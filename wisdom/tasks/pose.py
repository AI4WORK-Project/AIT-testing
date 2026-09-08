"""PyTorch-side trt_pose loading and WISDOM pose objectives.

The scalar energy produced here is a differentiable attribution surrogate. It
is deliberately not presented as a pose-accuracy metric. For unlabeled data,
the pruning loss measures behavioral drift from the unpruned model output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import torch
import torch.nn as nn

from wisdom.core.task import TaskBatch, extract_model_inputs
from wisdom.utils.checkpoints import load_pytorch_model


@dataclass(frozen=True)
class PoseTopology:
    keypoints: tuple[str, ...]
    skeleton: tuple[tuple[int, int], ...]

    @property
    def num_parts(self) -> int:
        return len(self.keypoints)

    @property
    def num_links(self) -> int:
        return len(self.skeleton)


@dataclass(frozen=True)
class PoseTargets:
    cmap: torch.Tensor
    paf: torch.Tensor
    mask: torch.Tensor


def load_pose_topology(path: str | Path) -> PoseTopology:
    topology_path = Path(path)
    with topology_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Pose topology '{topology_path}' must contain a JSON object.")

    keypoints = payload.get("keypoints")
    skeleton = payload.get("skeleton")
    if not isinstance(keypoints, list) or not keypoints:
        raise ValueError("Pose topology requires a nonempty 'keypoints' list.")
    if not all(isinstance(item, str) and item for item in keypoints):
        raise ValueError("Every pose keypoint must be a nonempty string.")
    if not isinstance(skeleton, list) or not skeleton:
        raise ValueError("Pose topology requires a nonempty 'skeleton' list.")

    normalized_skeleton: list[tuple[int, int]] = []
    for link in skeleton:
        valid_link = (
            isinstance(link, list)
            and len(link) == 2
            and all(isinstance(endpoint, int) and not isinstance(endpoint, bool) for endpoint in link)
            and all(1 <= endpoint <= len(keypoints) for endpoint in link)
        )
        if not valid_link:
            raise ValueError(
                "Every skeleton link must contain exactly two integer, one-based "
                f"endpoints in 1..{len(keypoints)}."
            )
        normalized_skeleton.append((link[0], link[1]))

    return PoseTopology(tuple(keypoints), tuple(normalized_skeleton))


def _module_is_within(module: ModuleType, expected_package: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is None:
        return False
    try:
        Path(origin).resolve().relative_to(expected_package)
    except ValueError:
        return False
    return True


def import_trt_pose_models(source_root: str | Path | None = None) -> ModuleType:
    """Import only the trt_pose PyTorch model definitions, never its plugins."""
    if source_root is None:
        return importlib.import_module("models_info.trt_pose.models")

    root = Path(source_root).expanduser().resolve()
    package_dir = root / "trt_pose"
    models_dir = package_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"Expected a trt_pose/models directory below source root '{root}'."
        )

    loaded_package = sys.modules.get("trt_pose")
    if loaded_package is not None and not _module_is_within(loaded_package, package_dir):
        origin = getattr(loaded_package, "__file__", "unknown")
        raise ImportError(
            f"trt_pose is already imported from '{origin}', which conflicts with "
            f"the requested source root '{root}'."
        )

    root_text = str(root)
    inserted = root_text not in sys.path
    if inserted:
        sys.path.insert(0, root_text)
    try:
        models = importlib.import_module("trt_pose.models")
    finally:
        if inserted:
            try:
                sys.path.remove(root_text)
            except ValueError:
                pass
    if not _module_is_within(models, models_dir):
        origin = getattr(models, "__file__", "unknown")
        raise ImportError(
            f"Imported trt_pose.models from '{origin}', not requested root '{root}'."
        )
    return models


def _terminal_leaf_module_name(branch_name: str, branch: nn.Module) -> str:
    """Return the terminal leaf in a packaged cmap/paf output branch."""
    leaf_name = branch_name
    leaf = branch
    while children := tuple(leaf.named_children()):
        child_name, leaf = children[-1]
        leaf_name = f"{leaf_name}.{child_name}"
    return leaf_name


def discover_trt_pose_output_layer_names(
    model: nn.Module,
    architecture: str,
) -> tuple[str, str]:
    """Find the packaged model's unique cmap/paf head and terminal output leaves."""
    paired_heads = [
        (name, module)
        for name, module in model.named_modules()
        if {"cmap_conv", "paf_conv"}.issubset(module._modules)
    ]
    if len(paired_heads) != 1:
        if not paired_heads:
            issue = "missing paired cmap_conv/paf_conv head"
        else:
            issue = (
                "ambiguous paired cmap_conv/paf_conv heads: "
                f"{[name or '<root>' for name, _ in paired_heads]}"
            )
        raise RuntimeError(f"trt_pose architecture '{architecture}' has {issue}.")

    head_name, head = paired_heads[0]
    cmap_branch_name = f"{head_name}.cmap_conv" if head_name else "cmap_conv"
    paf_branch_name = f"{head_name}.paf_conv" if head_name else "paf_conv"
    return (
        _terminal_leaf_module_name(cmap_branch_name, head.cmap_conv),
        _terminal_leaf_module_name(paf_branch_name, head.paf_conv),
    )


def _inactive_backbone_layer_names(model: nn.Module, models: ModuleType) -> tuple[str, ...]:
    """Retain torchvision classifier weights, but never attribute unused heads.

    These NVIDIA backbone wrappers explicitly execute feature-only forwards.
    Do not apply this rule to arbitrary pose models or all Linear modules.
    """
    inactive: list[str] = []
    for class_name, classifier_path in (
        ("ResNetBackbone", "resnet.fc"),
        ("DenseNetBackbone", "densenet.classifier"),
        ("MnasnetBackbone", "backbone.classifier"),
    ):
        backbone_type = getattr(models, class_name, None)
        if backbone_type is None:
            continue
        for name, module in model.named_modules():
            if not isinstance(module, backbone_type):
                continue
            classifier = module.get_submodule(classifier_path)
            prefix = f"{name}.{classifier_path}" if name else classifier_path
            for leaf_name, leaf in classifier.named_modules():
                if isinstance(leaf, (nn.Conv2d, nn.Linear)):
                    inactive.append(f"{prefix}.{leaf_name}" if leaf_name else prefix)
    return tuple(inactive)


def build_trt_pose_model(
    *,
    architecture: str,
    topology_path: str | Path,
    architecture_kwargs: dict[str, object] | None = None,
    source_root: str | Path | None = None,
) -> nn.Module:
    topology = load_pose_topology(topology_path)
    models = import_trt_pose_models(source_root)
    factories = getattr(models, "MODELS", {})
    if architecture not in factories:
        available = ", ".join(sorted(factories))
        raise ValueError(
            f"Unknown trt_pose architecture '{architecture}'. Available: {available}."
        )
    kwargs = dict(architecture_kwargs or {})
    if "pretrained" in kwargs:
        raise ValueError("Do not set 'pretrained'; WISDOM always constructs without downloads.")
    model = factories[architecture](
        topology.num_parts,
        2 * topology.num_links,
        pretrained=False,
        **kwargs,
    )
    if not isinstance(model, nn.Module):
        raise TypeError(f"trt_pose architecture '{architecture}' did not return nn.Module.")
    model._wisdom_output_layer_names = discover_trt_pose_output_layer_names(  # type: ignore[attr-defined]
        model,
        architecture,
    )
    model._wisdom_inactive_layer_names = _inactive_backbone_layer_names(model, models)
    return model


def load_trt_pose_checkpoint(
    checkpoint_path: str | Path,
    *,
    topology_path: str | Path,
    architecture: str = "resnet18_baseline_att",
    architecture_kwargs: dict[str, object] | None = None,
    source_root: str | Path | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> nn.Module:
    """Reconstruct an explicit trt_pose architecture and load its state dict."""

    def factory() -> nn.Module:
        return build_trt_pose_model(
            architecture=architecture,
            topology_path=topology_path,
            architecture_kwargs=architecture_kwargs,
            source_root=source_root,
        )

    return load_pytorch_model(
        checkpoint_path,
        device=device,
        checkpoint_format="state-dict",
        model_factory=factory,
        strict=strict,
    )


def _validate_pose_output(output: object) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (
        isinstance(output, (tuple, list))
        and len(output) == 2
        and all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in output)
    )
    if not valid:
        raise TypeError("A pose model must return exactly two 4-D tensors: (cmap, paf).")
    cmap, paf = output
    if cmap.shape[0] != paf.shape[0]:
        raise ValueError("cmap and paf outputs must have matching batch sizes.")
    return cmap, paf


def pose_attribution_objective(
    cmap: torch.Tensor,
    paf: torch.Tensor,
) -> torch.Tensor:
    """Return a gradient-preserving cmap/PAF energy surrogate, not accuracy."""
    cmap_energy = cmap.square().flatten(1).mean(1)
    paf_energy = paf.square().flatten(1).mean(1)
    return (0.5 * (cmap_energy + paf_energy)).unsqueeze(1)


def pose_confidence_surrogate(cmap: torch.Tensor) -> torch.Tensor:
    """Return a bounded, image-only pose confidence surrogate for BO.

    This is intentionally distinct from :func:`pose_attribution_objective`:
    the latter remains the signed-output-independent squared cmap/PAF energy
    used for Captum attribution, while this value is a bounded heatmap peak
    summary and is never presented as pose accuracy.
    """
    if cmap.ndim != 4:
        raise ValueError("Pose confidence maps must be 4-D")
    return cmap.sigmoid().flatten(2).amax(dim=2).mean()


class PoseAttributionWrapper(nn.Module):
    """Expose tuple-output pose behavior as a one-column Captum objective."""

    def __init__(self, pose_model: nn.Module) -> None:
        super().__init__()
        self.pose_model = pose_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        cmap, paf = _validate_pose_output(self.pose_model(inputs))
        return pose_attribution_objective(cmap, paf)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"Pose prediction and target shapes differ: {pred.shape} != {target.shape}."
        )
    try:
        expanded = mask.expand_as(pred)
    except RuntimeError as exc:
        raise ValueError(
            f"Pose mask shape {mask.shape} cannot expand to output shape {pred.shape}."
        ) from exc
    return (expanded * (pred - target).square()).sum() / expanded.sum().clamp_min(1)


def relative_mse(
    pred: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred.shape != reference.shape:
        raise ValueError(
            f"Pose prediction and reference shapes differ: {pred.shape} != {reference.shape}."
        )
    return (pred - reference).square().mean() / reference.square().mean().clamp_min(eps)


class PoseAdapter:
    """Task adapter for supervised pose loss or unlabeled behavioral drift."""

    def __init__(
        self,
        pose_model: nn.Module,
        *,
        output_layer_names: tuple[str, str] | None = None,
    ) -> None:
        self.analysis_model = PoseAttributionWrapper(pose_model)
        names = output_layer_names or getattr(
            pose_model,
            "_wisdom_output_layer_names",
            None,
        )
        if not isinstance(names, tuple) or len(names) != 2:
            raise ValueError(
                "PoseAdapter requires the exact (cmap, paf) output layer names "
                "for this architecture."
            )
        raw_modules = dict(pose_model.named_modules())
        missing = [name for name in names if name not in raw_modules]
        if missing:
            raise ValueError(f"Pose output layer name(s) not found: {missing}.")
        self._output_layer_names = names

    @property
    def pose_model(self) -> nn.Module:
        return self.analysis_model.pose_model

    def prepare_batch(self, batch: object, device: str) -> TaskBatch:
        inputs = extract_model_inputs(batch).to(device)
        if inputs.ndim != 4:
            raise ValueError("Pose model inputs must be a 4-D (B,C,H,W) tensor.")

        targets: PoseTargets | None = None
        if isinstance(batch, (tuple, list)):
            if len(batch) == 4:
                _, cmap, paf, mask = batch
                targets = self._prepare_targets(cmap, paf, mask, inputs, device)
            elif len(batch) != 1:
                raise TypeError(
                    "Pose batches must be a Tensor, (images,), or "
                    "(images, cmap, paf, mask)."
                )
        elif isinstance(batch, Mapping):
            pose_keys = ("cmap", "paf", "mask")
            provided = [key in batch for key in pose_keys]
            if any(provided) and not all(provided):
                raise TypeError("Pose mappings must provide cmap, paf, and mask together.")
            if all(provided):
                targets = self._prepare_targets(
                    batch["cmap"],
                    batch["paf"],
                    batch["mask"],
                    inputs,
                    device,
                )
        return TaskBatch(inputs=inputs, targets=targets)

    @staticmethod
    def _prepare_targets(
        cmap: object,
        paf: object,
        mask: object,
        inputs: torch.Tensor,
        device: str,
    ) -> PoseTargets:
        if not all(isinstance(value, torch.Tensor) for value in (cmap, paf, mask)):
            raise TypeError("Pose cmap, paf, and mask targets must all be Tensors.")
        cmap = cmap.to(device)
        paf = paf.to(device)
        mask = mask.to(device)
        if cmap.ndim != 4 or paf.ndim != 4 or mask.ndim != 4:
            raise ValueError("Pose cmap, paf, and mask targets must all be 4-D.")
        if not (cmap.shape[0] == paf.shape[0] == mask.shape[0] == inputs.shape[0]):
            raise ValueError("Pose inputs and targets must have matching batch sizes.")
        return PoseTargets(cmap=cmap, paf=paf, mask=mask)

    def captum_target(self, batch: TaskBatch) -> int:
        return 0

    def excluded_layer_names(self) -> tuple[str, ...]:
        inactive = getattr(self.pose_model, "_wisdom_inactive_layer_names", ())
        return tuple(f"pose_model.{name}" for name in (*self._output_layer_names, *inactive))

    def excluded_layer_prefixes(self) -> tuple[str, ...]:
        return ()

    def capture_reference(
        self,
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if isinstance(batch.targets, PoseTargets):
            return None
        with torch.no_grad():
            cmap, paf = _validate_pose_output(self.pose_model(batch.inputs))
        return cmap.detach(), paf.detach()

    def loss(
        self,
        batch: TaskBatch,
        reference: object | None = None,
    ) -> torch.Tensor:
        cmap, paf = _validate_pose_output(self.pose_model(batch.inputs))
        if isinstance(batch.targets, PoseTargets):
            return 0.5 * (
                masked_mse(cmap, batch.targets.cmap, batch.targets.mask)
                + masked_mse(paf, batch.targets.paf, batch.targets.mask)
            )
        if not (
            isinstance(reference, (tuple, list))
            and len(reference) == 2
            and all(isinstance(item, torch.Tensor) for item in reference)
        ):
            raise ValueError("Unlabeled pose loss requires captured (cmap, paf) references.")
        return 0.5 * (
            relative_mse(cmap, reference[0])
            + relative_mse(paf, reference[1])
        )
