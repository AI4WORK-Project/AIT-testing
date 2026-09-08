#!/usr/bin/env python
"""Thin trt_pose orchestration for WISDOM pretraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wisdom.core.task import PreparedTask, TaskEvaluation, reserve_validation_split
from wisdom.core.wisdom_train import train_wisdom_pose
from wisdom.tasks.pose import (
    PoseAdapter,
    PoseTargets,
    build_trt_pose_model,
    pose_confidence_surrogate,
)
from wisdom.utils.checkpoints import load_pytorch_model
from wisdom.utils.cli_options import add_selection_arguments, positive_int


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class PoseImageDataset(Dataset):
    """Deterministic unlabeled RGB inputs using trt_pose example normalization."""

    def __init__(
        self,
        image_source: str,
        *,
        image_size: int = 224,
        max_images: int | None = None,
    ) -> None:
        source = Path(image_source)
        if source.is_dir():
            paths = [
                path
                for path in sorted(source.iterdir())
                if path.suffix.lower() in _IMAGE_EXTENSIONS
            ]
        elif source.is_file() and source.suffix.lower() in _IMAGE_EXTENSIONS:
            paths = [source]
        else:
            raise FileNotFoundError(
                f"Pose image source must be an image or directory: {image_source}"
            )
        self.paths = paths[:max_images] if max_images is not None else paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def _parse_json_mapping(value: str) -> dict[str, object]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return payload


def _parse_method_csvs(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --method-out-csv value '{item}'. Expected method=path."
            )
        method, path = item.split("=", 1)
        method = method.strip().lower()
        if not method:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Empty method.")
        mapping[method] = path
    return mapping


def _architecture_factory(args: argparse.Namespace):
    return lambda: build_trt_pose_model(
        architecture=args.pose_architecture,
        topology_path=args.pose_topology,
        architecture_kwargs=args.pose_model_kwargs,
        source_root=_pose_model_source_root(args),
    )


def _pose_model_source_root(args: argparse.Namespace) -> str | None:
    """Return an explicit model-definition root, never an inferred sibling path."""
    if hasattr(args, "weights_path"):
        return args.model_path
    return getattr(args, "trt_pose_root", None)


def load_pose_model(args: argparse.Namespace) -> nn.Module:
    """Load separate pose architecture and weights under the safe policy."""
    checkpoint_path = getattr(args, "weights_path", args.model_path)
    return load_pytorch_model(
        checkpoint_path,
        device=args.device,
        checkpoint_format=args.checkpoint_format,
        # The factory is deliberately supplied for both safe formats.  ``auto``
        # only recognizes state dictionaries in the common loader, and module
        # mode is the sole explicit trusted pickle opt-in.
        model_factory=(None if args.checkpoint_format == "module" else _architecture_factory(args)),
    )


def _output_layer_names(
    args: argparse.Namespace,
    model: nn.Module,
) -> tuple[str, str] | None:
    if getattr(args, "pose_output_layers", None):
        return tuple(args.pose_output_layers)
    return getattr(model, "_wisdom_output_layer_names", None)


def pose_pck(
    predicted_cmap: torch.Tensor,
    target_cmap: torch.Tensor,
    threshold: float = 0.05,
    mask: torch.Tensor | None = None,
) -> float | None:
    """Compute PCK from heatmap peaks for visible target keypoints only."""
    correct, visible = _pose_pck_counts(predicted_cmap, target_cmap, threshold, mask)
    if not visible:
        return None
    return correct / visible


def _pose_pck_counts(
    predicted_cmap: torch.Tensor,
    target_cmap: torch.Tensor,
    threshold: float,
    mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Return correct and visible keypoint counts using the public PCK rule."""
    if predicted_cmap.shape != target_cmap.shape or predicted_cmap.ndim != 4:
        raise ValueError(
            "Predicted and target confidence maps must have the same 4-D shape"
        )
    _, _, height, width = predicted_cmap.shape
    predicted_index = predicted_cmap.flatten(2).argmax(dim=2)
    target_index = target_cmap.flatten(2).argmax(dim=2)
    predicted_xy = torch.stack(
        (predicted_index % width, predicted_index // width), dim=-1
    ).float()
    target_xy = torch.stack(
        (target_index % width, target_index // width), dim=-1
    ).float()
    visible = target_cmap.abs().flatten(2).amax(dim=2) > 0
    if mask is not None:
        try:
            expanded_mask = mask.expand_as(target_cmap)
        except RuntimeError as exc:
            raise ValueError("Pose PCK mask must expand to the target confidence maps") from exc
        valid_at_target = expanded_mask.flatten(2).gather(2, target_index.unsqueeze(-1)).squeeze(-1)
        visible = visible & (valid_at_target > 0)
    if not bool(visible.any()):
        return 0, 0
    normalizer = math.sqrt(height * height + width * width)
    distances = torch.linalg.vector_norm(predicted_xy - target_xy, dim=-1) / normalizer
    return int((distances[visible] <= threshold).sum().item()), int(visible.sum().item())


def _evaluation_adapter(model: nn.Module) -> PoseAdapter:
    """Resolve the model's established heads for metric-only evaluation."""
    output_names = getattr(model, "_wisdom_output_layer_names", None)
    if output_names is None:
        modules = dict(model.named_modules())
        conventional = ("cmap_head", "paf_head")
        if all(name in modules for name in conventional):
            output_names = conventional
    return PoseAdapter(model, output_layer_names=output_names)


def evaluate_pose(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    adapter: PoseAdapter | None = None,
) -> TaskEvaluation:
    """Evaluate labeled pose batches honestly, or image-only confidence behavior.

    No class targets or cross-entropy are introduced for pose.  A loader is
    either fully supervised or fully image-only so a mixed source cannot turn
    partial labels into a misleading aggregate metric.
    """
    adapter = adapter if adapter is not None else _evaluation_adapter(model)
    total_samples = 0
    total_loss = 0.0
    correct_keypoints = 0
    visible_keypoints = 0
    total_confidence = 0.0
    supervised: bool | None = None

    model.eval().to(device)
    with torch.no_grad():
        for raw_batch in loader:
            batch = adapter.prepare_batch(raw_batch, device)
            is_supervised = isinstance(batch.targets, PoseTargets)
            if supervised is None:
                supervised = is_supervised
            elif supervised != is_supervised:
                raise ValueError("Pose evaluation loader mixes supervised and image-only batches.")

            sample_count = batch.inputs.shape[0]
            total_samples += sample_count
            cmap, _ = model(batch.inputs)
            if is_supervised:
                assert isinstance(batch.targets, PoseTargets)
                total_loss += float(adapter.loss(batch).item()) * sample_count
                correct, visible = _pose_pck_counts(cmap, batch.targets.cmap, 0.05, batch.targets.mask)
                correct_keypoints += correct
                visible_keypoints += visible
            else:
                total_confidence += float(pose_confidence_surrogate(cmap).item()) * sample_count

    if total_samples == 0:
        raise ValueError("Pose evaluation loader contains no samples.")
    if supervised:
        pck = correct_keypoints / visible_keypoints if visible_keypoints else None
        metrics = {"loss": total_loss / total_samples, "pck": pck, "pose_confidence_surrogate": None}
        return TaskEvaluation(metrics=metrics, bo_metric_name="pck", bo_metric_value=pck)

    confidence = total_confidence / total_samples
    metrics = {"loss": None, "pck": None, "pose_confidence_surrogate": confidence}
    return TaskEvaluation(
        metrics=metrics,
        bo_metric_name="pose_confidence_surrogate",
        bo_metric_value=confidence,
    )


def _build_pose_loader(image_source: str, args: argparse.Namespace) -> DataLoader:
    dataset = PoseImageDataset(
        image_source,
        image_size=args.image_size,
        max_images=getattr(args, "max_images", None),
    )
    if not len(dataset):
        raise FileNotFoundError(f"No supported images found in {image_source}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
    )


def prepare_pose_inference(args: argparse.Namespace) -> PreparedTask:
    """Prepare packaged pose architecture, weights, and separate role loaders."""
    model = load_pose_model(args)
    output_layer_names = _output_layer_names(args, model)
    build_loader = _build_pose_loader(args.build_data_path, args)
    validation_loader = (
        _build_pose_loader(args.validation_data_path, args)
        if args.validation_data_path
        else None
    )
    test_loader = _build_pose_loader(args.test_data_path, args)
    if getattr(args, "bo", False) and validation_loader is None:
        build_loader, validation_loader = reserve_validation_split(
            build_loader,
            seed=getattr(args, "seed", 42),
        )

    def score_trainer(out_csv: str) -> str:
        return train_wisdom_pose(
            model=model,
            train_loader=build_loader,
            out_csv=out_csv,
            output_layer_names=output_layer_names,
            top_m=args.top_m_neurons,
            methods=args.methods,
            voting_mode=args.voting_mode,
            selection_mode=args.selection_mode,
            n_groups=args.num_groups,
            num_layers=args.num_layers,
            device=args.device,
            checkpoint_path=args.trainer_checkpoint,
            checkpoint_every=args.checkpoint_every,
        )

    adapter = PoseAdapter(model, output_layer_names=output_layer_names)
    return PreparedTask(
        task="pose",
        adapter=adapter,
        build_loader=build_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        score_trainer=score_trainer,
        evaluate=lambda loader: evaluate_pose(model, loader, args.device, adapter=adapter),
        model_name=args.pose_architecture,
        checkpoint_format=args.checkpoint_format,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WISDOM consensus pretraining for trt_pose models"
    )
    parser.add_argument("--model-path", required=True, help="Pose checkpoint path")
    parser.add_argument("--pose-topology", required=True, help="human_pose.json path")
    parser.add_argument(
        "--pose-architecture",
        default="resnet18_baseline_att",
        help="Explicit key from trt_pose.models.MODELS",
    )
    parser.add_argument(
        "--pose-model-kwargs",
        type=_parse_json_mapping,
        default={},
        help="Architecture keyword arguments as a JSON object",
    )
    parser.add_argument(
        "--pose-output-layers",
        nargs=2,
        metavar=("CMAP_LAYER", "PAF_LAYER"),
        default=None,
        help="Exact raw output-layer names for a custom pose architecture",
    )
    parser.add_argument(
        "--trt-pose-root",
        default=None,
        help="Root of the adjacent NVIDIA trt_pose clone",
    )
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "module", "state-dict"],
        default="state-dict",
        help="Use module only for a trusted serialized nn.Module",
    )
    parser.add_argument("--img-dir", required=True, help="Unlabeled image or directory")
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--max-images", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-m", type=positive_int, default=20)
    parser.add_argument(
        "--methods", nargs="+", default=["lgxa", "lig"],
        help="Attribution methods (default: lgxa lig); see README for all method IDs.",
    )
    parser.add_argument(
        "--voting-mode",
        default="fine-grained",
        choices=["fine-grained", "coarse"],
    )
    add_selection_arguments(parser)
    parser.add_argument(
        "--out-csv",
        default="neuron_eval_out/wisdom_pose_scores.csv",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default=None, help="Trainer resume checkpoint")
    parser.add_argument("--checkpoint-every", type=positive_int, default=50)
    parser.add_argument("--method-out-csv", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    model = load_pose_model(args)
    dataset = PoseImageDataset(
        args.img_dir,
        image_size=args.image_size,
        max_images=args.max_images,
    )
    if not dataset:
        raise FileNotFoundError(f"No supported images found in {args.img_dir}")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": max(0, args.num_workers),
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    csv_path = train_wisdom_pose(
        model,
        loader,
        args.out_csv,
        output_layer_names=_output_layer_names(args, model),
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        selection_mode=args.selection_mode,
        n_groups=args.num_groups,
        num_layers=args.num_layers,
        device=args.device,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        method_out_csvs=_parse_method_csvs(args.method_out_csv) or None,
    )
    print(f"\nDone. CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
