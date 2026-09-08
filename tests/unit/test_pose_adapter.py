from __future__ import annotations

import json

import pytest
import torch

from ..helpers import TinyPoseModel
from wisdom.tasks.pose import (
    PoseAdapter,
    PoseAttributionWrapper,
    discover_trt_pose_output_layer_names,
    load_pose_topology,
    masked_mse,
    pose_attribution_objective,
    pose_confidence_surrogate,
    relative_mse,
)
from wisdom.attribution.captum_backend import batch_per_layer_scores


def test_energy_is_per_sample_and_differentiable() -> None:
    torch.manual_seed(7)
    wrapper = PoseAttributionWrapper(TinyPoseModel().eval())
    images = torch.randn(2, 3, 8, 8, requires_grad=True)

    score = wrapper(images)

    assert score.shape == (2, 1)
    score.sum().backward()
    assert images.grad is not None
    assert images.grad.abs().sum() > 0


def test_pose_surrogate_is_captum_targetable_without_class_labels() -> None:
    wrapper = PoseAttributionWrapper(TinyPoseModel().eval())
    scores = batch_per_layer_scores(
        wrapper,
        torch.randn(2, 3, 8, 8),
        target=0,
        device="cpu",
        method="lgxa",
        target_layers=("pose_model.backbone.0",),
    )
    assert scores["pose_model.backbone.0"].shape == (4,)


def test_signed_paf_does_not_cancel() -> None:
    cmap = torch.zeros(1, 1, 2, 2)
    paf = torch.tensor(
        [[[[1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]]
    )

    torch.testing.assert_close(
        pose_attribution_objective(cmap, paf),
        torch.tensor([[0.5]]),
    )


def test_pose_confidence_surrogate_is_bounded_and_requires_heatmaps() -> None:
    """Fail if confidence ceases to be a bounded heatmap-only BO surrogate."""
    cmap = torch.tensor([[[[-10.0, 10.0]]]])

    value = pose_confidence_surrogate(cmap)

    assert 0.0 <= value.item() <= 1.0
    assert value.item() == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
    with pytest.raises(ValueError, match="4-D"):
        pose_confidence_surrogate(torch.zeros(1, 1, 2))


def test_mask_and_relative_normalization() -> None:
    pred = torch.tensor([[[[3.0, 9.0]]]])
    target = torch.tensor([[[[1.0, 1.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])
    torch.testing.assert_close(masked_mse(pred, target, mask), torch.tensor(4.0))

    reference = torch.full((1, 1, 2, 2), 2.0)
    torch.testing.assert_close(
        relative_mse(reference + 1.0, reference),
        torch.tensor(0.25),
    )


def test_unlabeled_pose_loss_is_behavioral_drift() -> None:
    raw = TinyPoseModel().eval()
    adapter = PoseAdapter(raw, output_layer_names=("cmap_head", "paf_head"))
    batch = adapter.prepare_batch(torch.randn(2, 3, 8, 8), "cpu")
    reference = adapter.capture_reference(batch)

    torch.testing.assert_close(adapter.loss(batch, reference), torch.tensor(0.0))
    with torch.no_grad():
        raw.cmap_head.bias.add_(0.5)
    assert adapter.loss(batch, reference) > 0
    assert adapter.captum_target(batch) == 0


def test_supervised_pose_loss_uses_targets_instead_of_class_labels() -> None:
    raw = TinyPoseModel().eval()
    adapter = PoseAdapter(raw, output_layer_names=("cmap_head", "paf_head"))
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        cmap, paf = raw(images)
    batch = adapter.prepare_batch(
        (images, cmap, paf, torch.ones(2, 1, 8, 8)),
        "cpu",
    )

    torch.testing.assert_close(adapter.loss(batch), torch.tensor(0.0))


def test_pose_adapter_preserves_stable_hook_names_and_excludes_only_heads() -> None:
    adapter = PoseAdapter(
        TinyPoseModel(),
        output_layer_names=("cmap_head", "paf_head"),
    )

    assert "pose_model.backbone.0" in dict(adapter.analysis_model.named_modules())
    assert adapter.excluded_layer_names() == (
        "pose_model.cmap_head",
        "pose_model.paf_head",
    )
    assert adapter.excluded_layer_prefixes() == ()


def test_pose_adapter_rejects_unknown_output_layer_names() -> None:
    with pytest.raises(ValueError, match="not found"):
        PoseAdapter(
            TinyPoseModel(),
            output_layer_names=("missing", "paf_head"),
        )


def test_pose_adapter_explicit_output_layers_override_model_metadata() -> None:
    raw = TinyPoseModel()
    raw._wisdom_output_layer_names = ("not_a_cmap_head", "not_a_paf_head")

    adapter = PoseAdapter(raw, output_layer_names=("cmap_head", "paf_head"))

    assert adapter.excluded_layer_names() == (
        "pose_model.cmap_head",
        "pose_model.paf_head",
    )


def test_pose_head_discovery_rejects_missing_or_ambiguous_paired_heads() -> None:
    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cmap_conv = torch.nn.Conv2d(1, 1, kernel_size=1)
            self.paf_conv = torch.nn.Conv2d(1, 2, kernel_size=1)

    with pytest.raises(RuntimeError, match="trt_pose architecture 'missing'.*missing"):
        discover_trt_pose_output_layer_names(torch.nn.Identity(), "missing")

    ambiguous = torch.nn.Module()
    ambiguous.first = Head()
    ambiguous.second = Head()
    with pytest.raises(RuntimeError, match="trt_pose architecture 'ambiguous'.*ambiguous"):
        discover_trt_pose_output_layer_names(ambiguous, "ambiguous")


def test_pose_wrapper_rejects_non_pose_output() -> None:
    class BadModel(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

    with pytest.raises(TypeError, match="two 4-D tensors"):
        PoseAttributionWrapper(BadModel())(torch.randn(1, 3, 8, 8))


def test_pose_topology_is_validated(tmp_path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps({"keypoints": ["a", "b", "c"], "skeleton": [[1, 2], [2, 3]]})
    )
    topology = load_pose_topology(valid)
    assert topology.num_parts == 3
    assert topology.num_links == 2

    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps({"keypoints": ["a", "b"], "skeleton": [[1, 3]]})
    )
    with pytest.raises(ValueError, match="one-based endpoints"):
        load_pose_topology(invalid)
