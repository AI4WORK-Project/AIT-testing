from __future__ import annotations

import json
from pathlib import Path

import torch
import pytest
import models_info

from wisdom.tasks.pose import (
    PoseAdapter,
    PoseAttributionWrapper,
    build_trt_pose_model,
    import_trt_pose_models,
    load_trt_pose_checkpoint,
)


def _human_topology(path: Path) -> Path:
    payload = {
        "keypoints": [f"part-{index}" for index in range(18)],
        "skeleton": [[(index % 18) + 1, ((index + 1) % 18) + 1] for index in range(21)],
    }
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize("architecture", [
    "resnet18_baseline_att", "densenet121_baseline_att", "mnasnet0_5_baseline_att",
])
def test_every_eligible_pose_layer_executes_for_default_attribution(tmp_path, architecture):
    """The unused ImageNet classifier must never become a Captum target layer."""
    from wisdom.attribution.captum_backend import batch_per_layer_scores
    from wisdom.core.layers import discover_eligible_layers

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_trt_pose_model(
            architecture=architecture,
            topology_path=_human_topology(tmp_path / "topology.json"),
            architecture_kwargs={"upsample_channels": 8, "num_upsample": 1},
        ).eval()
        adapter = PoseAdapter(model)
        layers = discover_eligible_layers(
            adapter.analysis_model, excluded_layers=adapter.excluded_layer_names(),
            excluded_prefixes=adapter.excluded_layer_prefixes(),
        )
        scores = batch_per_layer_scores(
            adapter.analysis_model, torch.ones(1, 3, 32, 32), target=0,
            device="cpu", method="lgxa", target_layers=layers,
        )
        assert layers and set(scores) == set(layers)
        assert all(torch.isfinite(score).all() for score in scores.values())
        assert any(score.abs().sum() > 0 for score in scores.values())
    finally:
        torch.set_num_threads(previous_threads)


def test_packaged_trt_pose_state_dict_round_trip_and_gradient(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("WISDOM_TRT_POSE_ROOT", raising=False)
    topology = _human_topology(tmp_path / "human_pose.json")
    models = import_trt_pose_models()
    assert Path(models.__file__).resolve().is_relative_to(
        Path(models_info.__file__).parent / "trt_pose" / "models"
    )

    source = build_trt_pose_model(
        architecture="resnet18_baseline_att",
        topology_path=topology,
    )
    assert len(source.state_dict()) == 172
    checkpoint = tmp_path / "pose.pth"
    torch.save(source.state_dict(), checkpoint)

    loaded = load_trt_pose_checkpoint(
        checkpoint,
        topology_path=topology,
        architecture="resnet18_baseline_att",
        device="cpu",
    )

    images = torch.randn(1, 3, 32, 32, requires_grad=True)
    cmap, paf = loaded(images)
    assert cmap.shape == (1, 18, 8, 8)
    assert paf.shape == (1, 42, 8, 8)
    PoseAttributionWrapper(loaded)(images).sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()


@pytest.mark.parametrize(
    ("architecture", "expected_output_layers"),
    [
        ("resnet18_baseline", ("1.cmap_conv.1", "1.paf_conv.1")),
        ("mnasnet0_5_baseline_att", ("1.cmap_conv", "1.paf_conv")),
    ],
)
def test_packaged_pose_heads_expose_terminal_output_layers_without_overrides(
    tmp_path: Path,
    monkeypatch,
    architecture: str,
    expected_output_layers: tuple[str, str],
) -> None:
    """Fail if packaged pose heads need architecture-specific metadata tables."""
    monkeypatch.delenv("WISDOM_TRT_POSE_ROOT", raising=False)
    model = build_trt_pose_model(
        architecture=architecture,
        topology_path=_human_topology(tmp_path / "human_pose.json"),
        architecture_kwargs={"upsample_channels": 8, "num_upsample": 1},
    ).eval()

    cmap, paf = model(torch.zeros(1, 3, 32, 32))

    assert cmap.shape[1] == 18
    assert paf.shape[1] == 42
    assert model._wisdom_output_layer_names == expected_output_layers
    adapter = PoseAdapter(model)
    inactive = "0.resnet.fc" if architecture == "resnet18_baseline" else "0.backbone.classifier.1"
    assert adapter.excluded_layer_names() == tuple(
        f"pose_model.{name}" for name in (*expected_output_layers, inactive)
    )
