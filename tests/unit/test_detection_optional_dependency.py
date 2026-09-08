from __future__ import annotations

import builtins
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from wisdom.utils.detection_loader import load_detection_model


MINIMAL_YOLO = {
    "nc": 2,
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "backbone": [
        [-1, 1, "Conv", [8, 3, 2]],
        [-1, 1, "Conv", [16, 3, 2]],
    ],
    "head": [[[1], 1, "Detect", [2]]],
}


@pytest.mark.parametrize("model_path", ["local-model.yaml", "yolov5s.pt"])
def test_missing_ultralytics_reports_dependency_sync_command(
    monkeypatch,
    model_path: str,
) -> None:
    real_import = builtins.__import__

    def reject_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ultralytics)
    with pytest.raises(RuntimeError, match=r"requires Ultralytics.*`uv sync`"):
        load_detection_model(model_path, device="cpu")


@pytest.mark.ultralytics
def test_detection_architecture_uses_local_yaml_and_safe_state_dict_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the shared safe loader or conflating config and weights breaks this."""
    pytest.importorskip("ultralytics")
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    from wisdom.utils import checkpoints
    from wisdom_yolo_train import load_detection_architecture

    yaml_path = tmp_path / "local-yolo.yaml"
    yaml_path.write_text(yaml.safe_dump(MINIMAL_YOLO), encoding="utf-8")
    source = DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False)
    weights_path = tmp_path / "weights.pth"
    torch.save(source.state_dict(), weights_path)

    calls: list[dict[str, object]] = []
    real_load = checkpoints.torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "load", recording_load)
    loaded = load_detection_architecture(
        str(yaml_path), str(weights_path), "state-dict", "cpu"
    )

    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in source.state_dict().items())
    assert not loaded.training
    assert {parameter.device.type for parameter in loaded.parameters()} == {"cpu"}
    assert calls == [{"map_location": torch.device("cpu"), "weights_only": True}]


@pytest.mark.ultralytics
def test_detection_state_dict_deserializes_on_cpu_before_device_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing the execution device directly to torch.load must fail this test."""
    pytest.importorskip("ultralytics")
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    from wisdom.utils import checkpoints
    from wisdom_yolo_train import load_detection_architecture

    yaml_path = tmp_path / "local-yolo.yaml"
    yaml_path.write_text(yaml.safe_dump(MINIMAL_YOLO), encoding="utf-8")
    weights_path = tmp_path / "weights.pth"
    torch.save(DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False).state_dict(), weights_path)
    calls: list[dict[str, object]] = []
    real_load = checkpoints.torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "load", recording_load)
    load_detection_architecture(str(yaml_path), str(weights_path), "state-dict", "meta")

    assert calls == [{"map_location": torch.device("cpu"), "weights_only": True}]


@pytest.mark.ultralytics
@pytest.mark.parametrize("container", ["module", "model", "ema"])
def test_trusted_detection_checkpoint_loads_without_yaml_and_preserves_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, container: str
) -> None:
    """Rejecting stock .pt containers or retaining FP16 breaks float32 inference."""
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from wisdom.utils.detection_loader import normalize_detection_output
    from wisdom_yolo_train import load_detection_architecture

    source = DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False).eval().half()
    payload = source
    if container == "model":
        payload = {"model": source, "ema": None, "epoch": -1}
    elif container == "ema":
        non_ema = deepcopy(source)
        with torch.no_grad():
            next(non_ema.parameters()).add_(1)
        payload = {"model": non_ema, "ema": source, "epoch": 2}
    checkpoint = tmp_path / "local-detector.pt"
    torch.save(payload, checkpoint)
    calls = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    loaded = load_detection_architecture(None, str(checkpoint), "module", "cpu")

    assert not loaded.training
    assert list(dict(loaded.named_modules())) == list(dict(source.named_modules()))
    assert all(
        torch.equal(value.float(), loaded.state_dict()[name])
        for name, value in source.state_dict().items()
    )
    assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())
    assert calls == [{"map_location": torch.device("cpu"), "weights_only": False}]
    with torch.no_grad():
        output = normalize_detection_output(loaded(torch.zeros(1, 3, 32, 32)), 2)
    assert output.shape == (1, 6, 64)
    assert torch.isfinite(output).all()


@pytest.mark.ultralytics
@pytest.mark.parametrize("fully_frozen", [True, False])
def test_detection_adapter_handles_exported_frozen_models_without_changing_weights(
    fully_frozen: bool,
) -> None:
    """Export-time freezing must not hide all layers; partial freezing stays intact."""
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from wisdom.core.layers import discover_eligible_layers
    from wisdom.tasks.detection import DetectionAdapter

    model = DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False).eval()
    (model if fully_frozen else model.model[0]).requires_grad_(False)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    adapter = DetectionAdapter(model, num_classes=2)
    eligible = discover_eligible_layers(
        adapter.analysis_model, excluded_prefixes=adapter.excluded_layer_prefixes(),
    )

    expected = ("yolo_model.model.0.conv", "yolo_model.model.1.conv")
    assert eligible == (expected if fully_frozen else expected[1:])
    if not fully_frozen:
        assert not model.model[0].conv.weight.requires_grad
    adapter.analysis_model(torch.zeros(1, 3, 32, 32)).sum().backward()
    assert model.model[1].conv.weight.grad is not None
    assert torch.isfinite(model.model[1].conv.weight.grad).all()
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original.items())
