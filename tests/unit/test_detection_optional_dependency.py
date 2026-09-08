from __future__ import annotations

import builtins
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
def test_missing_ultralytics_reports_optional_install_command(
    monkeypatch,
    model_path: str,
) -> None:
    real_import = builtins.__import__

    def reject_ultralytics(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_ultralytics)
    with pytest.raises(RuntimeError, match=r"uv sync --extra detection"):
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
