from __future__ import annotations

import pytest
import torch

from ..helpers import TinyClassifier
from wisdom.utils import checkpoints


@pytest.mark.parametrize("key", [None, "state_dict", "model_state_dict"])
def test_state_dict_formats(tmp_path, key: str | None) -> None:
    source = TinyClassifier()
    state = source.state_dict()
    payload = state if key is None else {key: state}
    path = tmp_path / "weights.pth"
    torch.save(payload, path)

    loaded = checkpoints.load_pytorch_model(
        path,
        device="cpu",
        model_factory=TinyClassifier,
    )

    assert loaded.training is False
    assert next(loaded.parameters()).device.type == "cpu"
    for expected, actual in zip(source.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_uniform_data_parallel_prefix_is_removed(tmp_path) -> None:
    state = {
        f"module.{key}": value
        for key, value in TinyClassifier().state_dict().items()
    }
    path = tmp_path / "prefixed.pth"
    torch.save(state, path)

    loaded = checkpoints.load_pytorch_model(
        path,
        device="cpu",
        model_factory=TinyClassifier,
    )

    assert set(loaded.state_dict()) == set(TinyClassifier().state_dict())


def test_mixed_data_parallel_prefix_is_not_removed(tmp_path) -> None:
    state = dict(TinyClassifier().state_dict())
    state["module.features.0.weight"] = state.pop("features.0.weight")
    path = tmp_path / "mixed-prefix.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match="module.features.0.weight"):
        checkpoints.load_pytorch_model(
            path,
            device="cpu",
            model_factory=TinyClassifier,
        )


def test_state_dict_requires_architecture_factory(tmp_path) -> None:
    path = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), path)

    with pytest.raises(ValueError, match="architecture factory"):
        checkpoints.load_pytorch_model(path, device="cpu")


def test_trusted_serialized_module_requires_explicit_format(tmp_path) -> None:
    path = tmp_path / "whole.pth"
    torch.save(TinyClassifier(), path)

    loaded = checkpoints.load_pytorch_model(
        path,
        device="cpu",
        checkpoint_format="module",
    )
    assert isinstance(loaded, TinyClassifier)
    assert loaded.training is False

    with pytest.raises(ValueError, match="checkpoint-format module"):
        checkpoints.load_pytorch_model(path, device="cpu")


def test_every_load_uses_explicit_map_location(tmp_path, monkeypatch) -> None:
    path = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), path)
    real_load = torch.load
    seen: list[torch.device | None] = []

    def recording_load(*args, **kwargs):
        seen.append(kwargs.get("map_location"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "load", recording_load)
    checkpoints.load_pytorch_model(
        path,
        device="cpu",
        model_factory=TinyClassifier,
    )

    assert seen == [torch.device("cpu")]


def test_state_dict_architecture_is_instantiated_before_checkpoint_load(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), path)
    real_load = torch.load
    events: list[str] = []

    def factory() -> TinyClassifier:
        events.append("factory")
        return TinyClassifier()

    def recording_load(*args, **kwargs):
        events.append("load")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(checkpoints.torch, "load", recording_load)
    checkpoints.load_pytorch_model(
        path,
        device="cpu",
        checkpoint_format="state-dict",
        model_factory=factory,
    )

    assert events[:2] == ["factory", "load"]


def test_strict_mismatch_reports_checkpoint_and_key(tmp_path) -> None:
    state = dict(TinyClassifier().state_dict())
    state.pop("classifier.bias")
    path = tmp_path / "bad.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match=r"(?s)bad\.pth.*classifier\.bias"):
        checkpoints.load_pytorch_model(
            path,
            device="cpu",
            model_factory=TinyClassifier,
        )


def test_ambiguous_nested_state_dict_is_rejected(tmp_path) -> None:
    state = TinyClassifier().state_dict()
    path = tmp_path / "ambiguous.pth"
    torch.save({"state_dict": state, "model_state_dict": state}, path)

    with pytest.raises(ValueError, match="Ambiguous state-dict keys"):
        checkpoints.load_pytorch_model(
            path,
            device="cpu",
            model_factory=TinyClassifier,
        )


def test_invalid_checkpoint_format_is_rejected_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint_format"):
        checkpoints.load_pytorch_model(
            tmp_path / "unused.pth",
            device="cpu",
            checkpoint_format="guess",
        )
