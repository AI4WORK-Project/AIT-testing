"""Explicit, device-safe loading for PyTorch models and state dictionaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import pickle

import torch
import torch.nn as nn


ModelFactory = Callable[[], nn.Module]
_STATE_KEYS = ("state_dict", "model_state_dict")
_CHECKPOINT_FORMATS = {"auto", "module", "state-dict"}


def _is_string_tensor_mapping(payload: object) -> bool:
    return (
        isinstance(payload, Mapping)
        and bool(payload)
        and all(isinstance(key, str) for key in payload)
        and all(isinstance(value, torch.Tensor) for value in payload.values())
    )


def extract_state_dict(payload: object) -> Mapping[str, torch.Tensor] | None:
    """Return a recognized state dict without guessing arbitrary container keys."""
    if _is_string_tensor_mapping(payload):
        return payload
    if not isinstance(payload, Mapping):
        return None

    matches = [key for key in _STATE_KEYS if key in payload]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous state-dict keys: {matches}")
    if len(matches) == 1:
        candidate = payload[matches[0]]
        if not _is_string_tensor_mapping(candidate):
            raise ValueError(
                f"Checkpoint key '{matches[0]}' is not a nonempty string-to-Tensor mapping."
            )
        return candidate
    return None


def normalize_data_parallel_prefix(
    state_dict: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> dict[str, torch.Tensor]:
    """Strip a uniform ``module.`` prefix only when the target does not use it."""
    source_keys = tuple(state_dict)
    target_keys = tuple(model.state_dict())
    strip_prefix = (
        bool(source_keys)
        and all(key.startswith("module.") for key in source_keys)
        and not any(key.startswith("module.") for key in target_keys)
    )
    if strip_prefix:
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def _finish_model(model: nn.Module, device: torch.device) -> nn.Module:
    model.to(device)
    model.eval()
    return model


def _instantiate_model(model_factory: ModelFactory | None) -> nn.Module | None:
    if model_factory is None:
        return None
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("The architecture factory must return torch.nn.Module.")
    return model


def load_pytorch_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    checkpoint_format: str = "auto",
    model_factory: ModelFactory | None = None,
    strict: bool = True,
) -> nn.Module:
    """Load a trusted module or safely reconstruct a model from a state dict.

    ``checkpoint_format='module'`` explicitly opts into Python pickle loading and
    must therefore only be used with a trusted checkpoint. ``auto`` is safe-first:
    it recognizes state dictionaries but never falls back to unsafe module loading.
    """
    if checkpoint_format not in _CHECKPOINT_FORMATS:
        options = ", ".join(sorted(_CHECKPOINT_FORMATS))
        raise ValueError(
            f"Invalid checkpoint_format '{checkpoint_format}'; expected one of: {options}."
        )

    path = Path(checkpoint_path)
    target_device = torch.device(device)

    if checkpoint_format == "module":
        payload = torch.load(
            path,
            map_location=target_device,
            weights_only=False,
        )
        if not isinstance(payload, nn.Module):
            raise TypeError(
                f"Checkpoint '{path}' is not a serialized torch.nn.Module; "
                "use checkpoint_format='state-dict' with an architecture factory."
            )
        return _finish_model(payload, target_device)

    model = _instantiate_model(model_factory)
    if checkpoint_format == "state-dict" and model is None:
        raise ValueError(
            "A state-dict checkpoint contains weights but no architecture. "
            "Provide an architecture factory before loading it."
        )

    try:
        payload = torch.load(
            path,
            map_location=target_device,
            weights_only=True,
        )
    except pickle.UnpicklingError as exc:
        if checkpoint_format == "auto":
            raise ValueError(
                f"Checkpoint '{path}' is not a safely loadable state dict. "
                "Use checkpoint-format module only for a trusted serialized nn.Module."
            ) from exc
        raise ValueError(
            f"Checkpoint '{path}' could not be loaded as a state dict."
        ) from exc

    state_dict = extract_state_dict(payload)
    if state_dict is None:
        if isinstance(payload, nn.Module):  # Defensive; weights-only should not return this.
            raise ValueError(
                f"Checkpoint '{path}' contains a module. Use checkpoint-format module "
                "only if the file is trusted."
            )
        raise ValueError(
            f"Checkpoint '{path}' does not contain a recognized state dict "
            f"(raw, '{_STATE_KEYS[0]}', or '{_STATE_KEYS[1]}')."
        )
    if model is None:
        raise ValueError(
            f"Checkpoint '{path}' contains weights but no architecture. "
            "Provide an architecture factory before loading a state dict."
        )

    normalized = normalize_data_parallel_prefix(state_dict, model)
    try:
        model.load_state_dict(normalized, strict=strict)
    except RuntimeError as exc:
        architecture = type(model).__qualname__
        raise RuntimeError(
            f"Failed to load checkpoint '{path}' strictly into architecture "
            f"'{architecture}': {exc}"
        ) from exc
    return _finish_model(model, target_device)
