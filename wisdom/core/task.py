"""Small task abstractions shared by classification, detection, and pose."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


_INPUT_KEYS = ("images", "image", "inputs", "input", "pixel_values")

# Normalized model inputs plus optional task targets (act like an immutable container.)
@dataclass(frozen=True)
class TaskBatch:
    inputs: torch.Tensor
    targets: object | None = None


def extract_model_inputs(batch: object) -> torch.Tensor:
    """Extract image/input tensors from the small set of supported batch forms."""
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, (tuple, list)) and batch and isinstance(batch[0], torch.Tensor):
        return batch[0]
    if isinstance(batch, Mapping):
        for key in _INPUT_KEYS:
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError(
        "Expected a Tensor, a tuple/list whose first item is a Tensor, or a "
        f"mapping with a Tensor under one of {_INPUT_KEYS}."
    )


def _loader_for_subset(loader: DataLoader, indices: list[int]) -> DataLoader:
    """Rebuild one of WISDOM's ordinary loaders over a deterministic subset."""
    kwargs: dict[str, object] = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": False,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
        "multiprocessing_context": loader.multiprocessing_context,
        "generator": loader.generator,
    }
    if loader.num_workers:
        kwargs["prefetch_factor"] = loader.prefetch_factor
        kwargs["persistent_workers"] = loader.persistent_workers
    pin_memory_device = getattr(loader, "pin_memory_device", "")
    if pin_memory_device:
        kwargs["pin_memory_device"] = pin_memory_device
    return DataLoader(Subset(loader.dataset, indices), **kwargs)


def reserve_validation_split(
    loader: DataLoader,
    *,
    fraction: float = 0.10,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Remove a deterministic holdout from a build loader for BO validation."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("Validation fraction must be between zero and one.")
    total = len(loader.dataset)
    if total < 2:
        raise ValueError("Automatic validation splitting requires at least two samples.")

    validation_count = max(1, int(total * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    validation_indices = sorted(indices[:validation_count])
    build_indices = sorted(indices[validation_count:])
    return (
        _loader_for_subset(loader, build_indices),
        _loader_for_subset(loader, validation_indices),
    )

# Protocol for batch preparation, Captum target, head exclusions, reference capture, and pruning loss.
class TaskAdapter(Protocol):
    analysis_model: nn.Module

    def prepare_batch(self, batch: object, device: str) -> TaskBatch: ...

    def captum_target(self, batch: TaskBatch) -> int | torch.Tensor | None: ...

    def excluded_layer_names(self) -> tuple[str, ...]: ...

    def excluded_layer_prefixes(self) -> tuple[str, ...]: ...

    def capture_reference(self, batch: TaskBatch) -> object | None: ...

    def loss(
        self,
        batch: TaskBatch,
        reference: object | None = None,
    ) -> torch.Tensor: ...

# Carries honest report metrics and an optional BO metric name/value.
@dataclass(frozen=True)
class TaskEvaluation:
    metrics: dict[str, float | None]
    bo_metric_name: str | None
    bo_metric_value: float | None

# Bundles adapter, three role loaders, score callback, evaluator, model name, and checkpoint format for the runner.
# prepared in `run_wisdom.py`
@dataclass(frozen=True)
class PreparedTask:
    task: str
    adapter: TaskAdapter
    build_loader: DataLoader
    validation_loader: DataLoader | None
    test_loader: DataLoader
    score_trainer: Callable[[str], str]
    evaluate: Callable[[DataLoader], TaskEvaluation]
    model_name: str
    checkpoint_format: str
