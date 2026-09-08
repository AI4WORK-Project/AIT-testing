"""Classification behavior isolated behind the shared task adapter contract."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from wisdom.core.layers import discover_eligible_layers
from wisdom.core.task import TaskBatch, extract_model_inputs


_TARGET_KEYS = ("labels", "label", "targets", "target")


class ClassificationAdapter:
    def __init__(self, model: nn.Module) -> None:
        self.analysis_model = model

    def prepare_batch(self, batch: object, device: str) -> TaskBatch:
        inputs = extract_model_inputs(batch).to(device)
        labels: object | None = None
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            labels = batch[1]
        elif isinstance(batch, Mapping):
            for key in _TARGET_KEYS:
                if key in batch:
                    labels = batch[key]
                    break
        if not isinstance(labels, torch.Tensor):
            raise TypeError(
                "Classification batches require class labels as a Tensor in "
                "(inputs, labels) or a labels/target mapping field."
            )
        return TaskBatch(inputs=inputs, targets=labels.to(device))

    def captum_target(self, batch: TaskBatch) -> torch.Tensor:
        if not isinstance(batch.targets, torch.Tensor):
            raise TypeError("Classification batch is missing Tensor class labels.")
        return batch.targets

    def excluded_layer_names(self) -> tuple[str, ...]:
        eligible = discover_eligible_layers(self.analysis_model)
        return eligible[-1:] if eligible else ()

    def excluded_layer_prefixes(self) -> tuple[str, ...]:
        return ()

    def capture_reference(self, batch: TaskBatch) -> None:
        return None

    def loss(
        self,
        batch: TaskBatch,
        reference: object | None = None,
    ) -> torch.Tensor:
        labels = self.captum_target(batch)
        return F.cross_entropy(self.analysis_model(batch.inputs), labels)
