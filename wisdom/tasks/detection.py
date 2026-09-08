"""Detection adapter preserving WISDOM's existing YOLO surrogate behavior."""

from __future__ import annotations

import torch
import torch.nn as nn

from wisdom.core.task import TaskBatch, extract_model_inputs
from wisdom.utils.detection_loader import detect_head_prefixes, infer_num_classes
from wisdom.utils.yolo_wrapper import YOLOWrapper


class DetectionAdapter:
    def __init__(self, model: nn.Module, num_classes: int | None = None) -> None:
        # Official .pt exports freeze every parameter. WISDOM needs eligible
        # layers/gradients for analysis, not optimizer updates. Restore gradients
        # only for fully frozen models; preserve intentional partial freezing.
        if not any(parameter.requires_grad for parameter in model.parameters()):
            model.requires_grad_(True)
        self.analysis_model = YOLOWrapper(
            model,
            num_classes=num_classes or infer_num_classes(model),
        )
        self._excluded_prefixes = tuple(detect_head_prefixes(self.analysis_model))

    @property
    def detection_model(self) -> nn.Module:
        return self.analysis_model.yolo_model

    def prepare_batch(self, batch: object, device: str) -> TaskBatch:
        return TaskBatch(inputs=extract_model_inputs(batch).to(device))

    def captum_target(self, batch: TaskBatch) -> torch.Tensor:
        return torch.zeros(
            batch.inputs.shape[0],
            dtype=torch.long,
            device=batch.inputs.device,
        )

    def excluded_layer_names(self) -> tuple[str, ...]:
        return ()

    def excluded_layer_prefixes(self) -> tuple[str, ...]:
        return self._excluded_prefixes

    def capture_reference(self, batch: TaskBatch) -> None:
        return None

    def loss(
        self,
        batch: TaskBatch,
        reference: object | None = None,
    ) -> torch.Tensor:
        return -self.analysis_model(batch.inputs).sum()
