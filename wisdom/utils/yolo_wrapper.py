# wisdom/utils/yolo_wrapper.py
"""
Wraps an Ultralytics YOLO detection model so that it exposes a
classification-like interface (input images → (B, num_classes) logits).

This wrapper is needed because:
  - Captum attribution methods expect model(x) → logits tensor with a
    'target' dimension (class index).
  - YOLO outputs raw predictions of shape (B, 4+nc, num_anchors).

The wrapper sums class scores across all anchor points, giving (B, nc).
"""
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from wisdom.utils.detection_loader import infer_num_classes, normalize_detection_output


class YOLOWrapper(nn.Module):
    """Make a YOLO DetectionModel behave like a classifier for Captum."""

    def __init__(self, yolo_torch_model: nn.Module, num_classes: int | None = None):
        super().__init__()
        self.yolo_model = yolo_torch_model
        self.nc = num_classes or infer_num_classes(yolo_torch_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = normalize_detection_output(self.yolo_model(x), num_classes=self.nc)
        cls_scores = preds[:, 4 : 4 + self.nc, :]
        return cls_scores.sum(dim=-1)  # (B, nc)


def yolo_surrogate_loss(
    model: nn.Module,
    images: torch.Tensor,
    device: str,
    wrapper: Optional[YOLOWrapper] = None,
) -> float:
    """
    Compute a surrogate detection 'loss' as the negative mean of summed
    class confidence.  Higher confidence → lower surrogate loss.
    When neurons are pruned and performance degrades, confidence drops
    and this value increases – matching the loss-gain logic in
    ConsensusWisdom.
    """
    model.eval()
    if wrapper is not None:
        wrapper.eval().to(device)
        with torch.no_grad():
            cls_scores = wrapper(images.to(device))  # (B, nc)
            # negative confidence → higher is worse
            return -cls_scores.sum().item()
    else:
        model.to(device)
        with torch.no_grad():
            preds = normalize_detection_output(
                model(images.to(device)),
                num_classes=infer_num_classes(model),
            )
            cls_scores = preds[:, 4:, :]
            return -cls_scores.sum().item()
