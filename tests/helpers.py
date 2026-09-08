from __future__ import annotations

import torch
import torch.nn as nn


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(4, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


class TinyPoseModel(nn.Module):
    def __init__(self, num_parts: int = 3, num_links: int = 2) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cmap_head = nn.Conv2d(4, num_parts, kernel_size=1)
        self.paf_head = nn.Conv2d(4, 2 * num_links, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        return self.cmap_head(features), self.paf_head(features)
