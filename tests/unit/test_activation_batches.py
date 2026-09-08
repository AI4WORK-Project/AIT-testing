from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..helpers import TinyClassifier
from wisdom.core.activation import collect_per_neuron_series


class MappingImages(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"images": self.images[index]}


@pytest.mark.parametrize("mapping_batch", [False, True])
def test_activation_collection_accepts_tensor_and_mapping_batches(
    mapping_batch: bool,
) -> None:
    images = torch.randn(4, 3, 8, 8)
    dataset = MappingImages(images) if mapping_batch else images
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    series = collect_per_neuron_series(
        TinyClassifier().eval(),
        loader,
        {"features.0": [0]},
        device="cpu",
    )

    assert series["features.0"][0].shape == (4,)


def test_activation_hooks_are_removed_when_forward_raises() -> None:
    class FailingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 2, kernel_size=1)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            self.conv(inputs)
            raise RuntimeError("intentional forward failure")

    model = FailingModel()
    loader = DataLoader(torch.randn(1, 3, 4, 4), batch_size=1)

    with pytest.raises(RuntimeError, match="intentional"):
        collect_per_neuron_series(model, loader, {"conv": [0]}, device="cpu")

    assert not model.conv._forward_hooks
