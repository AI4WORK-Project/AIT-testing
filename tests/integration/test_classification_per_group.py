from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from ..helpers import TinyClassifier
from wisdom.core.wisdom_train import train_wisdom_classification
from wisdom.utils.io_cache import read_layer_scores_csv


def test_classification_per_group_training_uses_requested_layers_and_groups(tmp_path) -> None:
    torch.manual_seed(7)
    model = TinyClassifier().eval()
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    output = tmp_path / "scores.csv"

    train_wisdom_classification(
        model,
        loader,
        str(output),
        top_m=1,
        methods=["la"],
        selection_mode="per-group",
        n_groups=2,
        num_layers=2,
        device="cpu",
    )

    scores = read_layer_scores_csv(str(output))
    assert tuple(scores) == ("features.0", "features.4")
    assert all(torch.count_nonzero(layer_scores) == 1 for layer_scores in scores.values())
