from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from ..helpers import TinyClassifier
from wisdom.core.wisdom import ClusteringConfig, WisdomConfig, WisdomIDC
from wisdom.core.wisdom_train import train_wisdom_classification
from wisdom.utils.io_cache import read_layer_scores_csv


@pytest.mark.parametrize("methods", [["la"], None])
def test_classification_pretraining_and_coverage(tmp_path, methods) -> None:
    torch.manual_seed(19)
    model = TinyClassifier().eval()
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    weights_before = {name: value.clone() for name, value in model.state_dict().items()}
    flags_before = [parameter.requires_grad for parameter in model.parameters()]
    with torch.no_grad():
        predictions_before = model(images).clone()

    csv_path = train_wisdom_classification(
        model,
        loader,
        str(tmp_path / "classification.csv"),
        top_m=1,
        methods=methods,
        device="cpu",
    )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights_before[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags_before
    with torch.no_grad():
        torch.testing.assert_close(model(images), predictions_before, rtol=0, atol=0)
    assert all(not module._forward_hooks and not module._backward_hooks for module in model.modules())

    scores = read_layer_scores_csv(csv_path)
    engine = WisdomIDC(
        model,
        cfg=WisdomConfig(top_m_neurons=1),
        cluster=ClusteringConfig(
            method="KMeans",
            params={"n_clusters": 2, "random_state": 7, "n_init": 10},
        ),
    )
    selected = engine.fit(loader, scores, device="cpu")
    rate, total, maximum = engine.coverage(loader, selected, device="cpu")

    assert 0.0 <= rate <= maximum <= 1.0
    assert total >= 1
