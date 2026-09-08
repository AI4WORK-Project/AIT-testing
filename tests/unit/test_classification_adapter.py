from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ..helpers import TinyClassifier
from wisdom.tasks.classification import ClassificationAdapter


def test_classification_adapter_preserves_cross_entropy_behavior() -> None:
    model = TinyClassifier().eval()
    adapter = ClassificationAdapter(model)
    images = torch.randn(2, 3, 8, 8)
    labels = torch.tensor([0, 2])
    batch = adapter.prepare_batch((images, labels), "cpu")

    torch.testing.assert_close(
        adapter.loss(batch),
        F.cross_entropy(model(images), labels),
    )
    assert adapter.captum_target(batch) is batch.targets
    assert adapter.analysis_model is model
    assert adapter.excluded_layer_names() == ("classifier",)


def test_classification_adapter_supports_mapping_batch() -> None:
    labels = torch.tensor([1, 2])
    batch = ClassificationAdapter(TinyClassifier()).prepare_batch(
        {"images": torch.randn(2, 3, 8, 8), "labels": labels},
        "cpu",
    )
    assert batch.targets is labels


def test_classification_adapter_requires_labels() -> None:
    with pytest.raises(TypeError, match="class labels"):
        ClassificationAdapter(TinyClassifier()).prepare_batch(
            torch.randn(2, 3, 8, 8),
            "cpu",
        )
