from __future__ import annotations

import pytest
import torch

from wisdom.core.task import extract_model_inputs


IMAGES = torch.randn(2, 3, 8, 8)


@pytest.mark.parametrize(
    "batch",
    [
        IMAGES,
        (IMAGES,),
        (IMAGES, torch.tensor([0, 1])),
        {"images": IMAGES, "labels": torch.tensor([0, 1])},
        {"pixel_values": IMAGES},
    ],
)
def test_supported_batches(batch: object) -> None:
    assert extract_model_inputs(batch) is IMAGES


def test_mapping_uses_documented_input_key_precedence() -> None:
    other = torch.randn_like(IMAGES)
    assert extract_model_inputs({"input": other, "images": IMAGES}) is IMAGES


@pytest.mark.parametrize(
    "batch",
    [(), [], {}, {"labels": torch.tensor([0])}, "bad", ("bad", IMAGES)],
)
def test_ambiguous_batches_fail(batch: object) -> None:
    with pytest.raises(TypeError, match="Tensor"):
        extract_model_inputs(batch)
