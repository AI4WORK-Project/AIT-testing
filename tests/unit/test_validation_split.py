from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from wisdom.core.task import reserve_validation_split


def _subset_indices(loader: DataLoader) -> list[int]:
    assert isinstance(loader.dataset, Subset)
    return list(loader.dataset.indices)


def test_reserve_validation_split_is_disjoint_reproducible_and_ten_percent() -> None:
    """Overlapping, nondeterministic, or incorrectly sized holdouts are regressions."""
    loader = DataLoader(TensorDataset(torch.arange(20)), batch_size=3, shuffle=False)
    global_rng_state = torch.random.get_rng_state()

    build, validation = reserve_validation_split(loader, fraction=0.10, seed=42)
    repeated_build, repeated_validation = reserve_validation_split(
        loader, fraction=0.10, seed=42
    )

    build_indices = _subset_indices(build)
    validation_indices = _subset_indices(validation)
    assert len(build.dataset) == 18
    assert len(validation.dataset) == 2
    assert set(build_indices).isdisjoint(validation_indices)
    assert set(build_indices) | set(validation_indices) == set(range(20))
    assert build_indices == sorted(build_indices)
    assert validation_indices == sorted(validation_indices)
    assert build_indices == _subset_indices(repeated_build)
    assert validation_indices == _subset_indices(repeated_validation)
    assert build.batch_size == validation.batch_size == 3
    assert build.collate_fn is loader.collate_fn
    assert validation.collate_fn is loader.collate_fn
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


def test_reserve_validation_split_rejects_a_dataset_too_small_to_partition() -> None:
    loader = DataLoader(TensorDataset(torch.arange(1)), batch_size=1)

    with pytest.raises(ValueError, match="at least two samples"):
        reserve_validation_split(loader, fraction=0.10, seed=42)
