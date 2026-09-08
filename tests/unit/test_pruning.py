from __future__ import annotations

import torch
import torch.nn as nn

from wisdom.pruning.weights_pruning import prune_linear_neurons


def test_linear_weight_pruning_zeros_output_rows_and_bias() -> None:
    layer = nn.Linear(3, 4, bias=True)
    original_weight = layer.weight.detach().clone()
    original_bias = layer.bias.detach().clone()

    prune_linear_neurons(layer, [1, 3])

    assert torch.count_nonzero(layer.weight[[1, 3]]) == 0
    assert torch.count_nonzero(layer.bias[[1, 3]]) == 0
    torch.testing.assert_close(layer.weight[[0, 2]], original_weight[[0, 2]])
    torch.testing.assert_close(layer.bias[[0, 2]], original_bias[[0, 2]])
