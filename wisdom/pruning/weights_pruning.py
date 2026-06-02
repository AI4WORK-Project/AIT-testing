# wisdom/pruning/weights_pruning.py
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn

def prune_linear_neurons(module: nn.Linear, neuron_indices: List[int], zero_incoming=False):
    with torch.no_grad():
        W = module.weight    # [out_features, in_features]
        b = module.bias      # [out_features] or None
        idx = torch.tensor(neuron_indices, dtype=torch.long, device=W.device)

        # Zero OUTGOING paths of chosen neurons: columns in W
        W[:, idx] = 0.0
        if zero_incoming:
            # Also zero row of those neurons if treating them as previous layer outputs
            W[idx, :] = 0.0
        if b is not None:
            # If neuron index refers to out_features, bias zeroing applies when zero_incoming==True.
            # Typically you do NOT touch bias here unless pruning output units.
            pass

def prune_conv_neurons(module: nn.Conv2d, channel_indices: List[int]):
    with torch.no_grad():
        W = module.weight    # [out_channels, in_channels, kH, kW]
        b = module.bias      # [out_channels] or None
        idx = torch.tensor(channel_indices, dtype=torch.long, device=W.device)
        W[idx, :, :, :] = 0.0
        if b is not None:
            b[idx] = 0.0

def prune_model_neurons(model: nn.Module, selection: Dict[str, List[int]]):
    """
    selection[layer_name] = [neuron_idx, ...]
    Applies in-place permanent pruning.
    """
    name2mod = dict(model.named_modules())
    for lname, idxs in selection.items():
        mod = name2mod.get(lname, None)
        if mod is None: 
            raise KeyError(f"Layer {lname} not found.")
        if isinstance(mod, nn.Linear):
            prune_linear_neurons(mod, idxs)
        elif isinstance(mod, nn.Conv2d):
            prune_conv_neurons(mod, idxs)
        else:
            raise TypeError(f"Unsupported module type for pruning: {type(mod)}")
