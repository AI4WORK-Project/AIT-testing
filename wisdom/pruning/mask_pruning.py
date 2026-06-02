# wisdom/pruning/mask_pruning.py
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn

class MaskHandles:
    def __init__(self, handles): self._handles = handles
    def remove(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass

def _hook_linear(idxs: List[int]):
    idx = torch.tensor(sorted(set(idxs)), dtype=torch.long)
    def hook(_mod, _inp, out: torch.Tensor):
        # out: (B, F)
        if out.dim() != 2: return out
        out[:, idx.to(out.device)] = 0.0
        return out
    return hook

def _hook_conv(idxs: List[int]):
    idx = torch.tensor(sorted(set(idxs)), dtype=torch.long)
    def hook(_mod, _inp, out: torch.Tensor):
        # out: (B, C, H, W)
        if out.dim() != 4: return out
        out[:, idx.to(out.device), :, :] = 0.0
        return out
    return hook

def mask_model_neurons(model: nn.Module, selection: Dict[str, List[int]]) -> MaskHandles:
    """
    selection[layer_name] = [neuron_idx, ...]
    Returns a handle object with .remove() to unmask.
    """
    name2mod = dict(model.named_modules())
    handles = []
    for lname, idxs in selection.items():
        mod = name2mod.get(lname, None)
        if mod is None: 
            raise KeyError(f"Layer {lname} not found.")
        if isinstance(mod, nn.Linear):
            h = mod.register_forward_hook(lambda m,i,o, idxs=idxs: _hook_linear(idxs)(m,i,o))
        elif isinstance(mod, nn.Conv2d):
            h = mod.register_forward_hook(lambda m,i,o, idxs=idxs: _hook_conv(idxs)(m,i,o))
        else:
            raise TypeError(f"Unsupported module type for masking: {type(mod)}")
        handles.append(h)
    return MaskHandles(handles)