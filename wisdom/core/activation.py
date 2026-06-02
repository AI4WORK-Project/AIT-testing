# src/wisdom/activation.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _spatial_mean(x: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W)->(B,C); (B,F)->(B,F)
    return x.mean(dim=(2,3)) if x.dim() == 4 else x


def collect_per_neuron_series(model: nn.Module,
                              dataloader: DataLoader,
                              selected: Dict[str, List[int]],
                              device: str = "cuda:0") -> Dict[str, Dict[int, np.ndarray]]:
    """
    Forward all samples once; keep only selected neurons.
    Returns: {layer: {idx: np.array(N,)}} where N = #samples
    """
    model.eval().to(device)
    name2mod = dict(model.named_modules())
    buffers: Dict[str, Dict[int, List[float]]] = {l: {i: [] for i in idxs} for l, idxs in selected.items()}
    handles = []

    def make_hook(layer_name: str, keep_idx: List[int]):
        idx = torch.tensor(sorted(set(keep_idx)), dtype=torch.long)
        def hook(_m, _inp, out: torch.Tensor):
            with torch.no_grad():
                v = _spatial_mean(out)               # (B,C) or (B,F)
                v = v[:, idx.to(v.device)]           # (B,K)
                cpu = v.detach().cpu()
                for col, j in enumerate(idx.tolist()):
                    buffers[layer_name][j].extend(cpu[:, col].tolist())
            return out
        return hook

    for lname, idxs in selected.items():
        mod = name2mod.get(lname, None)
        if mod is None: raise KeyError(f"Layer {lname} not found")
        handles.append(mod.register_forward_hook(make_hook(lname, idxs)))

    with torch.no_grad():
        for x, _ in dataloader:
            _ = model(x.to(device))

    for h in handles:
        try: h.remove()
        except Exception: pass

    out: Dict[str, Dict[int, np.ndarray]] = {}
    for lname, dct in buffers.items():
        out[lname] = {i: np.asarray(vals, dtype=np.float64) for i, vals in dct.items()}
    return out


def collect_per_neuron_once(model: nn.Module,
                            batch: torch.Tensor,
                            selected: Dict[str, List[int]],
                            device: str = "cuda:0") -> Dict[str, Dict[int, float]]:
    """
    Single forward (assume batch size 1 slice provided).
    Returns {layer:{idx: scalar}}
    """
    model.eval().to(device)
    assert batch.size(0) >= 1
    x = batch.to(device)

    name2mod = dict(model.named_modules())
    sample_buf: Dict[str, Dict[int, float]] = {l: {} for l in selected}
    handles = []

    def make_hook(layer_name: str, keep_idx: List[int]):
        idx = torch.tensor(sorted(set(keep_idx)), dtype=torch.long)
        def hook(_m, _inp, out: torch.Tensor):
            with torch.no_grad():
                v = _spatial_mean(out)         # (B,C) or (B,F)
                v0 = v[0:1, idx.to(v.device)]  # first sample only
                cpu = v0.detach().cpu().squeeze(0)  # (K,)
                for col, j in enumerate(idx.tolist()):
                    sample_buf[layer_name][j] = float(cpu[col].item())
            return out
        return hook
    

    for lname, idxs in selected.items():
        mod = name2mod.get(lname, None)
        if mod is None: raise KeyError(f"Layer {lname} not found")
        handles.append(mod.register_forward_hook(make_hook(lname, idxs)))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        try: h.remove()
        except Exception: pass

    return sample_buf
