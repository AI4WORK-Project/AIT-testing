# wisdom/attribution/registry.py
from __future__ import annotations
from typing import Dict, List, Type
import torch

from .captum_backend import per_layer_scores
from .base import AttributionMethod

REGISTRY: Dict[str, Type[AttributionMethod]] = {}

def register(name: str):
    def deco(cls: Type[AttributionMethod]):
        REGISTRY[name.lower()] = cls
        cls.name = name.lower()
        return cls
    return deco

def get(name: str) -> Type[AttributionMethod]:
    try:
        return REGISTRY[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown attribution method '{name}'. Registered: {list(REGISTRY)}")

def ensemble_scores(model, dataloader, device, methods: List[str]) -> Dict[str, torch.Tensor]:
    acc: Dict[str, torch.Tensor] = {}
    for m in methods:
        s = per_layer_scores(model, dataloader, device, m)
        for k,v in s.items():
            acc[k] = v.clone() if k not in acc else (acc[k] + v)
    # simple normalization (you can swap in your pruning-weighted voting)
    for k in acc:
        t = acc[k]; denom = t.abs().sum().clamp_min(1e-12)
        acc[k] = (t / denom) * torch.numel(t)
    return acc