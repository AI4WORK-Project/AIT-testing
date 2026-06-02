# wisdom/attribution/base.py
from __future__ import annotations
from typing import Dict, Iterable, Optional
import torch
from torch.utils.data import DataLoader

class AttributionMethod:
    """
    Plugin interface for attribution methods.

    Implement EITHER:
      - attribute_model(): to compute per-layer relevance in one shot, OR
      - attribute_layer(): to compute relevance for a specific layer.

    You can implement both; the framework will use the most specific one.
    """
    name: str = "base"

    def attribute_model(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda:0",
        target_layers: Optional[Iterable[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Return {layer_name: relevance per neuron (1D per layer)} aggregated over dataloader.
        If target_layers is provided, return only those.
        """
        raise NotImplementedError

    def attribute_layer(
        self,
        model: torch.nn.Module,
        layer_name: str,
        dataloader: DataLoader,
        device: str = "cuda:0",
        **kwargs
    ) -> torch.Tensor:
        """
        Return relevance vector (1D) for the given layer, aggregated over dataloader.
        """
        raise NotImplementedError