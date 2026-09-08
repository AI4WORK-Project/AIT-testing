# wisdom/attribution/captum_backend.py
from typing import Dict, Tuple, List, Iterable, Optional, Callable, Union
import torch
from torch.utils.data import DataLoader
from captum.attr import (
    LayerConductance, LayerActivation, InternalInfluence, LayerGradientXActivation,
    LayerGradCam, LayerDeepLift, LayerDeepLiftShap, LayerGradientShap,
    LayerIntegratedGradients, LayerFeatureAblation, LayerLRP
)

from .base import AttributionMethod

ATTRS = {
    "lc": LayerConductance, 
    "la": LayerActivation, 
    "ii": InternalInfluence,
    "lgxa": LayerGradientXActivation, 
    "lgc": LayerGradCam, 
    "ldl": LayerDeepLift,
    "ldls": LayerDeepLiftShap, 
    "lgs": LayerGradientShap, 
    "lig": LayerIntegratedGradients,
    "lfa": LayerFeatureAblation, 
    "lrp": LayerLRP,
}

# Dispatch function for different Captum methods.
def _call_dispatch(name: str) -> Callable:
    """Return a call(images, target)->attr function tailored to the method."""
    key = name.lower()
    def _call(A, images, target):
        if key == "la":
            return A.attribute(images)
        elif key in ("ldl", "ldls", "lgs"):
            kwargs = {"baselines": torch.zeros_like(images)}
            if target is not None:
                kwargs["target"] = target
            return A.attribute(images, **kwargs)
        kwargs = {} if target is None else {"target": target}
        return A.attribute(images, **kwargs)
    return _call

def _accumulate_layer_attr(
    A_obj, 
    call_attr: Callable, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: str, 
    target_is_label=True) -> torch.Tensor:
    agg = None
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device) if target_is_label else None
        # call the provided function to compute attributions
        attr = call_attr(A_obj, images, labels)
        # collapse batch then spatial dims to per-neuron vector
        # attr shapes: (B, C) or (B, C, H, W)
        if attr.dim() == 4:
            vec = attr.sum(dim=(0,2,3)).detach().cpu()
        else:
            vec = attr.sum(dim=0).detach().cpu()
        agg = vec if agg is None else (agg + vec)
    return agg / len(dataloader.dataset)

class CaptumLayerAttribution(AttributionMethod):
    """Generic wrapper for Captum's per-layer attribution classes."""
    def __init__(self, name: str, ctor, call):
        self.name = name.lower()
        self._ctor = ctor
        self._call = call

    def _candidate_layers(self, model) -> Iterable[tuple[str, torch.nn.Module]]:
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                yield n, m

    def attribute_model(self, model, dataloader, device="cuda:0",
                        target_layers: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, torch.Tensor]:
        model = model.to(device)
        names = set(target_layers) if target_layers else None
        out: Dict[str, torch.Tensor] = {}
        for lname, layer in self._candidate_layers(model):
            if names and lname not in names:
                continue
            A = self._ctor(model, layer)
            out[lname] = _accumulate_layer_attr(A, self._call, model, dataloader, device)
        return out
    
    def attribute_layer(self, model, layer_name, dataloader, device="cuda:0", **kwargs) -> torch.Tensor:
        model = model.to(device)
        layer = dict(model.named_modules())[layer_name]
        A = self._ctor(model, layer)
        return _accumulate_layer_attr(A, self._call, model, dataloader, device)

    def attribute_batch(self, model, images, labels=None, device="cuda:0",
                        target_layers: Optional[Iterable[str]] = None,
                        target=None, **kwargs) -> Dict[str, torch.Tensor]:
        """One-batch version: returns {layer_name: per-neuron vector}."""
        if labels is not None and target is not None:
            raise ValueError("Pass either labels or target, not both.")
        resolved_target = labels if target is None else target
        model = model.to(device).eval()
        names = set(target_layers) if target_layers else None
        images = images.to(device)
        if isinstance(resolved_target, torch.Tensor):
            resolved_target = resolved_target.to(device)

        out: Dict[str, torch.Tensor] = {}
        for lname, layer in self._candidate_layers(model):
            if names and lname not in names:
                continue
            A = self._ctor(model, layer)
            attr = self._call(A, images, resolved_target)
            if attr.dim() == 4:
                vec = attr.sum(dim=(0,2,3)).detach().cpu()
            else:
                vec = attr.sum(dim=0).detach().cpu()
            out[lname] = vec
        return out

def per_layer_scores(model: torch.nn.Module, dataloader: DataLoader, device: str, method: str,
                     target_layers: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    model = model.to(device).eval()
    scores: Dict[str, torch.Tensor] = {}
    layers: List[Tuple[str, torch.nn.Module]] = [
        (n, m) for n, m in model.named_modules()
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
    ]
    call = _call_dispatch(method)
    for name, layer in layers:
        if target_layers and name not in set(target_layers):
            continue
        A = ATTRS[method](model, layer)
        agg = None
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            attr = call(A, images, labels)
            if attr.dim() == 4:
                part = attr.sum(dim=(0,2,3)).detach().cpu()
            else:
                part = attr.sum(dim=0).detach().cpu()
            agg = part if agg is None else (agg + part)
        scores[name] = agg / len(dataloader.dataset)
    return scores

def get(method: str) -> CaptumLayerAttribution:
    """Return a CaptumLayerAttribution for the given method key (e.g., 'lrp', 'ldl', ...)."""
    key = method.lower()
    if key not in ATTRS:
        raise ValueError(f"Unknown Captum method '{method}'. Options: {list(ATTRS.keys())}")
    return CaptumLayerAttribution(name=key, ctor=ATTRS[key], call=_call_dispatch(key))

def batch_per_layer_scores(model: torch.nn.Module,
                           images: torch.Tensor,
                           labels: torch.Tensor | None = None,
                           device: str = "cuda:0",
                           method: str = "lgxa",
                           target_layers: Optional[Iterable[str]] = None,
                           *,
                           target: int | torch.Tensor | None = None) -> Dict[str, torch.Tensor]:
    """One-batch attribution via CaptumLayerAttribution."""
    return get(method).attribute_batch(
        model,
        images,
        labels,
        device=device,
        target_layers=target_layers,
        target=target,
    )


def adaptive_per_layer_scores(model: torch.nn.Module,
                              data: Union[DataLoader, Tuple[torch.Tensor, torch.Tensor]],
                              device: str,
                              method: str,
                              target_layers: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """
    If `data` is a DataLoader → average over dataset (like per_layer_scores).
    If `data` is (images, labels) → single-batch attribution (like batch_per_layer_scores).
    """
    if isinstance(data, DataLoader):
        return get(method).attribute_model(model, data, device=device, target_layers=target_layers)
    else:
        images, labels = data
        return get(method).attribute_batch(model, images, labels, device=device, target_layers=target_layers)

# def per_layer_scores(model: torch.nn.Module,
#                      dataloader: DataLoader,
#                      device: str,
#                      method: str) -> Dict[str, torch.Tensor]:
#     """Return {layer_name: relevance per neuron (C,) or (out_channels,)} over the dataloader."""
#     model = model.to(device).eval()
#     scores: Dict[str, torch.Tensor] = {}
#     # discover candidate layers
#     layers: List[Tuple[str, torch.nn.Module]] = [
#         (n, m) for n, m in model.named_modules()
#         if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
#     ]
#     for name, layer in layers:
#         A = ATTRS[method](model, layer)
#         agg = None
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             if method in ("ldl", "ldls", "lgs"):
#                 attr = A.attribute(images, baselines=torch.zeros_like(images), target=labels)
#             elif method == "la":
#                 attr = A.attribute(images)
#             else:
#                 attr = A.attribute(images, target=labels)
#             if attr.dim() == 4:
#                 part = attr.sum(dim=(0,2,3)).detach().cpu()
#             else:
#                 part = attr.sum(dim=0).detach().cpu()
#             agg = part if agg is None else (agg + part)
#         if isinstance(layer, torch.nn.Linear):
#             scores[name] = agg / len(dataloader.dataset)
#         else:
#             scores[name] = agg / len(dataloader.dataset)
#     return scores
