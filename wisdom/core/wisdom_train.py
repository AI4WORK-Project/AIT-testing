# core/wisdom_train.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Tuple

import math
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wisdom.utils.io_cache import save_layer_scores_csv
from wisdom.attribution.captum_backend import batch_per_layer_scores

# -----------------------------
# Config
# -----------------------------
@dataclass
class WisdomTrainConfig:
    methods: List[str] = field(default_factory=lambda: ["lrp", "ldl", "lig"])
    device: str = "cuda:0"
    voting_weights: Optional[List[float]] = None
    voting_mode: str = "fine-grained"  # "fine-grained" | "coarse"
    pruning_augmentations: Optional[List[Dict]] = None
    out_csv: Optional[str] = None


# -----------------------------
# Helper functions
# -----------------------------
def _is_trainable_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv2d, nn.Linear))

def _trainable_modules(model: nn.Module) -> Tuple[List[str], List[nn.Module]]:
    names, mods = [], []
    for n, m in model.named_modules():
        if _is_trainable_module(m):
            names.append(n)
            mods.append(m)
    return names, mods

def _criterion():
    return nn.CrossEntropyLoss()

def _eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    model.eval().to(device)
    x = x.to(device); y = y.to(device)
    with torch.no_grad():
        out = model(x)
        loss = _criterion()(out, y).item()
    return loss

def _voting_init(layer_scores: Dict[str, torch.Tensor],
                 trainable_names: List[str],
                 trainable_mods: List[nn.Module],
                 excluded_layer: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if layer_scores:
        return layer_scores
    for lname, m in zip(trainable_names, trainable_mods):
        if excluded_layer and lname == excluded_layer:
            continue
        if isinstance(m, nn.Conv2d):
            layer_scores[lname] = torch.zeros(m.out_channels, dtype=torch.float32)
        elif isinstance(m, nn.Linear):
            layer_scores[lname] = torch.zeros(m.out_features, dtype=torch.float32)
    return layer_scores

def _voting_neurons(layer_index_pairs: List[Tuple[str, int]],
                    layer_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Assign rank points (higher rank gets more points) like prepare_data.voting_neurons.
    Input list is assumed sorted DESC by importance; we intentionally enumerate reversed.
    """
    for rank, (layer_name, neuron_index) in enumerate(reversed(layer_index_pairs), start=1):
        if layer_name in layer_scores and 0 <= neuron_index < layer_scores[layer_name].numel():
            layer_scores[layer_name][neuron_index] += rank
    return layer_scores

def _weighted_top_neurons(important_neurons_dict: Dict[str, List[Tuple[str, float, int]]],
                          loss_gains: Dict[str, float],
                          top_k: int) -> List[Tuple[Tuple[str, int], float]]:
    """
    Same spirit as prepare_data.weighted_top_neurons:
      - normalize loss_gains to weights,
      - accumulate weight * score per (layer, idx),
      - return top_k ((layer, idx), weighted_score).
    """
    total_gain = sum(max(0.0, g) for g in loss_gains.values()) or 1.0
    weights = {m: max(0.0, g) / total_gain for m, g in loss_gains.items()}
    weighted_scores: Dict[Tuple[str, int], float] = {}
    for method, triples in important_neurons_dict.items():
        w = weights.get(method, 0.0)
        if w == 0.0:
            continue
        for (layer_name, score, idx) in triples:
            key = (layer_name, int(idx))
            weighted_scores[key] = weighted_scores.get(key, 0.0) + float(score) * w
    sorted_neurons = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_neurons[:top_k]


# -----------------------------
# Attribution (per-batch, Captum-friendly)
# -----------------------------
def _compute_batch_importance_captum(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    method: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-layer importance on ONE batch, reduced to per-neuron vectors.
    Returns {layer_name: tensor(#neurons,)}.
    """
    try:
        from captum.attr import (
            LayerLRP, LayerDeepLift, LayerIntegratedGradients, LayerGradientShap,
            LayerFeatureAblation, LayerActivation, InternalInfluence,
            LayerGradientXActivation, LayerGradCam
        )
    except Exception as e:
        raise RuntimeError(
            "Captum is required for _compute_batch_importance_captum. Please install captum or "
            "wire this to your attribution/registry."
        ) from e

    name2ctor = {
        "lrp": LayerLRP,
        "ldl": LayerDeepLift,
        "lig": LayerIntegratedGradients,
        "lgs": LayerGradientShap,
        "lfa": LayerFeatureAblation,
        "la":  LayerActivation,
        "ii":  InternalInfluence,
        "lgxa": LayerGradientXActivation,
        "lgc": LayerGradCam,
    }
    key = method.lower()
    if key not in name2ctor:
        raise ValueError(f"Unknown Captum method '{method}'. Options: {list(name2ctor.keys())}")

    model = model.to(device).eval()
    images = images.to(device); labels = labels.to(device)

    out: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for lname, layer in model.named_modules():
            if not _is_trainable_module(layer):
                continue
            A = name2ctor[key](model, layer)
            if key == "la":
                attr = A.attribute(images)
            elif key in ("ldl", "lgs"):
                attr = A.attribute(images, baselines=torch.zeros_like(images), target=labels)
            else:
                attr = A.attribute(images, target=labels)

            # reduce to per-neuron vector
            if attr.dim() == 4:  # (B,C,H,W)
                vec = attr.sum(dim=(0, 2, 3)).detach().cpu()
            else:                # (B,F)
                vec = attr.sum(dim=0).detach().cpu()
            out[lname] = vec

    return out


def _select_top_neurons_all(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_neurons: int,
    filter_layer: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """
    Flatten all layers' importance and pick top-M across layers (optionally excluding final layer).
    Returns:
      - indices_by_layer: {layer: 1D LongTensor of selected indices}
      - selected_triplets: list of (layer_name, score, idx) sorted desc by score
    """
    flattened: List[Tuple[str, float, int]] = []
    for layer_name, scores in importance_scores_dict.items():
        if filter_layer and layer_name == filter_layer:
            continue
        if scores.dim() == 1:
            for idx, s in enumerate(scores):
                flattened.append((layer_name, float(s.item()), int(idx)))
        else:
            # Should not happen because we reduce per layer, but keep safe
            mean_attr = scores.mean(dim=tuple(range(1, scores.dim())))
            for idx, s in enumerate(mean_attr):
                flattened.append((layer_name, float(s.item()), int(idx)))

    flattened.sort(key=lambda x: x[1], reverse=True)
    selected = flattened if top_m_neurons == -1 else flattened[:top_m_neurons]

    by_layer: Dict[str, List[int]] = {}
    for layer_name, _, idx in selected:
        by_layer.setdefault(layer_name, []).append(idx)

    indices_by_layer = {
        layer: torch.tensor(sorted(idxs), dtype=torch.long) for layer, idxs in by_layer.items()
    }
    return indices_by_layer, selected


# -----------------------------
# Pruning backends (new style)
# -----------------------------
class _WeightsGuard:
    def __init__(self, model: nn.Module):
        self.model = model
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def restore(self):
        self.model.load_state_dict(self.state, strict=True)

def _mask_prune_once(model: nn.Module, selection: Dict[str, List[int]], x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    from pruning.mask_pruning import mask_model_neurons
    handle = mask_model_neurons(model, selection)
    try:
        return _eval_loss(model, x, y, device)
    finally:
        try: handle.remove()
        except Exception: pass

def _weights_prune_once(model: nn.Module, selection: Dict[str, List[int]], x: torch.Tensor, y: torch.Tensor, device: str) -> float:
    from pruning.weights_pruning import prune_model_neurons
    guard = _WeightsGuard(model)
    try:
        prune_model_neurons(model, selection)
        return _eval_loss(model, x, y, device)
    finally:
        guard.restore()


# -----------------------------
# Main Trainer
# -----------------------------
class ConsensusWisdom:
    """
    Multi-method voting + optional pruning to identify important neurons.
    Steps per batch:
      1) Get important neurons for each attribution method.
      2) Identify optimal method by pruning its top neurons and measuring loss gain.
      3) Initialize voting buffers on first batch.
      4) Update votes:
         - fine-grained: pick TOP-K neurons across methods via weighted_top_neurons (weights from loss gains),
           then vote them by rank (like prepare_data.voting_neurons).
         - coarse: take only the optimal method's neurons and vote by rank.
      5) After all batches, save CSV.
    """

    def __init__(self, model: nn.Module, device: str = "cuda:0"):
        self.model = model
        self.device = device
        self.model.eval().to(device)
        self.trainable_names, self.trainable_mods = _trainable_modules(model)

    # -------- public API --------
    def fit(
        self,
        train_loader: DataLoader,
        cfg: WisdomTrainConfig,
        top_m_neurons: int,
        final_layer: Optional[str] = None,
        prune_mode: str = "mask",  # "mask" | "weights"
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Returns (layer_scores, csv_path).
        """
        assert cfg.out_csv, "Please provide cfg.out_csv to save layer scores."
        final_layer = final_layer or (self.trainable_names[-1] if self.trainable_names else None)

        layer_scores: Dict[str, torch.Tensor] = {}
        init_done = False

        # for images, labels in train_loader:
        for images, labels in tqdm(train_loader):
            # 1) Important neurons per method on THIS batch
            important_neurons_dict: Dict[str, List[Tuple[str, float, int]]] = {}
            for method in cfg.methods:
                imp = batch_per_layer_scores(
                    model=self.model,
                    images=images,
                    labels=labels,
                    device=cfg.device,
                    method=method,
                    target_layers=None, # all layers
                )
                _, selected_triplets = _select_top_neurons_all(
                    imp, top_m_neurons=top_m_neurons, filter_layer=final_layer
                )
                # list of (layer_name, score, idx)
                important_neurons_dict[method] = selected_triplets

            # 2) Identify optimal method by loss gain after pruning
            base_loss = _eval_loss(self.model, images, labels, cfg.device)

            loss_gains: Dict[str, float] = {}
            for method, triplets in important_neurons_dict.items():
                selection: Dict[str, List[int]] = {}
                for (layer_name, _score, idx) in triplets:
                    selection.setdefault(layer_name, []).append(int(idx))

                if prune_mode == "mask":
                    pruned_loss = _mask_prune_once(self.model, selection, images, labels, cfg.device)
                elif prune_mode == "weights":
                    pruned_loss = _weights_prune_once(self.model, selection, images, labels, cfg.device)
                else:
                    raise ValueError("prune_mode must be 'mask' or 'weights'")

                loss_gains[method] = pruned_loss - base_loss

            optimal_method = max(loss_gains, key=loss_gains.get)

            # 3) Voting buffers
            if not init_done:
                layer_scores = _voting_init(layer_scores, self.trainable_names, self.trainable_mods, excluded_layer=final_layer)
                init_done = True

            # 4) Vote according to mode
            if cfg.voting_mode == "coarse":
                # Optimal-only neurons, by rank
                opt_triplets = important_neurons_dict[optimal_method]
                # sort descending by raw score
                opt_triplets = sorted(opt_triplets, key=lambda t: t[1], reverse=True)
                layer_index_pairs = [(layer, idx) for (layer, _score, idx) in opt_triplets]
                _voting_neurons(layer_index_pairs, layer_scores)
            else:
                # fine-grained: across methods, weighted by loss gains → pick TOP-K → vote by rank
                top_across = _weighted_top_neurons(important_neurons_dict, loss_gains, top_k=top_m_neurons)
                layer_index_pairs = [pair for (pair, _wscore) in top_across]
                _voting_neurons(layer_index_pairs, layer_scores)

        # 5) Save CSV
        out_csv = save_layer_scores_csv(layer_scores, cfg.out_csv)
        return layer_scores, out_csv
