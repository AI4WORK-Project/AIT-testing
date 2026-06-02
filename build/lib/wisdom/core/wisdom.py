# wisdom/wisdom.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .activation import collect_per_neuron_series, collect_per_neuron_once
from .compute import combinations_coverage
from wisdom.utils.common import stable_selection_hash
from wisdom.clustering.assign import fit_per_neuron, assign_clusters


@dataclass
class ClusteringConfig:
    method: str = "KMeans"                 # KMeans, MeanShift, DBSCAN, Agglomerative, ...
    params: dict = field(default_factory=lambda: {"n_clusters": 2, "random_state": 42})
    use_silhouette: bool = False           # IDC-faithful k-search per neuron
    k_max: int = 10
    meanshift_quantile: float = 0.3
    meanshift_nsamples: int = 500

@dataclass
class WisdomConfig:
    top_m_neurons: int = 10
    test_all_classes: bool = True
    cache_path: Optional[str] = None       # cluster cache path


class WisdomIDC:
    """
    Unified IDC implementation:
      @impl="idc": KMeans + Silhouette per neuron (IDC-faithful)
      @impl="wisdom": pluggable clustering (factory), robust MeanShift, etc.
    """

    def __init__(self,
                 model: nn.Module,
                 impl: str = "wisdom",
                 cfg: Optional[WisdomConfig] = None,
                 cluster: Optional[ClusteringConfig] = None):
        self.model = model
        self.impl = impl.lower()
        self.cfg = cfg or WisdomConfig()
        self.cluster = cluster or ClusteringConfig()
        self.groups: Dict[str, Dict[int, dict]] = {}
        self.cluster_sizes: Dict[str, int] = {}

    # -------- selection --------
    @staticmethod
    def _flatten_scores(layer_scores: Dict[str, torch.Tensor], exclude_last: Optional[str] = None):
        flat = []
        for name, s in layer_scores.items():
            if exclude_last and name == exclude_last:
                continue
            if s.dim() == 1:
                for i, v in enumerate(s): flat.append((name, float(v), int(i)))
            else:
                m = s.mean(dim=tuple(range(1, s.dim()))) if s.dim() > 1 else s
                for i, v in enumerate(m): flat.append((name, float(v), int(i)))
        return sorted(flat, key=lambda x: x[1], reverse=True)

    def select_top_neurons(self,
                           layer_scores: Dict[str, torch.Tensor],
                           exclude_last: Optional[str] = None) -> Dict[str, List[int]]:
        flat = self._flatten_scores(layer_scores, exclude_last)
        if self.cfg.top_m_neurons > 0:
            flat = flat[:self.cfg.top_m_neurons]
        sel: Dict[str, List[int]] = {}
        for lname, _, idx in flat:
            sel.setdefault(lname, []).append(int(idx))
        return sel

    # -------- clustering fit --------
    def _cache_tag(self, selected: Dict[str, List[int]]) -> str:
        return stable_selection_hash(selected, self.impl, self.cluster)

    def fit_clusters(self,
                     build_loader: DataLoader,
                     selected: Dict[str, List[int]],
                     device: str = "cuda:0") -> None:
        series = collect_per_neuron_series(self.model, build_loader, selected, device=device)
        self.groups = fit_per_neuron(
            series,
            method=self.cluster.method,
            params=self.cluster.params,
            use_silhouette=(self.impl == "idc") or self.cluster.use_silhouette,
            k_max=self.cluster.k_max,
            meanshift_q=self.cluster.meanshift_quantile,
            meanshift_ns=self.cluster.meanshift_nsamples,
            cache_path=self.cfg.cache_path,
            cache_tag=self._cache_tag(selected),
            selected=selected,
        )
        self.cluster_sizes = {f"{l}:{i}": self.groups[l][i]["centers"].shape[0]
                              for l in self.groups for i in self.groups[l]}
        
    # -------- coverage --------

    def coverage(self,
                 test_loader: DataLoader,
                 selected: Dict[str, List[int]],
                 device: str = "cuda:0") -> Tuple[float, float, int]:
        assignments = []
        with torch.no_grad():
            for x, _ in test_loader:
                for b in range(x.size(0)):
                    acts = collect_per_neuron_once(self.model, x[b:b+1], selected, device=device)
                    assn = assign_clusters(self.groups, acts)
                    flat = {f"{l}:{i}": assn[l][i] for l in assn for i in assn[l]}
                    assignments.append(flat)
        return combinations_coverage(assignments, self.cluster_sizes)
