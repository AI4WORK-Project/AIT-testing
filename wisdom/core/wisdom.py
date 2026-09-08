# wisdom/wisdom.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Mapping, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .activation import collect_per_neuron_series, collect_per_neuron_once
from .compute import combinations_coverage
from .layers import build_layer_plan, get_group_names as _get_group_names
from .task import extract_model_inputs
from wisdom.utils.common import stable_selection_hash
from wisdom.clustering.assign import fit_per_neuron, assign_clusters


def get_group_names(n_groups: int = 3) -> tuple[str, ...]:
    return _get_group_names(n_groups)


def build_layer_groups(layer_names: Iterable[str], n_groups: int = 3) -> Dict[str, List[str]]:
    plan = build_layer_plan(tuple(layer_names), n_groups=n_groups)
    return {group_name: list(names) for group_name, names in plan.groups.items()}


def _load_scores_frame(
    csv_path: str,
    *,
    strip_prefix: str = 'yolo_model.',
    exclude_prefixes: Optional[Iterable[str]] = None,
    exclude_layers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df['__layer_name__'] = [
        layer_name.replace(strip_prefix, '') if strip_prefix and str(layer_name).startswith(strip_prefix) else str(layer_name)
        for layer_name in df['LayerName']
    ]
    if exclude_prefixes:
        prefixes = tuple(str(prefix) for prefix in exclude_prefixes)
        df = df[~df['__layer_name__'].str.startswith(prefixes)]
    if exclude_layers:
        blocked = {str(layer_name) for layer_name in exclude_layers}
        df = df[~df['__layer_name__'].isin(blocked)]
    return df


def load_layerwise_top_neurons(
    csv_path: str,
    per_layer_k: int = 5,
    strip_prefix: str = 'yolo_model.',
    *,
    exclude_prefixes: Optional[Iterable[str]] = None,
    exclude_layers: Optional[Iterable[str]] = None,
) -> Dict[str, List[int]]:
    if per_layer_k <= 0:
        raise ValueError(f'per_layer_k must be positive, got {per_layer_k}')
    df = _load_scores_frame(
        csv_path,
        strip_prefix=strip_prefix,
        exclude_prefixes=exclude_prefixes,
        exclude_layers=exclude_layers,
    )
    neurons: Dict[str, List[int]] = {}
    for layer_name, group in df.groupby('__layer_name__'):
        top = group.nlargest(per_layer_k, 'Score')
        top = top[top['Score'] > 0]
        if len(top) > 0:
            neurons[str(layer_name)] = sorted(int(index) for index in top['NeuronIndex'].tolist())
    return neurons


def load_groupwise_top_neurons(
    csv_path: str,
    per_group_k: int = 5,
    strip_prefix: str = 'yolo_model.',
    *,
    n_groups: int = 3,
    exclude_prefixes: Optional[Iterable[str]] = None,
    exclude_layers: Optional[Iterable[str]] = None,
    layer_groups: Mapping[str, Iterable[str]] | None = None,
) -> Dict[str, List[int]]:
    if per_group_k <= 0:
        raise ValueError(f'per_group_k must be positive, got {per_group_k}')
    df = _load_scores_frame(
        csv_path,
        strip_prefix=strip_prefix,
        exclude_prefixes=exclude_prefixes,
        exclude_layers=exclude_layers,
    )
    if layer_groups is None:
        resolved_groups: Dict[str, List[str]] = build_layer_groups(
            df['__layer_name__'].unique().tolist(),
            n_groups=n_groups,
        )
    else:
        resolved_groups = {
            str(group_name): [str(layer_name) for layer_name in layer_names]
            for group_name, layer_names in layer_groups.items()
        }
        flattened = [layer_name for names in resolved_groups.values() for layer_name in names]
        duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
        if duplicates:
            raise ValueError(f'Layers appear in more than one group: {duplicates}')
        stale_layers = sorted(set(df['__layer_name__']) - set(flattened))
        if stale_layers:
            raise ValueError(f'Score CSV contains layers not eligible for this model: {stale_layers}')
    layer_to_group = {
        layer_name: group_name
        for group_name, layer_names in resolved_groups.items()
        for layer_name in layer_names
    }
    df = df[df['Score'] > 0].copy()
    df['__group_name__'] = df['__layer_name__'].map(layer_to_group)

    neurons: Dict[str, List[int]] = {}
    for group_name in resolved_groups:
        group = df[df['__group_name__'] == group_name]
        if group.empty:
            continue
        top = group.nlargest(per_group_k, 'Score')
        for layer_name, layer_rows in top.groupby('__layer_name__'):
            neurons.setdefault(str(layer_name), []).extend(int(index) for index in layer_rows['NeuronIndex'].tolist())

    return {
        layer_name: sorted(dict.fromkeys(indices))
        for layer_name, indices in neurons.items()
    }


def split_selected_by_layer(selected: Dict[str, List[int]]) -> Dict[str, Dict[str, List[int]]]:
    return {
        layer_name: {layer_name: list(indices)}
        for layer_name, indices in selected.items()
        if indices
    }


def split_selected_by_group(
    selected: Dict[str, List[int]],
    n_groups: int = 3,
    *,
    layer_groups: Mapping[str, Iterable[str]] | None = None,
) -> Dict[str, Dict[str, List[int]]]:
    if layer_groups is None:
        layer_groups = build_layer_groups(selected.keys(), n_groups=n_groups)
    else:
        layer_groups = {
            str(group_name): [str(layer_name) for layer_name in layer_names]
            for group_name, layer_names in layer_groups.items()
        }
    scopes: Dict[str, Dict[str, List[int]]] = {group_name: {} for group_name in layer_groups}
    layer_to_group = {
        layer_name: group_name
        for group_name, layer_names in layer_groups.items()
        for layer_name in layer_names
    }
    for layer_name, indices in selected.items():
        if not indices:
            continue
        group_name = layer_to_group.get(layer_name)
        if group_name is None:
            continue
        scopes[group_name][layer_name] = list(indices)
    return {group_name: scope for group_name, scope in scopes.items() if scope}


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
    selection_mode: str = 'global'
    n_groups: int = 3
    layer_groups: Optional[Dict[str, Tuple[str, ...]]] = None


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

    def select_top_neurons_all(
        self,
        layer_scores: Dict[str, torch.Tensor],
        exclude_last: Optional[str] = None,
    ) -> Dict[str, List[int]]:
        return self.select_top_neurons(layer_scores, exclude_last=exclude_last)

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

    @property
    def total_combination(self) -> int:
        total = 1
        for size in self.cluster_sizes.values():
            total *= int(size)
        return total

    def fit(
        self,
        build_loader: DataLoader,
        layer_scores: Dict[str, torch.Tensor],
        exclude_last: Optional[str] = None,
        device: str = "cuda:0",
    ) -> Dict[str, List[int]]:
        selected = self.select_top_neurons(layer_scores, exclude_last=exclude_last)
        self.fit_clusters(build_loader, selected, device=device)
        return selected

    def fit_selected(
        self,
        build_loader: DataLoader,
        selected: Dict[str, Union[torch.Tensor, List[int]]],
        device: str = "cuda:0",
    ) -> Dict[str, List[int]]:
        normalized = {
            layer: indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
            for layer, indices in selected.items()
        }
        self.fit_clusters(build_loader, normalized, device=device)
        return normalized

    @staticmethod
    def _normalize_selected(selected: Dict[str, Union[torch.Tensor, List[int]]]) -> Dict[str, List[int]]:
        return {
            layer: indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
            for layer, indices in selected.items()
        }

    def _scope_keys(self, selected: Dict[str, List[int]]) -> Dict[str, List[str]]:
        if self.cfg.selection_mode in {'per-layer', 'single-layer'}:
            return {
                layer_name: [f'{layer_name}:{index}' for index in indices]
                for layer_name, indices in selected.items()
                if indices
            }
        if self.cfg.selection_mode == 'per-group':
            grouped = split_selected_by_group(
                selected,
                n_groups=self.cfg.n_groups,
                layer_groups=self.cfg.layer_groups,
            )
            return {
                group_name: [
                    f'{layer_name}:{index}'
                    for layer_name, indices in group_selected.items()
                    for index in indices
                ]
                for group_name, group_selected in grouped.items()
            }
        return {
            'overall': [
                f'{layer_name}:{index}'
                for layer_name, indices in selected.items()
                for index in indices
            ]
        }

    def _collect_assignments(
        self,
        test_loader: DataLoader,
        selected: Dict[str, List[int]],
        device: str = 'cuda:0',
    ) -> List[Dict[str, int]]:
        assignments: List[Dict[str, int]] = []
        with torch.no_grad():
            for raw_batch in test_loader:
                x = extract_model_inputs(raw_batch)
                for batch_index in range(x.size(0)):
                    acts = collect_per_neuron_once(self.model, x[batch_index:batch_index + 1], selected, device=device)
                    assigned = assign_clusters(self.groups, acts)
                    assignments.append({f'{layer}:{index}': assigned[layer][index] for layer in assigned for index in assigned[layer]})
        return assignments

    # ------- coverage details --------
    def coverage_details(
        self,
        test_loader: DataLoader,
        selected: Optional[Dict[str, Union[torch.Tensor, List[int]]]] = None,
        device: str = 'cuda:0',
        layer_scores: Optional[Dict[str, torch.Tensor]] = None,
        exclude_last: Optional[str] = None,
    ) -> Dict[str, Union[float, int, Dict[str, Dict[str, Union[float, int]]]]]:
        """
        Reports:
            - overall coverage plus per-scope rate, 
            - combinations, 
            - maximum possible coverage
            - monitored-neuron count
            Non-global overall values are means across scopes.
        """
        if selected is None:
            if layer_scores is None:
                raise ValueError('Either selected or layer_scores must be provided.')
            selected = self.select_top_neurons(layer_scores, exclude_last=exclude_last)

        normalized = self._normalize_selected(selected)
        assignments = self._collect_assignments(test_loader, normalized, device=device)
        scope_keys = self._scope_keys(normalized)

        scope_details: Dict[str, Dict[str, Union[float, int]]] = {}
        scope_rates: List[float] = []
        scope_totals: List[int] = []
        scope_maxima: List[float] = []

        for scope_name, keys in scope_keys.items():
            scope_sizes = {key: self.cluster_sizes[key] for key in keys if key in self.cluster_sizes}
            scope_assignments = [{key: assignment[key] for key in keys if key in assignment} for assignment in assignments]
            if scope_sizes:
                rate, total, max_coverage = combinations_coverage(scope_assignments, scope_sizes)
            else:
                rate, total, max_coverage = 0.0, 0, 0.0
            scope_details[scope_name] = {
                'coverage_rate': float(rate),
                'total_combinations': int(total),
                'max_coverage': float(max_coverage),
                'monitored_neurons': len(keys),
            }
            scope_rates.append(float(rate))
            scope_totals.append(int(total))
            scope_maxima.append(float(max_coverage))

        if self.cfg.selection_mode == 'global' and 'overall' in scope_details:
            overall_rate = float(scope_details['overall']['coverage_rate'])
            overall_total = int(scope_details['overall']['total_combinations'])
            overall_max = float(scope_details['overall']['max_coverage'])
        else:
            overall_rate = float(np.mean(scope_rates)) if scope_rates else 0.0
            overall_total = int(round(float(np.mean(scope_totals)))) if scope_totals else 0
            overall_max = float(np.mean(scope_maxima)) if scope_maxima else 0.0

        return {
            'coverage_rate': overall_rate,
            'total_combinations': overall_total,
            'max_coverage': overall_max,
            'scope_details': scope_details,
        }
        
    # -------- coverage --------

    def coverage(self,
                 test_loader: DataLoader,
                 selected: Optional[Dict[str, Union[torch.Tensor, List[int]]]] = None,
                 device: str = "cuda:0",
                 layer_scores: Optional[Dict[str, torch.Tensor]] = None,
                 exclude_last: Optional[str] = None) -> Tuple[float, int, float]:
        if selected is None:
            if layer_scores is None:
                raise ValueError("Either selected or layer_scores must be provided.")
            selected = self.select_top_neurons(layer_scores, exclude_last=exclude_last)
        details = self.coverage_details(
            test_loader,
            selected=selected,
            device=device,
            layer_scores=layer_scores,
            exclude_last=exclude_last,
        )
        return details['coverage_rate'], details['total_combinations'], details['max_coverage']

    def save_to_json(
        self,
        coverage_rate: float,
        max_coverage: float,
        model_name: str,
        testing_layer: str,
        file_path: str = "coverage_rate.json",
    ) -> None:
        import json

        with open(file_path, "w") as handle:
            json.dump(
                {
                    "Total Combination": self.total_combination,
                    "Max Coverage": max_coverage,
                    "Coverage Rate": coverage_rate,
                    "Model Name": model_name,
                    "Testing Layer": testing_layer,
                },
                handle,
            )
