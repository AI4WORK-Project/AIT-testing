# wisdom/clustering/assign.py
from __future__ import annotations
from fileinput import filename
from typing import Dict, Optional, List

import numpy as np
import pickle
from sklearn.metrics import silhouette_score, pairwise_distances_argmin
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

from .factory import make
from wisdom.utils.io_cache import save_cluster_groups, load_cluster_groups


def best_k_silhouette(values: np.ndarray, k_max=10, random_state=42) -> int:
    X = values.reshape(-1, 1)
    uniq = np.unique(values)
    if uniq.size <= 1:
        return 1
    k_max = int(max(2, min(k_max, max(2, uniq.size))))
    best_k, best_s = 2, -1.0
    for k in range(2, k_max+1):
        try:
            km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels, metric="euclidean")
            if s > best_s:
                best_s, best_k = s, k
        except Exception:
            continue
    return max(1, best_k)

def robust_bandwidth(values: np.ndarray, quantile=0.3, n_samples=500) -> float:
    X = values.reshape(-1, 1)
    q = float(np.clip(quantile, 1e-3, 0.99))
    n_samp = int(min(max(50, int(n_samples)), len(values)))
    try:
        bw = estimate_bandwidth(X, quantile=q, n_samples=n_samp)
        if not np.isfinite(bw) or bw <= 0:
            raise ValueError
    except Exception:
        std = float(np.std(values))
        n = len(values)
        bw = 1.06 * std * (n ** (-1/5)) if std > 0 else 1.0
    return float(bw)

def ensure_centers(model, X: np.ndarray):
    if hasattr(model, "cluster_centers_"):
        return model.cluster_centers_
    labels = getattr(model, "labels_", None)
    if labels is None:
        raise RuntimeError("Estimator lacks predict(), labels_, and cluster_centers_.")
    L = np.asarray(labels)
    centers = np.vstack([X[L==c].mean(axis=0) for c in np.unique(L)])
    model.cluster_centers_ = centers
    return centers

def safe_predict(model, X: np.ndarray):
    if hasattr(model, "predict"):
        return model.predict(X)
    centers = ensure_centers(model, X)
    return pairwise_distances_argmin(X, centers)


def groups_from_legacy_pickle(
    pkl_path: str,
    selected: Dict[str, List[int]],
) -> Dict[str, Dict[int, dict]]:
    with open(pkl_path, "rb") as f:
        legacy = pickle.load(f)

    groups: Dict[str, Dict[int, dict]] = {}
    for layer, est_list in legacy.items():
        # if the legacy file has clusters for a subset, we expect selected[layer]
        # to have the same length and order.
        sel_ids = selected.get(layer, [])
        if len(sel_ids) != len(est_list):
            pass

        groups[layer] = {}
        for pos, est in enumerate(est_list):
            if pos >= len(sel_ids):
                break
            neuron_id = int(sel_ids[pos])

            method = getattr(est, "__class__", type(est)).__name__
            try:
                params = est.get_params(deep=False)
            except Exception:
                params = {}

            centers = getattr(est, "cluster_centers_", None)
            if centers is None:
                centers = np.array([[0.0]], dtype=np.float64)
            else:
                centers = np.asarray(centers, dtype=np.float64)
                if centers.ndim == 1:
                    centers = centers.reshape(-1, 1)
                elif centers.shape[1] != 1:
                    centers = centers[:, :1].astype(np.float64)

            labels = getattr(est, "labels_", None)
            labels = np.asarray(labels, dtype=np.int32) if labels is not None else np.empty((0,), dtype=np.int32)

            groups[layer][neuron_id] = {
                "method": method,
                "params": params,
                "centers": centers,
                "labels": labels,
            }
    return groups

def fit_per_neuron(
    activations: Dict[str, Dict[int, np.ndarray]],
    method: str = "KMeans",
    params: Optional[dict] = None,
    use_silhouette: bool = False,
    k_max: int = 10,
    meanshift_q: float = 0.3,
    meanshift_ns: int = 500,
    cache_path: Optional[str] = None,
    cache_tag: Optional[str] = None,
    selected: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Dict[int, dict]]:
    """
    activations[layer][idx] = np.array(N,)
    Returns groups[layer][idx] = {"method","params","centers:(C,1)","labels:(N,)"}
    """
    if cache_tag or cache_path:
        if cache_path is not None:
            if cache_path.lower().endswith(".pkl"):
                cached = groups_from_legacy_pickle(cache_path, selected)
            else: 
                cached = load_cluster_groups(cache_path, cache_tag)
            if cached is not None:
                print(f"[INFO] Cached clusters loaded from {cache_path}")
                return cached

    params = dict(params or {})
    groups: Dict[str, Dict[int, dict]] = {}
    for layer, dct in activations.items():
        groups[layer] = {}
        for idx, vals in dct.items():
            v = np.asarray(vals, dtype=np.float64).reshape(-1)
            if np.unique(v).size <= 1:
                centers = np.array([[float(v[0])]], dtype=np.float64)
                labels = np.zeros((v.shape[0],), dtype=np.int32)
                groups[layer][idx] = {"method": "Trivial", "params": {},
                                      "centers": centers, "labels": labels}
                continue

            m = method.lower()
            if m == "kmeans" and use_silhouette:
                k = best_k_silhouette(v, k_max=k_max, random_state=int(params.get("random_state", 42)))
                est = KMeans(n_clusters=k, random_state=int(params.get("random_state", 42)), n_init="auto")
                labels = est.fit_predict(v.reshape(-1,1))
                centers = est.cluster_centers_
            elif m == "meanshift":
                p = params.copy()
                if "bandwidth" not in p or p["bandwidth"] in (None, 0, "auto"):
                    p["bandwidth"] = robust_bandwidth(v, quantile=meanshift_q, n_samples=meanshift_ns)
                est = MeanShift(**p)
                labels = est.fit_predict(v.reshape(-1,1))
                centers = ensure_centers(est, v.reshape(-1,1))
            else:
                est = make(method, **params)
                labels = est.fit_predict(v.reshape(-1,1))
                centers = ensure_centers(est, v.reshape(-1,1))

            groups[layer][idx] = {
                "method": method,
                "params": params,
                "centers": centers.astype(np.float64),
                "labels": labels.astype(np.int32),
            }

    if cache_tag:
        save_cluster_groups(cache_path, cache_tag, groups)
    return groups

def assign_clusters(
    groups: Dict[str, Dict[int, dict]],
    sample_acts: Dict[str, Dict[int, float]]
) -> Dict[str, Dict[int, int]]:
    """
    sample_acts[layer][idx] = scalar
    Returns assigned[layer][idx] = cluster_id
    """
    out: Dict[str, Dict[int, int]] = {}
    for layer, idx_map in sample_acts.items():
        out[layer] = {}
        for idx, val in idx_map.items():
            info = groups[layer][idx]
            centers = info["centers"].reshape(-1,1)   # (C,1)
            d = np.abs(centers.squeeze(1) - float(val))
            out[layer][idx] = int(np.argmin(d))
    return out