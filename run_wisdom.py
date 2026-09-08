#!/usr/bin/env python
"""Explicit WISDOM/IDC inference-runner foundation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

def run_bo(*args: Any, **kwargs: Any) -> Any:
    """Lazily import the optional-search dependency when BO is requested."""
    from wisdom.utils.search import run_bo as packaged_run_bo

    return packaged_run_bo(*args, **kwargs)


if TYPE_CHECKING:
    import pandas as pd
    import torch
    from wisdom.core.layers import LayerPlan
    from wisdom.core.task import PreparedTask
    from wisdom.core.wisdom import WisdomIDC


@dataclass(frozen=True)
class ScoreArtifact:
    path: Path
    status: str


def _default_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a nonnegative integer, got {value!r}")
    return parsed


def comma_separated_floats(value: str) -> tuple[float, ...]:
    """Parse custom normalization values without importing task packages."""
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        )
    try:
        parsed = tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        ) from exc
    if any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        )
    return parsed


def _parse_json_mapping(value: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("pose-model-kwargs must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("pose-model-kwargs must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explicit WISDOM/IDC inference runner")
    parser.add_argument("--mode", choices=["wisdom", "idc"], required=True)
    parser.add_argument("--task", choices=["classification", "detection", "pose"], required=True)
    parser.add_argument("--weights-path", required=True)
    parser.add_argument(
        "--model-path", default=None,
        help="Local YOLO architecture YAML for state-dict loading; not needed for a trusted module .pt.",
    )
    parser.add_argument(
        "--checkpoint-format", choices=["auto", "module", "state-dict"], default="state-dict",
        help="Use module only for trusted pickles (including Ultralytics .pt model/ema containers).",
    )
    parser.add_argument("--model-factory", default=None)
    parser.add_argument("--build-data-path", required=True)
    parser.add_argument(
        "--validation-data-path",
        default=None,
        help=(
            "Explicit BO validation data. With --bo, omit this option to reserve "
            "a deterministic 10%% holdout from build data."
        ),
    )
    parser.add_argument("--test-data-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--wisdom-csv", default=None)
    parser.add_argument("--idc-csv", default=None)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument(
        "--normalize",
        choices=["none", "imagenet", "cifar", "mnist", "custom"],
        default="none",
    )
    parser.add_argument(
        "--normalize-mean",
        type=comma_separated_floats,
        default=None,
        metavar="M1,M2,...",
        help="Per-channel mean required by --normalize custom",
    )
    parser.add_argument(
        "--normalize-std",
        type=comma_separated_floats,
        default=None,
        metavar="S1,S2,...",
        help="Positive per-channel standard deviation required by --normalize custom",
    )
    parser.add_argument("--pose-topology", default=None)
    parser.add_argument("--pose-architecture", default="resnet18_baseline_att")
    parser.add_argument("--pose-model-kwargs", type=_parse_json_mapping, default={})
    parser.add_argument("--pose-output-layers", nargs=2, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--voting-mode", choices=["fine-grained", "coarse"], default="fine-grained")
    parser.add_argument("--trainer-checkpoint", default=None)
    parser.add_argument("--checkpoint-every", type=positive_int, default=50)
    parser.add_argument("--selection-mode", choices=["global", "per-group", "per-layer"], default="global")
    parser.add_argument("--top-m-neurons", type=positive_int, default=10)
    parser.add_argument("--num-groups", type=positive_int, default=3)
    parser.add_argument("--num-layers", type=positive_int, default=None)
    parser.add_argument("--cluster-method", default="KMeans")
    parser.add_argument("--n-clusters", type=positive_int, default=2)
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bo", action="store_true")
    parser.add_argument("--bo-backend", choices=["auto", "sklearn", "botorch"], default="auto")
    parser.add_argument("--bo-init", type=positive_int, default=3)
    parser.add_argument("--bo-iter", type=nonnegative_int, default=3)
    parser.add_argument("--bo-candidate-pool-size", type=positive_int, default=32)
    parser.add_argument("--bo-cluster-methods", default="KMeans,MiniBatchKMeans,Birch")
    parser.add_argument("--bo-n-clusters", default="2,3,4")
    parser.add_argument("--bo-output-json", default=None)
    parser.add_argument("--plot-neurons", action="store_true")
    parser.add_argument("--plot-top-k", type=positive_int, default=None)
    return parser


def _resolved_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _nonempty_csv_values(value: str, *, option: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{option} must contain at least one value")
    return values


_CLUSTER_CAPABILITIES = {
    "kmeans": (True, True),
    "minibatchkmeans": (True, True),
    "birch": (True, False),
    "agglomerativeclustering": (True, False),
    "spectralclustering": (True, True),
    "meanshift": (False, False),
    "dbscan": (False, False),
    "optics": (False, False),
}


def _normalized_cluster_method(method: str) -> str:
    normalized = method.strip()
    if not normalized:
        raise ValueError("--cluster-method must be nonblank")
    return normalized


def _cluster_capability(method: str) -> tuple[bool, bool] | None:
    """Return (uses_cluster_count, uses_seed), leaving unknown names to the factory."""
    return _CLUSTER_CAPABILITIES.get(_normalized_cluster_method(method).lower())


def _effective_cluster_count(method: str, requested_count: int | None) -> int | None:
    capability = _cluster_capability(method)
    return requested_count if capability is None or capability[0] else None


def _score_artifact_path(args: argparse.Namespace) -> Path | None:
    if args.mode == "wisdom":
        configured = getattr(args, "wisdom_csv", None)
        return Path(configured) if configured else Path(args.output_json).with_name(
            f"{Path(args.output_json).stem}_wisdom_scores.csv"
        )
    configured = getattr(args, "idc_csv", None)
    return Path(configured) if configured else None


def _score_path_lifecycle(args: argparse.Namespace) -> tuple[Path | None, bool]:
    """Return the score path and whether this run will write it, without loading it."""
    path = _score_artifact_path(args)
    if path is None or args.mode != "wisdom":
        return path, False
    return path, not (path.exists() and path.stat().st_size > 0)


def _bo_output_path(args: argparse.Namespace) -> Path:
    configured = getattr(args, "bo_output_json", None)
    return Path(configured) if configured else Path(args.output_json).with_name(
        f"{Path(args.output_json).stem}_bo.json"
    )


def _plot_output_path(args: argparse.Namespace) -> Path:
    output_path = Path(args.output_json)
    top_k = args.plot_top_k or args.top_m_neurons
    return output_path.with_name(f"{output_path.stem}_top_{top_k}_neurons.pdf")


def _validate_writable_paths(args: argparse.Namespace) -> None:
    """Reject resolved writable-artifact aliases without touching the filesystem."""
    artifacts: list[tuple[str, Path]] = [("--output-json", Path(args.output_json))]
    score_path, score_is_writable = _score_path_lifecycle(args)
    if score_path is not None and score_is_writable:
        artifacts.append(("--wisdom-csv" if args.mode == "wisdom" else "--idc-csv", score_path))
    if getattr(args, "bo", False):
        artifacts.append(("--bo-output-json", _bo_output_path(args)))
    if getattr(args, "plot_neurons", False):
        artifacts.append(("--plot-neurons destination", _plot_output_path(args)))
    for option in ("cache_path",):
        value = getattr(args, option, None)
        if value:
            artifacts.append((f"--{option.replace('_', '-')}", Path(value)))
    if score_is_writable and getattr(args, "trainer_checkpoint", None):
        artifacts.append(("--trainer-checkpoint", Path(args.trainer_checkpoint)))

    resolved_artifacts = [(name, _resolved_path(str(path))) for name, path in artifacts]
    for index, (name, path) in enumerate(resolved_artifacts):
        for other_name, other_path in resolved_artifacts[index + 1:]:
            if path == other_path:
                raise ValueError(f"Writable artifacts {name} and {other_name} resolve to the same path: {path}")

    inputs = [
        ("--weights-path", getattr(args, "weights_path", None)),
        ("--model-path", getattr(args, "model_path", None)),
        ("--build-data-path", getattr(args, "build_data_path", None)),
        ("--validation-data-path", getattr(args, "validation_data_path", None)),
        ("--test-data-path", getattr(args, "test_data_path", None)),
        ("--pose-topology", getattr(args, "pose_topology", None)),
    ]
    if score_path is not None and not score_is_writable:
        inputs.append(("--wisdom-csv" if args.mode == "wisdom" else "--idc-csv", str(score_path)))
    for artifact_name, artifact_path in resolved_artifacts:
        for input_name, input_value in inputs:
            if input_value and artifact_path == _resolved_path(str(input_value)):
                raise ValueError(f"Writable artifact {artifact_name} collides with input {input_name}: {artifact_path}")


def _validated_bo_cluster_methods(value: str) -> list[str]:
    """Reject BO methods that cannot vary its n_clusters search dimension."""
    methods = list(
        dict.fromkeys(
            _normalized_cluster_method(method)
            for method in _nonempty_csv_values(value, option="--bo-cluster-methods")
        )
    )
    incompatible = [
        method
        for method in methods
        if (capability := _cluster_capability(method)) is not None and not capability[0]
    ]
    if incompatible:
        raise ValueError(
            "--bo-cluster-methods must contain only count-compatible methods; "
            f"got {', '.join(incompatible)}"
        )
    for method in methods:
        if _cluster_capability(method) is None:
            from wisdom.clustering.factory import make as make_clusterer

            make_clusterer(method)
    return methods


def _validate_classification_normalization(args: argparse.Namespace) -> None:
    mode = getattr(args, "normalize", "none")
    mean = getattr(args, "normalize_mean", None)
    std = getattr(args, "normalize_std", None)
    if mode != "custom":
        if mean is not None or std is not None:
            raise ValueError(
                "--normalize-mean and --normalize-std are only valid with "
                "--normalize custom."
            )
        return
    if mean is None or std is None:
        raise ValueError(
            "--normalize custom requires both --normalize-mean and --normalize-std."
        )
    expected_channels = 1 if getattr(args, "grayscale", False) else 3
    if len(mean) != expected_channels or len(std) != expected_channels:
        raise ValueError(
            f"--normalize custom requires {expected_channels} values for both "
            "mean and std."
        )
    if any(value <= 0 for value in std):
        raise ValueError("Custom normalization std values must be positive.")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument relationships only; do not load a model or dataset."""
    mode = getattr(args, "mode", None)
    if mode == "wisdom" and getattr(args, "idc_csv", None):
        raise ValueError("--idc-csv is only valid in IDC mode")
    if mode == "idc" and getattr(args, "wisdom_csv", None):
        raise ValueError("--wisdom-csv is only valid in WISDOM mode")
    if mode == "idc" and getattr(args, "bo", False):
        raise ValueError("IDC mode does not support BO")
    if getattr(args, "n_clusters", 0) <= 0:
        raise ValueError("--n-clusters must be positive")
    _normalized_cluster_method(args.cluster_method)

    _validate_writable_paths(args)

    task = getattr(args, "task", None)
    if task == "classification":
        _validate_classification_normalization(args)
    if task == "classification" and getattr(args, "checkpoint_format", "state-dict") in {"auto", "state-dict"} and not getattr(args, "model_factory", None):
        raise ValueError("classification state-dict checkpoints require --model-factory")
    if task == "detection":
        model_path = getattr(args, "model_path", None)
        if not model_path and getattr(args, "checkpoint_format", "state-dict") != "module":
            raise ValueError("detection state-dict checkpoints require a local --model-path")
        if "://" in str(model_path):
            raise ValueError("detection --model-path must be local")
    if task == "pose":
        if not getattr(args, "pose_topology", None):
            raise ValueError("pose requires --pose-topology")
        if not getattr(args, "pose_architecture", None):
            raise ValueError("pose requires --pose-architecture")

    build_path, test_path = getattr(args, "build_data_path", None), getattr(args, "test_data_path", None)
    if build_path and test_path and _resolved_path(build_path) == _resolved_path(test_path):
        raise ValueError("--build-data-path and --test-data-path must resolve to different paths")
    if getattr(args, "bo", False):
        validation_path = getattr(args, "validation_data_path", None)
        if validation_path:
            validation = _resolved_path(validation_path)
            if build_path and validation == _resolved_path(build_path):
                raise ValueError("--validation-data-path must differ from --build-data-path")
            if test_path and validation == _resolved_path(test_path):
                raise ValueError("--validation-data-path must differ from --test-data-path")
        _validated_bo_cluster_methods(getattr(args, "bo_cluster_methods", ""))
        cluster_counts = _nonempty_csv_values(getattr(args, "bo_n_clusters", ""), option="--bo-n-clusters")
        try:
            if any(int(value) <= 0 for value in cluster_counts):
                raise ValueError
        except ValueError as exc:
            raise ValueError("--bo-n-clusters must contain positive integers") from exc


def _prepare_task(args: argparse.Namespace) -> PreparedTask:
    if args.task == "classification":
        from wisdom_classification_train import prepare_classification_inference
        return prepare_classification_inference(args)
    if args.task == "detection":
        from wisdom_yolo_train import prepare_detection_inference
        return prepare_detection_inference(args)
    from wisdom_pose_train import prepare_pose_inference
    return prepare_pose_inference(args)


def _validate_score_csv(path: Path) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid score CSV '{path}': {exc}") from exc
    required = {"LayerName", "NeuronIndex", "Score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Score CSV '{path}' is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"Score CSV '{path}' must contain at least one data row")
    names = frame["LayerName"]
    if names.isna().any() or any(not str(name).strip() for name in names):
        raise ValueError(f"Score CSV '{path}' contains an empty LayerName")
    indices = pd.to_numeric(frame["NeuronIndex"], errors="coerce")
    if indices.isna().any() or not np.isfinite(indices).all() or not np.equal(indices, np.floor(indices)).all() or (indices < 0).any():
        raise ValueError(f"Score CSV '{path}' contains invalid nonnegative NeuronIndex values")
    scores = pd.to_numeric(frame["Score"], errors="coerce")
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError(f"Score CSV '{path}' contains non-finite Score values")
    validated = frame.copy()
    validated["LayerName"] = validated["LayerName"].astype(str)
    validated["NeuronIndex"], validated["Score"] = indices.astype(int), scores.astype(float)
    return validated


def _resolve_score_artifact(args: argparse.Namespace, prepared: PreparedTask) -> ScoreArtifact:
    if args.mode == "wisdom":
        if getattr(args, "idc_csv", None):
            raise ValueError("--idc-csv cannot be used in WISDOM mode")
        path = _score_artifact_path(args)
        assert path is not None
        if path.exists() and path.stat().st_size > 0:
            _validate_score_csv(path)
            return ScoreArtifact(path, "reused")
        path.parent.mkdir(parents=True, exist_ok=True)
        prepared.score_trainer(str(path))
        _validate_score_csv(path)
        return ScoreArtifact(path, "generated")

    if getattr(args, "wisdom_csv", None):
        raise ValueError("--wisdom-csv cannot be used in IDC mode")
    path = _score_artifact_path(args)
    if path is None:
        raise ValueError("IDC mode requires an existing nonempty IDC score CSV via --idc-csv")
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("IDC mode requires an existing nonempty IDC score CSV")
    _validate_score_csv(path)
    return ScoreArtifact(path, "reused")


def _score_tensors(
    frame: pd.DataFrame, layers: tuple[str, ...]
) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    import torch

    result: dict[str, torch.Tensor] = {}
    source_indices: dict[str, list[int]] = {}
    for layer in layers:
        rows = frame[(frame["LayerName"] == layer) & (frame["Score"] > 0)].sort_values(
            "NeuronIndex"
        )
        if rows.empty:
            continue
        result[layer] = torch.tensor(rows["Score"].tolist(), dtype=torch.float)
        source_indices[layer] = [int(index) for index in rows["NeuronIndex"].tolist()]
    return result, source_indices


def _resolve_selected_neurons(args: argparse.Namespace, prepared: PreparedTask, score_path: str | Path) -> tuple[dict[str, list[int]], LayerPlan]:
    from wisdom.core.layers import build_layer_plan, discover_eligible_layers
    from wisdom.core.wisdom import (
        WisdomConfig,
        WisdomIDC,
        load_groupwise_top_neurons,
        load_layerwise_top_neurons,
    )

    frame = _validate_score_csv(Path(score_path))
    model = prepared.adapter.analysis_model
    unknown = sorted(set(frame["LayerName"]) - set(dict(model.named_modules())))
    if unknown:
        raise ValueError(f"Score CSV layer names were not found in the analysis model: {unknown}")
    eligible = discover_eligible_layers(model, excluded_layers=prepared.adapter.excluded_layer_names(), excluded_prefixes=prepared.adapter.excluded_layer_prefixes())
    plan = build_layer_plan(eligible, n_groups=args.num_groups if args.selection_mode == "per-group" else None, num_layers=args.num_layers)
    excluded = set(frame["LayerName"]) - set(plan.eligible_layers)
    if args.selection_mode == "global":
        selector = WisdomIDC(model, cfg=WisdomConfig(top_m_neurons=args.top_m_neurons))
        score_tensors, source_indices = _score_tensors(frame, plan.eligible_layers)
        dense_selection = selector.select_top_neurons(score_tensors)
        selected = {
            layer: [source_indices[layer][position] for position in positions]
            for layer, positions in dense_selection.items()
        }
    elif args.selection_mode == "per-layer":
        selected = load_layerwise_top_neurons(str(score_path), args.top_m_neurons, strip_prefix="", exclude_layers=excluded)
    else:
        selected = load_groupwise_top_neurons(str(score_path), args.top_m_neurons, strip_prefix="", exclude_layers=excluded, layer_groups=plan.groups)
        missing = [group for group, layers in plan.groups.items() if not any(selected.get(layer) for layer in layers)]
        if missing:
            raise ValueError(f"per-group selection found no neurons for group(s): {missing}")
    selected = {layer: sorted(int(index) for index in indices) for layer, indices in selected.items() if indices}
    if not selected:
        raise ValueError("selection found no neurons with a positive score")
    return selected, plan


def _cluster_parameters(method: str, n_clusters: int | None, seed: int) -> dict[str, Any]:
    capability = _cluster_capability(method)
    if capability is None:
        return {}
    uses_count, uses_seed = capability
    if not uses_count:
        return {}
    if n_clusters is None:
        raise ValueError(f"{method} requires an effective cluster count")
    if uses_seed:
        return {"n_clusters": n_clusters, "random_state": seed}
    return {"n_clusters": n_clusters}


def _build_engine(args: argparse.Namespace, prepared: PreparedTask, layer_plan: LayerPlan, cluster_method: str | None = None, n_clusters: int | None = None) -> WisdomIDC:
    from wisdom.clustering.factory import make as make_clusterer
    from wisdom.core.wisdom import ClusteringConfig, WisdomConfig, WisdomIDC

    requested_method = _normalized_cluster_method(
        cluster_method if cluster_method is not None else args.cluster_method
    )
    if args.mode == "idc":
        method, params, impl, silhouette = "KMeans", {"n_clusters": 2, "random_state": args.seed}, "idc", True
    else:
        requested_count = n_clusters if n_clusters is not None else _effective_cluster_count(
            requested_method, args.n_clusters
        )
        method, params, impl, silhouette = requested_method, _cluster_parameters(requested_method, requested_count, args.seed), "wisdom", False
    make_clusterer(method, **params)
    cluster = ClusteringConfig(method=method, params=params, use_silhouette=silhouette)
    config = WisdomConfig(top_m_neurons=args.top_m_neurons, cache_path=args.cache_path, selection_mode=args.selection_mode, n_groups=args.num_groups, layer_groups=layer_plan.groups)
    return WisdomIDC(prepared.adapter.analysis_model, impl=impl, cfg=config, cluster=cluster)


def _suite_sizes(total: int, points: int = 5) -> list[int]:
    """Return deterministic, increasing validation-prefix sizes."""
    import numpy as np

    if total < 1:
        raise ValueError("Validation data must contain at least one sample")
    if total == 1:
        return [1]
    count = min(max(2, points), total)
    return sorted({max(1, round(value)) for value in np.linspace(1, total, count)} | {total})


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    import numpy as np

    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x, y = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _validation_prefix_loader(loader: Any, size: int) -> Any:
    """Make an ordered validation prefix without changing loader batching semantics."""
    from torch.utils.data import DataLoader, Subset

    return DataLoader(
        Subset(loader.dataset, range(size)),
        batch_size=loader.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor if loader.num_workers else None,
        persistent_workers=loader.persistent_workers if loader.num_workers else False,
        pin_memory_device=loader.pin_memory_device,
    )


def _coverage_objective(
    args: argparse.Namespace,
    prepared: PreparedTask,
    selected: dict[str, list[int]],
    layer_plan: LayerPlan,
) -> Callable[[dict[str, object]], float]:
    """Build the validation-only correlation objective used for BO candidates."""
    validation_loader = prepared.validation_loader
    if validation_loader is None:
        raise ValueError("BO requires validation data")
    sizes = _suite_sizes(len(validation_loader.dataset))

    def objective(candidate: dict[str, object]) -> float:
        try:
            method = str(candidate["cluster_method"])
            n_clusters = int(candidate["n_clusters"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("BO candidate must define cluster_method and n_clusters") from exc
        engine = _build_engine(args, prepared, layer_plan, method, n_clusters)
        engine.fit_selected(prepared.build_loader, selected, args.device)
        coverage_values: list[float] = []
        task_values: list[float] = []
        metric_name: str | None = None
        for size in sizes:
            prefix = _validation_prefix_loader(validation_loader, size)
            coverage = engine.coverage_details(prefix, selected, args.device)
            evaluation = prepared.evaluate(prefix)
            if evaluation.bo_metric_name is None or evaluation.bo_metric_value is None:
                raise ValueError(f"BO requires an available validation metric for task '{prepared.task}'")
            if metric_name is None:
                metric_name = evaluation.bo_metric_name
            elif evaluation.bo_metric_name != metric_name:
                raise ValueError(f"BO validation metric changed for task '{prepared.task}'")
            coverage_values.append(float(coverage["coverage_rate"]))
            task_values.append(float(evaluation.bo_metric_value))
        return _pearson_correlation(coverage_values, task_values)

    return objective


def _cluster_counts(args: argparse.Namespace, build_samples: int) -> list[int]:
    try:
        values = [int(value) for value in _nonempty_csv_values(args.bo_n_clusters, option="--bo-n-clusters")]
    except ValueError as exc:
        raise ValueError("--bo-n-clusters must contain integers") from exc
    counts = list(dict.fromkeys(values))
    if any(count < 2 or count > build_samples for count in counts):
        raise ValueError(f"--bo-n-clusters must contain values from 2 to build sample count ({build_samples})")
    return counts


def _resolve_cluster_configuration(
    args: argparse.Namespace,
    prepared: PreparedTask,
    selected: dict[str, list[int]],
    layer_plan: LayerPlan,
) -> dict[str, object]:
    """Return configured clustering or a truthfully reported BO winner."""
    configured_method = _normalized_cluster_method(args.cluster_method)
    if not args.bo:
        if args.mode == "idc":
            return {
                "source": "configured",
                "cluster_method": "KMeans",
                "n_clusters": 2,
            }
        effective_count = _effective_cluster_count(configured_method, args.n_clusters)
        if effective_count is not None and effective_count > len(prepared.build_loader.dataset):
            raise ValueError("n_clusters cannot exceed the build sample count")
        return {
            "source": "configured",
            "cluster_method": configured_method,
            "n_clusters": effective_count,
        }
    if prepared.validation_loader is None:
        raise ValueError("BO requires validation data")
    build_samples = len(prepared.build_loader.dataset)
    methods = _validated_bo_cluster_methods(args.bo_cluster_methods)
    counts = _cluster_counts(args, build_samples)
    validation_evaluation = prepared.evaluate(prepared.validation_loader)
    validation_metric = validation_evaluation.bo_metric_name
    if validation_metric is None or validation_evaluation.bo_metric_value is None:
        raise ValueError(f"BO requires an available validation metric for task '{prepared.task}'")
    output_path = _bo_output_path(args)
    result, history_path = run_bo(
        {"cluster_method": methods, "n_clusters": counts},
        _coverage_objective(args, prepared, selected, layer_plan),
        random_state=args.seed,
        backend=args.bo_backend,
        n_init=args.bo_init,
        n_iter=args.bo_iter,
        candidate_pool_size=args.bo_candidate_pool_size,
        out_path=str(output_path),
        payload_extras={"validation_metric": validation_metric},
    )
    return {
        "source": "bo",
        "backend": result.backend,
        "best_cluster_method": str(result.best_config["cluster_method"]),
        "best_n_clusters": int(result.best_config["n_clusters"]),
        "best_score": float(result.best_score),
        "validation_metric": validation_metric,
        "history_json": history_path,
    }


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_neuron_plot(score_path: Path, top_k: int, output_path: Path) -> str:
    from wisdom.utils.visualization import viz_topk_neurons_score

    destination = output_path.with_name(f"{output_path.stem}_top_{top_k}_neurons.pdf")
    return str(viz_topk_neurons_score(str(score_path), top_k=top_k, output_path=destination))


def _print_summary(summary: dict[str, object]) -> None:
    model = summary["model"]
    score = summary["score_csv"]
    selection = summary["selection"]
    coverage = summary["coverage"]
    metrics = summary["metrics"]
    print(f"Task: {summary['task']}")
    print(f"Mode: {summary['mode']}")
    print(f"Model: {model['name']}")
    print(f"Weights: {model['weights_path']}")
    print(f"Score CSV: {score['path']} ({score['status']})")
    print(f"Selected layers/neurons: {selection['selected_layers']}/{selection['selected_neurons']}")
    print(f"Coverage: {_format_value(coverage['coverage_rate'])}")
    for name, value in metrics.items():
        print(f"{name}: {_format_value(value)}")
    print(f"Output JSON: {summary['output_json']}")
    print(json.dumps(summary["cluster_configuration"], indent=2))


def run(args: argparse.Namespace) -> dict[str, object]:
    """Fit the final engine on build data and report test-only metrics and coverage."""
    prepared = _prepare_task(args)
    score = _resolve_score_artifact(args, prepared)
    selected, layer_plan = _resolve_selected_neurons(args, prepared, score.path)
    cluster_configuration = _resolve_cluster_configuration(args, prepared, selected, layer_plan)
    if cluster_configuration["source"] == "bo":
        cluster_method = str(cluster_configuration["best_cluster_method"])
        n_clusters = int(cluster_configuration["best_n_clusters"])
    else:
        cluster_method = str(cluster_configuration["cluster_method"])
        n_clusters = cluster_configuration["n_clusters"]
    final_engine = _build_engine(args, prepared, layer_plan, cluster_method, n_clusters)
    final_engine.fit_selected(prepared.build_loader, selected, args.device)
    evaluation = prepared.evaluate(prepared.test_loader)
    coverage_details = final_engine.coverage_details(prepared.test_loader, selected, args.device)
    output_path = _resolved_path(args.output_json)
    visualization_path = None
    if args.plot_neurons:
        visualization_path = _write_neuron_plot(score.path, args.plot_top_k or args.top_m_neurons, output_path)
    resolved_validation = _resolved_path(args.validation_data_path) if args.validation_data_path else None
    build_samples = len(prepared.build_loader.dataset)
    validation_samples = (
        len(prepared.validation_loader.dataset) if prepared.validation_loader else None
    )
    validation_source = (
        "path"
        if resolved_validation
        else "build_holdout"
        if validation_samples is not None
        else None
    )
    validation_fraction = (
        validation_samples / (build_samples + validation_samples)
        if validation_source == "build_holdout"
        else None
    )
    summary: dict[str, object] = {
        "task": args.task,
        "mode": args.mode,
        "model": {
            "name": prepared.model_name,
            "model_path": str(_resolved_path(args.model_path)) if args.model_path else None,
            "weights_path": str(_resolved_path(args.weights_path)),
            "checkpoint_format": prepared.checkpoint_format,
        },
        "score_csv": {"path": str(score.path), "status": score.status},
        "data": {
            "build_path": str(_resolved_path(args.build_data_path)),
            "validation_path": str(resolved_validation) if resolved_validation else None,
            "validation_source": validation_source,
            "validation_fraction": validation_fraction,
            "test_path": str(_resolved_path(args.test_data_path)),
            "build_samples": build_samples,
            "validation_samples": validation_samples,
            "test_samples": len(prepared.test_loader.dataset),
        },
        "selection": {
            "mode": args.selection_mode,
            "selected_layers": len(selected),
            "selected_neurons": sum(len(indices) for indices in selected.values()),
            "num_groups": len(layer_plan.groups),
            "num_layers": len(layer_plan.eligible_layers),
        },
        "cluster_configuration": cluster_configuration,
        "coverage": coverage_details,
        "metrics": evaluation.metrics,
        "visualization_path": visualization_path,
        "output_json": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    _print_summary(summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
