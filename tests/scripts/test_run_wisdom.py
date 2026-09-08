from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import run_wisdom
from tests.helpers import TinyClassifier
from wisdom.core.layers import build_layer_plan
from wisdom.core.task import PreparedTask, TaskEvaluation
from wisdom.tasks.classification import ClassificationAdapter
from wisdom.utils.search import SearchResult


def _namespace(**overrides: object) -> Namespace:
    values = {
        "mode": "wisdom", "task": "classification", "wisdom_csv": None,
        "idc_csv": None, "output_json": "result.json", "device": "cpu",
        "selection_mode": "global", "top_m_neurons": 1, "num_groups": 3,
        "num_layers": None, "cluster_method": "KMeans", "n_clusters": 2,
        "cache_path": None, "seed": 42,
    }
    values.update(overrides)
    return Namespace(**values)


def _prepared(score_trainer) -> PreparedTask:
    model = TinyClassifier().eval()
    loader = DataLoader(
        TensorDataset(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1])), batch_size=2
    )
    return PreparedTask(
        task="classification", adapter=ClassificationAdapter(model),
        build_loader=loader, validation_loader=None, test_loader=loader,
        score_trainer=score_trainer,
        evaluate=lambda current: TaskEvaluation(
            {"accuracy": None, "loss": None, "f1": 0.0}, "f1", 0.0
        ),
        model_name="TinyClassifier", checkpoint_format="state-dict",
    )


def _parse(*extra: str):
    return run_wisdom.build_parser().parse_args(
        [
            "--mode", "wisdom", "--task", "classification",
            "--weights-path", "weights.pth", "--model-factory", "tests.helpers:TinyClassifier",
            "--build-data-path", "build", "--test-data-path", "test",
            "--output-json", "result.json", *extra,
        ]
    )


def _write_scores(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _scores_for_tiny(path: Path) -> Path:
    return _write_scores(
        path,
        [
            {"LayerName": "features.0", "NeuronIndex": 0, "Score": 1.0},
            {"LayerName": "features.0", "NeuronIndex": 1, "Score": 9.0},
            {"LayerName": "features.2", "NeuronIndex": 0, "Score": 8.0},
            {"LayerName": "features.2", "NeuronIndex": 1, "Score": 2.0},
            {"LayerName": "features.4", "NeuronIndex": 0, "Score": 7.0},
            {"LayerName": "features.4", "NeuronIndex": 1, "Score": 3.0},
        ],
    )


def test_parser_exposes_all_cluster_bo_controls_with_defaults() -> None:
    args = _parse()
    assert args.cluster_method == "KMeans" and args.n_clusters == 2
    assert args.bo_backend == "auto"
    assert (args.bo_init, args.bo_iter, args.bo_candidate_pool_size) == (3, 3, 32)
    assert args.bo_cluster_methods == "KMeans,MiniBatchKMeans,Birch"
    assert args.bo_n_clusters == "2,3,4"


def test_parser_accepts_custom_normalization_statistics() -> None:
    """Dropping either custom-stat option must break the public runner contract."""
    args = _parse(
        "--normalize",
        "custom",
        "--normalize-mean",
        "0.1,0.2,0.3",
        "--normalize-std",
        "0.4,0.5,0.6",
    )

    assert args.normalize == "custom"
    assert args.normalize_mean == (0.1, 0.2, 0.3)
    assert args.normalize_std == (0.4, 0.5, 0.6)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ("--normalize", "custom", "--normalize-mean", "0.1,0.2,0.3"),
            "both.*normalize-mean.*normalize-std",
        ),
        (
            (
                "--normalize",
                "custom",
                "--normalize-mean",
                "0.1",
                "--normalize-std",
                "0.2",
            ),
            "3 values",
        ),
        (
            (
                "--normalize",
                "custom",
                "--normalize-mean",
                "0.1,0.2,0.3",
                "--normalize-std",
                "0.2,0,0.2",
            ),
            "positive",
        ),
        (
            (
                "--normalize",
                "imagenet",
                "--normalize-mean",
                "0.1,0.2,0.3",
                "--normalize-std",
                "0.2,0.2,0.2",
            ),
            "only valid",
        ),
    ],
)
def test_runner_validates_custom_normalization_configuration(
    extra: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_wisdom._validate_args(_parse(*extra))


def test_runner_import_does_not_load_optional_plot_or_task_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import run_wisdom, sys; "
            "blocked = {name for name in sys.modules if name.startswith(('matplotlib', 'ultralytics', 'botorch'))}; "
            "assert not blocked, blocked",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--mode", "idc", "--bo", "--validation-data-path", "validation"), "IDC"),
        (("--idc-csv", "idc.csv"), "idc-csv"),
        (("--wisdom-csv", "w.csv", "--mode", "idc"), "wisdom-csv"),
    ],
)
def test_invalid_mode_and_bo_combinations_fail(extra: tuple[str, ...], message: str) -> None:
    args = _parse(*extra)
    with pytest.raises(ValueError, match=message):
        run_wisdom._validate_args(args)


def test_bo_without_validation_path_is_allowed_for_default_build_holdout() -> None:
    """Requiring an explicit validation directory would break the 10% fallback."""
    run_wisdom._validate_args(_parse("--bo"))


def test_pth_suffix_never_changes_explicit_task() -> None:
    assert _parse("--weights-path", "pose.pth").task == "classification"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (_namespace(task="pose", pose_topology=None, pose_architecture="resnet18_baseline_att"), "pose-topology"),
        (_namespace(task="pose", pose_topology="topology.json", pose_architecture=""), "pose-architecture"),
        (_namespace(task="detection", model_path=None), "model-path"),
        (_namespace(task="classification", checkpoint_format="state-dict", model_factory=None), "model-factory"),
        (_namespace(build_data_path="same", test_data_path="same", model_factory="factory"), "build-data-path"),
        (_namespace(build_data_path="build", test_data_path="test", validation_data_path="test", bo=True, model_factory="factory"), "validation-data-path"),
    ],
)
def test_task_architecture_and_dataset_roles_are_validated(args: Namespace, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_wisdom._validate_args(args)


def test_pose_bo_is_allowed_with_explicit_validation_data() -> None:
    args = run_wisdom.build_parser().parse_args(
        [
            "--mode", "wisdom", "--task", "pose", "--weights-path", "pose.pth",
            "--pose-topology", "human_pose.json", "--build-data-path", "build",
            "--validation-data-path", "validation", "--test-data-path", "test",
            "--output-json", "result.json", "--bo",
        ]
    )
    run_wisdom._validate_args(args)


def test_validation_rejects_nonpositive_cluster_count_before_preparation() -> None:
    with pytest.raises(ValueError, match="n-clusters"):
        run_wisdom._validate_args(_namespace(model_factory="factory", n_clusters=0))


@pytest.mark.parametrize("method", ["", "   "])
def test_validation_rejects_blank_configured_cluster_methods(method: str) -> None:
    """Letting an empty method silently become factory-default KMeans must fail this test."""
    with pytest.raises(ValueError, match="cluster-method.*nonblank"):
        run_wisdom._validate_args(_namespace(model_factory="factory", cluster_method=method))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"output_json": "artifacts/../weights.pth"},
            "output-json.*weights-path",
        ),
        (
            {"wisdom_csv": "artifacts/result.json"},
            "output-json.*wisdom-csv",
        ),
        (
            {"bo": True, "bo_output_json": "artifacts/result.json"},
            "output-json.*bo-output-json",
        ),
        (
            {"wisdom_csv": "artifacts/../weights.pth"},
            "wisdom-csv.*weights-path",
        ),
        (
            {"bo": True, "bo_output_json": "artifacts/../build"},
            "bo-output-json.*build-data-path",
        ),
    ],
)
def test_validation_rejects_resolved_writable_artifact_aliases(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    """Removing resolved output/input collision checks must fail this test."""
    values: dict[str, object] = {
        "model_factory": "factory",
        "weights_path": str(tmp_path / "weights.pth"),
        "build_data_path": str(tmp_path / "build"),
        "validation_data_path": str(tmp_path / "validation"),
        "test_data_path": str(tmp_path / "test"),
        "output_json": str(tmp_path / "artifacts" / "result.json"),
        "bo": False,
        "bo_output_json": None,
        "bo_cluster_methods": "KMeans",
        "bo_n_clusters": "2",
    }
    for key, value in overrides.items():
        values[key] = str(tmp_path / str(value)) if isinstance(value, str) else value

    with pytest.raises(ValueError, match=message):
        run_wisdom._validate_args(_args(**values))


def test_validation_rejects_derived_bo_and_plot_artifacts_that_alias_inputs(tmp_path: Path) -> None:
    """Removing default-output derivation before path validation must fail this test."""
    output = tmp_path / "artifacts" / "result.json"
    args = _args(
        model_factory="factory",
        bo=True,
        bo_cluster_methods="KMeans",
        bo_n_clusters="2",
        output_json=str(output),
        weights_path=str(output.with_name("result_bo.json")),
        build_data_path=str(tmp_path / "build"),
        validation_data_path=str(tmp_path / "validation"),
        test_data_path=str(tmp_path / "test"),
    )
    with pytest.raises(ValueError, match="bo-output-json.*weights-path"):
        run_wisdom._validate_args(args)

    args.bo = False
    args.weights_path = str(output.with_name("result_top_1_neurons.pdf"))
    args.plot_neurons = True
    with pytest.raises(ValueError, match="plot.*weights-path"):
        run_wisdom._validate_args(args)


def test_validation_allows_inactive_wisdom_checkpoint_to_alias_a_reused_score(tmp_path: Path) -> None:
    """Treating reused WISDOM artifacts as writable must fail this test."""
    score_path = _scores_for_tiny(tmp_path / "reused.csv")
    before = score_path.read_bytes()

    run_wisdom._validate_args(
        _args(
            model_factory="factory",
            wisdom_csv=str(score_path),
            trainer_checkpoint=str(score_path),
            output_json=str(tmp_path / "result.json"),
            build_data_path=str(tmp_path / "build"),
            test_data_path=str(tmp_path / "test"),
        )
    )

    assert score_path.read_bytes() == before


def test_validation_allows_inactive_idc_checkpoint_to_alias_its_read_only_score(tmp_path: Path) -> None:
    """Treating IDC's never-run trainer checkpoint as writable must fail this test."""
    score_path = _scores_for_tiny(tmp_path / "idc.csv")

    run_wisdom._validate_args(
        _args(
            mode="idc",
            model_factory="factory",
            idc_csv=str(score_path),
            trainer_checkpoint=str(score_path),
            output_json=str(tmp_path / "result.json"),
            build_data_path=str(tmp_path / "build"),
            test_data_path=str(tmp_path / "test"),
        )
    )


@pytest.mark.parametrize("destination", ["output_json", "bo_output_json", "cache_path"])
def test_validation_still_protects_reused_wisdom_scores_from_active_destinations(
    tmp_path: Path, destination: str
) -> None:
    """Dropping reused scores from read-input validation must fail this test."""
    score_path = _scores_for_tiny(tmp_path / "reused.csv")
    values: dict[str, object] = {
        "model_factory": "factory",
        "wisdom_csv": str(score_path),
        "output_json": str(tmp_path / "result.json"),
        "build_data_path": str(tmp_path / "build"),
        "validation_data_path": str(tmp_path / "validation"),
        "test_data_path": str(tmp_path / "test"),
        "bo": destination == "bo_output_json",
        "bo_cluster_methods": "KMeans",
        "bo_n_clusters": "2",
    }
    values[destination] = str(score_path)

    with pytest.raises(ValueError, match="wisdom-csv"):
        run_wisdom._validate_args(_args(**values))


def test_validation_still_protects_reused_wisdom_scores_from_plot_output(tmp_path: Path) -> None:
    """Letting a plot overwrite a reused score CSV must fail this test."""
    output = tmp_path / "result.json"
    score_path = _scores_for_tiny(output.with_name("result_top_1_neurons.pdf"))

    with pytest.raises(ValueError, match="wisdom-csv"):
        run_wisdom._validate_args(
            _args(
                model_factory="factory",
                wisdom_csv=str(score_path),
                output_json=str(output),
                plot_neurons=True,
                build_data_path=str(tmp_path / "build"),
                test_data_path=str(tmp_path / "test"),
            )
        )


@pytest.mark.parametrize("initial", ["missing", "empty"])
def test_wisdom_missing_or_empty_csv_trains_exactly_once(tmp_path: Path, initial: str) -> None:
    score_path = tmp_path / "scores.csv"
    if initial == "empty":
        score_path.touch()
    calls: list[str] = []

    def train_scores(path: str) -> str:
        calls.append(path)
        _write_scores(Path(path), [{"LayerName": "features.0", "NeuronIndex": 0, "Score": 1.0}])
        return path

    artifact = run_wisdom._resolve_score_artifact(
        _namespace(mode="wisdom", wisdom_csv=str(score_path), idc_csv=None), _prepared(train_scores)
    )
    assert calls == [str(score_path)]
    assert artifact.path == score_path and artifact.status == "generated"


def test_existing_valid_wisdom_csv_is_reused(tmp_path: Path) -> None:
    score_path = _scores_for_tiny(tmp_path / "scores.csv")
    artifact = run_wisdom._resolve_score_artifact(
        _namespace(wisdom_csv=str(score_path)),
        _prepared(lambda path: pytest.fail("valid score CSV was regenerated")),
    )
    assert artifact.status == "reused"


def test_malformed_nonempty_wisdom_csv_is_not_overwritten(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.csv"
    score_path.write_text("LayerName,NeuronIndex\nfeatures.0,0\n", encoding="utf-8")
    original = score_path.read_bytes()
    with pytest.raises(ValueError, match="Score"):
        run_wisdom._resolve_score_artifact(
            _namespace(wisdom_csv=str(score_path)),
            _prepared(lambda path: pytest.fail("malformed CSV was overwritten")),
        )
    assert score_path.read_bytes() == original


def test_omitted_wisdom_csv_derives_from_output_stem(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"
    calls: list[str] = []

    def train_scores(path: str) -> str:
        calls.append(path)
        _write_scores(Path(path), [{"LayerName": "features.0", "NeuronIndex": 0, "Score": 1.0}])
        return path

    artifact = run_wisdom._resolve_score_artifact(
        _namespace(output_json=str(output), wisdom_csv=None), _prepared(train_scores)
    )
    expected = tmp_path / "coverage_wisdom_scores.csv"
    assert calls == [str(expected)]
    assert artifact.path == expected


def test_wisdom_generation_validates_the_requested_score_destination(tmp_path: Path) -> None:
    requested = tmp_path / "requested.csv"
    elsewhere = _write_scores(
        tmp_path / "elsewhere.csv",
        [{"LayerName": "features.0", "NeuronIndex": 0, "Score": 1.0}],
    )
    with pytest.raises(ValueError):
        run_wisdom._resolve_score_artifact(
            _namespace(wisdom_csv=str(requested)), _prepared(lambda path: str(elsewhere))
        )
    assert not requested.exists()


@pytest.mark.parametrize("value", [None, "", "missing.csv"])
def test_idc_csv_is_never_generated(tmp_path: Path, value: str | None) -> None:
    path = None if value is None else str(tmp_path / value)
    with pytest.raises(ValueError, match="existing nonempty IDC"):
        run_wisdom._resolve_score_artifact(
            _namespace(mode="idc", wisdom_csv=None, idc_csv=path),
            _prepared(lambda path: pytest.fail("IDC invoked WISDOM pretraining")),
        )


def test_mode_specific_score_paths_never_substitute(tmp_path: Path) -> None:
    idc_path = _scores_for_tiny(tmp_path / "idc.csv")
    with pytest.raises(ValueError, match="idc-csv"):
        run_wisdom._resolve_score_artifact(
            _namespace(mode="wisdom", wisdom_csv=None, idc_csv=str(idc_path)), _prepared(lambda path: path)
        )


def test_global_top_m_is_total_across_considered_layers(tmp_path: Path) -> None:
    selected, plan = run_wisdom._resolve_selected_neurons(
        _namespace(top_m_neurons=2), _prepared(lambda path: path), _scores_for_tiny(tmp_path / "scores.csv")
    )
    assert plan.eligible_layers == ("features.0", "features.2", "features.4")
    assert selected == {"features.0": [1], "features.2": [0]}


def test_global_selection_never_manufactures_sparse_csv_indices(tmp_path: Path) -> None:
    score_path = _write_scores(
        tmp_path / "scores.csv",
        [{"LayerName": "features.0", "NeuronIndex": 2, "Score": 1.0}],
    )

    selected, _ = run_wisdom._resolve_selected_neurons(
        _namespace(top_m_neurons=3), _prepared(lambda path: path), score_path
    )

    assert selected == {"features.0": [2]}


def test_per_layer_top_m_is_applied_independently(tmp_path: Path) -> None:
    selected, _ = run_wisdom._resolve_selected_neurons(
        _namespace(selection_mode="per-layer", top_m_neurons=1),
        _prepared(lambda path: path), _scores_for_tiny(tmp_path / "scores.csv")
    )
    assert selected == {"features.0": [1], "features.2": [0], "features.4": [0]}


def test_per_group_selection_covers_each_discovered_group(tmp_path: Path) -> None:
    selected, plan = run_wisdom._resolve_selected_neurons(
        _namespace(selection_mode="per-group", num_groups=3, top_m_neurons=1),
        _prepared(lambda path: path), _scores_for_tiny(tmp_path / "scores.csv")
    )
    assert tuple(plan.groups) == ("early", "middle", "late")
    assert {layer for group in plan.groups.values() for layer in group} == set(plan.eligible_layers)
    assert set(selected) == set(plan.eligible_layers)


def test_num_layers_limits_evenly_before_grouping(tmp_path: Path) -> None:
    selected, plan = run_wisdom._resolve_selected_neurons(
        _namespace(selection_mode="per-group", num_groups=2, num_layers=2, top_m_neurons=1),
        _prepared(lambda path: path), _scores_for_tiny(tmp_path / "scores.csv")
    )
    assert plan.eligible_layers == ("features.0", "features.4")
    assert set(selected) == {"features.0", "features.4"}


class _WrappedModel(nn.Module):
    def __init__(self, wrapper_name: str) -> None:
        super().__init__()
        setattr(
            self,
            wrapper_name,
            nn.Sequential(nn.Conv2d(3, 2, 1), nn.Conv2d(2, 2, 1)),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return next(self.children())(inputs)


@pytest.mark.parametrize("wrapper_name", ["yolo_model", "pose_model"])
def test_wrapper_prefixed_score_names_resolve_without_rewriting(tmp_path: Path, wrapper_name: str) -> None:
    model = _WrappedModel(wrapper_name).eval()
    loader = DataLoader(TensorDataset(torch.zeros(1, 3, 8, 8), torch.tensor([0])), batch_size=1)
    adapter = ClassificationAdapter(model)
    prepared = PreparedTask("classification", adapter, loader, None, loader, lambda path: path,
                            lambda current: TaskEvaluation({}, None, None), "wrapped", "state-dict")
    score_path = _write_scores(tmp_path / "scores.csv", [{"LayerName": f"{wrapper_name}.0", "NeuronIndex": 0, "Score": 1.0}])
    selected, _ = run_wisdom._resolve_selected_neurons(_namespace(), prepared, score_path)
    assert selected == {f"{wrapper_name}.0": [0]}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"LayerName": "missing.layer", "NeuronIndex": 0, "Score": 1.0}], "not found"),
        ([{"LayerName": "features.0", "NeuronIndex": 0, "Score": 0.0}], "no neurons"),
    ],
)
def test_selection_rejects_unknown_layers_and_empty_groups(tmp_path: Path, rows: list[dict[str, object]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_wisdom._resolve_selected_neurons(_namespace(), _prepared(lambda path: path), _write_scores(tmp_path / "scores.csv", rows))


def test_engine_honors_wisdom_cluster_config_and_idc_forces_silhouette() -> None:
    prepared = _prepared(lambda path: path)
    plan = build_layer_plan(("features.0",), num_layers=1)
    engine = run_wisdom._build_engine(_namespace(), prepared, plan, cluster_method="Birch", n_clusters=3)
    assert engine.model is prepared.adapter.analysis_model
    assert engine.impl == "wisdom"
    assert engine.cluster.method == "Birch"
    assert engine.cluster.params["n_clusters"] == 3
    assert engine.cfg.layer_groups == plan.groups

    idc = run_wisdom._build_engine(_namespace(mode="idc"), prepared, plan, cluster_method="Birch", n_clusters=3)
    assert idc.impl == "idc"
    assert idc.cluster.method == "KMeans"
    assert idc.cluster.use_silhouette is True


@pytest.mark.parametrize("method", ["", "   "])
def test_direct_cluster_helpers_reject_blank_configured_methods(method: str) -> None:
    """Reporting or building an empty configured method must fail before factory defaults."""
    prepared = _prepared_with_validation_metric(lambda loader: 0.0)
    with pytest.raises(ValueError, match="cluster-method.*nonblank"):
        run_wisdom._resolve_cluster_configuration(
            _args(cluster_method=method), prepared, {"features.0": [0]}, _plan()
        )
    with pytest.raises(ValueError, match="cluster-method.*nonblank"):
        run_wisdom._build_engine(
            _namespace(cluster_method=method), prepared, _plan(), cluster_method=method, n_clusters=2
        )


def test_configured_cluster_method_is_trimmed_before_reporting_and_execution() -> None:
    """Reporting a spelling different from the constructed clusterer must fail this test."""
    prepared = _prepared_with_validation_metric(lambda loader: 0.0)
    configuration = run_wisdom._resolve_cluster_configuration(
        _args(cluster_method="  Birch  "), prepared, {"features.0": [0]}, _plan()
    )
    engine = run_wisdom._build_engine(
        _namespace(cluster_method="  Birch  "), prepared, _plan(), n_clusters=2
    )

    assert configuration["cluster_method"] == engine.cluster.method == "Birch"
    assert configuration["n_clusters"] == engine.cluster.params["n_clusters"] == 2


def test_cluster_capabilities_pass_spectral_count_and_seed_but_leave_countless_methods_default() -> None:
    """Replacing capability-aware parameters with a generic count must fail this test."""
    prepared = _prepared(lambda path: path)
    plan = build_layer_plan(("features.0",), num_layers=1)

    spectral = run_wisdom._build_engine(
        _namespace(seed=17), prepared, plan, cluster_method="SpectralClustering", n_clusters=3
    )
    assert spectral.cluster.params == {"n_clusters": 3, "random_state": 17}

    for method in ("MeanShift", "DBSCAN", "OPTICS"):
        engine = run_wisdom._build_engine(
            _namespace(), prepared, plan, cluster_method=method, n_clusters=None
        )
        assert engine.cluster.params == {}


@pytest.mark.parametrize("method", ["MeanShift", "DBSCAN", "OPTICS"])
def test_configured_countless_methods_report_none_and_build_without_a_count(method: str) -> None:
    """Reporting the requested count for methods that ignore it must fail this test."""
    prepared = _prepared_with_validation_metric(lambda loader: 0.0)
    configuration = run_wisdom._resolve_cluster_configuration(
        _args(cluster_method=method, n_clusters=3), prepared, {"features.0": [0]}, _plan()
    )
    assert configuration == {
        "source": "configured",
        "cluster_method": method,
        "n_clusters": None,
    }


def test_bo_rejects_count_incompatible_methods_before_search() -> None:
    """Allowing a count-irrelevant BO dimension must fail this test."""
    with pytest.raises(ValueError, match="count-compatible"):
        run_wisdom._validate_args(
            _args(
                model_factory="factory",
                bo=True,
                bo_cluster_methods="KMeans,MeanShift",
                bo_n_clusters="2",
            )
        )


def test_direct_cluster_configuration_rejects_count_incompatible_methods_before_search(monkeypatch) -> None:
    """Allowing direct run orchestration to invoke BO for MeanShift must fail this test."""
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_bo was called")

    monkeypatch.setattr(run_wisdom, "run_bo", fail_if_called)
    with pytest.raises(ValueError, match="count-compatible"):
        run_wisdom._resolve_cluster_configuration(
            _args(bo=True, bo_cluster_methods="MeanShift", bo_n_clusters="2"),
            _prepared_with_validation_metric(lambda loader: 0.0),
            {"features.0": [0]},
            _plan(),
        )
    assert called is False


def _args(**overrides: object) -> Namespace:
    values = vars(_namespace()).copy()
    values.update(
        {
            "bo": False,
            "bo_backend": "auto",
            "bo_init": 3,
            "bo_iter": 3,
            "bo_candidate_pool_size": 32,
            "bo_cluster_methods": "KMeans,MiniBatchKMeans,Birch",
            "bo_n_clusters": "2,3,4",
            "bo_output_json": None,
            "build_data_path": "build",
            "validation_data_path": "validation",
            "test_data_path": "test",
            "weights_path": "weights.pth",
            "model_path": None,
            "plot_neurons": False,
            "plot_top_k": None,
        }
    )
    values.update(overrides)
    return Namespace(**values)


def _plan():
    return build_layer_plan(("features.0",), num_layers=1)


def _prepared_with_validation_metric(metric_fn, *, task: str = "classification") -> PreparedTask:
    base = _prepared(lambda path: path)
    build = DataLoader(
        TensorDataset(torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    validation = DataLoader(
        TensorDataset(torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])), batch_size=2
    )

    def evaluate(loader):
        if loader is base.test_loader:
            raise AssertionError("BO objective must not evaluate test data")
        value = float(metric_fn(loader))
        metric = "pose_confidence_surrogate" if task == "pose" else "f1"
        metrics = {"loss": None, "pck": None, metric: value} if task == "pose" else {
            "accuracy": None, "loss": None, "f1": value
        }
        return TaskEvaluation(metrics, metric, value)

    return PreparedTask(
        task=task,
        adapter=base.adapter,
        build_loader=build,
        validation_loader=validation,
        test_loader=base.test_loader,
        score_trainer=base.score_trainer,
        evaluate=evaluate,
        model_name=base.model_name,
        checkpoint_format=base.checkpoint_format,
    )


def test_coverage_objective_uses_fresh_engines_and_validation_prefixes_only(monkeypatch) -> None:
    seen_sizes: list[int] = []
    engines = []

    class FakeEngine:
        def fit_selected(self, loader, selected, device):
            assert loader is prepared.build_loader
            return selected

        def coverage_details(self, loader, selected, device):
            size = len(loader.dataset)
            seen_sizes.append(size)
            return {"coverage_rate": size / 4.0, "max_coverage": 1.0, "total_combinations": 4, "scope_details": {}}

    def new_engine(*args, **kwargs):
        engine = FakeEngine()
        engines.append(engine)
        return engine

    prepared = _prepared_with_validation_metric(lambda loader: len(loader.dataset) / 4.0)
    monkeypatch.setattr(run_wisdom, "_build_engine", new_engine)
    objective = run_wisdom._coverage_objective(_args(bo=True), prepared, {"features.0": [0]}, _plan())

    assert objective({"cluster_method": "KMeans", "n_clusters": 2}) == pytest.approx(1.0)
    assert objective({"cluster_method": "Birch", "n_clusters": 2}) == pytest.approx(1.0)
    prefix_count = len(run_wisdom._suite_sizes(len(prepared.validation_loader.dataset)))
    assert seen_sizes[:prefix_count] == sorted(seen_sizes[:prefix_count])
    assert seen_sizes[prefix_count:] == sorted(seen_sizes[prefix_count:])
    assert seen_sizes[prefix_count - 1] == len(prepared.validation_loader.dataset)
    assert len(engines) == 2


def test_pearson_helpers_are_deterministic_for_degenerate_series() -> None:
    assert run_wisdom._suite_sizes(4, points=3) == [1, 2, 4]
    assert run_wisdom._suite_sizes(1) == [1]
    assert run_wisdom._pearson_correlation([1.0], [1.0]) == 0.0
    assert run_wisdom._pearson_correlation([1.0, 1.0], [1.0, 2.0]) == 0.0


def test_cluster_configuration_propagates_bo_options_and_pose_surrogate(monkeypatch, tmp_path: Path) -> None:
    prepared = _prepared_with_validation_metric(lambda loader: len(loader.dataset) / 4.0, task="pose")
    captured = {}

    def fake_run_bo(*args, **kwargs):
        captured.update(kwargs)
        return SearchResult({"cluster_method": "Birch", "n_clusters": 3}, 0.75,
                            [({"cluster_method": "Birch", "n_clusters": 3}, 0.75)], "sklearn"), str(tmp_path / "bo.json")

    monkeypatch.setattr(run_wisdom, "run_bo", fake_run_bo)
    configuration = run_wisdom._resolve_cluster_configuration(
        _args(bo=True, bo_backend="sklearn", bo_init=2, bo_iter=4, bo_candidate_pool_size=17,
              bo_output_json=str(tmp_path / "requested.json")),
        prepared, {"features.0": [0]}, _plan(),
    )

    assert configuration == {
        "source": "bo", "backend": "sklearn", "best_cluster_method": "Birch",
        "best_n_clusters": 3, "best_score": 0.75,
        "validation_metric": "pose_confidence_surrogate", "history_json": str(tmp_path / "bo.json"),
    }
    assert captured == {
        "random_state": 42, "backend": "sklearn", "n_init": 2, "n_iter": 4,
        "candidate_pool_size": 17, "out_path": str(tmp_path / "requested.json"),
        "payload_extras": {"validation_metric": "pose_confidence_surrogate"},
    }


def test_idc_reports_and_runs_its_forced_effective_cluster_configuration(monkeypatch, tmp_path: Path, capsys) -> None:
    base = _prepared(lambda path: path)
    build = DataLoader(
        TensorDataset(torch.zeros(4, 3, 8, 8), torch.tensor([0, 1, 0, 1])), batch_size=2
    )
    prepared = PreparedTask(
        task=base.task, adapter=base.adapter, build_loader=build, validation_loader=None,
        test_loader=base.test_loader, score_trainer=base.score_trainer, evaluate=base.evaluate,
        model_name=base.model_name, checkpoint_format=base.checkpoint_format,
    )
    output = tmp_path / "idc.json"
    seen = []

    class FakeEngine:
        def fit_selected(self, loader, selected, device):
            assert loader is prepared.build_loader
            return selected

        def coverage_details(self, loader, selected, device):
            assert loader is prepared.test_loader
            return {"coverage_rate": 0.5, "max_coverage": 1.0, "total_combinations": 2, "scope_details": {}}

    monkeypatch.setattr(run_wisdom, "_prepare_task", lambda args: prepared)
    monkeypatch.setattr(run_wisdom, "_resolve_score_artifact", lambda args, prepared: run_wisdom.ScoreArtifact(tmp_path / "idc.csv", "reused"))
    monkeypatch.setattr(run_wisdom, "_resolve_selected_neurons", lambda args, prepared, path: ({"features.0": [0]}, _plan()))
    monkeypatch.setattr(run_wisdom, "_build_engine", lambda *args: seen.append(args[-2:]) or FakeEngine())

    summary = run_wisdom.run(_args(mode="idc", cluster_method="Birch", n_clusters=3, output_json=str(output)))
    terminal = capsys.readouterr().out

    expected = {"source": "configured", "cluster_method": "KMeans", "n_clusters": 2}
    assert summary["cluster_configuration"] == expected
    assert json.loads(output.read_text())["cluster_configuration"] == expected
    assert json.dumps(expected, indent=2) in terminal
    assert seen == [("KMeans", 2)]


def test_run_writes_honest_summary_with_a_new_final_engine(monkeypatch, tmp_path: Path, capsys) -> None:
    output = tmp_path / "nested" / "coverage.json"
    base = _prepared(lambda path: path)
    build = DataLoader(
        TensorDataset(torch.zeros(18, 3, 8, 8), torch.arange(18) % 3), batch_size=2
    )
    validation = DataLoader(
        TensorDataset(torch.zeros(2, 3, 8, 8), torch.tensor([0, 1])), batch_size=1
    )
    prepared = PreparedTask(
        task=base.task,
        adapter=base.adapter,
        build_loader=build,
        validation_loader=validation,
        test_loader=base.test_loader,
        score_trainer=base.score_trainer,
        evaluate=base.evaluate,
        model_name=base.model_name,
        checkpoint_format=base.checkpoint_format,
    )
    engines = []

    class FakeEngine:
        def __init__(self, method, count):
            self.method, self.count = method, count

        def fit_selected(self, loader, selected, device):
            assert loader is prepared.build_loader
            return selected

        def coverage_details(self, loader, selected, device):
            assert loader is prepared.test_loader
            return {"coverage_rate": 0.5, "max_coverage": 1.0, "total_combinations": 2, "scope_details": {}}

    monkeypatch.setattr(run_wisdom, "_prepare_task", lambda args: prepared)
    monkeypatch.setattr(run_wisdom, "_resolve_score_artifact", lambda args, prepared: run_wisdom.ScoreArtifact(tmp_path / "scores.csv", "reused"))
    monkeypatch.setattr(run_wisdom, "_resolve_selected_neurons", lambda args, prepared, path: ({"features.0": [0]}, _plan()))
    monkeypatch.setattr(run_wisdom, "_resolve_cluster_configuration", lambda *args: {
        "source": "bo", "backend": "sklearn", "best_cluster_method": "Birch", "best_n_clusters": 3,
        "best_score": 0.75, "validation_metric": "f1", "history_json": str(tmp_path / "bo.json"),
    })

    def build_engine(args, prepared, plan, cluster_method=None, n_clusters=None):
        engine = FakeEngine(cluster_method, n_clusters)
        engines.append(engine)
        return engine

    monkeypatch.setattr(run_wisdom, "_build_engine", build_engine)
    summary = run_wisdom.run(
        _args(
            output_json=str(output),
            bo=True,
            validation_data_path=None,
        )
    )
    terminal = capsys.readouterr().out

    assert output.exists() and json.loads(output.read_text()) == summary
    assert summary["metrics"] == {"accuracy": None, "loss": None, "f1": 0.0}
    assert summary["coverage"]["coverage_rate"] == 0.5
    assert summary["data"]["validation_path"] is None
    assert summary["data"]["validation_source"] == "build_holdout"
    assert summary["data"]["validation_fraction"] == pytest.approx(0.10)
    assert engines[-1].method == "Birch" and engines[-1].count == 3
    assert "N/A" in terminal and "TinyClassifier" in terminal
    assert json.dumps(summary["cluster_configuration"], indent=2) in terminal
