from __future__ import annotations

import importlib.util
import json

import pytest

from wisdom.utils.search import BOSearch, run_bo


SPACE = {"cluster_method": ["KMeans", "Birch"], "n_clusters": [2, 3]}


def objective(config: dict[str, object]) -> float:
    return float(config["cluster_method"] == "Birch") + 0.1 * int(config["n_clusters"])


def test_sklearn_exhausts_tiny_space_without_duplicates(tmp_path) -> None:
    result, written = run_bo(
        SPACE,
        objective,
        backend="sklearn",
        random_state=17,
        n_init=2,
        n_iter=2,
        candidate_pool_size=4,
        out_path=tmp_path / "search.json",
        payload_extras={"validation_metric": "f1"},
    )
    assert result.backend == "sklearn"
    assert result.best_config == {"cluster_method": "Birch", "n_clusters": 3}
    assert len(result.history) == len({tuple(sorted(cfg.items())) for cfg, _ in result.history}) == 4
    payload = json.loads((tmp_path / "search.json").read_text())
    assert written == str(tmp_path / "search.json")
    assert payload["backend_used"] == "sklearn"
    assert payload["best_config"] == result.best_config
    assert payload["validation_metric"] == "f1"


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({}, "must not be empty"),
        ({"method": []}, "nonempty"),
        ({"x": (3, 1)}, "lower bound"),
        ({"x": (1, 3, "unknown")}, "kind"),
    ],
)
def test_invalid_search_spaces_fail_before_objective(space, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        BOSearch(space)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_init": 0}, "n_init"),
        ({"n_iter": -1}, "n_iter"),
        ({"candidate_pool_size": 0}, "candidate_pool_size"),
        ({"backend": "magic"}, "backend"),
    ],
)
def test_invalid_search_controls_fail(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        run_bo({"x": [1, 2]}, lambda cfg: float(cfg["x"]), **kwargs)


@pytest.mark.bo
def test_actual_botorch_backend_runs_acquisition_without_cross_fallback() -> None:
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    result, _ = run_bo(
        {"n_clusters": [2, 3, 4]},
        lambda cfg: -abs(int(cfg["n_clusters"]) - 4),
        backend="botorch",
        random_state=23,
        n_init=2,
        n_iter=1,
        candidate_pool_size=8,
    )
    assert result.backend == "botorch"
    assert len(result.history) == 3
    assert len({tuple(sorted(cfg.items())) for cfg, _ in result.history}) == 3
    assert result.best_config == {"n_clusters": 4}


def test_auto_reports_the_backend_that_is_really_installed() -> None:
    expected = (
        "botorch"
        if importlib.util.find_spec("botorch") and importlib.util.find_spec("gpytorch")
        else "sklearn"
    )
    result, _ = run_bo(
        {"n_clusters": [2, 3, 4]},
        lambda cfg: float(cfg["n_clusters"]),
        backend="auto",
        random_state=29,
        n_init=2,
        n_iter=1,
        candidate_pool_size=8,
    )
    assert result.backend == expected


def test_seed_reproduces_sklearn_history() -> None:
    kwargs = dict(backend="sklearn", random_state=31, n_init=2, n_iter=2, candidate_pool_size=4)
    first, _ = run_bo(SPACE, objective, **kwargs)
    second, _ = run_bo(SPACE, objective, **kwargs)
    assert first.history == second.history


def test_sklearn_honors_a_single_candidate_pool_entry() -> None:
    class CountingBOSearch(BOSearch):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.unseen_draws = 0

        def _random_unseen_config(self, seen):
            self.unseen_draws += 1
            return super()._random_unseen_config(seen)

    search = CountingBOSearch({"x": (0.0, 1.0)}, random_state=37, backend="sklearn")
    search.optimize(lambda config: float(config["x"]), n_init=1, n_iter=1, candidate_pool_size=1)
    assert search.unseen_draws == 2


def test_sklearn_deduplicates_finite_categorical_values() -> None:
    result, _ = run_bo(
        {"x": ["a", "a"]},
        lambda config: float(config["x"] == "a"),
        backend="sklearn",
        random_state=41,
        n_init=2,
        n_iter=1,
    )
    assert result.history == [({"x": "a"}, 1.0)]


@pytest.mark.bo
def test_botorch_deduplicates_finite_categorical_values() -> None:
    pytest.importorskip("botorch")
    pytest.importorskip("gpytorch")
    result, _ = run_bo(
        {"x": ["a", "a"]},
        lambda config: float(config["x"] == "a"),
        backend="botorch",
        random_state=43,
        n_init=2,
        n_iter=0,
    )
    assert result.history == [({"x": "a"}, 1.0)]


def test_integer_dimensions_reject_fractional_bounds() -> None:
    with pytest.raises(ValueError, match="integral bounds"):
        BOSearch({"x": (1.5, 3, "int")})
