from __future__ import annotations

from collections import OrderedDict

import pytest
import torch.nn as nn
import pandas as pd
import torch

from wisdom.core.layers import (
    build_layer_plan,
    discover_eligible_layers,
    limit_layers_evenly,
)
from wisdom.core.wisdom import (
    WisdomConfig,
    WisdomIDC,
    load_groupwise_top_neurons,
    split_selected_by_group,
)
from wisdom.core.wisdom_train import (
    ConsensusWisdom,
    WisdomTrainConfig,
    _select_top_neurons_per_group,
    _select_top_neurons_per_layer,
)


@pytest.mark.parametrize("mode", ["per-group", "per-layer"])
def test_multimethod_voting_cannot_starve_a_low_scale_scope(tmp_path, monkeypatch, mode):
    """A global weighted cutoff must not replace the requested M per scope."""
    from torch.utils.data import DataLoader, TensorDataset
    import wisdom.core.wisdom_train as training

    model = nn.Sequential(OrderedDict([
        ("front", nn.Linear(2, 2)), ("back", nn.Linear(2, 2)), ("head", nn.Linear(2, 2)),
    ])).eval()
    loader = DataLoader(TensorDataset(torch.ones(1, 2), torch.zeros(1, dtype=torch.long)))
    method_scores = {
        "first": {"front": torch.tensor([100., 0.]), "back": torch.tensor([1., 0.])},
        "second": {"front": torch.tensor([0., 90.]), "back": torch.tensor([0., 0.9])},
    }
    # Fixed method attributions/gains isolate consensus ranking from Captum noise.
    monkeypatch.setattr(training, "batch_per_layer_scores", lambda **kw: method_scores[kw["method"]])
    monkeypatch.setattr(training, "_prune_and_evaluate", lambda *args: 100.)
    cfg = WisdomTrainConfig(
        methods=["first", "second"], device="cpu", voting_mode="fine-grained",
        selection_mode=mode, n_groups=2, out_csv=str(tmp_path / "scores.csv"),
    )

    scores, _ = ConsensusWisdom(model, device="cpu").fit(loader, cfg, top_m_neurons=1)

    assert scores["front"].nonzero().flatten().tolist() == [0]
    assert scores["back"].nonzero().flatten().tolist() == [0]


def _scaled_model(repeats: int) -> nn.Module:
    layers = OrderedDict()
    for index in range(repeats):
        layers[f"stage_{index}"] = nn.Conv2d(3, 3, kernel_size=1)
        layers[f"relu_{index}"] = nn.ReLU()
    layers["classifier"] = nn.Linear(3, 2)
    return nn.Sequential(layers)


def test_discovery_preserves_model_registration_order_and_exclusions() -> None:
    model = nn.Sequential(
        OrderedDict(
            [
                ("stage_z", nn.Conv2d(3, 4, kernel_size=1)),
                ("stage_a", nn.Conv2d(4, 4, kernel_size=1)),
                ("activation", nn.ReLU()),
                ("head", nn.Linear(4, 2)),
            ]
        )
    )

    assert discover_eligible_layers(model) == ("stage_z", "stage_a", "head")
    assert discover_eligible_layers(model, excluded_layers=("head",)) == (
        "stage_z",
        "stage_a",
    )
    assert discover_eligible_layers(model, excluded_prefixes=("stage_",)) == ("head",)


def test_different_model_scales_discover_their_own_layer_counts() -> None:
    small = discover_eligible_layers(_scaled_model(2))
    large = discover_eligible_layers(_scaled_model(5))

    assert small == ("stage_0", "stage_1", "classifier")
    assert large == (
        "stage_0",
        "stage_1",
        "stage_2",
        "stage_3",
        "stage_4",
        "classifier",
    )
    assert len(large) > len(small)


def test_even_layer_limit_and_requested_group_count_cover_layers_once() -> None:
    names = tuple(f"layer_{index}" for index in range(8))

    assert limit_layers_evenly(names, 4) == (
        "layer_0",
        "layer_2",
        "layer_5",
        "layer_7",
    )
    plan = build_layer_plan(names, n_groups=3, num_layers=7)

    assert tuple(plan.groups) == ("early", "middle", "late")
    assert [len(group) for group in plan.groups.values()] == [3, 2, 2]
    flattened = tuple(layer for group in plan.groups.values() for layer in group)
    assert flattened == plan.eligible_layers
    assert len(flattened) == len(set(flattened)) == 7


@pytest.mark.parametrize(
    ("names", "n_groups", "num_layers", "message"),
    [
        (("a", "b"), 0, None, "n_groups"),
        (("a", "b"), 3, None, "eligible layers"),
        (("a", "b"), 1, 0, "num_layers"),
        (("a", "b"), 1, 3, "num_layers"),
        ((), 1, None, "eligible layers"),
    ],
)
def test_invalid_group_and_layer_counts_fail_clearly(
    names: tuple[str, ...],
    n_groups: int,
    num_layers: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_layer_plan(names, n_groups=n_groups, num_layers=num_layers)


def test_global_plan_does_not_validate_an_irrelevant_group_count() -> None:
    plan = build_layer_plan(("a", "b"), n_groups=None, num_layers=1)

    assert plan.groups == {}
    assert plan.eligible_layers == ("a",)


def test_sparse_selected_layers_keep_original_group_membership() -> None:
    names = tuple(f"layer_{index}" for index in range(6))
    plan = build_layer_plan(names, n_groups=3)
    selected = {
        "layer_0": [0],
        "layer_1": [0],
        "layer_4": [0],
        "layer_5": [0],
    }

    grouped = split_selected_by_group(selected, layer_groups=plan.groups)

    assert set(grouped["early"]) == {"layer_0", "layer_1"}
    assert "middle" not in grouped
    assert set(grouped["late"]) == {"layer_4", "layer_5"}


def test_coverage_scopes_reuse_configured_groups() -> None:
    names = tuple(f"layer_{index}" for index in range(6))
    plan = build_layer_plan(names, n_groups=3)
    engine = WisdomIDC(
        nn.Identity(),
        cfg=WisdomConfig(selection_mode="per-group", layer_groups=plan.groups),
    )

    scopes = engine._scope_keys(
        {
            "layer_0": [0],
            "layer_1": [0],
            "layer_4": [0],
            "layer_5": [0],
        }
    )

    assert scopes == {
        "early": ["layer_0:0", "layer_1:0"],
        "late": ["layer_4:0", "layer_5:0"],
    }


def test_groupwise_csv_selection_uses_model_derived_groups(tmp_path) -> None:
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"LayerName": "a", "NeuronIndex": 0, "Score": 1.0},
            {"LayerName": "b", "NeuronIndex": 0, "Score": 9.0},
            {"LayerName": "c", "NeuronIndex": 0, "Score": 8.0},
            {"LayerName": "d", "NeuronIndex": 0, "Score": 2.0},
        ]
    ).to_csv(csv_path, index=False)
    plan = build_layer_plan(("a", "b", "c", "d"), n_groups=2)

    selected = load_groupwise_top_neurons(
        str(csv_path),
        per_group_k=1,
        strip_prefix="",
        layer_groups=plan.groups,
    )

    assert selected == {"b": [0], "c": [0]}


def test_groupwise_csv_selection_rejects_stale_layers(tmp_path) -> None:
    csv_path = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            {"LayerName": "a", "NeuronIndex": 0, "Score": 1.0},
            {"LayerName": "stale", "NeuronIndex": 0, "Score": 100.0},
        ]
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="stale"):
        load_groupwise_top_neurons(
            str(csv_path),
            per_group_k=1,
            strip_prefix="",
            layer_groups={"front": ("a",), "back": ("b",)},
        )


def test_discovery_ignores_modules_with_no_trainable_parameters() -> None:
    model = _scaled_model(2)
    model.stage_1.weight.requires_grad_(False)
    model.stage_1.bias.requires_grad_(False)

    assert discover_eligible_layers(model) == ("stage_0", "classifier")


def test_per_group_neuron_selection_uses_arbitrary_discovered_groups() -> None:
    plan = build_layer_plan(("stem", "branch", "neck", "tail"), n_groups=4)
    scores = {
        "stem": torch.tensor([1.0, 2.0]),
        "branch": torch.tensor([3.0, 1.0]),
        "neck": torch.tensor([2.0, 4.0]),
        "tail": torch.tensor([5.0, 1.0]),
    }

    selected_by_layer, triplets = _select_top_neurons_per_group(
        scores,
        top_m_per_group=1,
        layer_groups=plan.groups,
    )

    assert {layer for layer, _, _ in triplets} == set(plan.eligible_layers)
    assert {layer: indices.tolist() for layer, indices in selected_by_layer.items()} == {
        "stem": [1],
        "branch": [0],
        "neck": [1],
        "tail": [0],
    }


def test_per_group_neuron_selection_rejects_nonpositive_top_m() -> None:
    with pytest.raises(ValueError, match="top_m_per_group"):
        _select_top_neurons_per_group(
            {"layer": torch.tensor([1.0])},
            top_m_per_group=0,
            layer_groups={"only": ("layer",)},
        )


def test_per_layer_selection_applies_top_m_to_each_considered_layer() -> None:
    scores = {
        "a": torch.tensor([1.0, 4.0]),
        "b": torch.tensor([3.0, 2.0]),
    }

    _, selected = _select_top_neurons_per_layer(scores, top_m_per_layer=1)

    assert {(layer, index) for layer, _score, index in selected} == {
        ("a", 1),
        ("b", 0),
    }
