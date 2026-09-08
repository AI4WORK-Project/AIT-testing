from __future__ import annotations

import pytest

from wisdom.core.layers import build_layer_plan, discover_eligible_layers
from wisdom.tasks.detection import DetectionAdapter


@pytest.mark.ultralytics
def test_two_actual_yolo_scales_evenly_limit_dynamic_per_group_layers() -> None:
    """Layer plans must derive from each real architecture, never fixed indices."""
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    layer_counts: dict[str, int] = {}
    for config_name in ("yolo11n.yaml", "yolo11m.yaml"):
        adapter = DetectionAdapter(
            DetectionModel(config_name, ch=3, nc=2, verbose=False).eval(),
            num_classes=2,
        )
        eligible = discover_eligible_layers(
            adapter.analysis_model,
            excluded_prefixes=adapter.excluded_layer_prefixes(),
        )
        plan = build_layer_plan(eligible, n_groups=4, num_layers=8)
        flattened = tuple(
            layer for group_layers in plan.groups.values() for layer in group_layers
        )
        assert len(eligible) >= 8
        assert len(plan.groups) == 4
        assert len(plan.eligible_layers) == 8
        assert len(flattened) == len(set(flattened)) == 8
        assert flattened == plan.eligible_layers
        assert plan.eligible_layers[0] == eligible[0]
        assert plan.eligible_layers[-1] == eligible[-1]
        assert not any(
            layer.startswith(adapter.excluded_layer_prefixes()) for layer in eligible
        )
        layer_counts[config_name] = len(eligible)

    assert layer_counts["yolo11n.yaml"] != layer_counts["yolo11m.yaml"]
    with pytest.raises(ValueError, match="num_layers"):
        build_layer_plan(("a", "b"), n_groups=1, num_layers=0)
    with pytest.raises(ValueError, match="eligible layers"):
        build_layer_plan(("a", "b"), n_groups=3)
