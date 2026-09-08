from __future__ import annotations

import torch
import pytest
from torch.utils.data import DataLoader

from ..helpers import TinyPoseModel
from wisdom.core.wisdom import ClusteringConfig, WisdomConfig, WisdomIDC
from wisdom.core.wisdom_train import train_wisdom_pose
from wisdom.tasks.pose import PoseAdapter
from wisdom.utils.io_cache import read_layer_scores_csv


def test_pose_pretraining_and_coverage_with_unlabeled_batches(tmp_path) -> None:
    torch.manual_seed(23)
    raw_model = TinyPoseModel().eval()
    adapter = PoseAdapter(
        raw_model,
        output_layer_names=("cmap_head", "paf_head"),
    )
    images = torch.randn(4, 3, 8, 8)
    loader = DataLoader(images, batch_size=4, shuffle=False)

    csv_path = train_wisdom_pose(
        raw_model,
        loader,
        str(tmp_path / "pose.csv"),
        output_layer_names=("cmap_head", "paf_head"),
        top_m=1,
        methods=["lgxa"],
        device="cpu",
    )

    scores = read_layer_scores_csv(csv_path)
    assert scores
    assert not any(name.endswith(("cmap_head", "paf_head")) for name in scores)
    engine = WisdomIDC(
        adapter.analysis_model,
        cfg=WisdomConfig(top_m_neurons=1),
        cluster=ClusteringConfig(
            method="KMeans",
            params={"n_clusters": 2, "random_state": 7, "n_init": 10},
        ),
    )
    selected = engine.fit(loader, scores, device="cpu")
    rate, total, maximum = engine.coverage(loader, selected, device="cpu")

    assert 0.0 <= rate <= maximum <= 1.0
    assert total >= 1


@pytest.mark.parametrize("method_source", ["api_default", "script_default"])
def test_pose_default_pretraining_runs_multiple_attribution_methods(
    tmp_path, method_source: str,
) -> None:
    """A one-method API or script default must not silently disable consensus."""
    from wisdom.attribution.captum_backend import ATTRS
    from wisdom_pose_train import build_parser

    torch.manual_seed(23)
    model = TinyPoseModel().eval()
    loader = DataLoader(torch.randn(2, 3, 8, 8), batch_size=2, shuffle=False)
    methods = None
    if method_source == "script_default":
        methods = build_parser().parse_args([
            "--model-path", "pose.pth", "--pose-topology", "human_pose.json",
            "--img-dir", "images",
        ]).methods

    # Request optional per-method artifacts; only methods actually executed
    # produce files. This checks real Captum/trainer work, not parser text.
    train_wisdom_pose(
        model, loader, str(tmp_path / "consensus.csv"),
        output_layer_names=("cmap_head", "paf_head"),
        top_m=1, methods=methods, device="cpu",
        method_out_csvs={name: str(tmp_path / f"{name}_method.csv") for name in ATTRS},
    )

    artifacts = list(tmp_path.glob("*_method.csv"))
    assert len(artifacts) >= 2, "Default pretraining must execute at least two methods"
    for path in [tmp_path / "consensus.csv", *artifacts]:
        scores = read_layer_scores_csv(path)
        assert scores
        assert all(torch.isfinite(values).all() for values in scores.values())
