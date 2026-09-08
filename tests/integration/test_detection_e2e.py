from __future__ import annotations

import pytest
import torch
from argparse import Namespace
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from wisdom.core.wisdom import ClusteringConfig, WisdomConfig, WisdomIDC
from wisdom.core.task import TaskEvaluation
from wisdom.core.wisdom_train import train_wisdom_yolo
from wisdom.tasks.detection import DetectionAdapter
from wisdom.utils.io_cache import read_layer_scores_csv
from wisdom_yolo_train import DetectionInferenceDataset, prepare_detection_inference


MINIMAL_YOLO = {
    "nc": 2,
    "depth_multiple": 1.0,
    "width_multiple": 1.0,
    "backbone": [
        [-1, 1, "Conv", [8, 3, 2]],
        [-1, 1, "Conv", [16, 3, 2]],
    ],
    "head": [[[1], 1, "Detect", [2]]],
}


@pytest.mark.ultralytics
def test_actual_detection_model_pretraining_and_coverage(tmp_path) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    torch.manual_seed(29)
    raw_model = DetectionModel(
        MINIMAL_YOLO,
        ch=3,
        nc=2,
        verbose=False,
    ).eval()
    images = torch.randn(4, 3, 32, 32)
    loader = DataLoader(images, batch_size=4, shuffle=False)

    csv_path = train_wisdom_yolo(
        model=raw_model,
        train_loader=loader,
        out_csv=str(tmp_path / "detection.csv"),
        top_m=1,
        methods=["la"],
        device="cpu",
    )

    scores = read_layer_scores_csv(csv_path)
    adapter = DetectionAdapter(raw_model, num_classes=2)
    assert scores
    assert all(
        not name.startswith(adapter.excluded_layer_prefixes())
        for name in scores
    )
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


def _write_yolo_images(root: Path, *, labels: str) -> None:
    image_dir = root / "images"
    label_dir = root / "labels"
    image_dir.mkdir(parents=True)
    if labels != "none":
        label_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (12, 8), color=(index, 0, 0)).save(image_dir / f"sample-{index}.png")
        if labels == "all" or (labels == "partial" and index == 0):
            (label_dir / f"sample-{index}.txt").write_text("1 0.5 0.5 0.25 0.5\n")


def test_detection_dataset_distinguishes_complete_partial_and_absent_labels(tmp_path: Path) -> None:
    """Silently treating partial labels as unlabeled must fail this test."""
    complete = tmp_path / "complete"
    absent = tmp_path / "absent"
    partial = tmp_path / "partial"
    _write_yolo_images(complete, labels="all")
    _write_yolo_images(absent, labels="none")
    _write_yolo_images(partial, labels="partial")

    assert len(DetectionInferenceDataset(str(complete / "images"), 16)[0]) == 2
    assert len(DetectionInferenceDataset(str(absent / "images"), 16)[0]) == 1
    with pytest.raises(ValueError, match="partial label"):
        DetectionInferenceDataset(str(partial / "images"), 16)


@pytest.mark.ultralytics
def test_detection_preparation_keeps_build_validation_and_test_separate(tmp_path: Path, monkeypatch) -> None:
    """Sharing a loader or dropping the wrapper prefix must fail this test."""
    pytest.importorskip("ultralytics")
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    paths = [tmp_path / name for name in ("build", "validation", "test")]
    for path in paths:
        _write_yolo_images(path, labels="all")
    model_yaml = tmp_path / "local-yolo.yaml"
    model_yaml.write_text(yaml.safe_dump(MINIMAL_YOLO), encoding="utf-8")
    weights = tmp_path / "weights.pth"
    torch.save(DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False).state_dict(), weights)
    args = Namespace(
        model_path=str(model_yaml), weights_path=str(weights), checkpoint_format="state-dict",
        build_data_path=str(paths[0] / "images"), validation_data_path=str(paths[1] / "images"),
        test_data_path=str(paths[2] / "images"), batch_size=2, image_size=16, imgsz=32, device="cpu",
        top_m_neurons=1, methods=["la"], voting_mode="fine-grained", selection_mode="per-group",
        num_groups=2, num_layers=None, trainer_checkpoint=None, checkpoint_every=50, num_workers=0,
        bo=True, seed=11,
    )

    prepared = prepare_detection_inference(args)

    assert prepared.build_loader.dataset.paths[0].is_relative_to(paths[0])
    assert prepared.validation_loader is not None
    assert prepared.validation_loader.dataset.paths[0].is_relative_to(paths[1])
    assert prepared.test_loader.dataset.paths[0].is_relative_to(paths[2])
    assert any(name.startswith("yolo_model.model.") for name, _ in prepared.adapter.analysis_model.named_modules())
    assert prepared.build_loader.dataset[0][0].shape == (3, 32, 32)

    seen = []
    monkeypatch.setattr(
        "wisdom_yolo_train.evaluate_detection",
        lambda model, loader, device, image_size, num_classes: seen.append(image_size)
        or TaskEvaluation({}, None, None),
    )
    prepared.evaluate(prepared.test_loader)
    assert seen == [32]
