from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from tests.helpers import TinyClassifier, TinyPoseModel
from wisdom_classification_train import (
    build_classification_loader,
    evaluate_classification,
    prepare_classification_inference,
)
from wisdom_yolo_train import collate_detection_batch


class CustomHeadsPoseModel(TinyPoseModel):
    def __init__(self):
        super().__init__()
        self.confidences = self.cmap_head
        self.affinities = self.paf_head
        del self.cmap_head, self.paf_head

    def forward(self, inputs):
        features = self.backbone(inputs)
        return self.confidences(features), self.affinities(features)


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


def test_classification_metrics_are_honest_and_bo_uses_weighted_f1() -> None:
    model = TinyClassifier().eval()
    images = torch.randn(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 1])

    result = evaluate_classification(
        model,
        DataLoader(TensorDataset(images, labels), batch_size=2),
        "cpu",
    )

    assert set(result.metrics) == {"accuracy", "loss", "f1"}
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.metrics["loss"] >= 0.0
    assert 0.0 <= result.metrics["f1"] <= 1.0
    assert result.bo_metric_name == "f1"
    assert result.bo_metric_value == result.metrics["f1"]


def test_classification_metrics_use_per_sample_cross_entropy_and_weighted_f1() -> None:
    logits = torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]
    )
    labels = torch.tensor([0, 1, 2, 1])

    result = evaluate_classification(
        torch.nn.Identity(),
        DataLoader(TensorDataset(logits, labels), batch_size=2),
        "cpu",
    )

    assert result.metrics == pytest.approx(
        {"accuracy": 0.75, "loss": 1.0359763, "f1": 2 / 3}
    )


def _write_imagefolder(root: Path, offset: int, *, samples_per_class: int = 1) -> None:
    for class_index in range(2):
        class_dir = root / f"class-{class_index}"
        class_dir.mkdir(parents=True)
        for sample_index in range(samples_per_class):
            Image.fromarray(
                np.full(
                    (8, 8, 3),
                    offset + class_index * 40 + sample_index,
                    np.uint8,
                )
            ).save(class_dir / f"sample-{sample_index}.png")


def _write_rgb_imagefolder(root: Path, rgb: tuple[int, int, int]) -> None:
    pixels = np.empty((8, 8, 3), dtype=np.uint8)
    pixels[:] = rgb
    for class_index in range(2):
        class_dir = root / f"class-{class_index}"
        class_dir.mkdir(parents=True)
        for sample_index in range(2):
            Image.fromarray(pixels).save(class_dir / f"sample-{sample_index}.png")


def _classification_args(checkpoint: Path, paths: list[Path]) -> Namespace:
    return Namespace(
        weights_path=str(checkpoint),
        checkpoint_format="state-dict",
        model_factory="tests.helpers:TinyClassifier",
        model_path=None,
        build_data_path=str(paths[0]),
        validation_data_path=str(paths[1]),
        test_data_path=str(paths[2]),
        batch_size=2,
        image_size=8,
        grayscale=False,
        normalize="none",
        device="cpu",
        methods=["la"],
        voting_mode="fine-grained",
        selection_mode="global",
        num_groups=3,
        num_layers=1,
        top_m_neurons=1,
        trainer_checkpoint=None,
        checkpoint_every=50,
        num_workers=0,
    )


def test_classification_preparation_keeps_build_validation_and_test_separate(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ("build", "validation", "test")]
    for offset, path in enumerate(paths):
        _write_imagefolder(path, offset)
    checkpoint = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), checkpoint)

    args = _classification_args(checkpoint, paths)
    args.bo = True
    args.seed = 11
    prepared = prepare_classification_inference(args)

    assert prepared.task == "classification"
    assert prepared.build_loader.dataset.root == str(paths[0])
    assert prepared.validation_loader is not None
    assert prepared.validation_loader.dataset.root == str(paths[1])
    assert prepared.test_loader.dataset.root == str(paths[2])


def test_classification_bo_reserves_holdout_before_score_pretraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using all build samples for scores after reserving validation is leakage."""
    build_path, test_path = tmp_path / "build", tmp_path / "test"
    _write_imagefolder(build_path, 0, samples_per_class=10)
    _write_imagefolder(test_path, 100, samples_per_class=2)
    checkpoint = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), checkpoint)
    args = _classification_args(checkpoint, [build_path, build_path, test_path])
    args.validation_data_path = None
    args.bo = True
    args.seed = 19
    seen_loaders: list[DataLoader] = []

    monkeypatch.setattr(
        "wisdom_classification_train.train_wisdom_classification",
        lambda **kwargs: seen_loaders.append(kwargs["train_loader"])
        or kwargs["out_csv"],
    )

    prepared = prepare_classification_inference(args)
    prepared.score_trainer(str(tmp_path / "scores.csv"))

    assert len(prepared.build_loader.dataset) == 18
    assert prepared.validation_loader is not None
    assert len(prepared.validation_loader.dataset) == 2
    assert seen_loaders == [prepared.build_loader]


def test_custom_normalization_reaches_build_bo_holdout_and_test(
    tmp_path: Path,
) -> None:
    """A custom transform omitted from any task loader must fail this test."""
    build_path, test_path = tmp_path / "build", tmp_path / "test"
    for path in (build_path, test_path):
        _write_rgb_imagefolder(path, (51, 102, 153))
    checkpoint = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), checkpoint)
    args = _classification_args(checkpoint, [build_path, build_path, test_path])
    args.validation_data_path = None
    args.bo = True
    args.normalize = "custom"
    args.normalize_mean = (0.1, 0.2, 0.3)
    args.normalize_std = (0.1, 0.2, 0.3)

    prepared = prepare_classification_inference(args)

    assert prepared.validation_loader is not None
    expected = torch.ones(3, 8, 8)
    for loader in (
        prepared.build_loader,
        prepared.validation_loader,
        prepared.test_loader,
    ):
        images, _ = next(iter(loader))
        torch.testing.assert_close(images[0], expected)


@pytest.mark.parametrize(
    ("mode", "mean", "std", "message"),
    [
        ("custom", (0.1, 0.2, 0.3), None, "both.*mean.*std"),
        ("custom", (0.1,), (0.2,), "3 values"),
        ("custom", (0.1, 0.2, 0.3), (0.2, 0.0, 0.2), "positive"),
        ("custom", (0.1, float("inf"), 0.3), (0.2, 0.2, 0.2), "finite"),
        ("imagenet", (0.1, 0.2, 0.3), (0.2, 0.2, 0.2), "only valid"),
    ],
)
def test_custom_normalization_rejects_invalid_statistics(
    tmp_path: Path,
    mode: str,
    mean: tuple[float, ...] | None,
    std: tuple[float, ...] | None,
    message: str,
) -> None:
    _write_rgb_imagefolder(tmp_path, (51, 102, 153))

    with pytest.raises(ValueError, match=message):
        build_classification_loader(
            str(tmp_path),
            batch_size=2,
            image_size=8,
            grayscale=False,
            normalize=mode,
            normalize_mean=mean,
            normalize_std=std,
        )


def test_classification_preparation_rejects_different_class_mappings(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ("build", "validation", "test")]
    for offset, path in enumerate(paths):
        _write_imagefolder(path, offset)
    (paths[2] / "class-1").rename(paths[2] / "different-class")
    checkpoint = tmp_path / "weights.pth"
    torch.save(TinyClassifier().state_dict(), checkpoint)

    with pytest.raises(ValueError, match="class mapping"):
        prepare_classification_inference(_classification_args(checkpoint, paths))


@pytest.mark.ultralytics
def test_detection_metrics_report_f1_only_when_labels_are_available() -> None:
    """Removing label-awareness must make this test fail."""
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel
    from wisdom_yolo_train import evaluate_detection

    model = DetectionModel(MINIMAL_YOLO, ch=3, nc=2, verbose=False).eval()
    images = torch.zeros(2, 3, 32, 32)
    labeled = DataLoader(
        [(images[0], [(0, 0.5, 0.5, 0.25, 0.25)]), (images[1], [])],
        batch_size=2,
        collate_fn=collate_detection_batch,
    )
    result = evaluate_detection(model, labeled, "cpu", image_size=32, num_classes=2)
    assert set(result.metrics) == {"precision", "recall", "f1"}
    assert all(value is not None and 0.0 <= value <= 1.0 for value in result.metrics.values())
    assert result.bo_metric_name == "f1"

    unlabeled = DataLoader(torch.zeros(2, 3, 32, 32), batch_size=2)
    missing = evaluate_detection(model, unlabeled, "cpu", image_size=32, num_classes=2)
    assert missing.metrics == {"precision": None, "recall": None, "f1": None}
    assert missing.bo_metric_name is None and missing.bo_metric_value is None


@pytest.mark.parametrize("same_class, expected", [(False, (2, 2, 2)), (True, (1, 1, 1))])
def test_detection_nms_respects_class_identity(same_class, expected) -> None:
    """Cross-class suppression must not turn two perfect detections into a miss."""
    from wisdom_yolo_train import _match_detection_batch

    predictions = torch.tensor([
        [16., 16.], [16., 16.], [8., 8.], [8., 8.],
        [0.9, 0.8 if same_class else 0.1],
        [0.1, 0.1 if same_class else 0.8],
    ])
    labels = [(0, 0.5, 0.5, 0.25, 0.25)]
    if not same_class:
        labels.append((1, 0.5, 0.5, 0.25, 0.25))
    assert _match_detection_batch(predictions, labels, 32, 2) == expected


def test_pose_pck_does_not_count_masked_out_targets() -> None:
    from wisdom_pose_train import evaluate_pose

    raw = TinyPoseModel().eval()
    images = torch.ones(1, 3, 8, 8)
    with torch.no_grad():
        cmap, paf = raw(images)
    result = evaluate_pose(
        raw,
        DataLoader(TensorDataset(images, cmap, paf, torch.zeros(1, 1, 8, 8))),
        "cpu",
    )
    assert result.metrics["pck"] is None
    assert result.bo_metric_value is None


def test_pose_pck_checks_mask_at_each_target_peak() -> None:
    from wisdom_pose_train import pose_pck

    target = torch.zeros(1, 2, 4, 4)
    target[0, 0, 1, 1] = target[0, 1, 2, 2] = 1
    predicted = torch.zeros_like(target)
    predicted[0, 0, 1, 1] = predicted[0, 1, 0, 0] = 1
    mask = torch.ones(1, 1, 4, 4)
    mask[0, 0, 2, 2] = 0

    assert pose_pck(predicted, target) == 0.5
    assert pose_pck(predicted, target, mask=mask) == 1.0


def test_pose_pck_uses_heatmap_peaks_and_confidence_surrogate_is_bounded() -> None:
    """Fail if pose evaluation treats unlabeled images as supervised accuracy data."""
    from wisdom_pose_train import evaluate_pose, pose_pck

    raw = TinyPoseModel().eval()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        cmap, paf = raw(images)
    assert pose_pck(cmap, cmap) == pytest.approx(1.0)

    supervised = DataLoader(
        TensorDataset(images, cmap, paf, torch.ones(2, 1, 8, 8)),
        batch_size=2,
    )
    result = evaluate_pose(raw, supervised, "cpu")
    assert result.metrics["pck"] == pytest.approx(1.0)
    assert result.metrics["loss"] == pytest.approx(0.0)
    assert result.metrics["pose_confidence_surrogate"] is None
    assert result.bo_metric_name == "pck"

    unlabeled = evaluate_pose(raw, DataLoader(images, batch_size=2), "cpu")
    assert unlabeled.metrics["loss"] is None
    assert unlabeled.metrics["pck"] is None
    assert 0.0 <= unlabeled.metrics["pose_confidence_surrogate"] <= 1.0
    assert unlabeled.bo_metric_name == "pose_confidence_surrogate"
    assert unlabeled.bo_metric_value == unlabeled.metrics["pose_confidence_surrogate"]


def test_pose_evaluation_weights_pck_by_visible_keypoints_across_batches() -> None:
    """Fail if dataset PCK weights a one-keypoint batch like a 100-keypoint batch."""
    from wisdom_pose_train import evaluate_pose

    raw = TinyPoseModel(num_parts=101).eval()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        cmap, paf = raw(images)

    targets = torch.zeros_like(cmap)
    targets[0, 0] = cmap[0, 0]
    for part in range(100):
        prediction_index = int(cmap[1, part].flatten().argmax())
        y, x = divmod(prediction_index, 8)
        targets[1, part, (y + 1) % 8, x] = 1.0
    result = evaluate_pose(
        raw,
        DataLoader(
            TensorDataset(images, targets, paf, torch.ones(2, 1, 8, 8)),
            batch_size=1,
        ),
        "cpu",
    )

    assert result.metrics["pck"] == pytest.approx(1 / 101)


@pytest.mark.parametrize("custom_heads", [False, True])
def test_pose_preparation_reports_explicit_architecture_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    custom_heads: bool,
) -> None:
    """Fail if pose preparation falls back to the generic Sequential class name."""
    import wisdom_pose_train

    image_paths = [tmp_path / name for name in ("build", "validation", "test")]
    for image_path in image_paths:
        image_path.mkdir()
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image_path / "sample.png")

    model = CustomHeadsPoseModel().eval() if custom_heads else TinyPoseModel().eval()
    if not custom_heads:
        model._wisdom_output_layer_names = ("cmap_head", "paf_head")
    monkeypatch.setattr(wisdom_pose_train, "load_pose_model", lambda args: model)
    args = Namespace(
        pose_architecture="mnasnet0_5_baseline_att",
        pose_output_layers=("confidences", "affinities") if custom_heads else None,
        build_data_path=str(image_paths[0]),
        validation_data_path=str(image_paths[1]),
        test_data_path=str(image_paths[2]),
        image_size=8,
        max_images=None,
        batch_size=1,
        num_workers=0,
        device="cpu",
        top_m_neurons=1,
        methods=["la"],
        voting_mode="fine-grained",
        selection_mode="global",
        num_groups=1,
        num_layers=1,
        trainer_checkpoint=None,
        checkpoint_every=1,
        checkpoint_format="state-dict",
        bo=True,
        seed=11,
    )

    prepared = wisdom_pose_train.prepare_pose_inference(args)

    assert prepared.model_name == "mnasnet0_5_baseline_att"
    result = prepared.evaluate(prepared.test_loader)
    assert 0.0 <= result.metrics["pose_confidence_surrogate"] <= 1.0


def test_detection_bo_reserves_default_validation_holdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detection preparation must honor the common no-path BO fallback."""
    import wisdom_yolo_train

    model = TinyClassifier().eval()
    build_source = DataLoader(
        TensorDataset(torch.arange(20).view(20, 1).float()), batch_size=4
    )
    test_source = DataLoader(
        TensorDataset(torch.arange(4).view(4, 1).float()), batch_size=2
    )
    monkeypatch.setattr(wisdom_yolo_train, "load_detection_architecture", lambda *args: model)
    monkeypatch.setattr(wisdom_yolo_train, "infer_num_classes", lambda model: 2)
    monkeypatch.setattr(
        wisdom_yolo_train,
        "_build_detection_loader",
        lambda source, args: build_source if source == "build" else test_source,
    )

    args = Namespace(
        model_path="model.yaml",
        weights_path="weights.pth",
        checkpoint_format="state-dict",
        build_data_path="build",
        validation_data_path=None,
        test_data_path="test",
        batch_size=4,
        imgsz=32,
        device="cpu",
        top_m_neurons=1,
        methods=["la"],
        voting_mode="fine-grained",
        selection_mode="global",
        num_groups=1,
        num_layers=1,
        trainer_checkpoint=None,
        checkpoint_every=1,
        num_workers=0,
        bo=True,
        seed=7,
    )

    prepared = wisdom_yolo_train.prepare_detection_inference(args)

    assert len(prepared.build_loader.dataset) == 18
    assert prepared.validation_loader is not None
    assert len(prepared.validation_loader.dataset) == 2


def test_pose_bo_reserves_default_validation_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pose preparation must honor the common no-path BO fallback."""
    import wisdom_pose_train

    build_path, test_path = tmp_path / "build", tmp_path / "test"
    build_path.mkdir()
    test_path.mkdir()
    for index in range(20):
        Image.fromarray(np.full((8, 8, 3), index, np.uint8)).save(
            build_path / f"sample-{index}.png"
        )
    for index in range(2):
        Image.fromarray(np.full((8, 8, 3), index, np.uint8)).save(
            test_path / f"sample-{index}.png"
        )

    model = TinyPoseModel().eval()
    model._wisdom_output_layer_names = ("cmap_head", "paf_head")
    monkeypatch.setattr(wisdom_pose_train, "load_pose_model", lambda args: model)
    args = Namespace(
        pose_architecture="resnet18_baseline_att",
        pose_output_layers=None,
        build_data_path=str(build_path),
        validation_data_path=None,
        test_data_path=str(test_path),
        image_size=8,
        max_images=None,
        batch_size=4,
        num_workers=0,
        device="cpu",
        top_m_neurons=1,
        methods=["la"],
        voting_mode="fine-grained",
        selection_mode="global",
        num_groups=1,
        num_layers=1,
        trainer_checkpoint=None,
        checkpoint_every=1,
        checkpoint_format="state-dict",
        bo=True,
        seed=7,
    )

    prepared = wisdom_pose_train.prepare_pose_inference(args)

    assert len(prepared.build_loader.dataset) == 18
    assert prepared.validation_loader is not None
    assert len(prepared.validation_loader.dataset) == 2
