"""Standalone, local and CPU-only acceptance coverage for ``run_wisdom``."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from tests.helpers import TinyClassifier
from wisdom.tasks.pose import build_trt_pose_model


REPOSITORY = Path(__file__).parents[1]

# This architecture is written to a temporary local YAML before use. It
# exercises actual Ultralytics reconstruction without a .pt artifact.
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


@pytest.fixture(autouse=True)
def deterministic_cpu(monkeypatch: pytest.MonkeyPatch):
    """Keep fixture construction deterministic; subprocesses receive the same seed."""
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(20260902)
    np.random.seed(20260902)
    torch.manual_seed(20260902)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    yield
    torch.set_num_threads(previous_threads)
    torch.use_deterministic_algorithms(
        previous_deterministic,
        warn_only=previous_warn_only,
    )


def _run(command: list[str], *, inference: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"})
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"run_wisdom failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    if not inference:
        return completed
    assert "Task:" in completed.stdout
    assert "Coverage:" in completed.stdout
    assert '"source":' in completed.stdout
    return completed


def _write_rgb(path: Path, value: int, *, size: int = 32) -> None:
    pixels = np.full((size, size, 3), value, dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def _write_imagefolder(root: Path, *, offset: int) -> None:
    for class_index in range(3):
        class_dir = root / f"class-{class_index}"
        class_dir.mkdir(parents=True)
        for image_index in range(2):
            _write_rgb(
                class_dir / f"sample-{image_index}.png",
                offset + class_index * 60 + image_index * 11,
                size=8,
            )


def _write_detection_split(root: Path, *, offset: int) -> None:
    images, labels = root / "images", root / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    for index in range(3):
        _write_rgb(images / f"sample-{index}.png", offset + index * 20)
        # A complete local YOLO label is required for the honest P/R/F1 path.
        (labels / f"sample-{index}.txt").write_text(
            f"{index % 2} 0.5 0.5 0.45 0.45\n", encoding="utf-8"
        )


def _write_pose_images(root: Path, *, offset: int) -> None:
    root.mkdir(parents=True)
    for index in range(2):
        _write_rgb(root / f"pose-{index}.png", offset + index * 30)


def _assert_finite_coverage(summary: dict[str, object]) -> None:
    coverage = summary["coverage"]
    assert isinstance(coverage, dict)
    for key in ("coverage_rate", "max_coverage"):
        assert math.isfinite(float(coverage[key]))
    assert 0.0 <= float(coverage["coverage_rate"]) <= float(coverage["max_coverage"]) <= 1.0
    assert int(coverage["total_combinations"]) >= 1
    assert coverage["scope_details"]


def _load_summary(path: Path) -> dict[str, object]:
    assert path.is_file()
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["output_json"] == str(path.resolve())
    _assert_finite_coverage(summary)
    return summary


def _score_layer_names(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["LayerName"] for row in rows)
    assert all(int(row["NeuronIndex"]) >= 0 for row in rows)
    assert all(math.isfinite(float(row["Score"])) for row in rows)
    assert any(float(row["Score"]) > 0.0 for row in rows)
    return {row["LayerName"] for row in rows}


def test_classification_runner_generates_reuses_scores_and_runs_sklearn_bo(
    tmp_path: Path,
) -> None:
    """Catch regressions in explicit task selection and score artifact lifecycle."""
    checkpoint = tmp_path / "tiny-classifier-state-dict.pth"
    torch.save(TinyClassifier().state_dict(), checkpoint)
    build_root, test_root = tmp_path / "build", tmp_path / "test"
    _write_imagefolder(build_root, offset=10)
    _write_imagefolder(test_root, offset=50)
    score_path, result_path = tmp_path / "scores.csv", tmp_path / "result.json"
    command = [
        sys.executable, "-m", "run_wisdom",
        "--mode", "wisdom", "--task", "classification",
        "--weights-path", str(checkpoint), "--checkpoint-format", "state-dict",
        "--model-factory", "tests.helpers:TinyClassifier",
        "--build-data-path", str(build_root),
        "--test-data-path", str(test_root),
        "--wisdom-csv", str(score_path), "--output-json", str(result_path),
        "--methods", "la", "--top-m-neurons", "1", "--num-layers", "1",
        "--cluster-method", "KMeans", "--n-clusters", "2",
        "--bo", "--bo-backend", "sklearn", "--bo-init", "1", "--bo-iter", "0",
        "--bo-cluster-methods", "KMeans", "--bo-n-clusters", "2",
        "--batch-size", "2", "--image-size", "8", "--device", "cpu",
    ]
    _run(command)
    generated = _load_summary(result_path)
    assert score_path.is_file() and score_path.stat().st_size > 0
    assert _score_layer_names(score_path)
    assert generated["task"] == "classification"
    assert generated["mode"] == "wisdom"
    assert generated["score_csv"] == {"path": str(score_path), "status": "generated"}
    assert generated["data"]["build_samples"] == 5
    assert generated["data"]["validation_samples"] == 1
    assert generated["data"]["validation_path"] is None
    assert generated["data"]["validation_source"] == "build_holdout"
    assert generated["data"]["validation_fraction"] == pytest.approx(1 / 6)
    assert generated["data"]["test_samples"] == 6
    assert generated["selection"]["selected_layers"] == 1
    assert generated["selection"]["selected_neurons"] == 1
    metrics = generated["metrics"]
    assert all(math.isfinite(float(metrics[key])) for key in ("accuracy", "loss", "f1"))
    assert 0.0 <= float(metrics["accuracy"]) <= 1.0
    assert float(metrics["loss"]) > 0.0
    configuration = generated["cluster_configuration"]
    assert configuration["source"] == "bo"
    assert configuration["backend"] == "sklearn"
    assert configuration["best_cluster_method"] == "KMeans"
    assert configuration["best_n_clusters"] == 2
    bo_path = Path(str(configuration["history_json"]))
    history = json.loads(bo_path.read_text(encoding="utf-8"))
    assert history["backend_used"] == "sklearn"
    assert len(history["history"]) == 1

    _run(command)
    reused = _load_summary(result_path)
    assert reused["score_csv"] == {"path": str(score_path), "status": "reused"}

    direct_csv = tmp_path / "direct-classification.csv"
    _run([
        sys.executable, "-m", "wisdom_classification_train",
        "--model-path", str(checkpoint), "--checkpoint-format", "state-dict",
        "--model-factory", "tests.helpers:TinyClassifier",
        "--imagefolder-root", str(build_root), "--image-size", "8", "--normalize", "none",
        "--methods", "la", "--top-m", "1", "--num-layers", "1",
        "--out-csv", str(direct_csv), "--device", "cpu",
    ], inference=False)
    assert _score_layer_names(direct_csv)


@pytest.mark.ultralytics
def test_detection_runner_uses_local_yolo_state_dict_and_labeled_metrics(tmp_path: Path) -> None:
    """Catch wrapper-prefix/head-filter and per-group runner regressions."""
    pytest.importorskip("ultralytics")
    import yaml
    from ultralytics.nn.tasks import DetectionModel

    model_path, checkpoint = tmp_path / "tiny-yolo.yaml", tmp_path / "tiny-yolo-state-dict.pth"
    model_path.write_text(yaml.safe_dump(MINIMAL_YOLO), encoding="utf-8")
    source = DetectionModel(str(model_path), ch=3, nc=2, verbose=False).eval()
    torch.save(source.state_dict(), checkpoint)
    build_root, test_root = tmp_path / "build", tmp_path / "test"
    _write_detection_split(build_root, offset=20)
    _write_detection_split(test_root, offset=80)
    score_path, result_path = tmp_path / "scores.csv", tmp_path / "result.json"
    _run([
        sys.executable, "-m", "run_wisdom",
        "--mode", "wisdom", "--task", "detection",
        "--model-path", str(model_path), "--weights-path", str(checkpoint),
        "--checkpoint-format", "state-dict",
        "--build-data-path", str(build_root / "images"),
        "--test-data-path", str(test_root / "images"),
        "--wisdom-csv", str(score_path), "--output-json", str(result_path),
        "--methods", "la", "--selection-mode", "per-group",
        "--top-m-neurons", "1", "--num-groups", "2", "--num-layers", "2",
        "--cluster-method", "KMeans", "--n-clusters", "2",
        "--batch-size", "2", "--imgsz", "32", "--device", "cpu",
    ])
    summary = _load_summary(result_path)
    assert summary["task"] == "detection"
    assert summary["model"]["name"] == "DetectionModel"
    assert summary["model"]["model_path"] == str(model_path.resolve())
    assert summary["score_csv"]["status"] == "generated"
    layer_names = _score_layer_names(score_path)
    assert layer_names and all(name.startswith("yolo_model.") for name in layer_names)
    assert not any(name.startswith("yolo_model.model.2.") for name in layer_names)
    assert summary["selection"] == {
        "mode": "per-group", "selected_layers": 2, "selected_neurons": 2,
        "num_groups": 2, "num_layers": 2,
    }
    metrics = summary["metrics"]
    assert all(math.isfinite(float(metrics[key])) and 0.0 <= float(metrics[key]) <= 1.0 for key in ("precision", "recall", "f1"))

    direct_csv = tmp_path / "direct-detection.csv"
    _run([
        sys.executable, "-m", "wisdom_yolo_train",
        "--weights", str(model_path), "--img-dir", str(build_root / "images"),
        "--imgsz", "32", "--num-images", "3", "--methods", "la", "--top-m", "1",
        "--selection-mode", "per-group", "--num-groups", "2", "--num-layers", "2",
        "--out-csv", str(direct_csv), "--device", "cpu",
    ], inference=False)
    assert _score_layer_names(direct_csv)


@pytest.mark.local_trt_pose
@pytest.mark.slow
def test_pose_runner_uses_packaged_resnet_and_image_only_surrogate(tmp_path: Path) -> None:
    """Catch accidental sibling-trt_pose imports and mislabeled image-only metrics."""
    topology_path = tmp_path / "topology.json"
    topology = {
        "keypoints": [f"part-{index}" for index in range(18)],
        "skeleton": [[(index % 18) + 1, ((index + 1) % 18) + 1] for index in range(21)],
    }
    topology_path.write_text(json.dumps(topology), encoding="utf-8")
    assert (len(topology["keypoints"]), len(topology["skeleton"])) == (18, 21)
    source = build_trt_pose_model(
        architecture="resnet18_baseline_att", topology_path=topology_path
    ).eval()
    assert len(source.state_dict()) == 172
    cmap, paf = source(torch.zeros(1, 3, 32, 32))
    assert cmap.shape == (1, 18, 8, 8)
    assert paf.shape == (1, 42, 8, 8)
    checkpoint = tmp_path / "pose-state-dict.pth"
    torch.save(source.state_dict(), checkpoint)
    build_root, test_root = tmp_path / "build", tmp_path / "test"
    _write_pose_images(build_root, offset=35)
    _write_pose_images(test_root, offset=95)
    score_path, result_path = tmp_path / "scores.csv", tmp_path / "result.json"
    _run([
        sys.executable, "-m", "run_wisdom",
        "--mode", "wisdom", "--task", "pose",
        "--weights-path", str(checkpoint), "--checkpoint-format", "state-dict",
        "--pose-topology", str(topology_path),
        "--pose-architecture", "resnet18_baseline_att",
        "--build-data-path", str(build_root), "--test-data-path", str(test_root),
        "--wisdom-csv", str(score_path), "--output-json", str(result_path),
        "--methods", "la", "--top-m-neurons", "1", "--num-layers", "1",
        "--cluster-method", "KMeans", "--n-clusters", "2",
        "--batch-size", "2", "--image-size", "32", "--device", "cpu",
    ])
    summary = _load_summary(result_path)
    assert summary["task"] == "pose"
    assert summary["model"]["name"] == "resnet18_baseline_att"
    assert summary["model"]["model_path"] is None
    assert summary["score_csv"]["status"] == "generated"
    assert score_path.is_file() and score_path.stat().st_size > 0
    layer_names = _score_layer_names(score_path)
    assert layer_names
    assert not any(name.endswith(("cmap_conv", "paf_conv")) for name in layer_names)
    assert summary["selection"]["selected_layers"] >= 1
    assert summary["selection"]["selected_neurons"] >= 1
    metrics = summary["metrics"]
    assert metrics["loss"] is None and metrics["pck"] is None
    assert math.isfinite(float(metrics["pose_confidence_surrogate"]))
    assert 0.0 <= float(metrics["pose_confidence_surrogate"]) <= 1.0

    direct_csv = tmp_path / "direct-pose.csv"
    _run([
        sys.executable, "-m", "wisdom_pose_train",
        "--model-path", str(checkpoint), "--checkpoint-format", "state-dict",
        "--pose-topology", str(topology_path), "--pose-architecture", "resnet18_baseline_att",
        "--img-dir", str(build_root), "--image-size", "32", "--methods", "lgxa",
        "--top-m", "1", "--num-layers", "1",
        "--out-csv", str(direct_csv), "--device", "cpu",
    ], inference=False)
    assert _score_layer_names(direct_csv)
