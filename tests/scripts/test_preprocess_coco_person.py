"""The bundled NVIDIA converter must produce the topology our pose model uses."""

import json
from pathlib import Path
import subprocess
import sys


def test_coco_person_converter_adds_neck_without_changing_source(tmp_path):
    root = Path(__file__).resolve().parents[2]
    keypoints = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
        [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
        [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],
        [4, 6], [5, 7],
    ]
    annotations = []
    for index, (left_visibility, right_visibility) in enumerate([(2, 2), (1, 2), (0, 2)]):
        points = [0, 0, 0] * 17
        points[15:21] = [10, 20, left_visibility, 30, 40, right_visibility]
        annotations.append({"id": index, "image_id": 1, "category_id": 1, "keypoints": points})
    payload = {
        "images": [{"id": 1, "file_name": "person.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person", "keypoints": keypoints, "skeleton": skeleton}],
        "annotations": annotations,
    }
    source = tmp_path / "coco.json"
    destination = tmp_path / "coco_trt_pose.json"
    source.write_text(json.dumps(payload))
    original = source.read_bytes()

    result = subprocess.run(
        [sys.executable, str(root / "datasets/preprocess_coco_person.py"), str(source), str(destination)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == original
    converted = json.loads(destination.read_text())
    assert converted["images"] == payload["images"]
    assert [ann["keypoints"][-3:] for ann in converted["annotations"]] == [
        [20, 30, 2], [20, 30, 1], [20, 30, 0],
    ]
    assert all(len(ann["keypoints"]) == 54 for ann in converted["annotations"])
    category = converted["categories"][0]
    assert category["keypoints"] == keypoints + ["neck"]
    assert len(category["skeleton"]) == 21
    assert [6, 7] not in category["skeleton"]
    assert category["skeleton"][-5:] == [[18, 1], [18, 6], [18, 7], [18, 12], [18, 13]]
    topology = json.loads((root / "datasets/human_pose.json").read_text())
    assert category["keypoints"] == topology["keypoints"]
    assert category["skeleton"] == topology["skeleton"]
