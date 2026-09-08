"""Task-specific adapters and model preparation helpers.

These exports do not import Ultralytics or trt_pose itself; those optional
dependencies are resolved only by their model-loading functions.
"""

from wisdom.tasks.classification import ClassificationAdapter
from wisdom.tasks.detection import DetectionAdapter
from wisdom.tasks.pose import (
    PoseAdapter,
    PoseAttributionWrapper,
    PoseTopology,
    build_trt_pose_model,
    load_pose_topology,
    load_trt_pose_checkpoint,
)

__all__ = [
    "ClassificationAdapter",
    "DetectionAdapter",
    "PoseAdapter",
    "PoseAttributionWrapper",
    "PoseTopology",
    "build_trt_pose_model",
    "load_pose_topology",
    "load_trt_pose_checkpoint",
]
