from __future__ import annotations

import subprocess
import sys

import pytest

import wisdom_classification_train
import wisdom_yolo_train


@pytest.mark.parametrize(
    "module_name",
    [
        "wisdom_classification_train",
        "wisdom_yolo_train",
        "wisdom_pose_train",
    ],
)
def test_task_script_help_is_import_safe(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--top-m" in result.stdout
    assert "--selection-mode" in result.stdout


@pytest.mark.parametrize(
    ("build_parser", "required_args"),
    [
        (
            wisdom_classification_train.build_parser,
            ["--model-path", "model.pth", "--imagefolder-root", "images"],
        ),
        (wisdom_yolo_train.build_parser, []),
    ],
)
def test_existing_task_parsers_share_selection_controls(
    build_parser,
    required_args: list[str],
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            *required_args,
            "--selection-mode",
            "per-group",
            "--num-groups",
            "2",
            "--num-layers",
            "4",
        ]
    )
    assert args.selection_mode == "per-group"
    assert args.num_groups == 2
    assert args.num_layers == 4
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--per-group" not in option_strings
    assert "--top-m-neurons" not in option_strings


def test_classification_parser_exposes_explicit_checkpoint_format() -> None:
    parser = wisdom_classification_train.build_parser()
    args = parser.parse_args(
        [
            "--model-path",
            "model.pth",
            "--imagefolder-root",
            "images",
            "--checkpoint-format",
            "state-dict",
            "--model-factory",
            "some_package:make_model",
        ]
    )
    assert args.checkpoint_format == "state-dict"
    assert args.model_factory == "some_package:make_model"


def test_classification_parser_accepts_custom_normalization_statistics() -> None:
    args = wisdom_classification_train.build_parser().parse_args(
        [
            "--model-path",
            "model.pth",
            "--imagefolder-root",
            "images",
            "--normalize",
            "custom",
            "--normalize-mean",
            "0.1,0.2,0.3",
            "--normalize-std",
            "0.4,0.5,0.6",
        ]
    )

    assert args.normalize_mean == (0.1, 0.2, 0.3)
    assert args.normalize_std == (0.4, 0.5, 0.6)


def test_classification_custom_normalization_requires_imagefolder_source() -> None:
    """Named torchvision transforms must not silently ignore custom statistics."""
    with pytest.raises(ValueError, match="imagefolder-root"):
        wisdom_classification_train.main(
            [
                "--model-path",
                "missing.pth",
                "--dataset",
                "cifar10",
                "--normalize",
                "custom",
                "--normalize-mean",
                "0.1,0.2,0.3",
                "--normalize-std",
                "0.4,0.5,0.6",
            ]
        )


def test_classification_imagefolder_loader_keeps_private_compatibility_alias() -> None:
    assert (
        wisdom_classification_train._build_imagefolder_loader
        is wisdom_classification_train.build_classification_loader
    )


def test_pose_parser_has_explicit_reconstruction_inputs() -> None:
    import wisdom_pose_train

    args = wisdom_pose_train.build_parser().parse_args(
        [
            "--model-path",
            "pose.pth",
            "--pose-topology",
            "human_pose.json",
            "--img-dir",
            "images",
            "--pose-architecture",
            "resnet18_baseline_att",
        ]
    )
    assert args.checkpoint_format == "state-dict"
    assert args.pose_architecture == "resnet18_baseline_att"
