from __future__ import annotations

import pytest

import wisdom_yolo_train


def test_yolo_script_reuses_selection_mode_and_exposes_group_and_layer_counts() -> None:
    parser = wisdom_yolo_train.build_parser()

    args = parser.parse_args(
        [
            "--selection-mode",
            "per-group",
            "--num-groups",
            "4",
            "--num-layers",
            "8",
        ]
    )

    assert args.selection_mode == "per-group"
    assert args.num_groups == 4
    assert args.num_layers == 8

    with pytest.raises(SystemExit):
        parser.parse_args(["--per-group"])


@pytest.mark.parametrize(
    ("option", "value"),
    [("--num-groups", "0"), ("--num-layers", "0")],
)
def test_yolo_script_rejects_nonpositive_selection_counts(
    option: str,
    value: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = wisdom_yolo_train.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([option, value])
    assert "must be positive" in capsys.readouterr().err
