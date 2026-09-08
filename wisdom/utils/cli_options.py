"""Shared argparse controls used by WISDOM's small task entry points."""

from __future__ import annotations

import argparse
import math


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def comma_separated_floats(value: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated sequence of finite floats."""
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        )
    try:
        parsed = tuple(float(item) for item in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        ) from exc
    if any(not math.isfinite(item) for item in parsed):
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of finite numbers"
        )
    return parsed


def add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--selection-mode",
        default="global",
        choices=["global", "per-group", "per-layer"],
        help=(
            "Scope for top-M: globally, independently in each dynamic layer "
            "group, or independently in each considered layer"
        ),
    )
    parser.add_argument(
        "--num-groups",
        type=positive_int,
        default=3,
        help="Number of dynamic groups used by per-group selection",
    )
    parser.add_argument(
        "--num-layers",
        type=positive_int,
        default=None,
        help="Number of evenly distributed eligible layers to consider (default: all)",
    )
