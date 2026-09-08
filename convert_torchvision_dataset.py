#!/usr/bin/env python
"""Convert local torchvision classification datasets to ImageFolder layout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import datasets


"""
python convert_torchvision_dataset.py \
    --dataset cifar10 \
    --data-root /shared/storage/cs/scratch/lrr550/datasets/ \
    --output-root /shared/storage/cs/scratch/lrr550/datasets/cifar-10-imagefolder \
    --validation-fraction 0.10 \
    --seed 42
"""

DATASET_FACTORIES = {
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "mnist": datasets.MNIST,
}


@dataclass(frozen=True)
class ConversionResult:
    destination: Path
    counts: dict[str, int]


def load_torchvision_splits(dataset_name: str, data_root: str | Path) -> tuple[Any, Any]:
    """Load an existing official train/test split without network access."""
    name = dataset_name.lower()
    try:
        factory = DATASET_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset_name}") from exc
    try:
        train_dataset = factory(root=str(data_root), train=True, download=False)
        test_dataset = factory(root=str(data_root), train=False, download=False)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"Dataset '{name}' is not available below '{data_root}'. "
            "Download it with torchvision before running this offline converter."
        ) from exc
    return train_dataset, test_dataset


def _class_directory(index: int, class_name: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(class_name)).strip("._-")
    return f"{index:03d}_{slug or 'class'}"


def _targets(dataset: Any) -> list[int]:
    raw_targets = getattr(dataset, "targets", None)
    if raw_targets is None:
        raise TypeError("A torchvision-style dataset with a 'targets' attribute is required.")
    if hasattr(raw_targets, "tolist"):
        raw_targets = raw_targets.tolist()
    targets = [int(target) for target in raw_targets]
    if len(targets) != len(dataset):
        raise ValueError("Dataset targets and samples have different lengths.")
    return targets


def _write_split(
    dataset: Any,
    indices: list[int],
    split_root: Path,
    class_directories: tuple[str, ...],
) -> int:
    for class_directory in class_directories:
        (split_root / class_directory).mkdir(parents=True, exist_ok=True)
    for index in indices:
        image, raw_target = dataset[index]
        target = int(raw_target)
        if not 0 <= target < len(class_directories):
            raise ValueError(f"Sample {index} has out-of-range class target {target}.")
        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Sample {index} returned {type(image).__name__}; expected a PIL image."
            )
        output = split_root / class_directories[target] / f"{index:06d}.png"
        image.save(output, format="PNG")
    return len(indices)


def export_imagefolder(
    *,
    dataset_name: str,
    train_dataset: Any,
    test_dataset: Any,
    destination: str | Path,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> ConversionResult:
    """Export disjoint build/validation/test splits without touching source data."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")

    requested_destination = Path(destination).expanduser()
    destination_path = requested_destination.parent.resolve() / requested_destination.name
    if os.path.lexists(destination_path):
        raise FileExistsError(f"Destination already exists: {destination_path}")

    classes = tuple(getattr(train_dataset, "classes", ()))
    if not classes:
        raise TypeError("A torchvision-style dataset with nonempty 'classes' is required.")
    if tuple(getattr(test_dataset, "classes", ())) != classes:
        raise ValueError("Training and test datasets expose different class definitions.")

    targets = _targets(train_dataset)
    all_train_indices = list(range(len(train_dataset)))
    try:
        build_indices, validation_indices = train_test_split(
            all_train_indices,
            test_size=validation_fraction,
            random_state=seed,
            shuffle=True,
            stratify=targets,
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create a stratified validation split; every class needs "
            "enough samples for both build and validation data."
        ) from exc

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}-",
            dir=destination_path.parent,
        )
    )
    class_directories = tuple(
        _class_directory(index, class_name) for index, class_name in enumerate(classes)
    )
    try:
        counts = {
            "build": _write_split(
                train_dataset,
                sorted(int(index) for index in build_indices),
                staging_path / "build",
                class_directories,
            ),
            "validation": _write_split(
                train_dataset,
                sorted(int(index) for index in validation_indices),
                staging_path / "validation",
                class_directories,
            ),
            "test": _write_split(
                test_dataset,
                list(range(len(test_dataset))),
                staging_path / "test",
                class_directories,
            ),
        }
        manifest = {
            "dataset": dataset_name.lower(),
            "source_format": "torchvision",
            "output_format": "ImageFolder",
            "validation_fraction": validation_fraction,
            "seed": seed,
            "classes": {
                class_directory: index
                for index, class_directory in enumerate(class_directories)
            },
            "counts": counts,
        }
        (staging_path / "conversion.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        if os.path.lexists(destination_path):
            raise FileExistsError(f"Destination already exists: {destination_path}")
        staging_path.rename(destination_path)
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    return ConversionResult(destination=destination_path, counts=counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an existing torchvision dataset to WISDOM ImageFolder splits"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FACTORIES))
    parser.add_argument(
        "--data-root",
        required=True,
        help="Parent containing the existing torchvision dataset; no download is attempted",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Destination (default: <data-root>/<dataset>-imagefolder)",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_root = Path(args.data_root).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else data_root / f"{args.dataset}-imagefolder"
    )
    train_dataset, test_dataset = load_torchvision_splits(args.dataset, data_root)
    result = export_imagefolder(
        dataset_name=args.dataset,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        destination=output_root,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        json.dumps(
            {"destination": str(result.destination), "counts": result.counts},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
