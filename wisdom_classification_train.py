#!/usr/bin/env python
"""Thin classification orchestration for WISDOM pretraining."""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from wisdom.core.task import PreparedTask, TaskEvaluation, reserve_validation_split
from wisdom.core.wisdom_train import train_wisdom_classification
from wisdom.tasks.classification import ClassificationAdapter
from wisdom.utils.checkpoints import load_pytorch_model
from wisdom.utils.cli_options import (
    add_selection_arguments,
    comma_separated_floats,
    positive_int,
)


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_transform(
    mode: str,
    grayscale: bool,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    key = mode.lower()
    if key != "custom" and (mean is not None or std is not None):
        raise ValueError(
            "--normalize-mean and --normalize-std are only valid with "
            "--normalize custom."
        )
    if key == "none":
        return None
    if key == "imagenet":
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    if key == "cifar":
        return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    if key == "mnist":
        if not grayscale:
            raise ValueError(
                "--normalize mnist requires --grayscale for ImageFolder datasets."
            )
        return transforms.Normalize((0.1307,), (0.3081,))
    if key == "custom":
        if mean is None or std is None:
            raise ValueError(
                "--normalize custom requires both --normalize-mean and "
                "--normalize-std."
            )
        expected_channels = 1 if grayscale else 3
        if len(mean) != expected_channels or len(std) != expected_channels:
            raise ValueError(
                f"--normalize custom requires {expected_channels} values for both "
                "mean and std."
            )
        if any(not math.isfinite(value) for value in (*mean, *std)):
            raise ValueError("Custom normalization mean and std must be finite.")
        if any(value <= 0 for value in std):
            raise ValueError("Custom normalization std values must be positive.")
        return transforms.Normalize(tuple(mean), tuple(std))
    raise ValueError(f"Unsupported normalization mode: {mode}")


def build_classification_loader(
    root: str,
    batch_size: int,
    image_size: int,
    grayscale: bool,
    normalize: str,
    *,
    normalize_mean: Sequence[float] | None = None,
    normalize_std: Sequence[float] | None = None,
    num_workers: int = 0,
) -> DataLoader:
    transform_steps = [transforms.Resize((image_size, image_size))]
    if grayscale:
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    else:
        transform_steps.append(transforms.Lambda(lambda image: image.convert("RGB")))
    transform_steps.append(transforms.ToTensor())
    normalizer = _normalize_transform(
        normalize,
        grayscale,
        mean=normalize_mean,
        std=normalize_std,
    )
    if normalizer is not None:
        transform_steps.append(normalizer)
    dataset = datasets.ImageFolder(
        root=root,
        transform=transforms.Compose(transform_steps),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


_build_imagefolder_loader = build_classification_loader


def _build_local_dataset(name: str, root: str):
    """Load an already-present torchvision dataset without network downloads."""
    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        factory = datasets.MNIST
    elif name in {"cifar10", "cifar100"}:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        factory = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    elif name == "imagenet":
        image_root = Path(root)
        if (image_root / "train").is_dir():
            image_root = image_root / "train"
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return datasets.ImageFolder(str(image_root), transform=transform)
    else:  # Parser choices make this defensive.
        raise ValueError(f"Unsupported classification dataset: {name}")

    try:
        return factory(root=root, train=True, download=False, transform=transform)
    except RuntimeError as exc:
        raise FileNotFoundError(
            f"Dataset '{name}' is not available below '{root}'. WISDOM does not "
            "download datasets from this entry point."
        ) from exc


def _parse_method_csvs(items: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --method-out-csv value '{item}'. Expected method=path."
            )
        method, path = item.split("=", 1)
        method = method.strip().lower()
        if not method:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Empty method.")
        mapping[method] = path
    return mapping


def _read_index_file(path: str | None) -> list[int] | None:
    if not path:
        return None
    return [
        int(line)
        for raw in Path(path).read_text(encoding="utf-8").splitlines()
        if (line := raw.strip())
    ]


def resolve_model_factory(spec: str | None) -> Callable[[], nn.Module] | None:
    if spec is None:
        return None
    if ":" not in spec:
        raise ValueError("--model-factory must use module.path:callable syntax.")
    module_name, attribute_name = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise ValueError(f"Model factory '{spec}' is not callable.")
    return factory


def load_classification_model(
    model_path: str,
    *,
    device: str,
    checkpoint_format: str,
    model_factory: str | None = None,
) -> nn.Module:
    return load_pytorch_model(
        model_path,
        device=device,
        checkpoint_format=checkpoint_format,
        model_factory=resolve_model_factory(model_factory),
    )


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> TaskEvaluation:
    """Measure classification accuracy, mean cross entropy, and weighted F1."""
    from sklearn.metrics import f1_score

    adapter = ClassificationAdapter(model)
    total_samples = 0
    total_correct = 0
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = adapter.prepare_batch(raw_batch, device)
            if not isinstance(batch.targets, torch.Tensor):
                raise TypeError("Classification batch is missing Tensor class labels.")
            logits = model(batch.inputs)
            labels = batch.targets
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            predictions = logits.argmax(dim=1)
            total_samples += labels.numel()
            total_correct += int((predictions == labels).sum().item())
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("Classification evaluation loader contains no samples.")

    metrics = {
        "accuracy": total_correct / total_samples,
        "loss": total_loss / total_samples,
        "f1": float(
            f1_score(
                all_labels,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }
    return TaskEvaluation(
        metrics=metrics,
        bo_metric_name="f1",
        bo_metric_value=metrics["f1"],
    )


def _validate_class_mappings(*loaders: DataLoader | None) -> None:
    present_loaders = [loader for loader in loaders if loader is not None]
    reference_mapping = present_loaders[0].dataset.class_to_idx
    if any(loader.dataset.class_to_idx != reference_mapping for loader in present_loaders[1:]):
        raise ValueError(
            "Classification build, validation, and test datasets must have identical "
            "class mapping."
        )


def prepare_classification_inference(args: argparse.Namespace) -> PreparedTask:
    """Load a classification task and distinct loaders for WISDOM inference."""
    model = load_classification_model(
        args.weights_path,
        device=args.device,
        checkpoint_format=args.checkpoint_format,
        model_factory=args.model_factory,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "grayscale": args.grayscale,
        "normalize": args.normalize,
        "normalize_mean": getattr(args, "normalize_mean", None),
        "normalize_std": getattr(args, "normalize_std", None),
        "num_workers": args.num_workers,
    }
    build_loader = build_classification_loader(args.build_data_path, **loader_kwargs)
    validation_loader = (
        build_classification_loader(args.validation_data_path, **loader_kwargs)
        if args.validation_data_path
        else None
    )
    test_loader = build_classification_loader(args.test_data_path, **loader_kwargs)
    _validate_class_mappings(build_loader, validation_loader, test_loader)
    if getattr(args, "bo", False) and validation_loader is None:
        build_loader, validation_loader = reserve_validation_split(
            build_loader,
            seed=getattr(args, "seed", 42),
        )

    def score_trainer(out_csv: str) -> str:
        return train_wisdom_classification(
            model=model,
            train_loader=build_loader,
            out_csv=out_csv,
            top_m=args.top_m_neurons,
            methods=args.methods,
            voting_mode=args.voting_mode,
            selection_mode=args.selection_mode,
            n_groups=args.num_groups,
            num_layers=args.num_layers,
            device=args.device,
            checkpoint_path=args.trainer_checkpoint,
            checkpoint_every=args.checkpoint_every,
        )

    return PreparedTask(
        task="classification",
        adapter=ClassificationAdapter(model),
        build_loader=build_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        score_trainer=score_trainer,
        evaluate=lambda loader: evaluate_classification(model, loader, args.device),
        model_name=type(model).__name__,
        checkpoint_format=args.checkpoint_format,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WISDOM consensus pretraining for classification models"
    )
    parser.add_argument("--model-path", required=True, help="Model/checkpoint path")
    parser.add_argument(
        "--checkpoint-format",
        choices=["auto", "module", "state-dict"],
        default="module",
        help=(
            "module loads a trusted serialized nn.Module; state-dict/auto require "
            "--model-factory because weights do not contain an architecture"
        ),
    )
    parser.add_argument(
        "--model-factory",
        default=None,
        help="Zero-argument architecture factory in module.path:callable form",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "cifar100", "imagenet"],
        help="Already-downloaded torchvision dataset",
    )
    source_group.add_argument(
        "--imagefolder-root",
        help="Classification data in torchvision ImageFolder layout",
    )
    parser.add_argument("--data-path", default="./datasets")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--image-size", type=positive_int, default=224)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument(
        "--normalize",
        choices=["none", "imagenet", "cifar", "mnist", "custom"],
        default="none",
    )
    parser.add_argument(
        "--normalize-mean",
        type=comma_separated_floats,
        default=None,
        metavar="M1,M2,...",
        help="Per-channel mean required by --normalize custom",
    )
    parser.add_argument(
        "--normalize-std",
        type=comma_separated_floats,
        default=None,
        metavar="S1,S2,...",
        help="Positive per-channel standard deviation required by --normalize custom",
    )
    parser.add_argument("--top-m", type=positive_int, default=20)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--voting-mode",
        default="fine-grained",
        choices=["fine-grained", "coarse"],
    )
    add_selection_arguments(parser)
    parser.add_argument(
        "--out-csv",
        default="neuron_eval_out/wisdom_classification_scores.csv",
    )
    parser.add_argument("--device", default=_default_device())
    parser.add_argument("--checkpoint", default=None, help="Trainer resume checkpoint")
    parser.add_argument("--checkpoint-every", type=positive_int, default=50)
    parser.add_argument("--index-file", default=None)
    parser.add_argument("--method-out-csv", nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> str:
    args = build_parser().parse_args(argv)
    if args.dataset and args.normalize == "custom":
        raise ValueError(
            "--normalize custom is supported with --imagefolder-root; named "
            "torchvision datasets retain their established preprocessing."
        )
    model = load_classification_model(
        args.model_path,
        device=args.device,
        checkpoint_format=args.checkpoint_format,
        model_factory=args.model_factory,
    )

    if args.dataset:
        dataset = _build_local_dataset(args.dataset, args.data_path)
        indices = _read_index_file(args.index_file)
        if indices is not None:
            dataset = Subset(dataset, indices)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        if args.index_file:
            raise ValueError("--index-file is only supported with --dataset.")
        train_loader = _build_imagefolder_loader(
            root=args.imagefolder_root,
            batch_size=args.batch_size,
            image_size=args.image_size,
            grayscale=args.grayscale,
            normalize=args.normalize,
            normalize_mean=args.normalize_mean,
            normalize_std=args.normalize_std,
        )

    csv_path = train_wisdom_classification(
        model=model,
        train_loader=train_loader,
        out_csv=args.out_csv,
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        selection_mode=args.selection_mode,
        n_groups=args.num_groups,
        num_layers=args.num_layers,
        device=args.device,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        method_out_csvs=_parse_method_csvs(args.method_out_csv) or None,
    )
    print(f"\nDone. CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
