#!/usr/bin/env python
"""
wisdom_yolo_train.py
====================
Detection WISDOM pretraining CLI wrapper.

Loads a pretrained YOLO detection model, resolves a dataset image source from a
YAML file or explicit path, and delegates the actual WISDOM pretraining work to
`wisdom.core.wisdom_train.train_wisdom_yolo(...)`.

Usage
-----
    python wisdom_yolo_train.py \
        --weights weights/yolo11n.pt \
        --data standalone/data/coco128.yaml \
        --batch-size 4 \
        --num-images 100 \
        --top-m 20 \
        --methods lgxa lig lgs \
        --voting-mode fine-grained \
        --out-csv wisdom_yolo_scores.csv \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import batched_nms, box_iou

from wisdom.core.wisdom_train import COCOImageDataset, collate_image_tuples, train_wisdom_yolo
from wisdom.core.task import PreparedTask, TaskEvaluation, reserve_validation_split
from wisdom.tasks.detection import DetectionAdapter
from wisdom.utils.checkpoints import load_pytorch_model
from wisdom.utils.cli_options import add_selection_arguments
from wisdom.utils.detection_loader import infer_num_classes, normalize_detection_output


# Backward-compatible wrapper: the packaged implementation now lives in
# wisdom.core.wisdom_train.
_collate = collate_image_tuples


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_image_paths(image_source: str | Path) -> list[Path]:
    source = Path(image_source)
    if source.is_dir():
        return [
            path for path in sorted(source.iterdir())
            if path.suffix.lower() in _IMAGE_EXTENSIONS
        ]
    if source.is_file() and source.suffix.lower() == ".txt":
        paths: list[Path] = []
        for raw_line in source.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path = Path(line)
            paths.append(path if path.is_absolute() else (source.parent / path).resolve())
        return paths
    if source.is_file() and source.suffix.lower() in _IMAGE_EXTENSIONS:
        return [source]
    raise FileNotFoundError(f"Unsupported image source: {image_source}")


def _parallel_label_path(image_path: Path) -> Path | None:
    parts = image_path.parts
    try:
        images_index = parts.index("images")
    except ValueError:
        return None
    return Path(*parts[:images_index], "labels", *parts[images_index + 1:]).with_suffix(".txt")


class DetectionInferenceDataset(Dataset):
    """Deterministic detection images, retaining labels only when all are present."""

    def __init__(
        self,
        image_source: str,
        image_size: int,
        load_labels: bool = True,
    ) -> None:
        self.paths = _resolve_image_paths(image_source)
        self.transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )
        self.label_paths: list[Path] | None = None
        if load_labels:
            candidates = [_parallel_label_path(path) for path in self.paths]
            if all(path is not None for path in candidates):
                resolved = [path for path in candidates if path is not None]
                present = [path.exists() for path in resolved]
                if any(present) and not all(present):
                    raise ValueError(
                        "Detection dataset has partial label presence; provide labels for "
                        "every image or no label files."
                    )
                if all(present):
                    self.label_paths = resolved

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def _read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
        labels: list[tuple[int, float, float, float, float]] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}; expected 5 fields.")
            try:
                class_id = int(fields[0])
                cx, cy, width, height = (float(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(f"Invalid YOLO label at {path}:{line_number}.") from exc
            labels.append((class_id, cx, cy, width, height))
        return labels

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.label_paths is None:
            return (tensor,)
        return tensor, self._read_labels(self.label_paths[index])


def collate_detection_batch(batch):
    images = torch.stack([item[0] for item in batch])
    if len(batch[0]) == 1:
        return (images,)
    return images, [item[1] for item in batch]


def _detection_model_factory(model_path: str) -> nn.Module:
    try:
        from ultralytics.nn.tasks import DetectionModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Detection support requires Ultralytics; restore project dependencies "
            "with `uv sync`."
        ) from exc
    path = Path(model_path)
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Detection --model-path must be a local Ultralytics YAML architecture.")
    return DetectionModel(str(path), ch=3, nc=None, verbose=False)


def load_detection_architecture(
    model_path: str | None,
    weights_path: str,
    checkpoint_format: str,
    device: str,
) -> nn.Module:
    """Load local weights, reconstructing from YAML only for state dictionaries.

    ``module`` explicitly trusts pickle: accept a full module or Ultralytics'
    model/ema container, preferring EMA as its own loader does. Load on CPU and
    convert saved FP16 parameters to FP32 for our float32 image tensors. Do not
    fuse layers: that would change the modules used by the neuron-score CSV.
    No download or package auto-install fallback is attempted.
    """
    if checkpoint_format == "module":
        payload = torch.load(
            Path(weights_path), map_location=torch.device("cpu"), weights_only=False,
        )
        model = payload
        if isinstance(payload, dict):
            model = payload.get("ema")
            if model is None:
                model = payload.get("model")
        if not isinstance(model, nn.Module):
            raise TypeError(
                "Trusted detection checkpoint must contain an nn.Module or an "
                "Ultralytics 'model'/'ema' module; for state dictionaries use "
                "checkpoint-format state-dict with a local --model-path YAML."
            )
        model = model.float()
    else:
        if not model_path:
            raise ValueError("Detection state-dict checkpoints require a local --model-path YAML.")
        model = load_pytorch_model(
            weights_path,
            device="cpu",
            checkpoint_format=checkpoint_format,
            model_factory=lambda: _detection_model_factory(model_path),
        )
    return model.to(device).eval()


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half_size = boxes[:, 2:] / 2
    return torch.cat((boxes[:, :2] - half_size, boxes[:, :2] + half_size), dim=1)


def _match_detection_batch(
    predictions: torch.Tensor,
    labels: list[tuple[int, float, float, float, float]],
    image_size: int,
    num_classes: int,
) -> tuple[int, int, int]:
    normalized = normalize_detection_output(predictions.unsqueeze(0), num_classes)[0]
    boxes = _xywh_to_xyxy(normalized[:4].transpose(0, 1) / image_size).clamp(0.0, 1.0)
    scores, classes = normalized[4:].max(dim=0)
    keep = scores >= 0.25
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if len(boxes):
        selected = batched_nms(boxes, scores, classes, iou_threshold=0.45)
        order = selected[scores[selected].argsort(descending=True)]
        boxes, classes = boxes[order], classes[order]

    target_classes = torch.tensor([label[0] for label in labels], device=boxes.device)
    target_boxes = torch.tensor([label[1:] for label in labels], dtype=boxes.dtype, device=boxes.device)
    if len(target_boxes):
        target_boxes = _xywh_to_xyxy(target_boxes).clamp(0.0, 1.0)
    matched: set[int] = set()
    true_positive = 0
    for box, class_id in zip(boxes, classes):
        candidates = [
            index for index in range(len(target_boxes))
            if index not in matched and target_classes[index] == class_id
        ]
        if not candidates:
            continue
        ious = box_iou(box.unsqueeze(0), target_boxes[candidates]).squeeze(0)
        best_iou, candidate_index = ious.max(dim=0)
        if best_iou >= 0.5:
            matched.add(candidates[int(candidate_index)])
            true_positive += 1
    return true_positive, len(boxes), len(labels)


def evaluate_detection(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    image_size: int,
    num_classes: int,
) -> TaskEvaluation:
    """Report detection quality only when the loader carries complete labels."""
    total_predictions = total_targets = true_positive = 0
    has_labels: bool | None = None
    model.eval().to(device)
    with torch.no_grad():
        for raw_batch in loader:
            if isinstance(raw_batch, torch.Tensor):
                images, labels = raw_batch, None
            elif isinstance(raw_batch, (tuple, list)) and raw_batch:
                images = raw_batch[0]
                labels = raw_batch[1] if len(raw_batch) > 1 else None
            else:
                raise TypeError("Detection loader must yield image tensors or (images, labels).")
            labeled_batch = labels is not None
            if has_labels is None:
                has_labels = labeled_batch
            elif has_labels != labeled_batch:
                raise ValueError("Detection evaluation loader mixes labeled and unlabeled batches.")
            if not labeled_batch:
                continue
            output = normalize_detection_output(model(images.to(device)), num_classes)
            for prediction, targets in zip(output, labels):
                tp, predictions, target_count = _match_detection_batch(
                    prediction, targets, image_size, num_classes
                )
                true_positive += tp
                total_predictions += predictions
                total_targets += target_count

    if not has_labels:
        return TaskEvaluation(
            metrics={"precision": None, "recall": None, "f1": None},
            bo_metric_name=None,
            bo_metric_value=None,
        )
    precision = true_positive / total_predictions if total_predictions else 0.0
    recall = true_positive / total_targets if total_targets else 0.0
    denominator = precision + recall
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / denominator if denominator else 0.0,
    }
    return TaskEvaluation(metrics=metrics, bo_metric_name="f1", bo_metric_value=metrics["f1"])


def _build_detection_loader(image_source: str, args: argparse.Namespace) -> DataLoader:
    dataset = DetectionInferenceDataset(image_source, args.imgsz)
    if not len(dataset):
        raise FileNotFoundError(f"No images found in {image_source}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_detection_batch,
        num_workers=args.num_workers,
    )


def prepare_detection_inference(args: argparse.Namespace) -> PreparedTask:
    """Load runner-only YOLO inference components with distinct data roles."""
    model = load_detection_architecture(
        args.model_path, args.weights_path, args.checkpoint_format, args.device
    )
    num_classes = infer_num_classes(model)
    build_loader = _build_detection_loader(args.build_data_path, args)
    validation_loader = (
        _build_detection_loader(args.validation_data_path, args)
        if args.validation_data_path
        else None
    )
    test_loader = _build_detection_loader(args.test_data_path, args)
    if getattr(args, "bo", False) and validation_loader is None:
        build_loader, validation_loader = reserve_validation_split(
            build_loader,
            seed=getattr(args, "seed", 42),
        )

    def score_trainer(out_csv: str) -> str:
        return train_wisdom_yolo(
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
            num_workers=args.num_workers,
            num_classes=num_classes,
        )

    return PreparedTask(
        task="detection",
        adapter=DetectionAdapter(model, num_classes=num_classes),
        build_loader=build_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        score_trainer=score_trainer,
        evaluate=lambda loader: evaluate_detection(
            model, loader, args.device, args.imgsz, num_classes
        ),
        model_name=type(model).__name__,
        checkpoint_format=args.checkpoint_format,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detection WISDOM pretraining wrapper")
    p.add_argument(
        "--weights", default="weights/yolo11n.pt",
        help="Trusted local Ultralytics .pt checkpoint; a model YAML instead creates random weights for testing.",
    )
    p.add_argument("--data", default="standalone/data/coco128.yaml", help="Dataset YAML whose train entry resolves the image source.")
    p.add_argument("--img-dir", default=None, help="Override image source: directory, txt list, or single image path.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for image decoding")
    p.add_argument("--num-images", type=int, default=100, help="Max images to use")
    p.add_argument("--top-m", type=int, default=20, help="Top-M neurons per method")
    p.add_argument("--methods", nargs="+", default=["lgxa", "lig", "lgs"])
    p.add_argument("--voting-mode", default="fine-grained", choices=["fine-grained", "coarse"])
    add_selection_arguments(p)
    p.add_argument("--out-csv", default="neuron_eval_out/wisdom_yolo_scores.csv")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint file for resume support")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save checkpoint every N batches (default 50)")
    p.add_argument(
        "--method-out-csv",
        nargs="*",
        default=[],
        help="Optional method-level CSV outputs in method=path form.",
    )
    return p


def _resolve_image_source(args) -> str:
    if args.img_dir:
        return args.img_dir
    import yaml

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)
    image_source = data_cfg.get("train", "")
    if not os.path.isabs(image_source):
        image_source = os.path.join(os.path.dirname(args.data), image_source)
    return image_source


def parse_args(argv: list[str] | None = None):
    return build_parser().parse_args(argv)


def parse_method_out_csvs(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Expected method=path.")
        method, path = item.split("=", 1)
        method = method.strip().lower()
        if not method:
            raise ValueError(f"Invalid --method-out-csv value '{item}'. Empty method.")
        mapping[method] = path
    return mapping


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    image_source = _resolve_image_source(args)
    method_out_csvs = parse_method_out_csvs(args.method_out_csv)

    csv_path = train_wisdom_yolo(
        weights=args.weights,
        img_dir=image_source,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        num_images=args.num_images,
        top_m=args.top_m,
        methods=args.methods,
        voting_mode=args.voting_mode,
        selection_mode=args.selection_mode,
        n_groups=args.num_groups,
        num_layers=args.num_layers,
        device=args.device,
        imgsz=args.imgsz,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
        method_out_csvs=method_out_csvs or None,
        num_workers=args.num_workers,
    )
    print(f"\nDone. CSV: {csv_path}")
    return csv_path


if __name__ == "__main__":
    main()
