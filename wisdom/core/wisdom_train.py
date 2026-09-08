# core/wisdom_train.py
from __future__ import annotations
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from wisdom.utils.io_cache import save_layer_scores_csv
from wisdom.utils.detection_loader import infer_num_classes
from wisdom.attribution.captum_backend import batch_per_layer_scores
from wisdom.core.layers import build_layer_plan, discover_eligible_layers
from wisdom.core.task import TaskAdapter

# -----------------------------
# Config
# -----------------------------
@dataclass
class WisdomTrainConfig:
    methods: List[str] = field(default_factory=lambda: ["lrp", "ldl", "lig"])
    device: str = "cuda:0"
    voting_weights: Optional[List[float]] = None
    voting_mode: str = "fine-grained"  # "fine-grained" | "coarse"
    selection_mode: str = "global"  # "global" | "per-group" | "per-layer"
    n_groups: int = 3
    num_layers: int | None = None
    pruning_augmentations: Optional[List[Dict]] = None
    out_csv: Optional[str] = None
    method_out_csvs: Optional[Dict[str, str]] = None
    diagnostics_csv: Optional[str] = None
    # Deprecated compatibility alias used only when no task_adapter is supplied.
    is_yolo: bool = False
    num_classes: int = 80  # COCO classes for YOLO


_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_image_paths(image_source: str | Path) -> list[Path]:
    source = Path(image_source)
    if source.is_dir():
        paths = [path for path in sorted(source.iterdir()) if path.suffix.lower() in _IMAGE_EXTENSIONS]
    elif source.is_file() and source.suffix.lower() == ".txt":
        paths = []
        for raw_line in source.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            path = Path(line)
            if not path.is_absolute():
                path = (source.parent / path).resolve()
            paths.append(path)
    elif source.is_file() and source.suffix.lower() in _IMAGE_EXTENSIONS:
        paths = [source]
    else:
        raise FileNotFoundError(f"Unsupported image source: {image_source}")
    return paths


class DetectionImageDataset(Dataset):
    """Returns unlabeled RGB image tensors from a directory, image list, or single image."""

    def __init__(self, image_source: str, max_images: int | None = None, imgsz: int = 640):
        self.paths = _resolve_image_paths(image_source)
        if max_images is not None:
            self.paths = self.paths[:max_images]
        self.transform = transforms.Compose([
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img),


COCOImageDataset = DetectionImageDataset


def collate_image_tuples(batch):
    """Stack single-element image tuples into a single image batch tuple."""
    imgs = torch.stack([b[0] for b in batch])
    return (imgs,)


# -----------------------------
# Helper functions
# -----------------------------
def _is_trainable_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Conv2d, nn.Linear))

def _trainable_modules(model: nn.Module) -> Tuple[List[str], List[nn.Module]]:
    names, mods = [], []
    for n, m in model.named_modules():
        if _is_trainable_module(m):
            names.append(n)
            mods.append(m)
    return names, mods

def _emit_training_summary(layer_scores: Dict[str, torch.Tensor], csv_path: str) -> None:
    print(f"Saved layer scores to {csv_path}")
    print(f"Layers scored: {len(layer_scores)}")
    total_scored = sum(t.numel() for t in layer_scores.values())
    non_zero = sum((t != 0).sum().item() for t in layer_scores.values())
    print(f"Total neurons: {total_scored}, non-zero scores: {non_zero}")

def _voting_init(layer_scores: Dict[str, torch.Tensor],
                 trainable_names: List[str],
                 trainable_mods: List[nn.Module],
                 excluded_layer: Optional[str] = None,
                 excluded_prefixes: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
    if layer_scores:
        return layer_scores
    for lname, m in zip(trainable_names, trainable_mods):
        if excluded_layer and lname == excluded_layer:
            continue
        if excluded_prefixes and any(lname.startswith(p) for p in excluded_prefixes):
            continue
        if isinstance(m, nn.Conv2d):
            layer_scores[lname] = torch.zeros(m.out_channels, dtype=torch.float32)
        elif isinstance(m, nn.Linear):
            layer_scores[lname] = torch.zeros(m.out_features, dtype=torch.float32)
    return layer_scores

def _voting_neurons(layer_index_pairs: List[Tuple[str, int]],
                    layer_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Assign rank points (higher rank gets more points) like prepare_data.voting_neurons.
    Input list is assumed sorted DESC by importance; we intentionally enumerate reversed.
    """
    for rank, (layer_name, neuron_index) in enumerate(reversed(layer_index_pairs), start=1):
        if layer_name in layer_scores and 0 <= neuron_index < layer_scores[layer_name].numel():
            layer_scores[layer_name][neuron_index] += rank
    return layer_scores

def _loss_gain_weights(loss_gains: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive pruning gains, with a fail-safe for all-negative batches."""

    if not loss_gains:
        raise ValueError("At least one pruning loss gain is required")
    positive_gains = {method: max(0.0, gain) for method, gain in loss_gains.items()}
    total_gain = sum(positive_gains.values())
    if total_gain > 0.0:
        return {
            method: gain / total_gain for method, gain in positive_gains.items()
        }
    # Masking can improve the loss for every method on a small/noisy batch.
    # The historical implementation then assigned zero weight to every method
    # and silently discarded the whole batch. Preserve the intended loss-gain
    # ordering by using the least harmful method, also coarse voting's choice.
    best_method = max(loss_gains, key=loss_gains.get)
    return {method: float(method == best_method) for method in loss_gains}


def _weighted_top_neurons(important_neurons_dict: Dict[str, List[Tuple[str, float, int]]],
                          loss_gains: Dict[str, float],
                          top_k: int,
                          layer_groups: Mapping[str, Iterable[str]] | None = None,
                          ) -> List[Tuple[Tuple[str, int], float]]:
    """
    Same spirit as prepare_data.weighted_top_neurons:
      - normalize loss_gains to weights,
      - accumulate weight * score per (layer, idx),
      - return top_k ((layer, idx), weighted_score), within each supplied scope.
    """
    weights = _loss_gain_weights(loss_gains)
    weighted_scores: Dict[Tuple[str, int], float] = {}
    for method, triples in important_neurons_dict.items():
        w = weights.get(method, 0.0)
        if w == 0.0:
            continue
        for (layer_name, score, idx) in triples:
            key = (layer_name, int(idx))
            weighted_scores[key] = weighted_scores.get(key, 0.0) + float(score) * w
    sorted_neurons = sorted(weighted_scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k == -1:
        return sorted_neurons
    if layer_groups is None:
        return sorted_neurons[:top_k]
    layer_to_group = {layer: group for group, layers in layer_groups.items() for layer in layers}
    counts: Dict[str, int] = {}
    selected = []
    for pair, score in sorted_neurons:
        group = layer_to_group[pair[0]]
        if counts.get(group, 0) < top_k:
            selected.append((pair, score))
            counts[group] = counts.get(group, 0) + 1
    return selected


def _select_top_neurons_all(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_neurons: int,
    filter_layer: Optional[str] = None,
    filter_prefixes: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """
    Flatten all layers' importance and pick top-M across layers (optionally excluding final layer
    or layers matching given prefixes, e.g. detection head).
    Returns:
      - indices_by_layer: {layer: 1D LongTensor of selected indices}
      - selected_triplets: list of (layer_name, score, idx) sorted desc by score
    """
    flattened: List[Tuple[str, float, int]] = []
    for layer_name, scores in importance_scores_dict.items():
        if filter_layer and layer_name == filter_layer:
            continue
        if filter_prefixes and any(layer_name.startswith(p) for p in filter_prefixes):
            continue
        if scores.dim() == 1:
            for idx, s in enumerate(scores):
                flattened.append((layer_name, float(s.item()), int(idx)))
        else:
            # Should not happen because we reduce per layer, but keep safe
            mean_attr = scores.mean(dim=tuple(range(1, scores.dim())))
            for idx, s in enumerate(mean_attr):
                flattened.append((layer_name, float(s.item()), int(idx)))

    flattened.sort(key=lambda x: x[1], reverse=True)
    selected = flattened if top_m_neurons == -1 else flattened[:top_m_neurons]

    by_layer: Dict[str, List[int]] = {}
    for layer_name, _, idx in selected:
        by_layer.setdefault(layer_name, []).append(idx)

    indices_by_layer = {
        layer: torch.tensor(sorted(idxs), dtype=torch.long) for layer, idxs in by_layer.items()
    }
    return indices_by_layer, selected


def _select_top_neurons_per_group(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_per_group: int,
    layer_groups: Mapping[str, Iterable[str]],
    filter_layer: Optional[str] = None,
    filter_prefixes: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """
    Select top-M neurons per explicit model-derived layer group.

    This ensures balanced representation across network depth, avoiding
    the early-layer dominance that occurs with global top-M selection.

    Returns same format as ``_select_top_neurons_all``.
    """
    if top_m_per_group <= 0:
        raise ValueError(
            f"top_m_per_group must be a positive integer, got {top_m_per_group}"
        )

    assigned_layers: set[str] = set()
    selected: List[Tuple[str, float, int]] = []
    for group_name, group_layers in layer_groups.items():
        names = tuple(str(name) for name in group_layers)
        if not names:
            raise ValueError(f"Layer group {group_name!r} must not be empty")
        duplicates = assigned_layers.intersection(names)
        if duplicates:
            raise ValueError(f"Layers appear in more than one group: {sorted(duplicates)}")
        assigned_layers.update(names)
        group_scores = {
            name: importance_scores_dict[name]
            for name in names
            if name in importance_scores_dict
        }
        _, group_selected = _select_top_neurons_all(
            group_scores,
            top_m_neurons=top_m_per_group,
            filter_layer=filter_layer,
            filter_prefixes=filter_prefixes,
        )
        selected.extend(group_selected)

    # Re-sort combined result DESC by score (for voting)
    selected.sort(key=lambda x: x[1], reverse=True)

    by_layer: Dict[str, List[int]] = {}
    for layer_name, _, idx in selected:
        by_layer.setdefault(layer_name, []).append(idx)

    indices_by_layer = {
        layer: torch.tensor(sorted(idxs), dtype=torch.long)
        for layer, idxs in by_layer.items()
    }
    return indices_by_layer, selected


def _select_top_neurons_per_layer(
    importance_scores_dict: Dict[str, torch.Tensor],
    top_m_per_layer: int,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, float, int]]]:
    """Select M neurons independently within every considered layer."""
    if top_m_per_layer == 0 or top_m_per_layer < -1:
        raise ValueError("top_m_per_layer must be positive, or -1 for all neurons")
    selected: List[Tuple[str, float, int]] = []
    for layer_name, scores in importance_scores_dict.items():
        _, layer_selected = _select_top_neurons_all(
            {layer_name: scores},
            top_m_neurons=top_m_per_layer,
        )
        selected.extend(layer_selected)
    selected.sort(key=lambda item: item[1], reverse=True)
    by_layer: Dict[str, List[int]] = {}
    for layer_name, _score, index in selected:
        by_layer.setdefault(layer_name, []).append(index)
    return (
        {
            layer_name: torch.tensor(sorted(indices), dtype=torch.long)
            for layer_name, indices in by_layer.items()
        },
        selected,
    )


# -----------------------------
# Pruning backends (new style)
# -----------------------------
class _WeightsGuard:
    def __init__(self, model: nn.Module):
        self.model = model
        self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def restore(self):
        self.model.load_state_dict(self.state, strict=True)


def _prune_and_evaluate(
    model: nn.Module,
    selection: Dict[str, List[int]],
    evaluate,
    prune_mode: str,
) -> float:
    """Apply one reversible pruning strategy and always restore model behavior."""
    if prune_mode not in {"mask", "weights"}:
        raise ValueError("prune_mode must be 'mask' or 'weights'")
    if prune_mode == "mask":
        from wisdom.pruning.mask_pruning import mask_model_neurons

        handle = mask_model_neurons(model, selection)
        try:
            return float(evaluate())
        finally:
            handle.remove()

    from wisdom.pruning.weights_pruning import prune_model_neurons

    guard = _WeightsGuard(model)
    try:
        prune_model_neurons(model, selection)
        return float(evaluate())
    finally:
        guard.restore()

# -----------------------------
# Main Trainer
# -----------------------------
class ConsensusWisdom:
    """
    Multi-method voting + optional pruning to identify important neurons.
    Steps per batch:
      1) Get important neurons for each attribution method.
      2) Identify optimal method by pruning its top neurons and measuring loss gain.
      3) Initialize voting buffers on first batch.
      4) Update votes:
         - fine-grained: pick TOP-K neurons across methods via weighted_top_neurons (weights from loss gains),
           then vote them by rank (like prepare_data.voting_neurons).
         - coarse: take only the optimal method's neurons and vote by rank.
      5) After all batches, save CSV.

    A task adapter supplies batch parsing, Captum targets, exclusions, and the
    task-specific loss. Omitting it retains the historical classification
    behavior; ``cfg.is_yolo`` remains a compatibility route for detection.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda:0",
        task_adapter: TaskAdapter | None = None,
    ):
        self.model = model
        self.device = device
        self.model.eval().to(device)
        self.trainable_names, self.trainable_mods = _trainable_modules(model)
        self.task_adapter = task_adapter

    # -------- public API --------
    def fit(
        self,
        train_loader: DataLoader,
        cfg: WisdomTrainConfig,
        top_m_neurons: int,
        final_layer: Optional[str] = None,
        prune_mode: str = "mask",  # "mask" | "weights"
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 50,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Returns (layer_scores, csv_path).

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to a ``.pt`` file for saving/resuming progress.  If the
            file already exists, training resumes from the saved batch
            index.  A checkpoint is written every *checkpoint_every*
            batches.
        checkpoint_every : int
            How often (in batches) to save a checkpoint.  Default 50.
        """
        import os
        assert cfg.out_csv, "Please provide cfg.out_csv to save layer scores."
        if cfg.selection_mode not in {"global", "per-group", "per-layer"}:
            raise ValueError(
                "selection_mode must be 'global', 'per-group', or 'per-layer'"
            )
        if not cfg.methods:
            raise ValueError("At least one attribution method is required")
        if top_m_neurons == 0 or top_m_neurons < -1:
            raise ValueError("top_m_neurons must be positive, or -1 for all neurons")
        if cfg.selection_mode == "per-group" and top_m_neurons == -1:
            raise ValueError("per-group top_m_neurons must be positive")
        if prune_mode not in {"mask", "weights"}:
            raise ValueError("prune_mode must be 'mask' or 'weights'")

        adapter = self.task_adapter
        if adapter is None:
            if cfg.is_yolo:
                from wisdom.tasks.detection import DetectionAdapter

                adapter = DetectionAdapter(self.model, num_classes=cfg.num_classes)
            else:
                from wisdom.tasks.classification import ClassificationAdapter

                adapter = ClassificationAdapter(self.model)

        analysis_model = adapter.analysis_model.eval().to(self.device)
        excluded_layers = (
            (final_layer,) if final_layer is not None else adapter.excluded_layer_names()
        )
        excluded_prefixes = adapter.excluded_layer_prefixes()
        eligible_layers = discover_eligible_layers(
            analysis_model,
            excluded_layers=excluded_layers,
            excluded_prefixes=excluded_prefixes,
        )
        layer_plan = build_layer_plan(
            eligible_layers,
            n_groups=cfg.n_groups if cfg.selection_mode == "per-group" else None,
            num_layers=cfg.num_layers,
        )
        analysis_modules = dict(analysis_model.named_modules())
        planned_names = list(layer_plan.eligible_layers)
        planned_modules = [analysis_modules[name] for name in planned_names]

        layer_scores: Dict[str, torch.Tensor] = {}
        method_layer_scores: Dict[str, Dict[str, torch.Tensor]] = {}
        diagnostics: List[Dict[str, object]] = []
        init_done = False

        # Checkpoint resume
        start_batch = 0
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            layer_scores = {k: v.clone() for k, v in ckpt["layer_scores"].items()}
            method_layer_scores = {
                method: {k: v.clone() for k, v in scores.items()}
                for method, scores in ckpt.get("method_layer_scores", {}).items()
            }
            diagnostics = list(ckpt.get("diagnostics", []))
            start_batch = ckpt["batch_idx"] + 1
            init_done = bool(layer_scores)
            print(f"[WISDOM] Resuming from checkpoint batch {start_batch}")

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            if batch_idx < start_batch:
                continue
            task_batch = adapter.prepare_batch(batch, cfg.device)
            images = task_batch.inputs
            target = adapter.captum_target(task_batch)

            # 1) Important neurons per method on THIS batch
            important_neurons_dict: Dict[str, List[Tuple[str, float, int]]] = {}
            for method in cfg.methods:
                imp = batch_per_layer_scores(
                    model=analysis_model,
                    images=images,
                    target=target,
                    device=cfg.device,
                    method=method,
                    target_layers=layer_plan.eligible_layers,
                )
                imp = {
                    layer_name: scores
                    for layer_name, scores in imp.items()
                    if layer_name in layer_plan.eligible_layers
                }
                # Select top neurons: global or per-group
                if cfg.selection_mode == "per-group":
                    _, selected_triplets = _select_top_neurons_per_group(
                        imp,
                        top_m_per_group=top_m_neurons,
                        layer_groups=layer_plan.groups,
                    )
                elif cfg.selection_mode == "per-layer":
                    _, selected_triplets = _select_top_neurons_per_layer(
                        imp,
                        top_m_per_layer=top_m_neurons,
                    )
                else:
                    _, selected_triplets = _select_top_neurons_all(
                        imp, top_m_neurons=top_m_neurons,
                    )
                important_neurons_dict[method] = selected_triplets

            # 2) Identify optimal method by loss gain after pruning
            reference = adapter.capture_reference(task_batch)

            def evaluate_loss() -> float:
                with torch.no_grad():
                    return float(adapter.loss(task_batch, reference).detach().item())

            base_loss = evaluate_loss()

            loss_gains: Dict[str, float] = {}
            for method, triplets in important_neurons_dict.items():
                selection: Dict[str, List[int]] = {}
                for (layer_name, _score, idx) in triplets:
                    selection.setdefault(layer_name, []).append(int(idx))

                pruned_loss = _prune_and_evaluate(
                    analysis_model,
                    selection,
                    evaluate_loss,
                    prune_mode,
                )

                loss_gains[method] = pruned_loss - base_loss

            optimal_method = max(loss_gains, key=loss_gains.get)
            loss_gain_weights = _loss_gain_weights(loss_gains)
            diagnostic_row: Dict[str, object] = {
                "batch_index": batch_idx,
                "batch_size": int(images.size(0)),
                "base_loss": float(base_loss),
                "optimal_method": optimal_method,
                "all_gains_nonpositive": all(gain <= 0.0 for gain in loss_gains.values()),
                "positive_gain_sum": float(
                    sum(max(0.0, gain) for gain in loss_gains.values())
                ),
            }
            for method in cfg.methods:
                diagnostic_row[f"{method}_loss_gain"] = float(loss_gains[method])
                diagnostic_row[f"{method}_weight"] = float(loss_gain_weights[method])
            diagnostics.append(diagnostic_row)

            # 3) Voting buffers
            if not init_done:
                layer_scores = _voting_init(
                    layer_scores,
                    planned_names,
                    planned_modules,
                )
                if cfg.method_out_csvs:
                    for method in cfg.methods:
                        if method not in cfg.method_out_csvs:
                            continue
                        method_layer_scores[method] = _voting_init(
                            {},
                            planned_names,
                            planned_modules,
                        )
                init_done = True

            if cfg.method_out_csvs:
                for method in cfg.methods:
                    if method not in cfg.method_out_csvs or method in method_layer_scores:
                        continue
                    method_layer_scores[method] = _voting_init(
                        {},
                        planned_names,
                        planned_modules,
                    )
                for method, triplets in important_neurons_dict.items():
                    if method not in method_layer_scores:
                        continue
                    ranked_triplets = sorted(triplets, key=lambda t: t[1], reverse=True)
                    layer_index_pairs = [(layer, idx) for (layer, _score, idx) in ranked_triplets]
                    _voting_neurons(layer_index_pairs, method_layer_scores[method])

            # 4) Vote according to mode
            if cfg.voting_mode == "coarse":
                opt_triplets = important_neurons_dict[optimal_method]
                opt_triplets = sorted(opt_triplets, key=lambda t: t[1], reverse=True)
                layer_index_pairs = [(layer, idx) for (layer, _score, idx) in opt_triplets]
                _voting_neurons(layer_index_pairs, layer_scores)
            else:
                voting_scopes = None
                if cfg.selection_mode == "per-group":
                    voting_scopes = layer_plan.groups
                elif cfg.selection_mode == "per-layer":
                    voting_scopes = {name: (name,) for name in layer_plan.eligible_layers}
                top_across = _weighted_top_neurons(
                    important_neurons_dict, loss_gains, top_k=top_m_neurons,
                    layer_groups=voting_scopes,
                )
                layer_index_pairs = [pair for (pair, _wscore) in top_across]
                _voting_neurons(layer_index_pairs, layer_scores)

            # Periodic checkpoint
            if checkpoint_path and (batch_idx + 1) % checkpoint_every == 0:
                torch.save(
                    {
                        "layer_scores": layer_scores,
                        "method_layer_scores": method_layer_scores,
                        "diagnostics": diagnostics,
                        "batch_idx": batch_idx,
                    },
                    checkpoint_path,
                )

        # 5) Save CSV
        # Remove checkpoint after successful completion
        if checkpoint_path and os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)
        out_csv = save_layer_scores_csv(layer_scores, cfg.out_csv)
        if cfg.method_out_csvs:
            for method, csv_path in cfg.method_out_csvs.items():
                if method in method_layer_scores:
                    save_layer_scores_csv(method_layer_scores[method], csv_path)
        if cfg.diagnostics_csv:
            diagnostic_path = Path(cfg.diagnostics_csv)
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "batch_index",
                "batch_size",
                "base_loss",
                "optimal_method",
                "all_gains_nonpositive",
                "positive_gain_sum",
                *[f"{method}_loss_gain" for method in cfg.methods],
                *[f"{method}_weight" for method in cfg.methods],
            ]
            with diagnostic_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(diagnostics)
        return layer_scores, out_csv


def train_wisdom_classification(
    model: nn.Module,
    train_loader: DataLoader,
    out_csv: str,
    top_m: int = 20,
    methods: list[str] | None = None,
    voting_mode: str = "fine-grained",
    device: str = "cuda:0",
    final_layer: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    method_out_csvs: dict[str, str] | None = None,
    selection_mode: str = "global",
    n_groups: int = 3,
    num_layers: int | None = None,
) -> str:
    """Run WISDOM consensus training on a classification model and save layer scores."""
    if methods is None:
        methods = ["lrp", "ldl", "lig"]

    cfg = WisdomTrainConfig(
        methods=methods,
        device=device,
        voting_mode=voting_mode,
        selection_mode=selection_mode,
        n_groups=n_groups,
        num_layers=num_layers,
        out_csv=out_csv,
        method_out_csvs=method_out_csvs,
    )
    trainer = ConsensusWisdom(model.eval(), device=device)
    layer_scores, csv_path = trainer.fit(
        train_loader,
        cfg,
        top_m_neurons=top_m,
        final_layer=final_layer,
        prune_mode="mask",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )
    _emit_training_summary(layer_scores, csv_path)
    return csv_path


def train_wisdom_pose(
    model: nn.Module,
    train_loader: DataLoader,
    out_csv: str,
    *,
    output_layer_names: tuple[str, str] | None = None,
    top_m: int = 20,
    methods: list[str] | None = None,
    voting_mode: str = "fine-grained",
    selection_mode: str = "global",
    n_groups: int = 3,
    num_layers: int | None = None,
    device: str = "cuda:0",
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    method_out_csvs: dict[str, str] | None = None,
) -> str:
    """Run WISDOM pretraining using pose behavioral sensitivity, not CE loss."""
    from wisdom.tasks.pose import PoseAdapter

    if methods is None:
        methods = ["lgxa", "lig"]
    adapter = PoseAdapter(model.eval(), output_layer_names=output_layer_names)
    cfg = WisdomTrainConfig(
        methods=methods,
        device=device,
        voting_mode=voting_mode,
        selection_mode=selection_mode,
        n_groups=n_groups,
        num_layers=num_layers,
        out_csv=out_csv,
        method_out_csvs=method_out_csvs,
    )
    trainer = ConsensusWisdom(
        adapter.analysis_model,
        device=device,
        task_adapter=adapter,
    )
    layer_scores, csv_path = trainer.fit(
        train_loader,
        cfg,
        top_m_neurons=top_m,
        prune_mode="mask",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )
    _emit_training_summary(layer_scores, csv_path)
    return csv_path


def train_wisdom_yolo(
    weights: str | None = None,
    img_dir: str | None = None,
    out_csv: str = "neuron_eval_out/wisdom_yolo_scores.csv",
    batch_size: int = 4,
    num_images: int = 100,
    top_m: int = 20,
    methods: list[str] | None = None,
    voting_mode: str = "fine-grained",
    selection_mode: str = "global",
    device: str = "cuda:0",
    imgsz: int = 640,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 50,
    method_out_csvs: dict[str, str] | None = None,
    num_workers: int = 0,
    n_groups: int = 3,
    num_layers: int | None = None,
    model: nn.Module | None = None,
    train_loader: DataLoader | None = None,
    num_classes: int | None = None,
) -> str:
    """Run WISDOM consensus training on a YOLO detection model and save layer scores."""
    from wisdom.tasks.detection import DetectionAdapter

    if methods is None:
        methods = ["lgxa", "lig", "lgs"]

    if model is None:
        if weights is None:
            raise ValueError("Provide either detection model=... or weights=...")
        from wisdom.utils.detection_loader import load_detection_model

        bundle = load_detection_model(weights, device=device)
        torch_model = bundle.model.eval()
        resolved_num_classes = bundle.num_classes
    else:
        torch_model = model.eval().to(device)
        resolved_num_classes = num_classes or infer_num_classes(torch_model)

    if train_loader is None:
        if img_dir is None:
            raise ValueError("Provide train_loader=... or img_dir=... for detection images")
        ds = DetectionImageDataset(img_dir, max_images=num_images, imgsz=imgsz)
        if len(ds) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": False,
            "collate_fn": collate_image_tuples,
            "num_workers": max(0, int(num_workers)),
            "pin_memory": str(device).startswith("cuda"),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = True
        train_loader = DataLoader(ds, **loader_kwargs)

    cfg = WisdomTrainConfig(
        methods=methods,
        device=device,
        voting_mode=voting_mode,
        selection_mode=selection_mode,
        n_groups=n_groups,
        num_layers=num_layers,
        out_csv=out_csv,
        method_out_csvs=method_out_csvs,
        num_classes=resolved_num_classes,
    )

    adapter = DetectionAdapter(torch_model, num_classes=resolved_num_classes)
    trainer = ConsensusWisdom(
        torch_model,
        device=device,
        task_adapter=adapter,
    )
    layer_scores, csv_path = trainer.fit(
        train_loader,
        cfg,
        top_m_neurons=top_m,
        prune_mode="mask",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )
    _emit_training_summary(layer_scores, csv_path)
    return csv_path
