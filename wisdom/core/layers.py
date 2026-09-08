from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch.nn as nn


_GROUP_NAME_PRESETS: dict[int, tuple[str, ...]] = {
    2: ("front", "back"),
    3: ("early", "middle", "late"),
    4: ("early", "mid_early", "mid_late", "late"),
    5: ("early", "mid_early", "middle", "mid_late", "late"),
}


# Immutable record of considered eligible layers and their groups, per-group layer order is preserved from the original model.
@dataclass(frozen=True)
class LayerPlan:
    eligible_layers: tuple[str, ...]
    groups: dict[str, tuple[str, ...]]


def get_group_names(n_groups: int) -> tuple[str, ...]:
    if not isinstance(n_groups, int) or isinstance(n_groups, bool) or n_groups <= 0:
        raise ValueError(f"n_groups must be a positive integer, got {n_groups!r}")
    return _GROUP_NAME_PRESETS.get(
        n_groups,
        tuple(f"group_{index + 1}" for index in range(n_groups)),
    )

# Walks named_modules() in registration order and retains trainable Conv2d/Linear modules after exact/prefix exclusions.
def discover_eligible_layers(
    model: nn.Module,
    *,
    excluded_layers: Iterable[str] = (),
    excluded_prefixes: Iterable[str] = (),
) -> tuple[str, ...]:
    blocked = {str(name) for name in excluded_layers}
    prefixes = tuple(str(prefix) for prefix in excluded_prefixes)
    discovered: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        if name in blocked or (prefixes and name.startswith(prefixes)):
            continue
        if not any(parameter.requires_grad for parameter in module.parameters(recurse=False)):
            continue
        discovered.append(name)
    return tuple(discovered)


def _unique_layer_names(layer_names: Sequence[str] | Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(name) for name in layer_names))

# Uses all eligible layers by default or selects an evenly distributed requested count
def limit_layers_evenly(
    layer_names: Sequence[str] | Iterable[str],
    count: int | None,
) -> tuple[str, ...]:
    names = _unique_layer_names(layer_names)
    if not names:
        raise ValueError("No eligible layers were discovered")
    if count is None:
        return names
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(names):
        raise ValueError(
            f"num_layers must be an integer in 1..{len(names)}, got {count!r}"
        )
    if count == 1:
        return (names[(len(names) - 1) // 2],)
    indices = [round(index * (len(names) - 1) / (count - 1)) for index in range(count)]
    return tuple(names[index] for index in indices)

# Applies the layer limit
# Validates groups <= considered layers
# Partitions every considered layer exactly once into balanced contiguous groups.
def build_layer_plan(
    layer_names: Sequence[str] | Iterable[str],
    *,
    n_groups: int | None = None,
    num_layers: int | None = None,
) -> LayerPlan:
    considered = limit_layers_evenly(layer_names, num_layers)
    if n_groups is None:
        return LayerPlan(eligible_layers=considered, groups={})

    group_names = get_group_names(n_groups)
    if n_groups > len(considered):
        raise ValueError(
            f"n_groups must not exceed the {len(considered)} eligible layers, "
            f"got {n_groups}"
        )

    base_size, remainder = divmod(len(considered), n_groups)
    groups: dict[str, tuple[str, ...]] = {}
    offset = 0
    for index, group_name in enumerate(group_names):
        size = base_size + (1 if index < remainder else 0)
        groups[group_name] = considered[offset : offset + size]
        offset += size
    return LayerPlan(eligible_layers=considered, groups=groups)
