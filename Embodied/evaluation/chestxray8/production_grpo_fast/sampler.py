"""Deterministic data order, accumulation-window, and distributed sharding.

Semantics (see DESIGN.md Sec 8/10 for the full argument):

- One epoch = one deterministic `Random(42 + epoch).shuffle(range(790))` pass
  over the fixed train80 manifest. No reshuffling mid-epoch, no dataset
  reads beyond `splits/train80_pairs_seed42.jsonl`.
- The 20 epochs / 15,800 prompt-group stream is treated as *continuous* for
  the purpose of the 4-groups-per-optimizer-step accumulation window: window
  `w` covers global group indices `[4w, 4w+1, 4w+2, 4w+3]`. Because
  `15800 / 4 == 3950` exactly, this yields exactly 3,950 full-size windows
  and never a partial (<4-group) optimizer update, at the cost of a handful
  of windows straddling an epoch boundary (790 % 4 == 2, so every epoch
  shifts the phase by 2 groups). Epoch boundaries stay meaningful only as
  checkpoint/logging markers, not accumulation resets.
- Distributed sharding splits each 4-group window contiguously across
  workers, as evenly as the world size allows (`worker_group_indices`):
  exactly even for world sizes 1, 2, 4; a 2+1+1 split for world size 3. The
  window always sums to exactly `GROUPS_PER_UPDATE` groups regardless of
  world size, so the fixed "4 groups/update" contract holds for every
  configuration this backend benchmarks -- see DESIGN.md Sec 3 for why the
  2-GPU (2+2) design is nonetheless the recommended production default over
  the 3-GPU (2+1+1) one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

TRAIN_SIZE = 790
EPOCHS = 20
GENERATIONS = 4
GROUPS_PER_UPDATE = 4
TOTAL_GROUPS = TRAIN_SIZE * EPOCHS
TOTAL_OPTIMIZER_STEPS = TOTAL_GROUPS // GROUPS_PER_UPDATE
assert TOTAL_GROUPS % GROUPS_PER_UPDATE == 0, "15,800 groups must divide evenly into 4-group windows"


def epoch_order(epoch: int, *, n: int = TRAIN_SIZE, seed: int = 42) -> List[int]:
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order


@dataclass(frozen=True)
class GroupSpec:
    global_group_index: int
    epoch: int
    index_in_epoch: int
    manifest_index: int
    sample_seed: int


def resolve_group(global_group_index: int, *, n: int = TRAIN_SIZE, seed: int = 42) -> GroupSpec:
    if global_group_index < 0:
        raise ValueError("global_group_index must be >= 0")
    epoch, index_in_epoch = divmod(global_group_index, n)
    manifest_index = epoch_order(epoch, n=n, seed=seed)[index_in_epoch]
    sample_seed = seed + epoch * n + manifest_index
    return GroupSpec(
        global_group_index=global_group_index,
        epoch=epoch,
        index_in_epoch=index_in_epoch,
        manifest_index=manifest_index,
        sample_seed=sample_seed,
    )


def rollout_seeds(sample_seed: int, generations: int = GENERATIONS) -> List[int]:
    return [int(sample_seed) * generations + j for j in range(generations)]


def window_index_and_slot(global_group_index: int, *, groups_per_update: int = GROUPS_PER_UPDATE) -> "tuple[int, int]":
    return divmod(global_group_index, groups_per_update)


def worker_group_indices(
    window_index: int,
    *,
    world_size: int,
    rank: int,
    groups_per_update: int = GROUPS_PER_UPDATE,
) -> List[int]:
    """Partition one 4-group window across `world_size` ranks, as evenly as
    possible, while always summing to exactly `groups_per_update` total.

    For world_size in {1, 2, 4} this is an exact even split (4, 2+2, 1+1+1+1).
    For world_size=3 (2-GPU-clean designs don't use a 3rd GPU; this is for
    the 3-GPU scaling-curve benchmark) it is 2+1+1: still exactly 4 groups
    globally, still exactly correct under `run_accumulation_window`'s
    global-divisor + SUM-allreduce reduction (see distributed.py), just with
    one rank doing twice the local work of the other two.
    """
    if world_size < 1 or world_size > groups_per_update:
        raise ValueError(f"world_size must be in [1, {groups_per_update}]")
    if not (0 <= rank < world_size):
        raise ValueError("rank out of range")
    base_count, remainder = divmod(groups_per_update, world_size)
    counts = [base_count + (1 if r < remainder else 0) for r in range(world_size)]
    start = window_index * groups_per_update + sum(counts[:rank])
    return [start + i for i in range(counts[rank])]


def resume_cursor_for_optimizer_step(optimizer_step: int, *, groups_per_update: int = GROUPS_PER_UPDATE) -> int:
    """First global_group_index of the not-yet-started window after `optimizer_step` completed steps."""
    return optimizer_step * groups_per_update
