#!/usr/bin/env python3
"""No-GPU correctness gate: multi-rank curriculum-scheduler synchronization.

Verifies `distributed.sync_iou_threshold_across_ranks` against a single-
process reference for a fixed, synthetic sequence of group IoUs, across
world_size in {1, 2, 4}: the resulting tau *after each window* must be
identical regardless of world_size (this backend's world-size-invariance
property -- see iou_threshold_scheduler.py's module docstring), and every
rank in a multi-GPU run must end a window holding the exact same tau.

Uses the CPU-only "gloo" backend (no GPU needed) via torch.multiprocessing.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import distributed as dist_mod  # noqa: E402
from production_grpo_fast import sampler  # noqa: E402
from production_grpo_fast.iou_threshold_scheduler import (  # noqa: E402
    IoUThresholdScheduler,
    SchedulerConfig,
)

# 3 windows * GROUPS_PER_UPDATE(4) = 12 groups, each with sampler.GENERATIONS(4)
# synthetic IoUs. Window 2's groups are crafted to clear tau0=0.3 comfortably
# (mean ~0.65, std 0 across the window) so a raise is exercised within this
# short test, not just the degenerate "never raises" path.
SYNTHETIC_GROUP_IOUS: List[List[float]] = [
    [0.05, 0.10, 0.02, 0.08],  # window 0, group 0 (global 0)
    [0.06, 0.09, 0.04, 0.07],  # window 0, group 1 (global 1)
    [0.03, 0.11, 0.05, 0.06],  # window 0, group 2 (global 2)
    [0.04, 0.08, 0.03, 0.09],  # window 0, group 3 (global 3)
    [0.65, 0.66, 0.64, 0.65],  # window 1, group 0 (global 4)
    [0.65, 0.66, 0.64, 0.65],  # window 1, group 1 (global 5)
    [0.65, 0.66, 0.64, 0.65],  # window 1, group 2 (global 6)
    [0.65, 0.66, 0.64, 0.65],  # window 1, group 3 (global 7)
    [0.50, 0.55, 0.52, 0.53],  # window 2, group 0 (global 8)
    [0.50, 0.55, 0.52, 0.53],  # window 2, group 1 (global 9)
    [0.50, 0.55, 0.52, 0.53],  # window 2, group 2 (global 10)
    [0.50, 0.55, 0.52, 0.53],  # window 2, group 3 (global 11)
]
N_WINDOWS = 3
TEST_CONFIG = SchedulerConfig(
    enabled=True, tau0=0.3, tau_target=0.8, window_size=4,
    delta_min_margin=0.10, decay="piecewise", min_window_fill=4,
)


def reference_tau_sequence() -> List[float]:
    """Single-process ground truth: feed all 12 groups through one scheduler
    in true global order, recording tau *after* each window of 4 groups."""
    sched = IoUThresholdScheduler(TEST_CONFIG)
    taus_after_window = []
    for w in range(N_WINDOWS):
        for g in range(sampler.GROUPS_PER_UPDATE):
            sched.step_group(SYNTHETIC_GROUP_IOUS[w * sampler.GROUPS_PER_UPDATE + g])
        taus_after_window.append(sched.tau)
    return taus_after_window


def _worker(rank: int, world_size: int, tmp_dir: str) -> None:
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29777"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    scheduler = IoUThresholdScheduler(TEST_CONFIG) if rank == 0 else None
    taus_after_window = []
    for window_index in range(N_WINDOWS):
        local_indices = sampler.worker_group_indices(window_index, world_size=world_size, rank=rank)
        local_group_ious = [
            SYNTHETIC_GROUP_IOUS[i] for i in local_indices
        ]
        tau = dist_mod.sync_iou_threshold_across_ranks(
            scheduler,
            local_group_ious=local_group_ious,
            window_index=window_index,
            world_size=world_size,
            rank=rank,
            worker_group_indices_fn=sampler.worker_group_indices,
        )
        taus_after_window.append(tau)

    torch.save(taus_after_window, Path(tmp_dir) / f"tau_rank{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def run_world_size(world_size: int, tmp_dir: str) -> None:
    mp.spawn(_worker, args=(world_size, tmp_dir), nprocs=world_size, join=True)


def main() -> None:
    reference = reference_tau_sequence()
    print(f"PASS reference tau sequence (single scheduler, 12 groups): {reference}")
    assert reference[0] == TEST_CONFIG.tau0, "window 0 (all-low IoUs) must not raise tau"
    assert reference[1] > reference[0], "window 1 (all-high IoUs) must raise tau"

    for world_size in (1, 2, 4):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_world_size(world_size, tmp_dir)
            per_rank_taus = [
                torch.load(Path(tmp_dir) / f"tau_rank{r}.pt") for r in range(world_size)
            ]
            for r, taus in enumerate(per_rank_taus):
                assert taus == reference, (
                    f"world_size={world_size} rank={r} tau sequence {taus} != "
                    f"single-process reference {reference}"
                )
            for r in range(1, world_size):
                assert per_rank_taus[r] == per_rank_taus[0], (
                    f"world_size={world_size}: rank {r} and rank 0 disagree on tau "
                    f"({per_rank_taus[r]} vs {per_rank_taus[0]})"
                )
            print(f"PASS world_size={world_size}: every rank's tau sequence matches the single-process reference")

    print("ALL IOU-THRESHOLD-SCHEDULER SYNC TESTS PASSED")


if __name__ == "__main__":
    main()
