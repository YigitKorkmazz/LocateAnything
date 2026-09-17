"""Synchronous multi-GPU data parallelism over independent prompt groups.

Each rank holds a full model replica (base weights frozen + identical LoRA
init, guaranteed by `broadcast_trainable_from_rank0`, not just by matching
RNG seeds across processes) and runs its own shard of a 4-group accumulation
window (see `sampler.worker_group_indices`). After every rank finishes its
local `.backward()` calls, `allreduce_sum_grads` SUM-reduces only the ~15M
LoRA-parameter gradients (there is nothing else to synchronize: activations
never cross ranks, so this is deliberately not `DistributedDataParallel`).
SUM is correct, not MEAN, because each rollout's loss was already divided by
the *global* `generations * groups_per_update` before `.backward()`
(see `grpo_step.run_accumulation_window`), so summing the already-normalized
local gradients reconstructs the exact global average regardless of how many
groups each rank ran.

GPU 0 was excluded by default in the original design because another user's
job was running on it during initial development (see DESIGN.md Sec 3) --
not because of any technical incompatibility. With explicit permission to
use all 4 physical GPUs, `ALLOWED_PHYSICAL_GPUS` below includes 0; the check
still exists to catch obviously-wrong indices (e.g. a typo'd GPU 4+).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

import torch
import torch.distributed as dist

ALLOWED_PHYSICAL_GPUS = (0, 1, 2, 3)


def refuse_gpu0(physical_gpus: Sequence[int]) -> None:
    for g in physical_gpus:
        if int(g) not in ALLOWED_PHYSICAL_GPUS:
            raise RuntimeError(f"physical GPU {g} is not in the allowed set {ALLOWED_PHYSICAL_GPUS}")


def init_process_group(
    rank: int,
    world_size: int,
    physical_gpus: Sequence[int],
    *,
    master_port: int = 29511,
) -> torch.device:
    refuse_gpu0(physical_gpus)
    if world_size != len(physical_gpus):
        raise ValueError("world_size must equal len(physical_gpus)")
    physical_gpu = int(physical_gpus[rank])
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    torch.cuda.set_device(physical_gpu)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return torch.device(f"cuda:{physical_gpu}")


def destroy_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_trainable_from_rank0(model: Any) -> None:
    """Force every rank to start from rank 0's exact LoRA-adapter weights.

    Relying on identical RNG seeds across independently-spawned processes to
    reproduce identical PEFT LoRA-A random init is fragile; broadcasting is
    the standard, bulletproof guarantee used by ordinary DDP initialization.
    """
    for p in model.parameters():
        if p.requires_grad:
            dist.broadcast(p.data, src=0)


def allreduce_sum_grads(model: Any) -> None:
    handles = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
    for h in handles:
        h.wait()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def sync_iou_threshold_across_ranks(
    scheduler: Any,
    *,
    local_group_ious: List[List[float]],
    window_index: int,
    world_size: int,
    rank: int,
    worker_group_indices_fn: Any,
) -> float:
    """Advance the (rank-0-owned) curriculum scheduler by one window's worth
    of groups, in true global group-index order, and return the tau to use
    for the NEXT window. See iou_threshold_scheduler.py's module docstring
    for why this -- not a per-rank scheduler -- is the correct design.

    `local_group_ious[i]` is this rank's i-th local group's list of G IoUs,
    in the same order as `worker_group_indices_fn(window_index, world_size,
    rank)`. `scheduler` must be non-None on rank 0 and is ignored (may be
    None) on every other rank.
    """
    if world_size <= 1:
        for ious in local_group_ious:
            scheduler.step_group(ious)
        return float(scheduler.tau)

    gathered: List[Optional[List[List[float]]]] = [None] * world_size
    dist.all_gather_object(gathered, local_group_ious)

    # NCCL (this backend's process-group backend during real training) only
    # broadcasts CUDA tensors, unlike the CPU-only "gloo" backend this
    # function's own unit test (test_iou_threshold_scheduler_sync.py) uses --
    # a CPU tensor here raised "No backend type associated with device type
    # cpu" the first time this ran on real GPUs. `torch.cuda.set_device` was
    # already called for this rank by `init_process_group`, so plain "cuda"
    # resolves to the right physical GPU.
    tau_device = "cuda" if torch.cuda.is_available() else "cpu"
    tau_holder = torch.zeros(1, dtype=torch.float64, device=tau_device)
    if rank == 0:
        if scheduler is None:
            raise ValueError("rank 0 must own the scheduler instance")
        by_global_index = {}
        for r in range(world_size):
            local_indices = worker_group_indices_fn(window_index, world_size=world_size, rank=r)
            r_groups = gathered[r]
            if len(local_indices) != len(r_groups):
                raise RuntimeError(
                    f"rank {r} reported {len(r_groups)} groups but "
                    f"worker_group_indices assigned it {len(local_indices)}"
                )
            for global_index, ious in zip(local_indices, r_groups):
                by_global_index[global_index] = ious
        for global_index in sorted(by_global_index):
            scheduler.step_group(by_global_index[global_index])
        tau_holder[0] = float(scheduler.tau)

    dist.broadcast(tau_holder, src=0)
    return float(tau_holder.item())
