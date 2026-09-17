#!/usr/bin/env python3
"""Production GRPO training entrypoint for LocateAnything-3B on ChestX-ray8.

Usage (tcsh-safe; see README.md Sec "production launch command" for the
fully-worked example):

    python train.py --output-dir OUT --gpus 1,2 --checkpoint-every-epoch

No heldout access: this file (transitively, this whole package) never reads
the held-out final-evaluation manifest or the historical internal-validation
manifest; the only dataset manifest path referenced anywhere below is
`TRAIN80_PATH` (train80, patient-disjoint), integrity-checked by SHA256
before use. `test_no_heldout_access.py` statically enforces this.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Must be set before `import torch` triggers any CUDA allocator init.
# `torch.utils.checkpoint`'s repeated forward/backward recompute over
# variable-length sequences, sustained over hundreds of prompt groups,
# fragments the default CUDA caching allocator badly enough to OOM with
# several GiB still "reserved but unallocated" (observed in production:
# GPU2 OOM'd on a 20MB allocation after 39/3950 steps with 4.1GiB reserved-
# but-unallocated). `expandable_segments` is PyTorch's own recommended fix
# for exactly this allocator symptom. `setdefault` respects an explicit
# override from the launching shell.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

CHEST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (str(CHEST_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rl.rewards import MedCLIPSemanticScorer  # noqa: E402
from rl.runtime import load_verified_pairs  # noqa: E402

from production_grpo_fast import checkpoint as ckpt_mod  # noqa: E402
from production_grpo_fast import distributed as dist_mod  # noqa: E402
from production_grpo_fast import grpo_step  # noqa: E402
from production_grpo_fast import iou_threshold_scheduler as iou_sched_mod  # noqa: E402
from production_grpo_fast import runtime as prod_runtime  # noqa: E402
from production_grpo_fast import sampler  # noqa: E402
from production_grpo_fast.rewards_adapter import build_reward_pipeline  # noqa: E402

TRAIN80_PATH = "splits/train80_pairs_seed42.jsonl"
TRAIN80_SHA256 = "a2b1c25f652f90e8d93e58b33ee74aa5eb09db4df7c6b37e760f5e5238c2e3b0"


def load_train80_pairs() -> List[Dict[str, Any]]:
    config = {"data": {"train_split": TRAIN80_PATH, "train_sha256": TRAIN80_SHA256}}
    pairs = load_verified_pairs(config, "train")
    if len(pairs) != sampler.TRAIN_SIZE:
        raise RuntimeError(f"expected {sampler.TRAIN_SIZE} train80 examples, found {len(pairs)}")
    return pairs


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def checkpoint_layer_set(n_layers: int = 36, checkpoint_every: int = 1) -> Optional[set]:
    if checkpoint_every <= 1:
        return None  # None => checkpoint all layers (safest / default)
    return {i for i in range(n_layers) if i % checkpoint_every == 0}


def epoch_of(last_processed_global_group_index: int) -> int:
    return last_processed_global_group_index // sampler.TRAIN_SIZE


def worker_main(
    rank: int,
    world_size: int,
    physical_gpus: Sequence[int],
    args: argparse.Namespace,
) -> None:
    if world_size > 1:
        device = dist_mod.init_process_group(rank, world_size, physical_gpus, master_port=args.master_port)
    else:
        physical_gpu = int(physical_gpus[0])
        dist_mod.refuse_gpu0(physical_gpus)
        device = torch.device(f"cuda:{physical_gpu}")
        torch.cuda.set_device(device)

    seed_all(42)
    output_dir = Path(args.output_dir)
    log_path = output_dir / f"train_log_rank{rank}.jsonl"

    def log(record: Dict[str, Any]) -> None:
        if rank != 0 and not args.log_all_ranks:
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    model, tokenizer, processor, load_report = prod_runtime.build_model_and_tokenizer(device)
    audit = prod_runtime.trainability_audit(model)
    if world_size > 1:
        dist_mod.broadcast_trainable_from_rank0(model)
        dist_mod.barrier()

    optimizer = prod_runtime.build_optimizer(model, lr=args.learning_rate, weight_decay=0.0)
    # Overridable so a later "extend the horizon" run (e.g. 3950 -> 5000
    # steps) can re-derive the linear-decay curve against the new target
    # instead of resuming a schedule that already reached LR=0 at the
    # original total_steps and would stay pinned at 0 forever otherwise.
    # `scheduler.load_state_dict` (in checkpoint.resume_checkpoint) only
    # restores `last_epoch`/step-count onto whatever lambda this call built,
    # so building it with the new total_steps here is what performs the
    # "migration" -- nothing about the already-completed steps changes.
    scheduler = prod_runtime.build_linear_schedule(
        optimizer, total_steps=args.total_optimizer_steps, warmup_steps=0
    )
    pairs = load_train80_pairs()
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))

    # MedLoc-R1 curriculum IoU-threshold scheduler (optional; off by default,
    # in which case reward_pipeline.iou_threshold stays at its fixed 0.5 and
    # nothing below in this block does anything). Only rank 0 owns the
    # stateful scheduler instance -- see distributed.sync_iou_threshold_
    # across_ranks and iou_threshold_scheduler.py's module docstring for why
    # a per-rank scheduler would be wrong here. `current_tau` is held by
    # every rank (it is just a float, needed to set reward_pipeline.
    # iou_threshold locally before each window).
    iou_sched_config = iou_sched_mod.SchedulerConfig(
        enabled=args.iou_sched_enabled,
        tau0=args.iou_sched_tau0,
        tau_target=args.iou_sched_tau_target,
        window_size=args.iou_sched_window,
        delta_min_margin=args.iou_sched_delta_margin,
        decay=args.iou_sched_decay,
        log_path=str(output_dir / f"iou_sched_log_rank{rank}.txt") if args.iou_sched_enabled else None,
    )
    scheduler_iou = (
        iou_sched_mod.IoUThresholdScheduler(iou_sched_config)
        if (args.iou_sched_enabled and rank == 0)
        else None
    )
    current_tau = float(args.iou_sched_tau0)

    next_group = 0
    if args.resume:
        target = ckpt_mod.latest_checkpoint(output_dir) if args.resume == "auto" else Path(args.resume)
        if target is not None:
            next_group = ckpt_mod.resume_checkpoint(target, model, optimizer, scheduler)
            if world_size > 1:
                dist_mod.broadcast_trainable_from_rank0(model)
                dist_mod.barrier()
            if args.iou_sched_enabled:
                sidecar = target / "iou_scheduler_state.json"
                if sidecar.is_file():
                    state = json.loads(sidecar.read_text(encoding="utf-8"))
                    current_tau = float(state["tau"])
                    if scheduler_iou is not None:
                        scheduler_iou.load_state_dict(state)
                elif rank == 0:
                    print(
                        f"[iou-sched] WARNING: {sidecar} not found; resuming curriculum "
                        f"from tau0={current_tau} instead of the checkpoint's actual tau"
                    )
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.json").write_text(
            json.dumps({"load_report": load_report, "trainability_audit": audit, "args": vars(args)}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    checkpoint_layers = checkpoint_layer_set(checkpoint_every=args.checkpoint_every_layer)
    allreduce_fn = dist_mod.allreduce_sum_grads if world_size > 1 else None

    # Fresh run: epoch 0 is in progress, not yet completed -- must not trigger
    # a checkpoint until it actually finishes (epoch_of(...) transitions from
    # 0 to 1), so the sentinel is 0, not -1. A -1 sentinel fired a spurious
    # checkpoint after the very first window, since epoch_of(...) == 0 > -1.
    last_epoch_checkpointed = epoch_of(next_group - 1) if next_group > 0 else 0
    last_window_checkpointed = -1
    windows_run = 0
    total_groups_seen = 0
    benchmark_start = time.monotonic()

    # Normally == sampler.TOTAL_GROUPS (15,800); an extension run passes a
    # larger --total-optimizer-steps and this naturally runs past the
    # original 20-epoch/790-example exposure count into epoch 21, 22, ...
    # (sampler.resolve_group/epoch_order are defined for any epoch >= 0).
    total_groups_for_this_run = args.total_optimizer_steps * sampler.GROUPS_PER_UPDATE
    while next_group < total_groups_for_this_run:
        if args.max_windows is not None and windows_run >= args.max_windows:
            break
        window_index, slot = sampler.window_index_and_slot(next_group)
        if slot != 0:
            raise RuntimeError("resume cursor is not aligned to a window boundary")
        local_indices = sampler.worker_group_indices(window_index, world_size=world_size, rank=rank)
        specs = [sampler.resolve_group(i) for i in local_indices]
        pairs_local = [pairs[s.manifest_index] for s in specs]
        seeds_local = [s.sample_seed for s in specs]

        if args.iou_sched_enabled:
            reward_pipeline.iou_threshold = current_tau

        step_start = time.monotonic()
        result = grpo_step.run_accumulation_window(
            model,
            optimizer,
            tokenizer,
            processor,
            reward_pipeline,
            pairs_local,
            device=device,
            sample_seeds=seeds_local,
            checkpoint_layers=checkpoint_layers,
            allreduce_fn=allreduce_fn,
            max_grad_norm=args.max_grad_norm,
            audit=(windows_run == 0 or (args.audit_interval > 0 and windows_run % args.audit_interval == 0)),
            kl_beta=args.kl_beta,
        )
        scheduler.step()
        step_elapsed = time.monotonic() - step_start

        tau_used_this_window = current_tau
        if args.iou_sched_enabled:
            current_tau = dist_mod.sync_iou_threshold_across_ranks(
                scheduler_iou,
                local_group_ious=result["group_ious"],
                window_index=window_index,
                world_size=world_size,
                rank=rank,
                worker_group_indices_fn=sampler.worker_group_indices,
            )

        windows_run += 1
        total_groups_seen += len(local_indices)
        next_group = (window_index + 1) * sampler.GROUPS_PER_UPDATE
        optimizer_step = next_group // sampler.GROUPS_PER_UPDATE

        elapsed_total = time.monotonic() - benchmark_start
        log(
            {
                "rank": rank,
                "world_size": world_size,
                "window_index": window_index,
                "optimizer_step": optimizer_step,
                "local_group_count": len(local_indices),
                "epoch_of_last_group": epoch_of(next_group - 1),
                "step_seconds": step_elapsed,
                "elapsed_seconds": elapsed_total,
                "local_groups_per_hour": 3600.0 * total_groups_seen / elapsed_total if elapsed_total else None,
                "mean_reward": sum(r for g in result["group_reports"] for r in g["rewards"]) / max(1, sum(len(g["rewards"]) for g in result["group_reports"])),
                "gradient_audit": result["gradient_audit"],
                "iou_sched_tau_used": tau_used_this_window if args.iou_sched_enabled else None,
                "iou_sched_tau_next": current_tau if args.iou_sched_enabled else None,
                "kl_beta": args.kl_beta,
                "mean_kl_penalty": (
                    sum(r["kl_penalty"] for g in result["group_reports"] for r in g["rollout_reports"] if "kl_penalty" in r)
                    / max(1, sum(1 for g in result["group_reports"] for r in g["rollout_reports"] if "kl_penalty" in r))
                    if args.kl_beta > 0.0
                    else None
                ),
            }
        )

        new_epoch = epoch_of(next_group - 1)
        epoch_boundary = args.checkpoint_every_epoch and new_epoch > last_epoch_checkpointed
        # Epoch boundaries are ~198 windows apart (~4-5 hours); a long-run
        # CUDA allocator fragmentation OOM (observed in production: crashed
        # after 39/3950 steps) can lose most of that if it's the only
        # checkpoint cadence. Also checkpoint every N windows unconditionally.
        interval_boundary = (
            args.checkpoint_every_windows > 0
            and windows_run % args.checkpoint_every_windows == 0
            and windows_run != last_window_checkpointed
        )
        if epoch_boundary or interval_boundary:
            if world_size > 1:
                dist_mod.barrier()
            if rank == 0:
                saved_to = ckpt_mod.save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    next_global_group_index=next_group,
                    config={"args": vars(args), "trainability_audit": audit},
                )
                if args.iou_sched_enabled:
                    (saved_to / "iou_scheduler_state.json").write_text(
                        json.dumps(scheduler_iou.state_dict(), indent=2, sort_keys=True), encoding="utf-8"
                    )
            if world_size > 1:
                dist_mod.barrier()
            last_epoch_checkpointed = new_epoch
            last_window_checkpointed = windows_run

    # Guard against a double-save at the same optimizer_step: when the run's
    # total window count happens to be a multiple of --checkpoint-every-windows
    # (or lands on an epoch boundary), the in-loop checkpoint above already
    # wrote this exact target, and save_checkpoint refuses to overwrite it.
    final_target = output_dir / f"checkpoint_step_{(next_group // sampler.GROUPS_PER_UPDATE):06d}"
    if (
        rank == 0
        and args.final_checkpoint
        and next_group % sampler.GROUPS_PER_UPDATE == 0
        and next_group > 0
        and not final_target.exists()
    ):
        saved_to = ckpt_mod.save_checkpoint(
            output_dir,
            model,
            optimizer,
            scheduler,
            next_global_group_index=next_group,
            config={"args": vars(args), "trainability_audit": audit, "final": True},
        )
        if args.iou_sched_enabled:
            (saved_to / "iou_scheduler_state.json").write_text(
                json.dumps(scheduler_iou.state_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )

    if world_size > 1:
        dist_mod.barrier()
        dist_mod.destroy_process_group()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="1,2", help="comma-separated physical GPU indices, e.g. 1,2 or 1,2,3")
    parser.add_argument("--resume", default=None, help="'auto' to resume the latest checkpoint in --output-dir, or an explicit checkpoint path")
    parser.add_argument("--max-windows", type=int, default=None, help="stop after this many accumulation windows (benchmark/smoke only)")
    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        default=sampler.TOTAL_OPTIMIZER_STEPS,
        help=(
            "horizon for both the linear LR schedule and the training loop's stop "
            f"condition. Default {sampler.TOTAL_OPTIMIZER_STEPS} == the fixed 20-epoch/"
            "15,800-group contract. Pass a larger value only for an explicit, deliberate "
            "extension run resuming from that run's final checkpoint -- this re-derives "
            "the linear-decay curve against the new target rather than resuming a "
            "schedule already pinned at LR=0."
        ),
    )
    parser.add_argument("--checkpoint-every-epoch", action="store_true", default=True)
    parser.add_argument("--no-checkpoint-every-epoch", dest="checkpoint_every_epoch", action="store_false")
    parser.add_argument(
        "--checkpoint-every-windows",
        type=int,
        default=25,
        help="also checkpoint every N accumulation windows (0 disables); epoch boundaries are "
        "~198 windows apart, too sparse to bound the loss from a long-run OOM/crash on their own",
    )
    parser.add_argument("--final-checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint-every-layer", type=int, default=1, help="1 = checkpoint all 36 decoder layers (default/safe); >1 = selective (see replay.py)")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--audit-interval", type=int, default=50)
    parser.add_argument("--master-port", type=int, default=29511)
    parser.add_argument("--log-all-ranks", action="store_true", default=True)

    # --- Continuous/curriculum spatial reward (MedLoc-R1, arXiv:2603.28120) ---
    # Off by default: reward_pipeline.iou_threshold then stays fixed at 0.5
    # (paper-faithful MedGround-R1 spatial reward), reproducing prior runs
    # exactly. See iou_threshold_scheduler.py for the mechanism and
    # distributed.sync_iou_threshold_across_ranks for the multi-GPU design.
    parser.add_argument("--iou-sched-enabled", action="store_true", default=False)
    parser.add_argument("--iou-sched-tau0", type=float, default=0.3)
    parser.add_argument("--iou-sched-tau-target", type=float, default=0.8)
    parser.add_argument("--iou-sched-window", type=int, default=30)
    parser.add_argument("--iou-sched-delta-margin", type=float, default=0.10)
    parser.add_argument("--iou-sched-decay", choices=["piecewise", "linear", "cosine"], default="piecewise")

    # --- KL penalty toward a frozen reference policy (base weights, LoRA
    # adapter disabled via PEFT's disable_adapter() -- no second model copy).
    # Default 0.04 matches MedGround-R1's own setting (beta=0.04, Sec 3
    # Implementation Details); this backend's earlier beta=0 run regressed
    # BELOW the zero-shot baseline (mean IoU 0.067 vs 0.129 zero-shot, see
    # heldout194_eval_step005000.json), and MedGround-R1 explicitly motivates
    # its KL term as "to preserve the original pretrained knowledge" (Sec
    # 2.1) -- exactly the failure mode observed. Pass --kl-beta 0.0 to
    # reproduce the old (no-KL) behavior exactly (skips the extra reference-
    # policy forward pass entirely).
    parser.add_argument("--kl-beta", type=float, default=0.04)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    physical_gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    dist_mod.refuse_gpu0(physical_gpus)
    world_size = len(physical_gpus)
    if world_size == 1:
        worker_main(0, 1, physical_gpus, args)
        return
    torch.multiprocessing.spawn(
        _spawn_entry,
        args=(world_size, physical_gpus, args),
        nprocs=world_size,
        join=True,
    )


def _spawn_entry(rank: int, world_size: int, physical_gpus: Sequence[int], args: argparse.Namespace) -> None:
    worker_main(rank, world_size, physical_gpus, args)


if __name__ == "__main__":
    main()
