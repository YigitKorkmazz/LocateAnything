#!/usr/bin/env python3
"""Warm-start variant of production_grpo_fast/train.py.

Isolation contract
-------------------
This file imports `production_grpo_fast` **read-only** and does not modify
any file under that package. It exists as a *separate* entrypoint so that a
second, concurrent GRPO run (this one) can be developed/iterated on without
any risk to another already-running `production_grpo_fast/train.py` process
(possibly on a different host, sharing this NFS mount) that depends on that
package's files staying byte-for-byte unchanged for its own future
`--resume auto`. See `evaluation/chestxray8/production_grpo_fast/README.md`
and `DESIGN.md` for the underlying GRPO/reward/checkpoint design this reuses
unmodified.

What's different from `production_grpo_fast/train.py`
--------------------------------------------------------
Exactly one behavior is added: `--init-adapter PATH`. When given, and only
on a fresh start (no existing checkpoint found for --resume in this run's
own --output-dir), the freshly-constructed LoRA adapter's weights are
overwritten with the weights at PATH (a directory containing
`adapter_model.safetensors`, or a direct path to that file) via
`peft.set_peft_model_state_dict` -- the exact same mechanism
`production_grpo_fast/checkpoint.py`'s own `resume_checkpoint` uses,
applied here to warm-start from a prior *supervised* fine-tune instead of
resuming this run's own GRPO state. Everything else (reward pipeline,
KL-to-reference regularization, checkpoint cadence/format, distributed
semantics, sampler/data-manifest contract) is the unmodified
`production_grpo_fast` code, imported directly.

Second addition: `--spatial-reward-mode bbox_aux` (see `bbox_aux_reward.py`):
total = format + semantic + R_iou + R_align + R_scale, where R_align/R_scale
are MedReasoner's corner-alignment (Eq. 3) and scale/aspect (Eq. 4) terms.
It is wired in by rebinding `production_grpo_fast.grpo_step.score_group` at
the start of every worker process (no package file is modified); the
`binary`/`continuous` modes delegate to the original `score_group`
unchanged. Per-window logs additionally carry per-rollout IoU (all modes)
and per-rollout R_align/R_scale/align_d/scale_d/pred_area plus per-group
gt_area (bbox_aux only). `--spatial-reward-mode bbox_aux_align` is the same
but without R_scale in the reward (total = format + semantic + R_iou +
R_align); R_scale/scale_d are still logged for it as diagnostics.

Verified warm-start checkpoint for this project: results/finetuning/
lora_direct_v1/checkpoint-450/adapter -- LoRA rank=8/alpha=16/dropout=0.05
and the same 7 target modules (504 tensors) as
`production_grpo_fast.runtime`'s LORA_RANK/LORA_ALPHA/LORA_DROPOUT, trained
with prompt_strategy="direct_disease" which matches (byte-for-byte, same
SHA256) the `splits/train80_pairs_seed42.jsonl` manifest this backend
trains on -- unlike `results/finetuning/lora` / `lora_v2`, which used
prompt_strategy="radiology_context" instead, or the `lora_projector_*`
variants, which (unlike this backend) also trained the projector.

Usage
-----
    python train_warmstart.py --output-dir OUT --gpus 1 \\
        --init-adapter /path/to/results/finetuning/lora_direct_v1/checkpoint-450/adapter \\
        --spatial-reward-mode continuous --kl-beta 0.04 --resume auto
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

# Must be set before `import torch` triggers any CUDA allocator init -- see
# production_grpo_fast/train.py's identical comment for the OOM this avoids.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# This file lives one directory deeper than production_grpo_fast/train.py
# would (production_grpo_fast_warmstart/ is a sibling of production_grpo_fast/,
# both directly under chestxray8/), so parents[1]/parents[3] resolve to the
# same CHEST_DIR/REPO_ROOT as the original -- `import production_grpo_fast`
# below therefore finds the real, unmodified package on sys.path.
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

from production_grpo_fast_warmstart import bbox_aux_reward  # noqa: E402

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
        return None
    return {i for i in range(n_layers) if i % checkpoint_every == 0}


def epoch_of(last_processed_global_group_index: int) -> int:
    return last_processed_global_group_index // sampler.TRAIN_SIZE


def load_init_adapter(model: Any, adapter_path: Path, *, rank: int) -> None:
    """Overwrite the freshly-constructed LoRA adapter with SFT weights.

    Same loading primitive as `production_grpo_fast.checkpoint.resume_checkpoint`
    (`peft.set_peft_model_state_dict` from a `safetensors` file), just pointed
    at a prior supervised fine-tune's adapter instead of this run's own GRPO
    checkpoint. Only called on a genuinely fresh start (see call site).
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter_file = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
    if not adapter_file.is_file():
        raise FileNotFoundError(f"--init-adapter: no adapter_model.safetensors at {adapter_file}")
    if rank == 0:
        print(f"[warm-start] loading initial LoRA weights from {adapter_file}", flush=True)
    set_peft_model_state_dict(model.language_model, load_file(str(adapter_file)))


def worker_main(
    rank: int,
    world_size: int,
    physical_gpus: Sequence[int],
    args: argparse.Namespace,
) -> None:
    # Per-process (spawn re-imports modules, so the parent's state is not inherited).
    use_bbox_aux = args.spatial_reward_mode in bbox_aux_reward.BBOX_AUX_MODES
    if use_bbox_aux:
        bbox_aux_reward.install_bbox_aux_scoring()
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

    # Resolve *before* building the model: whether this is a genuinely fresh
    # start (no checkpoint of THIS run yet) decides whether --init-adapter
    # applies at all -- a resumed run must not re-clobber its own,
    # already-further-trained adapter with the original SFT weights.
    resume_target = None
    if args.resume:
        resume_target = ckpt_mod.latest_checkpoint(output_dir) if args.resume == "auto" else Path(args.resume)

    model, tokenizer, processor, load_report = prod_runtime.build_model_and_tokenizer(device)

    warm_started_from = None
    if args.init_adapter and resume_target is None:
        adapter_path = Path(args.init_adapter)
        load_init_adapter(model, adapter_path, rank=rank)
        warm_started_from = str(adapter_path)

    audit = prod_runtime.trainability_audit(model)
    if world_size > 1:
        dist_mod.broadcast_trainable_from_rank0(model)
        dist_mod.barrier()

    optimizer = prod_runtime.build_optimizer(model, lr=args.learning_rate, weight_decay=0.0)
    scheduler = prod_runtime.build_linear_schedule(
        optimizer, total_steps=args.total_optimizer_steps, warmup_steps=0
    )
    pairs = load_train80_pairs()
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))

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
    if args.resume and resume_target is not None:
        next_group = ckpt_mod.resume_checkpoint(resume_target, model, optimizer, scheduler)
        if world_size > 1:
            dist_mod.broadcast_trainable_from_rank0(model)
            dist_mod.barrier()
        if args.iou_sched_enabled:
            sidecar = resume_target / "iou_scheduler_state.json"
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
            json.dumps(
                {
                    "load_report": load_report,
                    "trainability_audit": audit,
                    "args": vars(args),
                    "warm_started_from": warm_started_from,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    checkpoint_layers = checkpoint_layer_set(checkpoint_every=args.checkpoint_every_layer)
    allreduce_fn = dist_mod.allreduce_sum_grads if world_size > 1 else None

    last_epoch_checkpointed = epoch_of(next_group - 1) if next_group > 0 else 0
    last_window_checkpointed = -1
    windows_run = 0
    total_groups_seen = 0
    benchmark_start = time.monotonic()

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
            spatial_reward_mode=args.spatial_reward_mode,
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
        group_stds = [float(np.std([float(r) for r in g["rewards"]])) for g in result["group_reports"]]
        log(
            {
                **bbox_aux_reward.window_bbox_metrics(result["group_reports"], bbox_aux=use_bbox_aux),
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
                # Population std (same normalization as group_relative_advantages) of each
                # prompt group's G rollout rewards; ~0 means the group gives no GRPO signal.
                "group_reward_stds": group_stds,
                "mean_group_reward_std": sum(group_stds) / max(1, len(group_stds)),
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
                    config={"args": vars(args), "trainability_audit": audit, "warm_started_from": warm_started_from},
                )
                if args.iou_sched_enabled:
                    (saved_to / "iou_scheduler_state.json").write_text(
                        json.dumps(scheduler_iou.state_dict(), indent=2, sort_keys=True), encoding="utf-8"
                    )
            if world_size > 1:
                dist_mod.barrier()
            last_epoch_checkpointed = new_epoch
            last_window_checkpointed = windows_run

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
            config={"args": vars(args), "trainability_audit": audit, "warm_started_from": warm_started_from, "final": True},
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
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Warm-start: path to a directory containing adapter_model.safetensors (or the "
        "file itself), loaded into the freshly-built LoRA adapter in place of its random "
        "init. Only applied on a fresh start (ignored if --resume finds an existing "
        "checkpoint in --output-dir, since that already reflects further-trained state).",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="stop after this many accumulation windows (benchmark/smoke only)")
    parser.add_argument(
        "--total-optimizer-steps",
        type=int,
        default=sampler.TOTAL_OPTIMIZER_STEPS,
        help=(
            "horizon for both the linear LR schedule and the training loop's stop "
            f"condition. Default {sampler.TOTAL_OPTIMIZER_STEPS} == the fixed 20-epoch/"
            "15,800-group contract."
        ),
    )
    parser.add_argument("--checkpoint-every-epoch", action="store_true", default=True)
    parser.add_argument("--no-checkpoint-every-epoch", dest="checkpoint_every_epoch", action="store_false")
    parser.add_argument(
        "--checkpoint-every-windows",
        type=int,
        default=25,
        help="also checkpoint every N accumulation windows (0 disables)",
    )
    parser.add_argument("--final-checkpoint", action="store_true", default=True)
    parser.add_argument("--checkpoint-every-layer", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--audit-interval", type=int, default=50)
    parser.add_argument("--master-port", type=int, default=29511)
    parser.add_argument("--log-all-ranks", action="store_true", default=True)
    parser.add_argument(
        "--spatial-reward-mode",
        choices=["binary", "continuous", *bbox_aux_reward.BBOX_AUX_MODES],
        default="binary",
    )
    parser.add_argument("--iou-sched-enabled", action="store_true", default=False)
    parser.add_argument("--iou-sched-tau0", type=float, default=0.3)
    parser.add_argument("--iou-sched-tau-target", type=float, default=0.8)
    parser.add_argument("--iou-sched-window", type=int, default=30)
    parser.add_argument("--iou-sched-delta-margin", type=float, default=0.10)
    parser.add_argument("--iou-sched-decay", choices=["piecewise", "linear", "cosine"], default="piecewise")
    parser.add_argument("--kl-beta", type=float, default=0.04)
    args = parser.parse_args(argv)
    if args.spatial_reward_mode in ("continuous", *bbox_aux_reward.BBOX_AUX_MODES) and args.iou_sched_enabled:
        raise ValueError(
            f"--spatial-reward-mode {args.spatial_reward_mode} and --iou-sched-enabled are mutually exclusive."
        )
    return args


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
