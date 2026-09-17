#!/usr/bin/env python3
"""Helper process for test_multi_gpu_gradient_reduction.py.

Two modes, always operating on the same fixed 4-group window (global group
indices 0..3):

  --mode reference   single process, all 4 groups locally, one accumulation
                      window, captures LoRA grads before the optimizer step.
  --mode worker      one rank of a distributed (world_size=2 or 3) run that
                      loads the *same* initial LoRA state as the reference
                      (so both start from an identical point), runs this
                      rank's shard of the same 4-group window, all-reduces,
                      and (rank 0 only) captures the resulting LoRA grads.

Writes `{out}/init_lora.pt` (reference mode) and `{out}/grads_<label>.pt`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

from production_grpo_fast import distributed as dist_mod  # noqa: E402
from production_grpo_fast import grpo_step, runtime as prod_runtime, sampler  # noqa: E402
from production_grpo_fast.rewards_adapter import build_reward_pipeline  # noqa: E402
from rl.rewards import MedCLIPSemanticScorer  # noqa: E402
from train import load_train80_pairs, seed_all  # noqa: E402

WINDOW_INDEX = 0


def _build(device: torch.device):
    seed_all(42)
    model, tokenizer, processor, _ = prod_runtime.build_model_and_tokenizer(device)
    prod_runtime.trainability_audit(model)
    optimizer = prod_runtime.build_optimizer(model, lr=1e-6, weight_decay=0.0)
    reward_pipeline = build_reward_pipeline(MedCLIPSemanticScorer(device=device))
    return model, tokenizer, processor, optimizer, reward_pipeline


def _lora_state(model) -> dict:
    from peft import get_peft_model_state_dict

    return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model.language_model).items()}


def _load_lora_state(model, state: dict) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(model.language_model, {k: v.to(dtype=torch.bfloat16) for k, v in state.items()})


def run_reference(device: torch.device, out: Path) -> None:
    model, tokenizer, processor, optimizer, reward_pipeline = _build(device)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(_lora_state(model), out / "init_lora.pt")

    pairs = load_train80_pairs()
    local_indices = sampler.worker_group_indices(WINDOW_INDEX, world_size=1, rank=0)
    specs = [sampler.resolve_group(i) for i in local_indices]
    result = grpo_step.run_accumulation_window(
        model, optimizer, tokenizer, processor, reward_pipeline,
        [pairs[s.manifest_index] for s in specs], device=device,
        sample_seeds=[s.sample_seed for s in specs],
        capture_grads_and_skip_step=True,
    )
    torch.save(result["grads"], out / "grads_reference.pt")
    print("REFERENCE_DONE", {k: float(v.norm()) for k, v in list(result["grads"].items())[:2]})


def run_worker(rank: int, world_size: int, physical_gpus, out: Path) -> None:
    device = dist_mod.init_process_group(rank, world_size, physical_gpus, master_port=29522)
    model, tokenizer, processor, optimizer, reward_pipeline = _build(device)
    init_state = torch.load(out / "init_lora.pt", map_location="cpu")
    _load_lora_state(model, init_state)
    dist_mod.barrier()

    pairs = load_train80_pairs()
    local_indices = sampler.worker_group_indices(WINDOW_INDEX, world_size=world_size, rank=rank)
    specs = [sampler.resolve_group(i) for i in local_indices]
    result = grpo_step.run_accumulation_window(
        model, optimizer, tokenizer, processor, reward_pipeline,
        [pairs[s.manifest_index] for s in specs], device=device,
        sample_seeds=[s.sample_seed for s in specs],
        allreduce_fn=dist_mod.allreduce_sum_grads,
        capture_grads_and_skip_step=True,
    )
    if rank == 0:
        torch.save(result["grads"], out / f"grads_worker_w{world_size}.pt")
        print("WORKER_DONE", world_size)
    dist_mod.barrier()
    dist_mod.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["reference", "worker"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="1,2")
    args = parser.parse_args()
    out = Path(args.out)
    physical_gpus = [int(x) for x in args.gpus.split(",")]
    dist_mod.refuse_gpu0(physical_gpus)

    if args.mode == "reference":
        run_reference(torch.device(f"cuda:{physical_gpus[0]}"), out)
        return

    world_size = len(physical_gpus)
    torch.multiprocessing.spawn(
        run_worker,
        args=(world_size, physical_gpus, out),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
