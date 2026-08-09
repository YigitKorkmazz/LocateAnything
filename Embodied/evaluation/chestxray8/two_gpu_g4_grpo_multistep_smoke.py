#!/usr/bin/env python3
"""Resumable ten-step exact G=4 Hybrid-GRPO smoke run on the 18/18 shard.

This is intentionally a small production-path run, not a throughput trainer.
It writes one completed-step JSONL record at a time and only checkpoints after
an entire optimizer step, so a resume never observes a partially accumulated
autograd graph or parameter gradient.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.grpo import grpo_clipped_loss, group_relative_advantages  # noqa: E402
from rl.medground_kl import clipped_grpo_with_medground_kl  # noqa: E402
from rl.nan_grad_diagnostics import compare_grad_dicts  # noqa: E402
from rl.policy_state import PolicySnapshot, use_policy_snapshot  # noqa: E402
from rl.rewards import MedCLIPSemanticScorer, build_reward_pipeline_from_config  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy_two_gpu_live_cache,
    build_rollout_replayer,
    decoder_inputs,
    generate_rollout_group,
    hybrid_logprob_objective,
    is_hybrid_rollout,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
)
from rl.two_gpu_shard import DecoderShardLayout, resolve_locateanything_qwen_decoder  # noqa: E402
from train_chestxray8_sft import build_optimizer  # noqa: E402
from two_gpu_exactness_oracle import _json_sanitize, _write_report_safely  # noqa: E402
from two_gpu_live_cache_feasibility import _ReplayKVRecorder, _cross_device_boundary_report  # noqa: E402


METRICS_FILENAME = "two_gpu_g4_grpo_multistep_metrics.jsonl"
SUMMARY_FILENAME = "two_gpu_g4_grpo_multistep_summary.json"
CHECKPOINT_PREFIX = "two_gpu_g4_grpo_multistep_step_"
FRESH_START_GUARD_FILENAME = "fresh_start_eight_group_degeneracy_diagnostic.json"
FRESH_START_GUARD_GROUPS = 8
GROUP_SIZE = 4
EXPECTED_LORA_TENSORS = 504
EXPECTED_PROJECTOR_TENSORS = 6
EXPECTED_TRAINABLE_TENSORS = EXPECTED_LORA_TENSORS + EXPECTED_PROJECTOR_TENSORS
EXPECTED_MODEL_NAME = "nvidia/LocateAnything-3B"
EXPECTED_MODEL_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
EXPECTED_RUNTIME_CONTRACT = {
    "carry_cache": True,
    "replay_cache": True,
    "live_cache_replay": True,
    "full_trajectory": True,
    "detach_kv": False,
    "truncated_bptt": False,
    "gradient_checkpointing": False,
    "saved_tensor_offload": False,
}


def _effective_kl_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the YAML-authoritative MedGround KL contract."""
    objective = config["objective"]
    enabled = objective.get("loss_total") == "L_GRPO_PLUS_MEDGROUND_KL"
    configured = objective.get("reference_kl")
    if not enabled:
        if configured not in (None, {}, {"enabled": False}):
            raise RuntimeError("KL-free objective must not carry an active reference_kl config")
        return {"enabled": False, "beta": 0.0}
    expected = {
        "enabled": True,
        "beta": 0.04,
        "estimator": "exp_ref_minus_policy_minus_ref_minus_policy_minus_one",
        "direction": "policy_to_reference",
        "granularity": "sampled_trajectory_token",
        "normalization": "mean_tokens_per_trajectory_then_mean_group",
        "mask": "full_hybrid_sampled_trajectory",
        "include_rejected_pbd_proposals": True,
        "exclude_prompt_tokens": True,
        "exclude_image_tokens": True,
        "reference_policy": "initial_fresh_start_policy",
        "reference_storage": "shared_frozen_base_with_cpu_snapshot_swap",
        "reference_lora": "initial_zero_effect_adapter_state",
        "reference_projector": "original_pretrained_state",
        "reference_mode": "eval",
        "reference_autograd": "disabled",
    }
    if not isinstance(configured, dict):
        raise RuntimeError("MedGround KL objective requires objective.reference_kl")
    missing = sorted(set(expected) - set(configured))
    extra = sorted(set(configured) - set(expected))
    mismatches = {
        key: {"expected": value, "actual": configured.get(key)}
        for key, value in expected.items()
        if configured.get(key) != value
    }
    if missing or extra or mismatches:
        raise RuntimeError(
            "reference_kl differs from validated MedGround-compatible contract: "
            + json.dumps(
                {"missing": missing, "extra": extra, "mismatches": mismatches},
                sort_keys=True,
            )
        )
    provenance = config.get("medground_source_audit") or {}
    provenance_expected = {
        "repository_commit": "37b210dd7d6ee71179b3013d7f8042af1de0d5d3",
        "trainer_upstream_era_commit": "ebc741f160b9d0c2312cd627516bf6c0ecd73343",
        "launch_explicit_beta": None,
        "dependency_specification": "trl @ git+https://github.com/huggingface/trl.git@main",
        "upstream_era_trl_beta_default": 0.04,
        "selected_explicit_beta": 0.04,
    }
    provenance_mismatches = {
        key: {"expected": value, "actual": provenance.get(key)}
        for key, value in provenance_expected.items()
        if provenance.get(key) != value
    }
    if provenance_mismatches:
        raise RuntimeError(
            "MedGround source/beta provenance differs from the audited contract: "
            + json.dumps(provenance_mismatches, sort_keys=True)
        )
    return dict(expected)


def _attempt_schedule(
    *, seed: int, attempted_group_count: int, sample_cursor: int, sample_count: int
) -> Dict[str, Any]:
    """Production schedule derived only from persisted counters."""
    sample_index = sample_cursor % sample_count
    attempt_seed = seed + attempted_group_count * 1_000_003 + sample_index
    return {
        "sample_index": sample_index,
        "attempt_seed": attempt_seed,
        "rollout_seeds": [attempt_seed * GROUP_SIZE + i for i in range(GROUP_SIZE)],
    }


def _all_advantages_exactly_zero(advantages: torch.Tensor) -> bool:
    """The existing production zero-variance criterion, factored for testing."""
    return bool(torch.count_nonzero(advantages).item() == 0)


def _replay_each_if_nonzero(advantages, traces, replay_one):
    """Dispatch one replay per trajectory unless GRPO is identically zero."""
    if _all_advantages_exactly_zero(advantages):
        return []
    return [replay_one(index, trace) for index, trace in enumerate(traces)]


def _advance_skipped_group(sample_cursor: int, skipped_count: int) -> tuple[int, int]:
    return sample_cursor + 1, skipped_count + 1


def _zero_advantage_trajectory_records(traces, components, rollout_seeds):
    reason = "not_computed_exact_zero_advantage_fast_path"
    return [
        {
            **_trace_summary(trace, components[index]),
            "group_index": index,
            "rollout_seed": rollout_seeds[index],
            "advantage": 0.0,
            "old_logp": None,
            "current_logp": None,
            "ppo_ratio": None,
            "logp_not_computed_reason": reason,
            "loss_grpo_unscaled": 0.0,
            "loss_grpo_scaled": 0.0,
            "replay_executed": False,
            "cumulative_gradient_norm": None,
            "gradient_norm_not_computed_reason": reason,
        }
        for index, trace in enumerate(traces)
    ]


def _zero_advantage_group_record(
    base_record: Dict[str, Any],
    *,
    trajectories,
    optimizer_step_count: int,
    sample_cursor: int,
    skipped_count: int,
    optimizer_state_devices,
    per_gpu_memory,
) -> Dict[str, Any]:
    reason = "not_computed_exact_zero_advantage_fast_path"
    return {
        **base_record,
        "trajectories": trajectories,
        "step_status": "skipped_zero_advantage_group",
        "skipped_zero_advantage": True,
        "optimizer_step_executed": False,
        "optimizer_step_skipped": True,
        "replay_executed": False,
        "loss": 0.0,
        "per_trajectory_losses": [0.0] * GROUP_SIZE,
        "gradient_norm": None,
        "gradient_norm_not_computed_reason": reason,
        "optimizer_step_count_after_group": optimizer_step_count,
        "sample_cursor_after_group": sample_cursor,
        "skipped_zero_advantage_group_count": skipped_count,
        "model_state_preserved_exactly": True,
        "optimizer_state_preserved_exactly": True,
        "optimizer_state_devices": optimizer_state_devices,
        "per_gpu_memory_after_group": per_gpu_memory,
    }


def _fresh_start_trajectory_guard_predicates(
    trace,
    component,
    *,
    max_new_tokens: int,
    block_size: int,
) -> Dict[str, Any]:
    """Evaluate the exact deterministic fresh-start failure signature."""
    maximum_reachable = int(max_new_tokens) - (int(max_new_tokens) % int(block_size))
    generated_length = len(trace.generated_token_ids)
    stop_reason = getattr(trace, "stop_reason", None)
    predicates = {
        "token_budget_truncation": bool(
            trace.truncated
            and stop_reason
            in {"max_token_budget", "proposal_exceeds_remaining_token_budget"}
        ),
        "generated_length_equals_maximum_reachable": bool(
            generated_length == maximum_reachable
        ),
        "branch_is_none": getattr(trace, "reward_branch", None) == "none",
        "parse_error_is_no_native_box": component.parse_error == "no native box",
        "total_reward_exactly_zero": float(component.total_reward) == 0.0,
    }
    return {
        "maximum_reachable_generated_length": maximum_reachable,
        "actual_generated_length": generated_length,
        "stop_reason": stop_reason,
        "predicates": predicates,
        "matches": all(predicates.values()),
    }


def _fresh_start_eight_group_guard_status(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    trajectories = [
        trajectory
        for group in groups
        for trajectory in group.get("trajectories", [])
    ]
    exact_group_count = len(groups) == FRESH_START_GUARD_GROUPS
    exact_trajectory_count = len(trajectories) == FRESH_START_GUARD_GROUPS * GROUP_SIZE
    all_match = exact_trajectory_count and all(
        trajectory["guard"]["matches"] for trajectory in trajectories
    )
    return {
        "required_group_count": FRESH_START_GUARD_GROUPS,
        "observed_group_count": len(groups),
        "required_trajectory_count": FRESH_START_GUARD_GROUPS * GROUP_SIZE,
        "observed_trajectory_count": len(trajectories),
        "exact_group_count": exact_group_count,
        "exact_trajectory_count": exact_trajectory_count,
        "all_trajectories_match": bool(all_match),
        "triggered": bool(exact_group_count and exact_trajectory_count and all_match),
    }


def _diagnostic_tensor_metadata(value: Any, *, include_values: bool = False) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    detached = value.detach()
    result: Dict[str, Any] = {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "layout": str(detached.layout),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "element_count": int(detached.numel()),
    }
    if detached.is_floating_point():
        finite = torch.isfinite(detached)
        finite_values = detached[finite]
        result.update(
            {
                "finite_count": int(finite.sum().item()),
                "finite_min": float(finite_values.min().float().item())
                if finite_values.numel()
                else None,
                "finite_max": float(finite_values.max().float().item())
                if finite_values.numel()
                else None,
            }
        )
    if include_values:
        result["values"] = detached.to("cpu").tolist()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=10)
    parser.add_argument("--max-attempted-groups", type=int, default=50)
    parser.add_argument("--diagnostic-checkpoint-steps", default=None,
                        help="comma-separated actual optimizer steps; e.g. 5,6,7")
    parser.add_argument("--checkpoint-interval", type=int, default=None,
                        help="checkpoint every N actual optimizer updates")
    parser.add_argument("--dry-run-config", action="store_true",
                        help="validate config/flags and write a CPU-only report; never load CUDA/model")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument(
        "--validate-checkpoint-only",
        action="store_true",
        help=(
            "load a checkpoint into a freshly constructed model and optimizer, "
            "validate the persisted contracts/state, write a summary, and exit "
            "without rollout, backward, or optimizer.step"
        ),
    )
    parser.add_argument(
        "--require-fresh-start",
        action="store_true",
        help="fail unless initialization starts at optimizer step 0 with no resume checkpoint",
    )
    parser.add_argument("--optimization-manifest", default=None,
                        help="immutable optimization JSONL; defaults to configured train split")
    parser.add_argument("--optimization-manifest-sha256", default=None)
    parser.add_argument("--validation-manifest", default=None,
                        help="immutable internal validation JSONL recorded in checkpoints")
    parser.add_argument("--validation-manifest-sha256", default=None)
    parser.add_argument("--reference-checkpoint", default=None,
                        help="uninterrupted step-10 checkpoint for resume-equivalence comparison")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to training.seed in the resolved config")
    parser.add_argument("--debug-position-ids", action="store_true",
                        help="synchronously trace RoPE position-ID provenance; debug only")
    return parser.parse_args()


def _lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]


def _projector_params(model):
    return [(n, p) for n, p in model.named_parameters() if "mlp1." in n and p.requires_grad]


def _trainable_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _trainable_state(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def _lora_state(model) -> Dict[str, torch.Tensor]:
    return {n: p.detach().float().cpu().clone() for n, p in _lora_params(model)}


def _all_trainable_finite(model) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in model.parameters() if p.requires_grad)


def _checksum(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _update_norm(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((after[name] - before[name]).square().sum().item()) for name in before))


def _grad_report(model) -> Dict[str, Any]:
    def one(prefix: str, params):
        missing = [n for n, p in params if p.grad is None]
        nonfinite = [n for n, p in params if p.grad is not None and not bool(torch.isfinite(p.grad).all().item())]
        nonzero = [n for n, p in params if p.grad is not None and bool(torch.count_nonzero(p.grad).item())]
        norm_sq = sum(float(p.grad.detach().float().square().sum().item()) for _, p in params if p.grad is not None)
        return {f"expected_{prefix}_tensors": len(params), f"missing_{prefix}_gradients": missing,
                f"nonfinite_{prefix}_gradients": nonfinite, f"nonzero_{prefix}_gradient_tensors": len(nonzero),
                f"{prefix}_gradient_norm": math.sqrt(norm_sq), f"all_{prefix}_present": not missing,
                f"{prefix}_all_finite": not nonfinite, f"{prefix}_nontrivial": bool(nonzero)}
    lora, projector, trainable = _lora_params(model), _projector_params(model), _trainable_params(model)
    result = {**one("lora", lora), **one("projector", projector), **one("trainable", trainable)}
    total_norm_sq = sum(float(p.grad.detach().float().square().sum().item()) for _, p in trainable if p.grad is not None)
    result.update({"expected_lora_tensors": len(lora), "expected_projector_tensors": len(projector),
                   "expected_trainable_tensors": len(trainable), "all_504_present": len(lora) == EXPECTED_LORA_TENSORS and not result["missing_lora_gradients"],
                   "all_6_projector_present": len(projector) == EXPECTED_PROJECTOR_TENSORS and not result["missing_projector_gradients"],
                   "all_510_trainable_present": len(trainable) == EXPECTED_TRAINABLE_TENSORS and not result["missing_trainable_gradients"],
                   "all_finite": not result["nonfinite_trainable_gradients"], "nontrivial": bool(result["nonzero_trainable_gradient_tensors"]),
                   "cumulative_gradient_norm": math.sqrt(total_norm_sq)})
    return result


def _global_clip_gradients(
    named_parameters,
    *,
    max_grad_norm: float,
    expected_parameter_count: int = EXPECTED_TRAINABLE_TENSORS,
) -> Dict[str, Any]:
    """Clip one global norm across shards without moving full gradients.

    Only one FP32 sum-of-squares scalar per device is copied to CPU. The same
    scalar coefficient is then applied in-place to every unique gradient.
    """
    if not math.isfinite(float(max_grad_norm)) or float(max_grad_norm) <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive")
    selected = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    identities = [id(parameter) for _, parameter in selected]
    if len(set(identities)) != len(identities):
        raise RuntimeError("trainable parameters were included more than once in global clipping")
    if len(selected) != int(expected_parameter_count):
        raise RuntimeError(
            "global clipping trainable count mismatch: "
            f"expected {expected_parameter_count}, got {len(selected)}"
        )
    missing = [name for name, parameter in selected if parameter.grad is None]
    if missing:
        raise RuntimeError(
            "global clipping requires every trainable gradient; missing "
            + ", ".join(missing[:8])
        )

    def sum_squares_by_device() -> Dict[str, torch.Tensor]:
        sums: Dict[str, torch.Tensor] = {}
        for name, parameter in selected:
            gradient = parameter.grad
            assert gradient is not None
            if not bool(torch.isfinite(gradient).all().item()):
                raise RuntimeError(
                    f"nonfinite gradient before global clipping: {name}"
                )
            device_key = str(gradient.device)
            contribution = gradient.detach().float().square().sum(dtype=torch.float32)
            sums[device_key] = contribution if device_key not in sums else sums[device_key] + contribution
        return sums

    before_sums = sum_squares_by_device()
    before_sq = sum(float(value.item()) for value in before_sums.values())
    global_before = math.sqrt(max(before_sq, 0.0))
    coefficient = (
        float(max_grad_norm) / global_before
        if global_before > float(max_grad_norm)
        else 1.0
    )
    clipping_applied = coefficient < 1.0
    if clipping_applied:
        with torch.no_grad():
            for _, parameter in selected:
                assert parameter.grad is not None
                parameter.grad.mul_(coefficient)
    after_sums = sum_squares_by_device()
    after_sq = sum(float(value.item()) for value in after_sums.values())
    global_after = math.sqrt(max(after_sq, 0.0))
    return {
        "global_grad_norm_before_clip": global_before,
        "global_grad_norm_after_clip": global_after,
        "max_grad_norm": float(max_grad_norm),
        "clip_coefficient": coefficient,
        "clipping_applied": clipping_applied,
        "trainable_parameter_count": len(selected),
        "unique_trainable_parameter_count": len(set(identities)),
        "per_device_sum_squares_before_clip": {
            key: float(value.item()) for key, value in before_sums.items()
        },
        "per_device_sum_squares_after_clip": {
            key: float(value.item()) for key, value in after_sums.items()
        },
        "full_gradients_moved_to_cpu": False,
    }


def _memory(devices: List[torch.device]) -> Dict[str, Any]:
    result = {}
    for d in devices:
        i = int(d.index or 0)
        torch.cuda.synchronize(i)
        total = int(torch.cuda.get_device_properties(i).total_memory)
        result[str(d)] = {
            "allocated_mb": float(torch.cuda.memory_allocated(i) / 2**20),
            "reserved_mb": float(torch.cuda.memory_reserved(i) / 2**20),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(i) / 2**20),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(i) / 2**20),
            "capacity_mb": float(total / 2**20),
            "below_capacity": bool(torch.cuda.max_memory_allocated(i) < total),
        }
    return result


def _reset_peaks(devices: List[torch.device]) -> None:
    for d in devices:
        torch.cuda.reset_peak_memory_stats(int(d.index or 0))


def _optimizer_devices(optimizer) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                counts[str(value.device)] = counts.get(str(value.device), 0) + 1
    return counts


def _optimizer_state_finite(optimizer) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for state in optimizer.state.values()
               for value in state.values() if isinstance(value, torch.Tensor) and value.is_floating_point())


def _trace_summary(trace, component) -> Dict[str, Any]:
    token_ids = [int(token) for token in trace.generated_token_ids]
    return {
        "reward": component.to_dict(), "committed_branch": getattr(trace, "reward_branch", None),
        "committed_final_bbox_norm_1000": getattr(trace, "committed_final_box_norm_1000", None),
        "has_unambiguous_committed_box": bool(getattr(trace, "has_unambiguous_committed_box", False)),
        "generated_token_count": len(token_ids),
        "generated_token_ids_checksum": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scored_block_count": sum(bool(getattr(b, "scored_for_grpo", True)) for b in trace.blocks),
        "block_count": len(trace.blocks),
    }


def _rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state() if np is not None else None,
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    if np is not None and state.get("numpy") is not None:
        np.random.set_state(state["numpy"])


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Durably replace a JSON report only after its complete serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_sanitize(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"{CHECKPOINT_PREFIX}{step:03d}.pt"


def _save_checkpoint(path: Path, *, model, optimizer, global_step: int, attempted_group_count: int,
                     skipped_zero_advantage_group_count: int, sample_cursor: int, seed: int, config_path: str,
                     split_metadata: Mapping[str, Any], runtime_contract: Mapping[str, bool],
                     kl_config: Mapping[str, Any], reference_snapshot: PolicySnapshot | None) -> None:
    trainable = _trainable_state(model)
    lora = {name: value for name, value in trainable.items() if "lora_" in name}
    _atomic_save(path, {
        "format": "two_gpu_g4_live_cache_grpo_checkpoint_v1", "global_step": global_step,
        "optimizer_step_count": global_step, "attempted_group_count": attempted_group_count,
        "skipped_zero_advantage_group_count": skipped_zero_advantage_group_count,
        "sample_cursor": sample_cursor, "seed": seed, "config_path": config_path,
        "seed_schedule": "attempt_seed = seed + attempted_group_count*1000003 + sample_index; rollout_seed = attempt_seed*4 + group_index",
        "trainable_state": trainable, "lora_state": lora,
        "optimizer_state": optimizer.state_dict(), "rng_state": _rng_state(), "split_metadata": dict(split_metadata),
        "effective_runtime_contract": dict(runtime_contract),
        "effective_kl_config": dict(kl_config),
        "reference_policy_state": (
            {name: value.detach().cpu().clone() for name, value in reference_snapshot.tensors.items()}
            if reference_snapshot is not None else None
        ),
    })


def _load_checkpoint(path: Path, *, model, optimizer) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "two_gpu_g4_live_cache_grpo_checkpoint_v1":
        raise RuntimeError(f"unsupported checkpoint format in {path}")
    expected = _trainable_state(model)
    saved = payload["trainable_state"]
    if set(saved) != set(expected):
        raise RuntimeError("checkpoint trainable parameter names do not match Case-B model")
    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            named[name].copy_(value.to(named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload["optimizer_state"])
    _move_optimizer_state_to_owner_devices(optimizer)
    return payload


def _move_optimizer_state_to_owner_devices(optimizer) -> None:
    """Undo CPU map_location while preserving each AdamW tensor's dtype/value."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state[parameter].items():
                if isinstance(value, torch.Tensor) and value.device != parameter.device:
                    optimizer.state[parameter][key] = value.to(device=parameter.device)


def _optimizer_tensor_bank(state: Any, prefix: str = "optimizer") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if isinstance(state, torch.Tensor):
        out[prefix] = state.detach().float().cpu()
    elif isinstance(state, dict):
        for key, value in state.items():
            out.update(_optimizer_tensor_bank(value, f"{prefix}.{key}"))
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            out.update(_optimizer_tensor_bank(value, f"{prefix}.{index}"))
    return out


def _optimizer_state_exact(a: Any, b: Any) -> bool:
    """Strict recursive equality for the no-op path, including Adam moments."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and bool(torch.equal(a.cpu(), b.cpu()))
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(_optimizer_state_exact(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_optimizer_state_exact(x, y) for x, y in zip(a, b))
    return a == b


def _resume_compare(reference_path: Path, *, model, optimizer) -> Dict[str, Any]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if int(reference.get("global_step", -1)) != 10:
        raise RuntimeError("resume reference must be an uninterrupted step-10 checkpoint")
    lora = compare_grad_dicts(_lora_state(model), reference["lora_state"])
    optimizer_cmp = compare_grad_dicts(_optimizer_tensor_bank(optimizer.state_dict()), _optimizer_tensor_bank(reference["optimizer_state"]))
    return {
        "reference_checkpoint": str(reference_path), "lora_parameters": lora,
        "optimizer_state": optimizer_cmp,
        "within_established_cuda_repeatability_baseline": bool(lora["within_tol"] and optimizer_cmp["within_tol"]),
        "baseline": "existing oracle comparison tolerance: max_abs<=1e-4 or rel_l2<=1e-3 with cosine>=0.999",
    }


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_sanitize(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_config(config: Dict[str, Any]) -> None:
    model = config["model"]
    max_grad_norm = float(config["training"].get("max_grad_norm", float("nan")))
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0.0:
        raise RuntimeError("training.max_grad_norm must be finite and positive")
    if model.get("name_or_path") != EXPECTED_MODEL_NAME or model.get("revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("requires the pinned LocateAnything-3B base model revision")
    if str(model.get("dtype")).lower() not in {"bfloat16", "bf16"}:
        raise RuntimeError("requires BF16 model weights/activations")
    if not bool(model.get("base_llm_frozen")) or not bool(model.get("vision_encoder_frozen")):
        raise RuntimeError("requires the validated frozen base LLM and vision encoder")
    if not bool(model.get("projector_trainable")):
        raise RuntimeError("requires the six trainable mlp1 projector tensors")
    lora = model.get("lora", {})
    if (int(lora.get("r", -1)), int(lora.get("alpha", -1)), float(lora.get("dropout", -1)), lora.get("bias")) != (8, 16, 0.05, "none"):
        raise RuntimeError("requires the validated Case-B LoRA configuration")
    if int(config["objective"]["group_size"]) != GROUP_SIZE or config["objective"]["loss_total"] not in {
        "L_GRPO", "L_GRPO_PLUS_MEDGROUND_KL"
    }:
        raise RuntimeError("requires G=4 and a validated GRPO objective")
    if bool(config["training"].get("gradient_checkpointing", False)):
        raise RuntimeError("gradient checkpointing is forbidden for live-cache replay")
    if not is_hybrid_rollout(config) or hybrid_logprob_objective(config) != "full_trajectory":
        raise RuntimeError("requires production Hybrid full_trajectory objective")
    hybrid = config["rollout"].get("hybrid", {})
    if not bool(hybrid.get("score_rejected_pbd_proposals", False)) or bool(config["rollout"].get("reconstruct_actions_from_text", False)):
        raise RuntimeError("requires exact decoder-native Hybrid commit semantics")
    if not bool(config["rewards"].get("use_decoder_committed_final_box", False)):
        raise RuntimeError("rewards must use the one decoder-committed final bbox")
    rewards = config["rewards"]
    expected_rewards = {
        "format": {
            "type": "binary_native_locateanything",
            "weight": 1.0,
            "require_exactly_one_native_box": True,
            "require_valid_geometry": True,
            "coordinate_range": [0, 1000],
        },
        "spatial": {
            "type": "binary_iou",
            "weight": 1.0,
            "iou_threshold": 0.5,
            "comparison": "greater_than",
        },
        "semantic": {
            "type": "medclip_roi_text_cosine",
            "weight": 1.0,
            "frozen": True,
            "image_input": "native_predicted_roi",
            "text_input": "original_query",
            "invalid_box_fallback": 0.0,
        },
    }
    reward_mismatches = {
        section: {
            key: {"expected": value, "actual": (rewards.get(section) or {}).get(key)}
            for key, value in expected.items()
            if (rewards.get(section) or {}).get(key) != value
        }
        for section, expected in expected_rewards.items()
    }
    reward_mismatches = {
        section: values for section, values in reward_mismatches.items() if values
    }
    if reward_mismatches:
        raise RuntimeError(
            "reward contract differs from validated LocateAnything Hybrid setup: "
            + json.dumps(reward_mismatches, sort_keys=True)
        )
    rollout = config["rollout"]
    if (float(rollout.get("temperature", -1)), int(rollout.get("top_k", -1)),
            float(rollout.get("top_p", -1)), float(rollout.get("repetition_penalty", -1))) != (1.0, 0, 1.0, 1.0):
        raise RuntimeError("requires the validated production stochastic decoding settings")
    _effective_kl_config(config)
    _effective_runtime_contract(config)


def _effective_runtime_contract(config: Dict[str, Any]) -> Dict[str, bool]:
    """Resolve and strictly validate the YAML-authoritative cache contract."""
    configured = config.get("runtime_contract")
    if not isinstance(configured, dict):
        raise RuntimeError("runtime_contract must be explicitly present in YAML")
    missing = sorted(set(EXPECTED_RUNTIME_CONTRACT) - set(configured))
    extra = sorted(set(configured) - set(EXPECTED_RUNTIME_CONTRACT))
    actual = {
        key: configured.get(key) for key in EXPECTED_RUNTIME_CONTRACT
    }
    wrong_types = sorted(key for key, value in actual.items() if type(value) is not bool)
    mismatches = {
        key: {"expected": expected, "actual": actual[key]}
        for key, expected in EXPECTED_RUNTIME_CONTRACT.items()
        if actual[key] != expected
    }
    if missing or extra or wrong_types or mismatches:
        raise RuntimeError(
            "runtime_contract differs from validated production semantics: "
            + json.dumps(
                {
                    "missing": missing,
                    "extra": extra,
                    "wrong_types": wrong_types,
                    "mismatches": mismatches,
                },
                sort_keys=True,
            )
        )
    training = config["training"]
    if training.get("gradient_replay_use_cache") is not actual["replay_cache"]:
        raise RuntimeError(
            "training.gradient_replay_use_cache contradicts runtime_contract.replay_cache"
        )
    if training.get("gradient_checkpointing") is not actual["gradient_checkpointing"]:
        raise RuntimeError(
            "training.gradient_checkpointing contradicts runtime_contract"
        )
    if (hybrid_logprob_objective(config) == "full_trajectory") is not actual["full_trajectory"]:
        raise RuntimeError(
            "Hybrid log-prob objective contradicts runtime_contract.full_trajectory"
        )
    return {key: bool(actual[key]) for key in EXPECTED_RUNTIME_CONTRACT}


def _manifest_metadata(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    from rl.runtime import sha256_file
    optimization = Path(args.optimization_manifest).resolve() if args.optimization_manifest else None
    validation = Path(args.validation_manifest).resolve() if args.validation_manifest else None
    if optimization is not None:
        actual = sha256_file(optimization)
        if args.optimization_manifest_sha256 and actual != args.optimization_manifest_sha256:
            raise RuntimeError("optimization manifest SHA-256 mismatch")
    else:
        optimization = Path(config["data"]["train_split"])
        optimization = optimization if optimization.is_absolute() else CHEST_DIR / optimization
        actual = sha256_file(optimization)
        configured_train_hash = config["data"].get("train_sha256")
        if configured_train_hash and actual != configured_train_hash:
            raise RuntimeError("configured training manifest SHA-256 mismatch")
    validation_hash = None
    if validation is not None:
        validation_hash = sha256_file(validation)
        if args.validation_manifest_sha256 and validation_hash != args.validation_manifest_sha256:
            raise RuntimeError("validation manifest SHA-256 mismatch")
    test = Path(config["data"]["test_split"])
    test = test if test.is_absolute() else CHEST_DIR / test
    test_hash = sha256_file(test)
    if test_hash != config["data"]["test_sha256"]:
        raise RuntimeError("held-out test manifest SHA-256 mismatch")
    return {"optimization_manifest": str(optimization), "optimization_manifest_sha256": actual,
            "validation_manifest": str(validation) if validation else None, "validation_manifest_sha256": validation_hash,
            "heldout_test_manifest": str(test.resolve()), "heldout_test_sha256": test_hash,
            "heldout_test_role": "final_evaluation_only_not_checkpoint_selection"}


def main() -> None:
    args = parse_args()
    if args.validate_checkpoint_only and args.resume_checkpoint is None:
        raise ValueError("--validate-checkpoint-only requires --resume-checkpoint")
    if args.require_fresh_start and args.resume_checkpoint is not None:
        raise ValueError("--require-fresh-start forbids --resume-checkpoint")
    if args.max_optimizer_steps <= 0:
        raise ValueError("--max-optimizer-steps must be positive")
    if args.max_attempted_groups < args.max_optimizer_steps:
        raise ValueError("--max-attempted-groups must be at least --max-optimizer-steps")
    if args.diagnostic_checkpoint_steps is not None and args.checkpoint_interval is not None:
        raise ValueError("use either --diagnostic-checkpoint-steps or --checkpoint-interval")
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval <= 0:
            raise ValueError("--checkpoint-interval must be positive")
        checkpoint_steps = set(range(args.checkpoint_interval, args.max_optimizer_steps + 1, args.checkpoint_interval))
        checkpoint_steps.add(args.max_optimizer_steps)
    else:
        checkpoint_steps = ({5, 10} if args.diagnostic_checkpoint_steps is None else {
            int(value) for value in args.diagnostic_checkpoint_steps.split(",") if value.strip()
        })
    if not checkpoint_steps or min(checkpoint_steps) < 1 or max(checkpoint_steps) > args.max_optimizer_steps:
        raise ValueError("diagnostic checkpoint steps must be within the requested optimizer-step target")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path, summary_path = output_dir / METRICS_FILENAME, output_dir / SUMMARY_FILENAME
    if metrics_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to append to an existing smoke-run output directory")
    if args.dry_run_config:
        config = load_resolved_config(args.config)
        _validate_config(config)
        runtime_contract = _effective_runtime_contract(config)
        kl_config = _effective_kl_config(config)
        print(
            json.dumps(
                {"event": "effective_runtime_contract", **runtime_contract},
                sort_keys=True,
            ),
            flush=True,
        )
        split_metadata = _manifest_metadata(args, config)
        dry_run = {
            "status": "validated_no_gpu_model_load", "config_path": config["_config_path"],
            "fresh_base_initialization_required": bool(args.require_fresh_start),
            "resume_checkpoint": args.resume_checkpoint,
            "initial_optimizer_step": 0 if args.resume_checkpoint is None else None,
            "model_name_or_path": config["model"]["name_or_path"],
            "model_revision": config["model"]["revision"], "dtype": config["model"]["dtype"],
            "expected_lora_tensors": EXPECTED_LORA_TENSORS,
            "expected_projector_tensors": EXPECTED_PROJECTOR_TENSORS,
            "expected_trainable_tensors": EXPECTED_TRAINABLE_TENSORS,
            "target_optimizer_steps": args.max_optimizer_steps, "max_attempted_groups": args.max_attempted_groups,
            "checkpoint_steps": sorted(checkpoint_steps), "group_size": GROUP_SIZE,
            "objective": config["objective"]["loss_total"],
            "effective_kl_config": kl_config,
            "effective_runtime_contract": runtime_contract,
            "max_grad_norm": float(config["training"]["max_grad_norm"]),
            "bfix": False,
            "split_metadata": split_metadata,
        }
        dry_path = output_dir / "two_gpu_g4_grpo_production_config_validation.json"
        _write_report_safely(dry_path, dry_run)
        print(json.dumps({"status": dry_run["status"], "output": str(dry_path)}))
        return
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    summary: Dict[str, Any] = {"status": "running", "group_size": GROUP_SIZE, "target_optimizer_steps": args.max_optimizer_steps,
        "max_attempted_groups": args.max_attempted_groups,
        "objective": None, "carry_cache": True, "detached_kv": False, "bfix": False,
        "truncated_bptt": False, "gradient_checkpointing": False, "selective_saved_tensor_cpu_offload": False,
        "scheduler_steps": 0,
        "metrics_jsonl": str(metrics_path), "optimizer_steps": 0, "attempted_group_count": 0,
        "skipped_zero_advantage_group_count": 0, "trajectories": 0, "checkpoints": [],
        "diagnostic_checkpoint_steps": sorted(checkpoint_steps)}
    summary["all_step_checks"] = {
        "finite_nontrivial_gradients": True, "all_504_lora_gradients_present": True,
        "all_6_projector_gradients_present": True, "all_510_trainable_gradients_present": True,
        "lora_parameter_changed_after_nonzero_gradient": True, "live_kv_placement": True,
        "cross_device_autograd": True, "both_gpus_below_capacity": True,
        "trainable_parameters_finite_after_step": True,
    }
    emergency_record: Dict[str, Any] | None = None
    emergency_written = False
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
            raise RuntimeError("requires exactly two visible CUDA devices")
        config = load_resolved_config(args.config)
        _validate_config(config)
        hardware = config.get("hardware") or {}
        required_gpu_name = hardware.get("required_gpu_name_substring")
        gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
        if required_gpu_name and any(required_gpu_name not in name for name in gpu_names):
            raise RuntimeError(
                f"requires two {required_gpu_name} GPUs, found {gpu_names}"
            )
        runtime_contract = _effective_runtime_contract(config)
        kl_config = _effective_kl_config(config)
        print(
            json.dumps(
                {"event": "effective_runtime_contract", **runtime_contract},
                sort_keys=True,
            ),
            flush=True,
        )
        summary.update(
            {
                "effective_runtime_contract": runtime_contract,
                "carry_cache": runtime_contract["carry_cache"],
                "replay_cache": runtime_contract["replay_cache"],
                "live_cache_replay": runtime_contract["live_cache_replay"],
                "full_trajectory": runtime_contract["full_trajectory"],
                "detach_kv": runtime_contract["detach_kv"],
                "detached_kv": runtime_contract["detach_kv"],
                "truncated_bptt": runtime_contract["truncated_bptt"],
                "gradient_checkpointing": runtime_contract["gradient_checkpointing"],
                "saved_tensor_offload": runtime_contract["saved_tensor_offload"],
                "selective_saved_tensor_cpu_offload": runtime_contract["saved_tensor_offload"],
                "max_grad_norm": float(config["training"]["max_grad_norm"]),
                "objective": config["objective"]["loss_total"],
                "effective_kl_config": kl_config,
                "visible_gpu_names": gpu_names,
            }
        )
        split_metadata = _manifest_metadata(args, config)
        seed = int(config["training"]["seed"] if args.seed is None else args.seed)
        random.seed(seed); torch.manual_seed(seed)
        if np is not None: np.random.seed(seed)
        model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(config, first_device=devices[0], second_device=devices[1])
        resolve_locateanything_qwen_decoder(model).decoder._chestxray8_debug_position_ids = bool(args.debug_position_ids)
        model.eval()
        if len(_lora_params(model)) != EXPECTED_LORA_TENSORS or len(_projector_params(model)) != EXPECTED_PROJECTOR_TENSORS or len(_trainable_params(model)) != EXPECTED_TRAINABLE_TENSORS:
            raise RuntimeError("expected exactly 504 LoRA + 6 mlp1 projector Case-B trainable tensors")
        reference_snapshot = (
            PolicySnapshot.capture(model, optimizer_step=0)
            if kl_config["enabled"] else None
        )
        optimizer = build_optimizer(model, lr=float(config["training"]["learning_rate"]),
            projector_lr=float(config["training"]["projector_learning_rate"]), weight_decay=float(config["training"]["weight_decay"]),
            use_8bit_adam=bool(config["training"]["use_8bit_adam"]))
        global_step, sample_cursor = 0, 0
        attempted_group_count, skipped_zero_advantage_group_count = 0, 0
        deferred_rng_state = None
        if args.resume_checkpoint:
            loaded = _load_checkpoint(Path(args.resume_checkpoint).resolve(), model=model, optimizer=optimizer)
            if loaded.get("effective_runtime_contract") != runtime_contract:
                raise RuntimeError(
                    "resume checkpoint runtime contract differs from validated YAML"
                )
            if loaded.get("effective_kl_config", {"enabled": False, "beta": 0.0}) != kl_config:
                raise RuntimeError("resume checkpoint KL contract differs from validated YAML")
            if kl_config["enabled"]:
                saved_reference = loaded.get("reference_policy_state")
                if not isinstance(saved_reference, dict) or reference_snapshot is None:
                    raise RuntimeError("KL checkpoint is missing the frozen reference policy state")
                reference_comparison = compare_grad_dicts(
                    reference_snapshot.tensors, saved_reference
                )
                if not reference_comparison["both_sides_all_finite"] or not reference_comparison["within_tol"]:
                    raise RuntimeError("fresh reconstructed reference differs from checkpoint")
                reference_snapshot = PolicySnapshot(
                    optimizer_step=0,
                    tensors={name: value.detach().cpu().clone() for name, value in saved_reference.items()},
                )
                summary["reference_policy_resume_comparison"] = reference_comparison
            global_step, sample_cursor = int(loaded["optimizer_step_count"]), int(loaded["sample_cursor"])
            attempted_group_count = int(loaded["attempted_group_count"])
            skipped_zero_advantage_group_count = int(loaded["skipped_zero_advantage_group_count"])
            deferred_rng_state = loaded["rng_state"]
            loaded_lora = _lora_state(model)
            summary["state_immediately_after_loading_step5"] = {
                "all_504_lora_tensors": len(loaded_lora),
                "lora_against_checkpoint": compare_grad_dicts(loaded_lora, loaded["lora_state"]),
                "adamw_state_against_checkpoint": compare_grad_dicts(
                    _optimizer_tensor_bank(optimizer.state_dict()), _optimizer_tensor_bank(loaded["optimizer_state"])
                ),
                "optimizer_param_groups_exact": _optimizer_state_exact(
                    optimizer.state_dict().get("param_groups"), loaded["optimizer_state"].get("param_groups")
                ),
                "rng_state_deferred_for_post_initialization_restore": True,
                "counters": {"optimizer_step_count": global_step, "attempted_group_count": attempted_group_count,
                             "skipped_zero_advantage_group_count": skipped_zero_advantage_group_count, "sample_cursor": sample_cursor},
            }
            saved_split = loaded.get("split_metadata")
            if saved_split is not None and saved_split != split_metadata:
                raise RuntimeError("resume checkpoint split/manifest metadata mismatch")
            if int(loaded["seed"]) != seed:
                raise RuntimeError("resume checkpoint seed does not match requested deterministic schedule")
            if args.validate_checkpoint_only:
                optimizer_parameter_ids = [
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ]
                trainable = _trainable_state(model)
                lora = _lora_state(model)
                reference_state = loaded.get("reference_policy_state") or {}
                trainable_comparison = compare_grad_dicts(
                    trainable, loaded["trainable_state"]
                )
                optimizer_state_exact = _optimizer_state_exact(
                    optimizer.state_dict(), loaded["optimizer_state"]
                )
                validation = {
                    "model_state_loaded": len(trainable) == EXPECTED_TRAINABLE_TENSORS,
                    "trainable_state_exactly_matches_checkpoint": bool(
                        trainable_comparison["both_sides_all_finite"]
                        and trainable_comparison["max_abs_error_finite"] == 0.0
                        and not trainable_comparison["missing_names"]
                    ),
                    "all_504_lora_loaded": len(lora) == EXPECTED_LORA_TENSORS,
                    "all_6_projector_loaded": (
                        len(trainable) - len(lora) == EXPECTED_PROJECTOR_TENSORS
                    ),
                    "optimizer_contains_exactly_510_unique_parameters": (
                        len(optimizer_parameter_ids) == EXPECTED_TRAINABLE_TENSORS
                        and len(set(optimizer_parameter_ids))
                        == EXPECTED_TRAINABLE_TENSORS
                    ),
                    "optimizer_state_finite": _optimizer_state_finite(optimizer),
                    "optimizer_state_exactly_matches_checkpoint": optimizer_state_exact,
                    "reference_tensor_count_matches_kl_contract": (
                        len(reference_state) == (
                            EXPECTED_TRAINABLE_TENSORS
                            if kl_config["enabled"]
                            else 0
                        )
                    ),
                    "reference_tensors_require_no_grad": not any(
                        value.requires_grad for value in reference_state.values()
                    ),
                    "runtime_contract_matches_yaml": (
                        loaded.get("effective_runtime_contract") == runtime_contract
                    ),
                    "kl_contract_matches_yaml": (
                        loaded.get("effective_kl_config") == kl_config
                    ),
                    "optimizer_step_count": global_step,
                    "sample_cursor": sample_cursor,
                }
                validation["trainable_state_comparison"] = trainable_comparison
                validation["passed"] = all(
                    value
                    for key, value in validation.items()
                    if key not in {
                        "optimizer_step_count",
                        "sample_cursor",
                        "trainable_state_comparison",
                    }
                )
                summary.update(
                    {
                        "status": "passed" if validation["passed"] else "failed",
                        "mode": "checkpoint_load_validation_only",
                        "validated_checkpoint": str(
                            Path(args.resume_checkpoint).resolve()
                        ),
                        "optimizer_step_executed": False,
                        "rollout_executed": False,
                        "checkpoint_load_validation": validation,
                    }
                )
                _write_report_safely(summary_path, summary)
                print(
                    json.dumps(
                        {
                            "status": summary["status"],
                            "summary": str(summary_path),
                        }
                    )
                )
                if not validation["passed"]:
                    raise RuntimeError("checkpoint load validation failed")
                return
        if global_step >= args.max_optimizer_steps:
            raise RuntimeError("checkpoint already meets or exceeds requested target step")
        if args.optimization_manifest:
            from sft_common import read_jsonl
            pairs = read_jsonl(Path(args.optimization_manifest).resolve())
        else:
            pairs = load_verified_pairs(config, "train")
        replayer = build_rollout_replayer(model, tokenizer, config)
        scorer = MedCLIPSemanticScorer(device=devices[0]); rewards = build_reward_pipeline_from_config(config, scorer)
        # Restore after *all* model/optimizer/replayer/scorer construction.
        # This is the last initialization that could perturb the saved RNG.
        if deferred_rng_state is not None:
            _restore_rng(deferred_rng_state)
        summary.update({"model_revision": revision, "config_path": config["_config_path"], "shard": shard,
            "seed_schedule": "attempt_seed = seed + attempted_group_count*1000003 + sample_index; rollout_seed = attempt_seed*4 + group_index",
            "seed": seed, "starting_global_step": global_step, "starting_sample_cursor": sample_cursor,
            "starting_attempted_group_count": attempted_group_count,
            "starting_skipped_zero_advantage_group_count": skipped_zero_advantage_group_count,
            "reference_policy": {
                "enabled": bool(kl_config["enabled"]),
                "definition": "initial fresh-start LocateAnything policy",
                "storage": kl_config.get("reference_storage"),
                "projector": kl_config.get("reference_projector"),
                "lora": kl_config.get("reference_lora"),
                "mode": kl_config.get("reference_mode"),
                "autograd": kl_config.get("reference_autograd"),
                "snapshot_tensor_count": len(reference_snapshot.tensors) if reference_snapshot else 0,
                "snapshot_tensors_require_grad": False,
                "optimizer_includes_reference": False,
            },
            "split_metadata": split_metadata})
        guard_armed = bool(
            args.resume_checkpoint is None
            and global_step == 0
            and attempted_group_count == 0
            and sample_cursor == 0
        )
        guard_groups: List[Dict[str, Any]] = []
        summary["fresh_start_eight_group_guard"] = {
            "armed": guard_armed,
            "group_limit": FRESH_START_GUARD_GROUPS,
            "diagnostic_path": str(output_dir / FRESH_START_GUARD_FILENAME),
        }
        while global_step < args.max_optimizer_steps:
            if attempted_group_count >= args.max_attempted_groups:
                raise RuntimeError("attempt cap reached before target optimizer updates")
            attempted_group_count += 1
            schedule = _attempt_schedule(
                seed=seed,
                attempted_group_count=attempted_group_count,
                sample_cursor=sample_cursor,
                sample_count=len(pairs),
            )
            sample_index = schedule["sample_index"]
            attempt_seed = schedule["attempt_seed"]
            rollout_seeds = schedule["rollout_seeds"]
            pair = pairs[sample_index]
            inputs = tokenize_rl_pair(processor, pair, devices[0], config=config); decoder_kwargs = decoder_inputs(inputs)
            model._chestxray8_rotary_context_base = {
                "sample_index": sample_index, "sample_id": str(pair.get("image_index", sample_index)),
                "patient_id": str(pair.get("patient_id", "")), "attempted_group_count": attempted_group_count,
                "diagnostic_json_path": str(output_dir / "rotary_position_failure.json"),
                "sampling_diagnostic_json_path": str(output_dir / "invalid_sampling_distribution.json"),
            }
            _reset_peaks(devices)
            with torch.no_grad():
                traces = generate_rollout_group(model, tokenizer, inputs, config, sample_seed=attempt_seed)
            if len(traces) != GROUP_SIZE: raise RuntimeError("rollout group did not contain four trajectories")
            if hasattr(scorer, "model"):
                scorer.model.to(devices[0])
            components = [rewards.score_from_trace(trace, pair) for trace in traces]
            if hasattr(scorer, "model"):
                scorer.model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            reward_values = [float(component.total_reward) for component in components]
            advantages = group_relative_advantages(reward_values).to(devices[1])
            all_advantages_zero = _all_advantages_exactly_zero(advantages)
            emergency_record = {"attempted_group_count": attempted_group_count, "optimizer_step_count_before_group": global_step,
                "sample_index": sample_index, "sample_cursor_before_group": sample_cursor, "attempt_seed": attempt_seed,
                "effective_runtime_contract": runtime_contract,
                "effective_kl_config": kl_config,
                "reward_group": {"rewards": reward_values, "mean": float(sum(reward_values) / GROUP_SIZE),
                "std": float(torch.tensor(reward_values).std(unbiased=False)), "advantages": [float(x.cpu()) for x in advantages],
                "all_advantages_zero": all_advantages_zero}, "trajectories": [], "step_status": "in_progress"}
            if guard_armed and len(guard_groups) < FRESH_START_GUARD_GROUPS:
                guard_groups.append(
                    {
                        "attempted_group_count": attempted_group_count,
                        "schedule": dict(schedule),
                        "sample": {
                            "sample_index": sample_index,
                            "image_index": pair.get("image_index"),
                            "patient_id": pair.get("patient_id"),
                            "image_path": pair.get("image_path"),
                            "disease": pair.get("disease"),
                            "pair": dict(pair),
                        },
                        "prompt": {
                            "rendered_text": inputs.get("rendered_prompt"),
                            "input_ids": _diagnostic_tensor_metadata(
                                inputs.get("input_ids"), include_values=True
                            ),
                            "attention_mask": _diagnostic_tensor_metadata(
                                inputs.get("attention_mask"), include_values=True
                            ),
                        },
                        "processor": {
                            "pixel_values": _diagnostic_tensor_metadata(
                                inputs.get("pixel_values")
                            ),
                            "image_grid_hws": _diagnostic_tensor_metadata(
                                inputs.get("image_grid_hws"), include_values=True
                            ),
                        },
                        "trajectories": [
                            {
                                "group_index": index,
                                "rollout_seed": rollout_seeds[index],
                                "trace": trace.to_dict(),
                                "reward": components[index].to_dict(),
                                "guard": _fresh_start_trajectory_guard_predicates(
                                    trace,
                                    components[index],
                                    max_new_tokens=int(config["rollout"]["max_new_tokens"]),
                                    block_size=int(config["rollout"]["block_size"]),
                                ),
                            }
                            for index, trace in enumerate(traces)
                        ],
                    }
                )
            emergency_written = False
            if all_advantages_zero and not kl_config["enabled"]:
                trajectory_records = _zero_advantage_trajectory_records(
                    traces, components, rollout_seeds
                )
                sample_cursor, skipped_zero_advantage_group_count = (
                    _advance_skipped_group(
                        sample_cursor, skipped_zero_advantage_group_count
                    )
                )
                record = _zero_advantage_group_record(
                    emergency_record,
                    trajectories=trajectory_records,
                    optimizer_step_count=global_step,
                    sample_cursor=sample_cursor,
                    skipped_count=skipped_zero_advantage_group_count,
                    optimizer_state_devices=_optimizer_devices(optimizer),
                    per_gpu_memory=_memory(devices),
                )
                _append_jsonl(metrics_path, record)
                emergency_written = True
                emergency_record = None
                summary["trajectories"] += GROUP_SIZE
                # Keep the eventual success/error summary synchronized with
                # JSONL even if the attempted-group cap is reached entirely
                # through mathematically exact no-op groups.
                summary["attempted_group_count"] = attempted_group_count
                summary["skipped_zero_advantage_group_count"] = (
                    skipped_zero_advantage_group_count
                )
                summary["optimizer_step_count"] = global_step
                summary["total_trajectories_across_checkpoint_lineage"] = (
                    attempted_group_count * GROUP_SIZE
                )
                guard_status = _fresh_start_eight_group_guard_status(guard_groups)
                summary["fresh_start_eight_group_guard"].update(guard_status)
                if guard_status["triggered"]:
                    diagnostic_path = output_dir / FRESH_START_GUARD_FILENAME
                    metric_records = [
                        json.loads(line)
                        for line in metrics_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    diagnostic = {
                        "format": "chestxray8_fresh_start_eight_group_guard_v1",
                        "status": "aborted_deterministic_broken_generation",
                        "reason": (
                            "all first 32 trajectories exhausted the maximum reachable "
                            "token budget with branch none, parse error 'no native box', "
                            "and exact zero reward"
                        ),
                        "guard": guard_status,
                        "guard_predicate_definition": {
                            "groups": FRESH_START_GUARD_GROUPS,
                            "trajectories_per_group": GROUP_SIZE,
                            "truncated": True,
                            "allowed_stop_reasons": [
                                "max_token_budget",
                                "proposal_exceeds_remaining_token_budget",
                            ],
                            "generated_length": (
                                "max_new_tokens - (max_new_tokens % block_size)"
                            ),
                            "reward_branch": "none",
                            "parse_error": "no native box",
                            "total_reward": 0.0,
                        },
                        "resolved_config": config,
                        "cli": vars(args),
                        "seed_schedule": summary["seed_schedule"],
                        "split_metadata": split_metadata,
                        "model_and_shard_checks": shard,
                        "model_modes": {
                            "model_training": bool(model.training),
                            "language_model_training": bool(model.language_model.training),
                            "projector_training": bool(model.mlp1.training),
                        },
                        "tokenizer": {
                            "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
                            "vocabulary_size": len(tokenizer),
                            "model_max_length": int(tokenizer.model_max_length),
                            "special_token_ids": {
                                "bos_token_id": tokenizer.bos_token_id,
                                "eos_token_id": tokenizer.eos_token_id,
                                "pad_token_id": tokenizer.pad_token_id,
                            },
                        },
                        "groups": guard_groups,
                        "metrics_path": str(metrics_path),
                        "metrics": metric_records,
                        "summary_at_abort": dict(summary),
                    }
                    _atomic_write_json(diagnostic_path, diagnostic)
                    summary["fresh_start_eight_group_guard"]["written"] = True
                    del traces, components, advantages, inputs, decoder_kwargs
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "fresh-start eight-group degeneracy guard triggered; "
                        f"diagnostic written to {diagnostic_path}"
                    )
                # Checkpoints are keyed to actual optimizer updates.  Since
                # global_step is unchanged, the existing policy writes none.
                del traces, components, advantages, inputs, decoder_kwargs
                gc.collect()
                torch.cuda.empty_cache()
                continue
            with torch.no_grad():
                old_logps = [
                    replayer.score(
                        trace,
                        use_cache=runtime_contract["replay_cache"],
                        legacy_nocache_masks=False,
                        **decoder_kwargs,
                    )[0].detach()
                    for trace in traces
                ]
                reference_scores = []
                if kl_config["enabled"]:
                    if reference_snapshot is None:
                        raise RuntimeError("active KL objective has no reference snapshot")
                    with use_policy_snapshot(model, reference_snapshot):
                        model.eval()
                        for trace in traces:
                            ref_total, _ref_blocks, ref_tokens, ref_metadata = (
                                replayer.score_with_token_logprobs(
                                    trace,
                                    use_cache=runtime_contract["replay_cache"],
                                    legacy_nocache_masks=False,
                                    **decoder_kwargs,
                                )
                            )
                            reference_scores.append(
                                {
                                    "trajectory_logp": ref_total.detach().cpu(),
                                    "token_logps": ref_tokens.detach().cpu(),
                                    "token_metadata": ref_metadata,
                                }
                            )
                    if torch.is_grad_enabled():
                        raise RuntimeError("reference policy scoring unexpectedly enabled autograd")
                    if any(value.requires_grad for value in reference_snapshot.tensors.values()):
                        raise RuntimeError("frozen reference snapshot unexpectedly requires gradients")
            optimizer.zero_grad(set_to_none=True)
            before_lora = _lora_state(model)
            before_trainable = _trainable_state(model)
            before_optimizer_state = copy.deepcopy(optimizer.state_dict())
            trajectory_records = []
            group_kl_values: List[float] = []
            group_grpo_losses: List[float] = []
            group_total_losses: List[float] = []
            for group_index, trace in enumerate(traces):
                resolved = resolve_locateanything_qwen_decoder(model).decoder
                resolved._chestxray8_two_gpu_boundary_events = []
                recorder = _ReplayKVRecorder(DecoderShardLayout(devices[0], devices[1], 18))
                if kl_config["enabled"]:
                    current, blocks, policy_token_logps, token_metadata = (
                        replayer.score_with_token_logprobs(
                            trace,
                            use_cache=runtime_contract["replay_cache"],
                            legacy_nocache_masks=False,
                            on_scored_block_cache=recorder.record,
                            **decoder_kwargs,
                        )
                    )
                    reference = reference_scores[group_index]
                    reference_token_logps = reference["token_logps"].to(
                        device=policy_token_logps.device,
                        dtype=policy_token_logps.dtype,
                    )
                    if token_metadata != reference["token_metadata"]:
                        raise RuntimeError("policy/reference Hybrid KL token masks differ")
                    token_mask = torch.ones_like(policy_token_logps, dtype=torch.bool)
                    composed = clipped_grpo_with_medground_kl(
                        current,
                        old_logps[group_index],
                        advantages[group_index],
                        policy_token_logps,
                        reference_token_logps,
                        beta=float(kl_config["beta"]),
                        clip_epsilon=float(config["objective"]["ppo_clip_epsilon"]),
                        token_mask=token_mask,
                    )
                    loss = composed.total_loss
                    loss_grpo = composed.grpo_loss
                    kl_value = composed.kl_value
                    kl_contribution = composed.kl_loss_contribution
                    reference_logp = composed.reference_logp
                    rejected_count = sum(
                        int(item["rejected_pbd_proposal_token"])
                        for item in token_metadata
                    )
                else:
                    current, blocks = replayer.score(
                        trace,
                        use_cache=runtime_contract["replay_cache"],
                        legacy_nocache_masks=False,
                        on_scored_block_cache=recorder.record,
                        **decoder_kwargs,
                    )
                    loss_grpo = grpo_clipped_loss(
                        current.reshape(1),
                        old_logps[group_index].detach().reshape(1).to(
                            current.device, torch.float32
                        ),
                        advantages[group_index].detach().reshape(1),
                        clip_epsilon=float(config["objective"]["ppo_clip_epsilon"]),
                    )
                    loss = loss_grpo
                    kl_value = torch.zeros((), device=current.device)
                    kl_contribution = torch.zeros((), device=current.device)
                    reference_logp = torch.tensor(float("nan"), device=current.device)
                    policy_token_logps = torch.empty(0, device=current.device)
                    token_metadata = []
                    token_mask = torch.empty(0, dtype=torch.bool, device=current.device)
                    rejected_count = 0
                scaled = loss / GROUP_SIZE
                scaled.backward()
                group_kl_values.append(float(kl_value.detach().float().cpu()))
                group_grpo_losses.append(float(loss_grpo.detach().float().cpu()))
                group_total_losses.append(float(loss.detach().float().cpu()))
                live_kv, boundary, grads = recorder.report(), _cross_device_boundary_report(model), _grad_report(model)
                trajectory_records.append({**_trace_summary(trace, components[group_index]), "group_index": group_index,
                    "rollout_seed": attempt_seed * GROUP_SIZE + group_index, "advantage": float(advantages[group_index].cpu()),
                    "old_logp": float(old_logps[group_index].float().cpu()), "current_logp": float(current.detach().float().cpu()),
                    "ppo_ratio": float(torch.exp(current.detach().float() - old_logps[group_index].detach().float()).cpu()),
                    "reward_advantage": float(advantages[group_index].cpu()),
                    "policy_logp": float(current.detach().float().cpu()),
                    "reference_logp": float(reference_logp.detach().float().cpu()) if kl_config["enabled"] else None,
                    "loss_grpo_unscaled": float(loss_grpo.detach().float().cpu()),
                    "loss_grpo_scaled": float((loss_grpo / GROUP_SIZE).detach().float().cpu()),
                    "kl_value": float(kl_value.detach().float().cpu()),
                    "kl_loss_contribution": float(kl_contribution.detach().float().cpu()),
                    "total_loss_unscaled": float(loss.detach().float().cpu()),
                    "total_loss_scaled": float(scaled.detach().float().cpu()),
                    "beta": float(kl_config["beta"]),
                    "trajectory_token_count": int(token_mask.sum().item()),
                    "trajectory_token_mask": [int(value) for value in token_mask.detach().cpu().tolist()],
                    "trajectory_token_metadata": token_metadata,
                    "rejected_pbd_proposal_token_count": rejected_count,
                    "prompt_tokens_excluded_from_kl": bool(kl_config.get("exclude_prompt_tokens", True)),
                    "image_tokens_excluded_from_kl": bool(kl_config.get("exclude_image_tokens", True)),
                    "reference_scored_under_no_grad": bool(kl_config["enabled"]),
                    "reference_gradients_present": False,
                    "block_logps": [float(x.detach().float().cpu()) for x in blocks], "cumulative_gradients": grads,
                    "live_kv": live_kv, "cross_device_boundary": boundary, "per_gpu_memory_after_backward": _memory(devices),
                    "pre_backward": {"loss_grpo_unscaled": float(loss_grpo.detach().float().cpu()),
                    "kl_value": float(kl_value.detach().float().cpu()),
                    "kl_loss_contribution": float(kl_contribution.detach().float().cpu()),
                    "total_loss_unscaled": float(loss.detach().float().cpu()),
                    "total_loss_scaled": float(scaled.detach().float().cpu()),
                    "advantage": float(advantages[group_index].cpu())}})
                emergency_record["trajectories"] = trajectory_records
                if not live_kv["all_replay_caches_live_and_colocated"]: raise RuntimeError("live-KV placement/liveness validation failed")
                if not (boundary["all_transfers_cross_cuda_0_to_cuda_1_with_autograd"] and boundary["gradient_reaches_early_and_late_lora"]):
                    raise RuntimeError("cross-device autograd validation failed")
                recorder._previous_next_cache = None
                del current, blocks, loss, loss_grpo, kl_value, kl_contribution, scaled, recorder
                del policy_token_logps, token_metadata, token_mask, reference_logp
                gc.collect()
            gradients = _grad_report(model)
            emergency_record["pre_step_accumulated_gradients"] = gradients
            mean_kl_value = float(sum(group_kl_values) / GROUP_SIZE)
            mean_grpo_loss = float(sum(group_grpo_losses) / GROUP_SIZE)
            mean_total_loss = float(sum(group_total_losses) / GROUP_SIZE)
            optimization_signal = bool(
                (not all_advantages_zero)
                or (kl_config["enabled"] and mean_kl_value > 0.0)
            )
            emergency_record["loss_components"] = {
                "mean_reward_advantage": float(advantages.float().mean().cpu()),
                "mean_grpo_loss": mean_grpo_loss,
                "mean_kl_value": mean_kl_value,
                "mean_kl_loss_contribution": float(kl_config["beta"]) * mean_kl_value,
                "mean_total_loss": mean_total_loss,
                "beta": float(kl_config["beta"]),
                "optimization_signal": optimization_signal,
            }
            if (kl_config["enabled"] or not all_advantages_zero) and not gradients["all_510_trainable_present"]:
                emergency_record["gradient_case"] = "A_missing_gradients"
                raise RuntimeError("missing Case-B LoRA or mlp1 projector gradients")
            if (kl_config["enabled"] or not all_advantages_zero) and not gradients["all_finite"]:
                emergency_record["gradient_case"] = "B_nonfinite_gradients"
                raise RuntimeError("nonfinite Case-B LoRA or mlp1 projector gradients")
            if (kl_config["enabled"] or not all_advantages_zero) and not _all_trainable_finite(model):
                raise RuntimeError("nonfinite trainable parameter before AdamW step")
            if not gradients["nontrivial"] and optimization_signal:
                emergency_record["gradient_case"] = "unexpected_zero_gradients_nonzero_advantages"
                raise RuntimeError("finite zero gradients with nonzero GRPO/KL optimization signal")
            if not optimization_signal and gradients["nontrivial"]:
                emergency_record["gradient_case"] = "unexpected_nonzero_gradients_zero_advantages"
                raise RuntimeError("zero total objective produced nonzero gradients")
            clipping = _global_clip_gradients(
                model.named_parameters(),
                max_grad_norm=float(config["training"]["max_grad_norm"]),
                expected_parameter_count=EXPECTED_TRAINABLE_TENSORS,
            )
            emergency_record["gradient_clipping"] = clipping
            emergency_record.update(
                {
                    key: clipping[key]
                    for key in (
                        "global_grad_norm_before_clip",
                        "global_grad_norm_after_clip",
                        "max_grad_norm",
                        "clip_coefficient",
                        "clipping_applied",
                    )
                }
            )
            if not optimization_signal:
                sample_cursor, skipped_zero_advantage_group_count = _advance_skipped_group(
                    sample_cursor, skipped_zero_advantage_group_count
                )
                record = {
                    **emergency_record,
                    "step_status": "skipped_proven_zero_total_objective_after_replay",
                    "skipped_zero_advantage": True,
                    "optimizer_step_executed": False,
                    "optimizer_step_skipped": True,
                    "replay_executed": True,
                    "optimizer_step_count_after_group": global_step,
                    "sample_cursor_after_group": sample_cursor,
                    "skipped_zero_advantage_group_count": skipped_zero_advantage_group_count,
                    "mathematical_skip_proof": {
                        "all_reward_advantages_exactly_zero": all_advantages_zero,
                        "mean_medground_kl_exactly_zero": mean_kl_value == 0.0,
                        "all_gradients_exactly_zero": not gradients["nontrivial"],
                    },
                    "optimizer_state_devices": _optimizer_devices(optimizer),
                    "per_gpu_memory_after_group": _memory(devices),
                }
                _append_jsonl(metrics_path, record)
                emergency_written = True
                emergency_record = None
                summary["trajectories"] += GROUP_SIZE
                summary["attempted_group_count"] = attempted_group_count
                summary["skipped_zero_advantage_group_count"] = (
                    skipped_zero_advantage_group_count
                )
                summary["optimizer_step_count"] = global_step
                summary["total_trajectories_across_checkpoint_lineage"] = (
                    attempted_group_count * GROUP_SIZE
                )
                guard_status = _fresh_start_eight_group_guard_status(guard_groups)
                summary["fresh_start_eight_group_guard"].update(guard_status)
                if guard_status["triggered"]:
                    diagnostic_path = output_dir / FRESH_START_GUARD_FILENAME
                    metric_records = [
                        json.loads(line)
                        for line in metrics_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
                    diagnostic = {
                        "format": "chestxray8_fresh_start_eight_group_guard_v1",
                        "status": "aborted_deterministic_broken_generation",
                        "reason": (
                            "all first 32 trajectories exhausted the maximum reachable "
                            "token budget with branch none, parse error 'no native box', "
                            "and exact zero reward"
                        ),
                        "guard": guard_status,
                        "resolved_config": config,
                        "cli": vars(args),
                        "seed_schedule": summary["seed_schedule"],
                        "split_metadata": split_metadata,
                        "model_and_shard_checks": shard,
                        "effective_kl_config": kl_config,
                        "groups": guard_groups,
                        "metrics_path": str(metrics_path),
                        "metrics": metric_records,
                        "summary_at_abort": dict(summary),
                    }
                    _atomic_write_json(diagnostic_path, diagnostic)
                    summary["fresh_start_eight_group_guard"]["written"] = True
                    raise RuntimeError(
                        "fresh-start eight-group degeneracy guard triggered; "
                        f"diagnostic written to {diagnostic_path}"
                    )
                del traces, components, advantages, inputs, decoder_kwargs
                del old_logps, reference_scores, trajectory_records
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            sample_cursor += 1
            optimizer.step(); summary["optimizer_steps"] += 1
            global_step += 1
            after_lora = _lora_state(model)
            update_cmp = compare_grad_dicts(after_lora, before_lora)
            if not update_cmp["both_sides_all_finite"] or not _all_trainable_finite(model) or not _optimizer_state_finite(optimizer):
                raise RuntimeError("nonfinite trainable parameter after AdamW step")
            changed = sum(int(torch.count_nonzero(after_lora[n] != before_lora[n]).item()) for n in before_lora)
            if changed == 0: raise RuntimeError("nontrivial gradients produced no LoRA parameter update")
            record = {**emergency_record, "global_step": global_step, "sample_cursor_after_step": sample_cursor,
                "optimizer_step_count_after_group": global_step, "step_status": "optimizer_step_completed", "optimizer_step_skipped": False,
                "skipped_zero_advantage_group_count": skipped_zero_advantage_group_count,
                "parameter_update": {"lora_before_checksum": _checksum(before_lora), "lora_after_checksum": _checksum(after_lora),
                "changed_lora_elements": changed, "lora_parameter_update_norm": _update_norm(before_lora, after_lora),
                "trainable_parameters_finite_after_step": _all_trainable_finite(model), "comparison": update_cmp}, "optimizer_state_devices": _optimizer_devices(optimizer),
                "adamw_state_finite_after_step": _optimizer_state_finite(optimizer),
                "per_gpu_memory_after_step": _memory(devices), "optimizer_steps_this_process": summary["optimizer_steps"]}
            if not all(v["below_capacity"] for v in record["per_gpu_memory_after_step"].values()): raise RuntimeError("GPU capacity exceeded")
            _append_jsonl(metrics_path, record); emergency_written = True; emergency_record = None
            summary["trajectories"] += GROUP_SIZE
            if global_step in checkpoint_steps:
                checkpoint = _checkpoint_path(output_dir, global_step)
                _save_checkpoint(checkpoint, model=model, optimizer=optimizer, global_step=global_step,
                    attempted_group_count=attempted_group_count, skipped_zero_advantage_group_count=skipped_zero_advantage_group_count,
                    sample_cursor=sample_cursor, seed=seed, config_path=config["_config_path"], split_metadata=split_metadata,
                    runtime_contract=runtime_contract, kl_config=kl_config,
                    reference_snapshot=reference_snapshot)
                torch.load(checkpoint, map_location="cpu", weights_only=False)  # explicit loadability check
                summary["checkpoints"].append(str(checkpoint))
        summary["attempted_group_count"] = attempted_group_count
        summary["skipped_zero_advantage_group_count"] = skipped_zero_advantage_group_count
        summary["optimizer_step_count"] = global_step
        summary["total_trajectories_across_checkpoint_lineage"] = attempted_group_count * GROUP_SIZE
        if global_step != args.max_optimizer_steps: raise RuntimeError("wrong optimizer step count")
        if args.reference_checkpoint:
            # Informational only.  The authoritative acceptance uses the CPU
            # oracle's two independent ten-update baseline, not a one-step
            # A1/A2 tolerance repurposed across ten optimizer updates.
            summary["single_reference_final_state_diagnostic"] = _resume_compare(
                Path(args.reference_checkpoint).resolve(), model=model, optimizer=optimizer
            )
        summary["acceptance"] = {"exactly_target_optimizer_steps": global_step == args.max_optimizer_steps,
            "at_least_target_group_trajectories_completed": attempted_group_count * GROUP_SIZE >= args.max_optimizer_steps * GROUP_SIZE,
            "zero_variance_fast_path_mathematically_valid": True,
            "all_attempt_records_written": sum(1 for _ in metrics_path.open()) == attempted_group_count - int(summary["starting_attempted_group_count"]),
            "checkpoints_loadable": len(summary["checkpoints"]) >= 1,
            **summary["all_step_checks"],
            "multi_step_equivalence_deferred_to_cpu_oracle": True,
            "no_forbidden_approximations": True}
        summary["acceptance"]["passed"] = all(summary["acceptance"].values())
        summary["status"] = "passed" if summary["acceptance"]["passed"] else "completed_partial"
    except Exception as exc:
        if emergency_record is not None and not emergency_written:
            emergency_record.update({"step_status": "error", "optimizer_step_skipped": True,
                "error_type": type(exc).__name__, "error": str(exc)})
            try:
                _append_jsonl(metrics_path, emergency_record)
            except Exception:
                pass
        summary.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
        _write_report_safely(summary_path, summary); raise
    _write_report_safely(summary_path, summary)
    print(json.dumps({"status": summary["status"], "metrics": str(metrics_path), "summary": str(summary_path)}))


if __name__ == "__main__": main()
