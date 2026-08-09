#!/usr/bin/env python3
"""One-trajectory diagnostic: live production-cached autograd under save_on_cpu.

Priority-A memory investigation. Does NOT run GRPO training.
Scientific constraints preserved:
  carry_cache=true, past_key_values not detached, model use_cache=true,
  no gradient checkpointing, no Bfix, no truncated-BPTT surrogate.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.exact_backends import (  # noqa: E402
    ACCEPTANCE_CRITERIA,
    SAVE_ON_CPU_CANDIDATE,
    exact_backend_comparison_table,
)
from rl.grpo import grpo_clipped_loss  # noqa: E402
from rl.lora_grad_diagnostics import (  # noqa: E402
    analyze_past_key_values_grad,
    cuda_memory_snapshot,
)
from rl.pbd_rl import _cache_length  # noqa: E402
from rl.policy_state import is_lora_name  # noqa: E402
from rl.replay_memory import assert_model_eval_for_replay  # noqa: E402
from rl.runtime import (  # noqa: E402
    build_policy,
    build_rollout_replayer,
    decoder_inputs,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
    write_json,
)
from verify_exact_replay_equivalence import (  # noqa: E402
    _cuda_sync,
    _parse_bool,
    _selected_grads,
    _tensor_meta,
    _trace_from_record,
    _zero_grads,
    apply_trainability_case,
    collect_environment_report,
    print_environment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(CHEST_DIR / "rl" / "chestxray8_grpo_hybrid_native.yaml"),
    )
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trainability-case",
        choices=("B", "D"),
        default="B",
        help="B=LoRA-only (failing Case B); D=LoRA+projector",
    )
    parser.add_argument(
        "--save-on-cpu",
        type=_parse_bool,
        default=True,
        help="Wrap gradient-bearing replay in torch.autograd.graph.save_on_cpu",
    )
    parser.add_argument(
        "--pin-memory",
        type=_parse_bool,
        default=True,
    )
    parser.add_argument(
        "--run-reference-without-offload",
        type=_parse_bool,
        default=False,
        help=(
            "Also run live-cache reference without save_on_cpu on this same "
            "trace (may OOM on full Hybrid traces; intended for short traces)"
        ),
    )
    parser.add_argument(
        "--backward",
        type=_parse_bool,
        default=True,
        help="Perform (loss).backward() after the trajectory score",
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    return parser.parse_args()


def _cpu_rss_bytes() -> int:
    # Linux: ru_maxrss is kilobytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    env = collect_environment_report(device)
    print_environment_report(env)

    if not hasattr(torch.autograd.graph, "save_on_cpu"):
        raise RuntimeError(
            "torch.autograd.graph.save_on_cpu is unavailable in this PyTorch build"
        )

    config = load_resolved_config(args.config)
    records = [
        json.loads(line)
        for line in Path(args.trace_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[int(args.trace_index)]
    trace = _trace_from_record(record.get("rollout_trace", record))
    pairs = load_verified_pairs(config, "train")
    pair = pairs[int(record.get("sample_index", 0))]
    model, tokenizer, processor, revision = build_policy(config, device)
    replayer = build_rollout_replayer(model, tokenizer, config)
    decoder_kwargs = decoder_inputs(
        tokenize_rl_pair(processor, pair, device, config=config)
    )

    model.eval()
    assert_model_eval_for_replay(model)
    trainability = apply_trainability_case(model, args.trainability_case)

    report: Dict[str, Any] = {
        "mode": "diagnose_save_on_cpu",
        "revision": revision,
        "environment": env,
        "candidate_backend": dict(SAVE_ON_CPU_CANDIDATE),
        "trainability": trainability,
        "save_on_cpu": bool(args.save_on_cpu),
        "pin_memory": bool(args.pin_memory),
        "backward_performed": False,
        "optimizer_step_performed": False,
        "stop_grad_past_key_values": False,
        "gradient_checkpointing": False,
        "bfix_used": False,
        "truncated_bptt_used": False,
        "carry_cache": True,
        "model_use_cache": True,
        "acceptance_criteria": list(ACCEPTANCE_CRITERIA),
        "blocks": [],
        "exact_backend_comparison_table": exact_backend_comparison_table(),
    }

    # Instrument per-block memory while preserving live (non-detached) cache.
    original_score_one = replayer._score_one_block
    scored_index = 0
    block_reports: List[Dict[str, Any]] = []

    def score_one_instrumented(**kwargs):
        nonlocal scored_index
        block = kwargs["block"]
        past = kwargs.get("past_key_values")
        cache_before = (
            _cache_length(past) if kwargs.get("carry_cache", True) else 0
        )
        kv_before = analyze_past_key_values_grad(past)
        if bool(args.save_on_cpu) and kv_before.get("any_grad_fn"):
            # Expected under live cache once grads are enabled and blocks>0.
            pass
        mem_before = cuda_memory_snapshot(device)
        _cuda_sync(device)
        try:
            value, next_cache = original_score_one(**kwargs)
            _cuda_sync(device)
            mem_after = cuda_memory_snapshot(device)
            kv_after = analyze_past_key_values_grad(next_cache)
            # CRITICAL: do not detach next_cache; return live tensors.
            block_reports.append(
                {
                    "scored_block_index": scored_index,
                    "block_source": getattr(block, "source", None),
                    "block_type": getattr(block, "block_type", None),
                    "cache_length_before": int(cache_before),
                    "cache_length_after": (
                        _cache_length(next_cache)
                        if next_cache is not None
                        else None
                    ),
                    "kv_grad_before": kv_before,
                    "kv_grad_after": kv_after,
                    "cuda_memory_before": mem_before,
                    "cuda_memory_after": mem_after,
                    "block_log_prob": float(value.detach().float().cpu()),
                    "block_log_prob_requires_grad": bool(value.requires_grad),
                    "status": "ok",
                }
            )
            scored_index += 1
            return value, next_cache
        except Exception as exc:
            _cuda_sync(device)
            block_reports.append(
                {
                    "scored_block_index": scored_index,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "cuda_memory_after": cuda_memory_snapshot(device),
                    "kv_grad_before": kv_before,
                }
            )
            raise

    old_logp = torch.tensor(
        float(record.get("old_log_prob", 0.0)),
        device=device,
        dtype=torch.float32,
    )
    advantage = torch.tensor(
        float(record.get("advantage", 1.0)) or 1.0,
        device=device,
        dtype=torch.float32,
    )

    def _run_once(*, use_save_on_cpu: bool) -> Dict[str, Any]:
        nonlocal scored_index, block_reports
        scored_index = 0
        block_reports = []
        _zero_grads(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        rss_before = _process_rss_bytes()
        maxrss_before = _cpu_rss_bytes()
        t0 = time.perf_counter()
        from contextlib import nullcontext

        ctx = (
            torch.autograd.graph.save_on_cpu(pin_memory=bool(args.pin_memory))
            if use_save_on_cpu
            else nullcontext()
        )
        out: Dict[str, Any] = {"save_on_cpu": use_save_on_cpu}
        try:
            replayer._score_one_block = score_one_instrumented  # type: ignore
            with torch.enable_grad(), ctx:
                current, block_logps = replayer.score(
                    trace,
                    use_cache=True,
                    legacy_nocache_masks=False,
                    **decoder_kwargs,
                )
                # Ensure past was live: at least one after-block should carry grad_fn
                # once LoRA requires_grad and scored_blocks>1.
                out["current_log_prob"] = float(current.detach().float().cpu())
                out["current_logp_requires_grad"] = bool(current.requires_grad)
                out["block_log_probs"] = [
                    float(v.detach().float().cpu()) for v in block_logps
                ]
                out["blocks"] = list(block_reports)
                if args.backward:
                    loss = grpo_clipped_loss(
                        current.reshape(1),
                        old_logp.reshape(1),
                        advantage.reshape(1),
                        clip_epsilon=float(args.clip_epsilon),
                    )
                    loss.backward()
                    out["backward_performed"] = True
                    out["loss_grpo"] = float(loss.detach().float().cpu())
                    grads = _selected_grads(model)
                    # Keep CPU copies for optional reference comparison.
                    out["selected_grads"] = {
                        name: g.detach().float().cpu().clone()
                        for name, g in grads.items()
                    }
                    out["selected_lora_grad_names"] = sorted(
                        name for name in grads if is_lora_name(name)
                    )
                    out["selected_lora_grad_count"] = sum(
                        1 for name in grads if is_lora_name(name)
                    )
                    out["selected_grad_l2"] = {
                        name: float(g.float().norm().cpu())
                        for name, g in list(grads.items())[:16]
                    }
                    del loss
                del current, block_logps
            out["status"] = "ok"
        except Exception as exc:
            out["status"] = "error"
            out["error_type"] = type(exc).__name__
            out["error_message"] = str(exc)
            out["traceback"] = traceback.format_exc()
            out["blocks"] = list(block_reports)
            print(out["traceback"])
        finally:
            replayer._score_one_block = original_score_one  # type: ignore
            if device.type == "cuda":
                torch.cuda.synchronize()
            out["runtime_seconds"] = float(time.perf_counter() - t0)
            out["cuda_memory_peak"] = cuda_memory_snapshot(device)
            out["cpu_maxrss_bytes"] = _cpu_rss_bytes()
            out["cpu_maxrss_delta_bytes"] = _cpu_rss_bytes() - maxrss_before
            rss_after = _process_rss_bytes()
            out["cpu_process_rss_before_bytes"] = rss_before
            out["cpu_process_rss_after_bytes"] = rss_after
            if rss_before is not None and rss_after is not None:
                out["cpu_process_rss_delta_bytes"] = rss_after - rss_before
            carries = [
                b.get("kv_grad_after", {}).get("carries_grad_into_trainable_params")
                for b in out.get("blocks") or []
                if b.get("status") == "ok"
            ]
            out["past_key_values_carry_grad_into_lora"] = {
                "any_block": any(bool(x) for x in carries),
                "num_checked": len(carries),
                "detached_kv": False,
            }
        return out

    print("=== SAVE_ON_CPU ONE-TRAJECTORY DIAGNOSTIC ===")
    print(
        json.dumps(
            {
                "save_on_cpu": bool(args.save_on_cpu),
                "pin_memory": bool(args.pin_memory),
                "trainability_case": args.trainability_case,
                "trace_index": args.trace_index,
                "backward": bool(args.backward),
            },
            indent=2,
        )
    )
    print("============================================")

    primary = _run_once(use_save_on_cpu=bool(args.save_on_cpu))
    report["primary"] = primary
    report["status"] = primary.get("status")
    report["backward_performed"] = bool(primary.get("backward_performed", False))

    if args.run_reference_without_offload:
        print("=== REFERENCE LIVE CACHE (no save_on_cpu) ===")
        reference = _run_once(use_save_on_cpu=False)
        report["reference_no_offload"] = reference
        if (
            primary.get("status") == "ok"
            and reference.get("status") == "ok"
            and args.backward
        ):
            # Forward compare.
            p_blocks = primary.get("block_log_probs") or []
            r_blocks = reference.get("block_log_probs") or []
            diffs = [abs(a - b) for a, b in zip(p_blocks, r_blocks)]
            report["forward_vs_reference"] = {
                "total_abs_diff": abs(
                    float(primary["current_log_prob"])
                    - float(reference["current_log_prob"])
                ),
                "max_block_abs_diff": max(diffs) if diffs else 0.0,
                "per_block_abs_diff": diffs,
            }
            report["memory_vs_reference"] = {
                "primary_peak_allocated": (
                    primary.get("cuda_memory_peak") or {}
                ).get("max_allocated_bytes"),
                "reference_peak_allocated": (
                    reference.get("cuda_memory_peak") or {}
                ).get("max_allocated_bytes"),
                "primary_runtime_seconds": primary.get("runtime_seconds"),
                "reference_runtime_seconds": reference.get("runtime_seconds"),
                "primary_cpu_rss_delta": primary.get("cpu_process_rss_delta_bytes"),
                "reference_cpu_rss_delta": reference.get(
                    "cpu_process_rss_delta_bytes"
                ),
            }
            p_grads = primary.get("selected_grads") or {}
            r_grads = reference.get("selected_grads") or {}
            names = sorted(set(p_grads) | set(r_grads))
            max_abs = 0.0
            max_rel = 0.0
            missing = []
            for name in names:
                if name not in p_grads or name not in r_grads:
                    missing.append(name)
                    continue
                diff = (p_grads[name] - r_grads[name]).abs()
                ref_norm = float(r_grads[name].abs().max().item()) + 1e-12
                max_abs = max(max_abs, float(diff.max().item()))
                max_rel = max(max_rel, float(diff.max().item()) / ref_norm)
            report["grad_vs_reference"] = {
                "num_compared": len(names) - len(missing),
                "missing_names": missing,
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "within_tol": max_abs <= 1e-4 or max_rel <= 1e-3,
            }

    # Drop bulky tensor dumps from JSON; keep summary fields.
    for key in ("primary", "reference_no_offload"):
        blob = report.get(key)
        if isinstance(blob, dict) and "selected_grads" in blob:
            blob["selected_grads"] = {
                name: {
                    "shape": list(t.shape),
                    "l2": float(t.float().norm().cpu()),
                }
                for name, t in list(blob["selected_grads"].items())[:32]
            }

    # Fill comparison table measured cell for this candidate.
    forward_exact = None
    grad_exact = None
    if "forward_vs_reference" in report:
        forward_exact = (
            report["forward_vs_reference"]["max_block_abs_diff"] <= 1e-5
        )
    if "grad_vs_reference" in report:
        grad_exact = bool(report["grad_vs_reference"].get("within_tol"))
    measured = {
        "live_production_cached_save_on_cpu": {
            "forward_exactness": forward_exact,
            "gradient_exactness": grad_exact,
            "optimizer_update_exactness": None,
            "peak_gpu_memory_bytes": (
                primary.get("cuda_memory_peak") or {}
            ).get("max_allocated_bytes"),
            "cpu_memory_bytes": primary.get("cpu_process_rss_after_bytes"),
            "runtime_seconds": primary.get("runtime_seconds"),
            "feasible_1x24gb": primary.get("status") == "ok",
            "feasible_2x24gb": None,
            "passes_acceptance": None,
            "notes": (
                "One-trajectory diagnostic only; no GRPO training. "
                f"status={primary.get('status')}"
            ),
        }
    }
    report["exact_backend_comparison_table"] = exact_backend_comparison_table(
        measured
    )

    write_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, default=str))
    if report.get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
