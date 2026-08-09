#!/usr/bin/env python3
"""LoRA-specific production-cached enable_grad diagnostics (no backward).

Manual GPU only. Does not change GRPO / rewards / default training path.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.lora_grad_diagnostics import (  # noqa: E402
    LORA_ISOLATION_GROUPS,
    WORKAROUND_MODES,
    analyze_past_key_values_grad,
    apply_lora_isolation,
    cuda_memory_snapshot,
    estimate_autograd_graph_nodes,
    peft_lora_forward_workaround,
    rich_tensor_meta,
    truncate_trace_scored_blocks,
)
from rl.pbd_rl import _cache_length  # noqa: E402
from rl.policy_state import is_lora_name, is_projector_name  # noqa: E402
from rl.replay_memory import _decoder_layers  # noqa: E402
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
    _tensor_meta,
    _trace_from_record,
    apply_trainability_case,
    capture_prepare_inputs,
    collect_environment_report,
    print_environment_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose LoRA CUDA failures on production-cached grad-forward"
    )
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
        choices=("B", "C"),
        default="B",
        help="B=LoRA-only (failing), C=projector-only (passing control)",
    )
    parser.add_argument(
        "--lora-isolation",
        choices=LORA_ISOLATION_GROUPS,
        default="all",
        help="Which LoRA target group remains trainable (Case B style)",
    )
    parser.add_argument(
        "--workaround",
        choices=WORKAROUND_MODES,
        default="none",
        help="Diagnostic PEFT forward patch (does not change GRPO objective)",
    )
    parser.add_argument(
        "--max-scored-blocks",
        type=int,
        default=None,
        help="Optional shorter-prefix diagnostic (keep first N scored blocks)",
    )
    parser.add_argument("--probe-block", type=int, default=14)
    parser.add_argument("--probe-layer", type=int, default=3)
    parser.add_argument(
        "--sync-after-layer",
        type=_parse_bool,
        default=True,
        help="Synchronize CUDA after every decoder layer (true|false)",
    )
    parser.add_argument(
        "--log-layer-memory",
        type=_parse_bool,
        default=True,
        help="Record CUDA memory after every transformer layer",
    )
    parser.add_argument(
        "--reset-peak-memory-each-block",
        type=_parse_bool,
        default=True,
    )
    return parser.parse_args()


def _freeze_non_case_params(model, trainability_case: str) -> Dict[str, Any]:
    if trainability_case == "B":
        # Caller applies isolation afterwards for LoRA subset.
        return apply_trainability_case(model, "B")
    return apply_trainability_case(model, "C")


def run_diagnosis(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    env = collect_environment_report(device)
    print_environment_report(env)

    config = load_resolved_config(args.config)
    records = [
        json.loads(line)
        for line in Path(args.trace_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[int(args.trace_index)]
    trace = _trace_from_record(record.get("rollout_trace", record))
    trace, truncate_report = truncate_trace_scored_blocks(
        trace, args.max_scored_blocks
    )

    pairs = load_verified_pairs(config, "train")
    pair = pairs[int(record.get("sample_index", 0))]
    model, tokenizer, processor, revision = build_policy(config, device)
    replayer = build_rollout_replayer(model, tokenizer, config)
    decoder_kwargs = decoder_inputs(
        tokenize_rl_pair(processor, pair, device, config=config)
    )

    model.eval()
    trainability = _freeze_non_case_params(model, args.trainability_case)
    isolation_report = None
    if args.trainability_case == "B":
        isolation_report = apply_lora_isolation(model, args.lora_isolation)
        # Recompute summary after isolation.
        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        trainability = {
            **trainability,
            "trainable_parameter_count": len(trainable),
            "first_trainable_parameter_name": trainable[0] if trainable else None,
            "lora_trainable_count": sum(1 for n in trainable if is_lora_name(n)),
            "projector_trainable_count": sum(
                1 for n in trainable if is_projector_name(n)
            ),
            "isolation": isolation_report,
        }

    lm_dtype = str(model.language_model.dtype).replace("torch.", "")
    report: Dict[str, Any] = {
        "mode": "diagnose_lora_grad_forward",
        "revision": revision,
        "environment": env,
        "trainability": trainability,
        "lora_isolation": isolation_report,
        "workaround": args.workaround,
        "truncate": truncate_report,
        "probe_block": int(args.probe_block),
        "probe_layer": int(args.probe_layer),
        "production_replay": {
            "carry_cache": True,
            "model_use_cache": True,
            "legacy_nocache_masks": False,
            "gradient_checkpointing": False,
            "force_math_sdpa": False,
            "backward_performed": False,
            "optimizer_step_performed": False,
        },
        "language_model_dtype": lm_dtype,
        "model_training": bool(model.training),
        "sync_after_layer": bool(args.sync_after_layer),
        "log_layer_memory": bool(args.log_layer_memory),
        "blocks": [],
        "layer_memory_events": [],
        "probe_events": {},
        "kv_grad_analysis_by_block": [],
    }

    layers = _decoder_layers(model)
    layer_originals = []
    current_scored_block: Dict[str, Any] = {"index": None}
    probe_sink: Dict[str, Any] = {}
    prepare_sink: Dict[str, Any] = {}
    block_reports: List[Dict[str, Any]] = []
    first_failing_block_index: Optional[int] = None
    first_failing_layer_index: Optional[int] = None

    def probe_block_getter():
        idx = current_scored_block.get("index")
        if idx is None:
            return None
        if int(idx) != int(args.probe_block):
            return None
        return idx

    def wrap_layers():
        for index, layer in enumerate(layers):
            original = layer.forward

            def make_forward(layer_index, orig):
                def forward(*f_args, **f_kwargs):
                    nonlocal first_failing_layer_index
                    hidden = f_args[0] if f_args else f_kwargs.get("hidden_states")
                    if (
                        current_scored_block.get("index") == int(args.probe_block)
                        and layer_index == int(args.probe_layer)
                        and torch.is_tensor(hidden)
                    ):
                        report["probe_events"]["hidden_before_layer"] = rich_tensor_meta(
                            hidden, name="hidden_states"
                        )
                    try:
                        out = orig(*f_args, **f_kwargs)
                        if args.sync_after_layer and device.type == "cuda":
                            torch.cuda.synchronize()
                        if args.log_layer_memory and device.type == "cuda":
                            block_idx = current_scored_block.get("index")
                            keep = (
                                block_idx == int(args.probe_block)
                                or layer_index
                                in {0, int(args.probe_layer), first_failing_layer_index}
                            )
                            if keep:
                                mem = cuda_memory_snapshot(device)
                                event = {
                                    "scored_block_index": block_idx,
                                    "layer_index": layer_index,
                                    "memory": mem,
                                }
                                if torch.is_tensor(hidden):
                                    event["hidden_meta"] = rich_tensor_meta(
                                        hidden, name="hidden_in"
                                    )
                                report["layer_memory_events"].append(event)
                        return out
                    except Exception as exc:
                        if first_failing_layer_index is None:
                            first_failing_layer_index = layer_index
                        mem = cuda_memory_snapshot(device)
                        report["layer_memory_events"].append(
                            {
                                "scored_block_index": current_scored_block.get("index"),
                                "layer_index": layer_index,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "memory": mem,
                                "hidden_meta": rich_tensor_meta(hidden, name="hidden_in")
                                if torch.is_tensor(hidden)
                                else None,
                            }
                        )
                        raise

                return forward

            layer_originals.append((layer, original))
            layer.forward = make_forward(index, original)  # type: ignore[method-assign]

    def unwrap_layers():
        for layer, original in layer_originals:
            layer.forward = original  # type: ignore[method-assign]

    original_score_one = replayer._score_one_block
    scored_index = 0

    def score_one_instrumented(**kwargs):
        nonlocal scored_index, first_failing_block_index, first_failing_layer_index
        block = kwargs["block"]
        past = kwargs.get("past_key_values")
        cache_before = (
            _cache_length(past) if kwargs.get("carry_cache", True) else 0
        )
        current_scored_block["index"] = scored_index
        if args.reset_peak_memory_each_block and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        kv_before = analyze_past_key_values_grad(past)
        mem_before = cuda_memory_snapshot(device)
        prepare_sink.clear()
        block_report: Dict[str, Any] = {
            "scored_block_index": scored_index,
            "block_source": getattr(block, "source", None),
            "block_type": getattr(block, "block_type", None),
            "prefix_length": int(getattr(block, "prefix_length", -1)),
            "cache_length_before": int(cache_before),
            "kv_grad_before": kv_before,
            "cuda_memory_before": mem_before,
        }
        report["kv_grad_analysis_by_block"].append(
            {
                "scored_block_index": scored_index,
                "phase": "before",
                "analysis": kv_before,
                "memory": mem_before,
            }
        )
        _cuda_sync(device)
        try:
            value, next_cache = original_score_one(**kwargs)
            _cuda_sync(device)
            mem_after = cuda_memory_snapshot(device)
            kv_after = analyze_past_key_values_grad(next_cache)
            roots = [value]
            if isinstance(next_cache, (tuple, list)):
                for layer_kv in next_cache[:2]:
                    if isinstance(layer_kv, (tuple, list)):
                        for tensor in layer_kv[:2]:
                            if torch.is_tensor(tensor):
                                roots.append(tensor)
            graph = estimate_autograd_graph_nodes(roots)
            block_report.update(
                {
                    "status": "ok",
                    "block_log_prob": float(value.detach().float().cpu()),
                    "block_log_prob_requires_grad": bool(value.requires_grad),
                    "block_log_prob_grad_fn": type(value.grad_fn).__name__
                    if value.grad_fn is not None
                    else None,
                    "cuda_memory_after": mem_after,
                    "kv_grad_after": kv_after,
                    "autograd_graph_after_block": graph,
                    "attention_mask_shape": prepare_sink.get("attention_mask_shape"),
                    "position_ids": prepare_sink.get("position_ids"),
                    "prepared_input_ids_meta": prepare_sink.get(
                        "prepared_input_ids_meta"
                    ),
                }
            )
            if scored_index == int(args.probe_block):
                block_report["probe_lora_modules"] = dict(probe_sink)
            report["kv_grad_analysis_by_block"].append(
                {
                    "scored_block_index": scored_index,
                    "phase": "after",
                    "analysis": kv_after,
                    "memory": mem_after,
                    "autograd_graph": graph,
                }
            )
            block_reports.append(block_report)
            scored_index += 1
            return value, next_cache
        except Exception as exc:
            _cuda_sync(device)
            mem_after = cuda_memory_snapshot(device)
            block_report.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "cuda_memory_after": mem_after,
                    "first_failing_layer_index": first_failing_layer_index,
                    "attention_mask_shape": prepare_sink.get("attention_mask_shape"),
                    "position_ids": prepare_sink.get("position_ids"),
                    "probe_lora_modules": dict(probe_sink)
                    if scored_index == int(args.probe_block)
                    else None,
                }
            )
            # Near-capacity heuristic: reserved within 256MiB of total or
            # free memory below 256MiB when driver invalid-argument appears.
            free_b = mem_after.get("free_bytes")
            total_b = mem_after.get("total_bytes")
            reserved = mem_after.get("reserved_bytes")
            near_capacity = False
            if free_b is not None and free_b < 256 * 1024 * 1024:
                near_capacity = True
            if (
                reserved is not None
                and total_b is not None
                and (total_b - reserved) < 256 * 1024 * 1024
            ):
                near_capacity = True
            block_report["near_capacity_allocation_suspected"] = near_capacity
            block_report["driver_invalid_argument"] = (
                "invalid argument" in str(exc).lower()
            )
            block_reports.append(block_report)
            if first_failing_block_index is None:
                first_failing_block_index = scored_index
            raise

    wrap_layers()
    try:
        replayer._score_one_block = score_one_instrumented  # type: ignore[method-assign]
        with peft_lora_forward_workaround(
            model,
            mode=str(args.workaround),
            probe_sink=probe_sink,
            probe_layer_index=int(args.probe_layer),
            probe_block_getter=probe_block_getter,
        ) as workaround_report, capture_prepare_inputs(
            model.language_model, prepare_sink
        ), torch.enable_grad():
            report["workaround_report"] = workaround_report
            report["torch_grad_enabled"] = bool(torch.is_grad_enabled())
            current, block_logps = replayer.score(
                trace,
                use_cache=True,
                legacy_nocache_masks=False,
                **decoder_kwargs,
            )
            report["status"] = "ok"
            report["current_log_prob"] = float(current.detach().float().cpu())
            report["current_logp_requires_grad"] = bool(current.requires_grad)
            report["current_log_prob_meta"] = _tensor_meta(current)
            report["num_block_logps"] = len(block_logps)
            report["autograd_graph_trajectory"] = estimate_autograd_graph_nodes(
                [current]
            )
            del current, block_logps
    except Exception as exc:
        report["status"] = "error"
        report["error_type"] = type(exc).__name__
        report["error_message"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    finally:
        replayer._score_one_block = original_score_one  # type: ignore[method-assign]
        unwrap_layers()

    report["blocks"] = block_reports
    report["first_failing_block_index"] = first_failing_block_index
    report["first_failing_layer_index"] = first_failing_layer_index
    # Explicit KV->LoRA grad_fn verdict across blocks.
    carries = [
        item["analysis"].get("carries_grad_into_trainable_params")
        for item in report["kv_grad_analysis_by_block"]
        if item.get("phase") == "after"
    ]
    report["past_key_values_carry_grad_into_lora"] = {
        "any_block_after_score": any(bool(x) for x in carries),
        "all_blocks_after_score": all(bool(x) for x in carries) if carries else False,
        "num_after_blocks_checked": len(carries),
        "explanation": (
            "If true, production-cached past_key_values retain autograd "
            "AccumulateGrad edges into trainable LoRA parameters from earlier "
            "blocks. This grows the live graph with cumulative cached blocks and "
            "is the primary LoRA-specific hypothesis for Case B failures."
        ),
    }
    # Summarize memory growth.
    mem_series = [
        {
            "scored_block_index": b["scored_block_index"],
            "allocated_before": (b.get("cuda_memory_before") or {}).get(
                "allocated_bytes"
            ),
            "allocated_after": (b.get("cuda_memory_after") or {}).get(
                "allocated_bytes"
            ),
            "reserved_after": (b.get("cuda_memory_after") or {}).get("reserved_bytes"),
            "max_allocated_after": (b.get("cuda_memory_after") or {}).get(
                "max_allocated_bytes"
            ),
            "status": b.get("status"),
        }
        for b in block_reports
    ]
    report["cuda_memory_series"] = mem_series
    if mem_series:
        allocs = [
            row["allocated_after"]
            for row in mem_series
            if row["allocated_after"] is not None
        ]
        if len(allocs) >= 2:
            report["cuda_memory_growth_bytes"] = int(allocs[-1] - allocs[0])
    return report


def main() -> None:
    args = parse_args()
    report = run_diagnosis(args)
    write_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2, default=str))
    if report.get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
