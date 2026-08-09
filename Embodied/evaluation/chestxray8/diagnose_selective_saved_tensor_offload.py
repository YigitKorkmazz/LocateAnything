#!/usr/bin/env python3
"""One-trajectory diagnostic: Priority-B selective saved-tensor CPU offload.

Replaces broad ``save_on_cpu`` with custom ``saved_tensors_hooks`` that:
  - keep attn_bias / attention masks GPU-resident (stride alignment),
  - offload only large activations (>= threshold, default 1 MiB),
  - dedupe overlapping views via shared storage ids,
  - restore via ``Tensor.set_(storage, offset, size, stride)`` (no contiguous()).

Does NOT run GRPO training. Preserves live KV autograd (no detach / BFIX /
truncated BPTT / mutable-cache checkpointing).
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.exact_backends import (  # noqa: E402
    ACCEPTANCE_CRITERIA,
    SELECTIVE_OFFLOAD_CANDIDATE,
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
from rl.nan_grad_diagnostics import (  # noqa: E402
    FirstNonfiniteGradProbe,
    aggregate_isfinite_flags,
    assert_trainable_grads_finite,
    compare_grad_dicts,
    report_trainable_grads,
    scalar_tensor_report,
)
from rl.lora_grad_diagnostics import truncate_trace_scored_blocks  # noqa: E402
from rl.selective_saved_tensor_offload import (  # noqa: E402
    DEFAULT_THRESHOLD_SWEEP,
    SelectiveSavedTensorOffload,
    selective_saved_tensor_offload_context,
)
from rl.inprocess_short_fixture_oracle import (  # noqa: E402
    run_inprocess_short_fixture_gradient_oracle,
)
from rl.short_ab_nan_compare import (  # noqa: E402
    orchestrate_isolated_ab_nan_compare,
    snapshot_trainable_init,
)
from rl.short_sequence_fixture import (  # noqa: E402
    DEFAULT_ORACLE_BLOCK_ADVANTAGES,
    build_short_sequence_gradient_oracle_fixture,
    save_short_sequence_fixture,
)
from verify_exact_replay_equivalence import (  # noqa: E402
    _cuda_sync,
    _parse_bool,
    _selected_grads,
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
    )
    parser.add_argument(
        "--threshold-bytes",
        type=int,
        default=256 << 10,
        help=(
            "Offload tensors whose storage or logical size is >= this many bytes. "
            "Default 256KiB = safest OOM-oriented choice after removing over-broad "
            "stride1 protection."
        ),
    )
    parser.add_argument("--pin-memory", type=_parse_bool, default=True)
    parser.add_argument("--protect-attn-bias", type=_parse_bool, default=True)
    parser.add_argument(
        "--verify-unpack-values",
        type=_parse_bool,
        default=True,
        help=(
            "Bounded allocation-free unpack sample verification "
            "(multi-index .item() only; default on for NaN hunt)"
        ),
    )
    parser.add_argument(
        "--sync-backward",
        type=_parse_bool,
        default=True,
        help="CUDA synchronize inside first-nonfinite grad hooks",
    )
    parser.add_argument(
        "--detect-anomaly",
        type=_parse_bool,
        default=False,
        help="torch.autograd.detect_anomaly(check_nan=True); short traces only",
    )
    parser.add_argument(
        "--hard-fail-nonfinite-grads",
        type=_parse_bool,
        default=True,
        help="Refuse optimizer comparison / exit non-zero if grads non-finite",
    )
    parser.add_argument(
        "--max-scored-blocks",
        type=int,
        default=None,
        help=(
            "Diagnostic: keep only the first N scored_for_grpo blocks from the "
            "existing trace (enables short-trace A/B without a separate file)"
        ),
    )
    parser.add_argument(
        "--short-ab-nan-compare",
        type=_parse_bool,
        default=False,
        help=(
            "Run A/B/C/D short-trace NaN compare in isolated subprocesses "
            "(requires --max-scored-blocks >= 3)"
        ),
    )
    parser.add_argument(
        "--isolated-condition",
        default=None,
        choices=(
            "A_no_offload_bf16",
            "B_selective_offload_bf16",
            "C_no_offload_fp32",
            "D_selective_offload_fp32",
        ),
        help="Run a single condition in a fresh orchestrated subprocess",
    )
    parser.add_argument(
        "--short-a-only",
        type=_parse_bool,
        default=False,
        help=(
            "Alias for --short-a0-only: pristine isolated A_no_offload_bf16 "
            "(no saved_tensors hooks / probes; trainable-only init)"
        ),
    )
    parser.add_argument(
        "--short-a0-only",
        type=_parse_bool,
        default=False,
        help=(
            "Isolated A0: pristine no-offload BF16, no hooks, no detect_anomaly, "
            "CUDA sync localization + LoRA sanity before replay"
        ),
    )
    parser.add_argument(
        "--short-a1-only",
        type=_parse_bool,
        default=False,
        help=(
            "Isolated A1: pristine no-offload BF16, no hooks, detect_anomaly on, "
            "CUDA sync localization + LoRA sanity before replay"
        ),
    )
    parser.add_argument(
        "--short-b-only",
        type=_parse_bool,
        default=False,
        help="Convenience: isolated B_selective_offload_bf16 only (4-block NaN gate)",
    )
    parser.add_argument(
        "--short-ab-only",
        type=_parse_bool,
        default=False,
        help=(
            "Isolated A vs B only (no C/D): no-offload BF16 vs selective-offload "
            "BF16 with nested SDPA identity guard; requires --max-scored-blocks >= 3. "
            "NOTE: production-trace A is not feasible on one 24GB GPU; prefer "
            "--short-sequence-ab-oracle."
        ),
    )
    parser.add_argument(
        "--short-sequence-ab-oracle",
        type=_parse_bool,
        default=False,
        help=(
            "LEVEL 1: in-process short-sequence A1/A2 + A/B gradient oracle on "
            "one model instance. Resizes the sample image so pristine A fits on "
            "one 24GB GPU; live carry_cache, >=3 scored MTP blocks, G0 SDPA "
            "guard on B. Isolated subprocess A/B is not used for Level-1."
        ),
    )
    parser.add_argument(
        "--production-b-feasibility",
        type=_parse_bool,
        default=False,
        help=(
            "LEVEL 2: production-trace B-only feasibility/invariants "
            "(forward logps, finite grads, unpack clean, peak memory). "
            "Alias-compatible with --short-b-only acceptance reporting."
        ),
    )
    parser.add_argument(
        "--short-fixture-target-prompt-tokens",
        type=int,
        default=192,
        help="LEVEL 1: max prompt tokens for the resized-image fixture",
    )
    parser.add_argument(
        "--short-fixture-image-max-sides",
        default="168,128,112,96,64",
        help="LEVEL 1: comma-separated max image sides tried until prompt fits",
    )
    parser.add_argument(
        "--log-mlp-lora-io-layers",
        default=None,
        help="Optional: comma-separated decoder layers for MLP LoRA IO probe (e.g. 29,30)",
    )
    parser.add_argument(
        "--identity-guard-level",
        default="G0",
        choices=("G0", "G1", "G2", "G3"),
        help=(
            "Short-fixture B-side identity guard: G0=SDPA, G1=SDPA+LoRA branches, "
            "G2=SDPA+LoRA+complete MLP modules, G3=global identity (no CPU offload)"
        ),
    )
    parser.add_argument(
        "--allow-storage-dedup",
        type=_parse_bool,
        default=False,
        help="Short-fixture: allow storage-pointer dedupe (default false / safe)",
    )
    parser.add_argument(
        "--full-unpack-verify",
        type=_parse_bool,
        default=False,
        help="Short-fixture: full per-save_id unpack verification",
    )
    parser.add_argument(
        "--probe-down-proj",
        type=_parse_bool,
        default=False,
        help="Short-fixture: instrument layers.0.mlp.down_proj for A/B",
    )
    parser.add_argument(
        "--oracle-repro-seed",
        type=int,
        default=424242,
        help="Identical A/B seed applied immediately before model forward",
    )
    parser.add_argument(
        "--threshold-sweep-bytes",
        default="1048576,524288,262144",
        help="Comma-separated thresholds for pre-backward prediction table",
    )
    parser.add_argument(
        "--max-unpack-telemetry-entries",
        type=int,
        default=512,
        help="Cap unpack telemetry rows retained in the JSON report",
    )
    parser.add_argument(
        "--max-tensor-log-entries",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--selective-offload",
        type=_parse_bool,
        default=True,
        help="If false, run live-cache with no saved-tensor hooks (reference)",
    )
    parser.add_argument(
        "--run-reference-without-offload",
        type=_parse_bool,
        default=False,
        help="Also run no-offload live-cache on the same trace (short traces)",
    )
    parser.add_argument("--backward", type=_parse_bool, default=True)
    parser.add_argument(
        "--compare-optimizer-update",
        type=_parse_bool,
        default=True,
        help="If reference succeeds, compare one AdamW step on selected grads",
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def _cpu_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _process_rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _grad_compare(
    a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    return compare_grad_dicts(a, b)


def _hypothesis_notes(out: Dict[str, Any]) -> Dict[str, Any]:
    """Heuristic cause tags from a single-run diagnostic payload."""
    pre = out.get("pre_backward_scalars") or {}
    if isinstance(pre, dict) and "all_forward_finite" in pre:
        forward_finite = bool(pre.get("all_forward_finite"))
    else:
        forward_finite = None
    unpack_nonfinite = (
        (out.get("selective_offload_summary") or {})
        .get("stats", {})
        .get("unpack_nonfinite_count")
    )
    unpack_mismatch = (
        (out.get("selective_offload_summary") or {})
        .get("stats", {})
        .get("unpack_value_mismatches")
    )
    grads = out.get("trainable_grad_report") or {}
    return {
        "forward_outputs_finite": forward_finite,
        "suggest_selective_pack_unpack_corruption": bool(
            (unpack_nonfinite or 0) > 0 or (unpack_mismatch or 0) > 0
        ),
        "suggest_invalid_restored_tensor_values": bool((unpack_nonfinite or 0) > 0),
        "suggest_shared_storage_lifetime_reuse": bool(
            (
                (out.get("selective_offload_summary") or {}).get(
                    "first_nonfinite_unpack"
                )
                or {}
            ).get("live_storage_reused")
        ),
        "suggest_diagnostic_loss_construction": bool(
            forward_finite
            and pre.get("loss")
            and not (pre.get("loss") or {}).get("isfinite", True)
        ),
        "suggest_bf16_attention_backward": bool(
            forward_finite
            and grads.get("any_grad_nonfinite")
            and not (unpack_nonfinite or 0)
            and not (unpack_mismatch or 0)
        ),
        "first_nonfinite_unpack": (out.get("selective_offload_summary") or {}).get(
            "first_nonfinite_unpack"
        ),
        "first_nonfinite_grad": out.get("first_nonfinite_grad_probe"),
    }


def _optimizer_update_compare(
    model: torch.nn.Module,
    grads_a: Dict[str, torch.Tensor],
    grads_b: Dict[str, torch.Tensor],
    *,
    lr: float,
) -> Dict[str, Any]:
    """Compare one AdamW step from identical params using two grad dicts."""
    names = sorted(set(grads_a) & set(grads_b))
    if not names:
        return {"status": "skip", "reason": "no_overlapping_grads"}

    # Build two identical leaf parameter clones and apply grads.
    params_a = []
    params_b = []
    for name in names:
        base = dict(model.named_parameters())[name].detach().float().cpu().clone()
        pa = base.clone().requires_grad_(True)
        pb = base.clone().requires_grad_(True)
        pa.grad = grads_a[name].float().cpu().clone()
        pb.grad = grads_b[name].float().cpu().clone()
        params_a.append(pa)
        params_b.append(pb)

    opt_a = torch.optim.AdamW(params_a, lr=lr)
    opt_b = torch.optim.AdamW(params_b, lr=lr)
    opt_a.step()
    opt_b.step()

    max_abs = 0.0
    max_rel = 0.0
    for pa, pb in zip(params_a, params_b):
        diff = (pa.detach() - pb.detach()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = float(pb.detach().abs().max().item()) + 1e-12
        max_rel = max(max_rel, float(diff.max().item()) / denom)
    return {
        "status": "ok",
        "num_params": len(names),
        "max_abs_update_diff": max_abs,
        "max_rel_update_diff": max_rel,
        "within_tol": max_abs <= 1e-6 or max_rel <= 1e-5,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    env = collect_environment_report(device)
    print_environment_report(env)

    if not hasattr(torch.autograd.graph, "saved_tensors_hooks"):
        raise RuntimeError(
            "torch.autograd.graph.saved_tensors_hooks unavailable in this PyTorch"
        )

    config = load_resolved_config(args.config)
    records = [
        json.loads(line)
        for line in Path(args.trace_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = records[int(args.trace_index)]
    trace = _trace_from_record(record.get("rollout_trace", record))
    truncate_report = {"truncated": False}
    if args.max_scored_blocks is not None:
        trace, truncate_report = truncate_trace_scored_blocks(
            trace, int(args.max_scored_blocks)
        )
    pairs = load_verified_pairs(config, "train")
    pair = pairs[int(record.get("sample_index", 0))]
    use_short_fixture = bool(args.short_sequence_ab_oracle)
    model, tokenizer, processor, revision = build_policy(config, device)
    replayer = build_rollout_replayer(model, tokenizer, config)
    # Avoid materializing the full production multimodal prompt when Level-1
    # short-sequence oracle will rebuild a resized-image fixture instead.
    if use_short_fixture:
        decoder_kwargs = {}
    else:
        decoder_kwargs = decoder_inputs(
            tokenize_rl_pair(processor, pair, device, config=config)
        )

    model.eval()
    assert_model_eval_for_replay(model)
    trainability = apply_trainability_case(model, args.trainability_case)

    isolated_conditions = None
    a_control = None  # "A0" | "A1" | None
    validation_level = None
    fixture_path = None
    fixture_meta = None
    if use_short_fixture:
        # Final Level-1 path: single-process A1/A2 + A/B (not isolated children).
        validation_level = "short_fixture_gradient_oracle_inprocess"
        if args.max_scored_blocks is None:
            args.max_scored_blocks = 3
        if int(args.max_scored_blocks) < 3:
            raise RuntimeError(
                "short-sequence A/B oracle requires --max-scored-blocks >= 3"
            )
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sides = [
            int(x)
            for x in str(args.short_fixture_image_max_sides).split(",")
            if x.strip()
        ]
        fixture = build_short_sequence_gradient_oracle_fixture(
            pair=pair,
            processor=processor,
            device=device,
            config=config,
            num_scored_blocks=int(args.max_scored_blocks),
            target_max_prompt_tokens=int(args.short_fixture_target_prompt_tokens),
            image_max_side_candidates=sides,
        )
        decoder_kwargs = fixture["decoder_kwargs"]
        oracle_trace = fixture["rollout_trace"]
        fixture_meta = dict(fixture["meta"])
        n_blocks = int(args.max_scored_blocks)
        base_advs = list(DEFAULT_ORACLE_BLOCK_ADVANTAGES)
        while len(base_advs) < n_blocks:
            base_advs.append(base_advs[-1])
        oracle_advs = base_advs[:n_blocks]
        fixture_meta["oracle_block_advantages"] = list(oracle_advs)
        print("=== IN-PROCESS SHORT-SEQUENCE GRADIENT ORACLE ===")
        print(
            json.dumps(
                {
                    "validation_level": validation_level,
                    "same_model_instance": True,
                    "same_cuda_process": True,
                    "identity_guard_level_b": "G0",
                    "compare_optimizer_update": bool(args.compare_optimizer_update),
                    "max_scored_blocks": n_blocks,
                    "oracle_block_advantages": oracle_advs,
                    "fixture_meta": fixture_meta,
                    "threshold_bytes": int(args.threshold_bytes),
                    "allow_storage_dedup": False,
                },
                indent=2,
                default=str,
            )
        )
        ab_report = run_inprocess_short_fixture_gradient_oracle(
            model=model,
            replayer=replayer,
            decoder_kwargs=decoder_kwargs,
            trace=oracle_trace,
            device=device,
            oracle_block_advantages=oracle_advs,
            threshold_bytes=int(args.threshold_bytes),
            pin_memory=bool(args.pin_memory),
            protect_attn_bias=bool(args.protect_attn_bias),
            verify_unpack_values=bool(args.verify_unpack_values),
            sync_backward=bool(args.sync_backward),
            detect_anomaly=False,
            clip_epsilon=float(args.clip_epsilon),
            lr=float(args.lr),
            compare_optimizer_update=bool(args.compare_optimizer_update),
            allow_storage_dedup=False,
            full_unpack_verify=False,
            include_projector=str(args.trainability_case) == "D",
            identity_guard_level_b="G0",
        )
        ab_report["revision"] = revision
        ab_report["environment"] = env
        ab_report["trainability"] = trainability
        ab_report["candidate_backend"] = dict(SELECTIVE_OFFLOAD_CANDIDATE)
        ab_report["short_sequence_fixture_meta"] = fixture_meta
        ab_report["oracle_block_advantages"] = oracle_advs
        ab_report["trace_truncate_outer"] = {
            "kind": "original_production_trace_metadata_only",
            "not_executed_by_oracle": True,
            "original_production_trace": truncate_report,
            "executed_fixture": {
                "generated_len": fixture_meta.get("generated_len")
                or fixture_meta.get("executed_generated_len"),
                "num_scored_blocks": fixture_meta.get("num_scored_blocks"),
                "prompt_len": fixture_meta.get("prompt_len"),
                "oracle_block_advantages": oracle_advs,
            },
        }
        write_json(out_path, ab_report)
        print(json.dumps(ab_report, indent=2, default=str))
        if ab_report.get("acceptance", {}).get("passes"):
            return
        raise SystemExit(2)
    elif bool(args.production_b_feasibility) or bool(args.short_b_only):
        isolated_conditions = ["B_selective_offload_bf16"]
        validation_level = "production_trace_b_feasibility"
    elif bool(args.short_a0_only) or bool(args.short_a_only):
        isolated_conditions = ["A_no_offload_bf16"]
        a_control = "A0"
    elif bool(args.short_a1_only):
        isolated_conditions = ["A_no_offload_bf16"]
        a_control = "A1"
    elif bool(args.short_ab_only):
        isolated_conditions = [
            "A_no_offload_bf16",
            "B_selective_offload_bf16",
        ]
        # A side of A-vs-B must remain pristine; B keeps selective/nested hooks.
        a_control = "A0"
    elif args.isolated_condition:
        isolated_conditions = [str(args.isolated_condition)]
        if isolated_conditions == ["A_no_offload_bf16"]:
            a_control = "A0"
    elif bool(args.short_ab_nan_compare):
        isolated_conditions = [
            "A_no_offload_bf16",
            "B_selective_offload_bf16",
            "C_no_offload_fp32",
            "D_selective_offload_fp32",
        ]
        a_control = "A0"

    if isolated_conditions is not None:
        scored_blocks_needed = (
            3
            if validation_level == "short_fixture_gradient_oracle"
            else int(args.max_scored_blocks or 0)
        )
        if validation_level == "short_fixture_gradient_oracle":
            if args.max_scored_blocks is None:
                args.max_scored_blocks = 3
            scored_blocks_needed = int(args.max_scored_blocks)
        if scored_blocks_needed < 3:
            raise RuntimeError(
                "isolated short-trace compare requires --max-scored-blocks >= 3"
            )
        out_path = Path(args.output_json)
        work_dir = out_path.parent / (out_path.stem + "_isolated_runs")
        work_dir.mkdir(parents=True, exist_ok=True)

        if validation_level == "short_fixture_gradient_oracle":
            sides = [
                int(x)
                for x in str(args.short_fixture_image_max_sides).split(",")
                if x.strip()
            ]
            fixture = build_short_sequence_gradient_oracle_fixture(
                pair=pair,
                processor=processor,
                device=device,
                config=config,
                num_scored_blocks=int(args.max_scored_blocks),
                target_max_prompt_tokens=int(args.short_fixture_target_prompt_tokens),
                image_max_side_candidates=sides,
            )
            fixture_path = work_dir / "short_sequence_fixture.pt"
            fixture_meta = save_short_sequence_fixture(fixture_path, fixture)
            print("=== SHORT-SEQUENCE GRADIENT ORACLE FIXTURE ===")
            print(json.dumps(fixture_meta, indent=2, default=str))
            # Drop GPU decoder tensors from parent before child processes.
            del fixture
            import gc

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        init_state_path = work_dir / "init_state.pt"
        include_projector = str(args.trainability_case) == "D"
        init_artifact = snapshot_trainable_init(
            model, include_projector=include_projector
        )
        torch.save(init_artifact, init_state_path)
        init_file_bytes = int(init_state_path.stat().st_size)
        init_meta = {
            "format": init_artifact.get("format"),
            "include_projector": bool(include_projector),
            "num_tensors": int(init_artifact.get("num_tensors") or 0),
            "cpu_bytes": int(init_artifact.get("cpu_bytes") or 0),
            "file_bytes": init_file_bytes,
            "tensor_names_head": list(init_artifact.get("tensor_names") or [])[:16],
        }
        # Parent must not retain the CPU tensor payload after save.
        if isinstance(init_artifact.get("tensors"), dict):
            init_artifact["tensors"].clear()
        del init_artifact
        import gc

        gc.collect()
        # Free parent model before isolated children reload it.
        del model, replayer, tokenizer, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        pristine_a = a_control in ("A0", "A1")
        detect_anomaly_a = None
        if a_control == "A0":
            detect_anomaly_a = False
        elif a_control == "A1":
            detect_anomaly_a = True
        # Nested SDPA hooks are B-only; A pristine must report enabled=false.
        nested_sdpa_for_run = any(
            c != "A_no_offload_bf16" for c in isolated_conditions
        )
        # Short-fixture oracle: pristine A without sync-localize noise.
        cuda_sync_a = bool(pristine_a) and validation_level != (
            "short_fixture_gradient_oracle"
        )
        run_lora_a = bool(pristine_a) and validation_level != (
            "short_fixture_gradient_oracle"
        )
        mlp_layers = None
        if args.log_mlp_lora_io_layers:
            mlp_layers = [
                int(x)
                for x in str(args.log_mlp_lora_io_layers).split(",")
                if x.strip()
            ]
        elif validation_level == "production_trace_b_feasibility":
            mlp_layers = [29, 30]
        oracle_advs = None
        short_guard = str(args.identity_guard_level)
        short_dedup = bool(args.allow_storage_dedup)
        short_full_verify = bool(args.full_unpack_verify)
        short_probe_down = bool(args.probe_down_proj)
        # Short-fixture corruption hunt: disable optimizer.step until grads match.
        short_compare_opt = bool(args.compare_optimizer_update)
        if validation_level == "short_fixture_gradient_oracle":
            n_blocks = int(args.max_scored_blocks)
            base_advs = list(DEFAULT_ORACLE_BLOCK_ADVANTAGES)
            while len(base_advs) < n_blocks:
                base_advs.append(base_advs[-1])
            oracle_advs = base_advs[:n_blocks]
            if fixture_meta is not None:
                fixture_meta["oracle_block_advantages"] = list(oracle_advs)
            # Defaults for the short-fixture corruption diagnostics.
            if not short_full_verify:
                short_full_verify = True
            if not short_probe_down:
                short_probe_down = True
            short_dedup = bool(args.allow_storage_dedup)  # keep explicit false default
            short_compare_opt = False
        print("=== ISOLATED SHORT-TRACE CONDITION RUNNER ===")
        print(
            json.dumps(
                {
                    "conditions": isolated_conditions,
                    "a_control": a_control,
                    "validation_level": validation_level,
                    "identity_guard_level": short_guard,
                    "allow_storage_dedup": short_dedup,
                    "full_unpack_verify": short_full_verify,
                    "probe_down_proj": short_probe_down,
                    "compare_optimizer_update": short_compare_opt,
                    "oracle_repro_seed": int(args.oracle_repro_seed),
                    "max_scored_blocks": int(args.max_scored_blocks),
                    "oracle_block_advantages": oracle_advs,
                    "trace_truncate": truncate_report,
                    "threshold_bytes": int(args.threshold_bytes),
                    "detect_anomaly_cli": bool(args.detect_anomaly),
                    "detect_anomaly_a": detect_anomaly_a,
                    "pristine_a": pristine_a,
                    "nested_sdpa_identity_hooks_enabled_for_any_condition": (
                        nested_sdpa_for_run
                    ),
                    "nested_sdpa_identity_hooks_enabled_for_A": False
                    if pristine_a
                    else nested_sdpa_for_run,
                    "cuda_sync_localize_a": cuda_sync_a,
                    "run_lora_sanity_a": run_lora_a,
                    "init_state_path": str(init_state_path),
                    "init_state_trainable_only": init_meta,
                    "fixture_path": str(fixture_path) if fixture_path else None,
                    "fixture_meta": fixture_meta,
                    "work_dir": str(work_dir),
                },
                indent=2,
            )
        )
        ab_report = orchestrate_isolated_ab_nan_compare(
            python_executable=sys.executable,
            config=str(args.config),
            trace_jsonl=str(args.trace_jsonl),
            trace_index=int(args.trace_index),
            output_dir=work_dir,
            max_scored_blocks=int(args.max_scored_blocks),
            trainability_case=str(args.trainability_case),
            threshold_bytes=int(args.threshold_bytes),
            pin_memory=bool(args.pin_memory),
            protect_attn_bias=bool(args.protect_attn_bias),
            verify_unpack_values=bool(args.verify_unpack_values),
            sync_backward=bool(args.sync_backward),
            detect_anomaly=bool(args.detect_anomaly),
            clip_epsilon=float(args.clip_epsilon),
            device=str(args.device),
            init_state_path=str(init_state_path),
            conditions=isolated_conditions,
            compare_optimizer_update=short_compare_opt,
            lr=float(args.lr),
            pristine_a=pristine_a,
            cuda_sync_localize_a=cuda_sync_a,
            run_lora_sanity_a=run_lora_a,
            detect_anomaly_a=detect_anomaly_a,
            fixture_path=str(fixture_path) if fixture_path else None,
            log_mlp_lora_io_layers=mlp_layers,
            validation_level=validation_level,
            oracle_block_advantages=oracle_advs,
            identity_guard_level=short_guard,
            allow_storage_dedup=short_dedup,
            full_unpack_verify=short_full_verify,
            probe_down_proj=short_probe_down,
            compare_optimizer_update_force=short_compare_opt,
            oracle_repro_seed=int(args.oracle_repro_seed),
        )
        ab_report["revision"] = revision
        ab_report["environment"] = env
        ab_report["trainability"] = trainability
        ab_report["candidate_backend"] = dict(SELECTIVE_OFFLOAD_CANDIDATE)
        # Keep original production-trace truncate metadata clearly labeled so it
        # cannot be confused with the executed short-sequence fixture lengths.
        if validation_level == "short_fixture_gradient_oracle":
            ab_report["trace_truncate_outer"] = {
                "kind": "original_production_trace_metadata_only",
                "not_executed_by_oracle": True,
                "original_production_trace": truncate_report,
                "executed_fixture": {
                    "generated_len": (fixture_meta or {}).get("generated_len")
                    or (fixture_meta or {}).get("executed_generated_len"),
                    "num_scored_blocks": (fixture_meta or {}).get("num_scored_blocks"),
                    "prompt_len": (fixture_meta or {}).get("prompt_len"),
                    "oracle_block_advantages": oracle_advs,
                },
            }
        else:
            ab_report["trace_truncate_outer"] = truncate_report
        ab_report["init_state_trainable_only"] = init_meta
        ab_report["a_control"] = a_control
        ab_report["short_sequence_fixture_meta"] = fixture_meta
        ab_report["oracle_block_advantages"] = oracle_advs
        a_run = (ab_report.get("runs") or {}).get("A_no_offload_bf16") or {}
        ab_report["a_nested_sdpa_identity_hooks_entered"] = a_run.get(
            "nested_sdpa_identity_hooks_entered"
        )
        ab_report["a_instrumentation"] = a_run.get("instrumentation")
        write_json(out_path, ab_report)
        print(json.dumps(ab_report, indent=2, default=str))
        if ab_report.get("acceptance", {}).get("passes"):
            return
        # A0/A1 acceptance: forward+backward ok, all grads finite, zero nested SDPA.
        if isolated_conditions == ["A_no_offload_bf16"]:
            a = a_run
            nested_entered = int(a.get("nested_sdpa_identity_hooks_entered") or 0)
            instr = a.get("instrumentation") or {}
            a_ok = (
                a.get("status") == "ok"
                and a.get("forward_status") == "ok"
                and a.get("backward_status") == "ok"
                and bool(
                    (a.get("trainable_grad_report") or {}).get(
                        "all_trainable_grads_finite"
                    )
                )
                and nested_entered == 0
                and not bool(instr.get("nested_sdpa_identity_hooks_enabled"))
                and not bool(instr.get("selective_saved_tensors_hooks"))
            )
            if a_ok:
                return
            raise SystemExit(2)
        raise SystemExit(2)

    report: Dict[str, Any] = {
        "mode": "diagnose_selective_saved_tensor_offload",
        "revision": revision,
        "environment": env,
        "candidate_backend": dict(SELECTIVE_OFFLOAD_CANDIDATE),
        "trainability": trainability,
        "trace_truncate": truncate_report,
        "max_scored_blocks": args.max_scored_blocks,
        "threshold_bytes": int(args.threshold_bytes),
        "pin_memory": bool(args.pin_memory),
        "protect_attn_bias": bool(args.protect_attn_bias),
        "selective_offload": bool(args.selective_offload),
        "backward_requested": bool(args.backward),
        "stop_grad_past_key_values": False,
        "gradient_checkpointing": False,
        "bfix_used": False,
        "truncated_bptt_used": False,
        "broad_save_on_cpu_used": False,
        "carry_cache": True,
        "model_use_cache": True,
        "acceptance_criteria": list(ACCEPTANCE_CRITERIA),
        "priority_a_baseline_note": (
            "broad save_on_cpu remains diagnostic-only: A1 forward OK (~12.4GB) "
            "but backward failed attn_bias strideH alignment; CPU RSS ~185GB"
        ),
    }

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
        mem_before = cuda_memory_snapshot(device)
        _cuda_sync(device)
        try:
            value, next_cache = original_score_one(**kwargs)
            _cuda_sync(device)
            mem_after = cuda_memory_snapshot(device)
            kv_after = analyze_past_key_values_grad(next_cache)
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

    def _run_once(*, use_selective_offload: bool) -> Dict[str, Any]:
        nonlocal scored_index, block_reports
        scored_index = 0
        block_reports = []
        _zero_grads(model)
        sweep = tuple(
            int(x.strip())
            for x in str(args.threshold_sweep_bytes).split(",")
            if x.strip()
        ) or DEFAULT_THRESHOLD_SWEEP

        manager: Optional[SelectiveSavedTensorOffload] = None
        if use_selective_offload:
            manager = SelectiveSavedTensorOffload(
                threshold_bytes=int(args.threshold_bytes),
                pin_memory=bool(args.pin_memory),
                protect_attn_bias=bool(args.protect_attn_bias),
                verify_unpack_values=bool(args.verify_unpack_values),
                max_log_entries=int(args.max_tensor_log_entries),
                track_unpack_memory=True,
            )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()
        rss_before = _process_rss_bytes()
        maxrss_before = _cpu_rss_bytes()
        t0 = time.perf_counter()
        out: Dict[str, Any] = {
            "selective_offload": use_selective_offload,
            "forward_status": "not_run",
            "backward_status": "not_run",
            "threshold_bytes": int(args.threshold_bytes),
        }
        hook_ctx = (
            selective_saved_tensor_offload_context(manager, model)
            if manager is not None
            else nullcontext()
        )
        try:
            replayer._score_one_block = score_one_instrumented  # type: ignore
            with torch.enable_grad(), hook_ctx:
                current, block_logps = replayer.score(
                    trace,
                    use_cache=True,
                    legacy_nocache_masks=False,
                    **decoder_kwargs,
                )
                out["forward_status"] = "ok"
                out["current_log_prob"] = float(current.detach().float().cpu())
                out["current_logp_requires_grad"] = bool(current.requires_grad)
                out["block_log_probs"] = [
                    float(v.detach().float().cpu()) for v in block_logps
                ]
                out["blocks"] = list(block_reports)

                # Pre-backward dry-run classification + threshold predictions.
                if manager is not None:
                    pre_backward = manager.threshold_prediction_report(sweep)
                    out["pre_backward_classification"] = pre_backward[
                        "classification"
                    ]
                    out["pre_backward_threshold_predictions"] = pre_backward[
                        "predictions"
                    ]
                    out["pre_backward_threshold_recommendation"] = pre_backward[
                        "recommendation"
                    ]
                    print("=== PRE-BACKWARD CLASSIFICATION / THRESHOLD PREDICT ===")
                    print(json.dumps(pre_backward, indent=2))
                    print("======================================================")

                if args.backward:
                    loss = grpo_clipped_loss(
                        current.reshape(1),
                        old_logp.reshape(1),
                        advantage.reshape(1),
                        clip_epsilon=float(args.clip_epsilon),
                    )
                    block_reports_scalar = [
                        scalar_tensor_report(v, name=f"block_logp[{i}]")
                        for i, v in enumerate(block_logps)
                    ]
                    pre_scalars = {
                        "current_log_prob": scalar_tensor_report(
                            current, name="current_log_prob"
                        ),
                        "block_log_probs": block_reports_scalar,
                        "loss": scalar_tensor_report(loss, name="loss_grpo"),
                        "old_log_prob": scalar_tensor_report(
                            old_logp, name="old_log_prob"
                        ),
                        "advantage": scalar_tensor_report(
                            advantage, name="advantage"
                        ),
                    }
                    finite_flags = [
                        bool(pre_scalars["current_log_prob"].get("isfinite")),
                        bool(pre_scalars["loss"].get("isfinite")),
                        *[bool(b.get("isfinite")) for b in block_reports_scalar],
                    ]
                    pre_scalars["all_forward_finite"] = aggregate_isfinite_flags(
                        finite_flags
                    )
                    out["pre_backward_scalars"] = pre_scalars
                    print("=== PRE-BACKWARD SCALARS ===")
                    print(json.dumps(pre_scalars, indent=2))
                    print("============================")

                    probe = FirstNonfiniteGradProbe(
                        model, sync_cuda=bool(args.sync_backward)
                    )
                    if bool(args.detect_anomaly):
                        try:
                            anomaly_ctx = torch.autograd.detect_anomaly(check_nan=True)
                        except TypeError:
                            anomaly_ctx = torch.autograd.detect_anomaly()
                    else:
                        anomaly_ctx = nullcontext()
                    try:
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        with anomaly_ctx:
                            loss.backward()
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        out["backward_status"] = "ok"
                        out["loss_grpo"] = float(loss.detach().float().cpu())
                        out["first_nonfinite_grad_probe"] = probe.report()
                        grad_report = report_trainable_grads(model)
                        out["trainable_grad_report"] = {
                            "num_trainable": grad_report["num_trainable"],
                            "any_grad_none": grad_report["any_grad_none"],
                            "any_grad_nonfinite": grad_report["any_grad_nonfinite"],
                            "all_trainable_grads_finite": grad_report[
                                "all_trainable_grads_finite"
                            ],
                            "num_nonfinite": grad_report["num_nonfinite"],
                            "num_grad_none": grad_report["num_grad_none"],
                            "parameters": grad_report["parameters"],
                        }
                        grads = _selected_grads(model)
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
                        # Finite-aware L2 (NaN-safe).
                        lora_l2 = {}
                        for name, g in grads.items():
                            if not is_lora_name(name):
                                continue
                            finite = torch.isfinite(g)
                            if finite.any():
                                lora_l2[name] = float(g[finite].float().norm().cpu())
                            else:
                                lora_l2[name] = float("nan")
                        out["selected_lora_grad_l2_finite"] = lora_l2
                        out["selected_lora_grad_l2"] = {
                            name: (
                                float(g.float().norm().cpu())
                                if torch.isfinite(g).all()
                                else float("nan")
                            )
                            for name, g in grads.items()
                            if is_lora_name(name)
                        }
                        out["optimizer_step_allowed"] = bool(
                            grad_report["all_trainable_grads_finite"]
                        )
                        if bool(args.hard_fail_nonfinite_grads):
                            try:
                                assert_trainable_grads_finite(model)
                            except RuntimeError as exc:
                                out["hard_fail_nonfinite_grads"] = str(exc)
                                out["optimizer_step_allowed"] = False
                                print(f"HARD_FAIL_NONFINITE_GRADS: {exc}")
                    except Exception as exc:
                        out["backward_status"] = "error"
                        out["backward_error_type"] = type(exc).__name__
                        out["backward_error_message"] = str(exc)
                        out["backward_traceback"] = traceback.format_exc()
                        out["first_nonfinite_grad_probe"] = probe.report()
                        print(out["backward_traceback"])
                    finally:
                        probe.close()
                    del loss
                del current, block_logps
            out["status"] = (
                "ok"
                if out["forward_status"] == "ok"
                and (
                    not args.backward
                    or out["backward_status"] == "ok"
                )
                else "error"
            )
        except Exception as exc:
            out["status"] = "error"
            if out["forward_status"] == "not_run":
                out["forward_status"] = "error"
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
            if manager is not None:
                summary = manager.summary()
                # Cap unpack telemetry; keep peak pointer always.
                tele = list(manager.unpack_telemetry)
                cap = int(args.max_unpack_telemetry_entries)
                peak_idx = summary.get("stats", {}).get("peak_unpack_call_index")
                if len(tele) > cap:
                    kept = tele[: cap // 2] + tele[-(cap // 2) :]
                    summary["unpack_telemetry_truncated"] = True
                    summary["unpack_telemetry_total"] = len(tele)
                    summary["unpack_telemetry_peak_index"] = peak_idx
                    out["unpack_telemetry"] = kept
                else:
                    out["unpack_telemetry"] = tele
                out["selective_offload_summary"] = summary
                out["saved_tensor_log"] = list(manager.tensor_log)
        return out

    print("=== SELECTIVE SAVED-TENSOR OFFLOAD ONE-TRAJECTORY DIAGNOSTIC ===")
    print(
        json.dumps(
            {
                "selective_offload": bool(args.selective_offload),
                "threshold_bytes": int(args.threshold_bytes),
                "pin_memory": bool(args.pin_memory),
                "protect_attn_bias": bool(args.protect_attn_bias),
                "trainability_case": args.trainability_case,
                "trace_index": args.trace_index,
                "backward": bool(args.backward),
            },
            indent=2,
        )
    )
    print("================================================================")

    primary = _run_once(use_selective_offload=bool(args.selective_offload))
    report["primary"] = primary
    report["status"] = primary.get("status")
    report["forward_status"] = primary.get("forward_status")
    report["backward_status"] = primary.get("backward_status")

    if args.run_reference_without_offload:
        print("=== REFERENCE LIVE CACHE (no saved-tensor hooks) ===")
        reference = _run_once(use_selective_offload=False)
        report["reference_no_offload"] = reference
        if (
            primary.get("forward_status") == "ok"
            and reference.get("forward_status") == "ok"
        ):
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
        if (
            primary.get("forward_status") == "ok"
            and reference.get("forward_status") == "ok"
        ):
            report["loss_vs_reference"] = {
                "primary_loss": (primary.get("pre_backward_scalars") or {})
                .get("loss", {})
                .get("value"),
                "reference_loss": (reference.get("pre_backward_scalars") or {})
                .get("loss", {})
                .get("value"),
                "primary_loss_isfinite": (primary.get("pre_backward_scalars") or {})
                .get("loss", {})
                .get("isfinite"),
                "reference_loss_isfinite": (reference.get("pre_backward_scalars") or {})
                .get("loss", {})
                .get("isfinite"),
            }
        if (
            primary.get("backward_status") == "ok"
            and reference.get("backward_status") == "ok"
        ):
            report["grad_vs_reference"] = _grad_compare(
                primary.get("selected_grads") or {},
                reference.get("selected_grads") or {},
            )
            primary_finite = bool(
                (primary.get("trainable_grad_report") or {}).get(
                    "all_trainable_grads_finite"
                )
            )
            reference_finite = bool(
                (reference.get("trainable_grad_report") or {}).get(
                    "all_trainable_grads_finite"
                )
            )
            report["nan_hypothesis"] = {
                "primary": _hypothesis_notes(primary),
                "reference": _hypothesis_notes(reference),
                "selective_vs_live": {
                    "forward_match": (
                        report.get("forward_vs_reference", {}).get(
                            "max_block_abs_diff", 1.0
                        )
                        <= 1e-5
                    ),
                    "both_grads_finite": primary_finite and reference_finite,
                    "grad_within_tol": bool(
                        (report.get("grad_vs_reference") or {}).get("within_tol")
                    ),
                },
            }
            if (
                args.compare_optimizer_update
                and primary_finite
                and reference_finite
            ):
                report["optimizer_update_vs_reference"] = _optimizer_update_compare(
                    model,
                    primary.get("selected_grads") or {},
                    reference.get("selected_grads") or {},
                    lr=float(args.lr),
                )
            elif args.compare_optimizer_update:
                report["optimizer_update_vs_reference"] = {
                    "status": "skipped_nonfinite_grads",
                    "primary_finite": primary_finite,
                    "reference_finite": reference_finite,
                    "note": "refusing optimizer.step until all trainable grads finite",
                }
        report["memory_vs_reference"] = {
            "primary_peak_allocated": (primary.get("cuda_memory_peak") or {}).get(
                "max_allocated_bytes"
            ),
            "reference_peak_allocated": (reference.get("cuda_memory_peak") or {}).get(
                "max_allocated_bytes"
            ),
            "primary_runtime_seconds": primary.get("runtime_seconds"),
            "reference_runtime_seconds": reference.get("runtime_seconds"),
            "primary_cpu_rss_after": primary.get("cpu_process_rss_after_bytes"),
            "reference_cpu_rss_after": reference.get("cpu_process_rss_after_bytes"),
            "primary_unique_offloaded_storage_bytes": (
                (primary.get("selective_offload_summary") or {})
                .get("stats", {})
                .get("unique_offloaded_storage_bytes")
            ),
        }

    # Drop bulky grad tensors before JSON write.
    for key in ("primary", "reference_no_offload"):
        blob = report.get(key)
        if not isinstance(blob, dict):
            continue
        grads = blob.pop("selected_grads", None)
        if isinstance(grads, dict):
            blob["selected_grads_summary"] = {
                name: {
                    "shape": list(t.shape),
                    "l2": float(t.float().norm().cpu()),
                }
                for name, t in list(grads.items())[:64]
            }

    offload_stats = (
        (primary.get("selective_offload_summary") or {}).get("stats") or {}
    )
    forward_exact = None
    grad_exact = None
    opt_exact = None
    if "forward_vs_reference" in report:
        forward_exact = (
            report["forward_vs_reference"]["max_block_abs_diff"] <= 1e-5
            and report["forward_vs_reference"]["total_abs_diff"] <= 1e-5
        )
    if "grad_vs_reference" in report:
        grad_exact = bool(report["grad_vs_reference"].get("within_tol"))
    if "optimizer_update_vs_reference" in report:
        opt_exact = bool(
            report["optimizer_update_vs_reference"].get("within_tol")
        )

    measured = {
        "live_production_cached_selective_saved_tensor_offload": {
            "forward_exactness": forward_exact,
            "gradient_exactness": grad_exact,
            "optimizer_update_exactness": opt_exact,
            "peak_gpu_memory_bytes": (primary.get("cuda_memory_peak") or {}).get(
                "max_allocated_bytes"
            ),
            "cpu_memory_bytes": primary.get("cpu_process_rss_after_bytes"),
            "runtime_seconds": primary.get("runtime_seconds"),
            "feasible_1x24gb": primary.get("status") == "ok",
            "feasible_2x24gb": None,
            "passes_acceptance": None,
            "notes": (
                "Priority-B one-trajectory selective saved_tensors_hooks. "
                f"forward={primary.get('forward_status')} "
                f"backward={primary.get('backward_status')} "
                f"unique_offloaded_bytes="
                f"{offload_stats.get('unique_offloaded_storage_bytes')} "
                f"protected_gpu_tensors="
                f"{offload_stats.get('gpu_resident_protected_tensors')}"
            ),
        },
        "live_production_cached_save_on_cpu": {
            "forward_exactness": True,
            "gradient_exactness": False,
            "optimizer_update_exactness": False,
            "peak_gpu_memory_bytes": int(12.4 * (1 << 30)),
            "cpu_memory_bytes": int(185 * (1 << 30)),
            "runtime_seconds": None,
            "feasible_1x24gb": False,
            "feasible_2x24gb": None,
            "passes_acceptance": False,
            "notes": (
                "A1 diagnostic baseline only: forward OK / ~12.4GB peak, "
                "backward failed attn_bias strideH alignment; CPU RSS ~185GB. "
                "Not a production backend."
            ),
        },
    }
    report["exact_backend_comparison_table"] = exact_backend_comparison_table(
        measured
    )

    if "nan_hypothesis" not in report and isinstance(primary, dict):
        report["nan_hypothesis"] = {"primary": _hypothesis_notes(primary)}

    # Acceptance gate for this diagnostic (not GRPO integration).
    # Do not mark finite checks false when the diagnostic itself errored before
    # evaluating them — use null / not_evaluated instead.
    if primary.get("backward_status") == "ok":
        primary_grads_finite: Any = bool(
            (primary.get("trainable_grad_report") or {}).get(
                "all_trainable_grads_finite"
            )
        )
    else:
        primary_grads_finite = None

    pre_backward = primary.get("pre_backward_scalars")
    if isinstance(pre_backward, dict) and "all_forward_finite" in pre_backward:
        primary_forward_finite: Any = bool(pre_backward.get("all_forward_finite"))
    else:
        primary_forward_finite = None

    report["acceptance"] = {
        "forward_values_finite": primary_forward_finite,
        "trainable_gradients_finite": primary_grads_finite,
        "forward_values_finite_status": (
            "evaluated" if primary_forward_finite is not None else "not_evaluated"
        ),
        "trainable_gradients_finite_status": (
            "evaluated" if primary_grads_finite is not None else "not_evaluated"
        ),
        "short_trace_grad_match": (report.get("grad_vs_reference") or {}).get(
            "within_tol"
        ),
        "optimizer_update_match": (report.get("optimizer_update_vs_reference") or {}).get(
            "within_tol"
        ),
        "detached_kv": False,
        "bfix": False,
        "truncated_bptt": False,
        "passes": bool(
            primary_forward_finite is True
            and primary_grads_finite is True
            and (
                not args.run_reference_without_offload
                or (
                    bool((report.get("grad_vs_reference") or {}).get("within_tol"))
                    and bool(
                        (report.get("optimizer_update_vs_reference") or {}).get(
                            "within_tol"
                        )
                    )
                )
            )
        ),
    }

    write_json(Path(args.output_json), report)
    # Compact stdout: avoid dumping thousands of tensor-log / grad-param lines.
    stdout_report = dict(report)
    primary_out = dict(stdout_report.get("primary") or {})
    if "saved_tensor_log" in primary_out:
        primary_out["saved_tensor_log"] = (
            f"<{len(primary_out['saved_tensor_log'])} entries in output JSON>"
        )
    trg = primary_out.get("trainable_grad_report")
    if isinstance(trg, dict) and "parameters" in trg:
        trg = dict(trg)
        trg["parameters"] = f"<{len(trg['parameters'])} params in output JSON>"
        primary_out["trainable_grad_report"] = trg
    stdout_report["primary"] = primary_out
    print(json.dumps(stdout_report, indent=2, default=str))
    if report.get("status") == "error":
        raise SystemExit(1)
    if bool(args.hard_fail_nonfinite_grads) and primary.get("backward_status") == "ok":
        if primary_grads_finite is False:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
