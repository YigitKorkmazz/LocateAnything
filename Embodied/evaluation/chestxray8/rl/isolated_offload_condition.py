#!/usr/bin/env python3
"""Run one selective-offload diagnostic condition in a fresh CUDA process.

Used so a CUDA allocator failure in one condition cannot poison later ones.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.lora_grad_diagnostics import truncate_trace_scored_blocks  # noqa: E402
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
from rl.short_ab_nan_compare import (  # noqa: E402
    pre_forward_cuda_and_lora_report,
    restore_trainable_init,
    run_one_short_condition,
)
from rl.short_sequence_fixture import (  # noqa: E402
    DEFAULT_ORACLE_BLOCK_ADVANTAGES,
    load_short_sequence_fixture,
)
from verify_exact_replay_equivalence import (  # noqa: E402
    _parse_bool,
    _trace_from_record,
    apply_trainability_case,
)


CONDITION_SPECS = {
    "A_no_offload_bf16": {"offload": False, "fp32": False},
    "B_selective_offload_bf16": {"offload": True, "fp32": False},
    "C_no_offload_fp32": {"offload": False, "fp32": True},
    "D_selective_offload_fp32": {"offload": True, "fp32": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-jsonl", required=True)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--condition", required=True, choices=sorted(CONDITION_SPECS))
    parser.add_argument("--max-scored-blocks", type=int, required=True)
    parser.add_argument("--trainability-case", choices=("B", "D"), default="B")
    parser.add_argument("--threshold-bytes", type=int, default=256 << 10)
    parser.add_argument("--pin-memory", type=_parse_bool, default=True)
    parser.add_argument("--protect-attn-bias", type=_parse_bool, default=True)
    parser.add_argument("--verify-unpack-values", type=_parse_bool, default=True)
    parser.add_argument("--sync-backward", type=_parse_bool, default=True)
    parser.add_argument("--detect-anomaly", type=_parse_bool, default=True)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--init-state-path",
        default=None,
        help=(
            "Optional trainable-only init artifact (LoRA [+ projector] + RNG) "
            "for identical initial weights across conditions; must be "
            "trainable_only_v1 (not a full model state_dict)"
        ),
    )
    parser.add_argument(
        "--pristine-no-hooks",
        type=_parse_bool,
        default=False,
        help=(
            "Disable selective/nested SDPA saved_tensors hooks, unpack "
            "verification, attention context tracking, and layer/grad probes"
        ),
    )
    parser.add_argument(
        "--cuda-sync-localize",
        type=_parse_bool,
        default=False,
        help="Sync after each scored block and each decoder layer; log progress",
    )
    parser.add_argument(
        "--run-lora-sanity",
        type=_parse_bool,
        default=False,
        help="Run first-MLP LoRA A/B sanity matmul before trajectory replay",
    )
    parser.add_argument(
        "--fixture-path",
        default=None,
        help="Optional short_sequence_gradient_oracle_v1 fixture (.pt)",
    )
    parser.add_argument(
        "--log-mlp-lora-io-layers",
        default=None,
        help="Comma-separated decoder layer indices for MLP LoRA IO probe",
    )
    parser.add_argument(
        "--oracle-block-advantages",
        default=None,
        help=(
            "Comma-separated per-scored-block advantages for short-fixture "
            "backend-gradient oracle (ratio init 1 via old=current.detach())"
        ),
    )
    parser.add_argument(
        "--identity-guard-level",
        default="G0",
        choices=("G0", "G1", "G2", "G3"),
        help="Nested identity-guard level for selective offload (B only)",
    )
    parser.add_argument(
        "--allow-storage-dedup",
        type=_parse_bool,
        default=False,
        help="Reuse CPU payloads by storage identity (unsafe; default off)",
    )
    parser.add_argument(
        "--full-unpack-verify",
        type=_parse_bool,
        default=False,
        help="Full per-save_id checksum + reference compare on unpack",
    )
    parser.add_argument(
        "--probe-down-proj",
        type=_parse_bool,
        default=False,
        help="Instrument layers.0.mlp.down_proj LoRA/base IO and grads",
    )
    parser.add_argument(
        "--oracle-repro-seed",
        type=int,
        default=None,
        help="Identical A/B seed applied immediately before model forward",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    spec = CONDITION_SPECS[args.condition]
    config = load_resolved_config(args.config)
    fixture_meta = None
    # Fresh process: load base + PEFT normally, then copy only trainable init.
    model, tokenizer, processor, revision = build_policy(config, device)
    replayer = build_rollout_replayer(model, tokenizer, config)

    if args.fixture_path:
        decoder_kwargs, trace, fixture_meta = load_short_sequence_fixture(
            Path(args.fixture_path), device
        )
        trunc = {
            "truncated": True,
            "source": "short_sequence_gradient_oracle_fixture",
            "kept_scored_blocks": sum(
                1 for b in trace.blocks if getattr(b, "scored_for_grpo", False)
            ),
            "fixture_meta": {
                k: fixture_meta.get(k)
                for k in (
                    "prompt_len",
                    "image_token_count",
                    "block0_window_length",
                    "num_scored_blocks",
                    "image_max_side",
                )
            },
        }
        # Production old_logp is unused in oracle mode (ratio init via detach).
        record = {
            "old_log_prob": 0.0,
            "advantage": 1.0,
            "sample_index": None,
            "oracle_backend_validation": True,
        }
    else:
        records = [
            json.loads(line)
            for line in Path(args.trace_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        record = records[int(args.trace_index)]
        trace = _trace_from_record(record.get("rollout_trace", record))
        trace, trunc = truncate_trace_scored_blocks(trace, int(args.max_scored_blocks))
        pairs = load_verified_pairs(config, "train")
        pair = pairs[int(record.get("sample_index", 0))]
        decoder_kwargs = decoder_inputs(
            tokenize_rl_pair(processor, pair, device, config=config)
        )

    model.eval()
    assert_model_eval_for_replay(model)
    trainability = apply_trainability_case(model, args.trainability_case)

    restore_report = None
    if args.init_state_path:
        artifact = torch.load(args.init_state_path, map_location="cpu")
        restore_report = restore_trainable_init(
            model, artifact, device=device, clear_cpu_refs=True
        )
        del artifact
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Re-assert trainability after in-place copy.
        trainability = apply_trainability_case(model, args.trainability_case)

    if spec["fp32"]:
        model.float()

    old_logp = torch.tensor(
        float(record.get("old_log_prob", 0.0)), device=device, dtype=torch.float32
    )
    advantage = torch.tensor(
        float(record.get("advantage", 1.0)) or 1.0,
        device=device,
        dtype=torch.float32,
    )
    oracle_advs = None
    if args.oracle_block_advantages:
        oracle_advs = [
            float(x) for x in str(args.oracle_block_advantages).split(",") if x.strip()
        ]
    elif args.fixture_path:
        # Short-fixture default: unequal-sign/magnitude advantages for KV paths.
        n_scored = sum(
            1 for b in trace.blocks if getattr(b, "scored_for_grpo", False)
        )
        base = list(DEFAULT_ORACLE_BLOCK_ADVANTAGES)
        while len(base) < n_scored:
            base.append(base[-1])
        oracle_advs = base[:n_scored]
        if fixture_meta is not None:
            oracle_advs = [
                float(x)
                for x in (
                    fixture_meta.get("oracle_block_advantages") or oracle_advs
                )
            ]

    if restore_report is not None:
        print(
            json.dumps(
                {
                    "event": "trainable_init_restore",
                    "condition": args.condition,
                    "num_restored": restore_report.get("num_restored"),
                    "bytes_restored": restore_report.get("bytes_restored"),
                    "storage_pointer_consistent": restore_report.get(
                        "storage_pointer_consistent"
                    ),
                    "first_lora_parameter": restore_report.get("first_lora_parameter"),
                },
                indent=2,
                default=str,
            )
        )
        pre = pre_forward_cuda_and_lora_report(model, device)
        print(
            json.dumps(
                {
                    "event": "pre_forward_after_restore",
                    "condition": args.condition,
                    **pre,
                    "restored_num_tensors": restore_report.get("num_restored"),
                    "restored_bytes": restore_report.get("bytes_restored"),
                },
                indent=2,
                default=str,
            )
        )

    try:
        # Condition A must never enable offload/nested SDPA even if misconfigured.
        use_offload = bool(spec["offload"]) and not bool(args.pristine_no_hooks)
        mlp_layers = None
        if args.log_mlp_lora_io_layers:
            mlp_layers = [
                int(x) for x in str(args.log_mlp_lora_io_layers).split(",") if x.strip()
            ]
        result = run_one_short_condition(
            label=args.condition,
            model=model,
            replayer=replayer,
            trace=trace,
            decoder_kwargs=decoder_kwargs,
            old_logp=old_logp,
            advantage=advantage,
            clip_epsilon=float(args.clip_epsilon),
            device=device,
            use_selective_offload=use_offload,
            use_fp32=bool(spec["fp32"]),
            threshold_bytes=int(args.threshold_bytes),
            pin_memory=bool(args.pin_memory),
            protect_attn_bias=bool(args.protect_attn_bias),
            verify_unpack_values=bool(args.verify_unpack_values),
            sync_backward=bool(args.sync_backward),
            detect_anomaly=bool(args.detect_anomaly),
            pristine_no_hooks=bool(args.pristine_no_hooks),
            cuda_sync_localize=bool(args.cuda_sync_localize),
            run_lora_sanity=bool(args.run_lora_sanity),
            log_mlp_lora_io_layers=mlp_layers,
            oracle_block_advantages=oracle_advs,
            identity_guard_level=str(args.identity_guard_level),
            allow_storage_dedup=bool(args.allow_storage_dedup),
            full_unpack_verify=bool(args.full_unpack_verify),
            probe_down_proj=bool(args.probe_down_proj),
            oracle_repro_seed=(
                int(args.oracle_repro_seed)
                if args.oracle_repro_seed is not None
                else None
            ),
        )
    except Exception as exc:
        result = {
            "label": args.condition,
            "status": "error",
            "forward_status": "error",
            "backward_status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    result["revision"] = revision
    result["trainability"] = trainability
    result["trace_truncate"] = trunc
    result["isolated_process"] = True
    result["condition"] = args.condition
    if fixture_meta is not None:
        result["short_sequence_fixture_meta"] = fixture_meta
    if restore_report is not None:
        result["trainable_init_restore"] = restore_report

    grads = result.pop("selected_grads", None)
    if isinstance(grads, dict):
        result["selected_grads_summary"] = {
            name: {
                "shape": list(t.shape),
                "all_finite": bool(torch.isfinite(t).all().item()),
                "l2_finite": (
                    float(t[torch.isfinite(t)].float().norm().cpu())
                    if torch.isfinite(t).any()
                    else float("nan")
                ),
            }
            for name, t in list(grads.items())[:128]
        }
        grad_path = Path(args.output_json).with_suffix(".grads.pt")
        torch.save(grads, grad_path)
        result["selected_grads_path"] = str(grad_path)

    write_json(Path(args.output_json), result)
    print(
        json.dumps(
            {
                "condition": args.condition,
                "status": result.get("status"),
                "forward_status": result.get("forward_status"),
                "backward_status": result.get("backward_status"),
                "grads_finite": (result.get("trainable_grad_report") or {}).get(
                    "all_trainable_grads_finite"
                ),
            },
            indent=2,
        )
    )
    if result.get("status") != "ok":
        raise SystemExit(1)
    if not bool(
        (result.get("trainable_grad_report") or {}).get("all_trainable_grads_finite")
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
