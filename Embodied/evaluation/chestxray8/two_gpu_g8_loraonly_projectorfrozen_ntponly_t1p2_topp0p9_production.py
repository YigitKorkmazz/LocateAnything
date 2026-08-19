#!/usr/bin/env python3
"""Guarded launcher for the T=1.2/top-p=0.9 pure-NTP G=8 ablation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_production as previous_ntp  # noqa: E402
from rl.runtime import is_ntp_only_rollout  # noqa: E402
from rl.spatial_density_diagnostics import validate_spatial_density_config  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_100.yaml"
OPTIMIZATION = previous_ntp.OPTIMIZATION
VALIDATION = previous_ntp.VALIDATION
OPTIMIZATION_SHA256 = previous_ntp.OPTIMIZATION_SHA256
VALIDATION_SHA256 = previous_ntp.VALIDATION_SHA256
EXECUTE_FLAG = "--execute-authorized-t1p2-topp0p9-ablation"
ORIGINAL_OPTIMIZER_CONTRACT_REPORT = production._optimizer_contract_report
PREFLIGHT_GPU0_PEAK_LIMIT_BYTES = 15 * 2**30


def validate_checkpoint_preflight_receipt(path: str | Path) -> dict:
    receipt_path = Path(path).resolve()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_scientific_contract = {
        "group_size": 8,
        "temperature": 1.2,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 512,
        "detach_kv": False,
        "projector_trainable": False,
        "learning_rate": 2e-5,
    }
    run = payload.get("run") or {}
    gradients = run.get("gradient_contract") or {}
    checkpoint = run.get("checkpoint") or {}
    memory = run.get("memory") or {}
    gpu0 = memory.get("cuda:0") or {}
    trace = payload.get("trace") or {}
    failures = []
    if payload.get("format") != "ntp_functional_kv_checkpoint_oracle_v1":
        failures.append("wrong oracle format")
    if payload.get("status") != "passed_checkpoint_preflight":
        failures.append("preflight status is not passed_checkpoint_preflight")
    if payload.get("mode") != "preflight" or payload.get("backend") != "checkpoint":
        failures.append("wrong preflight mode/backend")
    if payload.get("scientific_contract") != expected_scientific_contract:
        failures.append("scientific contract mismatch")
    if int(trace.get("generated_token_count", -1)) != 72:
        failures.append("preflight trace is not exactly 72 tokens")
    if payload.get("model_revision") != production.EXPECTED_MODEL_REVISION:
        failures.append("model revision mismatch")
    if bool(payload.get("optimizer_constructed")) or bool(
        payload.get("optimizer_step_called")
    ):
        failures.append("preflight unexpectedly used an optimizer")
    if not bool(checkpoint.get("enabled")):
        failures.append("checkpoint backend was not entered")
    if checkpoint.get("use_reentrant") is not False:
        failures.append("checkpoint backend was not non-reentrant")
    if int(checkpoint.get("wrapped_layer_count", -1)) != 36:
        failures.append("checkpoint backend did not wrap 36 layers")
    calls_by_layer = {
        int(index): int(count)
        for index, count in dict(
            checkpoint.get("checkpoint_calls_by_layer") or {}
        ).items()
    }
    if {index for index, count in calls_by_layer.items() if count > 0} != set(
        range(36)
    ):
        failures.append("preflight did not exercise checkpointing in all 36 layers")
    if int(gpu0.get("max_memory_allocated", PREFLIGHT_GPU0_PEAK_LIMIT_BYTES)) >= PREFLIGHT_GPU0_PEAK_LIMIT_BYTES:
        failures.append("GPU0 peak allocation is not below 15 GiB")
    if int(gradients.get("lora_gradient_tensors_present", -1)) != 504:
        failures.append("not all 504 LoRA gradients are present")
    if int(gradients.get("lora_gradient_tensors_finite", -1)) != 504:
        failures.append("not all 504 LoRA gradients are finite")
    if int(gradients.get("projector_gradient_tensors_present", -1)) != 0:
        failures.append("projector received a preflight gradient")
    if failures:
        raise RuntimeError(
            "checkpoint preflight receipt rejected: " + "; ".join(failures)
        )
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "status": payload["status"],
        "generated_token_count": 72,
        "gpu0_peak_allocated": int(gpu0["max_memory_allocated"]),
        "gpu0_peak_limit": PREFLIGHT_GPU0_PEAK_LIMIT_BYTES,
        "lora_gradients_present": 504,
        "lora_gradients_finite": 504,
        "projector_gradients_present": 0,
    }


def validate_sampling_ablation_contract(config):
    """Assert that temperature and top-p are the only scientific changes."""
    baseline = resolve_experiment_config(previous_ntp.DEFAULT_SPEC)
    for section in (
        "prompt",
        "model",
        "objective",
        "rewards",
        "policy_state",
        "data",
        "training",
        "runtime_contract",
        "evaluation",
        "medclip",
    ):
        assert config[section] == baseline[section], f"{section} differs from prior NTP run"

    expected_rollout = copy.deepcopy(baseline["rollout"])
    expected_rollout["temperature"] = 1.2
    expected_rollout["top_p"] = 0.9
    assert config["rollout"] == expected_rollout
    assert is_ntp_only_rollout(config)
    assert config["rollout"]["ntp_only"]["generation_mode"] == "slow"
    assert int(config["objective"]["group_size"]) == 8
    assert float(config["rollout"]["temperature"]) == 1.2
    assert float(config["rollout"]["top_p"]) == 0.9
    assert int(config["rollout"]["top_k"]) == 0
    assert float(config["rollout"]["repetition_penalty"]) == 1.0
    assert production._effective_kl_config(config) == {"enabled": False, "beta": 0.0}
    assert config["model"]["projector_trainable"] is False
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert float(config["training"]["learning_rate"]) == 2e-5
    assert int(config["training"]["max_optimizer_steps"]) == 100
    assert int(config["training"]["checkpoint_interval"]) == 25
    assert int(config["training"]["seed"]) == 42
    assert validate_spatial_density_config(config, group_size=8) == 25

    # Exercise every previously validated NTP invariant after undoing only the
    # two intended sampling changes.  No production code path is substituted.
    prior_sampling_copy = copy.deepcopy(config)
    prior_sampling_copy["rollout"]["temperature"] = 1.0
    prior_sampling_copy["rollout"]["top_p"] = 1.0
    previous_ntp.validate_ntponly_contract(prior_sampling_copy)


def _optimizer_contract_report_with_lr_assertion(model, optimizer):
    report = ORIGINAL_OPTIMIZER_CONTRACT_REPORT(model, optimizer)
    if report["optimizer_group_learning_rates"] != [2e-5]:
        raise RuntimeError("LoRA-only optimizer must have exactly one LR=2e-5 group")
    report["optimizer_learning_rate_contract"] = {
        "expected": 2e-5,
        "all_groups_exact": True,
    }
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(EXECUTE_FLAG, action="store_true", dest="execute")
    parser.add_argument("--config", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=100)
    parser.add_argument("--max-attempted-groups", type=int, default=800)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--require-fresh-start", action="store_true")
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="validate the fully forwarded contract without CUDA/model loading",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-preflight-json",
        help="required passed 72-token functional-KV replay preflight receipt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolve_experiment_config(args.config)
    validate_sampling_ablation_contract(config)
    if not args.execute and not args.dry_run_config:
        raise SystemExit("REFUSED: add {} to run the ablation".format(EXECUTE_FLAG))
    if not args.require_fresh_start:
        raise SystemExit("sampling ablation requires --require-fresh-start")
    if args.max_optimizer_steps != 100:
        raise SystemExit("sampling diagnostic requires exactly 100 optimizer updates")
    if args.checkpoint_interval != 25:
        raise SystemExit("sampling diagnostic requires checkpoint interval 25")
    if args.seed != 42:
        raise SystemExit("sampling diagnostic requires seed 42")
    preflight_receipt = None
    if args.execute:
        if not args.checkpoint_preflight_json:
            raise SystemExit(
                "production launch requires --checkpoint-preflight-json"
            )
        preflight_receipt = validate_checkpoint_preflight_receipt(
            args.checkpoint_preflight_json
        )

    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = 504
    production.METRICS_FILENAME = "two_gpu_g8_ntponly_t1p2_topp0p9_metrics.jsonl"
    production.SUMMARY_FILENAME = "two_gpu_g8_ntponly_t1p2_topp0p9_summary.json"
    production.CHECKPOINT_PREFIX = "two_gpu_g8_ntponly_t1p2_topp0p9_step_"
    production.PRE_REPLAY_TRACE_METADATA_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_pre_replay_trace_metadata.jsonl"
    )
    production.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    production.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    production.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    production.load_resolved_config = lambda unused: config
    production._validate_config = validate_sampling_ablation_contract
    production._optimizer_contract_report = _optimizer_contract_report_with_lr_assertion

    sys.argv = [
        sys.argv[0],
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--max-optimizer-steps", "100",
        "--max-attempted-groups", str(args.max_attempted_groups),
        "--checkpoint-interval", "25",
        "--optimization-manifest", str(OPTIMIZATION.resolve()),
        "--optimization-manifest-sha256", OPTIMIZATION_SHA256,
        "--validation-manifest", str(VALIDATION.resolve()),
        "--validation-manifest-sha256", VALIDATION_SHA256,
        "--seed", "42",
        "--require-fresh-start",
    ]
    if args.dry_run_config:
        sys.argv.append("--dry-run-config")
    print(
        json.dumps(
            {
                "event": "startup_sampling_ablation_contract",
                "run_name": "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_100",
                "generation_mode": "slow",
                "ntp_only": True,
                "group_size": 8,
                "temperature": 1.2,
                "top_p": 0.9,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "kl_enabled": False,
                "lora_trainable_tensors": 504,
                "projector_trainable_tensors": 0,
                "total_trainable_tensors": 504,
                "optimizer_learning_rate": 2e-5,
                "fresh_start": True,
                "seed": 42,
                "checkpoint_steps": [25, 50, 75, 100],
                "pre_replay_trace_metadata": True,
                "graph_free_cuda_cleanup": True,
                "per_replay_peak_memory_diagnostics": True,
                "checkpoint_backend": {
                    "enabled": True,
                    "mode": "functional_kv_decoder_layer",
                    "use_reentrant": False,
                    "detach_kv": False,
                },
                "checkpoint_preflight_receipt": preflight_receipt,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    production.main()


if __name__ == "__main__":
    main()
