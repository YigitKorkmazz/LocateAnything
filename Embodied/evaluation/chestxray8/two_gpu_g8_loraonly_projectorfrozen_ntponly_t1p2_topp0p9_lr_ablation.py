#!/usr/bin/env python3
"""Shared guarded launcher for the two 50-update LoRA-LR diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_production as reference  # noqa: E402
from rl.runtime import is_ntp_only_rollout, sha256_file  # noqa: E402
from rl.spatial_density_diagnostics import validate_spatial_density_config  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

REFERENCE_SPEC = reference.DEFAULT_SPEC
OPTIMIZATION = reference.OPTIMIZATION
VALIDATION = reference.VALIDATION
OPTIMIZATION_SHA256 = reference.OPTIMIZATION_SHA256
VALIDATION_SHA256 = reference.VALIDATION_SHA256
ORIGINAL_OPTIMIZER_CONTRACT_REPORT = production._optimizer_contract_report
PREFLIGHT_GPU0_PEAK_LIMIT_BYTES = reference.PREFLIGHT_GPU0_PEAK_LIMIT_BYTES
ALLOWED_RUNS = {
    1e-5: "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_50",
    5e-6: "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR5E6_50",
}


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def validate_lr_ablation_contract(
    config: Dict[str, Any], *, expected_lr: float, expected_run_name: str
) -> Dict[str, Any]:
    """Prove the resolved config differs only by LR plus run-duration metadata."""
    if expected_lr not in ALLOWED_RUNS:
        raise AssertionError(f"unsupported LR diagnostic: {expected_lr!r}")
    assert ALLOWED_RUNS[expected_lr] == expected_run_name
    assert config["experiment"] == expected_run_name

    baseline = resolve_experiment_config(REFERENCE_SPEC)
    normalized = copy.deepcopy(config)
    normalized.pop("_config_path", None)
    expected = copy.deepcopy(baseline)
    expected.pop("_config_path", None)
    normalized["experiment"] = expected["experiment"]
    normalized["training"]["learning_rate"] = expected["training"]["learning_rate"]
    normalized["training"]["max_optimizer_steps"] = expected["training"]["max_optimizer_steps"]
    assert normalized == expected, (
        "resolved LR diagnostic differs outside the allowlist: experiment, "
        "training.learning_rate, training.max_optimizer_steps"
    )

    # Re-run every invariant already validated for the successful reference.
    reference.validate_sampling_ablation_contract(normalized)

    assert int(config["objective"]["group_size"]) == 8
    assert is_ntp_only_rollout(config)
    assert config["rollout"]["ntp_only"]["generation_mode"] == "slow"
    assert float(config["rollout"]["temperature"]) == 1.2
    assert float(config["rollout"]["top_p"]) == 0.9
    assert int(config["rollout"]["top_k"]) == 0
    assert float(config["rollout"]["repetition_penalty"]) == 1.0
    assert int(config["rollout"]["max_new_tokens"]) == 512
    assert production._effective_kl_config(config) == {"enabled": False, "beta": 0.0}
    assert config["model"]["projector_trainable"] is False
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert float(config["training"]["learning_rate"]) == expected_lr
    assert int(config["training"]["max_optimizer_steps"]) == 50
    assert int(config["training"]["checkpoint_interval"]) == 25
    assert int(config["training"]["seed"]) == 42
    assert config["runtime_contract"]["detach_kv"] is False
    assert validate_spatial_density_config(config, group_size=8) == 25

    rewards = config["rewards"]
    assert rewards["parser"] == "native_locateanything"
    assert rewards["format"] == {
        "type": "binary_native_locateanything",
        "weight": 1.0,
        "require_exactly_one_native_box": True,
        "require_valid_geometry": True,
        "coordinate_range": [0, 1000],
    }
    assert rewards["spatial"] == {
        "type": "binary_iou",
        "weight": 1.0,
        "iou_threshold": 0.5,
        "comparison": "greater_than",
    }
    assert rewards["semantic"] == {
        "type": "medclip_roi_text_cosine",
        "weight": 1.0,
        "frozen": True,
        "image_input": "native_predicted_roi",
        "text_input": "original_query",
        "invalid_box_fallback": 0.0,
    }
    assert sha256_file(OPTIMIZATION) == OPTIMIZATION_SHA256
    assert sha256_file(VALIDATION) == VALIDATION_SHA256
    assert _line_count(OPTIMIZATION) == 710
    assert _line_count(VALIDATION) == 80
    return {
        "scientific_difference": {"training.learning_rate": expected_lr},
        "procedural_differences": {
            "experiment": expected_run_name,
            "training.max_optimizer_steps": 50,
        },
        "optimization_examples": 710,
        "internal_validation_examples": 80,
    }


def validate_known_lr_config(config: Dict[str, Any]) -> Dict[str, Any]:
    lr = float(config["training"]["learning_rate"])
    if lr not in ALLOWED_RUNS:
        raise AssertionError(f"config LR is not a prepared diagnostic: {lr!r}")
    return validate_lr_ablation_contract(
        config, expected_lr=lr, expected_run_name=ALLOWED_RUNS[lr]
    )


def validate_checkpoint_preflight_receipt(
    path: str | Path, *, expected_lr: float
) -> Dict[str, Any]:
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
        "learning_rate": expected_lr,
    }
    run = payload.get("run") or {}
    gradients = run.get("gradient_contract") or {}
    checkpoint = run.get("checkpoint") or {}
    gpu0 = (run.get("memory") or {}).get("cuda:0") or {}
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
    if bool(payload.get("optimizer_constructed")) or bool(payload.get("optimizer_step_called")):
        failures.append("preflight unexpectedly used an optimizer")
    if not bool(checkpoint.get("enabled")):
        failures.append("checkpoint backend was not entered")
    if checkpoint.get("use_reentrant") is not False:
        failures.append("checkpoint backend was not non-reentrant")
    if int(checkpoint.get("wrapped_layer_count", -1)) != 36:
        failures.append("checkpoint backend did not wrap 36 layers")
    calls_by_layer = {
        int(index): int(count)
        for index, count in dict(checkpoint.get("checkpoint_calls_by_layer") or {}).items()
    }
    if {index for index, count in calls_by_layer.items() if count > 0} != set(range(36)):
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
        raise RuntimeError("checkpoint preflight receipt rejected: " + "; ".join(failures))
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "status": payload["status"],
        "learning_rate": expected_lr,
        "generated_token_count": 72,
        "gpu0_peak_allocated": int(gpu0["max_memory_allocated"]),
        "gpu0_peak_limit": PREFLIGHT_GPU0_PEAK_LIMIT_BYTES,
        "lora_gradients_present": 504,
        "lora_gradients_finite": 504,
        "projector_gradients_present": 0,
    }


def _optimizer_contract_validator(expected_lr: float):
    def validate(model, optimizer):
        report = ORIGINAL_OPTIMIZER_CONTRACT_REPORT(model, optimizer)
        if report["optimizer_group_learning_rates"] != [expected_lr]:
            raise RuntimeError(
                f"LoRA-only optimizer must have exactly one LR={expected_lr} group"
            )
        report["optimizer_learning_rate_contract"] = {
            "expected": expected_lr,
            "all_groups_exact": True,
        }
        return report

    return validate


def _parse_args(default_spec: Path, execute_flag: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(execute_flag, action="store_true", dest="execute")
    parser.add_argument("--config", default=str(default_spec))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=50)
    parser.add_argument("--max-attempted-groups", type=int, default=800)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--require-fresh-start", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-preflight-json")
    return parser.parse_args()


def main_for_run(
    *, default_spec: Path, expected_lr: float, run_name: str, file_tag: str, execute_flag: str
) -> None:
    args = _parse_args(default_spec, execute_flag)
    config = resolve_experiment_config(args.config)
    equality = validate_lr_ablation_contract(
        config, expected_lr=expected_lr, expected_run_name=run_name
    )
    if not args.execute and not args.dry_run_config:
        raise SystemExit(f"REFUSED: add {execute_flag} to run the LR diagnostic")
    if not args.require_fresh_start:
        raise SystemExit("LR diagnostic requires --require-fresh-start")
    if args.max_optimizer_steps != 50:
        raise SystemExit("LR diagnostic requires exactly 50 optimizer updates")
    if args.checkpoint_interval != 25:
        raise SystemExit("LR diagnostic requires checkpoint interval 25")
    if args.seed != 42:
        raise SystemExit("LR diagnostic requires seed 42")

    receipt = None
    if args.execute:
        if not args.checkpoint_preflight_json:
            raise SystemExit("production launch requires --checkpoint-preflight-json")
        receipt = validate_checkpoint_preflight_receipt(
            args.checkpoint_preflight_json, expected_lr=expected_lr
        )

    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = 504
    production.METRICS_FILENAME = f"two_gpu_g8_ntponly_t1p2_topp0p9_{file_tag}_metrics.jsonl"
    production.SUMMARY_FILENAME = f"two_gpu_g8_ntponly_t1p2_topp0p9_{file_tag}_summary.json"
    production.CHECKPOINT_PREFIX = f"two_gpu_g8_ntponly_t1p2_topp0p9_{file_tag}_step_"
    production.PRE_REPLAY_TRACE_METADATA_FILENAME = (
        f"two_gpu_g8_ntponly_t1p2_topp0p9_{file_tag}_pre_replay_trace_metadata.jsonl"
    )
    production.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    production.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    production.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    production.load_resolved_config = lambda unused: config
    production._validate_config = lambda current: validate_lr_ablation_contract(
        current, expected_lr=expected_lr, expected_run_name=run_name
    )
    production._optimizer_contract_report = _optimizer_contract_validator(expected_lr)

    assert production.GROUP_SIZE == 8
    assert production.EXPECTED_LORA_TENSORS == 504
    assert production.EXPECTED_PROJECTOR_TENSORS == 0
    assert production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING is True
    assert config["runtime_contract"]["detach_kv"] is False

    sys.argv = [
        sys.argv[0],
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--max-optimizer-steps", "50",
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
                "event": "startup_lr_ablation_contract",
                "run_name": run_name,
                "scientific_equality": equality,
                "fresh_start": True,
                "group_size": 8,
                "generation_mode": "slow",
                "ntp_only": True,
                "temperature": 1.2,
                "top_p": 0.9,
                "top_k": 0,
                "repetition_penalty": 1.0,
                "kl_enabled": False,
                "lora_trainable_tensors": 504,
                "projector_trainable_tensors": 0,
                "optimizer_learning_rate": expected_lr,
                "seed": 42,
                "target_optimizer_updates": 50,
                "checkpoint_steps": [25, 50],
                "pre_replay_trace_metadata": True,
                "graph_free_cuda_cleanup": True,
                "per_replay_peak_memory_diagnostics": True,
                "checkpoint_backend": {
                    "enabled": True,
                    "mode": "functional_kv_decoder_layer",
                    "use_reentrant": False,
                    "current_policy_differentiable_replay_only": True,
                    "detach_kv": False,
                },
                "checkpoint_preflight_receipt": receipt,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    production.main()

