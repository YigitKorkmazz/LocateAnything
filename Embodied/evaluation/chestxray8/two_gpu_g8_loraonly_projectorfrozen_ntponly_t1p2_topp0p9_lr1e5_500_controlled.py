#!/usr/bin/env python3
"""Launch exactly one validation-gated segment of the LR=1e-5 run."""

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
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_50 as diagnostic  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation as lr_ablation  # noqa: E402
from rl.spatial_density_diagnostics import validate_spatial_density_config  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr1e5_500.yaml"
RUN_NAME = "G8_KL0_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_LR1E5_500_CONTROLLED"
EXECUTE_FLAG = "--execute-authorized-lr1e5-controlled-segment"
TARGET_STEPS = (100, 200, 300, 400, 500)
INTERNAL_VALIDATION_SHA256 = lr_ablation.VALIDATION_SHA256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_long_run_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    baseline = resolve_experiment_config(diagnostic.DEFAULT_SPEC)
    normalized = copy.deepcopy(config)
    expected = copy.deepcopy(baseline)
    normalized.pop("_config_path", None)
    expected.pop("_config_path", None)
    normalized["experiment"] = expected["experiment"]
    normalized["training"]["max_optimizer_steps"] = expected["training"]["max_optimizer_steps"]
    normalized["training"]["checkpoint_interval"] = expected["training"]["checkpoint_interval"]
    assert normalized == expected, (
        "long-run config differs outside experiment/duration/checkpoint cadence"
    )
    lr_ablation.validate_lr_ablation_contract(
        normalized,
        expected_lr=1e-5,
        expected_run_name=diagnostic.RUN_NAME,
    )
    assert config["experiment"] == RUN_NAME
    assert float(config["training"]["learning_rate"]) == 1e-5
    assert int(config["training"]["max_optimizer_steps"]) == 500
    assert int(config["training"]["checkpoint_interval"]) == 100
    assert config["runtime_contract"]["detach_kv"] is False
    assert validate_spatial_density_config(config, group_size=8) == 25
    return {
        "scientific_differences_from_lr1e5_50": {},
        "procedural_differences": {
            "experiment": RUN_NAME,
            "training.max_optimizer_steps": 500,
            "training.checkpoint_interval": 100,
        },
    }


def _checkpoint_metadata(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state") or {}
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "format": payload.get("format"),
        "global_step": int(payload.get("global_step", -1)),
        "seed": int(payload.get("seed", -1)),
        "lora_tensors": sum("lora_" in name for name in state),
        "projector_tensors": sum("mlp1" in name for name in state),
        "trainable_tensors": len(state),
        "runtime_contract": payload.get("effective_runtime_contract"),
        "kl_contract": payload.get("effective_kl_config"),
        "split_metadata": payload.get("split_metadata"),
    }


def validate_guard_receipt(path: Path, *, previous_step: int, target_step: int, resume: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    if payload.get("format") != "lr1e5_500_internal_validation_guard_v1":
        failures.append("wrong guard format")
    if payload.get("run_name") != RUN_NAME:
        failures.append("wrong run name")
    if payload.get("decision") != "continue":
        failures.append("guard did not authorize continuation")
    if int(payload.get("evaluated_checkpoint_step", -1)) != previous_step:
        failures.append("guard evaluated the wrong checkpoint")
    if int(payload.get("authorized_next_target_step", -1)) != target_step:
        failures.append("guard does not authorize this target")
    if payload.get("internal_validation_sha256") != INTERNAL_VALIDATION_SHA256:
        failures.append("guard used the wrong internal-validation manifest")
    checkpoint = payload.get("evaluated_checkpoint") or {}
    if Path(checkpoint.get("path", "")).resolve() != resume:
        failures.append("guard checkpoint path differs from resume checkpoint")
    if checkpoint.get("sha256") != _sha256(resume):
        failures.append("guard checkpoint hash differs from resume checkpoint")
    if failures:
        raise RuntimeError("validation guard receipt rejected: " + "; ".join(failures))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "evaluated_checkpoint_step": previous_step,
        "authorized_next_target_step": target_step,
        "decision": "continue",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(EXECUTE_FLAG, action="store_true", dest="execute")
    parser.add_argument("--config", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-step", type=int, choices=TARGET_STEPS, required=True)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--validation-guard-json")
    parser.add_argument("--checkpoint-preflight-json")
    parser.add_argument("--require-fresh-start", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_experiment_config(args.config)
    equality = validate_long_run_contract(config)
    target = int(args.target_step)
    previous = target - 100
    resume = Path(args.resume_checkpoint).resolve() if args.resume_checkpoint else None

    if target == 100:
        if not args.require_fresh_start or resume or args.validation_guard_json:
            raise SystemExit("step100 requires fresh start and forbids resume/guard inputs")
    else:
        if args.require_fresh_start or resume is None or not args.validation_guard_json:
            raise SystemExit("resumed segments require checkpoint+guard and forbid fresh start")
        metadata = _checkpoint_metadata(resume)
        if metadata != {
            **metadata,
            "global_step": previous,
            "seed": 42,
            "lora_tensors": 504,
            "projector_tensors": 0,
            "trainable_tensors": 504,
        }:
            raise RuntimeError("resume checkpoint tensor/step/seed contract mismatch")
        if metadata["runtime_contract"] != production.EXPECTED_RUNTIME_CONTRACT:
            raise RuntimeError("resume checkpoint runtime contract mismatch")
        if metadata["kl_contract"] != {"enabled": False, "beta": 0.0}:
            raise RuntimeError("resume checkpoint KL contract mismatch")
        validate_guard_receipt(
            Path(args.validation_guard_json).resolve(),
            previous_step=previous,
            target_step=target,
            resume=resume,
        )

    if args.seed != 42:
        raise SystemExit("controlled run requires seed 42")
    if not args.execute and not args.dry_run_config:
        raise SystemExit(f"REFUSED: add {EXECUTE_FLAG} to run one segment")
    preflight = None
    if args.execute:
        if not args.checkpoint_preflight_json:
            raise SystemExit("execution requires the passed LR=1e-5 checkpoint preflight")
        preflight = lr_ablation.validate_checkpoint_preflight_receipt(
            args.checkpoint_preflight_json, expected_lr=1e-5
        )

    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = 504
    production.METRICS_FILENAME = "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_metrics.jsonl"
    production.SUMMARY_FILENAME = "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_summary.json"
    production.CHECKPOINT_PREFIX = "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_step_"
    production.PRE_REPLAY_TRACE_METADATA_FILENAME = "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e5_500_pre_replay_trace_metadata.jsonl"
    production.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    production.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    production.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    production.load_resolved_config = lambda unused: config
    production._validate_config = validate_long_run_contract
    production._optimizer_contract_report = lr_ablation._optimizer_contract_validator(1e-5)

    assert production.GROUP_SIZE == 8
    assert production.EXPECTED_LORA_TENSORS == 504
    assert production.EXPECTED_PROJECTOR_TENSORS == 0
    assert production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING is True
    assert config["runtime_contract"]["detach_kv"] is False

    sys.argv = [
        sys.argv[0],
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--max-optimizer-steps", str(target),
        "--max-attempted-groups", str(target * 8),
        "--checkpoint-interval", "100",
        "--optimization-manifest", str(lr_ablation.OPTIMIZATION.resolve()),
        "--optimization-manifest-sha256", lr_ablation.OPTIMIZATION_SHA256,
        "--validation-manifest", str(lr_ablation.VALIDATION.resolve()),
        "--validation-manifest-sha256", lr_ablation.VALIDATION_SHA256,
        "--seed", "42",
    ]
    if target == 100:
        sys.argv.append("--require-fresh-start")
    else:
        sys.argv.extend(["--resume-checkpoint", str(resume)])
    if args.dry_run_config:
        sys.argv.append("--dry-run-config")
    print(json.dumps({
        "event": "startup_lr1e5_500_controlled_segment",
        "run_name": RUN_NAME,
        "scientific_equality": equality,
        "segment": {"previous_step": previous, "target_step": target},
        "fresh_start": target == 100,
        "resume_checkpoint": str(resume) if resume else None,
        "group_size": 8,
        "ntp_only_slow_mode": True,
        "temperature": 1.2,
        "top_p": 0.9,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "learning_rate": 1e-5,
        "kl_enabled": False,
        "lora_trainable_tensors": 504,
        "projector_trainable_tensors": 0,
        "checkpoint_backend": {
            "enabled": True,
            "scope": "current_policy_differentiable_replay_only",
            "detach_kv": False,
        },
        "checkpoint_preflight_receipt": preflight,
        "checkpoint_at_segment_end": target,
        "next_action_required": "pinned_internal_validation_then_guard",
    }, sort_keys=True), flush=True)
    production.main()


if __name__ == "__main__":
    main()

