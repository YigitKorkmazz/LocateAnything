#!/usr/bin/env python3
"""G=8 KL-free LoRA-only projector-frozen diagnostic launcher."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
from rl.runtime import load_resolved_config as load_base_config  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_hybrid_native_g8_loraonly_projectorfrozen_100.yaml"
BASE_CONFIG = HERE / "rl/chestxray8_grpo_hybrid_native.yaml"
OPTIMIZATION = HERE / "splits/production_train90_validation10_seed42/optimization90_of_train80_seed42.jsonl"
VALIDATION = HERE / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
OPTIMIZATION_SHA256 = "4edbed2071e4010c83c45ce8d33d341e0813fba2276150c82c990f1c7e9822f6"
VALIDATION_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
EXECUTE_FLAG = "--execute-authorized-projector-frozen-ablation"
ORIGINAL_PRODUCTION_VALIDATOR = production._validate_config


def validate_loraonly_projectorfrozen_contract(config):
    base = load_base_config(BASE_CONFIG)
    for section in ("prompt", "rollout", "rewards", "runtime_contract", "evaluation"):
        assert config[section] == base[section], f"{section} differs from validated base"
    assert config["model"]["name_or_path"] == "nvidia/LocateAnything-3B"
    assert config["model"]["revision"] == production.EXPECTED_MODEL_REVISION
    assert config["model"]["lora"] == base["model"]["lora"]
    assert config["model"]["base_llm_frozen"] is True
    assert config["model"]["vision_encoder_frozen"] is True
    assert config["model"]["projector_trainable"] is False
    assert config["objective"]["group_size"] == 8
    assert config["objective"]["loss_total"] == "L_GRPO"
    assert config["objective"]["reference_kl"] == {"enabled": False}
    assert config["objective"]["ppo_clip_epsilon"] == base["objective"]["ppo_clip_epsilon"]
    assert config["objective"]["old_policy_sync_interval_optimizer_steps"] == 1
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert float(config["training"]["learning_rate"]) == 2e-5
    assert float(config["training"]["projector_learning_rate"]) == 1e-5
    assert float(config["training"]["max_grad_norm"]) == 1.0
    assert int(config["training"]["max_optimizer_steps"]) == 100
    assert int(config["training"]["checkpoint_interval"]) == 25
    assert config["data"]["train_sha256"] == OPTIMIZATION_SHA256
    assert config["data"]["internal_validation_sha256"] == VALIDATION_SHA256

    # Reuse every production invariant after restoring only the legacy
    # Case-B trainability flag on a validation copy. The real config remains
    # projector-frozen and is checked above and again after model creation.
    validation_copy = copy.deepcopy(config)
    validation_copy["model"]["projector_trainable"] = True
    previous_group_size = production.GROUP_SIZE
    try:
        production.GROUP_SIZE = 8
        ORIGINAL_PRODUCTION_VALIDATOR(validation_copy)
    finally:
        production.GROUP_SIZE = previous_group_size
    assert production._effective_kl_config(config) == {"enabled": False, "beta": 0.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(EXECUTE_FLAG, action="store_true", dest="execute")
    parser.add_argument("--config", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=100)
    parser.add_argument("--max-attempted-groups", type=int, default=800)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--diagnostic-checkpoint-steps", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--validate-checkpoint-only", action="store_true")
    parser.add_argument("--require-fresh-start", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolve_experiment_config(args.config)
    validate_loraonly_projectorfrozen_contract(config)
    if not args.execute:
        raise SystemExit("REFUSED: add {} to run the ablation".format(EXECUTE_FLAG))
    if not 0 < args.max_optimizer_steps <= 100:
        raise SystemExit("projector-frozen diagnostic is capped at 100 optimizer updates")
    if args.checkpoint_interval != 25:
        raise SystemExit("projector-frozen diagnostic requires checkpoint interval 25")
    if args.validate_checkpoint_only and not args.resume_checkpoint:
        raise SystemExit("--validate-checkpoint-only requires --resume-checkpoint")
    if args.require_fresh_start and args.resume_checkpoint:
        raise SystemExit("fresh start cannot also resume")

    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = 504
    production.METRICS_FILENAME = "two_gpu_g8_loraonly_projectorfrozen_metrics.jsonl"
    production.SUMMARY_FILENAME = "two_gpu_g8_loraonly_projectorfrozen_summary.json"
    production.CHECKPOINT_PREFIX = "two_gpu_g8_loraonly_projectorfrozen_step_"
    production.load_resolved_config = lambda unused: config
    production._validate_config = validate_loraonly_projectorfrozen_contract

    forwarded = [
        sys.argv[0],
        "--config", str(Path(args.config).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--max-optimizer-steps", str(args.max_optimizer_steps),
        "--max-attempted-groups", str(args.max_attempted_groups),
        "--optimization-manifest", str(OPTIMIZATION.resolve()),
        "--optimization-manifest-sha256", OPTIMIZATION_SHA256,
        "--validation-manifest", str(VALIDATION.resolve()),
        "--validation-manifest-sha256", VALIDATION_SHA256,
        "--seed", str(args.seed),
    ]
    if args.validate_checkpoint_only:
        forwarded.extend([
            "--diagnostic-checkpoint-steps", "1",
            "--resume-checkpoint", str(Path(args.resume_checkpoint).resolve()),
            "--validate-checkpoint-only",
        ])
    elif args.diagnostic_checkpoint_steps:
        forwarded.extend(["--diagnostic-checkpoint-steps", args.diagnostic_checkpoint_steps])
    else:
        forwarded.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
    if args.resume_checkpoint and not args.validate_checkpoint_only:
        forwarded.extend(["--resume-checkpoint", str(Path(args.resume_checkpoint).resolve())])
    if args.require_fresh_start:
        forwarded.append("--require-fresh-start")
    sys.argv = forwarded
    print(
        json.dumps(
            {
                "event": "launch_g8_loraonly_projectorfrozen",
                "group_size": 8,
                "lora_trainable_tensors": 504,
                "projector_trainable_tensors": 0,
                "total_trainable_tensors": 504,
                "target_steps": args.max_optimizer_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    production.main()


if __name__ == "__main__":
    main()
