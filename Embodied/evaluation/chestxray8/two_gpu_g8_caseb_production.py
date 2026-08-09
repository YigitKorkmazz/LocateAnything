#!/usr/bin/env python3
"""Explicitly gated G=8 Case-B adapter for the validated production runner."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
from rl.runtime import load_resolved_config as load_base_config  # noqa: E402

DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml"
OPTIMIZATION = HERE / "splits/production_train90_validation10_seed42/optimization90_of_train80_seed42.jsonl"
VALIDATION = HERE / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
OPTIMIZATION_SHA256 = "4edbed2071e4010c83c45ce8d33d341e0813fba2276150c82c990f1c7e9822f6"
VALIDATION_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
EXECUTE_FLAG = "--execute-authorized-g8-run"


def deep_merge(base, changes):
    merged = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_experiment_config(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    if not (isinstance(raw, dict) and "base_config" in raw and "overrides" in raw):
        return load_base_config(path)
    base_path = path.parent / raw["base_config"]
    config = deep_merge(load_base_config(base_path), raw["overrides"])
    config["_config_path"] = str(path)
    return config


def validate_g8_caseb_contract(config):
    assert config["model"]["name_or_path"] == "nvidia/LocateAnything-3B"
    assert config["model"]["revision"] == production.EXPECTED_MODEL_REVISION
    assert config["model"]["projector_trainable"] is True
    assert config["objective"]["group_size"] == 8
    assert config["objective"]["loss_total"] == "L_GRPO"
    assert config["objective"]["reference_kl"] == {"enabled": False}
    assert config["policy_state"]["synchronized_parameters"] == ["lora", "mlp1_projector"]
    assert float(config["training"]["learning_rate"]) == 2e-5
    assert float(config["training"]["projector_learning_rate"]) == 1e-5
    assert float(config["training"]["max_grad_norm"]) == 1.0
    assert int(config["training"]["max_optimizer_steps"]) == 150
    assert int(config["training"]["checkpoint_interval"]) == 25
    assert config["data"]["train_sha256"] == OPTIMIZATION_SHA256
    assert config["data"]["internal_validation_sha256"] == VALIDATION_SHA256
    rewards = config["rewards"]
    assert rewards["parser"] == "native_locateanything"
    assert rewards["use_decoder_committed_final_box"] is True
    assert [rewards[name]["weight"] for name in ("format", "spatial", "semantic")] == [1.0, 1.0, 1.0]
    assert rewards["spatial"] == {
        "type": "binary_iou", "weight": 1.0, "iou_threshold": 0.5, "comparison": "greater_than"
    }
    assert rewards["semantic"] == {
        "type": "medclip_roi_text_cosine", "weight": 1.0, "frozen": True,
        "image_input": "native_predicted_roi", "text_input": "original_query", "invalid_box_fallback": 0.0,
    }
    assert config["rollout"]["hybrid"]["logprob_objective"] == "full_trajectory"
    assert config["rollout"]["hybrid"]["score_rejected_pbd_proposals"] is True
    assert config["runtime_contract"] == production.EXPECTED_RUNTIME_CONTRACT
    assert config["training"]["gradient_checkpointing"] is False
    previous = production.GROUP_SIZE
    try:
        production.GROUP_SIZE = 8
        production._validate_config(config)
    finally:
        production.GROUP_SIZE = previous
    assert production._effective_kl_config(config) == {"enabled": False, "beta": 0.0}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(EXECUTE_FLAG, action="store_true", dest="execute")
    parser.add_argument("--config", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-optimizer-steps", type=int, default=150)
    parser.add_argument("--max-attempted-groups", type=int, default=1200)
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
    validate_g8_caseb_contract(config)
    if not args.execute:
        raise SystemExit("REFUSED: add {} to run the validated G=8 path".format(EXECUTE_FLAG))
    if args.validate_checkpoint_only and not args.resume_checkpoint:
        raise SystemExit("--validate-checkpoint-only requires --resume-checkpoint")
    if args.require_fresh_start and args.resume_checkpoint:
        raise SystemExit("fresh start cannot also resume")

    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 6
    production.EXPECTED_TRAINABLE_TENSORS = 510
    production.METRICS_FILENAME = "two_gpu_g8_caseb_metrics.jsonl"
    production.SUMMARY_FILENAME = "two_gpu_g8_caseb_summary.json"
    production.CHECKPOINT_PREFIX = "two_gpu_g8_caseb_step_"
    production.load_resolved_config = lambda unused: config

    forwarded = [
        sys.argv[0], "--config", str(Path(args.config).resolve()),
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
        forwarded.extend(["--diagnostic-checkpoint-steps", "1", "--resume-checkpoint", str(Path(args.resume_checkpoint).resolve()), "--validate-checkpoint-only"])
    elif args.diagnostic_checkpoint_steps:
        forwarded.extend(["--diagnostic-checkpoint-steps", args.diagnostic_checkpoint_steps])
    else:
        forwarded.extend(["--checkpoint-interval", str(args.checkpoint_interval)])
    if args.resume_checkpoint and not args.validate_checkpoint_only:
        forwarded.extend(["--resume-checkpoint", str(Path(args.resume_checkpoint).resolve())])
    if args.require_fresh_start:
        forwarded.append("--require-fresh-start")
    sys.argv = forwarded
    print(json.dumps({"event": "launch_g8_caseb", "group_size": 8, "trainable_tensors": 510, "target_steps": args.max_optimizer_steps}), flush=True)
    production.main()


if __name__ == "__main__":
    main()
