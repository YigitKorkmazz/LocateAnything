#!/usr/bin/env python3
"""Guarded manual launcher for missing conditions B/C/D of the bbox 2x2."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as production  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation as lr_tools  # noqa: E402
from rl.runtime import is_ntp_only_rollout, sha256_file  # noqa: E402
from rl.spatial_density_diagnostics import validate_spatial_density_config  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

EXECUTE_FLAG = "--execute-authorized-bbox-size-2x2"
REFERENCE_CONFIG = HERE / (
    "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_"
    "t1p2_topp0p9_lr1e5_500.yaml"
)
OPTIMIZATION = HERE / (
    "splits/production_train90_validation10_seed42/"
    "optimization90_of_train80_seed42.jsonl"
)
VALIDATION = HERE / (
    "splits/production_train90_validation10_seed42/"
    "validation10_of_train80_seed42.jsonl"
)
OPTIMIZATION_SHA256 = (
    "4edbed2071e4010c83c45ce8d33d341e0813fba2276150c82c990f1c7e9822f6"
)
VALIDATION_SHA256 = (
    "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
)
RESULTS_ROOT = (
    HERE.parent.parent
    / "results"
    / "chestxray8_hybrid_grpo_native"
)
CHECKPOINT_STEPS = (100, 200, 300, 400, 500)

KL004 = {
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

CONDITIONS = {
    "B": {
        "learning_rate": 1e-5,
        "effective_kl": KL004,
        "run_name": "G8_BBOX_SIZE_2X2_B_LR1E5_KL004_500",
        "config": HERE / "rl/chestxray8_bbox_size_2x2_B_lr1e5_kl004_500.yaml",
        "output_dir": RESULTS_ROOT
        / "G8_BBOX_SIZE_2X2_B_LR1E5_KL004_500_FUNCTIONAL_KV_CKPT",
        "file_tag": "bbox_size_2x2_B_lr1e5_kl004_500",
    },
    "C": {
        "learning_rate": 1e-6,
        "effective_kl": {"enabled": False, "beta": 0.0},
        "run_name": "G8_BBOX_SIZE_2X2_C_LR1E6_KL0_500",
        "config": HERE / "rl/chestxray8_bbox_size_2x2_C_lr1e6_kl0_500.yaml",
        "output_dir": RESULTS_ROOT
        / "G8_BBOX_SIZE_2X2_C_LR1E6_KL0_500_FUNCTIONAL_KV_CKPT",
        "file_tag": "bbox_size_2x2_C_lr1e6_kl0_500",
    },
    "D": {
        "learning_rate": 1e-6,
        "effective_kl": KL004,
        "run_name": "G8_BBOX_SIZE_2X2_D_LR1E6_KL004_500",
        "config": HERE / "rl/chestxray8_bbox_size_2x2_D_lr1e6_kl004_500.yaml",
        "output_dir": RESULTS_ROOT
        / "G8_BBOX_SIZE_2X2_D_LR1E6_KL004_500_FUNCTIONAL_KV_CKPT",
        "file_tag": "bbox_size_2x2_D_lr1e6_kl004_500",
    },
}


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _training_only_manifest_metadata(args: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    optimization = Path(args.optimization_manifest).resolve()
    validation = Path(args.validation_manifest).resolve()
    if optimization != OPTIMIZATION.resolve():
        raise RuntimeError("optimization manifest path differs from the pinned contract")
    if validation != VALIDATION.resolve():
        raise RuntimeError("validation metadata path differs from the pinned contract")
    if sha256_file(optimization) != OPTIMIZATION_SHA256:
        raise RuntimeError("optimization manifest hash differs from the pinned contract")
    if sha256_file(validation) != VALIDATION_SHA256:
        raise RuntimeError("validation metadata hash differs from the pinned contract")
    return {
        "optimization_manifest": str(optimization),
        "optimization_manifest_sha256": OPTIMIZATION_SHA256,
        "validation_manifest": str(validation),
        "validation_manifest_sha256": VALIDATION_SHA256,
        "heldout_test_manifest": None,
        "heldout_test_sha256": None,
        "heldout_test_role": "forbidden",
    }


def validate_condition_contract(condition: str, config: Dict[str, Any]) -> Dict[str, Any]:
    spec = CONDITIONS[condition]
    reference = resolve_experiment_config(REFERENCE_CONFIG)
    normalized = copy.deepcopy(config)
    expected = copy.deepcopy(reference)
    normalized.pop("_config_path", None)
    expected.pop("_config_path", None)

    # These are the complete scientific/procedural allowlist relative to A.
    normalized["experiment"] = expected["experiment"]
    normalized.pop("medground_source_audit", None)
    normalized["objective"]["loss_total"] = expected["objective"]["loss_total"]
    normalized["objective"]["reference_kl"] = expected["objective"]["reference_kl"]
    normalized["training"]["learning_rate"] = expected["training"]["learning_rate"]
    assert normalized == expected, (
        f"condition {condition} differs from A outside run name, LR, and KL"
    )

    assert config["experiment"] == spec["run_name"]
    assert config["model"]["name_or_path"] == "nvidia/LocateAnything-3B"
    assert config["model"]["revision"] == production.EXPECTED_MODEL_REVISION
    assert config["model"]["projector_trainable"] is False
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert int(config["objective"]["group_size"]) == 8
    assert float(config["training"]["learning_rate"]) == spec["learning_rate"]
    assert production._effective_kl_config(config) == spec["effective_kl"]
    assert int(config["training"]["seed"]) == 42
    assert int(config["training"]["max_optimizer_steps"]) == 500
    assert int(config["training"]["checkpoint_interval"]) == 100
    assert config["training"]["resume_from_checkpoint"] is None
    assert config["training"]["use_8bit_adam"] is False
    assert "torch.optim.AdamW(" in inspect.getsource(production.build_optimizer)
    assert is_ntp_only_rollout(config)
    assert config["rollout"]["ntp_only"]["generation_mode"] == "slow"
    assert (
        float(config["rollout"]["temperature"]),
        float(config["rollout"]["top_p"]),
        int(config["rollout"]["top_k"]),
        float(config["rollout"]["repetition_penalty"]),
        int(config["rollout"]["max_new_tokens"]),
    ) == (1.2, 0.9, 0, 1.0, 512)
    assert config["runtime_contract"] == production.EXPECTED_RUNTIME_CONTRACT
    assert config["runtime_contract"]["detach_kv"] is False
    assert validate_spatial_density_config(config, group_size=8) == 25

    rewards = config["rewards"]
    assert rewards == reference["rewards"]
    assert [rewards[name]["weight"] for name in ("format", "spatial", "semantic")] == [
        1.0,
        1.0,
        1.0,
    ]
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

    output_dir = Path(spec["output_dir"]).resolve()
    checkpoint_prefix = f"two_gpu_g8_{spec['file_tag']}_step_"
    return {
        "condition": condition,
        "learning_rate": spec["learning_rate"],
        "effective_kl": spec["effective_kl"],
        "seed": 42,
        "group_size": 8,
        "optimizer": "torch.optim.AdamW",
        "trainable_tensors": {"lora": 504, "projector": 0, "total": 504},
        "fresh_start": True,
        "intermediate_validation": False,
        "heldout194_used": False,
        "spatial_density_interval": 25,
        "output_dir": str(output_dir),
        "checkpoint_paths": [
            str(output_dir / f"{checkpoint_prefix}{step}.pt")
            for step in CHECKPOINT_STEPS
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument(EXECUTE_FLAG, action="store_true", dest="execute")
    parser.add_argument("--dry-run-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.execute == args.dry_run_config:
        raise SystemExit(f"choose exactly one of {EXECUTE_FLAG} or --dry-run-config")
    spec = CONDITIONS[args.condition]
    config = resolve_experiment_config(spec["config"])
    contract = validate_condition_contract(args.condition, config)
    print(json.dumps({"event": "bbox_size_2x2_contract", **contract}, sort_keys=True))
    if args.dry_run_config:
        return

    # The trainer also checks the live model and optimizer before its first step.
    production.GROUP_SIZE = 8
    production.EXPECTED_LORA_TENSORS = 504
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = 504
    production.METRICS_FILENAME = f"two_gpu_g8_{spec['file_tag']}_metrics.jsonl"
    production.SUMMARY_FILENAME = f"two_gpu_g8_{spec['file_tag']}_summary.json"
    production.CHECKPOINT_PREFIX = f"two_gpu_g8_{spec['file_tag']}_step_"
    production.PRE_REPLAY_TRACE_METADATA_FILENAME = (
        f"two_gpu_g8_{spec['file_tag']}_pre_replay_trace_metadata.jsonl"
    )
    production.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    production.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    production.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    production.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    production.load_resolved_config = lambda unused: config
    production._validate_config = lambda candidate: validate_condition_contract(
        args.condition, candidate
    )
    production._optimizer_contract_report = lr_tools._optimizer_contract_validator(
        spec["learning_rate"]
    )
    production._manifest_metadata = _training_only_manifest_metadata

    if "--resume-checkpoint" in sys.argv:
        raise SystemExit("matched B/C/D branches forbid resume and require fresh start")
    sys.argv = [
        sys.argv[0],
        "--config",
        str(Path(spec["config"]).resolve()),
        "--output-dir",
        str(Path(spec["output_dir"]).resolve()),
        "--max-optimizer-steps",
        "500",
        "--max-attempted-groups",
        "4000",
        "--checkpoint-interval",
        "100",
        "--optimization-manifest",
        str(OPTIMIZATION.resolve()),
        "--optimization-manifest-sha256",
        OPTIMIZATION_SHA256,
        "--validation-manifest",
        str(VALIDATION.resolve()),
        "--validation-manifest-sha256",
        VALIDATION_SHA256,
        "--seed",
        "42",
        "--require-fresh-start",
    ]
    production.main()


if __name__ == "__main__":
    main()
