#!/usr/bin/env python3
"""Launch the fresh G=8 LR=1e-6, MedGround-KL beta=0.04 run to step 3000."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import two_gpu_g4_grpo_multistep_smoke as p  # noqa: E402
import two_gpu_g8_caseb_production as resolver  # noqa: E402
import two_gpu_g8_loraonly_projectorfrozen_ntponly_t1p2_topp0p9_lr_ablation as lr_tools  # noqa: E402
from rl.runtime import is_ntp_only_rollout, sha256_file  # noqa: E402
from rl.spatial_density_diagnostics import validate_spatial_density_config  # noqa: E402


CONFIG = HERE / (
    "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_"
    "t1p2_topp0p9_lr1e6_kl004_3000.yaml"
)
SCIENTIFIC_REFERENCE_CONFIG = HERE / (
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
OUTPUT_DIR = Path(
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/"
    "chestxray8_hybrid_grpo_native/"
    "G8_MEDGROUND_KL004_LORAONLY_PROJECTORFROZEN_NTPONLY_T1P2_TOPP0P9_"
    "LR1E6_3000_FUNCTIONAL_KV_CKPT"
)
CHECKPOINT_PREFIX = (
    "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e6_kl004_3000_step_"
)
CHECKPOINT_STEPS = (500, 1000, 1500, 2000, 2500, 3000)
EXPECTED_KL = {
    "enabled": True,
    "beta": 0.04,
    "estimator": "exp_ref_minus_policy_minus_ref_minus_policy_minus_one",
    "direction": "policy_to_reference",
    "granularity": "sampled_trajectory_token",
    "normalization": "mean_tokens_per_trajectory_then_mean_group",
    # This is the validated implementation's general trajectory-mask name.
    # Pure-NTP trace validation guarantees that no PBD/rejected tokens exist.
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


def _line_count(path: Path) -> int:
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def _validate_experiment_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    reference = resolver.resolve_experiment_config(SCIENTIFIC_REFERENCE_CONFIG)
    normalized = copy.deepcopy(config)
    expected = copy.deepcopy(reference)
    normalized.pop("_config_path", None)
    expected.pop("_config_path", None)

    # Normalize the only requested scientific changes and procedural metadata.
    normalized["experiment"] = expected["experiment"]
    normalized.pop("medground_source_audit", None)
    normalized["objective"]["loss_total"] = expected["objective"]["loss_total"]
    normalized["objective"]["reference_kl"] = expected["objective"]["reference_kl"]
    normalized["training"]["learning_rate"] = expected["training"]["learning_rate"]
    normalized["training"]["max_optimizer_steps"] = expected["training"]["max_optimizer_steps"]
    normalized["training"]["checkpoint_interval"] = expected["training"]["checkpoint_interval"]
    assert normalized == expected, (
        "experiment differs from selected LR=1e-5/KL-off configuration outside "
        "LR, KL, duration, checkpoint cadence, run name, and KL provenance"
    )

    assert config["model"]["name_or_path"] == "nvidia/LocateAnything-3B"
    assert config["model"]["revision"] == p.EXPECTED_MODEL_REVISION
    assert config["model"]["projector_trainable"] is False
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert config["objective"]["group_size"] == 8
    assert config["objective"]["loss_total"] == "L_GRPO_PLUS_MEDGROUND_KL"
    assert config["objective"]["reference_kl"] == EXPECTED_KL
    assert p._effective_kl_config(config) == EXPECTED_KL
    assert float(config["training"]["learning_rate"]) == 1e-6
    assert int(config["training"]["max_optimizer_steps"]) == 3000
    assert int(config["training"]["checkpoint_interval"]) == 500
    assert int(config["training"]["seed"]) == 42
    assert config["training"]["resume_from_checkpoint"] is None
    assert is_ntp_only_rollout(config)
    rollout = config["rollout"]
    assert rollout["ntp_only"] == {
        "enabled": True,
        "generation_mode": "slow",
        "pbd_enabled": False,
        "mtp_enabled": False,
        "hybrid_fallback_enabled": False,
        "rejected_proposal_trajectory_enabled": False,
    }
    assert rollout["hybrid"]["enabled"] is False
    assert rollout["hybrid"]["score_rejected_pbd_proposals"] is False
    assert (
        float(rollout["temperature"]),
        float(rollout["top_p"]),
        int(rollout["top_k"]),
        float(rollout["repetition_penalty"]),
        int(rollout["max_new_tokens"]),
    ) == (1.2, 0.9, 0, 1.0, 512)

    rewards = config["rewards"]
    assert rewards == reference["rewards"]
    assert rewards["parser"] == "native_locateanything"
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
    assert config["runtime_contract"] == p.EXPECTED_RUNTIME_CONTRACT
    assert config["runtime_contract"]["detach_kv"] is False
    assert validate_spatial_density_config(config, group_size=8) == 25
    assert config["data"]["train_sha256"] == OPTIMIZATION_SHA256
    assert config["data"]["internal_validation_sha256"] == VALIDATION_SHA256
    assert sha256_file(OPTIMIZATION) == OPTIMIZATION_SHA256
    assert sha256_file(VALIDATION) == VALIDATION_SHA256
    assert _line_count(OPTIMIZATION) == 710
    assert _line_count(VALIDATION) == 80
    return {
        "learning_rate": 1e-6,
        "effective_kl": EXPECTED_KL,
        "group_size": 8,
        "trainable_tensors": {"lora": 504, "projector": 0, "total": 504},
        "fresh_start": True,
        "initial_optimizer_step": 0,
        "optimizer_state": "fresh_empty_adamw_state_before_first_optimizer_step",
        "seed": 42,
        "heldout_test_used": False,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
    }


def _training_only_manifest_metadata(args: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Verify only optimization/validation manifests; never open the held-out test."""
    optimization = Path(args.optimization_manifest).resolve()
    validation = Path(args.validation_manifest).resolve()
    optimization_hash = sha256_file(optimization)
    validation_hash = sha256_file(validation)
    if optimization != OPTIMIZATION.resolve() or optimization_hash != OPTIMIZATION_SHA256:
        raise RuntimeError("optimization manifest differs from the pinned contract")
    if validation != VALIDATION.resolve() or validation_hash != VALIDATION_SHA256:
        raise RuntimeError("validation metadata manifest differs from the pinned contract")
    if args.optimization_manifest_sha256 != OPTIMIZATION_SHA256:
        raise RuntimeError("optimization manifest CLI SHA differs from the pinned contract")
    if args.validation_manifest_sha256 != VALIDATION_SHA256:
        raise RuntimeError("validation manifest CLI SHA differs from the pinned contract")
    return {
        "optimization_manifest": str(optimization),
        "optimization_manifest_sha256": optimization_hash,
        "validation_manifest": str(validation),
        "validation_manifest_sha256": validation_hash,
        "heldout_test_manifest": None,
        "heldout_test_sha256": None,
        "heldout_test_role": "forbidden_during_training_and_checkpoint_selection",
    }


def main() -> None:
    config = resolver.resolve_experiment_config(CONFIG)
    contract = _validate_experiment_contract(config)

    # These constants are checked again against the live model/optimizer at startup.
    p.GROUP_SIZE = 8
    p.EXPECTED_LORA_TENSORS = 504
    p.EXPECTED_PROJECTOR_TENSORS = 0
    p.EXPECTED_TRAINABLE_TENSORS = 504
    p.METRICS_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e6_kl004_3000_metrics.jsonl"
    )
    p.SUMMARY_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e6_kl004_3000_summary.json"
    )
    p.CHECKPOINT_PREFIX = CHECKPOINT_PREFIX
    p.PRE_REPLAY_TRACE_METADATA_FILENAME = (
        "two_gpu_g8_ntponly_t1p2_topp0p9_lr1e6_kl004_3000_"
        "pre_replay_trace_metadata.jsonl"
    )
    p.PERSIST_PRE_REPLAY_TRACE_METADATA = True
    p.SEMANTICS_PRESERVING_REPLAY_CUDA_CLEANUP = True
    p.PER_REPLAY_PEAK_MEMORY_DIAGNOSTICS = True
    p.EXACT_FUNCTIONAL_KV_LAYER_CHECKPOINTING = True
    p.load_resolved_config = lambda unused: config
    p._validate_config = _validate_experiment_contract
    p._optimizer_contract_report = lr_tools._optimizer_contract_validator(1e-6)
    p._manifest_metadata = _training_only_manifest_metadata

    # Fresh-start is enforced twice: no resume argument and the trainer guard.
    assert config["training"]["resume_from_checkpoint"] is None
    assert "--resume-checkpoint" not in sys.argv
    sys.argv = [
        sys.argv[0],
        "--config",
        str(CONFIG.resolve()),
        "--output-dir",
        str(OUTPUT_DIR.resolve()),
        "--max-optimizer-steps",
        "3000",
        "--max-attempted-groups",
        "24000",
        "--checkpoint-interval",
        "500",
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
    assert "--resume-checkpoint" not in sys.argv
    assert "--require-fresh-start" in sys.argv
    print({"event": "launch_fresh_lr1e6_medground_kl004_3000", **contract}, flush=True)
    p.main()


if __name__ == "__main__":
    main()
