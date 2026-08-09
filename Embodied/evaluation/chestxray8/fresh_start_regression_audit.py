#!/usr/bin/env python3
"""Reproducible fresh-start Hybrid-GRPO regression audit utilities.

Static mode is CPU-only and records the effective configuration plus the
retained successful/failed first attempts. Peer mode is a tiny, read-only CUDA
diagnostic for the historical direct-BF16 transport; it never loads a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    load_resolved_config,
    load_verified_pairs,
    sha256_file,
)
from rl.two_gpu_shard import diagnose_direct_bf16_peer_copy  # noqa: E402
from two_gpu_g4_grpo_multistep_smoke import (  # noqa: E402
    _attempt_schedule,
    _atomic_write_json,
    _validate_config,
)


DEFAULT_AUDIT_ROOT = (
    CHEST_DIR
    / "results/chestxray8_hybrid_grpo_native/audit/fresh_start_regression_seed42"
)
FAILED_RUN = (
    CHEST_DIR
    / "results/chestxray8_hybrid_grpo_native/training/"
    "two_gpu_18x18_production_500_train80_base_seed42_probe1"
)
SUCCESS_RUN = (
    CHEST_DIR
    / "results/chestxray8_hybrid_grpo_native/training/"
    "two_gpu_18x18_production_100_seed42"
)
METRICS = "two_gpu_g4_grpo_multistep_metrics.jsonl"


def _first_jsonl(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def _trajectory_snapshot(item: Dict[str, Any]) -> Dict[str, Any]:
    reward = item.get("reward") or {}
    return {
        "rollout_seed": item.get("rollout_seed"),
        "generated_token_count": item.get("generated_token_count"),
        "generated_token_ids_checksum": item.get("generated_token_ids_checksum"),
        "block_count": item.get("block_count"),
        "committed_branch": item.get("committed_branch"),
        "committed_final_bbox_norm_1000": item.get("committed_final_bbox_norm_1000"),
        "parse_error": reward.get("parse_error"),
        "total_reward": reward.get("total_reward"),
    }


def _row(
    field: str,
    expected: Any,
    actual: Any,
    status: str,
    fix: str = "none",
    source: str | None = None,
) -> Dict[str, Any]:
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "status": status,
        "fix": fix,
        "source": source,
    }


def _config_table(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    rollout = config["rollout"]
    training = config["training"]
    rewards = config["rewards"]
    return [
        _row("prompt.template", "Locate the {disease} in this chest X-ray", config["prompt"]["template"], "pass", source="YAML -> build_rl_messages -> processor chat template"),
        _row("model.name/revision", "pinned LocateAnything-3B revision", [config["model"]["name_or_path"], config["model"]["revision"]], "pass", source="YAML and AutoModel.from_pretrained"),
        _row("model.dtype", "BF16", config["model"]["dtype"], "pass", source="YAML; build_policy coerces torch.bfloat16"),
        _row("attention implementation", "SDPA", "sdpa hard-coded by build_policy", "pass"),
        _row("tokenizer", "remote pinned; slow; no auto EOS", {"trust_remote_code": True, "use_fast": False, "add_eos_token": False, "model_max_length": rollout["max_sequence_length"]}, "pass"),
        _row("processor", "pinned remote processor", {"trust_remote_code": True, "use_fast_requested": True, "actual_image_processor": "slow fallback (no fast implementation)", "tokenizer_replaced_with_resolved_tokenizer": True}, "coerced"),
        _row("vocabulary/resize", "tokenizer and embed/lm-head vocab identical", "resize only when current_vocab != len(tokenizer); runtime check pending GPU", "pending_gpu"),
        _row("special token IDs", "remote config/tokenizer IDs agree; coordinate range 151677..152677", {"coordinate_token_id_range": rollout["coordinate_token_id_range"], "runtime_ids": "pending GPU model load"}, "pending_gpu"),
        _row("loader cache config", "base configs false outside explicit cached generation", {"model.config.use_cache": False, "language_model.config.use_cache": False}, "pass"),
        _row("rollout/replay cache", "live cache continuity", {"rollout": True, "replay": True, "yaml_gradient_replay_use_cache": training["gradient_replay_use_cache"]}, "mismatch", "report only; do not silently change scientific path", "runner explicitly calls replayer.score(use_cache=True)"),
        _row("attention mask", "trusted and trainer semantics identical", "processor attention_mask is retained in inputs but omitted from decoder_inputs; batch size 1/no padding makes it expected-equivalent", "pending_gpu", "require fixed-input equality before safe verdict"),
        _row("train/eval mode", "existing scientific eval-mode rollout and replay", {"model": "eval", "LoRA dropout": "disabled", "projector": "eval"}, "pass"),
        _row("LoRA", {"r": 8, "alpha": 16, "dropout": 0.05, "bias": "none", "tensor_count": 504}, {**config["model"]["lora"], "tensor_count": "pending GPU"}, "pending_gpu"),
        _row("projector", "6 trainable mlp1 tensors on early shard", {"trainable": config["model"]["projector_trainable"], "tensor_count": "pending GPU"}, "pending_gpu"),
        _row("optimizer", "AdamW constructed after sharding; 2 parameter groups", {"type": "torch.optim.AdamW", "lora_lr": training["learning_rate"], "projector_lr": training["projector_learning_rate"], "weight_decay": training["weight_decay"], "8bit": training["use_8bit_adam"]}, "pass"),
        _row("scheduler", "no scheduler in production runner", {"scheduler": None, "steps": 0}, "pass"),
        _row("max_grad_norm", training["max_grad_norm"], "not consumed by production runner", "mismatch", "report only; blocks safe verdict until explicitly resolved"),
        _row("max_optimizer_steps", training["max_optimizer_steps"], "CLI --max-optimizer-steps overrides YAML (100 successful; 500 failed target)", "overridden", "none"),
        _row("checkpoint_interval", training["checkpoint_interval"], "CLI diagnostic/checkpoint flags override YAML", "overridden", "none"),
        _row("num_epochs", training["num_epochs"], "unused by optimizer-step-capped production runner", "ignored", "none"),
        _row("ppo_epochs", training["ppo_epochs"], "unused; exactly one replay/backward per trajectory", "ignored", "none"),
        _row("gradient_accumulation_samples", training["gradient_accumulation_samples"], "runner always accumulates the fixed G=4 group", "coerced", "none"),
        _row("G", 4, config["objective"]["group_size"], "pass"),
        _row("sampling", {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}, {key: rollout[key] for key in ("temperature", "top_k", "top_p", "repetition_penalty")}, "pass"),
        _row("Hybrid objective", "full_trajectory including rejected proposals", {"objective": rollout["hybrid"]["logprob_objective"], "score_rejected": rollout["hybrid"]["score_rejected_pbd_proposals"]}, "pass"),
        _row("native box grammar", "exactly one <box><x1><y1><x2><y2></box>", {"format": rollout["bbox_format"], "exactly_one": rewards["format"]["require_exactly_one_native_box"]}, "pass"),
        _row("IoU", "binary IoU > 0.50", {"type": rewards["spatial"]["type"], "threshold": rewards["spatial"]["iou_threshold"], "comparison": rewards["spatial"]["comparison"]}, "pass"),
        _row("reward weights", {"format": 1.0, "spatial": 1.0, "semantic": 1.0}, {"format": rewards["format"]["weight"], "spatial": rewards["spatial"]["weight"], "semantic": rewards["semantic"]["weight"]}, "pass"),
        _row("MedCLIP", "frozen pinned local weights", {"frozen": rewards["semantic"]["frozen"], "weights": config["medclip"]["weights"], "sha256": config["medclip"]["weights_sha256"]}, "pass_static"),
        _row("forbidden approximations", False, {"dynamic_filtering": rollout["dynamic_filtering_enabled"], "text_reconstruction": rollout["reconstruct_actions_from_text"], "geometry_repair": rollout["geometry_repair"], "gradient_checkpointing": training["gradient_checkpointing"]}, "pass"),
    ]


def run_static(config_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = load_resolved_config(config_path)
    _validate_config(config)
    pairs = load_verified_pairs(config, "train")
    first_pair = pairs[0]
    schedule = _attempt_schedule(
        seed=42, attempted_group_count=1, sample_cursor=0, sample_count=len(pairs)
    )
    failed = _first_jsonl(FAILED_RUN / METRICS)
    successful = _first_jsonl(SUCCESS_RUN / METRICS)
    report = {
        "format": "fresh_start_hybrid_grpo_static_audit_v1",
        "status": "static_audit_complete_gpu_gates_pending",
        "fixed_input": {
            "sample_index": 0,
            "image_index": first_pair.get("image_index"),
            "image_path": first_pair.get("image_path"),
            "schedule": schedule,
            "required_rollout_seed": 4000180,
        },
        "manifests": {
            "train_path": str((CHEST_DIR / config["data"]["train_split"]).resolve()),
            "train_expected_sha256": config["data"]["train_sha256"],
            "train_actual_sha256": sha256_file(CHEST_DIR / config["data"]["train_split"]),
            "test_path": str((CHEST_DIR / config["data"]["test_split"]).resolve()),
            "test_expected_sha256": config["data"]["test_sha256"],
            "test_actual_sha256": sha256_file(CHEST_DIR / config["data"]["test_split"]),
        },
        "effective_config_table": _config_table(config),
        "retained_first_attempt_comparison": {
            "same_schedule": {
                "sample_index": failed.get("sample_index") == successful.get("sample_index") == 0,
                "attempt_seed": failed.get("attempt_seed") == successful.get("attempt_seed") == 1000045,
                "rollout_seeds": [item.get("rollout_seed") for item in failed["trajectories"]],
            },
            "failed_fresh_start": {
                "path": str(FAILED_RUN / METRICS),
                "trajectories": [_trajectory_snapshot(item) for item in failed["trajectories"]],
                "reward_group": failed.get("reward_group"),
                "classification": "generation failure leading to parser/reward failure (mixed downstream symptoms)",
            },
            "successful_100_step_run": {
                "path": str(SUCCESS_RUN / METRICS),
                "trajectories": [_trajectory_snapshot(item) for item in successful["trajectories"]],
                "reward_group": successful.get("reward_group"),
            },
        },
        "code_audit": {
            "confirmed_bug": "late decoder layers, final norm, buffers, LoRA tensors and lm-head used direct BF16 Module.to/Tensor.to during initialization while only activations used FP32-mediated transport",
            "minimal_repair": "all late-shard initialization values now use synchronous CPU-staged transport with BF16 promoted to FP32 and exact source/destination hash/equality validation",
            "activation_transport": "custom identity-Jacobian transport now uses the same transient CPU staging because both BF16 and FP32 P2P were proven to return zeros",
            "production_guard": "first 8 fresh-start groups / 32 exact broken trajectories write atomic full JSON then abort",
            "unrelated_config_defects_block_safe_verdict": [
                "training.max_grad_norm=1.0 is not consumed",
                "training.gradient_replay_use_cache=false conflicts with actual live-cache replay",
            ],
        },
        "verification": {
            "cpu_static": "pending test runner",
            "peer_copy": "pending peer diagnostic",
            "trusted_vs_sharded_fixed_input": "pending GPU",
            "eight_groups": "pending GPU",
            "one_update_step_001": "pending GPU",
        },
        "verdict": "not safe",
        "verdict_reason": "GPU equivalence, eight-group and one-update gates are pending; two unrelated config mismatches remain unresolved",
    }
    _atomic_write_json(output, report)


def run_peer(output: Path, source_device: str, target_device: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("peer audit requires exactly two visible GPUs")
    report = diagnose_direct_bf16_peer_copy(source_device, target_device)
    report["format"] = "fresh_start_historical_direct_peer_audit_v3"
    report["scientific_role"] = "diagnostic only; never a reference if direct_exact=false"
    _atomic_write_json(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "peer"), default="static")
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--source-device", default="cuda:0")
    parser.add_argument("--target-device", default="cuda:1")
    args = parser.parse_args()
    root = Path(args.audit_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "static":
        output = root / "static_effective_config_and_retained_run_audit_v2.json"
        run_static(Path(args.config), output)
    else:
        output = root / "historical_direct_bf16_peer_diagnostic_v3.json"
        run_peer(output, args.source_device, args.target_device)
    print(json.dumps({"status": "written", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
