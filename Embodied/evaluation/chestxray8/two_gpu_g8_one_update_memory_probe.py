#!/usr/bin/env python3
"""Explicitly gated G=8 one-update RTX3090 memory-probe adapter.

Prepared by the localization audit.  Merely importing this module performs no
work.  A deliberate acknowledgement flag is required before it delegates to
the already validated two-GPU live-cache production runner.
"""

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

DEFAULT_SPEC = HERE / "rl/chestxray8_grpo_hybrid_native_g8_lora_only_probe.yaml"
ACK = "--i-understand-this-runs-one-g8-update"
ORIGINAL_PRODUCTION_VALIDATOR = production._validate_config


def deep_merge(base, changes):
    merged = copy.deepcopy(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_probe_config(spec_path):
    spec_path = Path(spec_path).resolve()
    spec = yaml.safe_load(spec_path.read_text())
    base_path = spec_path.parent / spec["base_config"]
    base = load_base_config(base_path)
    config = deep_merge(base, spec["overrides"])
    config["_config_path"] = str(spec_path)
    validate_probe_contract(config, spec)
    return config


def validate_probe_contract(config, spec):
    assert spec["probe"]["optimizer_updates"] == 1
    assert config["hardware"] == {
        "required_gpu_name_substring": "RTX 3090",
        "visible_gpu_count": 2,
        "decoder_layer_split": [18, 18],
    }
    assert config["objective"]["group_size"] == 8
    assert config["objective"]["loss_total"] == "L_GRPO"
    assert config["objective"]["reference_kl"] == {"enabled": False}
    assert config["model"]["projector_trainable"] is False
    assert config["policy_state"]["synchronized_parameters"] == ["lora"]
    assert float(config["training"]["learning_rate"]) == 1e-5
    assert int(config["training"]["checkpoint_interval"]) == 25
    rewards = config["rewards"]
    assert [rewards[name]["weight"] for name in ("format", "spatial", "semantic")] == [1.0, 1.0, 1.0]
    assert rewards["spatial"]["iou_threshold"] == 0.5
    assert rewards["spatial"]["comparison"] == "greater_than"
    assert rewards["parser"] == "native_locateanything"
    assert rewards["semantic"] == {
        "type": "medclip_roi_text_cosine",
        "weight": 1.0,
        "frozen": True,
        "image_input": "native_predicted_roi",
        "text_input": "original_query",
        "invalid_box_fallback": 0.0,
    }


def validate_with_production_contract(config):
    # Reuse all production invariants.  Its legacy validator requires Case-B,
    # so validate a copy with that one trainability flag restored, after the
    # LoRA-only contract above has already been asserted on the real config.
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
    parser.add_argument(ACK, action="store_true", dest="acknowledged")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-attempted-groups", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = resolve_probe_config(args.spec)
    validate_with_production_contract(config)
    if not args.acknowledged:
        raise SystemExit(
            "REFUSED: prepared only; add {} to deliberately run one G=8 optimizer update".format(ACK)
        )

    production.GROUP_SIZE = 8
    production.EXPECTED_PROJECTOR_TENSORS = 0
    production.EXPECTED_TRAINABLE_TENSORS = production.EXPECTED_LORA_TENSORS
    production.METRICS_FILENAME = "two_gpu_g8_one_update_memory_probe_metrics.jsonl"
    production.SUMMARY_FILENAME = "two_gpu_g8_one_update_memory_probe_summary.json"
    production.CHECKPOINT_PREFIX = "two_gpu_g8_one_update_memory_probe_step_"
    production.load_resolved_config = lambda unused: config
    production._validate_config = lambda actual: validate_with_production_contract(actual)

    sys.argv = [
        sys.argv[0],
        "--config", str(Path(args.spec).resolve()),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--max-optimizer-steps", "1",
        "--max-attempted-groups", str(args.max_attempted_groups),
        "--diagnostic-checkpoint-steps", "1",
        "--require-fresh-start",
        "--optimization-manifest", str((HERE / config["data"]["train_split"]).resolve()),
        "--optimization-manifest-sha256", config["data"]["train_sha256"],
        "--validation-manifest", str((HERE / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl").resolve()),
        "--validation-manifest-sha256", "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838",
        "--seed", str(args.seed),
    ]
    print(json.dumps({"event": "launching_explicitly_acknowledged_probe", "group_size": 8, "optimizer_updates": 1}))
    production.main()


if __name__ == "__main__":
    main()
