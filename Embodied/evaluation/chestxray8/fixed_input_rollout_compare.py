#!/usr/bin/env python3
"""Fixed sample-0/seed-4000180 trusted-vs-sharded rollout comparison."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from fresh_start_regression_audit import DEFAULT_AUDIT_ROOT  # noqa: E402
from rl.rewards import parse_native_locateanything_completion  # noqa: E402
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy,
    build_policy_two_gpu_live_cache,
    generate_rollout_group,
    load_resolved_config,
    load_verified_pairs,
    tokenize_rl_pair,
)
from rl.two_gpu_shard import resolve_locateanything_qwen_decoder  # noqa: E402
from two_gpu_g4_grpo_multistep_smoke import _atomic_save, _atomic_write_json  # noqa: E402


ATTEMPT_SEED = 1_000_045
ROLLOUT_SEED = 4_000_180


def _tensor_meta(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    return {
        "shape": list(detached.shape),
        "stride": list(detached.stride()),
        "layout": str(detached.layout),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_count": int(finite.sum().item()),
        "element_count": int(detached.numel()),
        "finite_min": float(finite_values.min().float().item()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max().float().item()) if finite_values.numel() else None,
    }


def _stage_tensor(output: Any, *, logits: bool = False) -> torch.Tensor | None:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(value, torch.Tensor):
        return None
    return value[:, -6:, :] if logits and value.dim() == 3 else value


def _install_first_forward_hooks(model):
    resolved = resolve_locateanything_qwen_decoder(model)
    captured: Dict[str, torch.Tensor] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    handles = []

    def output_hook(name: str, *, logits: bool = False):
        def hook(_module, _inputs, output):
            if name in captured:
                return
            value = _stage_tensor(output, logits=logits)
            if value is not None:
                metadata[name] = _tensor_meta(value)
                captured[name] = value.detach().to("cpu").clone()
        return hook

    def input_hook(name: str):
        def hook(_module, inputs):
            if name in captured or not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            value = inputs[0]
            metadata[name] = _tensor_meta(value)
            captured[name] = value.detach().to("cpu").clone()
        return hook

    handles.extend(
        [
            resolved.decoder.layers[17].register_forward_hook(output_hook("layer_17_output")),
            resolved.decoder.layers[18].register_forward_pre_hook(input_hook("layer_18_input")),
            resolved.decoder.layers[18].register_forward_hook(output_hook("layer_18_output")),
            resolved.norm.register_forward_hook(output_hook("final_norm_output")),
            resolved.lm_head.register_forward_hook(output_hook("logits_last_6", logits=True)),
        ]
    )
    return captured, metadata, handles


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)


def run_path(mode: str, root: Path) -> None:
    stem = "trusted_single_device" if mode == "single" else "repaired_two_device"
    json_path = root / f"{stem}_sample0_seed4000180.json"
    tensor_path = root / f"{stem}_sample0_seed4000180_stages.pt"
    if json_path.exists() or tensor_path.exists():
        raise FileExistsError(f"refusing to overwrite {stem} artifacts")
    config = load_resolved_config(DEFAULT_HYBRID_NATIVE_CONFIG)
    _seed_everything(int(config["training"]["seed"]))
    if mode == "single":
        if torch.cuda.device_count() != 1:
            raise RuntimeError("single mode requires exactly one visible GPU")
        model, tokenizer, processor, revision = build_policy(config, torch.device("cuda:0"))
        shard = None
    else:
        if torch.cuda.device_count() != 2:
            raise RuntimeError("sharded mode requires exactly two visible GPUs")
        model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(config)
    model.eval()
    pair = load_verified_pairs(config, "train")[0]
    inputs = tokenize_rl_pair(processor, pair, torch.device("cuda:0"), config=config)
    captured, metadata, handles = _install_first_forward_hooks(model)
    events = []
    try:
        with torch.no_grad():
            traces = generate_rollout_group(
                model,
                tokenizer,
                inputs,
                config,
                sample_seed=ATTEMPT_SEED,
                diagnostic_observer=events.append,
            )
    finally:
        for handle in handles:
            handle.remove()
    trace = traces[0]
    parsed = parse_native_locateanything_completion(trace.decoded_text or "")
    report = {
        "format": "fresh_start_fixed_input_rollout_v1",
        "path": stem,
        "model_revision": revision,
        "sample_index": 0,
        "image_index": pair.get("image_index"),
        "image_path": pair.get("image_path"),
        "rendered_prompt": inputs["rendered_prompt"],
        "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
        "attention_mask": inputs["attention_mask"][0].detach().cpu().tolist(),
        "pixel_values": _tensor_meta(inputs["pixel_values"]),
        "image_grid_hws": inputs["image_grid_hws"].detach().cpu().tolist(),
        "attempt_seed": ATTEMPT_SEED,
        "rollout_seed": ROLLOUT_SEED,
        "trace": trace.to_dict(),
        "parser": {
            "format_valid": parsed.format_valid,
            "final_box_norm_1000": parsed.final_box_norm_1000,
            "think_text": parsed.think_text,
            "error": parsed.error,
        },
        "observer_events": [event for event in events if event.get("seed") == ROLLOUT_SEED],
        "first_forward_stage_metadata": metadata,
        "stage_tensor_artifact": str(tensor_path),
        "shard": shard,
        "model_modes": {
            "model_training": model.training,
            "language_model_training": model.language_model.training,
            "projector_training": model.mlp1.training,
        },
    }
    _atomic_save(tensor_path, {"format": "fixed_input_first_forward_stages_v1", "metadata": metadata, "tensors": captured})
    _atomic_write_json(json_path, report)
    del traces, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "written", "output": str(json_path)}))


def _comparison(left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    if left.shape != right.shape:
        return {"shape_equal": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    a = left.float()
    b = right.float()
    absolute = (a - b).abs()
    denominator = torch.maximum(a.abs(), b.abs()).clamp_min(1e-12)
    relative = absolute / denominator
    return {
        "shape_equal": True,
        "exact": bool(torch.equal(left, right)),
        "max_absolute_error": float(absolute.max().item()),
        "mean_absolute_error": float(absolute.mean().item()),
        "max_relative_error": float(relative.max().item()),
        "left_finite_count": int(torch.isfinite(left).sum().item()),
        "right_finite_count": int(torch.isfinite(right).sum().item()),
        "dtype_equal": left.dtype == right.dtype,
        "stride_equal": left.stride() == right.stride(),
        "layout_equal": left.layout == right.layout,
    }


def compare(root: Path) -> None:
    output = root / "trusted_vs_repaired_sample0_seed4000180_comparison.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    single_json = json.loads((root / "trusted_single_device_sample0_seed4000180.json").read_text())
    shard_json = json.loads((root / "repaired_two_device_sample0_seed4000180.json").read_text())
    single = torch.load(root / "trusted_single_device_sample0_seed4000180_stages.pt", map_location="cpu", weights_only=False)
    sharded = torch.load(root / "repaired_two_device_sample0_seed4000180_stages.pt", map_location="cpu", weights_only=False)
    stages = {
        key: _comparison(single["tensors"][key], sharded["tensors"][key])
        for key in single["tensors"].keys() & sharded["tensors"].keys()
    }
    first_divergence = next((key for key in ("layer_17_output", "layer_18_input", "layer_18_output", "final_norm_output", "logits_last_6") if not stages.get(key, {}).get("exact")), None)
    report = {
        "format": "fresh_start_fixed_input_comparison_v1",
        "prompt_text_equal": single_json["rendered_prompt"] == shard_json["rendered_prompt"],
        "prompt_ids_equal": single_json["prompt_token_ids"] == shard_json["prompt_token_ids"],
        "attention_mask_equal": single_json["attention_mask"] == shard_json["attention_mask"],
        "image_path_equal": single_json["image_path"] == shard_json["image_path"],
        "image_grid_equal": single_json["image_grid_hws"] == shard_json["image_grid_hws"],
        "first_forward_stages": stages,
        "first_divergence": first_divergence,
        "generated_token_ids_equal": single_json["trace"]["generated_token_ids"] == shard_json["trace"]["generated_token_ids"],
        "decoded_text_equal": single_json["trace"]["decoded_text"] == shard_json["trace"]["decoded_text"],
        "branch_equal": single_json["trace"]["reward_branch"] == shard_json["trace"]["reward_branch"],
        "parser_equal": single_json["parser"] == shard_json["parser"],
    }
    report["passed"] = bool(
        report["prompt_text_equal"]
        and report["prompt_ids_equal"]
        and report["attention_mask_equal"]
        and report["image_path_equal"]
        and report["image_grid_equal"]
        and first_divergence is None
        and report["generated_token_ids_equal"]
        and report["branch_equal"]
        and report["parser_equal"]
    )
    _atomic_write_json(output, report)
    print(json.dumps({"status": "written", "passed": report["passed"], "output": str(output)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("single", "sharded", "compare"), required=True)
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    args = parser.parse_args()
    root = Path(args.audit_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    compare(root) if args.mode == "compare" else run_path(args.mode, root)


if __name__ == "__main__":
    main()
