#!/usr/bin/env python3
"""Inference-only held-out evaluation of base versus a Hybrid-GRPO checkpoint."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from rl.hybrid_rl import StochasticHybridRLDecoder  # noqa: E402
from rl.pbd_rl import PBDSamplingConfig  # noqa: E402
from rl.rewards import (  # noqa: E402
    BOX_RE,
    COMPLETION_PARSERS,
    MedCLIPSemanticScorer,
    build_reward_pipeline_from_config,
    resolve_parser_name,
)
from rl.runtime import (  # noqa: E402
    DEFAULT_HYBRID_NATIVE_CONFIG,
    build_policy_two_gpu_live_cache,
    load_resolved_config,
    sampling_from_config,
    sha256_file,
    tokenize_rl_pair,
    write_json,
)


PINNED_BASE_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
EXPECTED_TRAINABLE_TENSORS = 510
EXPECTED_LORA_TENSORS = 504
EXPECTED_PROJECTOR_TENSORS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260808)
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _checkpoint_preflight(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state")
    if not isinstance(state, dict) or len(state) != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError("checkpoint does not contain exactly 510 trainable tensors")
    lora_count = sum("lora_" in name for name in state)
    projector_count = sum("mlp1" in name for name in state)
    if (lora_count, projector_count) != (
        EXPECTED_LORA_TENSORS,
        EXPECTED_PROJECTOR_TENSORS,
    ):
        raise RuntimeError(
            f"checkpoint tensor composition mismatch: LoRA={lora_count}, "
            f"projector={projector_count}"
        )
    global_step = payload.get("global_step", payload.get("optimizer_step_count"))
    if int(global_step) != 500:
        raise RuntimeError(f"expected optimizer step 500, found {global_step!r}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError("checkpoint trainable_state contains a non-tensor value")
    result = {
        "path": str(path),
        "sha256": sha256_file(path),
        "loadable": True,
        "global_step": int(global_step),
        "trainable_tensor_count": len(state),
        "lora_tensor_count": lora_count,
        "projector_tensor_count": projector_count,
        "optimizer_constructed_or_loaded": False,
    }
    del payload, state
    gc.collect()
    return result


def _overlap(left: Sequence[Dict[str, Any]], right: Sequence[Dict[str, Any]], key: str) -> List[str]:
    left_values = {str(row[key]) for row in left if row.get(key) is not None}
    right_values = {str(row[key]) for row in right if row.get(key) is not None}
    return sorted(left_values & right_values)


def _preflight(args: argparse.Namespace, config: Dict[str, Any]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    checkpoint = Path(args.checkpoint).resolve()
    manifest = Path(args.manifest).resolve()
    train_manifest = Path(args.train_manifest).resolve()
    if config["model"]["revision"] != PINNED_BASE_REVISION:
        raise RuntimeError("configured LocateAnything base revision is not the pinned revision")
    expected_hash = str(args.manifest_sha256)
    manifest_hash = sha256_file(manifest)
    if manifest_hash != expected_hash or config["data"]["test_sha256"] != expected_hash:
        raise RuntimeError("held-out test manifest SHA-256 mismatch")
    configured_test = Path(config["data"]["test_split"])
    if not configured_test.is_absolute():
        configured_test = CHEST_DIR / configured_test
    if configured_test.resolve() != manifest:
        raise RuntimeError("explicit manifest is not the configured held-out test manifest")
    train_hash = sha256_file(train_manifest)
    if train_hash != config["data"]["train_sha256"]:
        raise RuntimeError("train80 manifest SHA-256 mismatch")
    test_rows, train_rows = _read_jsonl(manifest), _read_jsonl(train_manifest)
    overlaps = {
        key: _overlap(test_rows, train_rows, key)
        for key in ("patient_id", "image_index", "image_path")
    }
    if any(overlaps.values()):
        raise RuntimeError(f"held-out/train80 overlap detected: {overlaps}")
    if int(args.bootstrap_replicates) <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    checkpoint_report = _checkpoint_preflight(checkpoint)
    return test_rows, {
        "status": "passed",
        "checkpoint": checkpoint_report,
        "pinned_base_revision": PINNED_BASE_REVISION,
        "test_manifest": str(manifest),
        "test_manifest_sha256": manifest_hash,
        "test_sample_count": len(test_rows),
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": train_hash,
        "zero_overlap": True,
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "training_performed": False,
        "optimizer_created": False,
        "inference_mode_required": True,
    }


def _load_trainable_checkpoint_exact(model, path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["trainable_state"]
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != expected:
        missing = sorted(expected - set(saved))
        extra = sorted(set(saved) - expected)
        raise RuntimeError(f"checkpoint trainable-state names mismatch: missing={missing}, extra={extra}")
    with torch.no_grad():
        for name, value in saved.items():
            target = named[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
        for name, value in saved.items():
            actual = named[name].detach().to(device="cpu", dtype=value.dtype)
            if not torch.equal(actual, value):
                raise RuntimeError(f"checkpoint tensor failed exact post-load comparison: {name}")
    report = {
        "exact_state_loaded": True,
        "trainable_tensor_count": len(saved),
        "lora_tensor_count": sum("lora_" in name for name in saved),
        "projector_tensor_count": sum("mlp1" in name for name in saved),
        "global_step": int(payload.get("global_step", payload.get("optimizer_step_count"))),
    }
    del payload, saved, named
    gc.collect()
    return report


def _bootstrap(
    values: Sequence[float],
    statistic: Callable[[Iterable[float]], float],
    replicates: int,
    seed: int,
) -> Dict[str, float]:
    if not values:
        return {"point": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng, n = random.Random(seed), len(values)
    point = float(statistic(values))
    samples = sorted(
        float(statistic(values[rng.randrange(n)] for _ in range(n)))
        for _ in range(replicates)
    )
    return {
        "point": point,
        "ci95_low": samples[int(0.025 * (replicates - 1))],
        "ci95_high": samples[int(0.975 * (replicates - 1))],
    }


def _aggregate(rows: Sequence[Dict[str, Any]], replicates: int, seed: int) -> Dict[str, Any]:
    specifications = {
        "valid_native_box_rate": ("valid_native_box", statistics.fmean),
        "malformed_or_no_box_rate": ("malformed_or_no_box", statistics.fmean),
        "mean_iou": ("iou", statistics.fmean),
        "median_iou": ("iou", statistics.median),
        "iou_gt_0_5_accuracy": ("iou_gt_0_5", statistics.fmean),
        "mean_semantic_medclip_score": ("semantic_reward", statistics.fmean),
        "mean_format_reward": ("format_reward", statistics.fmean),
        "mean_spatial_reward": ("spatial_reward", statistics.fmean),
        "mean_semantic_reward": ("semantic_reward", statistics.fmean),
        "mean_total_reward": ("total_reward", statistics.fmean),
        "pbd_branch_rate": ("pbd_branch", statistics.fmean),
        "ntp_fallback_branch_rate": ("ntp_fallback_branch", statistics.fmean),
        "none_branch_rate": ("none_branch", statistics.fmean),
        "mean_generated_tokens": ("generated_token_count", statistics.fmean),
        "truncation_rate": ("truncated", statistics.fmean),
    }
    metrics = {}
    for index, (name, (field, statistic)) in enumerate(specifications.items()):
        values = [float(row[field]) for row in rows]
        metrics[name] = _bootstrap(values, statistic, replicates, seed + index)
    branches = Counter(str(row["committed_branch"]) for row in rows)
    parse_errors = Counter(
        str(row["parse_error"])
        for row in rows
        if row.get("parse_error") is not None
    )
    return {
        "n_samples": len(rows),
        "metrics": metrics,
        "branch_counts": dict(sorted(branches.items())),
        "parse_error_counts": dict(sorted(parse_errors.items())),
        "malformed_samples": [
            {
                "sample_index": row["sample_index"],
                "image_index": row["image_index"],
                "disease": row["disease"],
                "parse_error": row["parse_error"],
                "stop_reason": row["stop_reason"],
                "committed_branch": row["committed_branch"],
            }
            for row in rows
            if row["malformed_or_no_box"]
        ],
    }


def _memory_report(devices: Sequence[torch.device]) -> Dict[str, Any]:
    report = {}
    for index, _device in enumerate(devices):
        with torch.cuda.device(index):
            report[f"cuda:{index}"] = {
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
    return report


def _evaluate_condition(
    label: str,
    checkpoint: Path | None,
    *,
    config: Dict[str, Any],
    pairs: Sequence[Dict[str, Any]],
    devices: Sequence[torch.device],
    output: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    for index, _device in enumerate(devices):
        with torch.cuda.device(index):
            torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(int(config["evaluation"]["seed"]))
    model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(
        config,
        first_device=devices[0],
        second_device=devices[1],
    )
    if revision != PINNED_BASE_REVISION:
        raise RuntimeError(f"loaded unexpected model revision: {revision}")
    checkpoint_load = None
    if checkpoint is not None:
        checkpoint_load = _load_trainable_checkpoint_exact(model, checkpoint)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("a model parameter still has requires_grad=True")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("a model parameter has a populated gradient")
    if any(module.training for module in model.modules()):
        raise RuntimeError("model or submodule is not in eval mode")
    sampling: PBDSamplingConfig = sampling_from_config(config)
    decoder = StochasticHybridRLDecoder(
        model,
        tokenizer,
        sampling=sampling,
        logprob_objective="full_trajectory",
    )
    scorer = MedCLIPSemanticScorer(device=devices[0])
    rewards = build_reward_pipeline_from_config(config, scorer)
    parser = COMPLETION_PARSERS[resolve_parser_name(config)]
    rows: List[Dict[str, Any]] = []
    path = output / f"{label}_per_sample.jsonl"
    with path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        if torch.is_grad_enabled():
            raise RuntimeError("gradient mode is enabled inside evaluation")
        for sample_index, pair in enumerate(pairs):
            seed = int(config["evaluation"]["seed"]) + sample_index
            inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
            trace = decoder.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                image_grid_hws=inputs["image_grid_hws"],
                max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                seed=seed,
                force_first_box_block=False,
            )
            if hasattr(scorer, "model"):
                scorer.model.to(devices[0])
            component = rewards.score_from_trace(trace, pair)
            if hasattr(scorer, "model"):
                scorer.model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
            text = trace.decoded_text or ""
            parsed = parser(text)
            box_count = len(list(BOX_RE.finditer(text)))
            branch = str(getattr(trace, "reward_branch", "none"))
            valid_native_box = bool(
                parsed.format_valid
                and component.format_valid
                and component.geometry_valid
                and getattr(trace, "has_unambiguous_committed_box", False)
            )
            row = {
                "condition": label,
                "sample_index": sample_index,
                "image_index": pair["image_index"],
                "image_path": pair["image_path"],
                "patient_id": pair["patient_id"],
                "disease": pair["disease"],
                "user_query": pair["user_query"],
                "rendered_prompt": inputs["rendered_prompt"],
                "seed": seed,
                "generated_token_count": len(trace.generated_token_ids),
                "generated_token_ids": [int(value) for value in trace.generated_token_ids],
                "generated_token_ids_checksum": hashlib.sha256(
                    json.dumps([int(value) for value in trace.generated_token_ids]).encode()
                ).hexdigest(),
                "decoded_text": text,
                "stop_reason": getattr(trace, "stop_reason", None),
                "truncated": float(bool(trace.truncated)),
                "stopped_on_eos": bool(trace.stopped_on_eos),
                "committed_branch": branch,
                "committed_final_bbox_norm_1000": getattr(
                    trace, "committed_final_box_norm_1000", None
                ),
                "native_box_count": box_count,
                "valid_native_box": float(valid_native_box),
                "malformed_or_no_box": float(not valid_native_box),
                "format_valid": float(component.format_valid),
                "geometry_valid": float(component.geometry_valid),
                "pbd_branch": float(branch == "pbd"),
                "ntp_fallback_branch": float(branch == "ntp_fallback"),
                "none_branch": float(branch == "none"),
                "iou": float(component.final_iou),
                "iou_gt_0_5": float(component.final_iou > 0.50),
                "format_reward": float(component.format_reward),
                "spatial_reward": float(component.spatial_reward),
                "semantic_reward": float(component.semantic_reward),
                "total_reward": float(component.total_reward),
                "parse_error": component.parse_error,
                "parser_diagnostic_error": parsed.error,
                "reward": component.to_dict(),
            }
            rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "event": "heldout_sample_complete",
                        "condition": label,
                        "sample": sample_index + 1,
                        "total": len(pairs),
                        "image_index": pair["image_index"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if torch.is_grad_enabled() is False:
        raise RuntimeError("global gradient mode was unexpectedly changed by evaluation")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("evaluation populated a model gradient")
    aggregate = _aggregate(rows, bootstrap_replicates, bootstrap_seed)
    result = {
        "condition": label,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_load": checkpoint_load,
        "model_revision": revision,
        "model_eval_mode_verified": True,
        "all_parameters_require_grad_false": True,
        "inference_mode_verified": True,
        "optimizer_created": False,
        "sampling": {
            "temperature": sampling.temperature,
            "top_k": sampling.top_k,
            "top_p": sampling.top_p,
            "repetition_penalty": sampling.repetition_penalty,
            "block_size": sampling.block_size,
            "max_new_tokens": int(config["evaluation"]["max_new_tokens"]),
        },
        "shard": shard,
        "per_sample_jsonl": str(path),
        "aggregate": aggregate,
        "peak_memory": _memory_report(devices),
    }
    write_json(output / f"{label}_aggregate.json", result)
    del scorer, rewards, decoder, model, tokenizer, processor
    gc.collect()
    torch.cuda.empty_cache()
    return rows, result


def _paired_comparison(
    base_rows: Sequence[Dict[str, Any]],
    step_rows: Sequence[Dict[str, Any]],
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    if len(base_rows) != len(step_rows):
        raise RuntimeError("paired conditions have different sample counts")
    specifications = {
        "valid_native_box_rate": "valid_native_box",
        "mean_iou": "iou",
        "iou_gt_0_5_accuracy": "iou_gt_0_5",
        "mean_semantic_medclip_score": "semantic_reward",
        "mean_total_reward": "total_reward",
    }
    pairs = []
    for base, step in zip(base_rows, step_rows):
        identity = (base["sample_index"], base["image_index"], base["seed"])
        if identity != (step["sample_index"], step["image_index"], step["seed"]):
            raise RuntimeError("base/step500 sample or seed alignment failure")
        pairs.append((base, step))
    deltas = {}
    for index, (name, field) in enumerate(specifications.items()):
        values = [float(step[field]) - float(base[field]) for base, step in pairs]
        deltas[name] = _bootstrap(
            values,
            statistics.fmean,
            replicates,
            seed + index,
        )
    changed_outcomes = [
        {
            "sample_index": base["sample_index"],
            "image_index": base["image_index"],
            "disease": base["disease"],
            "base_valid_native_box": bool(base["valid_native_box"]),
            "step500_valid_native_box": bool(step["valid_native_box"]),
            "base_iou": base["iou"],
            "step500_iou": step["iou"],
            "base_total_reward": base["total_reward"],
            "step500_total_reward": step["total_reward"],
        }
        for base, step in pairs
        if any(
            base[field] != step[field]
            for field in ("valid_native_box", "iou", "semantic_reward", "total_reward")
        )
    ]
    return {
        "definition": "step500 - base, paired by sample_index/image_index/seed",
        "same_samples_and_seeds_verified": True,
        "n_paired_samples": len(pairs),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "paired_deltas": deltas,
        "changed_outcomes": changed_outcomes,
    }


def _point(result: Dict[str, Any], metric: str) -> float:
    return float(result["aggregate"]["metrics"][metric]["point"])


def _write_markdown(
    output: Path,
    preflight: Dict[str, Any],
    base: Dict[str, Any],
    step: Dict[str, Any],
    comparison: Dict[str, Any],
    command: str,
) -> None:
    metrics = [
        ("Valid native box rate", "valid_native_box_rate"),
        ("Malformed/no-box rate", "malformed_or_no_box_rate"),
        ("Mean IoU", "mean_iou"),
        ("Median IoU", "median_iou"),
        ("IoU > 0.5 accuracy", "iou_gt_0_5_accuracy"),
        ("Mean semantic MedCLIP", "mean_semantic_medclip_score"),
        ("Mean format reward", "mean_format_reward"),
        ("Mean spatial reward", "mean_spatial_reward"),
        ("Mean semantic reward", "mean_semantic_reward"),
        ("Mean total reward", "mean_total_reward"),
        ("PBD branch", "pbd_branch_rate"),
        ("NTP-fallback branch", "ntp_fallback_branch_rate"),
        ("None branch", "none_branch_rate"),
        ("Mean generated tokens", "mean_generated_tokens"),
        ("Truncation rate", "truncation_rate"),
    ]
    lines = [
        "# Final held-out ChestX-ray8 GRPO evaluation",
        "",
        f"- Held-out samples: {preflight['test_sample_count']}",
        f"- Test SHA-256: `{preflight['test_manifest_sha256']}`",
        f"- Base revision: `{preflight['pinned_base_revision']}`",
        "- Split isolation: zero patient, image-ID, and image-path overlap with train80",
        "- Execution: `torch.inference_mode()`, eval mode, all parameters frozen, no optimizer",
        "",
        "## Base versus step 500",
        "",
        "| Metric | Base | Step 500 | Step500 - base |",
        "|---|---:|---:|---:|",
    ]
    paired = comparison["paired_deltas"]
    for label, metric in metrics:
        delta = paired.get(metric, {}).get("point")
        delta_text = f"{delta:.8f}" if delta is not None else "—"
        lines.append(
            f"| {label} | {_point(base, metric):.8f} | {_point(step, metric):.8f} | {delta_text} |"
        )
    lines.extend([
        "",
        "## Paired bootstrap 95% confidence intervals",
        "",
        "| Delta | Point | 95% CI |",
        "|---|---:|---:|",
    ])
    for name, value in paired.items():
        lines.append(
            f"| {name} | {value['point']:.8f} | [{value['ci95_low']:.8f}, {value['ci95_high']:.8f}] |"
        )
    for condition in (base, step):
        lines.extend([
            "",
            f"## Failed/malformed samples: {condition['condition']}",
            "",
            f"Count: {len(condition['aggregate']['malformed_samples'])}",
            "",
            "| Index | Image | Disease | Error | Stop | Branch |",
            "|---:|---|---|---|---|---|",
        ])
        for item in condition["aggregate"]["malformed_samples"]:
            lines.append(
                f"| {item['sample_index']} | {item['image_index']} | {item['disease']} | "
                f"{item['parse_error']} | {item['stop_reason']} | {item['committed_branch']} |"
            )
    lines.extend(["", "## Exact command", "", "```bash", command, "```", ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("requires exactly two visible CUDA devices (physical GPUs 2 and 3)")
    config = load_resolved_config(args.config)
    pairs, preflight = _preflight(args, config)
    write_json(output / "preflight.json", preflight)
    print(json.dumps({"event": "preflight_passed", **preflight}, sort_keys=True), flush=True)
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    base_rows, base = _evaluate_condition(
        "base",
        None,
        config=config,
        pairs=pairs,
        devices=devices,
        output=output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    step_rows, step = _evaluate_condition(
        "step500",
        Path(args.checkpoint).resolve(),
        config=config,
        pairs=pairs,
        devices=devices,
        output=output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    comparison = _paired_comparison(
        base_rows,
        step_rows,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    comparison.update(
        {
            "preflight": preflight,
            "base_aggregate_json": str(output / "base_aggregate.json"),
            "step500_aggregate_json": str(output / "step500_aggregate.json"),
            "training_performed": False,
            "optimizer_created": False,
        }
    )
    write_json(output / "comparison.json", comparison)
    command = " ".join(sys.argv)
    _write_markdown(output, preflight, base, step, comparison, command)
    write_json(
        output / "completion.json",
        {
            "status": "passed",
            "output_dir": str(output),
            "report": str(output / "REPORT.md"),
            "comparison": str(output / "comparison.json"),
            "training_performed": False,
            "optimizer_created": False,
        },
    )
    print(
        json.dumps(
            {
                "event": "heldout_evaluation_complete",
                "output_dir": str(output),
                "report": str(output / "REPORT.md"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
