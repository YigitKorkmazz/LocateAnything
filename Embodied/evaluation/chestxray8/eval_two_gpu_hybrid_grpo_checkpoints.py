#!/usr/bin/env python3
"""Evaluation-only comparison of base, step-50, and step-100 Hybrid-GRPO."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR)); sys.path.insert(0, str(REPO_ROOT))
INTERNAL_VALIDATION_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"

from rl.hybrid_rl import StochasticHybridRLDecoder  # noqa: E402
from rl.pbd_rl import PBDSamplingConfig  # noqa: E402
from rl.rewards import BOX_RE, COMPLETION_PARSERS, MedCLIPSemanticScorer, build_reward_pipeline_from_config, resolve_parser_name  # noqa: E402
from rl.runtime import DEFAULT_HYBRID_NATIVE_CONFIG, build_policy_two_gpu_live_cache, load_verified_pairs, sampling_from_config, sha256_file, tokenize_rl_pair, write_json  # noqa: E402
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_HYBRID_NATIVE_CONFIG)); p.add_argument("--output-dir", required=True)
    p.add_argument("--step50-checkpoint"); p.add_argument("--step100-checkpoint")
    p.add_argument("--manifest", default=None, help="immutable JSONL; defaults to configured held-out test split")
    p.add_argument("--manifest-sha256", default=None)
    p.add_argument("--split-label", default="heldout_test")
    p.add_argument("--condition", action="append", default=[], metavar="LABEL=CHECKPOINT",
                   help="repeatable explicit condition; use LABEL=base for untrained base")
    p.add_argument("--bootstrap-replicates", type=int, default=2000); p.add_argument("--bootstrap-seed", type=int, default=20260804)
    return p.parse_args()


def _load_trainable_checkpoint(model, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved, named = payload["trainable_state"], dict(model.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    if set(saved) != expected: raise RuntimeError(f"checkpoint trainable state mismatch: {path}")
    with torch.no_grad():
        for name, value in saved.items(): named[name].copy_(value.to(named[name].device, dtype=named[name].dtype))


def _bootstrap(values: List[float], metric: str, replicates: int, seed: int) -> Dict[str, float]:
    if not values: return {"point": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    fn = statistics.median if metric.startswith("median_") else statistics.fmean
    rng, n = random.Random(seed), len(values)
    samples = sorted(fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(replicates))
    return {"point": float(fn(values)), "ci95_low": float(samples[int(.025 * (replicates - 1))]), "ci95_high": float(samples[int(.975 * (replicates - 1))])}


def _aggregate(rows: List[Dict[str, Any]], replicates: int, seed: int) -> Dict[str, Any]:
    def vals(key): return [float(row[key]) for row in rows]
    def valid_vals(key): return [float(row[key]) for row in rows if row["valid_native_box"]]
    metrics = {"format_valid_rate": vals("format_valid"), "geometry_valid_rate": vals("geometry_valid"),
        "valid_native_box_rate": vals("valid_native_box"),
        "malformed_or_no_box_rate": vals("malformed_or_no_box"),
        "exactly_one_box_rate": vals("exactly_one_box"), "pbd_branch_rate": vals("pbd_branch"), "ntp_branch_rate": vals("ntp_branch"),
        "none_branch_rate": vals("none_branch"), "mean_iou": vals("iou"), "median_iou": vals("iou"),
        "iou_at_025": vals("iou_at_025"), "iou_gt_0_5": vals("iou_gt_0_5"), "mean_medclip_semantic_reward": vals("semantic_reward"),
        "near_full_image_box_rate": vals("near_full_image"), "near_full_image_box_rate_among_valid": valid_vals("near_full_image"),
        "mean_box_area_norm01_among_valid": valid_vals("box_area_norm01"),
        "total_reward_mean": vals("total_reward"), "total_reward_median": vals("total_reward"), "malformed_output_rate": vals("malformed_output"),
        "average_generated_token_count": vals("generated_token_count"), "truncation_rate": vals("truncated")}
    result = {name: _bootstrap(value, name, replicates, seed + index) for index, (name, value) in enumerate(metrics.items())}
    token_counts = vals("generated_token_count")
    branches = {name: sum(row["committed_branch"] == name for row in rows) for name in ("pbd", "ntp_fallback", "none")}
    return {"n_samples": len(rows), "metrics": result, "branch_distribution": branches,
            "truncation_count": sum(int(row["truncated"]) for row in rows),
            "generated_token_count_summary": {"mean": float(statistics.fmean(token_counts)) if token_counts else 0.,
             "p95": float(sorted(token_counts)[max(0, math.ceil(.95 * len(token_counts)) - 1)]) if token_counts else 0.,
             "max": float(max(token_counts)) if token_counts else 0.}}


@torch.inference_mode()
def _evaluate_condition(label: str, checkpoint: Path | None, *, config, pairs, devices, output: Path, bootstrap_replicates: int, bootstrap_seed: int) -> Dict[str, Any]:
    torch.manual_seed(int(config["evaluation"]["seed"]))
    model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(config, first_device=devices[0], second_device=devices[1])
    if checkpoint is not None: _load_trainable_checkpoint(model, checkpoint)
    model.requires_grad_(False); model.eval(); sampling: PBDSamplingConfig = sampling_from_config(config)
    decoder = StochasticHybridRLDecoder(model, tokenizer, sampling=sampling, logprob_objective="full_trajectory")
    scorer = MedCLIPSemanticScorer(device=devices[0]); rewards = build_reward_pipeline_from_config(config, scorer)
    parser = COMPLETION_PARSERS[resolve_parser_name(config)]; rows = []
    for sample_index, pair in enumerate(pairs):
        inputs = tokenize_rl_pair(processor, pair, devices[0], config=config)
        trace = decoder.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], image_grid_hws=inputs["image_grid_hws"],
            max_new_tokens=int(config["evaluation"]["max_new_tokens"]), seed=int(config["evaluation"]["seed"]) + sample_index, force_first_box_block=False)
        if hasattr(scorer, "model"): scorer.model.to(devices[0])
        component = rewards.score_from_trace(trace, pair)
        if hasattr(scorer, "model"): scorer.model.to("cpu"); gc.collect(); torch.cuda.empty_cache()
        text = trace.decoded_text or ""; parsed = parser(text); box_count = len(list(BOX_RE.finditer(text)))
        branch = getattr(trace, "reward_branch", "none")
        final_box = getattr(trace, "committed_final_box_norm_1000", None)
        valid_native = bool(component.format_valid and component.geometry_valid and final_box is not None)
        if valid_native:
            width = (final_box[2] - final_box[0]) / 1000.0
            height = (final_box[3] - final_box[1]) / 1000.0
            box_area = width * height
            near_full = width >= 0.9 and height >= 0.9
        else:
            box_area, near_full = 0.0, False
        rows.append({"condition": label, "sample_index": sample_index, "image_index": pair["image_index"], "disease": pair["disease"],
            "seed": int(config["evaluation"]["seed"]) + sample_index, "generated_token_count": len(trace.generated_token_ids),
            "generated_token_ids_checksum": __import__("hashlib").sha256(json.dumps([int(x) for x in trace.generated_token_ids]).encode()).hexdigest(),
            "committed_branch": branch, "committed_final_bbox_norm_1000": final_box,
            "format_valid": float(component.format_valid), "geometry_valid": float(component.geometry_valid), "exactly_one_box": float(box_count == 1),
            "valid_native_box": float(valid_native), "box_area_norm01": float(box_area), "near_full_image": float(near_full),
            "malformed_or_no_box": float(not valid_native), "truncated": float(bool(getattr(trace, "truncated", False))),
            "pbd_branch": float(branch == "pbd"), "ntp_branch": float(branch == "ntp_fallback"), "none_branch": float(branch == "none"),
            "iou": float(component.final_iou), "iou_at_025": float(component.final_iou >= .25), "iou_gt_0_5": float(component.final_iou > .50),
            "semantic_reward": float(component.semantic_reward), "total_reward": float(component.total_reward),
            "malformed_output": float(parsed.error is not None), "parse_error": parsed.error, "reward": component.to_dict()})
    path = output / f"{label}_per_sample.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    del scorer, decoder, model; gc.collect(); torch.cuda.empty_cache()
    return {"checkpoint": str(checkpoint) if checkpoint else None, "model_revision": revision, "shard": shard, "per_sample_jsonl": str(path),
            "aggregate": _aggregate(rows, bootstrap_replicates, bootstrap_seed)}


def main() -> None:
    args = parse_args(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2: raise RuntimeError("requires CUDA_VISIBLE_DEVICES=2,3")
    config = resolve_experiment_config(args.config)
    if args.manifest:
        manifest = Path(args.manifest).resolve(); actual_hash = sha256_file(manifest)
        if args.manifest_sha256 and actual_hash != args.manifest_sha256: raise RuntimeError("evaluation manifest SHA-256 mismatch")
        if args.split_label == "internal_validation" and actual_hash != INTERNAL_VALIDATION_SHA256:
            raise RuntimeError("internal-validation mode requires the pinned 80-example manifest")
        pairs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        manifest = None; actual_hash = config["data"]["test_sha256"]; pairs = load_verified_pairs(config, "test")
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    report = {"evaluation": "deterministic_seeded_native_hybrid_commit", "config_path": config["_config_path"], "split_label": args.split_label,
        "manifest": str(manifest) if manifest else config["data"]["test_split"], "manifest_sha256": actual_hash,
        "same_samples_seeds_decoding_reward_code": True, "bootstrap_replicates": args.bootstrap_replicates,
        "heldout_test_used": False if args.split_label == "internal_validation" else None,
        "command": "CUDA_VISIBLE_DEVICES=0,1 " + " ".join(sys.argv), "conditions": {}}
    conditions = []
    if args.condition:
        for value in args.condition:
            if "=" not in value: raise ValueError("--condition must be LABEL=CHECKPOINT or LABEL=base")
            label, path = value.split("=", 1); conditions.append((label, None if path == "base" else Path(path).resolve()))
    else:
        if not args.step50_checkpoint or not args.step100_checkpoint:
            raise ValueError("legacy evaluation requires --step50-checkpoint and --step100-checkpoint")
        conditions = [("base_locateanything_3b", None), ("step_050", Path(args.step50_checkpoint).resolve()), ("step_100", Path(args.step100_checkpoint).resolve())]
    for label, checkpoint in conditions:
        report["conditions"][label] = _evaluate_condition(label, checkpoint, config=config, pairs=pairs, devices=devices, output=output,
            bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    if "step_100" in report["conditions"]:
        base = report["conditions"]["step_100"]["aggregate"]["metrics"]
        report["paired_differences_vs_step100"] = {label: {metric: float(value["point"] - base[metric]["point"])
            for metric, value in condition["aggregate"]["metrics"].items()} for label, condition in report["conditions"].items() if label != "step_100"}
    write_json(output / "aggregate_metrics.json", report)
    print(json.dumps({"output": str(output), "aggregate": str(output / "aggregate_metrics.json")}))


if __name__ == "__main__": main()
