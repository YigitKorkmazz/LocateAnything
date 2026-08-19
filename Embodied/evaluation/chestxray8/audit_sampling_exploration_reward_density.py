#!/usr/bin/env python3
"""Offline frozen-base sampling sweep for spatial reward density (no training).

Uses the pinned 80-sample internal-validation split only. Never touches the
194 held-out test set. Model weights are not updated.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

from eval_locateanything_bbox import box_iou  # noqa: E402
from rl.pbd_rl import PBDSamplingConfig  # noqa: E402
from rl.rewards import (  # noqa: E402
    COMPLETION_PARSERS,
    resolve_parser_name,
    spatial_reward_from_box,
)
from rl.runtime import (  # noqa: E402
    build_policy,
    build_policy_two_gpu_live_cache,
    build_rollout_decoder,
    is_ntp_only_rollout,
    sha256_file,
    tokenize_rl_pair,
    write_json,
)
from two_gpu_g8_caseb_production import resolve_experiment_config  # noqa: E402

DEFAULT_CONFIG = CHEST_DIR / "rl/chestxray8_grpo_native_g8_loraonly_projectorfrozen_ntponly_100.yaml"
DEFAULT_MANIFEST = (
    CHEST_DIR
    / "splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl"
)
EXPECTED_MANIFEST_SHA256 = "f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838"
TEMPERATURES = (0.7, 1.0, 1.2, 1.5)
TOP_PS = (0.9, 0.95, 1.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--manifest-sha256", default=EXPECTED_MANIFEST_SHA256)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-examples", type=int, default=40)
    p.add_argument("--rollouts-per-example", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--group-sims", type=int, default=2000)
    p.add_argument("--temperatures", default=",".join(str(x) for x in TEMPERATURES))
    p.add_argument("--top-ps", default=",".join(str(x) for x in TOP_PS))
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="skip generation; aggregate existing condition JSONLs",
    )
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="place the entire frozen policy on the sole visible CUDA device "
        "(cuda:0 after CUDA_VISIBLE_DEVICES remapping). Required for one "
        "independent worker replica per physical GPU.",
    )
    return p.parse_args()


def _visible_cuda_mapping() -> Dict[str, Any]:
    """Report process-local CUDA view vs host CUDA_VISIBLE_DEVICES."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    names = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            names.append(
                {
                    "torch_device": f"cuda:{index}",
                    "name": props.name,
                    "total_memory_mib": int(props.total_memory / 2**20),
                }
            )
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "torch_cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
        "devices": names,
    }


def _load_pairs(path: Path, expected_sha: str, n_examples: int, seed: int) -> List[Dict[str, Any]]:
    actual = sha256_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"manifest sha mismatch: {actual}")
    pairs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(pairs) != 80:
        raise RuntimeError(f"expected 80-sample internal validation, found {len(pairs)}")
    if n_examples > len(pairs):
        raise RuntimeError("n-examples exceeds manifest size")
    # Fixed subset of the 80-split; not held-out test.
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(pairs)), n_examples))
    return [pairs[i] for i in indices]


def _area_fraction(box: Optional[Sequence[float]]) -> Optional[float]:
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, (x2 - x1) / 1000.0) * max(0.0, (y2 - y1) / 1000.0)


def _coord_std(boxes: Sequence[Sequence[float]]) -> float:
    if len(boxes) < 2:
        return 0.0
    arr = np.asarray(boxes, dtype=np.float64)
    return float(arr.std(axis=0).mean())


def _mean_pairwise_iou(boxes: Sequence[Sequence[float]]) -> float:
    if len(boxes) < 2:
        return float("nan")
    vals = [box_iou(a, b) for a, b in combinations(boxes, 2)]
    return float(statistics.fmean(vals)) if vals else float("nan")


def _score_rollout(trace, pair, parser) -> Dict[str, Any]:
    text = trace.decoded_text or ""
    parsed = parser(text)
    box = getattr(trace, "committed_final_box_norm_1000", None)
    unambiguous = bool(getattr(trace, "has_unambiguous_committed_box", False))
    if unambiguous and box is not None:
        spatial, iou = spatial_reward_from_box(
            box, pair["gt_boxes_norm_1000"], threshold=0.5
        )
        spatial = float(spatial)
        iou = float(iou)
    else:
        spatial, iou = 0.0, 0.0
    valid = bool(parsed.format_valid and unambiguous and box is not None)
    area = _area_fraction(box) if valid else None
    near_full = False
    if valid and box is not None:
        width = (box[2] - box[0]) / 1000.0
        height = (box[3] - box[1]) / 1000.0
        near_full = width >= 0.9 and height >= 0.9
    return {
        "iou": iou,
        "spatial_reward": spatial,
        "format_valid": float(parsed.format_valid),
        "valid_native_box": float(valid),
        "near_full_image": float(near_full),
        "box_area_norm01": float(area) if area is not None else 0.0,
        "committed_final_bbox_norm_1000": list(box) if box is not None else None,
        "reward_branch": getattr(trace, "reward_branch", None),
        "truncated": float(bool(getattr(trace, "truncated", False))),
        "generated_token_count": len(trace.generated_token_ids),
        "parse_error": parsed.error,
    }


def _simulate_groups_from_pools(
    pools: Sequence[Sequence[float]],
    *,
    group_size: int,
    n_sims: int,
    seed: int,
) -> Dict[str, float]:
    usable = [list(p) for p in pools if len(p) >= group_size]
    if not usable:
        return {
            "P_all_spatial_0": float("nan"),
            "P_mixed_spatial": float("nan"),
            "P_all_spatial_1": float("nan"),
            "expected_positives_per_group": float("nan"),
            "P_group_has_ge1_iou_gt_0_5": float("nan"),
        }
    rng = random.Random(seed)
    all0 = mixed = all1 = positives = has_pos = 0
    for _ in range(n_sims):
        pool = usable[rng.randrange(len(usable))]
        chosen = rng.sample(pool, group_size)
        spat = [1 if iou > 0.5 else 0 for iou in chosen]
        n_pos = sum(spat)
        positives += n_pos
        if n_pos == 0:
            all0 += 1
        elif n_pos == group_size:
            all1 += 1
        else:
            mixed += 1
        if n_pos >= 1:
            has_pos += 1
    n = float(n_sims)
    return {
        "P_all_spatial_0": all0 / n,
        "P_mixed_spatial": mixed / n,
        "P_all_spatial_1": all1 / n,
        "expected_positives_per_group": positives / n,
        "P_group_has_ge1_iou_gt_0_5": has_pos / n,
    }


def _aggregate_condition(
    rows: List[Dict[str, Any]],
    *,
    group_sims: int,
    seed: int,
) -> Dict[str, Any]:
    ious = [float(r["iou"]) for r in rows]
    n = len(ious)
    if n == 0:
        raise RuntimeError("no rows to aggregate")

    def rate(pred):
        return float(np.mean([pred(r) for r in rows]))

    # per-prompt pools
    by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(int(row["sample_index"]), []).append(row)
    iou_pools = [[float(r["iou"]) for r in rs] for rs in by_sample.values()]
    pairwise = []
    coord_stds = []
    for rs in by_sample.values():
        boxes = [
            r["committed_final_bbox_norm_1000"]
            for r in rs
            if r["valid_native_box"] and r["committed_final_bbox_norm_1000"] is not None
        ]
        if len(boxes) >= 2:
            pairwise.append(_mean_pairwise_iou(boxes))
            coord_stds.append(_coord_std(boxes))
    qs = np.quantile(ious, [0.5, 0.9, 0.95, 0.99])
    sims = {
        f"G{g}": _simulate_groups_from_pools(
            iou_pools, group_size=g, n_sims=group_sims, seed=seed + g
        )
        for g in (4, 8, 16)
    }
    return {
        "n_rollouts": n,
        "n_examples": len(by_sample),
        "rollouts_per_example_mean": float(np.mean([len(v) for v in by_sample.values()])),
        "mean_iou": float(statistics.fmean(ious)),
        "median_iou": float(statistics.median(ious)),
        "P_iou_gt_0_1": float(np.mean([x > 0.1 for x in ious])),
        "P_iou_gt_0_2": float(np.mean([x > 0.2 for x in ious])),
        "P_iou_gt_0_3": float(np.mean([x > 0.3 for x in ious])),
        "P_iou_gt_0_4": float(np.mean([x > 0.4 for x in ious])),
        "P_iou_gt_0_5": float(np.mean([x > 0.5 for x in ious])),
        "iou_p90": float(qs[1]),
        "iou_p95": float(qs[2]),
        "iou_p99": float(qs[3]),
        "iou_max": float(max(ious)),
        "valid_box_rate": rate(lambda r: r["valid_native_box"] > 0.5),
        "near_full_box_rate": rate(lambda r: r["near_full_image"] > 0.5),
        "mean_box_area_fraction_among_valid": float(
            statistics.fmean(
                [r["box_area_norm01"] for r in rows if r["valid_native_box"] > 0.5]
            )
            if any(r["valid_native_box"] > 0.5 for r in rows)
            else 0.0
        ),
        "mean_pairwise_bbox_iou_within_prompt": float(
            statistics.fmean(pairwise) if pairwise else float("nan")
        ),
        "mean_coordinate_std_within_prompt": float(
            statistics.fmean(coord_stds) if coord_stds else 0.0
        ),
        "malformed_or_no_box_rate": rate(lambda r: r["valid_native_box"] < 0.5),
        "truncation_rate": rate(lambda r: r["truncated"] > 0.5),
        "simulated_groups": sims,
        "promising_screen": {
            "P_iou_gt_0_5_ge_5pct": float(np.mean([x > 0.5 for x in ious])) >= 0.05,
            "G8_mixed_ge_25pct": sims["G8"]["P_mixed_spatial"] >= 0.25
            if not math.isnan(sims["G8"]["P_mixed_spatial"])
            else False,
        },
    }


def _condition_tag(temperature: float, top_p: float) -> str:
    return f"temp{temperature:g}_topp{top_p:g}"


@torch.inference_mode()
def _generate_condition(
    *,
    model,
    tokenizer,
    processor,
    config: Dict[str, Any],
    pairs: Sequence[Dict[str, Any]],
    temperature: float,
    top_p: float,
    rollouts_per_example: int,
    seed: int,
    output_jsonl: Path,
) -> List[Dict[str, Any]]:
    # Mutate a shallow copy of rollout sampling only for this condition.
    local = json.loads(json.dumps(config))
    local["rollout"]["temperature"] = float(temperature)
    local["rollout"]["top_p"] = float(top_p)
    local["rollout"]["top_k"] = 0
    decoder = build_rollout_decoder(model, tokenizer, local)
    parser = COMPLETION_PARSERS[resolve_parser_name(config)]
    devices0 = next(model.parameters()).device
    # True frozen base: disable randomly-initialized LoRA adapters (B=0 already,
    # but disable_adapter is the explicit base-policy path).
    disable = getattr(model.language_model, "disable_adapter", None)
    rows: List[Dict[str, Any]] = []
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if output_jsonl.exists():
        output_jsonl.unlink()

    from contextlib import nullcontext

    adapter_ctx = disable() if callable(disable) else nullcontext()
    with adapter_ctx:
        model.eval()
        for sample_index, pair in enumerate(pairs):
            inputs = tokenize_rl_pair(processor, pair, devices0, config=config)
            for rollout_index in range(rollouts_per_example):
                rollout_seed = (
                    int(seed) * 1_000_003
                    + sample_index * 10_007
                    + rollout_index * 97
                    + int(round(temperature * 1000))
                    + int(round(top_p * 1000))
                )
                trace = decoder.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    image_grid_hws=inputs["image_grid_hws"],
                    max_new_tokens=int(config["evaluation"]["max_new_tokens"]),
                    seed=rollout_seed,
                    force_first_box_block=False,
                )
                scored = _score_rollout(trace, pair, parser)
                row = {
                    "sample_index": sample_index,
                    "image_index": pair.get("image_index"),
                    "disease": pair.get("disease"),
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": 0,
                    "rollout_index": rollout_index,
                    "rollout_seed": rollout_seed,
                    "decoding_mode": "ntp_only" if is_ntp_only_rollout(config) else "hybrid",
                    **scored,
                }
                rows.append(row)
                with output_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "event": "sample_complete",
                        "temperature": temperature,
                        "top_p": top_p,
                        "sample_index": sample_index,
                        "rollouts": rollouts_per_example,
                    }
                ),
                flush=True,
            )
    del decoder
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def _rank_conditions(aggregates: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for tag, agg in aggregates.items():
        ranked.append(
            {
                "condition": tag,
                "temperature": agg["temperature"],
                "top_p": agg["top_p"],
                "mean_iou": agg["mean_iou"],
                "P_iou_gt_0_5": agg["P_iou_gt_0_5"],
                "G8_mixed_rate": agg["simulated_groups"]["G8"]["P_mixed_spatial"],
                "valid_box_rate": agg["valid_box_rate"],
                "near_full_box_rate": agg["near_full_box_rate"],
                "mean_pairwise_bbox_iou": agg["mean_pairwise_bbox_iou_within_prompt"],
            }
        )
    ranked.sort(
        key=lambda r: (
            -float(r["G8_mixed_rate"]) if not math.isnan(r["G8_mixed_rate"]) else 1e9,
            -float(r["mean_iou"]),
            -float(r["valid_box_rate"]),
        )
    )
    return ranked


def _verdict(ranked: Sequence[Dict[str, Any]], aggregates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    production_mixed = 0.07
    production_p05 = 0.014
    useful = []
    for row in ranked:
        tag = row["condition"]
        agg = aggregates[tag]
        destructive = (
            agg["valid_box_rate"] < 0.40
            or agg["near_full_box_rate"] > 0.50
            or agg["mean_iou"] < 0.03
        )
        promising = (
            agg["P_iou_gt_0_5"] >= 0.05 or agg["simulated_groups"]["G8"]["P_mixed_spatial"] >= 0.25
        )
        improved = (
            agg["P_iou_gt_0_5"] >= 2 * production_p05
            or agg["simulated_groups"]["G8"]["P_mixed_spatial"] >= 2 * production_mixed
        )
        useful.append(
            {
                "condition": tag,
                "promising": promising,
                "improved_vs_production": improved,
                "destructive": destructive,
                "usable": promising and not destructive,
            }
        )
    usable_rows = [u for u in useful if u["usable"]]
    improved_non_destructive = [
        u for u in useful if u["improved_vs_production"] and not u["destructive"]
    ]
    if usable_rows:
        code, text = "A", "SAMPLING CAN RESCUE REWARD DENSITY"
        recommended = usable_rows[0]["condition"]
    elif improved_non_destructive:
        code, text = "B", "SAMPLING HELPS BUT NOT ENOUGH"
        recommended = improved_non_destructive[0]["condition"]
    else:
        # pick best non-destructive by ranking if any
        nondestructive = [u for u in useful if not u["destructive"]]
        code, text = "C", "SAMPLING CANNOT RESCUE IT"
        recommended = nondestructive[0]["condition"] if nondestructive else ranked[0]["condition"]
    return {
        "code": code,
        "text": text,
        "recommended_condition": recommended,
        "condition_flags": useful,
        "production_reference": {
            "mixed_spatial_groups": "6-7%",
            "rollout_iou_gt_0_5": "1.1-1.4%",
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = resolve_experiment_config(args.config)
    if not is_ntp_only_rollout(config):
        raise RuntimeError("this audit expects an NTP-only config")
    temperatures = [float(x) for x in args.temperatures.split(",") if x.strip()]
    top_ps = [float(x) for x in args.top_ps.split(",") if x.strip()]
    pairs = _load_pairs(
        Path(args.manifest).resolve(),
        args.manifest_sha256,
        args.n_examples,
        args.seed,
    )
    meta = {
        "role": "offline_frozen_base_sampling_exploration_audit",
        "heldout_test_used": False,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": args.manifest_sha256,
        "n_examples": len(pairs),
        "rollouts_per_example": args.rollouts_per_example,
        "temperatures": temperatures,
        "top_ps": top_ps,
        "decoding": "ntp_only",
        "model_updates": False,
        "rewards_modified": False,
        "example_image_indices": [p.get("image_index") for p in pairs],
    }
    write_json(output / "RUN_META.json", meta)

    aggregates: Dict[str, Dict[str, Any]] = {}
    if not args.analyze_only:
        cuda_map = _visible_cuda_mapping()
        print(json.dumps({"event": "cuda_mapping", **cuda_map}, sort_keys=True), flush=True)
        torch.manual_seed(args.seed)
        if args.single_gpu:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError(
                    "--single-gpu requires exactly one visible CUDA device "
                    f"(saw {cuda_map})"
                )
            device = torch.device("cuda:0")
            model, tokenizer, processor, revision = build_policy(config, device)
            shard = {
                "mode": "single_gpu",
                "device": str(device),
                "CUDA_VISIBLE_DEVICES": cuda_map["CUDA_VISIBLE_DEVICES"],
                "physical_gpu_via_visible_devices": cuda_map["CUDA_VISIBLE_DEVICES"],
            }
        else:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
                raise RuntimeError("default mode requires exactly two visible CUDA devices")
            devices = [torch.device("cuda:0"), torch.device("cuda:1")]
            model, tokenizer, processor, revision, shard = build_policy_two_gpu_live_cache(
                config, first_device=devices[0], second_device=devices[1]
            )
        model.requires_grad_(False)
        model.eval()
        # Fail loudly if any parameter landed off the intended single device.
        if args.single_gpu:
            bad = sorted(
                {
                    str(parameter.device)
                    for parameter in model.parameters()
                    if parameter.device.type != "cuda" or parameter.device.index not in (0, None)
                }
            )
            if bad:
                raise RuntimeError(f"single-gpu replica has parameters off cuda:0: {bad}")
        load_name = (
            "model_load_"
            + "_".join(_condition_tag(t, p) for t in temperatures for p in top_ps)
            + ".json"
            if args.single_gpu
            else "model_load.json"
        )
        write_json(
            output / load_name,
            {
                "revision": revision,
                "shard": shard,
                "checkpoint": None,
                "adapters": "disabled_for_base",
                "cuda_mapping": cuda_map,
                "single_gpu": bool(args.single_gpu),
            },
        )
        for temperature in temperatures:
            for top_p in top_ps:
                tag = _condition_tag(temperature, top_p)
                print(
                    json.dumps({"event": "condition_start", "condition": tag}),
                    flush=True,
                )
                jsonl = output / f"{tag}_per_rollout.jsonl"
                rows = _generate_condition(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    config=config,
                    pairs=pairs,
                    temperature=temperature,
                    top_p=top_p,
                    rollouts_per_example=args.rollouts_per_example,
                    seed=args.seed,
                    output_jsonl=jsonl,
                )
                agg = _aggregate_condition(rows, group_sims=args.group_sims, seed=args.seed)
                agg["temperature"] = temperature
                agg["top_p"] = top_p
                agg["per_rollout_jsonl"] = str(jsonl)
                aggregates[tag] = agg
                write_json(output / f"{tag}_aggregate.json", agg)
                print(
                    json.dumps(
                        {
                            "event": "condition_done",
                            "condition": tag,
                            "P_iou_gt_0_5": agg["P_iou_gt_0_5"],
                            "G8_mixed": agg["simulated_groups"]["G8"]["P_mixed_spatial"],
                            "mean_iou": agg["mean_iou"],
                            "valid_box_rate": agg["valid_box_rate"],
                        }
                    ),
                    flush=True,
                )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        for temperature in temperatures:
            for top_p in top_ps:
                tag = _condition_tag(temperature, top_p)
                jsonl = output / f"{tag}_per_rollout.jsonl"
                if not jsonl.exists():
                    raise FileNotFoundError(jsonl)
                rows = [
                    json.loads(line)
                    for line in jsonl.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                agg = _aggregate_condition(rows, group_sims=args.group_sims, seed=args.seed)
                agg["temperature"] = temperature
                agg["top_p"] = top_p
                agg["per_rollout_jsonl"] = str(jsonl)
                aggregates[tag] = agg
                write_json(output / f"{tag}_aggregate.json", agg)

    ranked = _rank_conditions(aggregates)
    verdict = _verdict(ranked, aggregates)
    summary = {
        "meta": meta,
        "production_reference": verdict["production_reference"],
        "ranking_table": ranked,
        "aggregates": aggregates,
        "verdict": verdict,
    }
    write_json(output / "sampling_exploration_summary.json", summary)

    lines = [
        "# Offline sampling exploration audit (frozen base, NTP-only)",
        "",
        "No training. No reward changes. Held-out test194 not used.",
        "",
        f"Examples: {meta['n_examples']} from internal-validation80 | "
        f"Rollouts/example: {meta['rollouts_per_example']}",
        "",
        f"## Verdict: {verdict['code']}) {verdict['text']}",
        "",
        f"Recommended condition: `{verdict['recommended_condition']}`",
        "",
        "## Ranking table",
        "",
        "| temperature | top_p | mean IoU | P(IoU>0.5) | G8 mixed | valid | near-full | pairwise bbox IoU |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        lines.append(
            "| {temperature:g} | {top_p:g} | {mean_iou:.4f} | {P_iou_gt_0_5:.4f} | "
            "{G8_mixed_rate:.4f} | {valid_box_rate:.4f} | {near_full_box_rate:.4f} | "
            "{mean_pairwise_bbox_iou:.4f} |".format(**row)
        )
    lines += ["", "## Simulated G8 detail", "",
              "| condition | P(all0) | P(mixed) | P(all1) | E[#pos] | P(≥1 pos) |",
              "|---|---:|---:|---:|---:|---:|"]
    for tag, agg in aggregates.items():
        g8 = agg["simulated_groups"]["G8"]
        lines.append(
            f"| {tag} | {g8['P_all_spatial_0']:.4f} | {g8['P_mixed_spatial']:.4f} | "
            f"{g8['P_all_spatial_1']:.4f} | {g8['expected_positives_per_group']:.4f} | "
            f"{g8['P_group_has_ge1_iou_gt_0_5']:.4f} |"
        )
    (output / "SAMPLING_EXPLORATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
