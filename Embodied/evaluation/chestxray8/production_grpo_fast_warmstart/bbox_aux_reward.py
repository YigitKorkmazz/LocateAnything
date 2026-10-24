"""`bbox_aux` spatial reward modes: raw IoU + MedReasoner alignment (+ scale).

    bbox_aux:        total = format + semantic + R_iou + R_align + R_scale
    bbox_aux_align:  total = format + semantic + R_iou + R_align

`bbox_aux_align` drops R_scale from the reward (it kept boxes tight but hurt
mean IoU / R@0.1 on ambiguous cases), leaving the corner-distance term as the
only auxiliary bbox term. R_scale/scale_d are still computed and logged for
that mode as diagnostics only.

Format and semantic are exactly what the unmodified
`rl.rewards.ProductionRewardPipeline` already computes; only the spatial part
is replaced. No smoothing, penalization, normalization or averaging of the
three bbox terms; every coefficient is 1 (the pipeline's format/spatial/
semantic weights are all 1.0 in `production_grpo_fast.rewards_adapter`).

- R_iou   = raw IoU (== `component.final_iou`, identical to the "continuous"
            mode's spatial reward).
- R_align = max(0, 1 - align_d), MedReasoner Eq. 3: align_d = mean over the
            4 box corners (TL, TR, BL, BR) of the L1 corner distance
            |dx|+|dy| between prediction and GT, divided by the GT diagonal.
- R_scale = max(0, 1 - scale_d / 2), MedReasoner Eq. 4: scale_d =
            sqrt(log(A_p/A_g)^2 + log(R_p/R_g)^2), A = w*h, R = w/h.

All coordinates are the 0..1000 normalized space of `gt_boxes_norm_1000` /
`final_box_norm_1000`.

No valid predicted box (None or invalid geometry) -> R_iou = R_align =
R_scale = 0. Multiple GT boxes -> the GT with max IoU is used for align and
scale; if every IoU is 0, the GT with the smallest align_d. Degenerate GT
boxes (zero width/height) are never selected; if no usable GT exists,
R_align = R_scale = 0 and the rollout is flagged `degenerate_gt=True` (also
counted in `DEGENERATE_GT_COUNT`).

This module never edits `production_grpo_fast` or `rl` files. It is wired in
at runtime by `install_bbox_aux_scoring()`, which rebinds the `score_group`
name inside the `production_grpo_fast.grpo_step` module namespace to
`score_group_with_bbox_aux`; that wrapper delegates every non-`bbox_aux`
mode to the original `rewards_adapter.score_group` unchanged.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rl.rewards import ProductionRewardPipeline, RewardComponents, is_valid_geometry
from eval_locateanything_bbox import box_iou

from production_grpo_fast import rewards_adapter as _rewards_adapter

SPATIAL_REWARD_MODE_BBOX_AUX = "bbox_aux"
SPATIAL_REWARD_MODE_BBOX_AUX_ALIGN = "bbox_aux_align"

# Which bbox terms each mode sums into spatial_reward (all coefficients 1).
BBOX_AUX_MODE_TERMS = {
    SPATIAL_REWARD_MODE_BBOX_AUX: ("r_iou", "r_align", "r_scale"),
    SPATIAL_REWARD_MODE_BBOX_AUX_ALIGN: ("r_iou", "r_align"),
}
BBOX_AUX_MODES = tuple(BBOX_AUX_MODE_TERMS)

# Captured once at import, before any patching, so delegation always reaches
# the real implementation even after `install_bbox_aux_scoring()`.
_ORIGINAL_SCORE_GROUP = _rewards_adapter.score_group

DEGENERATE_GT_COUNT = 0


@dataclass(frozen=True)
class BBoxAuxRewardComponents(RewardComponents):
    r_iou: float = 0.0
    r_align: float = 0.0
    r_scale: float = 0.0
    align_d: Optional[float] = None
    scale_d: Optional[float] = None
    pred_area: Optional[float] = None
    gt_area: Optional[float] = None
    matched_gt_index: Optional[int] = None
    degenerate_gt: bool = False


def _wh(box: Sequence[float]) -> Tuple[float, float]:
    return float(box[2]) - float(box[0]), float(box[3]) - float(box[1])


def box_area(box: Sequence[float]) -> float:
    w, h = _wh(box)
    return w * h


def alignment_distance(pred: Sequence[float], gt: Sequence[float]) -> float:
    """MedReasoner Eq. 3: mean corner L1 distance / GT diagonal."""
    px1, py1, px2, py2 = (float(v) for v in pred)
    gx1, gy1, gx2, gy2 = (float(v) for v in gt)
    gw, gh = gx2 - gx1, gy2 - gy1
    diag = math.hypot(gw, gh)
    if not (gw > 0 and gh > 0 and diag > 0):
        raise ValueError(f"degenerate GT box {gt!r}")
    corners_pred = ((px1, py1), (px2, py1), (px1, py2), (px2, py2))
    corners_gt = ((gx1, gy1), (gx2, gy1), (gx1, gy2), (gx2, gy2))
    total = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(corners_pred, corners_gt))
    return (total / 4.0) / diag


def scale_distance(pred: Sequence[float], gt: Sequence[float]) -> float:
    """MedReasoner Eq. 4: sqrt(log(A_p/A_g)^2 + log(R_p/R_g)^2)."""
    pw, ph = _wh(pred)
    gw, gh = _wh(gt)
    if not (gw > 0 and gh > 0):
        raise ValueError(f"degenerate GT box {gt!r}")
    if not (pw > 0 and ph > 0):
        raise ValueError(f"invalid predicted box {pred!r}")
    log_area = math.log((pw * ph) / (gw * gh))
    log_aspect = math.log((pw / ph) / (gw / gh))
    return math.sqrt(log_area * log_area + log_aspect * log_aspect)


def _is_usable_gt(gt: Sequence[float]) -> bool:
    w, h = _wh(gt)
    return w > 0 and h > 0 and math.isfinite(w) and math.isfinite(h)


def bbox_aux_terms(
    pred_box: Optional[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Compute r_iou/r_align/r_scale (+ diagnostics) for one predicted box."""
    usable = [(i, gt) for i, gt in enumerate(gt_boxes) if _is_usable_gt(gt)]
    out: Dict[str, Any] = {
        "r_iou": 0.0,
        "r_align": 0.0,
        "r_scale": 0.0,
        "align_d": None,
        "scale_d": None,
        "pred_area": None,
        "gt_area": float(box_area(usable[0][1])) if len(usable) == 1 else None,
        "matched_gt_index": None,
        "degenerate_gt": len(usable) != len(gt_boxes) or not usable,
    }
    if pred_box is None or not is_valid_geometry(pred_box):
        return out
    out["pred_area"] = float(box_area(pred_box))
    if not usable:
        return out

    ious = [(float(box_iou(pred_box, gt)), i, gt) for i, gt in usable]
    best_iou = max(iou for iou, _, _ in ious)
    if best_iou > 0.0:
        _, idx, gt = next(t for t in ious if t[0] == best_iou)
        align_d = alignment_distance(pred_box, gt)
    else:
        align_d, idx, gt = min(((alignment_distance(pred_box, g), i, g) for _, i, g in ious), key=lambda t: t[0])
    scale_d = scale_distance(pred_box, gt)

    out.update(
        r_iou=best_iou,
        r_align=max(0.0, 1.0 - align_d),
        r_scale=max(0.0, 1.0 - scale_d / 2.0),
        align_d=float(align_d),
        scale_d=float(scale_d),
        gt_area=float(box_area(gt)),
        matched_gt_index=int(idx),
    )
    return out


def as_bbox_aux(
    pipeline: ProductionRewardPipeline,
    component: RewardComponents,
    gt_boxes: Sequence[Sequence[float]],
    *,
    mode: str = SPATIAL_REWARD_MODE_BBOX_AUX,
) -> BBoxAuxRewardComponents:
    global DEGENERATE_GT_COUNT
    reward_terms = BBOX_AUX_MODE_TERMS[mode]
    pred = component.final_box_norm_1000
    terms = bbox_aux_terms(pred, gt_boxes)
    if terms["degenerate_gt"]:
        DEGENERATE_GT_COUNT += 1
    # R_iou is the pipeline's own raw IoU (identical to "continuous" mode).
    # For a valid box this equals our best_iou; for an invalid one both are 0.
    r_iou = float(component.final_iou) if terms["pred_area"] is not None else 0.0
    terms = dict(terms, r_iou=r_iou)
    spatial = sum(terms[name] for name in reward_terms)
    total = (
        pipeline.format_weight * component.format_reward
        + pipeline.spatial_weight * spatial
        + pipeline.semantic_weight * component.semantic_reward
    )
    base = {f.name: getattr(component, f.name) for f in dataclasses.fields(RewardComponents)}
    base.update(spatial_reward=float(spatial), total_reward=float(total))
    result = BBoxAuxRewardComponents(**base, **terms)
    for name in ("total_reward", "spatial_reward", "r_iou", "r_align", "r_scale"):
        if not math.isfinite(getattr(result, name)):
            raise RuntimeError(f"non-finite bbox_aux {name}: {result!r}")
    return result


def score_group_with_bbox_aux(
    pipeline: ProductionRewardPipeline,
    pair: Dict[str, Any],
    traces: Sequence[Any],
    *,
    spatial_reward_mode: str = _rewards_adapter.SPATIAL_REWARD_MODE_BINARY,
) -> Tuple[List[float], List[RewardComponents]]:
    """Drop-in for `rewards_adapter.score_group` adding the `bbox_aux*` modes."""
    if spatial_reward_mode not in BBOX_AUX_MODES:
        return _ORIGINAL_SCORE_GROUP(pipeline, pair, traces, spatial_reward_mode=spatial_reward_mode)
    gt_boxes = pair["gt_boxes_norm_1000"]
    components = [
        as_bbox_aux(pipeline, pipeline.score_from_trace(trace, pair), gt_boxes, mode=spatial_reward_mode)
        for trace in traces
    ]
    totals = [float(c.total_reward) for c in components]
    return totals, components


def install_bbox_aux_scoring() -> None:
    """Rebind `production_grpo_fast.grpo_step.score_group` to the wrapper.

    `execute_prompt_group` resolves `score_group` as a module global at call
    time, so this takes effect for every subsequent call in this process.
    Must be called inside each spawned worker (spawn re-imports modules).
    """
    from production_grpo_fast import grpo_step

    grpo_step.score_group = score_group_with_bbox_aux


def window_bbox_metrics(group_reports: Sequence[Dict[str, Any]], *, bbox_aux: bool) -> Dict[str, Any]:
    """Per-window logging fields built from `run_accumulation_window` reports."""
    rollout_iou = [[float(v) for v in g["ious"]] for g in group_reports]
    flat_iou = [v for g in rollout_iou for v in g]
    n = len(flat_iou)
    record: Dict[str, Any] = {
        "rollout_iou": rollout_iou,
        "mean_iou": sum(flat_iou) / n if n else None,
        "frac_rollouts_iou_zero": sum(1 for v in flat_iou if v == 0.0) / n if n else None,
    }
    keys = ("r_align", "r_scale", "align_d", "scale_d", "pred_area")
    if not bbox_aux:
        record.update({f"rollout_{k}": None for k in keys})
        record.update(mean_r_align=None, mean_r_scale=None, gt_area=None)
        return record
    comps = [g["reward_components"] for g in group_reports]
    for k in keys:
        record[f"rollout_{k}"] = [[c.get(k) for c in g] for g in comps]
    flat_align = [c["r_align"] for g in comps for c in g]
    flat_scale = [c["r_scale"] for g in comps for c in g]
    record["mean_r_align"] = sum(flat_align) / len(flat_align) if flat_align else None
    record["mean_r_scale"] = sum(flat_scale) / len(flat_scale) if flat_scale else None
    # One value per group: the (single) GT box area; train80 has exactly one
    # GT per pair. For a hypothetical multi-GT pair this is the area of the GT
    # matched by the first rollout that had a valid box (None if none did).
    record["gt_area"] = [
        next((c["gt_area"] for c in g if c.get("gt_area") is not None), None) for g in comps
    ]
    record["degenerate_gt_count"] = DEGENERATE_GT_COUNT
    return record
