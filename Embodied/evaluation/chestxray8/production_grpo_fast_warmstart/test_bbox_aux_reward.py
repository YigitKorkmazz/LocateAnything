#!/usr/bin/env python3
"""CPU-only tests for `bbox_aux_reward` (plain asserts; pytest-compatible).

Run:  PYTHONPATH=$(pwd) python evaluation/chestxray8/production_grpo_fast_warmstart/test_bbox_aux_reward.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

CHEST_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(CHEST_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.rewards import PARSER_NATIVE, ProductionRewardPipeline  # noqa: E402

from production_grpo_fast import grpo_step  # noqa: E402
from production_grpo_fast import rewards_adapter  # noqa: E402
from production_grpo_fast_warmstart import bbox_aux_reward as bar  # noqa: E402

GT = (300, 200, 500, 600)  # w=200, h=400


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def scaled_about_center(box, area_factor):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = math.sqrt(area_factor)
    w, h = (box[2] - box[0]) * s, (box[3] - box[1]) * s
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class FakeSemantic:
    """Deterministic stand-in for MedCLIP: a function of the box only."""

    def score(self, image_path, box, query):
        return 0.1 + (sum(box) % 97) / 1000.0


def make_pipeline():
    # Same construction as rewards_adapter.build_reward_pipeline, fake scorer.
    return ProductionRewardPipeline(
        FakeSemantic(),
        format_weight=1.0,
        spatial_weight=1.0,
        semantic_weight=1.0,
        iou_threshold=0.5,
        parser_name=PARSER_NATIVE,
        invalid_box_semantic_fallback=0.0,
    )


def trace(box, *, text=None, unambiguous=True):
    if text is None:
        text = "" if box is None else "<box><{}><{}><{}><{}></box>".format(*box)
    return SimpleNamespace(
        decoded_text=text,
        committed_final_box_norm_1000=box,
        has_unambiguous_committed_box=unambiguous,
        reward_branch="test",
    )


def random_box(rng, allow_invalid=False):
    x1, x2 = sorted(rng.randint(0, 1000) for _ in range(2))
    y1, y2 = sorted(rng.randint(0, 1000) for _ in range(2))
    if not allow_invalid:
        x2 = max(x2, x1 + 1) if x1 < 1000 else x2
        y2 = max(y2, y1 + 1) if y1 < 1000 else y2
        if x1 == 1000:
            x1 = 999
        if y1 == 1000:
            y1 = 999
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------- terms ----


def test_identical_box():
    t = bar.bbox_aux_terms(GT, [GT])
    assert close(t["r_iou"], 1.0) and close(t["r_align"], 1.0) and close(t["r_scale"], 1.0), t
    assert close(t["align_d"], 0.0) and close(t["scale_d"], 0.0)
    assert close(t["pred_area"], 80000.0) and close(t["gt_area"], 80000.0)


def test_alignment_formula_by_hand():
    pred = (310, 220, 520, 590)  # dx1=10, dy1=20, dx2=20, dy2=10
    # corners: TL 10+20, TR 20+20, BL 10+10, BR 20+10 -> mean 30
    expected = 30.0 / math.hypot(200, 400)
    assert close(bar.alignment_distance(pred, GT), expected)
    t = bar.bbox_aux_terms(pred, [GT])
    assert close(t["r_align"], 1.0 - expected)


def test_disjoint_near_vs_far():
    near = (510, 200, 710, 600)  # just right of GT, same size
    far = (800, 0, 1000, 400)
    tn, tf = bar.bbox_aux_terms(near, [GT]), bar.bbox_aux_terms(far, [GT])
    assert tn["r_iou"] == 0.0 and tf["r_iou"] == 0.0
    assert tn["r_align"] > tf["r_align"], (tn, tf)
    assert tn["r_align"] > 0.0


def test_scale_area_multiples():
    for factor, expected in ((2, 1 - math.log(2) / 2), (4, 1 - math.log(4) / 2), (8, 0.0)):
        t = bar.bbox_aux_terms(scaled_about_center(GT, factor), [GT])
        assert close(t["r_scale"], expected, 1e-9), (factor, t["r_scale"], expected)
    assert close(1 - math.log(2) / 2, 0.6534264097, 1e-9)
    assert close(1 - math.log(4) / 2, 0.3068528194, 1e-9)


def test_aspect_mismatch_lowers_scale():
    # same area (80000) and center, aspect 1 instead of 0.5
    side = math.sqrt(80000)
    cx, cy = 400, 400
    square = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    t = bar.bbox_aux_terms(square, [GT])
    assert close(t["scale_d"], math.log(2), 1e-9)
    assert t["r_scale"] < 1.0
    assert close(t["r_scale"], 1 - math.log(2) / 2, 1e-9)


def test_invalid_or_none_pred():
    for pred in (None, (500, 500, 500, 600), (500, 500, 400, 600), (10, 600, 20, 600)):
        t = bar.bbox_aux_terms(pred, [GT])
        assert t["r_iou"] == 0.0 and t["r_align"] == 0.0 and t["r_scale"] == 0.0, (pred, t)
        assert t["align_d"] is None and t["scale_d"] is None and t["pred_area"] is None


def test_multi_gt_selection():
    gt_a = (0, 0, 100, 100)
    gt_b = (500, 500, 700, 700)
    pred = (480, 480, 690, 690)  # overlaps b only
    t = bar.bbox_aux_terms(pred, [gt_a, gt_b])
    assert t["matched_gt_index"] == 1
    assert close(t["align_d"], bar.alignment_distance(pred, gt_b))
    # Max-IoU wins even when another GT has a smaller align_d.
    big = (100, 100, 900, 900)
    small_inside = (450, 450, 550, 550)
    pred2 = (440, 440, 560, 560)
    t2 = bar.bbox_aux_terms(pred2, [small_inside, big])
    ious = [bar.box_iou(pred2, small_inside), bar.box_iou(pred2, big)]
    assert t2["matched_gt_index"] == int(ious[1] > ious[0])
    # Max-IoU GT has a *larger* align_d than the other one here:
    pred3 = (100, 100, 200, 200)
    gt_hi_iou = (195, 195, 205, 205)  # tiny overlap -> IoU > 0, but align_d ~7
    gt_lo_d = (210, 100, 1000, 1000)  # disjoint -> IoU 0, align_d ~0.71
    assert bar.box_iou(pred3, gt_hi_iou) > 0 and bar.box_iou(pred3, gt_lo_d) == 0
    assert bar.alignment_distance(pred3, gt_lo_d) < bar.alignment_distance(pred3, gt_hi_iou)
    t3 = bar.bbox_aux_terms(pred3, [gt_lo_d, gt_hi_iou])
    assert t3["matched_gt_index"] == 1 and close(t3["r_iou"], bar.box_iou(pred3, gt_hi_iou))
    assert t3["r_align"] == 0.0
    # All IoU == 0 -> smallest align_d.
    gt_near = (300, 300, 400, 400)
    gt_far = (900, 900, 1000, 1000)
    pred4 = (410, 300, 510, 400)
    t4 = bar.bbox_aux_terms(pred4, [gt_far, gt_near])
    assert t4["r_iou"] == 0.0 and t4["matched_gt_index"] == 1
    assert close(t4["scale_d"], bar.scale_distance(pred4, gt_near))


def test_degenerate_gt_no_nan():
    t = bar.bbox_aux_terms((10, 10, 50, 50), [(100, 100, 100, 200)])
    assert t["degenerate_gt"] and t["r_align"] == 0.0 and t["r_scale"] == 0.0
    # mixed: degenerate one is skipped, usable one is matched
    t2 = bar.bbox_aux_terms((10, 10, 50, 50), [(100, 100, 100, 200), (10, 10, 50, 50)])
    assert t2["matched_gt_index"] == 1 and close(t2["r_align"], 1.0)


def test_fuzz_finite_and_bounded():
    rng = random.Random(0)
    for _ in range(5000):
        gts = [random_box(rng) for _ in range(rng.choice((1, 1, 1, 2, 3)))]
        pred = random_box(rng, allow_invalid=rng.random() < 0.1)
        t = bar.bbox_aux_terms(pred, gts)
        for k in ("r_iou", "r_align", "r_scale"):
            assert math.isfinite(t[k]) and 0.0 <= t[k] <= 1.0, (k, t, pred, gts)
        for k in ("align_d", "scale_d", "pred_area", "gt_area"):
            if t[k] is not None:
                assert math.isfinite(t[k]) and t[k] >= 0.0, (k, t)
        json.dumps(t, allow_nan=False)


# ------------------------------------------------------- score_group ----


def test_bbox_aux_totals_match_formula():
    pipe = make_pipeline()
    rng = random.Random(1)
    pair = {"gt_boxes_norm_1000": [list(GT)], "image_path": "x", "user_query": "q"}
    boxes = [GT, scaled_about_center(GT, 2), None, (900, 900, 950, 990), (500, 500, 500, 600)]
    boxes = [tuple(int(round(v)) for v in b) if b is not None else None for b in boxes]
    boxes += [random_box(rng) for _ in range(200)]
    traces = [trace(b) for b in boxes]
    traces.append(trace((300, 200, 500, 600), text="garbage no box"))  # box committed, format invalid
    traces.append(trace(None, unambiguous=False))
    totals, comps = bar.score_group_with_bbox_aux(pipe, pair, traces, spatial_reward_mode="bbox_aux")
    base = [pipe.score_from_trace(t, pair) for t in traces]
    for total, c, b in zip(totals, comps, base):
        assert isinstance(c, bar.BBoxAuxRewardComponents)
        assert c.format_reward == b.format_reward and c.semantic_reward == b.semantic_reward
        assert c.final_iou == b.final_iou and c.r_iou == b.final_iou
        assert close(c.spatial_reward, c.r_iou + c.r_align + c.r_scale)
        assert close(total, c.format_reward + c.semantic_reward + c.r_iou + c.r_align + c.r_scale)
        assert close(c.total_reward, total)
        if c.final_box_norm_1000 is None:
            assert c.r_iou == c.r_align == c.r_scale == 0.0
        d = c.to_dict()
        for k in ("r_iou", "r_align", "r_scale", "align_d", "scale_d", "pred_area", "gt_area", "total_reward"):
            assert k in d
        json.dumps(d, allow_nan=False)
    # format-invalid but box committed: format 0, bbox terms still computed (like continuous mode)
    assert comps[-2].format_reward == 0.0 and close(comps[-2].r_align, 1.0)
    assert comps[0].format_reward == 1.0 and close(totals[0], 1.0 + comps[0].semantic_reward + 3.0)


def test_bbox_aux_align_drops_scale_only():
    pipe = make_pipeline()
    rng = random.Random(3)
    pair = {"gt_boxes_norm_1000": [list(GT)], "image_path": "x", "user_query": "q"}
    boxes = [GT, None, (310, 220, 520, 590)] + [random_box(rng, allow_invalid=rng.random() < 0.1) for _ in range(200)]
    traces = [trace(b) for b in boxes]
    full_totals, full = bar.score_group_with_bbox_aux(pipe, pair, traces, spatial_reward_mode="bbox_aux")
    totals, comps = bar.score_group_with_bbox_aux(pipe, pair, traces, spatial_reward_mode="bbox_aux_align")
    for total, c, f, ft in zip(totals, comps, full, full_totals):
        assert close(c.spatial_reward, c.r_iou + c.r_align)
        assert close(total, c.format_reward + c.semantic_reward + c.r_iou + c.r_align)
        assert close(total, ft - f.r_scale)
        # diagnostics are identical to the full mode, only the reward sum differs
        for k in ("r_iou", "r_align", "r_scale", "align_d", "scale_d", "pred_area", "gt_area"):
            assert getattr(c, k) == getattr(f, k), k
        json.dumps(c.to_dict(), allow_nan=False)
    assert close(totals[0], 1.0 + comps[0].semantic_reward + 2.0)
    # an enlarged box is no longer penalized by R_scale
    big = tuple(int(round(v)) for v in scaled_about_center(GT, 4))
    (t_full,), _ = bar.score_group_with_bbox_aux(pipe, pair, [trace(big)], spatial_reward_mode="bbox_aux")
    (t_align,), (c_align,) = bar.score_group_with_bbox_aux(pipe, pair, [trace(big)], spatial_reward_mode="bbox_aux_align")
    assert c_align.r_scale > 0 and close(t_full - t_align, c_align.r_scale)


def test_binary_continuous_unchanged():
    pipe = make_pipeline()
    rng = random.Random(2)
    pair = {"gt_boxes_norm_1000": [list(GT)], "image_path": "x", "user_query": "q"}
    traces = [trace(random_box(rng, allow_invalid=rng.random() < 0.1)) for _ in range(300)]
    traces += [trace(None, unambiguous=False), trace(GT)]
    original = rewards_adapter.score_group
    try:
        for mode in ("binary", "continuous"):
            ref_totals, ref_comps = original(pipe, pair, traces, spatial_reward_mode=mode)
            bar.install_bbox_aux_scoring()
            assert grpo_step.score_group is bar.score_group_with_bbox_aux
            new_totals, new_comps = grpo_step.score_group(pipe, pair, traces, spatial_reward_mode=mode)
            assert new_totals == ref_totals, mode
            assert [c.to_dict() for c in new_comps] == [c.to_dict() for c in ref_comps], mode
            assert all(type(c) is type(r) for c, r in zip(new_comps, ref_comps))
    finally:
        grpo_step.score_group = original
    # unknown modes are still rejected by the original implementation
    try:
        bar.score_group_with_bbox_aux(pipe, pair, traces[:1], spatial_reward_mode="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_window_metrics():
    pipe = make_pipeline()
    pair = {"gt_boxes_norm_1000": [list(GT)], "image_path": "x", "user_query": "q"}
    reports = []
    for boxes in ([GT, None, (900, 900, 950, 990), (310, 220, 520, 590)], [GT, GT, GT, GT]):
        totals, comps = bar.score_group_with_bbox_aux(pipe, pair, [trace(b) for b in boxes], spatial_reward_mode="bbox_aux")
        reports.append({"rewards": totals, "reward_components": [c.to_dict() for c in comps], "ious": [c.final_iou for c in comps]})
    rec = bar.window_bbox_metrics(reports, bbox_aux=True)
    json.dumps(rec, allow_nan=False)
    assert len(rec["rollout_iou"]) == 2 and len(rec["rollout_r_align"][0]) == 4
    assert rec["gt_area"] == [80000.0, 80000.0]
    assert close(rec["frac_rollouts_iou_zero"], 2 / 8)
    assert rec["rollout_align_d"][0][1] is None and rec["rollout_pred_area"][0][1] is None
    assert close(rec["mean_r_align"], sum(v for g in rec["rollout_r_align"] for v in g) / 8)
    rec2 = bar.window_bbox_metrics(reports, bbox_aux=False)
    assert rec2["mean_r_align"] is None and rec2["rollout_iou"] == rec["rollout_iou"]
    json.dumps(rec2, allow_nan=False)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
