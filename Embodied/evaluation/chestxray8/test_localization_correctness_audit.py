"""CPU-only localization-coordinate and reward-path oracle tests.

These tests do not load a model, run inference, or evaluate a held-out sample.
They verify the geometry contract used by saved ChestX-ray8 manifests and GRPO.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

CHEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CHEST_DIR))
sys.path.insert(0, str(REPO_ROOT))

import eval_final_two_gpu_hybrid_grpo as evaluator  # noqa: E402
import eval_locateanything_bbox as bbox_eval  # noqa: E402
import two_gpu_g4_grpo_multistep_smoke as trainer  # noqa: E402
from rl import rewards  # noqa: E402
from sft_common import xyxy_pixels_to_norm1000  # noqa: E402


TRAIN_MANIFEST = CHEST_DIR / "splits/train80_pairs_seed42.jsonl"


class _ZeroSemantic:
    def score(self, image_path, box_norm_1000, query):
        return 0.0


def _native_completion(disease, box):
    coords = "".join("<{}>".format(int(value)) for value in box)
    return "<ref>{}</ref><box>{}</box>".format(disease, coords)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_exact_transformed_gt_is_iou_one_and_spatial_reward_one():
    pair = _read_jsonl(TRAIN_MANIFEST)[0]
    gt = pair["gt_boxes_norm_1000"][0]
    spatial, iou = rewards.spatial_reward_from_box(
        gt, pair["gt_boxes_norm_1000"], threshold=0.5
    )
    assert iou == 1.0
    assert spatial == 1.0

    pipeline = rewards.ProductionRewardPipeline(
        _ZeroSemantic(), parser_name=rewards.PARSER_NATIVE, iou_threshold=0.5
    )
    trace = SimpleNamespace(
        decoded_text=_native_completion(pair["disease"], gt),
        committed_final_box_norm_1000=tuple(gt),
        has_unambiguous_committed_box=True,
        reward_branch="pbd",
    )
    result = pipeline.score_from_trace(trace, pair)
    assert result.format_reward == 1.0
    assert result.final_iou == 1.0
    assert result.spatial_reward == 1.0


def test_asymmetric_rectangular_image_proves_xy_and_width_height_order():
    width, height = 1600, 900
    pixel_box = [160.0, 180.0, 1200.0, 720.0]
    normalized = xyxy_pixels_to_norm1000([pixel_box], width, height)[0]
    assert normalized == [100, 200, 750, 800]
    restored = bbox_eval.convert_boxes_to_pixels([normalized], width, height)[0]
    assert all(abs(a - b) <= 1e-12 for a, b in zip(restored, pixel_box))
    assert math.isclose(bbox_eval.box_iou(pixel_box, restored), 1.0)


def test_pixel_normalized_pixel_roundtrip_has_only_expected_rounding():
    cases = [
        (1600, 900, [17.3, 41.7, 1388.2, 811.4]),
        (901, 1603, [0.2, 13.9, 900.1, 1500.7]),
        (1024, 1024, [339.166137566138, 119.195767195767, 511.458201058202, 470.281481481481]),
    ]
    for width, height, pixel_box in cases:
        normalized = xyxy_pixels_to_norm1000([pixel_box], width, height)[0]
        restored = bbox_eval.convert_boxes_to_pixels([normalized], width, height)[0]
        assert abs(restored[0] - pixel_box[0]) <= width / 2000.0 + 1e-9
        assert abs(restored[2] - pixel_box[2]) <= width / 2000.0 + 1e-9
        assert abs(restored[1] - pixel_box[1]) <= height / 2000.0 + 1e-9
        assert abs(restored[3] - pixel_box[3]) <= height / 2000.0 + 1e-9


def test_1000_is_full_boundary_and_999_is_one_normalized_unit_inside():
    width, height = 1600, 900
    full = bbox_eval.convert_boxes_to_pixels([[0, 0, 1000, 1000]], width, height)[0]
    almost = bbox_eval.convert_boxes_to_pixels([[0, 0, 999, 999]], width, height)[0]
    assert full == [0.0, 0.0, 1600.0, 900.0]
    assert all(abs(a - b) <= 1e-9 for a, b in zip(almost, [0.0, 0.0, 1598.4, 899.1]))
    assert math.isclose(
        bbox_eval.box_iou([0, 0, 999, 999], [0, 0, 1000, 1000]),
        0.998001,
    )
    assert rewards.is_valid_geometry((0, 0, 1000, 1000))
    assert rewards.is_valid_geometry((0, 0, 999, 999))


def test_saved_train_manifest_dimensions_and_roundtrip_are_consistent():
    # All train records are metadata-checked; no model inference occurs.
    for pair in _read_jsonl(TRAIN_MANIFEST):
        with Image.open(pair["image_path"]) as image:
            width, height = image.size
        assert width == pair["image_width"]
        assert height == pair["image_height"]
        expected = xyxy_pixels_to_norm1000(
            pair["gt_boxes_xyxy_px"], width, height
        )
        assert expected == pair["gt_boxes_norm_1000"]
        restored = bbox_eval.convert_boxes_to_pixels(expected, width, height)
        for original, roundtrip in zip(pair["gt_boxes_xyxy_px"], restored):
            assert abs(roundtrip[0] - original[0]) <= width / 2000.0 + 1e-9
            assert abs(roundtrip[2] - original[2]) <= width / 2000.0 + 1e-9
            assert abs(roundtrip[1] - original[1]) <= height / 2000.0 + 1e-9
            assert abs(roundtrip[3] - original[3]) <= height / 2000.0 + 1e-9


def test_train_and_evaluator_share_exact_reward_and_iou_implementations():
    assert rewards.box_iou is bbox_eval.box_iou
    assert trainer.build_reward_pipeline_from_config is rewards.build_reward_pipeline_from_config
    assert evaluator.build_reward_pipeline_from_config is rewards.build_reward_pipeline_from_config
    assert trainer.MedCLIPSemanticScorer is rewards.MedCLIPSemanticScorer
    assert evaluator.MedCLIPSemanticScorer is rewards.MedCLIPSemanticScorer


def test_iou_is_invariant_to_independent_axis_scaling():
    a = [100, 200, 750, 800]
    b = [250, 100, 900, 700]
    normalized_iou = bbox_eval.box_iou(a, b)
    width, height = 1600, 900
    pixel_a = bbox_eval.convert_boxes_to_pixels([a], width, height)[0]
    pixel_b = bbox_eval.convert_boxes_to_pixels([b], width, height)[0]
    assert math.isclose(bbox_eval.box_iou(pixel_a, pixel_b), normalized_iou)
    assert math.isfinite(normalized_iou)


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("{} localization oracle tests passed".format(len(tests)))
