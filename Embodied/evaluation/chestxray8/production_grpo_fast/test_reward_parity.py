#!/usr/bin/env python3
"""No-GPU correctness gate for the reward wiring (format/spatial/semantic).

`rewards_adapter.build_reward_pipeline` is a thin, fixed-weight wrapper
around the existing, unmodified `rl.rewards.ProductionRewardPipeline`. This
test does not re-derive reward math (that math is upstream and out of
scope to re-verify here); it checks that this backend wires it up exactly
to spec: parser=native_locateanything, weights 1/1/1, strict IoU>0.5, and
that rewards are computed from `RolloutTrace.committed_final_box_norm_1000`
(never mined from raw text), using a fake semantic scorer so no MedCLIP /
GPU is required.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast.rewards_adapter import build_reward_pipeline, score_group  # noqa: E402
from rl.rewards import PARSER_NATIVE  # noqa: E402


class FixedSemanticScorer:
    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def score(self, image_path, box_norm_1000, query) -> float:
        self.calls += 1
        return self.value


@dataclass
class FakeTrace:
    decoded_text: str
    committed_final_box_norm_1000: Optional[Sequence[int]]
    has_unambiguous_committed_box: bool
    reward_branch: str


PAIR = {
    "image_path": "/dev/null",
    "user_query": "Locate the Cardiomegaly in this chest X-ray",
    "gt_boxes_norm_1000": [[100, 100, 300, 300]],
}


def test_pipeline_uses_native_parser_and_unit_weights() -> None:
    pipeline = build_reward_pipeline(FixedSemanticScorer(0.5))
    assert pipeline.parser_name == PARSER_NATIVE
    assert pipeline.format_weight == 1.0
    assert pipeline.spatial_weight == 1.0
    assert pipeline.semantic_weight == 1.0
    assert pipeline.iou_threshold == 0.5


def test_format_reward_requires_exactly_one_committed_box() -> None:
    scorer = FixedSemanticScorer(0.0)
    pipeline = build_reward_pipeline(scorer)
    no_box = FakeTrace("<ref>Cardiomegaly</ref>", None, False, "none")
    totals, components = score_group(pipeline, PAIR, [no_box])
    assert components[0].format_reward == 0.0
    assert totals[0] == 0.0
    assert scorer.calls == 0, "semantic scorer must not run for an unparseable trace"


def test_spatial_reward_strict_iou_greater_than_half() -> None:
    scorer = FixedSemanticScorer(0.2)
    pipeline = build_reward_pipeline(scorer)
    # GT is [100,100,300,300] (200x200=40000). A box that reproduces exactly
    # IoU==0.5 must score spatial=0 (strict '>' not '>='); a slightly larger
    # overlap must score spatial=1.
    exact_half = FakeTrace("<ref>x</ref><box><100><100><300><500></box>", (100, 100, 300, 500), True, "ntp_only")
    # intersection 200x200=40000, union = 40000(gt)+80000(pred)-40000=80000 -> IoU=0.5 exactly
    totals, components = score_group(pipeline, PAIR, [exact_half])
    assert components[0].final_iou == 0.5
    assert components[0].spatial_reward == 0.0, "IoU==0.5 must not satisfy strict '>' 0.5"

    clearly_over = FakeTrace("<ref>x</ref><box><100><100><300><300></box>", (100, 100, 300, 300), True, "ntp_only")
    totals, components = score_group(pipeline, PAIR, [clearly_over])
    assert components[0].final_iou == 1.0
    assert components[0].spatial_reward == 1.0


def test_semantic_reward_is_raw_scorer_cosine_and_zero_on_invalid_box() -> None:
    scorer = FixedSemanticScorer(0.37)
    pipeline = build_reward_pipeline(scorer)
    valid = FakeTrace("<ref>x</ref><box><100><100><300><300></box>", (100, 100, 300, 300), True, "ntp_only")
    totals, components = score_group(pipeline, PAIR, [valid])
    assert components[0].semantic_reward == 0.37
    assert scorer.calls == 1

    invalid_geometry = FakeTrace("<ref>x</ref><box><300><300><100><100></box>", (300, 300, 100, 100), True, "ntp_only")
    totals, components = score_group(pipeline, PAIR, [invalid_geometry])
    assert components[0].semantic_reward == 0.0, "invalid geometry must fall back to 0.0, not call the scorer"


def test_total_is_unweighted_sum() -> None:
    scorer = FixedSemanticScorer(0.4)
    pipeline = build_reward_pipeline(scorer)
    trace = FakeTrace("<ref>x</ref><box><100><100><300><300></box>", (100, 100, 300, 300), True, "ntp_only")
    totals, components = score_group(pipeline, PAIR, [trace])
    c = components[0]
    assert totals[0] == c.format_reward + c.spatial_reward + c.semantic_reward
    assert totals[0] == 1.0 + 1.0 + 0.4


def main() -> None:
    tests = [obj for name, obj in globals().items() if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} REWARD WIRING TESTS PASSED")


if __name__ == "__main__":
    main()
