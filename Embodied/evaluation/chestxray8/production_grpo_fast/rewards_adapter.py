"""Reward pipeline construction: format(1) + spatial(1) + semantic(1).

Thin wrapper around the existing, unmodified `rl.rewards.ProductionRewardPipeline`
and `rl.medclip_loader.load_pinned_medclip`. No reward math is reimplemented
here; this module only fixes the weights/parser/threshold to the spec and
scores from `RolloutTrace.committed_final_box_norm_1000` (never a regex
match mined from raw completion text).

Spatial reward mode
--------------------
`score_group`'s `spatial_reward_mode` selects how the *spatial* component is
derived from the same underlying `RewardComponents` the pipeline already
computes (which always carries the raw, unthresholded `final_iou` alongside
the fixed-threshold binary `spatial_reward` -- see
`rl.rewards.spatial_reward_from_box`, unmodified):

  "binary"     (default): MedGround-R1 Sec. 2.3, unchanged prior behavior --
               spatial_reward = 1.0 if IoU > iou_threshold else 0.0.

  "continuous": Visual-RFT (arXiv:2503.01785, Sec. 3.2.1, Eq. 6) R_IoU --
               spatial_reward = raw IoU directly, no threshold. Matches the
               supervisor-provided reference implementation's
               SPATIAL_REWARD_MODE=continuous exactly: Visual-RFT's full
               detection reward is R_IoU + R_conf + R_format, but R_conf
               (a per-box confidence term) has no counterpart in this
               single-box grounding setup and is omitted -- R_format is
               already this backend's separate format_reward. An unparsed /
               geometrically-invalid completion already carries final_iou =
               0.0 from the pipeline (see `rl.rewards.spatial_reward_from_box`:
               "unmatched bboxes have an IoU of 0", matching Visual-RFT's own
               convention), so no separate invalid-box handling is needed
               here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Sequence, Tuple

from rl.rewards import (
    PARSER_NATIVE,
    MedCLIPSemanticScorer,
    ProductionRewardPipeline,
    RewardComponents,
)

SPATIAL_REWARD_MODE_BINARY = "binary"
SPATIAL_REWARD_MODE_CONTINUOUS = "continuous"
SPATIAL_REWARD_MODES = (SPATIAL_REWARD_MODE_BINARY, SPATIAL_REWARD_MODE_CONTINUOUS)


def build_reward_pipeline(semantic_scorer: MedCLIPSemanticScorer) -> ProductionRewardPipeline:
    return ProductionRewardPipeline(
        semantic_scorer,
        format_weight=1.0,
        spatial_weight=1.0,
        semantic_weight=1.0,
        iou_threshold=0.5,
        parser_name=PARSER_NATIVE,
        invalid_box_semantic_fallback=0.0,
    )


def _as_continuous(pipeline: ProductionRewardPipeline, component: RewardComponents) -> RewardComponents:
    spatial = float(component.final_iou)
    total = (
        pipeline.format_weight * component.format_reward
        + pipeline.spatial_weight * spatial
        + pipeline.semantic_weight * component.semantic_reward
    )
    return dataclasses.replace(component, spatial_reward=spatial, total_reward=float(total))


def score_group(
    pipeline: ProductionRewardPipeline,
    pair: Dict[str, Any],
    traces: Sequence[Any],
    *,
    spatial_reward_mode: str = SPATIAL_REWARD_MODE_BINARY,
) -> Tuple[List[float], List[RewardComponents]]:
    if spatial_reward_mode not in SPATIAL_REWARD_MODES:
        raise ValueError(f"spatial_reward_mode must be one of {SPATIAL_REWARD_MODES}, got {spatial_reward_mode!r}")
    components = [pipeline.score_from_trace(trace, pair) for trace in traces]
    if spatial_reward_mode == SPATIAL_REWARD_MODE_CONTINUOUS:
        components = [_as_continuous(pipeline, c) for c in components]
    totals = [float(c.total_reward) for c in components]
    for value in totals:
        if value != value:  # NaN check without importing math
            raise RuntimeError("non-finite reward encountered")
    return totals, components
