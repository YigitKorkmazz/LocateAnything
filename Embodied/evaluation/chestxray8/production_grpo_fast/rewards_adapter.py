"""Reward pipeline construction: format(1) + spatial(1) + semantic(1).

Thin wrapper around the existing, unmodified `rl.rewards.ProductionRewardPipeline`
and `rl.medclip_loader.load_pinned_medclip`. No reward math is reimplemented
here; this module only fixes the weights/parser/threshold to the spec and
scores from `RolloutTrace.committed_final_box_norm_1000` (never a regex
match mined from raw completion text).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from rl.rewards import (
    PARSER_NATIVE,
    MedCLIPSemanticScorer,
    ProductionRewardPipeline,
    RewardComponents,
)


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


def score_group(
    pipeline: ProductionRewardPipeline,
    pair: Dict[str, Any],
    traces: Sequence[Any],
) -> Tuple[List[float], List[RewardComponents]]:
    components = [pipeline.score_from_trace(trace, pair) for trace in traces]
    totals = [float(c.total_reward) for c in components]
    for value in totals:
        if value != value:  # NaN check without importing math
            raise RuntimeError("non-finite reward encountered")
    return totals, components
