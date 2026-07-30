"""ChestX-ray8 reward-only GRPO components."""

from .grpo import grpo_clipped_loss, group_relative_advantages
from .pbd_rl import (
    BlockTrace,
    PBDSamplingConfig,
    PBDRolloutReplayer,
    RolloutTrace,
    SlotTrace,
    StochasticPBDRLDecoder,
    build_filtered_categorical,
    sample_pbd_block,
    score_pbd_block,
)
from .prompt import CHAIN_OF_BOX_PROMPT_TEMPLATE, build_chain_of_box_prompt
from .rewards import ProductionRewardPipeline, parse_chain_of_box_completion

__all__ = [
    "BlockTrace",
    "CHAIN_OF_BOX_PROMPT_TEMPLATE",
    "PBDSamplingConfig",
    "PBDRolloutReplayer",
    "ProductionRewardPipeline",
    "RolloutTrace",
    "SlotTrace",
    "StochasticPBDRLDecoder",
    "build_chain_of_box_prompt",
    "build_filtered_categorical",
    "grpo_clipped_loss",
    "group_relative_advantages",
    "parse_chain_of_box_completion",
    "sample_pbd_block",
    "score_pbd_block",
]
