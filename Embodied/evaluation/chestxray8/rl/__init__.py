"""ChestX-ray8 reward-only GRPO components."""

from .final_prediction import (
    attach_pbd_final_prediction,
    box_from_coord_box_tokens,
    box_from_token_span,
)
from .grpo import grpo_clipped_loss, group_relative_advantages
from .hybrid_rl import (
    DEFAULT_LOGPROB_OBJECTIVE,
    HYBRID_POLICY_DOC,
    LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED,
    LOGPROB_OBJECTIVE_FULL_TRAJECTORY,
    VALID_LOGPROB_OBJECTIVES,
    HybridRolloutReplayer,
    StochasticHybridRLDecoder,
    classify_hybrid_mtp_proposal,
    resolve_logprob_objective,
    sample_hybrid_mtp_block,
)
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
from .prompt import (
    CHAIN_OF_BOX_PROMPT_TEMPLATE,
    build_chain_of_box_prompt,
    build_native_locateanything_prompt,
    build_rl_messages,
)
from .rewards import (
    ProductionRewardPipeline,
    build_reward_pipeline_from_config,
    parse_chain_of_box_completion,
    parse_native_locateanything_completion,
    spatial_reward_from_box,
)

__all__ = [
    "BlockTrace",
    "CHAIN_OF_BOX_PROMPT_TEMPLATE",
    "DEFAULT_LOGPROB_OBJECTIVE",
    "HYBRID_POLICY_DOC",
    "HybridRolloutReplayer",
    "LOGPROB_OBJECTIVE_CONDITIONAL_COMMITTED",
    "LOGPROB_OBJECTIVE_FULL_TRAJECTORY",
    "PBDSamplingConfig",
    "PBDRolloutReplayer",
    "ProductionRewardPipeline",
    "RolloutTrace",
    "SlotTrace",
    "StochasticHybridRLDecoder",
    "StochasticPBDRLDecoder",
    "VALID_LOGPROB_OBJECTIVES",
    "attach_pbd_final_prediction",
    "box_from_coord_box_tokens",
    "box_from_token_span",
    "build_chain_of_box_prompt",
    "build_filtered_categorical",
    "build_native_locateanything_prompt",
    "build_reward_pipeline_from_config",
    "build_rl_messages",
    "grpo_clipped_loss",
    "group_relative_advantages",
    "parse_chain_of_box_completion",
    "parse_native_locateanything_completion",
    "classify_hybrid_mtp_proposal",
    "resolve_logprob_objective",
    "sample_hybrid_mtp_block",
    "sample_pbd_block",
    "score_pbd_block",
    "spatial_reward_from_box",
]
