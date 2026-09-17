"""One prompt group -> rollouts -> rewards -> advantages -> backward, and the
4-groups-per-optimizer-step accumulation window around it.

`execute_prompt_group` computes one independent GRPO group end-to-end
(generation, reward, within-group advantage normalization, differentiable
replay, `.backward()`), matching MedGround-R1/rl.grpo semantics exactly:
advantages are normalized only within their own G=4 group
(`rl.grpo.group_relative_advantages`), never across groups.

`run_accumulation_window` composes `GROUPS_PER_UPDATE` (=4) such calls into
one optimizer step by dividing every rollout's loss by
`generations * groups_per_update` (not just `generations`) before
`.backward()`, and only clipping/stepping after the whole window -- see
DESIGN.md Sec 8 for why this reduction is exact for both the single-process
and the cross-GPU-allreduce case.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

import torch

from rl.grpo import group_relative_advantages
from rl.rewards import RewardComponents
from rl.runtime import tokenize_rl_pair

from . import runtime as prod_runtime
from .generation import cache_group_visual_features, cached_old_logprob_sum, generate_group_shared_prefill
from .replay import CachedVisualNTPRolloutReplayer, score_and_backward
from .rewards_adapter import score_group
from .sampler import GENERATIONS

NATIVE_PROMPT_CONFIG = {"prompt": {"mode": "native_locateanything"}}

SAMPLING = {
    "temperature": 1.2,
    "top_p": 0.9,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "block_size": 6,
}
MAX_NEW_TOKENS = 256
CLIP_EPSILON = 0.2
GROUPS_PER_UPDATE = 4


def prepare_group_inputs(processor: Any, pair: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return tokenize_rl_pair(processor, pair, device, config=NATIVE_PROMPT_CONFIG)


def _sampling_config():
    from rl.pbd_rl import PBDSamplingConfig

    cfg = PBDSamplingConfig(**SAMPLING)
    cfg.validate()
    return cfg


def execute_prompt_group(
    model: Any,
    tokenizer: Any,
    processor: Any,
    reward_pipeline: Any,
    pair: Dict[str, Any],
    optimizer_or_none: Optional[torch.optim.Optimizer],
    *,
    device: torch.device,
    sample_seed: int,
    generations: int = GENERATIONS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    loss_divisor: float,
    clip_epsilon: float = CLIP_EPSILON,
    checkpoint_layers: Optional[Set[int]] = None,
    do_backward: bool = True,
    kl_beta: float = 0.0,
) -> Dict[str, Any]:
    del optimizer_or_none
    inputs = prepare_group_inputs(processor, pair, device)
    cached_visual_features = cache_group_visual_features(model, inputs["pixel_values"], inputs["image_grid_hws"])

    sampling = _sampling_config()
    seeds = [int(sample_seed) * generations + j for j in range(generations)]
    model.eval()
    with torch.inference_mode():
        traces = generate_group_shared_prefill(
            model,
            tokenizer,
            inputs["input_ids"],
            cached_visual_features,
            sampling=sampling,
            max_new_tokens=max_new_tokens,
            seeds=seeds,
        )

    totals, components = score_group(reward_pipeline, pair, traces)
    advantages = group_relative_advantages(totals).to(device)

    decoder_kwargs = {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "image_grid_hws": inputs["image_grid_hws"],
    }
    replayer = CachedVisualNTPRolloutReplayer(model, tokenizer, cached_visual_features=cached_visual_features)

    rollout_reports: List[Dict[str, Any]] = []
    for trace, advantage in zip(traces, advantages):
        old_logp = cached_old_logprob_sum(trace, device=device, dtype=torch.float32)
        if do_backward:
            report = score_and_backward(
                replayer,
                decoder_kwargs,
                trace,
                old_logp=old_logp,
                advantage=advantage,
                loss_divisor=loss_divisor,
                clip_epsilon=clip_epsilon,
                checkpoint_layers=checkpoint_layers,
                kl_beta=kl_beta,
            )
        else:
            with torch.no_grad():
                current_logp, _ = replayer.score(trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs)
            report = {
                "current_log_prob": float(current_logp.float().cpu()),
                "old_log_prob": float(old_logp.float().cpu()),
                "ppo_ratio": float(torch.exp(current_logp.float() - old_logp.float()).cpu()),
                "loss": None,
            }
        rollout_reports.append(report)

    return {
        "traces": traces,
        "rewards": totals,
        "reward_components": [c.to_dict() for c in components],
        "advantages": [float(a) for a in advantages.detach().cpu()],
        "rollout_reports": rollout_reports,
        "sample_seed": sample_seed,
        "ious": [float(c.final_iou) for c in components],
    }


def run_accumulation_window(
    model: Any,
    optimizer: torch.optim.Optimizer,
    tokenizer: Any,
    processor: Any,
    reward_pipeline: Any,
    pairs: Sequence[Dict[str, Any]],
    *,
    device: torch.device,
    sample_seeds: Sequence[int],
    groups_per_update: int = GROUPS_PER_UPDATE,
    generations: int = GENERATIONS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    clip_epsilon: float = CLIP_EPSILON,
    max_grad_norm: float = 1.0,
    checkpoint_layers: Optional[Set[int]] = None,
    allreduce_fn: Optional[Callable[[Any], None]] = None,
    audit: bool = False,
    capture_grads_and_skip_step: bool = False,
    kl_beta: float = 0.0,
) -> Dict[str, Any]:
    """Run this rank's local groups of one accumulation window and step.

    `pairs`/`sample_seeds` are this rank's *local* groups only (already
    sharded by `sampler.worker_group_indices`); `loss_divisor` is always the
    *global* `generations * groups_per_update` so that summing gradients
    across ranks (via `allreduce_fn`, expected to be a SUM all-reduce)
    reconstructs the exact mean over the global 4-group/16-rollout batch
    regardless of how many local groups this rank ran.
    """
    if len(pairs) != len(sample_seeds):
        raise ValueError("pairs and sample_seeds must have equal length")
    loss_divisor = float(generations * groups_per_update)
    optimizer.zero_grad(set_to_none=True)
    group_reports: List[Dict[str, Any]] = []
    for pair, seed in zip(pairs, sample_seeds):
        group_reports.append(
            execute_prompt_group(
                model,
                tokenizer,
                processor,
                reward_pipeline,
                pair,
                optimizer,
                device=device,
                sample_seed=seed,
                generations=generations,
                max_new_tokens=max_new_tokens,
                loss_divisor=loss_divisor,
                clip_epsilon=clip_epsilon,
                checkpoint_layers=checkpoint_layers,
                do_backward=True,
                kl_beta=kl_beta,
            )
        )
        # `torch.utils.checkpoint` recompute + many sequential groups without
        # an intervening `optimizer.step()` fragments the CUDA caching
        # allocator over long runs (observed: OOM after dozens of groups on
        # a GPU that was comfortably under budget for any single group).
        # `.grad` accumulation itself is untouched by this -- only cached,
        # already-freed allocator blocks are released back to the pool.
        gc.collect()
        torch.cuda.empty_cache()
    if allreduce_fn is not None:
        allreduce_fn(model)

    grad_audit = prod_runtime.gradient_audit(model) if audit else None
    if audit:
        prod_runtime.assert_gradient_audit_ok(grad_audit)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(max_grad_norm))
    group_ious = [report["ious"] for report in group_reports]
    if capture_grads_and_skip_step:
        grads = {n: p.grad.detach().float().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer.zero_grad(set_to_none=True)
        return {"group_reports": group_reports, "gradient_audit": grad_audit, "grads": grads, "group_ious": group_ious}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"group_reports": group_reports, "gradient_audit": grad_audit, "group_ious": group_ious}
