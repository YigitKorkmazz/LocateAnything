#!/usr/bin/env python3
"""GPU test: shared-prefill serial decode vs the unmodified reference decoder.

Compares `generation.generate_group_shared_prefill` (this backend) against
`rl.ntp_rl.StochasticNTPRLDecoder.generate` (unmodified upstream reference,
one call per rollout, from-scratch prefill every time) on the same prompt
and the same per-rollout seeds. Also checks `cached_old_logprob_sum` against
a from-scratch teacher-forced replay of the resulting trace.

Requires a GPU and the pinned model; not run as part of a fast/no-GPU suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import grpo_step, runtime as prod_runtime, sampler  # noqa: E402
from production_grpo_fast.generation import (  # noqa: E402
    cache_group_visual_features,
    cached_old_logprob_sum,
    generate_group_shared_prefill,
)
from production_grpo_fast.replay import CachedVisualNTPRolloutReplayer  # noqa: E402
from rl.ntp_rl import NTPRolloutReplayer, StochasticNTPRLDecoder  # noqa: E402


def main() -> None:
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
    torch.cuda.set_device(device)
    model, tokenizer, processor, _ = prod_runtime.build_model_and_tokenizer(device)

    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
    from train import load_train80_pairs  # noqa: E402

    pairs = load_train80_pairs()
    spec = sampler.resolve_group(1)
    pair = pairs[spec.manifest_index]
    inputs = grpo_step.prepare_group_inputs(processor, pair, device)
    cached_visual = cache_group_visual_features(model, inputs["pixel_values"], inputs["image_grid_hws"])
    sampling = grpo_step._sampling_config()
    seeds = sampler.rollout_seeds(spec.sample_seed)

    model.eval()
    with torch.inference_mode():
        fast_traces = generate_group_shared_prefill(
            model, tokenizer, inputs["input_ids"], cached_visual,
            sampling=sampling, max_new_tokens=grpo_step.MAX_NEW_TOKENS, seeds=seeds,
        )

    reference_decoder = StochasticNTPRLDecoder(model, tokenizer, sampling=sampling)
    reference_traces = []
    with torch.inference_mode():
        for seed in seeds:
            trace = reference_decoder.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                image_grid_hws=inputs["image_grid_hws"],
                max_new_tokens=grpo_step.MAX_NEW_TOKENS,
                seed=seed,
            )
            reference_traces.append(trace)

    for i, (fast, ref) in enumerate(zip(fast_traces, reference_traces)):
        assert fast.generated_token_ids == ref.generated_token_ids, (
            f"trajectory {i}: token mismatch\nfast={fast.generated_token_ids}\nref ={ref.generated_token_ids}"
        )
        assert fast.committed_final_box_norm_1000 == ref.committed_final_box_norm_1000
        fast_logps = [s.log_prob_old for b in fast.blocks for s in b.slots]
        ref_logps = [s.log_prob_old for b in ref.blocks for s in b.slots]
        max_delta = max((abs(a - b) for a, b in zip(fast_logps, ref_logps)), default=0.0)
        assert max_delta == 0.0, f"trajectory {i}: filtered logprob delta {max_delta}"
    print(f"PASS generation parity: {len(fast_traces)} trajectories, token-for-token identical, 0.0 logprob delta")

    replayer = CachedVisualNTPRolloutReplayer(model, tokenizer, cached_visual_features=cached_visual)
    decoder_kwargs = {
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "image_grid_hws": inputs["image_grid_hws"],
    }
    max_cached_vs_replay_delta = 0.0
    for trace in fast_traces:
        cached = float(cached_old_logprob_sum(trace, device=device, dtype=torch.float32).cpu())
        with torch.no_grad():
            replay_logp, _ = replayer.score(trace, use_cache=True, legacy_nocache_masks=False, **decoder_kwargs)
        delta = abs(cached - float(replay_logp.float().cpu()))
        max_cached_vs_replay_delta = max(max_cached_vs_replay_delta, delta)
    print(f"PASS cached-old-logprob vs from-scratch replay: max abs delta = {max_cached_vs_replay_delta}")
    assert max_cached_vs_replay_delta < 1e-3, "cached old logprob diverges from from-scratch replay beyond bf16 tolerance"
    print("ALL GENERATION PARITY TESTS PASSED")


if __name__ == "__main__":
    main()
