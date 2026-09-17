# production_grpo_fast — architecture proposal (Phase 2)

Design for a new, high-throughput GRPO backend for `nvidia/LocateAnything-3B`
on the ChestX-ray8 grounding task. Written after reading the existing `rl/*`
scientific modules and after finding that this exact throughput problem has
already been explored empirically in this directory (uncommitted, dated
today) via `single_gpu_locateanything_grpo_fast_v1..v8*.py` and
`multigpu_v6_data_parallel_probe.py`, with JSON reports under
`results/chestxray8_profile_isolated/` and `results/chestxray8_single_gpu_grpo/`.
Those reports supply real measurements on this hardware and are cited below.
None of that code is modified or imported as a dependency; it is used only as
validated prior art. This backend is implemented fresh in this directory.

## 1. What the prior exploration already proved

Three empirically-validated facts materially shape this design, so the new
backend adopts them rather than re-deriving from scratch (each is
independently re-verified by this backend's own correctness tests, not taken
on faith):

1. **Generation-time log-probs can replace a redundant "old-policy replay" pass.**
   `LocateAnything`'s NTP (slow/autoregressive) sampler already records the
   exact filtered log-prob of every sampled token on `SlotTrace.log_prob_old`.
   `results/chestxray8_profile_isolated/v6_cached_old_research/v6_cached_old_logprobs_report.json`
   shows token-for-token identity (`filtered_logprob_abs_delta.max == 0.0`
   over 1527 tokens, `support_changed_count == 0`) between that cached value
   and a from-scratch teacher-forced replay of the same trajectory, and a
   full one-optimizer-update A/B (`old_replay` vs `cached_old`) gives
   `max_loss_delta == 0.0`. Using the cached value removes an entire
   full-sequence forward pass per rollout (was ~7s/group, ~15–17% of total
   time) for free.
2. **Batched (padded) multi-sequence decoding is numerically unsafe at
   top_p=0.9** and was rejected in favor of: one shared single-row prompt
   prefill (prompt tokens are identical across the G=4 rollouts of a group,
   so the prefill forward is deterministic and reusable), then G independent
   serial single-token decode loops branching from a cloned KV cache. This
   keeps every sampled token in the same nucleus a serial replay would
   produce (`generation_vs_old_replay.support_changed_count == 0` in the same
   report) while eliminating 3 of 4 redundant prompt-prefill forward passes
   per group.
3. **The differentiable current-policy replay (forward+backward through all
   36 decoder layers, kept fully live-cached for exact GRPO gradients) is the
   dominant cost**, not generation: `results/.../gpu1_structure/run_profile20/profile_summary.json`
   and the v6 report attribute ~51% of wall time to backward recomputation
   and ~30% to the current-policy forward, vs. ~14% for the actual token
   sampling loop and <3% for vision+MedCLIP combined. Checkpointing all 36
   layers is required to fit in 24GiB (`fits_on_24gib_without_checkpoint:
   false` in the v7 report) but checkpointing *every* layer is not optimal —
   a partial/selective checkpoint schedule recovered ~10% more throughput
   while still fitting the VRAM budget. `torch.compile`/CUDA graphs were
   tried and rejected (v8: `best_safe_backend: "v6 eager"`).
4. A prior synchronous multi-GPU averaged-gradient probe (`multigpu_v6_data_parallel_probe.py`)
   confirms the *design* (independent groups per GPU, mean/sum LoRA-gradient
   reduction, one Adam step) is the right shape, but its own numerical parity
   gate did not pass at the tolerance it set for itself
   (`multigpu_v6_data_parallel_report.json`: `parity_pass: false`,
   `update_vs_ref1.cosine_similarity` 0.90–0.99). Reward/advantage were
   bit-identical across GPUs (`reward_max_abs_delta: 0.0`); only the backward
   pass differed, at a magnitude consistent with ordinary bf16 cross-device
   float non-associativity (comparable to what any multi-GPU bf16 DDP job
   exhibits) rather than a semantic defect. This backend re-runs that same
   class of check with a tolerance appropriate to *statistical* RL training
   (where rollout/reward Monte-Carlo variance already dwarfs this noise) and
   reports the number rather than asserting bit-exactness — see §8.

## 2. Non-negotiables carried over unchanged

Model `nvidia/LocateAnything-3B` @ revision `c32291ca5e996f5a7a485845b4f57a233936bba0`
(per the user's spec — the recent probes above used a different, newer
revision; this backend does not follow them there). Native output format,
0–1000 xyxy, reward weights 1/1/1, spatial IoU>0.5 (strict), MedCLIP cosine
via the pinned loader, LoRA r=8/alpha=16/dropout=.05 on
`q/k/v/o_proj, gate/up/down_proj` across all 36 layers (504 tensors,
14,966,784 params), projector and vision frozen, BF16, seed 42, G=4,
sampling `T=1.2, top_p=0.9, top_k=0, rep_penalty=1.0`, `max_new_tokens=256`,
KL off (β=0), AdamW lr=1e-6, wd=0, clip=1.0, linear decay, 0 warmup, 790
train examples / no internal validation split / heldout194 never touched,
**4 prompt groups per optimizer step (16 completions/step)**.

## 3. Process layout and GPU assignment

GPU0 is excluded: it was occupied by another user's job during Phase 1 and
the pre-existing project convention in this same directory
(`multigpu_v6_data_parallel_probe.py`) already treats it as off-limits for
this exact kind of experiment. Available: GPU 1/2/3 (RTX 3090, 24GiB each).

- **1 GPU**: single process, 4 prompt groups accumulated sequentially, then
  one optimizer step. This is the reference implementation everything else
  is checked against, and the safest fallback.
- **2 GPUs (primary production design)**: one `torch.distributed` (NCCL)
  process per GPU, full model replica each. Each worker sequentially runs 2
  prompt groups (its disjoint shard of the epoch's shuffled order),
  accumulating `.backward()` without an optimizer step. Workers then
  `all_reduce(SUM)` the LoRA `.grad` tensors (only ~15M params → ~60MB fp32,
  all-reduce is not a bottleneck), clip, and both step their (identical)
  optimizer state identically — no `DistributedDataParallel` wrapper needed,
  since only backward-accumulated gradients need to be synchronized, not
  activations. `2 workers × 2 groups = 4 groups/step`, matching the fixed
  contract exactly, and matching the "natural design" the task spec itself
  suggests.
- **3 GPUs**: 4 is not evenly divisible by 3 workers while keeping every
  worker's local group count equal (the spec requires *exactly* 4
  groups/step, not "approximately"). Rather than give one worker a
  different local batch than the others (which the spec says to avoid
  unless "clean"), the 3-GPU configuration keeps the 2-GPU policy design
  (2×2) on GPUs 1/2 and uses GPU 3 to run the MedCLIP reward model plus
  rollout image preprocessing as a dedicated, pipelined service, so reward
  scoring for group *N* overlaps with generation/replay of group *N+1*
  instead of serializing on either policy GPU. This is offered as a
  **benchmarked alternative**, not a silent redefinition of the 4-group
  contract. A second, purely diagnostic 3-GPU point (3×un-even accumulation)
  is also benchmarked for raw scaling curve purposes only and is explicitly
  labeled as not used for the production contract.

## 4. Generation strategy

Reimplementation of the validated "shared single-row prefill, then G
independent serial NTP decode loops from a cloned KV cache" design (§1.2),
built directly on the existing, unmodified `rl/pbd_rl.py` primitives
(`PBDSamplingConfig`, `_sample_slot`, `_forward_language_model`,
`_truncate_legacy_cache`, `BlockTrace`/`RolloutTrace`) and
`rl/hybrid_rl.resolve_hybrid_token_ids`, and validated against
`rl.ntp_rl.StochasticNTPRLDecoder` (the "safe", unmodified reference decoder)
token-for-token on fixed seeds. Every sampled token's filtered log-prob is
retained on the trace (`cached_old_logprob_sum`) for direct reuse as
`π_old`, so old-policy replay is never re-run in the hot path.

## 5. Visual-feature strategy

`model.extract_feature(pixel_values, image_grid_hws)` → `model.mlp1(...)` is
computed once per prompt group under `torch.no_grad()`, then injected as an
already-projected `visual_features` tensor into: the shared prefill, all G
decode loops, and current-policy replay for all 4 rollouts (subclassing the
decoder/replayer's feature-fetch hook, matching the validated
`CachedVisualNTPRLDecoder`/`CachedVisualNTPRolloutReplayer` pattern). This is
exact because the vision encoder and projector are frozen and deterministic;
correctness is checked by asserting the cached tensor has
`requires_grad is False` and by comparing rewards against an uncached
from-scratch run on a fixed trace.

## 6. Reward strategy

`rl.rewards.ProductionRewardPipeline` with `parser_name=PARSER_NATIVE`,
weights 1/1/1, `iou_threshold=0.5`, is used unmodified, scored from
`RolloutTrace.committed_final_box_norm_1000` (never a regex mined from raw
text). `rl.medclip_loader.load_pinned_medclip` is used unmodified. Batching
the up to 4 valid ROI crops of a group into one MedCLIP forward call (and
caching the text-query embedding once per group, since it's identical across
the 4 rollouts) is implemented as an opt-in, parity-tested path — measured
at <1% of per-group time, so it is a minor win, included for completeness
but not relied on for the headline throughput numbers.

## 7. Current-policy scoring / gradient computation

Teacher-forced replay of the exact committed token sequence, `model.eval()`,
BF16 activations, full trajectory scored (not truncated BPTT — that backend
is explicitly rejected upstream and is not reachable from this code). Kept
gradient-live (no `.detach()` on `past_key_values`) so LoRA gradients are
exact, using selective functional KV-layer checkpointing (§1.3) tuned to the
largest checkpoint-free fraction of the 36 layers that keeps peak VRAM under
budget for `max_new_tokens=256` — measured empirically in Phase 6 rather than
copied from the earlier probes (different max_new_tokens profile: 256 here
vs the historical 512 in some configs). `grpo_clipped_loss` (`rl.grpo`,
unmodified) is used unmodified; nothing besides
`current_log_probs, old_log_probs, advantages, clip_epsilon` feeds the loss.

## 8. Gradient accumulation / reduction / optimizer semantics

Each rollout's loss is divided by `G * GROUPS_PER_UPDATE = 4 * 4 = 16` (not
just `G=4`) before `.backward()`, and `optimizer.zero_grad()` /
`optimizer.step()` bracket a 4-group *window*, not a single group. This
makes "4 groups/update" a pure accumulation-window change, not a change to
the per-rollout gradient math already proven correct upstream. Cross-GPU
reduction is `all_reduce(SUM)` of the already-16-normalized local
gradients — mathematically the mean over the true global batch regardless of
how the 4 groups are split across workers. Correctness gate: single-process
4-group sequential reference vs. 2-GPU (2×2) and 3-GPU distributed runs on
matched seeds, checked on cosine similarity, relative gradient-norm delta,
and the resulting parameter update — reported with an explicit numerical
tolerance and an explanation (not silently assumed bit-exact; see §1.4).

**790 is not evenly divisible by 4.** Rather than emit a smaller (non-4-group)
update at every epoch boundary — which would silently violate the fixed
4-groups/update contract 20 times over the run — the accumulation window is
defined over the *continuous* 15,800-group stream (20 epochs × 790), not
reset per epoch. Since `15800 / 4 = 3950` exactly, this yields exactly 3,950
optimizer steps with **no partial-size update ever occurring**; a handful of
accumulation windows straddle an epoch boundary (the epoch's last 2 groups +
next epoch's first 2 groups). Epoch boundaries remain checkpoint points
(§9); the checkpoint additionally records how many of the 4 groups in the
*currently open* window have already been accumulated, so resume never
double-accumulates or drops a group.

## 9. Checkpoint / resume

Saved once per epoch (atomic temp-dir-then-rename, matching the validated
pattern in the prior single-GPU work): LoRA adapter (`safetensors`),
optimizer state dict, linear-schedule state, global optimizer step, epoch,
global 0..789 sample cursor *within the epoch's deterministic
`Random(42+epoch).shuffle`*, groups-attempted-in-open-window (0–3) plus the
identity of those groups' seeds, and full RNG state (`python`, `numpy`,
`torch`, `torch.cuda` — per rank in the multi-GPU case). Resume restores all
of this before the next group is drawn, so no example is duplicated or
skipped and Adam/scheduler state is exact, not reset.

## 10. Data order / sampler

`order = list(range(790)); Random(42 + epoch).shuffle(order)`, one full pass
per epoch, reused verbatim from the validated prior pattern. In the
multi-GPU case the epoch's `order` is computed identically on every rank
(same seed) and then sliced round-robin by rank so ranks cover disjoint,
collectively-exhaustive indices with no duplicates. `heldout194`
(`splits/test_pairs_seed42.jsonl`) is never referenced by any file in this
backend; a startup assertion checks the resolved train manifest path/hash
against `splits/train80_pairs_seed42.jsonl` only.

## 11. Expected bottlenecks going in (revised after Phase 9 profiling)

Current-policy forward+backward (§1.3) is expected to remain the largest
single cost even after selective checkpointing, since it is fundamentally a
36-layer, bf16, ~250-token backward pass through a 3B-parameter decoder with
live KV-cache gradients — LoRA lets us skip an optimizer over the full
model, but not skip the backward pass itself. Multi-GPU throughput is
expected to scale close to linearly in *rollout* throughput (each worker's
2 groups are fully independent right up to the gradient all-reduce, which is
cheap for a 15M-parameter LoRA), with the practical ceiling set by
NCCL/barrier overhead and any load imbalance from variable completion length
between workers.
