# production_grpo_fast — final report

High-throughput GRPO training backend for `nvidia/LocateAnything-3B` on
ChestX-ray8 grounding. New backend, new files, at
`evaluation/chestxray8/production_grpo_fast/`. No existing file in the
repository was modified. See `DESIGN.md` for the Phase-2 architecture
proposal this implementation follows; this file is the Phase-10 final
report, written after implementation, correctness gating, and benchmarking
on real hardware (GPUs 1-3; GPU 0 excluded throughout, see §2).

**The full 20-epoch / 15,800-group / 3,950-optimizer-step production run has
NOT been launched.** Everything below is implementation, unit/integration
correctness gates, and bounded benchmarks (52 prompt groups / 208
completions per GPU configuration), as instructed.

---

## 1. Proposed architecture

See `DESIGN.md` for the full proposal. Summary: reuse the existing,
unmodified `rl/*` scientific primitives (parsers, rewards, MedCLIP, model
loading, PBD/NTP sampling internals) as a fixed reference; build a new
orchestration layer around them that (a) shares one prompt-prefill forward
across a group's G=4 rollouts and reuses the frozen vision+projector
features for the whole group, (b) reads each rollout's `π_old` directly
from its generation-time filtered log-prob instead of re-running a
redundant teacher-forced replay pass, (c) scores the current policy with a
differentiable, functional-KV-checkpointed replay across all 36 decoder
layers, (d) accumulates exactly 4 prompt groups per optimizer step by
dividing every rollout's loss by `G * groups_per_update = 16` before
`.backward()` and only stepping after the window closes, and (e)
synchronizes that window across GPU workers with a plain SUM all-reduce of
LoRA gradients (no `DistributedDataParallel` needed — only ~15M parameters
of gradient ever cross the wire).

Two of these three techniques ((a)/(b)) were not invented from scratch: they
were found, already validated, in uncommitted prior exploration sitting in
this same directory (`single_gpu_locateanything_grpo_fast_v6_cached_old_logprobs.py`
and `multigpu_v6_data_parallel_probe.py`, with JSON reports under
`results/chestxray8_profile_isolated/`). That prior work is *not* imported
or modified — it was read as design reference and its central claims
(exact old-logprob caching; exact shared-prefill decode) were independently
re-verified from scratch against the unmodified `rl.ntp_rl` reference
decoder in this backend's own test suite (§17), because "a prior report
said it was exact" is not the same as "this backend's own generation code
is exact."

## 2. Why 1, 2, or 3 GPUs

All 4 physical RTX 3090s were available at the start of this session, but
GPU 0 was occupied by another user's job during Phase 1 (`nvidia-smi` showed
an unrelated `locateanything` process holding ~15GB / 100% util). The
pre-existing project convention in this exact directory
(`multigpu_v6_data_parallel_probe.py` hard-refuses GPU 0) independently
confirms this is the established norm here, not a one-off. This backend
follows the same rule everywhere: `distributed.refuse_gpu0` is called on
every multi-GPU launch path and is unit-tested.

Within GPUs 1-3, the task's own "exactly 4 groups/update" contract argues
for 2 GPUs as the clean primary design (2 workers × 2 groups each), and 3
GPUs as a valid-but-marginal alternative (2+1+1 groups): the spec allows an
uneven split as long as it sums to exactly 4, so 3 GPUs *can* honor the
contract, but because the busiest worker still processes 2 groups either
way, 3-GPU wall-clock time is set by the same critical path as 2-GPU — the
third GPU can only reduce the *other* two workers' idle-waiting, not the
window's floor. Measured throughput confirms this: 3-GPU beats 2-GPU by only
~13%, far short of the extra 50% capacity added (§20/§22). **2 GPUs is the
recommended production default**; 3 GPUs is offered as a validated,
correctness-gated option for when the wall-clock floor still matters enough
to justify the marginal gain and a spare GPU is otherwise idle.

## 3. Files created

All under `evaluation/chestxray8/production_grpo_fast/` (new directory, no
existing file touched):

| File | Purpose |
|---|---|
| `DESIGN.md` | Phase-2 architecture proposal |
| `README.md` | this report |
| `runtime.py` | model/tokenizer/LoRA/optimizer construction, trainability + gradient audits |
| `generation.py` | shared-prefill + G serial NTP decodes, visual-feature cache, cached-`π_old` |
| `replay.py` | differentiable current-policy replay, selective functional-KV-layer checkpointing |
| `rewards_adapter.py` | fixed-weight wrapper around the existing reward pipeline |
| `sampler.py` | deterministic epoch order, continuous 4-group accumulation windows, distributed sharding |
| `grpo_step.py` | one prompt group end-to-end; the 4-group accumulation window |
| `distributed.py` | NCCL process-group setup, LoRA-gradient SUM all-reduce, GPU-0 refusal |
| `checkpoint.py` | save/resume (adapter + optimizer + scheduler + RNG + cursor) |
| `train.py` | main entrypoint (single- or multi-GPU via `torch.multiprocessing.spawn`) |
| `benchmark.py` | throughput harness driving `train.py --max-windows` |
| `config.yaml` | documented, non-authoritative mirror of the pinned contract (code constants are authoritative) |
| `test_sampler.py` | no-GPU: data order / window / distributed-shard correctness |
| `test_reward_parity.py` | no-GPU: reward wiring (parser/weights/strict-IoU/semantic fallback) |
| `test_no_heldout_access.py` | no-GPU: static scan for forbidden manifest references |
| `test_trainability_audit.py` | GPU: 504-tensor / 14,966,784-parameter contract |
| `test_generation_parity.py` | GPU: shared-prefill decode vs. unmodified reference decoder |
| `test_one_optimizer_update.py` | GPU: one real 4-group/16-rollout/1-step update |
| `test_checkpoint_resume.py` | GPU: bit-exact save/resume |
| `test_multi_gpu_gradient_reduction.py` | GPU: 2-GPU and 3-GPU vs. single-process reference |
| `_dp_reference_worker.py` | helper subprocess for the multi-GPU gradient test |

## 4. Existing utilities reused (unmodified)

`train_chestxray8_sft.{load_tokenizer_and_processor, load_locateanything_model,
apply_llm_lora, build_optimizer}`, `sft_common.LLM_LORA_TARGET_MODULES`,
`rl.runtime.{tokenize_rl_pair, load_verified_pairs}`, `rl.prompt.*` (via
`tokenize_rl_pair`'s native-mode config), `rl.rewards.{ProductionRewardPipeline,
MedCLIPSemanticScorer, PARSER_NATIVE}`, `rl.medclip_loader.load_pinned_medclip`
(transitively), `rl.grpo.{group_relative_advantages, grpo_clipped_loss}`,
`rl.pbd_rl.{PBDSamplingConfig, _sample_slot, _forward_language_model,
_truncate_legacy_cache, _cache_length, _unwrap_lm_output, BlockTrace,
RolloutTrace}`, `rl.hybrid_rl.resolve_hybrid_token_ids`,
`rl.ntp_rl.{NTPRolloutReplayer, StochasticNTPRLDecoder, validate_ntp_only_trace}`,
`rl.final_prediction.box_from_token_span`, `rl.replay_memory.{_decoder_layers,
assert_model_eval_for_replay}`, `eval_locateanything_bbox.{BOX_RE, box_iou,
build_final_user_query, DIRECT_DISEASE_QUERY_TEMPLATE}` (transitively).

## 5. Model / trainability audit

Loaded via `train_chestxray8_sft.load_locateanything_model` at revision
`c32291ca5e996f5a7a485845b4f57a233936bba0` (the user-specified revision —
**not** the newer revision some of the uncommitted prior exploration
scripts used). Measured on real hardware:

```
trainable_tensors:      504
trainable_parameters:   14,966,784
lora_tensors:            504
non_lora_trainable:      0
projector_trainable:     0
vision_trainable:        0
```

Exact match to spec. LoRA r=8/alpha=16/dropout=0.05 on
`{q,k,v,o}_proj, {gate,up,down}_proj` across all 36 `language_model` decoder
layers (36 × 7 × 2 = 504). Vision encoder and projector (`mlp1`) are frozen
by construction: `runtime.build_model_and_tokenizer` never calls
`unfreeze_mlp1`.

## 6. Dataset / sampler semantics

Train manifest: `splits/train80_pairs_seed42.jsonl`, loaded via
`rl.runtime.load_verified_pairs` with an explicit SHA256 check
(`a2b1c25f...c2e3b0`) against 790 expected rows. No internal validation
split. `splits/test_pairs_seed42.jsonl` (heldout194) and
`splits/val_pairs_seed42.jsonl` are never referenced anywhere in this
package — enforced by a static test (`test_no_heldout_access.py`) that
scans every source file for those literal filenames, not just documented by
convention.

One epoch = `list(range(790))` shuffled by `Random(42 + epoch)`, exactly
reproduced every time it's needed (no persisted "epoch order" object). The
20-epoch / 15,800-group stream is treated as *continuous* for the
4-groups/update accumulation window (window `w` = global groups
`[4w..4w+3]`): since `15800 / 4 = 3950` exactly, this produces exactly 3,950
full-size windows and **never a partial (<4-group) optimizer update**, at
the cost of a handful of windows straddling an epoch boundary (790 mod 4 ==
2). See DESIGN.md §8 for the full argument and `test_sampler.py` for the
exhaustive no-duplicate/no-skip check across every epoch boundary and every
supported world size (1/2/3/4).

## 7. Prompt / output semantics

Prompt: `pair["user_query"]` verbatim (already `"Locate the {Disease} in
this chest X-ray"` in the manifest), routed through
`rl.runtime.tokenize_rl_pair(..., config={"prompt": {"mode":
"native_locateanything"}})` — the existing native-mode path, not the
Chain-of-Box wrapper. Output grammar
(`<ref>{Disease}</ref><box><x1><y1><x2><y2></box>`, 0-1000 xyxy) is never
touched by this backend; it is exactly what the pinned model emits and
exactly what `rl.rewards.PARSER_NATIVE` parses.

## 8. Generation implementation

`generation.generate_group_shared_prefill`: one batch-size-1 prompt-prefill
forward (using the group-local cached visual features, §9), then G
independent batch-size-1 serial NTP decode loops, each branching from a
*clone* of the post-prefill KV cache with its own seeded `torch.Generator`.
Built directly on the unmodified `rl.pbd_rl` sampling primitives
(`_sample_slot`, `PBDSamplingConfig`) — no new sampling math. Padded
multi-row batched decoding was considered and rejected (matches prior
finding: batched top_p=0.9 nucleus differs from serial nucleus, producing
-inf log-probs for some tokens).

**Independently re-verified** (`test_generation_parity.py`, real GPU, real
model): 4 trajectories generated via the shared-prefill path are
**token-for-token identical**, with **0.0 max filtered-logprob delta**,
against `rl.ntp_rl.StochasticNTPRLDecoder` (the unmodified upstream
reference, one full from-scratch generate() per rollout) on the same
prompt and seeds.

`π_old` is the sum of `SlotTrace.log_prob_old` already recorded by
`_sample_slot` at sampling time (`generation.cached_old_logprob_sum`) — no
separate old-policy replay pass is run. Independently re-verified against a
from-scratch teacher-forced replay of the resulting trace: **max abs delta
= 1.9e-6** (bf16 rounding noise), far inside the 1e-3 tolerance.

## 9. Reward implementation

`rewards_adapter.build_reward_pipeline` is a fixed-parameter wrapper around
the unmodified `rl.rewards.ProductionRewardPipeline`:
`parser=PARSER_NATIVE`, `format_weight=spatial_weight=semantic_weight=1.0`,
`iou_threshold=0.5` (strict `>`), `invalid_box_semantic_fallback=0.0`.
Rewards are scored from `RolloutTrace.committed_final_box_norm_1000` /
`has_unambiguous_committed_box`, never mined from raw completion text.
`test_reward_parity.py` independently checks (no GPU, fake semantic scorer):
format=0 with no committed box and the semantic scorer never called; strict
`IoU > 0.5` (an exact IoU==0.5 fixture scores spatial=0, not 1); semantic is
literally the scorer's raw return value, and falls back to 0.0 (not the
scorer's output) for invalid geometry; total is the unweighted sum of the
three components.

## 10. MedCLIP implementation

`rl.medclip_loader.load_pinned_medclip` used unmodified (own SHA256 +
source-commit verification of the MedCLIP weights/backbones). Ran
unbatched (one ROI + one query per call) in this benchmark; profiling
inherited from prior work (§16) puts MedCLIP at well under 1% of per-group
time, so a batched-4-ROIs / cached-text-embedding variant was designed but
not prioritized for implementation — noted as a low-value, low-risk future
optimization rather than pursued here.

## 11. Current-policy scoring implementation

`replay.CachedVisualNTPRolloutReplayer` (subclass of the unmodified
`rl.ntp_rl.NTPRolloutReplayer`) teacher-forces the exact committed token
sequence under `model.eval()`, BF16, with `past_key_values` kept
gradient-live (never detached) — the "scientific" full-trajectory-gradient
path, not truncated BPTT. `replay.selective_kv_layer_checkpointing` wraps
some or all of the 36 decoder layers in `torch.utils.checkpoint` (functional,
non-reentrant, immutable-K/V-snapshot recompute — the same mechanism as the
prior `rl/exact_kv_checkpoint.py` oracle, generalized to an arbitrary layer
subset); the default (used for every correctness gate and every benchmark
below) checkpoints **all 36 layers**, which is required to fit the backward
pass in 24GiB.

## 12. GRPO loss

`rl.grpo.grpo_clipped_loss`, unmodified, called with exactly
`(current_log_probs, old_log_probs, advantages, clip_epsilon=0.2)` — no
KL term, no logits/hidden-state dependence (this is the same function the
existing production trainer's `assert_grpo_loss_depends_only_on_trajectory_logprob`
check already guards). `β=0` (KL off) by never invoking
`rl.medground_kl` at all, not by passing a zero coefficient into it.

## 13. Advantage normalization

`rl.grpo.group_relative_advantages`, unmodified: `(r - mean) / (std + eps)`
computed independently per 4-rollout group, never pooled across groups.
Verified live on real rollouts (`test_one_optimizer_update.py`): 4 real
groups' advantages each have within-group mean ≈ 0; one group happened to
get all-zero rewards (all 4 completions failed format validity) and
correctly produced all-zero advantages (not skipped, not NaN) via the
`+eps` denominator — an edge case real training will hit routinely, handled
by the existing, unmodified scientific reference rather than a new
skip-degenerate-groups heuristic this backend chose not to add (see
DESIGN.md for why: the spec does not authorize dropping a group, and
dropping one would break the fixed "4 groups/update" contract).

## 14. Optimizer configuration

`torch.optim.AdamW` (via `train_chestxray8_sft.build_optimizer`, single
param group since the projector is frozen), `lr=1e-6`, `weight_decay=0`,
default betas `(0.9, 0.999)`, `eps=1e-8`, `clip_grad_norm_` at `1.0` (applied
once, over the whole accumulated window's gradient, after all 4 groups'
`.backward()` calls), linear decay over exactly 3,950 total steps, 0
warmup (`runtime.build_linear_schedule`).

## 15. Exact prompt groups per Adam update

**Exactly 4**, always — not "approximately," not occasionally 1 or 2 at
epoch boundaries. `sampler.GROUPS_PER_UPDATE = 4` is asserted at import time
against `TOTAL_GROUPS % 4 == 0` (15,800 / 4 = 3,950 exactly). Each rollout's
loss is divided by `generations * groups_per_update = 16` before
`.backward()` (`grpo_step.run_accumulation_window`), so 4 groups × 4
rollouts = 16 completions is the true unit of one optimizer step, matching
the spec exactly.

## 16. Gradient accumulation / reduction design

Single process (1 GPU): `zero_grad` → 4× (generate, reward, advantage,
differentiable replay, `.backward()`, each rollout loss pre-divided by 16)
→ `clip_grad_norm_` → `step`. Multi-GPU (2 or 3 workers): each rank runs its
shard of the same 4-group window (2+2 for world_size=2, 2+1+1 for
world_size=3 — `sampler.worker_group_indices`), then `distributed.allreduce_sum_grads`
SUM-reduces every LoRA `.grad` tensor across ranks before the identical
`clip_grad_norm_` → `step` runs on every rank. SUM (not MEAN) is correct
here specifically because every rollout's loss was already divided by the
*global* 16, not by each rank's local count — summing the pre-normalized
local gradients reconstructs the exact global average regardless of how
unevenly the 4 groups are split across workers. Every rank starts from
*exactly* the same LoRA weights via an explicit `dist.broadcast` from rank 0
(`distributed.broadcast_trainable_from_rank0`) rather than relying on
independently-seeded RNG streams across processes to coincidentally agree.

## 17. One-update numerical parity

`test_one_optimizer_update.py` (single process, real GPU, real model, real
4-group/16-rollout window): trainability audit correct, per-group
within-group advantage normalization correct, gradient audit clean
(504/504 finite LoRA gradients, 0 projector/vision/frozen-non-LoRA
gradients, 252/504 nonzero — the other 252 are the LoRA-A matrices, whose
gradient is mathematically exactly zero at initialization because
`dL/dA = Bᵀ · dL/d(output)` and PEFT initializes `B=0`; this is expected
LoRA-at-init behavior, not a bug), and a finite, nonzero optimizer update
norm (`0.00284`).

`test_multi_gpu_gradient_reduction.py` (three real subprocesses: a
single-process 4-group reference, a 2-GPU 2+2 run, a 3-GPU 2+1+1 run, all
starting from an *identical* broadcast initial LoRA state and processing
the identical fixed 4-group window):

| comparison | cosine similarity | relative norm delta | max abs delta |
|---|---|---|---|
| 2-GPU vs. single-process reference | 1.0018 | 9.5e-7 | 1.7e-3 |
| 3-GPU vs. single-process reference | 1.0018 | 1.4e-6 | 2.1e-3 |

(Cosine similarity printed slightly above 1.0 is floating-point rounding in
the similarity computation itself over a 504-tensor, ~15M-element flattened
vector — the true value cannot exceed 1.0; this indicates near-numerical
identity, not partial correlation.) Both configurations pass with enormous
margin against a floor of 0.999 cosine / 5% relative norm delta — a much
tighter result than a prior, uncommitted exploratory probe achieved on the
same class of comparison (which reported cosine similarities as low as 0.90
and explicitly did not clear its own bar). The most likely reason for the
improvement: this backend forces bit-identical starting weights across
ranks via an explicit broadcast (rather than relying on matching RNG seeds
across independently-spawned processes) and uses the mathematically-correct
global-normalize-then-SUM reduction throughout.

## 18. Gradient audit

Enforced by `runtime.assert_gradient_audit_ok` and exercised on every
correctness gate and (every-50-windows, plus always on window 0) during
real training: exactly 504 finite LoRA gradients, exactly 0 projector
gradients, exactly 0 vision gradients, exactly 0 frozen-non-LoRA gradients,
strictly >0 nonzero LoRA gradients. Never observed to fail in any run in
this work.

## 19-21. Throughput (1 / 2 / 3 GPU)

Method: `benchmark.py` runs `train.py`'s real training loop (unchanged
generation/reward/replay/backward/optimizer code — the benchmark is not a
separate fast-path) for 13 accumulation windows = **52 prompt groups / 208
completions** per configuration, on GPUs 1-3 only, writing per-rank JSONL
logs. This is close to, but did not reach, the "≥50 groups" the spec asked
for as a *preference* ("if practical") — a full 20-epoch run is many days
(§27), so 52 groups is the point chosen to balance a statistically
meaningful sample against session wall-clock time.

**Important methodology caveat**: the 1-GPU and 2-GPU benchmarks below ran
*concurrently* (to save wall-clock time), sharing the machine's 32 CPU
cores and system memory bandwidth with each other; the 3-GPU benchmark then
ran alone. A separate 10-group isolated measurement (no concurrent
benchmark running) gave **98.9 groups/hour** for 1 GPU — noticeably faster
than the 82.4 groups/hour measured under concurrent load, and close to the
prior validated single-GPU reference design's 102.1 groups/hour. The 1-GPU
number below is therefore a conservative, contention-affected figure, not
this backend's ceiling; the 2-GPU number may be mildly affected in the same
direction; the 3-GPU number is measured in isolation and should not be.

| GPUs | groups/hr | sec/group | optimizer steps/hr | completions/hr | scaling vs 1-GPU (ideal=N) |
|---|---|---|---|---|---|
| 1 (measured, concurrent load) | 82.4 | 43.7 | 20.6 | 330 | 1.00x |
| 1 (isolated, no concurrent load) | 98.9 | 36.4 | 24.7 | 396 | -- |
| 2 (2+2) | 141.2 | 25.5 | 35.3 | 565 | 1.71x (86% efficient) |
| 3 (2+1+1) | 159.1 | 22.6 | 39.8 | 636 | 1.93x (64% efficient) |

3-GPU beats 2-GPU by only **1.13x** (not the 1.5x its extra GPU would give
under ideal scaling) — exactly the outcome §2 predicted from the 2+1+1
split's critical path being set by the 2-group worker regardless of how
many idle-faster workers surround it.

## 22. Scaling efficiency

2-GPU: 86% of ideal (2.0x) linear scaling. 3-GPU: 64% of ideal (3.0x)
linear scaling relative to 1-GPU, but only a further 13% on top of 2-GPU.
The marginal-GPU story is the important one for a purchasing/scheduling
decision: **GPU #2 buys ~71% more throughput; GPU #3 buys only ~13% more**
under this fixed-contract design. This is a structural property of forcing
exactly 4 groups/update with an odd worker count, not a bug — see §2.

## 23. Per-GPU VRAM

Sampled live via `nvidia-smi` during the 3-GPU benchmark and cross-checked
against a dedicated 10-group memory-stability diagnostic on an idle GPU:

- Idle / model-loaded baseline: **7.94 GiB allocated** per rank (base
  frozen 3B model + LoRA adapters + optimizer states, all ranks identical).
- Peak during active generation + replay + backward: **~12.2-15.2 GiB
  reserved** per rank, stable across 10 consecutive groups with no growth
  (see §26 for why an explicit `torch.cuda.empty_cache()` between groups was
  necessary to keep it stable). Consistent across all three world sizes —
  each rank runs the same per-group workload regardless of how many other
  ranks exist.
- Comfortable headroom under the 24 GiB budget (~9-12 GiB free at peak) on
  every configuration tested; no OOM in any of the 3 × 52-group benchmark
  runs after the fix in §26.

## 24. CPU / IO utilization

32 physical CPUs available; this workload is GPU-bound (generation,
replay, and backward together are essentially all CUDA kernel time — see
§25). No CPU or storage-IO bottleneck was observed or expected: image
loading is one small PNG per prompt group, MedCLIP CPU-side preprocessing
is a single 224×224 crop+resize, and JSON logging is negligible relative to
30-40 seconds of GPU work per group. A dedicated per-core CPU profile was
not built for this report, since every other measurement (GPU utilization
during active phases, wall-clock breakdown inherited from the prior
validated stage profiler, §25) is consistent with this workload having no
meaningful CPU/IO contention to characterize.

## 25. Main bottleneck

Not rebuilt from scratch as a new profiler in this backend (a deliberate
scope choice — see below), but inherited from, and structurally consistent
with, the prior validated stage-level profiling this design was built from
(`results/chestxray8_profile_isolated/*/profile_summary.json`,
`v6_cached_old_research/v6_cached_old_logprobs_report.json`): of one
prompt group's wall time, roughly **half is the current-policy backward
pass** (checkpoint recomputation through all 36 decoder layers), **another
~30% is the current-policy forward pass** (the same replay, before
backward), and only **~14% is the actual token-sampling generation loop**;
vision+projector and MedCLIP together are consistently under 3%. This
backend's own measured group-level throughput (§19-21) tracks that prior
report's `groups_per_hour` closely enough (82-99 vs. its 102) to support
reusing its stage breakdown rather than re-deriving it, but it was not
independently re-measured stage-by-stage inside `production_grpo_fast`
itself, and that gap should be closed before this backend is used to guide
*further* architecture changes (e.g. before spending more effort on
selective/partial-layer checkpointing, §11, actually re-measure this
backend's own stage split rather than trusting the inherited one to still
hold after the loss-divisor / windowing / multi-GPU changes made here).

## 26. Checkpoint / resume behavior

Saved once per epoch boundary — precisely, right after the accumulation
window that completes at or after the epoch boundary (never mid-window;
`checkpoint.save_checkpoint` asserts `next_global_group_index %
GROUPS_PER_UPDATE == 0`). Contents: LoRA adapter (`safetensors`, via
`model.language_model.save_pretrained`), optimizer state dict, linear-
schedule state, `next_global_group_index` (the resume cursor — see
DESIGN.md §9/§10 for why rollout sampling is fully re-derivable from this
one integer and does not depend on RNG history), and full
`python`/`numpy`/`torch`/`torch.cuda` RNG state (saved for defense-in-depth
even though current correctness does not depend on it). Only rank 0 writes,
because SUM-allreduced gradients keep every rank's optimizer step
bit-identical. **Independently verified** (`test_checkpoint_resume.py`,
real model, real optimizer with 3 real Adam steps of state):
LoRA weights, Adam moment buffers, and scheduler LR all match **bit-for-bit**
after a save→fresh-model→resume round trip; the resumed cursor matches
exactly; RNG state round-trips.

Real memory-stability fix found and applied during this work: without an
explicit `gc.collect()` + `torch.cuda.empty_cache()` after every prompt
group, the CUDA caching allocator fragments over dozens of sequential
`torch.utils.checkpoint` forward/backward cycles and eventually OOMs (first
observed as a real OOM partway through the initial 2-GPU benchmark attempt,
at group ~20-something of a run that had been comfortably under budget for
any single group). A dedicated 10-group diagnostic confirmed allocated
memory is then perfectly flat (7.94 GiB) across all 10 groups with the fix
applied, vs. an unbounded climb toward OOM without it. This fix costs a
small amount of throughput (§25) but is required for the multi-hour/multi-
day runs this backend exists to make practical, and every benchmark number
in §19-21 already includes its cost.

## 27. Estimated runtime for the full run

Using the measured (concurrent-load) throughput figures as the conservative
estimate:

| horizon | 1 GPU | 2 GPU | 3 GPU |
|---|---|---|---|
| 1 epoch (790 groups) | 9.6 hr | 5.6 hr | 5.0 hr |
| 5,000 groups | 60.7 hr (2.5 d) | 35.4 hr (1.5 d) | 31.4 hr (1.3 d) |
| full 15,800 groups / 3,950 steps | **191.8 hr (8.0 d)** | **111.9 hr (4.7 d)** | **99.3 hr (4.1 d)** |

(Using the isolated 1-GPU figure instead: 1 epoch 8.0 hr, 5,000 groups 50.6
hr / 2.1 d, full run 159.8 hr / 6.7 d.) These are long. The full run is
fundamentally expensive because it is 3,950 optimizer steps, each requiring
16 full differentiable 36-layer-backward passes through a 3B-parameter
decoder over ~250-token sequences — LoRA reduces the optimizer's footprint,
not the backward pass's compute. 2 GPUs is the practical recommendation for
actually running this to completion in a reasonable calendar-time window.

## 28. Scientific limitations

- Stage-level profiling for *this* backend (as opposed to its validated
  predecessor) was not independently rebuilt (§25) — the bottleneck
  attribution is inherited, not freshly measured here.
- Benchmarks used 52 prompt groups per configuration, not the "≥50 if
  practical" spec's ideal upper end for capturing tail variance in
  completion length; two of the three benchmark runs shared machine
  resources concurrently (§19-21 caveat).
- MedCLIP batching (§10) was designed but not implemented; current
  semantic-reward cost is unbatched (already <1% of group time, so this is
  a genuinely low-priority gap, not a correctness one).
- The automatic epoch-boundary checkpoint *trigger* inside `train.py`'s
  main loop was validated by code review plus the underlying
  save/resume primitives (§26), not by an actual multi-hour live run
  that crosses a real 790-group epoch boundary — doing so was impractical
  within this session and was not requested.
- Selective (partial-layer, not all-36) KV checkpointing is implemented
  and parameterized (`replay.selective_kv_layer_checkpointing`,
  `train.py --checkpoint-every-layer`) but was not benchmarked in this
  report; all correctness gates and all throughput numbers above use the
  default (all 36 layers checkpointed, the most memory-conservative,
  correctness-first choice). A prior exploratory probe suggested partial
  checkpointing could recover meaningful throughput on 512-token
  sequences; whether it transfers to this backend's 256-token config is an
  open, flagged question, not a claim made here.
- 3-GPU's 2+1+1 split was validated for gradient correctness and measured
  for throughput, but its "2+1+1 doesn't beat 2+2 by much" finding is from
  one 52-group sample; it is a structural, analytically-predicted result
  (§2) and not solely a statistical artifact, but was not re-confirmed at
  a larger sample size.

## 29. Safe for production?

**Yes, for the 2-GPU (2+2) design**, with the caveats in §28 understood and
accepted: every correctness gate specified in the task (parser, native
coordinates, reward parity/weights/threshold, within-group-only advantage
normalization, trainability audit, gradient audit, multi-GPU gradient
reduction, checkpoint/resume exactness, no-heldout-access) passed on real
hardware against the real model, and the memory-stability issue found
during this work has a verified fix already included in every number
reported here. **Single-GPU is also safe** (it is the reference every other
configuration is checked against) and is the right choice if only one GPU
is available or if bit-for-bit-closest-to-single-process behavior is
wanted. **3-GPU (2+1+1) is validated and safe but not recommended as the
default** — it adds real operational complexity (a 3rd process, uneven
per-worker load) for a measured ~13% throughput gain over 2-GPU; use it
only when a 3rd GPU is otherwise idle and the wall-clock floor matters more
than efficiency-per-GPU.

Not yet run, and therefore not itself validated: the actual 20-epoch
production job. Everything upstream of "does the full run behave like the
tested slice" has been checked; the full run itself has not been launched,
per the explicit instruction not to.

## 30. Production launch command (tcsh-safe)

```tcsh
setenv HF_HOME /auto/data2/ykorkmaz/hf_cache
cd /auto/k2/ykorkmaz/LocateAnything-playground/Embodied/evaluation/chestxray8/production_grpo_fast
python train.py --output-dir /auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/chestxray8_production_grpo_fast/PRODUCTION_G4_LR1E6_E20_2GPU_SEED42 --gpus 1,2 --checkpoint-every-epoch
```

To resume after an interruption, add `--resume auto` (resumes the latest
checkpoint found in `--output-dir`) or `--resume /path/to/checkpoint_step_NNNNNN`
for an explicit checkpoint.
