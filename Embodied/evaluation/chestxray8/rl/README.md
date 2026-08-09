# ChestX-ray8 reward-only GRPO (native LocateAnything)

This directory implements MedGround-R1-style **reward-only GRPO**
(`L_total = L_GRPO` only: no NTP/MTP CE, bbox CE, coordinate regression, or
reasoning supervision).

There are three resolved native/CoB configs:

| Config | Decoder | Log-prob objective |
|--------|---------|--------------------|
| `chestxray8_grpo_pbd.yaml` | stochastic PBD | Chain-of-Box prompt + CoB format |
| `chestxray8_grpo_pbd_native.yaml` | stochastic PBD-only | native format; committed PBD box |
| `chestxray8_grpo_hybrid_native.yaml` | **Hybrid (primary)** | **`full_trajectory`** |
| `chestxray8_grpo_hybrid_native_conditional_ablation.yaml` | Hybrid ablation | `conditional_committed_output` surrogate |

## Shared rewards (PBD-only and Hybrid)

Format, spatial, and semantic definitions are shared:

1. **Format** — native LocateAnything grammar on the emitted completion
   (or CoB `<think>/<answer>` in the CoB config).
2. **Spatial** — MedGround-R1: `R_spatial = 1` iff
   `IoU(committed_final_box, GT) > 0.5`, else `0`.
3. **Semantic** — MedGround-R1: crop the committed final bbox ROI and score
   frozen MedCLIP image/text cosine similarity.

Spatial and semantic always consume the **single decoder-committed final
bbox** from `RolloutTrace` (`score_from_trace`). They never mine an arbitrary
`BOX_RE` match from malformed text, never use a rejected PBD proposal, and
never select a box by GT IoU among candidates.

Unambiguous committed box iff `completed_box_count == 1`; otherwise format /
spatial go to zero and semantic uses the configured invalid-box fallback.

## Hybrid log-prob objectives

The NTP fallback branch is a **deterministic** function of the sampled PBD
proposal `a_pbd`. Therefore rejected proposal tokens that determine the gate
are part of the Hybrid rollout trajectory.

### Production gate vs saved viability (Aug 2026)

Production Hybrid falls back **only** when `handle_pattern` returns
`error_box` for a **single** MTP proposal (malformed / incomplete coordinate
frame). It does **not** fall back for a finished sequence that simply
contains multiple otherwise-valid `coord_box` blocks.

The first Hybrid viability run (`viability_100_seed42_full_trajectory`) had
**0** NTP fallbacks because Hybrid RL incorrectly reused PBD-only
`sample_pbd_block`, which forces coordinate + `</box>` supports after
`<box>` and makes `error_box` unreachable. The 86 multi-`coord_box` failures
were **post-hoc** `completed_box_count != 1` rejections after the full
completion was already generated — not online Hybrid fallbacks.

Hybrid RL now samples MTP blocks with unrestricted (full-vocab) supports and
applies the same per-block `handle_pattern` gate as production.

### Primary: `full_trajectory` (`chestxray8_grpo_hybrid_native.yaml`)

```text
log π(τ) = Σ_i log π(a_pbd,i) + Σ_j log π(a_ntp,j)
```

Includes the full sampled PBD proposal (accepted or rejected slots) plus
committed NTP fallback tokens. This **is** the probability of the complete
Hybrid trajectory. Reward still uses only the committed final bbox.

### Surrogate ablation: `conditional_committed_output`

```text
log π̃(o) = Σ_{committed prefix} log π(a_i) + Σ_j log π(a_ntp,j)
```

Excludes rejected proposal tokens. This is **not** `log π(τ)`. Use only via
`chestxray8_grpo_hybrid_native_conditional_ablation.yaml`.

See `hybrid_rl.py` for the formal definition.

## Entry points

- Viability (100 samples, zero optimizer updates): `run_pbd_rl_viability.py`
- Full GRPO training: `run_pbd_rl_train.py` / `train_pbd_grpo.py`
- Multimode evaluation: `run_pbd_rl_eval.py`

## Memory-safe exact GRPO replay

Training uses sequential G=4 **production-matching truncated-BP** replay
(`rl/grpo_train_step.py`):

1. Group advantages over all G=4 rewards.
2. Old/reference scoring: sequential `no_grad`, production cached semantics.
3. Current policy (one trajectory at a time):
   - Pass A: `no_grad` production-cached score for the PPO ratio / loss scalar.
   - Pass B: same production windows, `position_ids`, and
     `one_gen_window` masks, but **stop-grad** carried `past_key_values`, with
     per-block `(∂ℓ/∂logπ · logp_i).backward()`.
4. One `optimizer.step()` after the G=4 group.

Live enable_grad through production KV is **unsupported on 24GB** (Case B:
LoRA graph retained in cache → ~6GB growth → disguised CUDA
`invalid argument` OOM). Cacheless Bfix is **rejected** (A ≠ Bfix ≈ 0.098;
see `rl/bfix_discrepancy.py`). Gradient checkpointing over mutable KV remains
hard-disabled. Reference log-probs are metric-only. Config knobs:

```yaml
training:
  replay_microbatch_size: 1
  gradient_replay_use_cache: false
  gradient_checkpointing: false  # required: KV-cache replay ≠ checkpoint-safe
```

CPU equivalence: `test_grpo_memory_safe_equivalence.py`.
Manual GPU equivalence (user-run): `verify_exact_replay_equivalence.py --mode full`.

Sampling for the first resolved experiments uses unfiltered categorical
draws (`temperature=1`, `top_k=0`, `top_p=1`, `repetition_penalty=1`).

## Policy memory

`PolicySnapshot` swaps LoRA + `mlp1` onto one shared frozen base for
old/current/reference scoring. Synchronize the old policy only after each
completed `optimizer.step()` (`old_policy_sync_interval=1`).

## MedCLIP identity

`medclip_loader.py` verifies pinned local backbone / weight hashes before
constructing `MedCLIPModel(vision_cls=MedCLIPVisionModelViT)`. Upstream
commit: `9c3396f20d5d54e4fae241b8cb06ca45848e98c9`.

## Viability (Hybrid primary — do not confuse with the ablation)

```bash
cd /auto/k2/ykorkmaz/LocateAnything-playground/Embodied/evaluation/chestxray8

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/auto/k2/ykorkmaz/LocateAnything-playground/Embodied:/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/evaluation/chestxray8 \
python -u run_pbd_rl_viability.py \
  --config rl/chestxray8_grpo_hybrid_native.yaml \
  --output-dir /auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/chestxray8_hybrid_grpo_native/viability_100_seed42_full_trajectory \
  --device cuda
```
