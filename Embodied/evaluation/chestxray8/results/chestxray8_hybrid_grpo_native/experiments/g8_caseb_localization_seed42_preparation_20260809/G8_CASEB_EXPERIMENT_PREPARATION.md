# G=8 Case-B localization experiment preparation

## Verdict

**SAFE to start the bounded fresh 0→150 run.**

This is an operational/correctness verdict, not a claim that G=8 is likely to
solve localization. The saved rollout distribution suggests that doubling the
group is unlikely to materially increase mixed binary-spatial groups. Do not
infer safety or benefit for a 150→500 continuation; apply the step-100
validation rule first and report its evidence.

The 150-step run was not started. Exactly one G=8 optimizer update and one
fresh checkpoint-reload validation were run.

## Preserved contract

The new configuration differs from the validated KL-free Case-B setup only in
group size, optimization/validation manifests, target length, and checkpoint
schedule.

- Native LocateAnything output/parser unchanged.
- Format reward unchanged.
- Spatial reward remains strict `IoU > 0.5` with weight 1.
- Frozen MedCLIP native-ROI/original-query reward unchanged with weight 1.
- Format/spatial/semantic weights remain `1/1/1`.
- Hybrid sampling and commit semantics unchanged.
- Full Hybrid trajectory objective, including rejected PBD proposal tokens.
- Exact live-cache replay; no KV detach.
- No truncated BPTT, gradient checkpointing, or saved-tensor offload.
- Fresh pinned LocateAnything-3B revision
  `c32291ca5e996f5a7a485845b4f57a233936bba0`.
- KL disabled: `L_GRPO`, effective beta `0.0`, no reference snapshot.
- 504 LoRA tensors plus all 6 mlp1 projector tensors trainable: 510 total.
- LoRA LR `2e-5`; projector LR `1e-5`; global `max_grad_norm=1.0`.
- Optimization manifest: 710 examples, SHA-256
  `4edbed2071e4010c83c45ce8d33d341e0813fba2276150c82c990f1c7e9822f6`.
- Internal validation: 80 examples, SHA-256
  `f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838`.
- Target 150 actual optimizer updates; checkpoints at
  25/50/75/100/125/150.
- The 194-case held-out test is not used in any prepared validation command.

## One-update RTX3090 result

The probe ran on two NVIDIA GeForce RTX 3090s with the existing 18/18 decoder
shard and completed one actual optimizer update.

| Check | Result |
|---|---|
| Status | passed |
| Actual optimizer updates | 1 |
| Attempted groups | 1 |
| Trajectories | 8 |
| 504 LoRA gradient tensors present | yes |
| 6 projector gradient tensors present | yes |
| 510 total trainable gradients present | yes |
| All gradients finite | yes |
| Nonzero projector gradients | 6/6 |
| Projector gradient norm | 38.82039446 |
| LoRA gradient norm | 13.40234812 |
| Total pre-clip gradient norm | 41.06879566 |
| Total post-clip gradient norm | 0.99996859 |
| AdamW state finite after step | yes |
| LoRA parameters changed | yes |
| Live KV and cross-device autograd checks | passed |
| OOM | no |

All 510 gradient *tensors* were present and finite. Of these, 258 contained a
nonzero value on this particular group: 252 LoRA plus all 6 projector tensors.
The remaining present LoRA gradients were exactly zero, which is permitted and
distinct from a missing gradient.

### Peak memory

| Device | Peak allocated | Peak reserved | Capacity | Reserved headroom |
|---|---:|---:|---:|---:|
| cuda:0 | 11,434.38 MiB | 12,772 MiB | 24,124.19 MiB | 11,352.19 MiB |
| cuda:1 | 10,513.23 MiB | 11,842 MiB | 24,124.19 MiB | 12,282.19 MiB |

Peak reserved use was 52.94% and 49.09% of the reported capacities. G=8 does
not retain eight replay graphs simultaneously; trajectories replay
sequentially and gradients accumulate, so the measured memory remains far
below 24 GiB.

### Checkpoint reload

`two_gpu_g8_caseb_step_001.pt` was loaded into a freshly constructed sharded
model and AdamW optimizer:

- all 510 trainable tensors loaded and matched exactly (`max_abs_error=0`);
- all 504 LoRA and 6 projector tensors present;
- optimizer has exactly 510 unique parameters;
- optimizer state matched the checkpoint exactly and was finite;
- optimizer step/sample cursor restored as `1/1`;
- runtime and KL-free contracts matched.

Checkpoint SHA-256:
`823d1c93ec400fd6751020605b8226d8187be4655338b86bfbf67bbbd3980d73`.

The first reload validator invocation produced a false failure because its
KL-free path demanded a 510-tensor reference-policy snapshot. KL is disabled,
so the correct reference count is zero. The assertion was corrected and the
fresh reload passed; full trainable and optimizer exact-equality checks were
added at the same time. The superseding artifact is
`step001_reload_validation_v2/two_gpu_g8_caseb_summary.json`.

## G=4 versus G=8 accumulation semantics

The runner obtains one standardized advantage vector using
`(reward - group_mean) / population_std`. Each trajectory is scored with the
same scalar clipped-GRPO function, and performs:

```text
(loss_i / GROUP_SIZE).backward()
```

Therefore the accumulated gradient is exactly the gradient of
`mean_i loss_i`. CPU tests compared sequential accumulation with the vector
mean for both G=4 and G=8 and passed at `1e-7` tolerance. All group summary
means also divide by `GROUP_SIZE`; there is no fixed `/4`, microbatch multiplier,
gradient-accumulation multiplier, or hidden batch normalization. G changes the
reward sample set and its within-group standardization, but not the objective's
mean-reduction semantics or effective learning-rate multiplier.

## Saved G=4 rollout diversity

The analysis covers all 578 saved groups (2,312 trajectories).

### Bbox diversity

- Valid boxes: 2,245/2,312 (`97.1021%`).
- Exact duplicate bbox rate among valid-valid pairs: `25.8888%`.
- Exact duplicate state rate including invalid `None`: `24.9135%`.
- Mean pairwise per-coordinate L1 distance: `0.0152274` in normalized 0–1
  coordinates; median only `0.0005`.
- Mean pairwise bbox IoU: `0.9444033`.
- Mean within-group coordinate population standard deviations:
  x1 `0.01796`, y1 `0.01475`, x2 `0.01352`, y2 `0.01385`.

The trajectories are therefore usually near-duplicates spatially, even when
their boxes are not exactly identical.

### Token-sequence diversity

Production logs retained exact generated-token SHA-256 checksums rather than
raw token sequences, so exact sequence equality is measurable but edit
distance is not.

- Exact duplicate token-sequence pair rate: `20.8189%`.
- Mean unique sequences per four-trajectory group: `3.1972`.
- Four unique sequences: 317/578 groups.
- At least one exact sequence duplicate: 261/578 groups.

### Best raw IoU per group

- Mean: `0.0824653`; median: `0.0476218`; maximum: `0.5577028`.
- 90th percentile: `0.2035667`; 95th percentile: `0.2526233`.
- Closest non-winning group was still `0.0443476` below the 0.5 threshold.
- Median non-winning distance below threshold: `0.4524029`.

| Best-IoU bin | Groups | Fraction |
|---|---:|---:|
| 0–.1 | 397 | 68.6851% |
| .1–.25 | 150 | 25.9516% |
| .25–.4 | 23 | 3.9792% |
| .4–.5 | 7 | 1.2111% |
| >.5 | 1 | 0.1730% |

Only 4/2,312 trajectories exceeded 0.5, and all four were in the same group.
Thus saved G=4 contained 577 all-failure groups, one all-success group, and no
mixed binary-spatial group.

### Will G=8 materially increase mixed groups?

Probably not under the observed distribution.

An optimistic pooled-i.i.d. calculation from the trajectory success rate
predicts mixed-group probability rising from `0.690%` at G=4 to `1.376%` at
G=8—about `2.06` mixed groups in 150 attempts. This assumption conflicts with
the observed perfect clustering. A heterogeneity-aware two-block estimate is
`0.345%`, only `0.52` mixed groups per 150 attempts. The very high pairwise box
IoU and only seven near-threshold losing groups reinforce the lower estimate.

G=8 doubles sampling opportunities, but it is unlikely to turn the binary
spatial component into a reliable training signal without a change in the
underlying rollout distribution. The probe nevertheless had nonzero total
reward diversity from the unchanged format/semantic components and produced
valid Case-B gradients.

## Validation and stopping rule

Prepared internal-validation output reports:

- mean and median IoU;
- strict IoU > 0.5 rate;
- valid native box rate;
- near-full-image rate overall and among valid boxes, defined as normalized
  width and height both at least 0.9;
- mean normalized box area among valid boxes;
- mean MedCLIP semantic score;
- PBD/NTP-fallback/none branch counts and rates.

At step 100, compare validation mean IoU with the frozen-base result produced
at step 25. If step 100 is not strictly higher, the prepared checker sets
`flag_unlikely_to_benefit_from_continuing_to_500=true`. It never stops or
continues a run automatically.

Exact commands and expected output locations are in `G8_CASEB_COMMANDS.md`.

## Files changed

- `rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml` — locked G=8 Case-B
  overlay.
- `two_gpu_g8_caseb_production.py` — explicitly gated production adapter.
- `test_g8_caseb_experiment_contract.py` — G=4/G=8 objective and static
  contract tests.
- `analyze_g4_rollout_diversity.py` — saved-output-only diversity analysis.
- `eval_two_gpu_hybrid_grpo_checkpoints.py` — added required localization
  validation metrics and overlay-config loading.
- `assess_g8_validation_stopping_rule.py` — report-only step-100 rule.
- `two_gpu_g4_grpo_multistep_smoke.py` — fixed KL-free reload reference-count
  assertion and added exact full trainable/optimizer reload verification.
- This report, `G8_CASEB_COMMANDS.md`, `g8_caseb_preparation.json`, saved
  diversity JSON, probe metrics/checkpoint/summary, and reload summaries.

## CPU/static checks

- 4/4 G=8 Case-B contract/objective tests passed.
- Python compilation passed for all new/modified scripts.
- YAML merge and the original production config validator passed with G=8.
- Git whitespace checks passed.
- No held-out prediction evaluation was performed.
- The full 0→150 run was not started.
