# Semantic reward diagnosis: does MedCLIP favor generic ROIs?

> **Scope and provenance.** This is a read-only analysis of saved evaluation JSONL and training metrics. No model was run, trained, or reevaluated. The 194-case evaluation is the previously observed comparative follow-up set, so p-values below are descriptive rather than confirmatory.

## Executive finding

**The direct size-incentive hypothesis is not supported, but the semantic reward is permissive of—and poorly aligned against—generic ROIs.** Semantic reward does not reliably rise with area: KL-free evaluation correlations are essentially zero, MedGround-KL correlations are negative or null, and saved within-image/query G=4 associations are negative or near chance. In 116 matched cases, expanding the base ROI to a KL-free near-full box changes semantic reward by only +0.00087 (95% CI includes zero) while IoU falls by 0.0596 (CI entirely below zero). Nevertheless, near-full and low-IoU boxes retain positive semantic scores, and semantic reward is negatively associated with IoU in both trained policies. Thus MedCLIP does not demonstrably drive expansion; it fails to penalize or distinguish the generic-box shortcut once it appears.

## Methods

For every valid evaluation prediction, width=`(x2−x1)/1000`, height=`(y2−y1)/1000`, area=`width×height`, and center=`((x1+x2)/2000,(y1+y2)/2000)`. Near-full means width and height are each ≥0.90. Area bins are left-closed/right-open, except `.9–1.0` includes 1.0. IoU strata are `<.1`, `.1–.5` inclusive, and `>.5`. Pearson measures linear association; Spearman measures monotone association. No multiplicity correction was applied. All per-box derived records are in the JSON.

## 1. Semantic reward correlations among valid evaluation boxes

| Condition | n | Pair | Pearson r (p) | Spearman ρ (p) |
|---|---:|---|---:|---:|
| Frozen base | 121 | semantic vs IoU | -0.1061 (0.2466) | -0.0860 (0.3483) |
| Frozen base | 121 | semantic vs area | -0.0747 (0.4153) | -0.0379 (0.6797) |
| Frozen base | 121 | semantic vs width | -0.1357 (0.1377) | -0.0656 (0.4750) |
| Frozen base | 121 | semantic vs height | -0.0088 (0.9240) | 0.0602 (0.5119) |
| KL-free step500 | 194 | semantic vs IoU | -0.1860 (0.0094) | -0.2242 (0.0017) |
| KL-free step500 | 194 | semantic vs area | 0.0218 (0.7632) | 0.0134 (0.8531) |
| KL-free step500 | 194 | semantic vs width | 0.0036 (0.9601) | -0.0682 (0.3446) |
| KL-free step500 | 194 | semantic vs height | 0.0480 (0.5064) | 0.0146 (0.8397) |
| MedGround-KL step500 | 177 | semantic vs IoU | -0.1226 (0.1041) | -0.2826 (1.38e-04) |
| MedGround-KL step500 | 177 | semantic vs area | -0.1596 (0.0338) | -0.0142 (0.8516) |
| MedGround-KL step500 | 177 | semantic vs width | -0.1859 (0.0132) | -0.0661 (0.3823) |
| MedGround-KL step500 | 177 | semantic vs height | -0.1900 (0.0113) | -0.0849 (0.2611) |

A positive area/width/height coefficient would mean larger boxes receive higher semantic reward. That pattern is absent: KL-free coefficients are near zero, and the KL run is negative by Pearson with near-zero rank association. Conversely, semantic–IoU association is negative for both trained policies (KL-free Pearson `r=-0.186`, Spearman `ρ=-0.224`; KL Spearman `ρ=-0.283`), the opposite of a useful grounding proxy. Disease-adjusted correlations are included in the JSON; their sign instability reinforces that no robust monotone size incentive is identified.

## 2. Semantic reward by normalized box-area bin

Cells are `n; mean semantic (median); mean IoU`. Empty bins are shown as —.

| Area bin | Base | KL-free | MedGround-KL |
|---|---:|---:|---:|
| 0-.1 | 16; 0.0219 (0.0206); 0.0653 | — | 13; 0.0316 (0.0301); 0.0381 |
| .1-.25 | 22; 0.0130 (0.0128); 0.2114 | — | 5; 0.0219 (0.0122); 0.0756 |
| .25-.5 | 32; 0.0160 (0.0151); 0.1531 | 1; 0.0190 (0.0190); 0.0000 | 7; 0.0128 (0.0132); 0.2137 |
| .5-.75 | 22; 0.0132 (0.0108); 0.1261 | — | 17; 0.0149 (0.0135); 0.1110 |
| .75-.9 | 16; 0.0139 (0.0148); 0.1255 | 2; 0.0163 (0.0163); 0.0863 | 28; 0.0141 (0.0114); 0.0906 |
| .9-1.0 | 13; 0.0179 (0.0144); 0.1149 | 191; 0.0178 (0.0145); 0.0730 | 107; 0.0186 (0.0145); 0.0410 |

## 3. Semantic reward by localization and near-full status

Cells are `n; mean semantic (median)`.

| Stratum | Base | KL-free | MedGround-KL |
|---|---:|---:|---:|
| IoU > .5 | 7; 0.0146 (0.0144) | — | 2; 0.0134 (0.0134) |
| IoU .1–.5 | 46; 0.0133 (0.0137) | 55; 0.0150 (0.0133) | 29; 0.0155 (0.0126) |
| IoU < .1 | 68; 0.0174 (0.0143) | 139; 0.0188 (0.0148) | 146; 0.0190 (0.0148) |
| Near-full | 14; 0.0180 (0.0151) | 192; 0.0178 (0.0146) | 113; 0.0189 (0.0147) |
| Not near-full | 107; 0.0154 (0.0138) | 2; 0.0132 (0.0132) | 64; 0.0175 (0.0134) |

Near-full minus non-near-full mean semantic differences are: base 0.0026, KL-free 0.0046, MedGround-KL 0.0014. The KL-free comparison has only two non-near-full boxes and is not an informative size-effect estimate. More importantly, low-IoU boxes have the highest mean semantic score in all three conditions, showing failure to reward localization quality. These contrasts are conditional on each model's generated-box distribution and are not randomized effects.

## 4. Matched KL-free near-full ROI versus smaller base ROI

The matched subset requires both predictions valid, KL-free near-full, and base area strictly smaller. This holds image, query, disease, and case constant. Paired CIs use 20,000 bootstrap resamples.

| Quantity | Base mean | KL-free mean | KL-free − base | Paired 95% CI | Higher / lower / tie |
|---|---:|---:|---:|---:|---:|
| Area | 0.4544 | 0.9897 | +0.5354 | [+0.4817, +0.5894] | 116 / 0 / 0 |
| Semantic reward | 0.0160 | 0.0168 | +0.0009 | [-0.0013, +0.0030] | 61 / 55 / 0 |
| IoU | 0.1406 | 0.0810 | -0.0596 | [-0.0834, -0.0373] | 26 / 90 / 0 |

There are 116 matched cases. In 61 cases the larger KL-free crop has higher semantic reward and in 55 it is lower—essentially balanced—while the paired mean semantic CI includes zero. By contrast, IoU is lower in 90 cases and its mean CI excludes zero; in 50 cases semantic rises while IoU falls. The matched evidence therefore rejects a reliable positive size effect but shows that a large localization loss is not accompanied by a semantic penalty.

## 5. Saved training-rollout corroboration

Within each saved G=4 group, candidates share image, query, disease, and step. Thus size–semantic comparisons within a group directly describe the local reward incentive available to GRPO. Only saved valid trajectories are used.

| Run | Valid trajectories | Varying-area groups | Overall Pearson / Spearman semantic–area | Within-group centered Pearson | Larger-area candidate has higher semantic (pair rate) | Max-area is max-semantic |
|---|---:|---:|---:|---:|---:|---:|
| KL-free | 2245 | 493 | -0.0708 / -0.0010 | -0.1157 | 52.3% | 164 / 493 (33.3%) |
| MedGround-KL | 1801 | 481 | -0.1031 / -0.1099 | -0.1390 | 49.0% | 122 / 481 (25.4%) |

## 6. Diagnosis

### Evidence

- **No robust marginal size incentive:** KL-free semantic–area Pearson/Spearman are `0.022/0.013`; KL values are `-0.160/-0.014`; matched semantic delta is +0.00087 with a CI spanning zero.
- **No within-image/query corroboration of expansion pressure:** centered semantic–area correlations in saved G=4 groups are `-0.116` (KL-free) and `-0.139` (KL); informative pair directions are 52.3% and 49.0% larger-area-higher-semantic.
- **Strong permissiveness:** the KL-free policy has 192/194 near-full boxes and zero IoU>.5 successes, yet their mean semantic reward is 0.0178. Matched near-full crops lose 0.0596 IoU without a detectable semantic loss.
- **Anti-grounding association:** low-IoU boxes have the highest mean semantic score in all three conditions, and semantic–IoU association is negative in both trained policies.
- MedGround-KL reduces near-full boxes, but it does not turn semantic score into a localization proxy.

### Mechanistic interpretation

The scorer crops the ROI, pads it to a square, resizes to 224×224, and compares its MedCLIP image embedding with the original disease query. A full-image ROI preserves global chest context and is unlikely to crop out the abnormality; a tight but imperfect ROI can exclude disease evidence or be distorted by padding/resizing. The observed result is **semantic invariance to a large loss of spatial specificity**, not a consistent increase with area. With binary spatial reward zero for almost all sub-threshold boxes, an approximately flat semantic score provides no counterforce against a generic-box shortcut; its negative IoU association may even rank some worse-localized candidates higher for image/text reasons unrelated to box correctness. This diagnoses this setup, not an intrinsic limitation of MedCLIP in every ROI design.

### Intended MedGround-R1 interpretation

A semantic ROI reward is useful when it distinguishes semantically informative spatial alternatives—especially negatives that are spatially different—while spatial reward anchors the solution. It should not replace grounding by making the whole image an equally rewarded semantic answer. Here it fails that intended role: large low-IoU crops retain positive scores, matched degradation in IoU produces no detectable semantic penalty, and semantic score is negatively associated with IoU. Precision is important: the repository's prior source audit records that the semantic function exists in MedGround-R1, but the audited original default launch selected accuracy/format rewards rather than semantic reward. Therefore this diagnosis applies to the current Hybrid configuration's MedGround-style semantic component, not to the original default MedGround-R1 run as an empirical claim.

## Bottom line

The saved evidence does **not** show that increasing ROI size reliably increases MedCLIP reward. It does show that generic near-full boxes remain positively rewarded and are not penalized when localization degrades, while semantic score can be anti-correlated with IoU. The correct diagnosis is **permissive/anti-grounding semantic reward, not a proven size-seeking reward gradient**. KL mitigates geometric collapse but does not make semantic reward a substitute for spatial grounding.
