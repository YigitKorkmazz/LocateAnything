# ChestX-ray8 localization correctness audit

## Binary verdict

**localization pipeline correct**

The unexpectedly low IoU is not explained by an x/y swap, xywh/xyxy error,
width/height swap, image-size mismatch, resize/padding offset, normalized-edge
convention, manifest misalignment, or a train/evaluator reward-path mismatch.

No training, model inference, or new held-out prediction evaluation was run.
The held-out manifest was inspected only as immutable metadata and existing
saved held-out metrics were read for context.

## Coordinate contract

The NIH `BBox_List_2017.csv` columns are positional because the header contains
unquoted commas: image, finding label, x, y, width, height. The loader converts
each source box to `[x, y, x + width, y + height]`. The documented NIH label
alias `Infiltrate` is canonicalized to `Infiltration` before grouping.

The pair builder opens the original image with Pillow and records its actual
`(width, height)`. It then computes:

```text
x_norm = round(clamp(x_pixel / image_width  * 1000, 0, 1000))
y_norm = round(clamp(y_pixel / image_height * 1000, 0, 1000))
```

The coordinate order is therefore `x1, y1, x2, y2`, with x scaled by width and
y by height. The evaluator's inverse is exactly `x/1000*width` and
`y/1000*height`.

- `1000` is the continuous right/bottom image edge; `[0,0,1000,1000]` is the
  full image.
- `999` is one normalized unit inside that edge, not an alternate full-image
  sentinel. Against the full box, `[0,0,999,999]` has IoU `0.998001`.
- The IoU implementation uses continuous `xyxy` extents; there is no inclusive
  `+1` pixel term.
- On an asymmetric `1600 x 900` oracle, pixel box
  `[160,180,1200,720]` maps to `[100,200,750,800]` and back exactly. This
  directly checks axis order and rejects a width/height swap.

### Resize, aspect ratio, and padding

The LocateAnything image processor first uses a uniform scale only if required
by its token limit, preserving aspect ratio. It then resizes width and height
to patch-grid multiples. It does not paste into a padded canvas, so the
coordinate-origin padding offsets are `(0,0)`.

GT and predictions are compared in normalized space. Independent positive x/y
scaling leaves IoU unchanged, which the rectangular-image oracle also tests.
Consequently the processor's tensor resize does not require a GT offset or
pixel remap. Image dimensions in the manifest come from the original Pillow
image, not the resized tensor.

## Oracle tests

All 7 CPU-only tests passed:

1. Exact transformed GT as the committed prediction gives IoU `1.0`.
2. The production spatial reward for that prediction is `1.0` under strict
   `IoU > 0.5`.
3. Pixel → normalized → pixel roundtrips are bounded by half of one normalized
   unit per axis.
4. An asymmetric image proves x/y and width/height ordering.
5. The `999/1000` boundary contract is exact.
6. IoU is invariant under independent x/y resize scaling.
7. Trainer and evaluator import the same production reward factory, semantic
   scorer, and IoU function objects.

Across all 790 train80 records, the largest roundtrip errors were
`0.5119931 px` in x and `0.5092910 px` in y, both within the expected
`1024/2000 = 0.512 px` nearest-token rounding bound.

Test source: `test_localization_correctness_audit.py`

## Train/evaluator reward-path identity

Both paths construct `rl.rewards.build_reward_pipeline_from_config` and call
the resulting `rewards.score_from_trace(trace, pair)`. That method reads the
same persisted `pair["gt_boxes_norm_1000"]` and calls
`spatial_reward_from_box`, which imports `box_iou` directly from
`eval_locateanything_bbox`.

The committed decoder box—not a box re-parsed from arbitrary malformed text—is
used in both training and held-out evaluation. The spatial decision is strict
`best_iou > 0.5` in both paths.

## Manifest and source alignment

Every record was independently matched to the raw source CSV by canonical
`(image_index, disease)`, and every source xywh box was reconverted to xyxy and
normalized coordinates.

| Manifest | Rows | Failures | SHA-256 status |
|---|---:|---:|---|
| train80 | 790 | 0 | matches |
| internal optimization | 710 | 0 | matches |
| internal validation | 80 | 0 | matches |
| held-out metadata only | 194 | 0 | matches |

Additional alignment checks passed:

- optimization ∪ validation equals train80;
- optimization ∩ validation is empty;
- train80 ∩ held-out image/disease keys is empty;
- recorded optimization/validation/test patient overlaps are all false;
- image filename, patient ID, disease, prompt phrase, query, GT count, Pillow
  width/height, source bbox, and normalized bbox agree for every checked row.

## First 50 completed KL-free groups

The CSV contains exactly 200 trajectory rows: 50 groups × 4 trajectories,
including invalid geometry as raw IoU `0`, before binary thresholding.

| Statistic | Value |
|---|---:|
| Mean raw IoU | 0.07131261 |
| Minimum | 0.00000000 |
| 25th percentile | 0.00000000 |
| Median | 0.01959521 |
| 75th percentile | 0.12178366 |
| Maximum | 0.33307382 |
| Trajectories with IoU > 0.5 | 0 / 200 |
| Groups with nonzero raw-IoU variance | 48 / 50 |
| Groups with nonzero binary-spatial variance | 0 / 50 |
| All-zero binary-spatial groups | 50 / 50 |

This shows that spatial variance was erased by thresholding, not by a broken
IoU computation: 48 of the first 50 groups had different raw IoUs, but all 200
values remained below `0.5`, so all four binary rewards were zero in each
group.

Across all 578 completed KL-free groups, 496 had nonzero raw-IoU variance, but
none had within-group binary-spatial variance. There were 577 all-zero groups
and one all-one group (4 successful trajectories). Binary group-relative
variance requires a *mixed* group for one image; marginal successes clustered
as four successes on the same image produce no within-group signal.

The already saved held-out base result is not contradictory. It had nonzero
`IoU > 0.5` accuracy (`4/194 = 0.02061856`) on different images under a
different saved evaluation schedule. Nonzero marginal accuracy does not imply
that four trajectories for the same training example will straddle the binary
threshold. No held-out inference was repeated for this audit.

Raw rows: `first50_klfree_raw_ious.csv`

## Diagnostic overlays

Twenty overlays were rendered from train80/internal-validation data only:

- 10 internal optimization examples;
- 10 internal validation examples;
- 0 held-out-test examples;
- transformed GT is red;
- saved fresh-base prediction is green for the 8 examples where an existing
  saved output was available;
- the remaining 12 are explicitly marked as having no saved base prediction.

Manual inspection of both predicted and GT-only overlays confirmed that boxes
land in the expected image coordinate frame. No inference was run to fill
missing base predictions.

Overlay index: `overlay_manifest.json`; images: `overlays/`.

## Prepared G=8 RTX3090 one-update memory probe

Prepared but **not run**:

- two visible RTX 3090 GPUs, existing 18/18 decoder shard;
- G=8 and exactly one optimizer update;
- format/spatial/semantic weights remain `1/1/1`;
- native LocateAnything parser and decoder-committed box unchanged;
- spatial remains strict binary `IoU > 0.5`;
- frozen MedCLIP ROI/text semantic reward unchanged;
- projector frozen; only 504 LoRA tensors synchronized/trainable;
- LoRA learning rate `1e-5`;
- KL initially disabled (`L_GRPO`, beta effectively zero);
- configuration retains checkpoint interval 25 for eventual validation
  studies; the isolated one-update probe would write a diagnostic step-1
  checkpoint only;
- immutable 710-example optimization and 80-example internal-validation
  manifests are pinned by SHA-256.

The launcher refuses to run unless the operator supplies the explicit
`--i-understand-this-runs-one-g8-update` acknowledgement. Its merged contract
passed CPU-only static validation. No launch command was executed.

Prepared spec: `rl/chestxray8_grpo_hybrid_native_g8_lora_only_probe.yaml`

Prepared launcher: `two_gpu_g8_one_update_memory_probe.py`

## Conclusion

The low localization score is a model/reward-signal behavior under the current
strict binary threshold, not evidence of a coordinate, preprocessing,
manifest, or reward-routing failure. In particular, the saved raw IoUs vary
within most groups while the thresholded spatial reward does not. That is the
mechanistic explanation for `0/578` spatial within-group variance.

Machine-readable result: `localization_correctness_audit.json`.
