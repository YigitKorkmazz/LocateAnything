# G8 Case-B internal validation report

## Scope

This report uses only the pinned 80-example internal-validation manifest. The 194-case held-out test was not evaluated or used for selection. These are model-selection results, not final test performance.

## Per-condition metrics

| Condition | Valid | Malformed/no-box | Mean IoU | Median IoU | IoU > .5 | Semantic | Total | Area (valid) | Near-full (all) | PBD/NTP/none | Tokens | Trunc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_base | 67.50% | 32.50% | 0.103986 | 0.039857 | 2.50% | 0.010396 | 0.710396 | 0.546294 | 11.25% | 72.50%/1.25%/26.25% | 19.82 | 0 |
| step_025 | 87.50% | 12.50% | 0.109672 | 0.065607 | 1.25% | 0.011919 | 0.899419 | 0.617436 | 16.25% | 88.75%/0.00%/11.25% | 18.51 | 0 |
| step_050 | 90.00% | 10.00% | 0.104062 | 0.059468 | 2.50% | 0.012898 | 0.937898 | 0.660531 | 31.25% | 91.25%/1.25%/7.50% | 17.56 | 0 |
| step_075 | 96.25% | 3.75% | 0.128912 | 0.066781 | 6.25% | 0.013275 | 1.038275 | 0.666780 | 33.75% | 95.00%/1.25%/3.75% | 17.38 | 0 |
| step_100 | 96.25% | 3.75% | 0.115347 | 0.061813 | 2.50% | 0.013112 | 1.000612 | 0.788398 | 56.25% | 95.00%/1.25%/3.75% | 17.64 | 0 |
| step_125 | 97.50% | 2.50% | 0.157196 | 0.086156 | 7.50% | 0.013814 | 1.063814 | 0.524473 | 16.25% | 100.00%/0.00%/0.00% | 17.46 | 0 |
| step_150 | 98.75% | 1.25% | 0.156744 | 0.078554 | 7.50% | 0.013647 | 1.076147 | 0.551092 | 12.50% | 100.00%/0.00%/0.00% | 17.51 | 0 |

Mean normalized area is among valid native boxes. Near-full paired/model-selection rate is over all 80 samples; near-full means width and height are both at least 0.9.

## Paired deltas versus frozen base

| Checkpoint | Mean IoU delta (95% CI) | Median IoU delta | IoU>.5 delta | Valid delta | Near-full delta |
|---|---:|---:|---:|---:|---:|
| step_025 | +0.005686 [-0.008149, +0.021207] | +0.025751 | -1.25% | +20.00% | +5.00% |
| step_050 | +0.000076 [-0.019765, +0.018065] | +0.019612 | +0.00% | +22.50% | +20.00% |
| step_075 | +0.024926 [+0.001760, +0.046154] | +0.026925 | +3.75% | +28.75% | +22.50% |
| step_100 | +0.011361 [-0.010063, +0.031309] | +0.021956 | +0.00% | +28.75% | +45.00% |
| step_125 | +0.053210 [+0.024851, +0.081029] | +0.046299 | +5.00% | +30.00% | +5.00% |
| step_150 | +0.052758 [+0.024308, +0.080881] | +0.038698 | +5.00% | +31.25% | +1.25% |

## Selection

Selected: **step_150** (`/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_150.pt`).

Rule: highest mean IoU; conditions within 0.005 are effectively tied, then higher strict IoU>.5, lower near-full rate, and non-collapsed valid-box rate.

## Step-100 stopping rule

Frozen-base mean IoU: `0.103986`; step-100: `0.115347`; delta: `+0.011361`.

Continuing beyond 150 justified by the existing step-100 rule: **yes**. No automatic action was taken.

## Verdict

**localization improved**

## Reproducibility

The initial combined command completed frozen base and steps 25/50/75, then encountered cross-condition CUDA allocator accumulation while starting step 100. Steps 100/125/150 were rerun successfully using the following fresh-process commands.

Evaluation commands used:

```bash
tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/all_checkpoints_internal_validation --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition frozen_base=base --condition step_025=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_025.pt --condition step_050=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_050.pt --condition step_075=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_075.pt --condition step_100=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_100.pt --condition step_125=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_125.pt --condition step_150=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_150.pt --bootstrap-replicates 2000 --bootstrap-seed 20260810'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_100_isolated --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_100=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_100.pt --bootstrap-replicates 2000 --bootstrap-seed 20260810'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_125_isolated --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_125=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_125.pt --bootstrap-replicates 2000 --bootstrap-seed 20260810'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_150_isolated --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_150=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_150.pt --bootstrap-replicates 2000 --bootstrap-seed 20260810'
```

Assembly command:

```bash
/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/assemble_g8_caseb_validation_conditions.py --partial-output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/all_checkpoints_internal_validation --step100-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_100_isolated/aggregate_metrics.json --step125-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_125_isolated/aggregate_metrics.json --step150-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_150_isolated/aggregate_metrics.json --bootstrap-replicates 2000 --bootstrap-seed 20260810
```

Summarizer command:

```bash
/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/summarize_g8_caseb_validation.py --evaluation-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/all_checkpoints_internal_validation/aggregate_metrics.json --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/all_checkpoints_internal_validation
```
