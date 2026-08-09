# Exact G=8 Case-B commands

Run from `/auto/k2/ykorkmaz/LocateAnything-playground`.

## One-update probe (completed)

```bash
tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/two_gpu_g8_caseb_production.py --execute-authorized-g8-run --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/experiments/g8_caseb_localization_seed42_preparation_20260809/one_update_probe --max-optimizer-steps 1 --max-attempted-groups 25 --diagnostic-checkpoint-steps 1 --require-fresh-start --seed 42'
```

## Fresh 0→150 run (prepared; not run)

The destination must not already exist.

```bash
tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/two_gpu_g8_caseb_production.py --execute-authorized-g8-run --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42 --max-optimizer-steps 150 --max-attempted-groups 1200 --checkpoint-interval 25 --require-fresh-start --seed 42'
```

Expected checkpoints:

```text
two_gpu_g8_caseb_step_025.pt
two_gpu_g8_caseb_step_050.pt
two_gpu_g8_caseb_step_075.pt
two_gpu_g8_caseb_step_100.pt
two_gpu_g8_caseb_step_125.pt
two_gpu_g8_caseb_step_150.pt
```

## Internal validation

The only evaluation manifest below is the pinned 80-example internal
validation set. The 194-case held-out test is not referenced.

Define these paths for readability:

```bash
TRAIN_OUT=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42
VAL_ROOT=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42
CONFIG=Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml
VAL_MANIFEST=Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl
VAL_SHA=f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838
PY=/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python
```

Step 25, including the frozen-base baseline:

```bash
tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_025 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition frozen_base=base --condition step_025=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_025.pt'
```

Steps 50, 75, 100, 125, and 150 use the same command shape:

```bash
tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_050 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_050=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_050.pt'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_075 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_075=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_075.pt'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_100 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_100=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_100.pt'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_125 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_125=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_125.pt'

tcsh -f -c 'setenv CUDA_VISIBLE_DEVICES 0,1; setenv PYTHONUNBUFFERED 1; /auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/eval_two_gpu_hybrid_grpo_checkpoints.py --config Embodied/evaluation/chestxray8/rl/chestxray8_grpo_hybrid_native_g8_caseb_150.yaml --output-dir Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_150 --manifest Embodied/evaluation/chestxray8/splits/production_train90_validation10_seed42/validation10_of_train80_seed42.jsonl --manifest-sha256 f22343be5c53aa61ef31dbd018070148f7c2d53580af0d176025f57a4d407838 --split-label internal_validation --condition step_150=Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/training/two_gpu_18x18_g8_caseb_150_optimization710_seed42/two_gpu_g8_caseb_step_150.pt'
```

Apply the step-100 stopping flag without automatically stopping or continuing:

```bash
/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python Embodied/evaluation/chestxray8/assess_g8_validation_stopping_rule.py --base-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_025/aggregate_metrics.json --step100-aggregate Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step_100/aggregate_metrics.json --output Embodied/evaluation/chestxray8/results/chestxray8_hybrid_grpo_native/validation/g8_caseb_150_optimization710_seed42/step100_stopping_rule.json
```
