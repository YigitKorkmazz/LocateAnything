# Joint NTP+MTP fine-tuning (ChestX-ray8)

Branch: `joint-ntp-mtp`

## Modes

| `--training-objective` | Behavior |
|---|---|
| `ntp_only` (**default**) | Unchanged AR SFT: assistant-only CE, forced causal attention |
| `joint_ntp_mtp` | Paper dual stream `x_vis+x_q+x_ntp+x_blk`, `L = λ_ntp L_ntp + λ_mtp L_mtp`, native Figure-4 block mask |

Evaluation is always standard AR generation (unchanged).

## Where things live

| Concern | Location |
|---|---|
| `x_blk` construction | `evaluation/chestxray8/joint_ntp_mtp.py` → `pack_joint_ntp_mtp` / `_pack_detection_blocks` |
| Figure-4 attention enabled | `train_chestxray8_sft.py` skips `patch_qwen_force_causal_attn_for_sft` in joint mode; LM uses native `create_block_diff_mask_by_pe_4d` |
| `loss_ntp` / `loss_mtp` | `joint_ntp_mtp.compute_joint_losses` |
| Combined loss | `forward_loss_joint` in `train_chestxray8_sft.py` |
| Leakage / packing asserts | `assert_packing_integrity`, `assert_attention_mask_invariants` |

## Commands (tcsh-friendly)

Shared env:

```tcsh
cd /auto/k2/ykorkmaz/LocateAnything-playground/Embodied
setenv CUDA_VISIBLE_DEVICES 1
setenv HF_HOME /auto/data2/ykorkmaz/.cache/huggingface
set PY=/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python
set REV=c32291ca5e996f5a7a485845b4f57a233936bba0
```

### Unit tests (no GPU required for packing/mask/loss)

```tcsh
$PY evaluation/chestxray8/test_joint_ntp_mtp.py
```

### Debug one-step (joint) — gradients + packing dump + attn visualization

```tcsh
$PY evaluation/chestxray8/train_chestxray8_sft.py \
  --experiment-type lora_projector \
  --training-objective joint_ntp_mtp \
  --lambda-ntp 1.0 \
  --lambda-mtp 1.0 \
  --mtp-block-size 6 \
  --attn-mask-check \
  --debug-one-step \
  --output-dir results/finetuning/joint_ntp_mtp_debug_one_step \
  --model-path nvidia/LocateAnything-3B \
  --model-revision $REV \
  --seed 42 \
  --learning-rate 2e-5 \
  --projector-learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --bf16 --grad-checkpoint \
  --max-train-samples 1
```

Inspect:

- `debug_one_step_report.json` → `loss_ntp`, `loss_mtp`, `loss_total`, `grad_group_norms`
- `joint_packing_dump.txt`
- `attn_mask_visualization.png`
- `attn_mask_check.json`

### NTP-only training (unchanged)

```tcsh
$PY evaluation/chestxray8/train_chestxray8_sft.py \
  --experiment-type lora_projector \
  --training-objective ntp_only \
  --output-dir results/finetuning/lora_projector_ntp_only_v1 \
  --model-path nvidia/LocateAnything-3B \
  --model-revision $REV \
  --seed 42 \
  --learning-rate 2e-5 \
  --projector-learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --bf16 --grad-checkpoint \
  --num-epochs 5 \
  --save-steps 25 \
  --logging-steps 1
```

### Joint NTP+MTP training

```tcsh
$PY evaluation/chestxray8/train_chestxray8_sft.py \
  --experiment-type lora_projector \
  --training-objective joint_ntp_mtp \
  --lambda-ntp 1.0 \
  --lambda-mtp 1.0 \
  --mtp-block-size 6 \
  --attn-mask-check \
  --output-dir results/finetuning/lora_projector_joint_ntp_mtp_v1 \
  --model-path nvidia/LocateAnything-3B \
  --model-revision $REV \
  --seed 42 \
  --learning-rate 2e-5 \
  --projector-learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --bf16 --grad-checkpoint \
  --num-epochs 5 \
  --save-steps 25 \
  --logging-steps 1
```

### 4-example overfit (joint, force full 80 steps)

```tcsh
$PY evaluation/chestxray8/train_chestxray8_sft.py \
  --experiment-type lora_projector \
  --training-objective joint_ntp_mtp \
  --lambda-ntp 1.0 --lambda-mtp 1.0 --mtp-block-size 6 \
  --overfit-samples 4 \
  --max-steps 80 \
  --overfit-eval-every 5 \
  --disable-overfit-early-stop \
  --output-dir results/finetuning/joint_ntp_mtp_overfit4 \
  --model-path nvidia/LocateAnything-3B \
  --model-revision $REV \
  --seed 42 \
  --learning-rate 2e-5 \
  --projector-learning-rate 1e-5 \
  --max-grad-norm 1.0 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --lora-rank 8 --lora-alpha 16 --lora-dropout 0.05 \
  --bf16 --grad-checkpoint \
  --save-steps 25 --logging-steps 1
```

Metrics are appended to `<output_dir>/overfit_metrics.jsonl` each eval step.

## Remaining differences vs the paper / official trainer

1. **Loss aggregation:** Official Embodied trainer uses one token-mean CE over NTP∪MTP labels. We use the paper form `λ_ntp·mean_CE(NTP) + λ_mtp·mean_CE(MTP)` (equal stream emphasis regardless of token counts).
2. **`pos_loss_list`:** Present in Qwen2 but unused for the training objective (logging-only in the HF code). We compute MTP CE explicitly over `stream_ids==MTP`.
3. **Packing:** Single-sample (no Magi stream packing / `sub_sample_lengths`). Fine for batch size 1.
4. **“Previous MTP blocks”:** Official `create_block_diff_mask_by_pe_4d` does **not** let MTP blocks attend other MTP blocks. Prior context is exposed via **prefix into `x0`** using PE at the block anchor (committed NTP tokens). This matches the HF implementation, not a literal cross-block MTP visibility graph.
5. **Generic (non-box) MTP branch:** Not used; ChestX-ray8 always has `</box>` / `</ref>`.
6. **Inference:** Unchanged AR decoding — intentional, so train-objective is the only variable.
