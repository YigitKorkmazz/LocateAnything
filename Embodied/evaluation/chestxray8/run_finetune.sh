#!/usr/bin/env bash
# ChestX-ray8 LocateAnything SFT helper commands.
# Prefer free GPUs among {1,3} by default (same policy as run_eval.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${LOCATEANYTHING_PYTHON:-/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/auto/k2/ykorkmaz/.cache/huggingface}"
export PYTHONUNBUFFERED=1

pick_free_gpu() {
  local preferred=("1" "3")
  local idx mem util
  while IFS=',' read -r idx mem util; do
    idx="$(echo "$idx" | tr -d ' ')"
    mem="$(echo "$mem" | tr -d ' MiB')"
    util="$(echo "$util" | tr -d ' %')"
    for p in "${preferred[@]}"; do
      if [[ "$idx" == "$p" ]] && (( mem < 200 )) && (( util == 0 )); then
        echo "$idx"
        return 0
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  return 1
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if GPU="$(pick_free_gpu)"; then
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "Auto-selected free GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  else
    echo "ERROR: No free GPU among preferred set {1,3}." >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2 || true
    exit 1
  fi
else
  echo "Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

cd "$ROOT"
CMD="${1:-}"
shift || true

case "$CMD" in
  prepare)
    exec "$PY" -u evaluation/chestxray8/prepare_sft_dataset.py "$@"
    ;;
  debug-tokenize)
    exec "$PY" -u evaluation/chestxray8/debug_tokenized_example.py "$@"
    ;;
  debug-full-sft)
    exec "$PY" -u evaluation/chestxray8/train_chestxray8_sft.py \
      --experiment-type full_sft \
      --output-dir results/finetuning/full_sft_debug \
      --debug-one-step \
      --max-train-samples 2 \
      --batch-size 1 \
      --gradient-accumulation-steps 1 \
      --num-workers 0 \
      "$@"
    ;;
  debug-lora)
    exec "$PY" -u evaluation/chestxray8/train_chestxray8_sft.py \
      --experiment-type lora \
      --output-dir results/finetuning/lora_debug \
      --debug-one-step \
      --max-train-samples 2 \
      --batch-size 1 \
      --gradient-accumulation-steps 1 \
      --num-workers 0 \
      --lora-rank 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      "$@"
    ;;
  train-full-sft)
    exec "$PY" -u evaluation/chestxray8/train_chestxray8_sft.py \
      --experiment-type full_sft \
      --output-dir results/finetuning/full_sft \
      --batch-size 1 \
      --gradient-accumulation-steps 8 \
      --num-epochs 3 \
      --learning-rate 2e-5 \
      --warmup-ratio 0.03 \
      --weight-decay 0.01 \
      --max-seq-length 4096 \
      --save-steps 50 \
      --logging-steps 1 \
      --num-workers 2 \
      "$@"
    ;;
  train-lora)
    exec "$PY" -u evaluation/chestxray8/train_chestxray8_sft.py \
      --experiment-type lora \
      --output-dir results/finetuning/lora \
      --batch-size 1 \
      --gradient-accumulation-steps 8 \
      --num-epochs 5 \
      --learning-rate 1e-4 \
      --warmup-ratio 0.03 \
      --weight-decay 0.01 \
      --max-seq-length 4096 \
      --save-steps 50 \
      --logging-steps 1 \
      --lora-rank 8 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --num-workers 2 \
      "$@"
    ;;
  eval-comparison)
    exec "$PY" -u evaluation/chestxray8/eval_finetuned_comparison.py "$@"
    ;;
  *)
    cat <<'EOF'
Usage: run_finetune.sh <command> [args...]

Commands:
  prepare            Build patient-level splits + ShareGPT JSONL
  debug-tokenize     Print one complete tokenized supervised example
  debug-full-sft     One forward/backward/optimizer step (full SFT)
  debug-lora         One forward/backward/optimizer step (LoRA)
  train-full-sft     Full-model SFT (manual long run)
  train-lora         LoRA SFT (manual long run)
  eval-comparison    Held-out test eval + comparison Excel

Environment:
  LOCATEANYTHING_PYTHON   python binary
  CUDA_VISIBLE_DEVICES    pin a GPU (otherwise auto-picks free among 1,3)
  HF_HOME                 Hugging Face cache
EOF
    exit 1
    ;;
esac
