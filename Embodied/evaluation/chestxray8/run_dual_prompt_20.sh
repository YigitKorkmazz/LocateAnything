#!/usr/bin/env bash
# Run Experiment A (bare_label) and Experiment B (radiology_context) on the
# SAME first N image-disease pairs, then build the comparison workbook.
# Prefers free GPUs among {1,3}.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${LOCATEANYTHING_PYTHON:-/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python}"
LIMIT="${1:-20}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/auto/k2/ykorkmaz/.cache/huggingface}"

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
    echo "ERROR: No free GPU among {1,3}." >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2 || true
    exit 1
  fi
fi

cd "$ROOT"
mkdir -p results/bare_label/visualizations results/radiology_context/visualizations

echo "===== Experiment A: bare_label (limit=${LIMIT}) ====="
"$PY" -u evaluation/chestxray8/eval_locateanything_bbox.py \
  --prompt-strategy bare_label \
  --limit "$LIMIT" \
  --visualize-count "$LIMIT" \
  --save-every 10 \
  --output-dir results/bare_label \
  2>&1 | tee results/bare_label/eval_chestxray8.log

echo "===== Experiment B: radiology_context (limit=${LIMIT}) ====="
"$PY" -u evaluation/chestxray8/eval_locateanything_bbox.py \
  --prompt-strategy radiology_context \
  --limit "$LIMIT" \
  --visualize-count "$LIMIT" \
  --save-every 10 \
  --output-dir results/radiology_context \
  2>&1 | tee results/radiology_context/eval_chestxray8.log

echo "===== Building comparison workbook ====="
"$PY" -u evaluation/chestxray8/compare_prompts.py \
  --bare-dir results/bare_label \
  --rad-dir results/radiology_context \
  --output results/chestxray8_prompt_comparison.xlsx

echo "STOPPING after ${LIMIT}-pair comparison (full 984-pair run not launched)."
