#!/usr/bin/env bash
# Convenience launcher for ChestX-ray8 LocateAnything bbox evaluation.
# Prefers free GPUs among {1,3} by default to avoid interrupting other users
# (on this host GPUs 0 and 2 are often occupied).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${LOCATEANYTHING_PYTHON:-/auto/k2/ykorkmaz/envs/miniconda3/envs/locateanything/bin/python}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/auto/k2/ykorkmaz/.cache/huggingface}"

pick_free_gpu() {
  # Prefer GPUs 1 and 3. Consider free if memory.used < 200 MiB and util == 0.
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
    echo "ERROR: No free GPU among preferred set {1,3}. Refusing to start." >&2
    echo "Set CUDA_VISIBLE_DEVICES manually only if you intentionally want another GPU." >&2
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv >&2 || true
    exit 1
  fi
else
  echo "Using pre-set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

cd "$ROOT"
exec "$PY" -u evaluation/chestxray8/eval_locateanything_bbox.py "$@"
