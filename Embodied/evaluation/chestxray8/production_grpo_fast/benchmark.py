#!/usr/bin/env python3
"""Throughput benchmark harness (Section 23 of the task spec).

Runs `train.py`'s real training loop (same model load, same generation,
reward, replay, backward, optimizer-step code paths) for a bounded number of
accumulation windows against a fresh output directory, then summarizes the
per-rank JSONL logs into the metrics the spec asks for. This never touches
`heldout194` and never runs a full 20-epoch job by itself.

Usage:
    python benchmark.py --gpus 1 --windows 5 --out results/bench_1gpu
    python benchmark.py --gpus 1,2 --windows 5 --out results/bench_2gpu
    python benchmark.py --gpus 1,2,3 --windows 5 --out results/bench_3gpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent


def run_benchmark(gpus: str, windows: int, output_dir: Path, python: str = sys.executable) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing benchmark output: {output_dir}")
    cmd = [
        python,
        str(HERE / "train.py"),
        "--output-dir",
        str(output_dir),
        "--gpus",
        gpus,
        "--max-windows",
        str(windows),
        "--no-checkpoint-every-epoch",
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    (output_dir / "benchmark_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "benchmark_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"benchmark subprocess failed (exit {proc.returncode}); see {output_dir}/benchmark_stderr.log")
    return summarize(output_dir, gpus=gpus, wall_seconds=elapsed)


def _load_rank_logs(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("train_log_rank*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(output_dir: Path, *, gpus: str, wall_seconds: float) -> Dict[str, Any]:
    rows = _load_rank_logs(output_dir)
    if not rows:
        raise RuntimeError(f"no log rows found under {output_dir}")
    world_size = max(r["world_size"] for r in rows)
    n_windows = max(r["optimizer_step"] for r in rows)
    total_groups = n_windows * 4
    total_completions = total_groups * 4
    seconds_per_group = wall_seconds / total_groups if total_groups else None
    report = {
        "gpus": gpus,
        "world_size": world_size,
        "optimizer_windows": n_windows,
        "total_prompt_groups": total_groups,
        "total_completions": total_completions,
        "wall_seconds": wall_seconds,
        "seconds_per_group": seconds_per_group,
        "prompt_groups_per_hour": 3600.0 / seconds_per_group if seconds_per_group else None,
        "optimizer_steps_per_hour": 3600.0 * n_windows / wall_seconds if wall_seconds else None,
        "completions_per_hour": 3600.0 * total_completions / wall_seconds if wall_seconds else None,
        "mean_reward": sum(r["mean_reward"] for r in rows) / len(rows),
        "per_rank_step_seconds": {
            rank: [r["step_seconds"] for r in rows if r["rank"] == rank]
            for rank in sorted({r["rank"] for r in rows})
        },
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run_benchmark(args.gpus, args.windows, Path(args.out))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
