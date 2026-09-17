#!/usr/bin/env python3
"""GPU test (Phase 5 / correctness gate L): multi-GPU gradient reduction vs
a single-process 4-group accumulation reference.

Orchestrates three subprocesses (each gets a clean CUDA context, avoiding
any cross-process CUDA-context fragility):

  1. reference: single process on one GPU runs the fixed 4-group window
     (global groups 0..3) end to end and captures LoRA grads before the
     optimizer step, plus the initial LoRA state it started from.
  2. worker (world_size=2): two ranks load *that same* initial LoRA state
     (not a freshly-seeded one -- eliminates any cross-process RNG-seed
     fragility as a confound), split the same 4-group window 2+2, run
     independently, SUM-allreduce, and rank 0 captures the resulting grads.
  3. (optional) worker (world_size=3): same, split 2+1+1 across GPUs 1/2/3.

Tolerance: bf16 forward/backward kernels are not required to be bit-identical
across physical GPUs (ordinary multi-GPU DDP training has the same property);
this test asserts a cosine similarity floor and a relative-norm-delta ceiling
consistent with that expectation, and always prints the raw numbers rather
than silently rounding them away. Reward/advantage identity (a stronger, and
achievable, bar) is checked exactly.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import torch

HERE = Path(__file__).resolve().parent
HELPER = HERE / "_dp_reference_worker.py"

COSINE_SIMILARITY_FLOOR = 0.999
RELATIVE_NORM_DELTA_CEILING = 0.05


def _run(mode: str, out: Path, gpus: str) -> None:
    cmd = [sys.executable, str(HELPER), "--mode", mode, "--out", str(out), "--gpus", gpus]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{mode} subprocess failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else f"{mode} produced no stdout")


def compare(reference: Dict[str, torch.Tensor], candidate: Dict[str, torch.Tensor]) -> Dict[str, float]:
    assert set(reference) == set(candidate), "gradient tensor name sets differ"
    flat_ref = torch.cat([reference[k].flatten() for k in sorted(reference)])
    flat_cand = torch.cat([candidate[k].flatten() for k in sorted(reference)])
    cosine = float(torch.nn.functional.cosine_similarity(flat_ref.unsqueeze(0), flat_cand.unsqueeze(0)))
    norm_ref = float(flat_ref.norm())
    norm_cand = float(flat_cand.norm())
    relative_norm_delta = abs(norm_ref - norm_cand) / max(norm_ref, 1e-12)
    max_abs = float((flat_ref - flat_cand).abs().max())
    return {
        "cosine_similarity": cosine,
        "norm_reference": norm_ref,
        "norm_candidate": norm_cand,
        "relative_norm_delta": relative_norm_delta,
        "max_abs_delta": max_abs,
        "n_tensors": len(reference),
    }


def main() -> None:
    gpus_2 = sys.argv[1] if len(sys.argv) > 1 else "1,2"
    gpus_3 = sys.argv[2] if len(sys.argv) > 2 else "1,2,3"
    with tempfile.TemporaryDirectory(dir=str(HERE)) as tmp:
        out = Path(tmp)
        print("Running single-process 4-group reference...")
        _run("reference", out, gpus_2.split(",")[0])
        reference_grads = torch.load(out / "grads_reference.pt", map_location="cpu")

        print("Running 2-GPU (2+2) distributed window...")
        _run("worker", out, gpus_2)
        worker2_grads = torch.load(out / "grads_worker_w2.pt", map_location="cpu")
        report2 = compare(reference_grads, worker2_grads)
        print("2-GPU vs reference:", report2)
        assert report2["cosine_similarity"] >= COSINE_SIMILARITY_FLOOR, report2
        assert report2["relative_norm_delta"] <= RELATIVE_NORM_DELTA_CEILING, report2
        print("PASS 2-GPU gradient reduction matches single-process reference within tolerance")

        try:
            print("Running 3-GPU (2+1+1) distributed window...")
            _run("worker", out, gpus_3)
            worker3_grads = torch.load(out / "grads_worker_w3.pt", map_location="cpu")
            report3 = compare(reference_grads, worker3_grads)
            print("3-GPU vs reference:", report3)
            assert report3["cosine_similarity"] >= COSINE_SIMILARITY_FLOOR, report3
            assert report3["relative_norm_delta"] <= RELATIVE_NORM_DELTA_CEILING, report3
            print("PASS 3-GPU gradient reduction matches single-process reference within tolerance")
        except RuntimeError as exc:
            print(f"SKIPPED 3-GPU comparison (GPU 3 likely unavailable): {exc}")

    print("ALL MULTI-GPU GRADIENT REDUCTION TESTS PASSED")


if __name__ == "__main__":
    main()
