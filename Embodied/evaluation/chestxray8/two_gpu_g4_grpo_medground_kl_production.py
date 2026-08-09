#!/usr/bin/env python3
"""Guarded production entrypoint for the fresh MedGround-KL experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from two_gpu_g4_grpo_multistep_smoke import main


def _argument_value(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"{name} is required for MedGround-KL production") from exc


if __name__ == "__main__":
    config = Path(_argument_value("--config")).name
    if config != "chestxray8_grpo_hybrid_native_medground_kl.yaml":
        raise RuntimeError("MedGround-KL entrypoint refuses a non-KL config")
    output = str(Path(_argument_value("--output-dir")).resolve())
    if "medground_kl" not in output.lower():
        raise RuntimeError("MedGround-KL output directory must contain 'medground_kl'")
    main()
