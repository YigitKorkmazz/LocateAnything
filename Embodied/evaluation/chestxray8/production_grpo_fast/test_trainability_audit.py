#!/usr/bin/env python3
"""GPU test: model load produces exactly the LoRA-only trainability contract.

504 tensors / 14,966,784 parameters, projector and vision fully frozen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from production_grpo_fast import runtime as prod_runtime  # noqa: E402


def main() -> None:
    device = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:1")
    torch.cuda.set_device(device)
    model, tokenizer, processor, load_report = prod_runtime.build_model_and_tokenizer(device)
    audit = prod_runtime.trainability_audit(model)
    assert audit["trainable_tensors"] == 504, audit
    assert audit["lora_tensors"] == 504, audit
    assert audit["trainable_parameters"] == 14_966_784, audit
    assert audit["non_lora_trainable_tensors"] == 0, audit
    assert audit["projector_trainable_tensors"] == 0, audit
    assert audit["vision_trainable_tensors"] == 0, audit
    assert load_report["revision"] == prod_runtime.PINNED_REVISION, load_report
    print("PASS trainability audit:", audit)
    print("PASS model revision:", load_report["revision"])
    print("ALL TRAINABILITY TESTS PASSED")


if __name__ == "__main__":
    main()
