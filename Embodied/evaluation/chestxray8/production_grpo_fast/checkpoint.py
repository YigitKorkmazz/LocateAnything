"""Checkpoint / resume.

Exact resume semantics
-----------------------
Rollout sampling is fully determined by `(epoch, manifest_index)` via
`sampler.resolve_group` / `sampler.rollout_seeds` -- there is no dependence
on the *history* of RNG consumption, only on explicitly-derived seeds. So
the one piece of state that must be exact for correctness is
`next_global_group_index`: the first not-yet-processed group in the
continuous 15,800-group / 3,950-window stream (see `sampler.py`). Resuming
from that index can neither duplicate nor skip a group, by construction.

Checkpoints are only ever taken at a completed accumulation-window boundary
(right after an `optimizer.step()`), never mid-window: `train.py` finishes
whichever window is open when an epoch boundary is crossed before checking
whether to checkpoint. This means `next_global_group_index` is always a
multiple of `GROUPS_PER_UPDATE`, so resume never needs to reconstruct a
partially-accumulated gradient -- there is never a partial-window state to
restore, and `optimizer_step = next_global_group_index // GROUPS_PER_UPDATE`
always holds. `python`/`numpy`/`torch`/`torch.cuda` RNG state is still saved
and restored for defense-in-depth (e.g. any future stochastic component),
even though current correctness does not depend on it.

Only rank 0 writes a checkpoint; because gradients are identical across
ranks after `distributed.allreduce_sum_grads` (SUM, not an approximation)
and every rank runs the same deterministic AdamW update from the same
starting weights, all ranks' trainable parameters stay in exact lockstep,
so one rank's adapter + optimizer state fully describes every rank's.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .sampler import GROUPS_PER_UPDATE


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    next_global_group_index: int,
    config: Dict[str, Any],
) -> Path:
    if next_global_group_index % GROUPS_PER_UPDATE != 0:
        raise RuntimeError("checkpoints must land on a completed accumulation-window boundary")
    optimizer_step = next_global_group_index // GROUPS_PER_UPDATE
    target = output_dir / f"checkpoint_step_{optimizer_step:06d}"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    temp = Path(tempfile.mkdtemp(prefix=".checkpoint_tmp_", dir=output_dir))
    try:
        # save_embedding_layers=False: PEFT marks embeddings as "resized" any
        # time resize_token_embeddings was called with a same-size vocab
        # (train_chestxray8_sft.load_locateanything_model does this check but
        # still triggers the flag), which otherwise dumps the full frozen
        # embed_tokens + lm_head (~1.3GiB, never LoRA-trained) into every
        # single checkpoint. Missing this parameter filled the disk (255GiB
        # across 191 checkpoints) and crashed a live run -- see incident notes.
        model.language_model.save_pretrained(temp / "adapter", safe_serialization=True, save_embedding_layers=False)
        torch.save(
            {
                "next_global_group_index": int(next_global_group_index),
                "optimizer_step": int(optimizer_step),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": rng_state(),
            },
            temp / "training_state.pt",
        )
        (temp / "resolved_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        temp.rename(target)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def resume_checkpoint(path: Path, model: Any, optimizer: torch.optim.Optimizer, scheduler: Any) -> int:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    adapter = path / "adapter" / "adapter_model.safetensors"
    if not adapter.is_file():
        raise FileNotFoundError(f"missing adapter weights: {adapter}")
    set_peft_model_state_dict(model.language_model, load_file(str(adapter)))
    # weights_only=False: this file is our own trusted checkpoint (contains
    # numpy/python RNG state alongside tensors, which PyTorch >=2.6's default
    # weights_only=True unpickler rejects).
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])
    return int(state["next_global_group_index"])


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    candidates = sorted(output_dir.glob("checkpoint_step_*"))
    return candidates[-1] if candidates else None
