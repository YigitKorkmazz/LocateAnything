#!/usr/bin/env python3
"""Regression test for additive checkpoint SHA256 provenance."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHEST_DIR))

from train_chestxray8_sft import save_checkpoint  # noqa: E402


class DummyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save_pretrained(self, path, save_embedding_layers=False):
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": self.weight.detach()}, output / "adapter_model.bin")


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = DummyLanguageModel()
        self.mlp1 = torch.nn.Linear(1, 1)


class DummySaver:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_hash_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        checkpoint = save_checkpoint(
            root,
            DummyModel(),
            DummySaver(),
            DummySaver(),
            {"stage_id": "test", "output_dir": str(root)},
            step=3,
            is_lora=True,
            save_projector=True,
        )
        payload = json.loads((checkpoint / "checkpoint_hashes.json").read_text())
        adapter = checkpoint / "adapter" / "adapter_model.bin"
        projector = checkpoint / "mlp1.pt"
        assert payload["step"] == 3
        assert payload["sha256"]["adapter/adapter_model.bin"] == sha256(adapter)
        assert payload["sha256"]["mlp1.pt"] == sha256(projector)


def main() -> None:
    test_checkpoint_hash_sidecar()
    print("CHECKPOINT HASH TEST PASSED")


if __name__ == "__main__":
    main()
