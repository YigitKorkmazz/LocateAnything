"""Current/old/reference parameter-state helpers for PBD GRPO."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator

import torch


def is_policy_trainable_name(name: str) -> bool:
    lower = name.lower()
    return "lora_" in lower or name.startswith("mlp1.")


def is_lora_name(name: str) -> bool:
    return "lora_" in name.lower()


def is_projector_name(name: str) -> bool:
    return name.startswith("mlp1.")


@dataclass
class PolicySnapshot:
    optimizer_step: int
    tensors: Dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model, *, optimizer_step: int) -> "PolicySnapshot":
        tensors = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
            if is_policy_trainable_name(name)
        }
        if not any(is_lora_name(name) for name in tensors):
            raise RuntimeError("policy snapshot contains no LoRA adapter tensors")
        if not any(is_projector_name(name) for name in tensors):
            raise RuntimeError("policy snapshot contains no projector tensors")
        return cls(optimizer_step=int(optimizer_step), tensors=tensors)

    def load_into(self, model) -> None:
        state = model.state_dict()
        missing = sorted(set(self.tensors) - set(state))
        if missing:
            raise RuntimeError(f"snapshot parameters missing from model: {missing[:10]}")
        with torch.no_grad():
            for name, source in self.tensors.items():
                state[name].copy_(source.to(device=state[name].device, dtype=state[name].dtype))


@contextmanager
def use_policy_snapshot(model, snapshot: PolicySnapshot) -> Iterator[None]:
    """Temporarily swap LoRA/projector state on a shared frozen base model."""
    restore = PolicySnapshot.capture(model, optimizer_step=snapshot.optimizer_step)
    snapshot.load_into(model)
    try:
        yield
    finally:
        restore.load_into(model)


class OldPolicyController:
    """Synchronize old policy exactly once after each completed optimizer step."""

    def __init__(self, current_model, old_model, *, sync_interval: int = 1) -> None:
        if sync_interval != 1:
            raise ValueError(
                "this experiment explicitly requires old_policy_sync_interval=1"
            )
        self.current_model = current_model
        self.old_model = old_model
        self.sync_interval = sync_interval
        self.last_synced_optimizer_step = -1
        self._freeze_old()

    def _freeze_old(self) -> None:
        self.old_model.eval()
        for parameter in self.old_model.parameters():
            parameter.requires_grad_(False)

    def assert_frozen(self) -> None:
        trainable = [
            name
            for name, parameter in self.old_model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise RuntimeError(f"old policy unexpectedly trainable: {trainable[:10]}")

    def synchronize_after_optimizer_step(self, optimizer_step: int) -> None:
        optimizer_step = int(optimizer_step)
        if optimizer_step <= self.last_synced_optimizer_step:
            raise RuntimeError(
                "old policy synchronization must follow a new completed optimizer step"
            )
        current_state = self.current_model.state_dict()
        old_state = self.old_model.state_dict()
        names = [name for name in current_state if is_policy_trainable_name(name)]
        if not any(is_lora_name(name) for name in names):
            raise RuntimeError("current policy has no LoRA state to synchronize")
        if not any(is_projector_name(name) for name in names):
            raise RuntimeError("current policy has no projector state to synchronize")
        missing = sorted(set(names) - set(old_state))
        if missing:
            raise RuntimeError(f"old policy missing synchronized tensors: {missing[:10]}")
        with torch.no_grad():
            for name in names:
                old_state[name].copy_(
                    current_state[name].detach().to(
                        device=old_state[name].device,
                        dtype=old_state[name].dtype,
                    )
                )
        self.last_synced_optimizer_step = optimizer_step
        self._freeze_old()
        self.assert_frozen()


def assert_approved_trainable_parameters(model) -> Dict[str, object]:
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    lora = [name for name in trainable if is_lora_name(name)]
    projector = [name for name in trainable if is_projector_name(name)]
    unapproved = [
        name
        for name in trainable
        if not is_lora_name(name) and not is_projector_name(name)
    ]
    vision = [name for name in trainable if name.startswith("vision_model.")]
    if not lora:
        raise RuntimeError("fresh LoRA adapters are missing")
    if not projector:
        raise RuntimeError("multimodal projector is unexpectedly frozen")
    if unapproved:
        raise RuntimeError(f"unapproved trainable parameters: {unapproved[:20]}")
    if vision:
        raise RuntimeError(f"vision encoder unexpectedly trainable: {vision[:20]}")
    return {
        "trainable_names": trainable,
        "lora_names": lora,
        "projector_names": projector,
        "unapproved_names": unapproved,
    }


def assert_frozen_modules(named_parameters: Iterable[tuple[str, torch.nn.Parameter]]) -> None:
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"expected frozen parameters, found: {trainable[:20]}")
