"""MedLoc-R1 (arXiv:2603.28120) performance-aware curriculum IoU threshold
scheduler, ported from the supervisor's single-GPU reference implementation
(`iou_threshold_scheduler.py`, same class/API) for use as this backend's
optional spatial-reward mode.

Motivation (unchanged from the reference): a fixed-threshold spatial reward
(reward = 1 if IoU > tau else 0) is sparse early in medical grounding -- no
completion in a group clears tau, so all rewards in the group are equal, all
GRPO advantages are 0, and the policy gradient vanishes. Starting at an easy
tau0 and raising it toward tau_target only once the policy is accurate,
consistent, and has margin (MedLoc-R1 Eq. 9) lets difficulty follow
readiness instead of stalling training.

Multi-GPU note (the one real difference from the reference)
-------------------------------------------------------------
The reference script is single-process: it calls `step_group` once per
G-sized group, strictly in the training stream's true order, so a threshold
raise decided after group k can affect group k+1 immediately (even within
what this backend calls one "accumulation window" of GROUPS_PER_UPDATE=4
groups). This backend runs up to 4 of those groups in *parallel*, one per
GPU rank, so a raise cannot be applied mid-window without serializing the
ranks (defeating the point of parallelism).

The chosen, explicitly-documented simplification: tau is fixed for every
group inside one accumulation window (the value decided at the end of the
*previous* window), and the scheduler still ingests every group's IoUs in
true global group-index order once the window completes (see
`production_grpo_fast.distributed.sync_iou_threshold_across_ranks`, which
gathers all ranks' groups, feeds them to the scheduler on rank 0 in
ascending global-group-index order, and broadcasts the resulting tau back).
This makes the schedule's evolution depend only on `GROUPS_PER_UPDATE` and
elapsed groups -- identical regardless of whether the run uses 1, 2, 3, or 4
GPUs -- matching this project's existing invariant that world_size may only
change wall-clock time, never training semantics (see distributed.py's
gradient-reduction docstring for the same principle applied to gradients).
It does NOT reproduce the reference script's mid-window raises bit-for-bit;
that would require serializing groups within a window, which is not done
here on purpose.
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, List, Optional, Sequence


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass
class SchedulerConfig:
    enabled: bool = False
    tau0: float = 0.3                 # initial (easy) threshold
    tau_target: float = 0.8           # final (hard) threshold
    window_size: int = 30             # N: sliding-window length (groups)
    delta_min_margin: float = 0.10    # Delta: required IoU surplus over tau

    # step-size schedule: "piecewise" | "linear" | "cosine"
    decay: str = "piecewise"

    # piecewise decay params (MedLoc-R1 defaults)
    pw_delta1: float = 0.15
    pw_delta2: float = 0.10
    pw_delta3: float = 0.05
    pw_beta1: float = 0.55            # boundary: tau >= beta1 -> use delta2
    pw_beta2: float = 0.75            # boundary: tau >= beta2 -> use delta3

    # linear / cosine decay param (single step size)
    delta0: float = 0.2

    # Per-tau acceptance thresholds P(tau) and S(tau), applied by "tau >=
    # tau_lo" (first matching row from the top wins).
    accept_stages: tuple = (
        (0.75, 0.55, 0.40),
        (0.60, 0.75, 0.35),
        (0.00, 0.80, 0.20),
    )

    # Minimum window fill before ANY raise is considered.
    min_window_fill: Optional[int] = None  # defaults to window_size if None

    log_path: Optional[str] = None


def config_from_env() -> SchedulerConfig:
    """Build a SchedulerConfig from IOU_SCHED_* environment variables.

    Kept for parity with the reference script's env-var interface; this
    backend's `train.py` normally builds `SchedulerConfig` directly from CLI
    args instead (see `--iou-sched-*` flags).
    """
    def _f(name, default):
        v = os.getenv(name)
        return float(v) if v is not None and v != "" else default

    def _i(name, default):
        v = os.getenv(name)
        return int(v) if v is not None and v != "" else default

    enabled = os.getenv("IOU_SCHED_ENABLED", "").strip().lower() in ("1", "true", "yes")
    decay = os.getenv("IOU_SCHED_DECAY", "piecewise").strip().lower()
    if decay not in ("piecewise", "linear", "cosine"):
        raise ValueError(f"IOU_SCHED_DECAY must be piecewise|linear|cosine, got {decay!r}")

    return SchedulerConfig(
        enabled=enabled,
        tau0=_f("IOU_SCHED_TAU0", 0.3),
        tau_target=_f("IOU_SCHED_TAU_TARGET", 0.8),
        window_size=_i("IOU_SCHED_WINDOW", 30),
        delta_min_margin=_f("IOU_SCHED_DELTA_MARGIN", 0.10),
        decay=decay,
        pw_delta1=_f("IOU_SCHED_PW_DELTA1", 0.15),
        pw_delta2=_f("IOU_SCHED_PW_DELTA2", 0.10),
        pw_delta3=_f("IOU_SCHED_PW_DELTA3", 0.05),
        pw_beta1=_f("IOU_SCHED_PW_BETA1", 0.55),
        pw_beta2=_f("IOU_SCHED_PW_BETA2", 0.75),
        delta0=_f("IOU_SCHED_DELTA0", 0.2),
        log_path=os.getenv("IOU_SCHED_LOG", None),
    )


# --------------------------------------------------------------------------
# Step-size schedules: delta(tau)
# --------------------------------------------------------------------------

def delta_piecewise(tau: float, c: SchedulerConfig) -> float:
    d = c.pw_delta1
    if tau >= c.pw_beta1:
        d -= (c.pw_delta1 - c.pw_delta2)
    if tau >= c.pw_beta2:
        d -= (c.pw_delta2 - c.pw_delta3)
    return d


def delta_linear(tau: float, c: SchedulerConfig) -> float:
    span = max(1e-8, c.tau_target - c.tau0)
    frac = (tau - c.tau0) / span
    return c.delta0 * (1.0 - frac)


def delta_cosine(tau: float, c: SchedulerConfig) -> float:
    span = max(1e-8, c.tau_target - c.tau0)
    frac = min(1.0, max(0.0, (tau - c.tau0) / span))
    return 0.5 * c.delta0 * (1.0 + math.cos(math.pi * frac))


def delta_for(tau: float, c: SchedulerConfig) -> float:
    if c.decay == "piecewise":
        return delta_piecewise(tau, c)
    if c.decay == "linear":
        return delta_linear(tau, c)
    if c.decay == "cosine":
        return delta_cosine(tau, c)
    raise ValueError(f"unknown decay {c.decay!r}")


def accept_params(tau: float, c: SchedulerConfig):
    for tau_lo, P, S in c.accept_stages:
        if tau >= tau_lo:
            return P, S
    return c.accept_stages[-1][1], c.accept_stages[-1][2]


# --------------------------------------------------------------------------
# Scheduler
# --------------------------------------------------------------------------

@dataclass
class _WindowStats:
    r_bar: float
    sigma_r: float
    mean_iou: float
    margin: float
    fill: int


class IoUThresholdScheduler:
    """Stateful curriculum threshold scheduler.

    In this backend there is exactly one *logical* instance per training run
    (owned by rank 0 when world_size > 1; see distributed.py), fed one group
    (G IoUs) at a time via `step_group`, in true global group-index order.
    """

    def __init__(self, config: SchedulerConfig):
        self.c = config
        self.tau: float = float(config.tau0)
        self.step_idx: int = 0
        self.n_raises: int = 0
        self._rewards: Deque[float] = deque(maxlen=config.window_size)
        self._ious: Deque[float] = deque(maxlen=config.window_size)
        self._log_path = config.log_path or os.getenv("LOG_PATH") or "iou_sched_log.txt"
        self._min_fill = config.min_window_fill or config.window_size
        self._log(
            f"[IoU-SCHED INIT] decay={config.decay} tau0={self.tau:.3f} "
            f"tau_target={config.tau_target:.3f} window={config.window_size} "
            f"Delta={config.delta_min_margin:.3f}"
        )

    def _log(self, msg: str) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass  # logging must never break training

    def binary_rewards(self, iou_list: Sequence[float]) -> List[float]:
        return [1.0 if float(v) > self.tau else 0.0 for v in iou_list]

    def step_group(self, iou_list: Sequence[float]) -> List[float]:
        """Process one group's IoUs; returns binary rewards under the tau
        that was active BEFORE this call (a possible raise only affects the
        NEXT call)."""
        iou_list = [float(v) for v in iou_list]
        rewards = self.binary_rewards(iou_list)

        group_mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        group_mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0.0
        self._rewards.append(group_mean_reward)
        self._ious.append(group_mean_iou)
        self.step_idx += 1

        self._maybe_update()
        return rewards

    def _stats(self) -> _WindowStats:
        rs = list(self._rewards)
        ious = list(self._ious)
        n = len(rs)
        if n == 0:
            return _WindowStats(0.0, 0.0, 0.0, -self.tau, 0)
        r_bar = sum(rs) / n
        var = sum((x - r_bar) ** 2 for x in rs) / n
        sigma = math.sqrt(var)
        mean_iou = sum(ious) / len(ious)
        margin = mean_iou - self.tau
        return _WindowStats(r_bar, sigma, mean_iou, margin, n)

    def _maybe_update(self) -> None:
        if self.tau >= self.c.tau_target:
            return

        st = self._stats()
        P, S = accept_params(self.tau, self.c)
        Delta = self.c.delta_min_margin

        c1 = st.r_bar >= P
        c2 = st.sigma_r <= S
        c3 = st.margin >= Delta
        window_ready = st.fill >= self._min_fill

        should_log = (self.step_idx % 25 == 0) or (c1 and c2 and c3)
        if should_log:
            self._log(self._decision_line(st, P, S, Delta, c1, c2, c3, window_ready))

        if window_ready and c1 and c2 and c3:
            self._raise_threshold(st, P, S, Delta)

    def _decision_line(self, st, P, S, Delta, c1, c2, c3, window_ready) -> str:
        def mark(ok):
            return "PASS" if ok else "FAIL"

        why1 = f"r_bar={st.r_bar:.3f} {'>=' if c1 else '<'} P={P:.2f}"
        why2 = f"sigma={st.sigma_r:.3f} {'<=' if c2 else '>'} S={S:.2f}"
        why3 = (
            f"margin={st.margin:.3f} (meanIoU={st.mean_iou:.3f}-tau={self.tau:.3f}) "
            f"{'>=' if c3 else '<'} Delta={Delta:.2f}"
        )
        fill = f"window={st.fill}/{self.c.window_size}" + ("" if window_ready else " NOT-READY")
        return (
            f"[IoU-SCHED group={self.step_idx}] tau={self.tau:.3f} {fill}\n"
            f"    (1) reward-sufficiency : {mark(c1)}  {why1}\n"
            f"    (2) stability          : {mark(c2)}  {why2}\n"
            f"    (3) IoU-margin         : {mark(c3)}  {why3}\n"
            f"    -> raise? {'YES (all criteria satisfied)' if (c1 and c2 and c3 and window_ready) else 'no'}"
        )

    def _raise_threshold(self, st, P, S, Delta) -> None:
        d = delta_for(self.tau, self.c)
        old = self.tau
        new = min(self.tau + d, self.c.tau_target)
        self.tau = new
        self.n_raises += 1
        keep = self.c.window_size // 2
        self._rewards = deque(list(self._rewards)[-keep:], maxlen=self.c.window_size)
        self._ious = deque(list(self._ious)[-keep:], maxlen=self.c.window_size)
        self._log(
            f"[IoU-SCHED RAISE #{self.n_raises} @group={self.step_idx}] "
            f"criteria satisfied (r_bar={st.r_bar:.3f}>=P={P:.2f}, "
            f"sigma={st.sigma_r:.3f}<=S={S:.2f}, margin={st.margin:.3f}>=Delta={Delta:.2f}); "
            f"delta({old:.3f})={d:.3f} via {self.c.decay}; "
            f"tau {old:.3f} -> {new:.3f}"
            + (" (reached target)" if new >= self.c.tau_target else "")
            + f"; window half-refreshed (kept newest {keep})"
        )

    def state_dict(self) -> dict:
        return {
            "tau": self.tau,
            "step_idx": self.step_idx,
            "n_raises": self.n_raises,
            "rewards": list(self._rewards),
            "ious": list(self._ious),
            "config": asdict(self.c),
        }

    def load_state_dict(self, state: dict) -> None:
        self.tau = float(state["tau"])
        self.step_idx = int(state["step_idx"])
        self.n_raises = int(state["n_raises"])
        self._rewards = deque(state["rewards"], maxlen=self.c.window_size)
        self._ious = deque(state["ious"], maxlen=self.c.window_size)
