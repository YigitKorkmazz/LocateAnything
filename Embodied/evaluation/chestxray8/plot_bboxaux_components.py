#!/usr/bin/env python3
"""Plot each bbox_aux reward component step by step for a warm-start run.

Panels: total reward, IoU, R_align, R_scale, format+semantic (residual =
total - iou - align [- scale]), fraction of rollouts with IoU 0, KL penalty.
For a `bbox_aux_align` run R_scale is not part of the reward (diagnostic only),
so it is not subtracted from the residual.
Raw per-step values (faint) + rolling mean (bold). Duplicate optimizer_steps
(resume re-appends) keep the LAST occurrence.

Usage: python plot_bboxaux_components.py [--run DIR] [--window 40] [--output PNG]
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = (
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/"
    "chestxray8_production_grpo_fast_warmstart/"
    "G4_LR1E6_E20_3GPU_NATIVE_PROJECTORFROZEN_SEED42_VISUALRFT_BBOXAUX_KL004_"
    "WARMSTART_SFTLORADIRECT1000STEPS_CKPT500"
)


def rolling(v, w):
    return [statistics.mean(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(RUN))
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    args.output = args.output or args.run / "bboxaux_reward_components.png"
    mode = json.loads((args.run / "resolved_config.json").read_text())["args"]["spatial_reward_mode"]
    scale_in_reward = mode == "bbox_aux"

    by_step = {}
    with open(args.run / "train_log_rank0.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_step[r["optimizer_step"]] = r
    steps = sorted(by_step)
    col = lambda k: [by_step[s][k] for s in steps]
    iou, al, sc, tot = col("mean_iou"), col("mean_r_align"), col("mean_r_scale"), col("mean_reward")
    fs = [t - a - b - (c if scale_in_reward else 0.0) for t, a, b, c in zip(tot, iou, al, sc)]

    panels = [
        ("Total reward", tot, "#444444"),
        ("R_iou (mean IoU)", iou, "#2a78d6"),
        ("R_align", al, "#eb6834"),
        ("R_scale" if scale_in_reward else "R_scale (diagnostic, not in reward)", sc, "#1baf7a"),
        ("Format + semantic (total − iou − align" + (" − scale)" if scale_in_reward else ")"), fs, "#9b59b6"),
        ("Fraction of rollouts with IoU = 0", col("frac_rollouts_iou_zero"), "#c0392b"),
        ("KL penalty (mean)", col("mean_kl_penalty"), "#7f8c8d"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), dpi=130, sharex=True)
    axes = axes.ravel()
    for ax, (title, vals, c) in zip(axes, panels):
        ax.plot(steps, vals, color=c, alpha=0.18, linewidth=0.7)
        sm = rolling(vals, args.window)
        ax.plot(steps, sm, color=c, linewidth=2)
        ax.annotate(f"{sm[-1]:.3f}", (steps[-1], sm[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, fontweight="bold", color=c)
        ax.set_title(f"{title}\nfirst {sm[0]:.3f} → last {sm[-1]:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].axis("off")
    for ax in axes[-3:-1]:
        ax.set_xlabel("optimizer step")
    fig.suptitle(f"{mode} GRPO, SFT ckpt500 warm-start ({args.window}-step rolling mean)", fontsize=12)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print("wrote", args.output, "steps", steps[0], "-", steps[-1], "n=", len(steps))
    for name, v in [("total", tot), ("iou", iou), ("align", al), ("scale", sc), ("fmt+sem", fs)]:
        n = max(1, len(v) // 10)
        print(f"{name:8s} first10%={statistics.mean(v[:n]):.3f} last10%={statistics.mean(v[-n:]):.3f}")


if __name__ == "__main__":
    main()
