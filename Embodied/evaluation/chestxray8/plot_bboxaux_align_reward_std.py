#!/usr/bin/env python3
"""Mean reward + within-group reward std for the bbox_aux_align warm-start run,
and an overlay against the bbox_aux (with R_scale) run.

Every step is aggregated over ALL ranks' logs, i.e. all 4 groups of the
update (the 3-GPU runs' rank0 alone only sees 2 of the 4 groups; the
bbox_aux_align run was 1 GPU for steps 1-475, then 3 GPUs). Duplicate
(step, rank) rows from a resume keep the LAST occurrence.

Writes:
  <NEW>/bboxaux_align_reward_and_std.png      -- new run alone (same layout as bboxaux_reward_and_std.png)
  <OUT>/bboxaux_vs_bboxaux_align_reward_std.png -- overlay; reward panel uses the terms both runs
      share (format + semantic + IoU + align), i.e. old mean_reward - mean_r_scale.
  <OUT>/bboxaux_vs_bboxaux_align_components.png -- overlay of every reward component, IoU=0
      fraction, median pred/GT box area and KL.

Usage: python plot_bboxaux_align_reward_std.py [--window 50]
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(
    "/auto/k2/ykorkmaz/LocateAnything-playground/Embodied/results/chestxray8_production_grpo_fast_warmstart"
)
NEW = BASE / "G4_LR1E6_E20_1GPU_NATIVE_PROJECTORFROZEN_SEED42_VISUALRFT_BBOXAUXALIGN_KL004_WARMSTART_SFTLORADIRECT1000STEPS_CKPT500"
OLD = BASE / "G4_LR1E6_E20_3GPU_NATIVE_PROJECTORFROZEN_SEED42_VISUALRFT_BBOXAUX_KL004_WARMSTART_SFTLORADIRECT1000STEPS_CKPT500"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#1f1f1e", "#5f5e5a", "#e4e3df", "#fcfcfb"


def load_steps(run_dir):
    """step -> dict(reward, reward_no_scale, std) aggregated over all ranks.

    A resume can re-run steps with a different world size (the bbox_aux run
    re-ran 2569-2572 on 1 GPU after a 3-GPU attempt); the attempt that counts
    is the one in rank0's LAST row for that step, so only rows with that
    world_size are kept.
    """
    rows = {}
    for path in sorted(run_dir.glob("train_log_rank*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["optimizer_step"], r["rank"], r["world_size"])] = r
    final_ws = {}
    for path in (run_dir / "train_log_rank0.jsonl",):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    final_ws[r["optimizer_step"]] = r["world_size"]
    by_step = {}
    for (step, _, ws), r in rows.items():
        if ws == final_ws[step]:
            by_step.setdefault(step, []).append(r)
    out = {}
    for step, rs in sorted(by_step.items()):
        n = [r["local_group_count"] for r in rs]  # every group has G=4 rollouts
        w = lambda k: sum(r[k] * c for r, c in zip(rs, n)) / sum(n)
        area = [a / ga for r in rs for g, ga in zip(r["rollout_pred_area"], r["gt_area"]) if ga
                for a in g if a is not None]
        out[step] = {
            "reward": w("mean_reward"),
            "reward_no_scale": w("mean_reward") - w("mean_r_scale"),
            "std": statistics.mean(s for r in rs for s in r["group_reward_stds"]),
            "iou": w("mean_iou"),
            "r_align": w("mean_r_align"),
            "r_scale": w("mean_r_scale"),
            "iou_zero": w("frac_rollouts_iou_zero"),
            "kl": w("mean_kl_penalty"),
            "area_ratio": statistics.median(area) if area else float("nan"),
            "n_groups": sum(n),
        }
    # format + semantic = total minus the bbox terms that run actually summed
    mode = json.loads((run_dir / "resolved_config.json").read_text())["args"]["spatial_reward_mode"]
    scale_in_reward = mode == "bbox_aux"
    for d in out.values():
        d["fmt_sem"] = d["reward"] - d["iou"] - d["r_align"] - (d["r_scale"] if scale_in_reward else 0.0)
    return out


def rolling(v, w):
    return [statistics.mean(v[max(0, i - w + 1): i + 1]) for i in range(len(v))]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)


def series(ax, steps, vals, color, window, label=None, fmt="{:.3f}", dy=0):
    ax.plot(steps, vals, color=color, alpha=0.22, linewidth=0.6)
    sm = rolling(vals, window)
    ax.plot(steps[window - 1:], sm[window - 1:], color=color, linewidth=2, label=label)
    ax.annotate(fmt.format(sm[-1]), (steps[-1], sm[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=10, color=INK)
    return sm


def figure_single(new, window, out):
    steps = sorted(new)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=150, sharex=True, facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    series(axes[0], steps, [new[s]["reward"] for s in steps], BLUE, window, fmt="{:.2f}")
    style(axes[0], "Mean reward (format + semantic + IoU + alignment, max 4)", "reward")
    series(axes[1], steps, [new[s]["std"] for s in steps], BLUE, window)
    style(axes[1], "Mean within-group reward std", "std (population, G=4 rollouts)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("optimizer step", color=INK2)
    fig.text(0.01, 0.005, f"light line: per-step value over all 4 groups; bold line: {window}-step rolling mean. "
             "GRPO bbox_aux_align reward (no R_scale), warm start from SFT ckpt-500, 3950 steps.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def figure_overlay(new, old, window, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), dpi=150, sharex=True, facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    # End-of-line reward labels nearly coincide: push the lower one down, the higher one up.
    end = lambda d, k: rolling([d[s][k] for s in sorted(d)], window)[-1]
    old_below = end(old, "reward_no_scale") <= end(new, "reward")
    for data, color, label, key, dy in (
        (old, ORANGE, "bbox_aux (IoU + align + scale)", "reward_no_scale", -7 if old_below else 7),
        (new, BLUE, "bbox_aux_align (IoU + align)", "reward", 7 if old_below else -7),
    ):
        steps = sorted(data)
        series(axes[0], steps, [data[s][key] for s in steps], color, window, label=label, fmt="{:.2f}", dy=dy)
        series(axes[1], steps, [data[s]["std"] for s in steps], color, window, label=label)
    style(axes[0], "Mean reward, shared terms only (format + semantic + IoU + align)", "reward")
    style(axes[1], "Mean within-group std of each run's own training reward", "std (population, G=4 rollouts)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("optimizer step", color=INK2)
    axes[0].legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, f"light line: per-step value over all 4 groups; bold line: {window}-step rolling mean. "
             "Both: warm start from SFT ckpt-500, 3950 steps, same groups and sampling seeds.\n"
             "Top: R_scale subtracted from the bbox_aux run so both curves have the same max (4). "
             "Bottom: the bbox_aux std includes the R_scale term, so it is expected to be higher.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def figure_components(new, old, window, out):
    panels = [
        ("reward_shared", "Reward, shared terms (format + semantic + IoU + align)", "{:.2f}"),
        ("iou", "R_iou (mean IoU)", "{:.3f}"),
        ("r_align", "R_align", "{:.3f}"),
        ("r_scale", "R_scale (in reward only for bbox_aux)", "{:.3f}"),
        ("fmt_sem", "Format + semantic", "{:.3f}"),
        ("iou_zero", "Fraction of rollouts with IoU = 0", "{:.3f}"),
        ("area_ratio", "Median predicted / GT box area", "{:.2f}"),
        ("kl", "KL penalty (mean)", "{:.3f}"),
    ]
    for d, key in ((old, "reward_no_scale"), (new, "reward")):
        for v in d.values():
            v["reward_shared"] = v[key]
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), dpi=130, sharex=True, facecolor=SURFACE)
    runs = ((old, ORANGE, "bbox_aux (IoU + align + scale)"), (new, BLUE, "bbox_aux_align (IoU + align)"))
    for ax, (key, title, fmt) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        ends = [rolling([d[s][key] for s in sorted(d)], window)[-1] for d, _, _ in runs]
        # end labels always split: the higher run's label above its line end, the lower one below
        for (d, color, label), end, other in zip(runs, ends, ends[::-1]):
            steps = sorted(d)
            series(ax, steps, [d[s][key] for s in steps], color, window, label=label, fmt=fmt,
                   dy=8 if end >= other else -8)
        style(ax, title, "")
        if key == "area_ratio":
            # per-step medians over 8-16 rollouts spike; clip to the bulk so the trend is readable
            raw = sorted(v[key] for d, _, _ in runs for v in d.values())
            ax.set_ylim(raw[int(0.005 * len(raw))], raw[int(0.995 * len(raw))])
            ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    for ax in axes[-1]:
        ax.set_xlabel("optimizer step", color=INK2)
    axes[0][0].legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.suptitle(f"bbox_aux vs bbox_aux_align GRPO, SFT ckpt-500 warm start ({window}-step rolling mean)",
                 fontsize=13, color=INK)
    fig.text(0.01, 0.005, "light line: per-step value over all 4 groups; bold line: rolling mean. "
             "Both runs: same groups and sampling seeds, 3950 steps. R_scale is logged for bbox_aux_align "
             "but not part of its reward.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.98))
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    new, old = load_steps(NEW), load_steps(OLD)
    for name, d in (("new", new), ("old", old)):
        steps = sorted(d)
        assert steps == list(range(1, 3951)), (name, steps[:3], steps[-3:], len(steps))
        assert all(d[s]["n_groups"] == 4 for s in steps), name
        n = len(steps) // 10
        for k in ("reward", "reward_no_scale", "std"):
            v = [d[s][k] for s in steps]
            print(f"{name} {k:16s} first10% {statistics.mean(v[:n]):.3f}  last10% {statistics.mean(v[-n:]):.3f}")
    figure_single(new, args.window, NEW / "bboxaux_align_reward_and_std.png")
    figure_overlay(new, old, args.window, BASE / "bboxaux_vs_bboxaux_align_reward_std.png")
    figure_components(new, old, args.window, BASE / "bboxaux_vs_bboxaux_align_components.png")


if __name__ == "__main__":
    main()
