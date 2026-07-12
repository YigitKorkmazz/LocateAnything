#!/usr/bin/env python3
"""Qualitative random visualizations from completed ChestX-ray8 evaluation.

Reads existing intermediate_results.jsonl only — does not run inference.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

SEED = 42
N_SAMPLES = 20
GT_COLOR = (0, 200, 0)
PRED_COLOR = (220, 40, 40)
TITLE_BG = (18, 18, 18)
TITLE_FG = (255, 255, 255)


def parse_boxes(value) -> List[List[float]]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [[float(x) for x in box] for box in value]
    parsed = ast.literal_eval(str(value))
    if not parsed:
        return []
    return [[float(x) for x in box] for box in parsed]


def load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def safe_disease_token(disease: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", disease).strip("_")


def image_index_stem(image_name: str) -> str:
    return Path(image_name).stem


def draw_sample(
    image: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    title: str,
) -> Image.Image:
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    box_font = load_font(18)

    for box in gt_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=4)

    for box in pred_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=4)

    if not pred_boxes:
        msg = "No prediction"
        msg_font = load_font(36)
        # Centered overlay with dark backing for readability.
        bbox = draw.textbbox((0, 0), msg, font=msg_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = vis.width // 2, vis.height // 2
        pad = 12
        draw.rectangle(
            [cx - tw // 2 - pad, cy - th // 2 - pad, cx + tw // 2 + pad, cy + th // 2 + pad],
            fill=(0, 0, 0),
        )
        draw.text((cx - tw // 2, cy - th // 2), msg, fill=(255, 80, 80), font=msg_font)

    title_font = load_font(22)
    # Wrap long titles to ~2–3 lines.
    max_chars = 92
    words = title.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    line_h = 28
    header_h = 16 + line_h * len(lines) + 12
    canvas = Image.new("RGB", (vis.width, vis.height + header_h), TITLE_BG)
    canvas.paste(vis, (0, header_h))
    tdraw = ImageDraw.Draw(canvas)
    y = 12
    for line in lines:
        tdraw.text((12, y), line, fill=TITLE_FG, font=title_font)
        y += line_h
    return canvas


def make_contact_sheet(images: Sequence[Image.Image], cols: int = 5, rows: int = 4) -> Image.Image:
    assert len(images) == cols * rows
    # Normalize thumbnails to a common size while preserving aspect.
    thumb_w, thumb_h = 512, 560  # slightly taller for title band
    thumbs = [im.copy() for im in images]
    resized = []
    for im in thumbs:
        im = im.convert("RGB")
        # Fit inside thumb box
        scale = min(thumb_w / im.width, thumb_h / im.height)
        nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), (30, 30, 30))
        tile.paste(im, ((thumb_w - nw) // 2, (thumb_h - nh) // 2))
        resized.append(tile)

    gap = 8
    sheet_w = cols * thumb_w + (cols + 1) * gap
    sheet_h = rows * thumb_h + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    for idx, tile in enumerate(resized):
        r, c = divmod(idx, cols)
        x = gap + c * (thumb_w + gap)
        y = gap + r * (thumb_h + gap)
        sheet.paste(tile, (x, y))
    return sheet


def load_samples(jsonl_path: Path) -> List[dict]:
    samples = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(record["sample_row"])
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/full_radiology_context"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    results_dir = args.results_dir
    jsonl_path = results_dir / "intermediate_results.jsonl"
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing results file: {jsonl_path}")

    out_dir = results_dir / "random_visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(jsonl_path)
    if len(samples) < args.n:
        raise RuntimeError(f"Need at least {args.n} samples, found {len(samples)}")

    rng = random.Random(args.seed)
    selected = rng.sample(samples, args.n)

    rendered: List[Image.Image] = []
    png_paths: List[Path] = []
    manifest = []

    for sample in selected:
        image_name = sample["Image"]
        disease = sample["Disease"]
        image_path = Path(sample["Image Path"])
        gt_boxes = parse_boxes(sample.get("Ground Truth Boxes"))
        pred_boxes = parse_boxes(sample.get("Predicted Boxes"))
        n_gt = int(sample.get("Number of GT Boxes", len(gt_boxes)))
        n_pred = int(sample.get("Number of Predicted Boxes", len(pred_boxes)))
        iou = float(sample.get("Maximum IoU") or 0.0)

        title = (
            f"{image_name} | {disease} | IoU={iou:.4f} | "
            f"GT boxes={n_gt} | Pred boxes={n_pred}"
        )

        image = Image.open(image_path).convert("RGB")
        vis = draw_sample(image, gt_boxes, pred_boxes, title)

        stem = f"{image_index_stem(image_name)}_{safe_disease_token(disease)}"
        png_path = out_dir / f"{stem}.png"
        vis.save(png_path, format="PNG")
        png_paths.append(png_path)
        rendered.append(vis)
        manifest.append(
            {
                "file": png_path.name,
                "image": image_name,
                "disease": disease,
                "iou": iou,
                "n_gt": n_gt,
                "n_pred": n_pred,
                "image_path": str(image_path),
            }
        )
        print(f"Saved {png_path}")

    # Multi-page PDF (one visualization per page), presentation resolution.
    pdf_path = out_dir / "chestxray8_random20_visualizations.pdf"
    # Upscale slightly for print/presentation clarity if needed.
    pdf_pages = []
    for im in rendered:
        # Keep native resolution; ensure RGB.
        page = im.convert("RGB")
        pdf_pages.append(page)
    pdf_pages[0].save(
        pdf_path,
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=150.0,
    )
    print(f"Saved {pdf_path}")

    # 5×4 contact sheet
    grid = make_contact_sheet(rendered, cols=5, rows=4)
    grid_path = out_dir / "chestxray8_random20_grid.png"
    grid.save(grid_path, format="PNG")
    print(f"Saved {grid_path}")

    manifest_path = out_dir / "sample_manifest_seed42.json"
    with manifest_path.open("w") as f:
        json.dump(
            {"seed": args.seed, "n": args.n, "samples": manifest},
            f,
            indent=2,
        )
    print(f"Saved {manifest_path}")
    print(f"Done. {len(rendered)} visualizations written to {out_dir}")


if __name__ == "__main__":
    main()
