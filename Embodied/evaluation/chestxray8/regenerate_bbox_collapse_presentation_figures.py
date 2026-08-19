#!/usr/bin/env python3
"""Regenerate presentation-clean bbox-area figures from existing CSV metrics."""

from __future__ import annotations

import csv
import html
import math
import subprocess
from pathlib import Path
from typing import Any

from reportlab.lib.colors import Color, HexColor
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = (
    HERE
    / "results/chestxray8_hybrid_grpo_native/analysis"
    / "bbox_size_collapse_existing_artifacts_20260816"
)
OVERALL_CSV = ANALYSIS_DIR / "validation_overall.csv"
DISEASE_CSV = ANALYSIS_DIR / "validation_disease.csv"

OVERALL_STEM = "presentation_overall_bbox_area_vs_step"
HEATMAP_STEM = "presentation_disease_wise_median_bbox_area_controlled"

SERIES = (
    (
        "G8 Hybrid projector-frozen",
        "Hybrid, projector frozen",
        "#4C78A8",
    ),
    (
        "G8 Hybrid Case-B projector-trainable",
        "Hybrid, projector trainable",
        "#F58518",
    ),
    (
        "G8 NTP controlled LR1e-5",
        "Controlled NTP, LR=1e-5",
        "#54A24B",
    ),
)

DISEASE_ORDER = (
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
)
CONTROLLED_STEPS = (0, 100, 200, 300, 400, 500)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def render_png(svg_path: Path, png_path: Path, density: int = 180) -> None:
    subprocess.run(
        ["/usr/bin/convert", "-density", str(density), str(svg_path), str(png_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def overall_plot(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    width, height = 1500, 900
    left, right, top, bottom = 155, 70, 100, 150
    plot_width = width - left - right
    plot_height = height - top - bottom

    plotted: list[tuple[str, str, str, list[tuple[int, float]]]] = []
    for source_name, display_name, color in SERIES:
        values = sorted(
            (
                int(row["step"]),
                float(row["median_area_valid_boxes"]),
            )
            for row in rows
            if row["experiment"] == source_name and row["median_area_valid_boxes"]
        )
        if not values:
            raise RuntimeError(f"No overall data found for {source_name}")
        plotted.append((source_name, display_name, color, values))

    all_values = [value for _, _, _, values in plotted for _, value in values]
    log_min = math.floor(min(math.log10(value) for value in all_values))
    log_max = 0.0
    x_min, x_max = 0, max(step for _, _, _, values in plotted for step, _ in values)

    def xy(step: int, value: float) -> tuple[float, float]:
        x = left + (step - x_min) / (x_max - x_min) * plot_width
        y = top + (log_max - math.log10(value)) / (log_max - log_min) * plot_height
        return x, y

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}</style>',
        f'<text x="{width / 2}" y="55" text-anchor="middle" font-size="38" '
        'font-weight="bold">Overall bbox area vs step</text>',
    ]

    for exponent in range(int(log_max), int(log_min) - 1, -1):
        value = 10**exponent
        _, y = xy(0, value)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" '
                'stroke="#d9d9d9" stroke-width="1.5"/>',
                f'<text x="{left-18}" y="{y+8}" text-anchor="end" '
                f'font-size="22">{value:g}</text>',
            ]
        )

    for step in CONTROLLED_STEPS:
        x, _ = xy(step, 1.0)
        svg.extend(
            [
                f'<line x1="{x}" y1="{height-bottom}" x2="{x}" '
                f'y2="{height-bottom+8}" stroke="#333" stroke-width="2"/>',
                f'<text x="{x}" y="{height-bottom+38}" text-anchor="middle" '
                f'font-size="22">{step}</text>',
            ]
        )

    svg.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
            'stroke="#333" stroke-width="3"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#333" stroke-width="3"/>',
        ]
    )

    for _, _, color, values in plotted:
        points = [xy(step, value) for step, value in values]
        svg.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" '
            f'fill="none" stroke="{color}" stroke-width="6" stroke-linejoin="round"/>'
        )
        for x, y in points:
            svg.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}"/>')

    legend_x, legend_y = 875, 125
    for index, (_, display_name, color, _) in enumerate(plotted):
        y = legend_y + index * 48
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+60}" y2="{y}" '
                f'stroke="{color}" stroke-width="6"/>',
                f'<circle cx="{legend_x+30}" cy="{y}" r="7" fill="{color}"/>',
                f'<text x="{legend_x+78}" y="{y+8}" font-size="23">'
                f'{html.escape(display_name)}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height-42}" text-anchor="middle" '
            'font-size="27">Optimizer step</text>',
            f'<text x="0" y="0" text-anchor="middle" font-size="27" '
            f'transform="translate(48 {top + plot_height / 2}) rotate(-90)">'
            'Median valid-box area fraction (log scale)</text>',
            "</svg>",
        ]
    )

    svg_path = ANALYSIS_DIR / f"{OVERALL_STEM}.svg"
    png_path = ANALYSIS_DIR / f"{OVERALL_STEM}.png"
    pdf_path = ANALYSIS_DIR / f"{OVERALL_STEM}.pdf"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    render_png(svg_path, png_path)

    scale = 0.5
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width * scale, height * scale))
    pdf.setTitle("Overall bbox area vs step")
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawCentredString(width * scale / 2, (height - 55) * scale, "Overall bbox area vs step")
    for exponent in range(int(log_max), int(log_min) - 1, -1):
        value = 10**exponent
        _, y_svg = xy(0, value)
        y = (height - y_svg) * scale
        pdf.setStrokeColor(HexColor("#d9d9d9"))
        pdf.setLineWidth(0.75)
        pdf.line(left * scale, y, (width - right) * scale, y)
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 11)
        pdf.drawRightString((left - 18) * scale, y - 3, f"{value:g}")
    pdf.setStrokeColor(HexColor("#333333"))
    pdf.setLineWidth(1.5)
    pdf.line(left * scale, bottom * scale, left * scale, (height - top) * scale)
    pdf.line(left * scale, bottom * scale, (width - right) * scale, bottom * scale)
    for step in CONTROLLED_STEPS:
        x_svg, _ = xy(step, 1.0)
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 11)
        pdf.drawCentredString(x_svg * scale, (bottom - 38) * scale, str(step))
    for _, _, color, values in plotted:
        points = [(x * scale, (height - y) * scale) for x, y in (xy(step, value) for step, value in values)]
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(3)
        pdf.drawPath(path)
        pdf.setFillColor(HexColor(color))
        for x, y in points:
            pdf.circle(x, y, 4, fill=1, stroke=0)
    for index, (_, display_name, color, _) in enumerate(plotted):
        y_svg = legend_y + index * 48
        y = (height - y_svg) * scale
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(3)
        pdf.line(legend_x * scale, y, (legend_x + 60) * scale, y)
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 11)
        pdf.drawString((legend_x + 78) * scale, y - 4, display_name)
    pdf.setFont("Helvetica", 13)
    pdf.drawCentredString(width * scale / 2, 21, "Optimizer step")
    pdf.saveState()
    pdf.translate(24, height * scale / 2)
    pdf.rotate(90)
    pdf.drawCentredString(0, 0, "Median valid-box area fraction (log scale)")
    pdf.restoreState()
    pdf.save()
    return png_path, pdf_path


def heat_color(t: float) -> tuple[int, int, int]:
    return (
        round(242 - 170 * t),
        round(245 - 110 * t),
        round(250 - 25 * t),
    )


def heatmap(rows: list[dict[str, str]]) -> tuple[Path, Path]:
    selected = [
        row
        for row in rows
        if row["experiment"] == "G8 NTP controlled LR1e-5"
        and int(row["step"]) in CONTROLLED_STEPS
    ]
    lookup = {
        (row["disease"], int(row["step"])): (
            float(row["median_area_valid_boxes"])
            if row["median_area_valid_boxes"]
            else None
        )
        for row in selected
    }
    values = [value for value in lookup.values() if value is not None and value > 0]
    if not values:
        raise RuntimeError("No controlled disease-level values found")
    log_min = min(math.log10(value) for value in values)
    log_max = max(math.log10(value) for value in values)

    width, height = 1500, 920
    left, top = 260, 190
    cell_w, cell_h = 190, 66
    grid_width = cell_w * len(CONTROLLED_STEPS)
    grid_height = cell_h * len(DISEASE_ORDER)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,sans-serif;fill:#222}</style>',
        f'<text x="{width/2}" y="55" text-anchor="middle" font-size="38" '
        'font-weight="bold">Disease-wise median bbox area across training</text>',
        f'<text x="{width/2}" y="92" text-anchor="middle" font-size="25">'
        'Controlled NTP, G=8, LR=1e-5</text>',
    ]

    for column, step in enumerate(CONTROLLED_STEPS):
        x = left + (column + 0.5) * cell_w
        svg.append(
            f'<text x="{x}" y="{top-22}" text-anchor="middle" font-size="24" '
            f'font-weight="bold">Step {step}</text>'
        )

    for row_index, disease in enumerate(DISEASE_ORDER):
        y = top + row_index * cell_h
        svg.append(
            f'<text x="{left-18}" y="{y+42}" text-anchor="end" font-size="23">'
            f'{html.escape(disease)}</text>'
        )
        for column, step in enumerate(CONTROLLED_STEPS):
            value = lookup.get((disease, step))
            x = left + column * cell_w
            if value is None or value <= 0:
                color = (238, 238, 238)
                text = "NA"
            else:
                t = (math.log10(value) - log_min) / (log_max - log_min)
                color = heat_color(t)
                text = f"{value:.6g}"
            fill = f"rgb({color[0]},{color[1]},{color[2]})"
            svg.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" '
                    f'fill="{fill}" stroke="white" stroke-width="2"/>',
                    f'<text x="{x+cell_w/2}" y="{y+41}" text-anchor="middle" '
                    f'font-size="20">{text}</text>',
                ]
            )

    bar_x, bar_y, bar_w, bar_h = left + 120, top + grid_height + 70, grid_width - 240, 24
    segments = 120
    for index in range(segments):
        t = index / (segments - 1)
        color = heat_color(t)
        svg.append(
            f'<rect x="{bar_x + index * bar_w / segments}" y="{bar_y}" '
            f'width="{bar_w / segments + 1}" height="{bar_h}" '
            f'fill="rgb({color[0]},{color[1]},{color[2]})" stroke="none"/>'
        )
    svg.extend(
        [
            f'<text x="{bar_x}" y="{bar_y+52}" text-anchor="middle" font-size="19">'
            f'{10**log_min:.3g}</text>',
            f'<text x="{bar_x+bar_w}" y="{bar_y+52}" text-anchor="middle" font-size="19">'
            f'{10**log_max:.3g}</text>',
            f'<text x="{bar_x+bar_w/2}" y="{bar_y+55}" text-anchor="middle" '
            'font-size="22" font-weight="bold">Median bbox area fraction (log-scaled color)</text>',
            "</svg>",
        ]
    )

    svg_path = ANALYSIS_DIR / f"{HEATMAP_STEM}.svg"
    png_path = ANALYSIS_DIR / f"{HEATMAP_STEM}.png"
    pdf_path = ANALYSIS_DIR / f"{HEATMAP_STEM}.pdf"
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    render_png(svg_path, png_path)

    scale = 0.5
    pdf = canvas.Canvas(str(pdf_path), pagesize=(width * scale, height * scale))
    pdf.setTitle("Disease-wise median bbox area across training")
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawCentredString(width * scale / 2, (height - 55) * scale, "Disease-wise median bbox area across training")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(width * scale / 2, (height - 92) * scale, "Controlled NTP, G=8, LR=1e-5")
    for column, step in enumerate(CONTROLLED_STEPS):
        x = (left + (column + 0.5) * cell_w) * scale
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawCentredString(x, (height - top + 22) * scale, f"Step {step}")
    for row_index, disease in enumerate(DISEASE_ORDER):
        y_svg = top + row_index * cell_h
        y = (height - y_svg - cell_h) * scale
        pdf.setFillColor(HexColor("#222222"))
        pdf.setFont("Helvetica", 11)
        pdf.drawRightString((left - 18) * scale, y + 23, disease)
        for column, step in enumerate(CONTROLLED_STEPS):
            value = lookup.get((disease, step))
            if value is None or value <= 0:
                color = (238, 238, 238)
                text = "NA"
            else:
                t = (math.log10(value) - log_min) / (log_max - log_min)
                color = heat_color(t)
                text = f"{value:.6g}"
            x = (left + column * cell_w) * scale
            pdf.setFillColor(Color(*(channel / 255 for channel in color)))
            pdf.rect(x, y, cell_w * scale, cell_h * scale, fill=1, stroke=0)
            pdf.setFillColor(HexColor("#222222"))
            pdf.setFont("Helvetica", 9)
            pdf.drawCentredString(x + cell_w * scale / 2, y + 13, text)
    for index in range(segments):
        t = index / (segments - 1)
        color = heat_color(t)
        pdf.setFillColor(Color(*(channel / 255 for channel in color)))
        pdf.rect(
            (bar_x + index * bar_w / segments) * scale,
            (height - bar_y - bar_h) * scale,
            (bar_w / segments + 1) * scale,
            bar_h * scale,
            fill=1,
            stroke=0,
        )
    pdf.setFillColor(HexColor("#222222"))
    pdf.setFont("Helvetica", 9)
    pdf.drawCentredString(bar_x * scale, (height - bar_y - 52) * scale, f"{10**log_min:.3g}")
    pdf.drawCentredString((bar_x + bar_w) * scale, (height - bar_y - 52) * scale, f"{10**log_max:.3g}")
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(
        (bar_x + bar_w / 2) * scale,
        (height - bar_y - 55) * scale,
        "Median bbox area fraction (log-scaled color)",
    )
    pdf.save()
    return png_path, pdf_path


def main() -> None:
    overall_paths = overall_plot(read_csv(OVERALL_CSV))
    heatmap_paths = heatmap(read_csv(DISEASE_CSV))
    print("Presentation figures regenerated from existing CSV metrics only.")
    for path in (*overall_paths, *heatmap_paths):
        print(path.resolve())


if __name__ == "__main__":
    main()
