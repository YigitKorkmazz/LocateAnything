#!/usr/bin/env python3
"""
Inference-debugging stage for ChestX-ray8 LocateAnything evaluation.

Does NOT run the 100-sample / full benchmark. Exercises native worker APIs,
dumps the full prompt/generation path, and runs a natural-image positive control.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from locateanything_worker import LocateAnythingWorker  # noqa: E402
from evaluation.chestxray8.eval_locateanything_bbox import (  # noqa: E402
    box_iou,
    compute_pair_metrics,
    convert_boxes_to_pixels,
    draw_visualization,
    parse_normalized_boxes,
)

DATASET_ROOT = Path("/auto/data2/ykorkmaz/nih-chest-xrays/data/versions/3")
BBOX_CSV = DATASET_ROOT / "BBox_List_2017.csv"
OUT_DIR = REPO_ROOT / "results" / "debug"
POSITIVE_CONTROL = OUT_DIR / "positive_control.jpg"
MODEL_PATH = "nvidia/LocateAnything-3B"

REF_RE = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
BOX_NUM_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
BOX_NONE_RE = re.compile(r"<box>\s*None\s*</box>", re.IGNORECASE)


def find_image(name: str) -> Path:
    for folder in sorted(DATASET_ROOT.glob("images_*")):
        candidate = folder / "images" / name
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(name)


def load_pneumonia_sample() -> Tuple[str, Path, List[List[float]]]:
    """Load one Pneumonia annotation from BBox_List_2017.csv only."""
    with BBOX_CSV.open(newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 6 or row[1].strip() != "Pneumonia":
                continue
            name = row[0].strip()
            x, y, w, h = map(float, row[2:6])
            gt = [[x, y, x + w, y + h]]
            return name, find_image(name), gt
    raise RuntimeError("No Pneumonia annotation found in BBox_List_2017.csv")


def parse_refs_and_boxes(answer: str) -> Dict[str, Any]:
    refs = REF_RE.findall(answer or "")
    num_boxes = [[float(g) for g in m.groups()] for m in BOX_NUM_RE.finditer(answer or "")]
    has_none = bool(BOX_NONE_RE.search(answer or ""))
    return {
        "references": refs,
        "normalized_boxes": num_boxes,
        "has_box_none": has_none,
    }


def instrumented_predict(
    worker: LocateAnythingWorker,
    image: Image.Image,
    question: str,
    *,
    generation_mode: str = "hybrid",
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.1,
) -> Dict[str, Any]:
    """
    Mirror LocateAnythingWorker._predict_standard, but capture:
    chat template text, generation config, and raw generated token IDs.
    """
    messages = worker._build_messages(image=image, question=question)
    templated = worker.processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = worker.processor.process_vision_info(messages)
    inputs = worker.processor(
        text=[templated], images=images, videos=videos, return_tensors="pt"
    ).to(worker.device)

    pixel_values = inputs["pixel_values"].to(worker.dtype)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs.get("image_grid_hws", None)
    top_k_for_generate = None if top_k <= 0 else top_k

    gen_cfg = {
        "generation_mode": generation_mode,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": True,
        "top_p": top_p,
        "top_k": top_k_for_generate,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }

    captured: Dict[str, Any] = {}
    original_generate = worker.model.generate

    def wrapped_generate(*args, **kwargs):
        # Call original and also recover token ids by re-running decode path
        # The HF remote generate returns a decoded string. We intercept by
        # temporarily patching tokenizer.batch_decode to capture ids.
        tok = kwargs.get("tokenizer", worker.tokenizer)
        orig_batch_decode = tok.batch_decode
        token_ids_holder: Dict[str, Any] = {}

        def capturing_batch_decode(generated_ids, *a, **kw):
            # generated_ids are already sliced to new tokens inside model.generate
            if torch.is_tensor(generated_ids):
                token_ids_holder["generated_token_ids"] = generated_ids.detach().cpu().tolist()
            else:
                token_ids_holder["generated_token_ids"] = generated_ids
            return orig_batch_decode(generated_ids, *a, **kw)

        tok.batch_decode = capturing_batch_decode  # type: ignore[method-assign]
        try:
            out = original_generate(*args, **kwargs)
        finally:
            tok.batch_decode = orig_batch_decode  # type: ignore[method-assign]
        captured.update(token_ids_holder)
        return out

    worker.model.generate = wrapped_generate  # type: ignore[method-assign]
    t0 = time.perf_counter()
    try:
        with torch.inference_mode():
            response = worker.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=worker.tokenizer,
                verbose=False,
                **gen_cfg,
            )
    finally:
        worker.model.generate = original_generate  # type: ignore[method-assign]
    elapsed = time.perf_counter() - t0

    answer = response[0] if isinstance(response, tuple) else response
    answer = str(answer)
    parsed = parse_refs_and_boxes(answer)
    w, h = image.size
    pixel_boxes = convert_boxes_to_pixels(parsed["normalized_boxes"], w, h)

    # Human-readable view of templated prompt with image placeholders preserved
    templated_printable = templated
    # Keep only a compact form of messages text content
    user_text_parts = []
    for msg in messages:
        for content in msg.get("content", []):
            if isinstance(content, dict) and content.get("type") == "text":
                user_text_parts.append(content["text"])

    return {
        "question_arg": question,
        "messages_text_parts": user_text_parts,
        "templated_prompt": templated_printable,
        "generation_config": gen_cfg,
        "input_id_length": int(input_ids.shape[-1]),
        "raw_generated_token_ids": captured.get("generated_token_ids"),
        "decoded_raw_output": answer,
        "parsed_references": parsed["references"],
        "parsed_normalized_boxes": parsed["normalized_boxes"],
        "has_box_none": parsed["has_box_none"],
        "predicted_pixel_boxes": pixel_boxes,
        "inference_time_sec": elapsed,
        "image_size": [w, h],
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report_mode(
    name: str,
    api_call: str,
    api_args: Dict[str, Any],
    task_mode: str,
    raw_user_query_before_template: str,
    dbg: Dict[str, Any],
    gt_boxes: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    print_section(name)
    print(f"Python function being called: {api_call}")
    print(f"function arguments: {json.dumps(api_args, indent=2, default=str)}")
    print(f"task identifier / mode: {task_mode}")
    print(f"raw user query before templating:\n  {raw_user_query_before_template}")
    print(f"final prompt after chat/task template:\n{dbg['templated_prompt']}")
    print(f"model generation configuration:\n{json.dumps(dbg['generation_config'], indent=2)}")
    print(f"input_ids length: {dbg['input_id_length']}")
    print(f"raw generated token IDs:\n{dbg['raw_generated_token_ids']}")
    print(f"decoded raw output:\n{dbg['decoded_raw_output']}")
    print(f"parsed references: {dbg['parsed_references']}")
    print(f"parsed boxes [0,1000]: {dbg['parsed_normalized_boxes']}")
    print(f"has <box>None</box>: {dbg['has_box_none']}")
    print(f"converted pixel boxes: {dbg['predicted_pixel_boxes']}")

    metrics = None
    if gt_boxes is not None:
        metrics = compute_pair_metrics(gt_boxes, dbg["predicted_pixel_boxes"])
        print(f"GT boxes: {gt_boxes}")
        print(f"IoU matrix: {metrics['iou_matrix']}")
        print(f"matches: {metrics['matches']}")
        print(
            f"Best IoU: {metrics['maximum_iou']:.4f} | "
            f"Mean matched IoU: {metrics['mean_matched_iou']:.4f} | "
            f"R@0.5: {metrics['recall_0_5']:.4f}"
        )

    return {
        "name": name,
        "api_call": api_call,
        "api_args": api_args,
        "task_mode": task_mode,
        "raw_user_query_before_template": raw_user_query_before_template,
        "final_prompt": dbg["templated_prompt"],
        "generation_config": dbg["generation_config"],
        "raw_generated_token_ids": dbg["raw_generated_token_ids"],
        "decoded_raw_output": dbg["decoded_raw_output"],
        "parsed_references": dbg["parsed_references"],
        "parsed_normalized_boxes": dbg["parsed_normalized_boxes"],
        "predicted_pixel_boxes": dbg["predicted_pixel_boxes"],
        "metrics": metrics,
        "inference_time_sec": dbg["inference_time_sec"],
    }


def save_debug_vis(
    image: Image.Image,
    label: str,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    raw_answer: str,
    out_path: Path,
) -> None:
    metrics = compute_pair_metrics(gt_boxes, pred_boxes)
    draw_visualization(
        image,
        label,
        gt_boxes,
        pred_boxes,
        metrics["gt_ious"],
        metrics["matches"],
        out_path,
        raw_answer=raw_answer,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_section("ENVIRONMENT")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"device={device}")
    if device == "cuda":
        print(f"visible GPU: {torch.cuda.get_device_name(0)}")
        print(f"device_count={torch.cuda.device_count()}")

    print_section("NATIVE API MAP (from locateanything_worker.py)")
    print(
        "detect(image, categories: list[str])\n"
        "  -> prompt: 'Locate all the instances that matches the following description: {a}</c>{b}.'\n"
        "  note: uses 'matches' (with s); categories joined by </c>\n\n"
        "ground_single(image, phrase: str)\n"
        "  -> prompt: 'Locate a single instance that matches the following description: {phrase}.'\n\n"
        "ground_multi(image, phrase: str)\n"
        "  -> prompt: 'Locate all the instances that match the following description: {phrase}.'\n"
        "  note: uses 'match' (no s); this is multi-box phrase grounding\n\n"
        "All of the above call predict(image, question=constructed_prompt).\n"
        "predict() does NOT add another task instruction; it only applies the chat template."
    )

    print_section("CURRENT EVAL PATH TRACE")
    print(
        "eval_locateanything_bbox.run_inference()\n"
        "  -> builds PROMPT_TEMPLATE = "
        "'Locate all the instances that match the following description: {disease}.'\n"
        "  -> LocateAnythingWorker.predict(image, prompt)\n"
        "  -> _predict_standard()\n"
        "  -> processor.py_apply_chat_template(messages)\n"
        "  -> model.generate(..., generation_mode='hybrid')\n\n"
        "DOUBLE-TEMPLATE CHECK:\n"
        "  Current eval passes the COMPLETE ground_multi instruction into predict().\n"
        "  It does NOT call ground_multi(image, full_instruction), which would double-wrap.\n"
        "  Equivalent native call: worker.ground_multi(image, disease)."
    )

    pneumonia_name, pneumonia_path, gt_boxes = load_pneumonia_sample()
    pneumonia_img = Image.open(pneumonia_path).convert("RGB")
    print_section("PNEUMONIA SAMPLE (GT from BBox_List_2017.csv only)")
    print(f"image: {pneumonia_name}")
    print(f"path: {pneumonia_path}")
    print(f"size: {pneumonia_img.size}")
    print(f"GT xyxy: {gt_boxes}")

    print("Loading model once...")
    worker = LocateAnythingWorker(MODEL_PATH, device=device)
    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Detailed dump for one call (Test A path mirrored through detect)
    # ------------------------------------------------------------------
    # Test A — object/category detection
    cats = ["Pneumonia"]
    question_a = (
        "Locate all the instances that matches the following description: "
        + "</c>".join(cats)
        + "."
    )
    dbg_a = instrumented_predict(worker, pneumonia_img, question_a)
    results.append(
        report_mode(
            "TEST A — object/category detection via detect()",
            api_call="LocateAnythingWorker.detect",
            api_args={"image": str(pneumonia_path), "categories": cats},
            task_mode="detect / category detection (note: 'matches' grammar)",
            raw_user_query_before_template=question_a,
            dbg=dbg_a,
            gt_boxes=gt_boxes,
        )
    )
    # Sanity: native detect() should produce the same question string
    assert (
        f"Locate all the instances that matches the following description: {cats[0]}."
        == question_a
    )

    # Test B — phrase grounding
    phrase_b = "pneumonia"
    question_b = f"Locate all the instances that match the following description: {phrase_b}."
    dbg_b = instrumented_predict(worker, pneumonia_img, question_b)
    results.append(
        report_mode(
            "TEST B — phrase grounding via ground_multi()",
            api_call="LocateAnythingWorker.ground_multi",
            api_args={"image": str(pneumonia_path), "phrase": phrase_b},
            task_mode="ground_multi / multi-box phrase grounding (note: 'match' grammar)",
            raw_user_query_before_template=question_b,
            dbg=dbg_b,
            gt_boxes=gt_boxes,
        )
    )

    # Test C — radiology-context phrase grounding
    phrase_c = "region showing pneumonia in the chest radiograph"
    question_c = f"Locate all the instances that match the following description: {phrase_c}."
    dbg_c = instrumented_predict(worker, pneumonia_img, question_c)
    results.append(
        report_mode(
            "TEST C — radiology-context phrase grounding via ground_multi()",
            api_call="LocateAnythingWorker.ground_multi",
            api_args={"image": str(pneumonia_path), "phrase": phrase_c},
            task_mode="ground_multi / multi-box phrase grounding",
            raw_user_query_before_template=question_c,
            dbg=dbg_c,
            gt_boxes=gt_boxes,
        )
    )

    for tag, dbg in [("A", dbg_a), ("B", dbg_b), ("C", dbg_c)]:
        save_debug_vis(
            pneumonia_img,
            f"Pneumonia/Test{tag}",
            gt_boxes,
            dbg["predicted_pixel_boxes"],
            dbg["decoded_raw_output"],
            OUT_DIR / f"pneumonia_test_{tag}.png",
        )

    # ------------------------------------------------------------------
    # Positive control on a natural image
    # ------------------------------------------------------------------
    if not POSITIVE_CONTROL.is_file():
        raise FileNotFoundError(
            f"Missing positive-control image: {POSITIVE_CONTROL}. "
            "Download a natural image first."
        )
    control_img = Image.open(POSITIVE_CONTROL).convert("RGB")
    print_section("POSITIVE CONTROL (natural image)")
    print(f"image: {POSITIVE_CONTROL} size={control_img.size}")
    print("Native API: detect(image, ['cat', 'couch'])  # COCO 000000039769.jpg")

    control_cats = ["cat", "couch"]
    question_pc = (
        "Locate all the instances that matches the following description: "
        + "</c>".join(control_cats)
        + "."
    )
    dbg_pc = instrumented_predict(worker, control_img, question_pc)
    results.append(
        report_mode(
            "POSITIVE CONTROL — natural image detect(['cat','couch'])",
            api_call="LocateAnythingWorker.detect",
            api_args={"image": str(POSITIVE_CONTROL), "categories": control_cats},
            task_mode="detect / category detection",
            raw_user_query_before_template=question_pc,
            dbg=dbg_pc,
            gt_boxes=None,
        )
    )

    # Also try ground_multi on a clear phrase
    phrase_pc2 = "cat"
    question_pc2 = f"Locate all the instances that match the following description: {phrase_pc2}."
    dbg_pc2 = instrumented_predict(worker, control_img, question_pc2)
    results.append(
        report_mode(
            "POSITIVE CONTROL — natural image ground_multi('cat')",
            api_call="LocateAnythingWorker.ground_multi",
            api_args={"image": str(POSITIVE_CONTROL), "phrase": phrase_pc2},
            task_mode="ground_multi / phrase grounding",
            raw_user_query_before_template=question_pc2,
            dbg=dbg_pc2,
            gt_boxes=None,
        )
    )

    # Draw positive-control preds (no GT)
    pc_vis = control_img.copy()
    draw = ImageDraw.Draw(pc_vis)
    for box in dbg_pc["predicted_pixel_boxes"]:
        draw.rectangle(box, outline=(255, 80, 80), width=3)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text(
        (8, 8),
        f"Predicted boxes: {len(dbg_pc['predicted_pixel_boxes'])}",
        fill=(255, 255, 0),
        font=font,
    )
    pc_out_label = (
        "None"
        if dbg_pc["has_box_none"]
        else f"{len(dbg_pc['predicted_pixel_boxes'])} box(es)"
    )
    draw.text((8, 26), f"Model output: {pc_out_label}", fill=(255, 255, 0), font=font)
    pc_vis.save(OUT_DIR / "positive_control_detect.png")

    # ------------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------------
    def has_boxes(dbg: Dict[str, Any]) -> bool:
        return len(dbg["predicted_pixel_boxes"]) > 0

    med_any = any(has_boxes(d) for d in (dbg_a, dbg_b, dbg_c))
    ctrl_any = any(has_boxes(d) for d in (dbg_pc, dbg_pc2))

    print_section("DIAGNOSTIC INTERPRETATION")
    print(f"Chest X-ray modes with boxes: "
          f"A={has_boxes(dbg_a)} B={has_boxes(dbg_b)} C={has_boxes(dbg_c)}")
    print(f"Natural-image control with boxes: "
          f"detect={has_boxes(dbg_pc)} ground_multi={has_boxes(dbg_pc2)}")

    if not ctrl_any:
        verdict = (
            "FAIL: natural-image positive control also returned no boxes. "
            "Model loading, prompt/task configuration, or inference call is likely wrong."
        )
        recommended_mode = None
    elif not med_any and ctrl_any:
        verdict = (
            "PASS (pipeline): natural-image control returned boxes, but all chest X-ray "
            "modes returned None. Inference path appears correct; LocateAnything is failing "
            "zero-shot on the medical domain."
        )
        recommended_mode = "ground_multi (current eval equivalent) — no medical mode succeeded"
    elif med_any:
        winners = []
        if has_boxes(dbg_a):
            winners.append("detect")
        if has_boxes(dbg_b) or has_boxes(dbg_c):
            winners.append("ground_multi")
        recommended_mode = ", ".join(winners)
        verdict = (
            f"PARTIAL: at least one chest X-ray task mode returned boxes ({recommended_mode}). "
            "Prefer the successful native mode for ChestX-ray8 evaluation."
        )
    else:
        verdict = "Unexpected state."
        recommended_mode = None

    print(verdict)
    print(f"Recommended mode for eval: {recommended_mode}")

    summary = {
        "pneumonia_image": pneumonia_name,
        "pneumonia_path": str(pneumonia_path),
        "gt_boxes": gt_boxes,
        "verdict": verdict,
        "recommended_mode": recommended_mode,
        "tests": results,
        "double_template_check": (
            "Eval uses predict(full ground_multi instruction). "
            "This is NOT double-templating. Equivalent to ground_multi(image, disease)."
        ),
    }
    out_json = OUT_DIR / "inference_debug_report.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_json}")
    print("STOPPING: not launching 100-sample or full evaluation.")


if __name__ == "__main__":
    main()
