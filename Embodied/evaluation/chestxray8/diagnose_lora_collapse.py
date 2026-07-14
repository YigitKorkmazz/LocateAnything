#!/usr/bin/env python3
"""Focused LoRA collapse diagnostics (no full train / no full test eval)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
CHEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CHEST_DIR))

from eaglevl.train.constants import number_tokens_list, special_tokens_list  # noqa: E402
from eval_locateanything_bbox import (  # noqa: E402
    BOX_RE,
    compute_pair_metrics,
    convert_boxes_to_pixels,
    parse_normalized_boxes,
)
from sft_common import IGNORE_INDEX, read_jsonl, set_global_seed  # noqa: E402
from train_chestxray8_sft import (  # noqa: E402
    PINNED_MODEL_REVISION,
    build_assistant_only_labels,
    load_tokenizer_and_processor,
    messages_from_pair,
    patch_lora_embedding_inplace_fix,
    patch_qwen2_pos_loss_list_bug,
    tokenize_messages,
)

OUT_DIR = REPO_ROOT / "results" / "finetuning" / "lora_diagnostics"


def tensor_checksum(t: torch.Tensor) -> str:
    x = t.detach().float().cpu().reshape(-1)
    n = min(4096, x.numel())
    s = float(x[:n].sum().item()) + float(x[:n].norm().item()) * 1e-3
    return f"sum0={s:.6f}|norm={float(t.detach().float().norm().item()):.4f}|shape={tuple(t.shape)}"


def load_base_worker(revision: str, device: str):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer
    from locateanything_worker import LocateAnythingWorker

    # Mirror LocateAnythingWorker but pin revision explicitly for diagnosis.
    worker = LocateAnythingWorker.__new__(LocateAnythingWorker)
    worker.device = device
    worker.dtype = torch.bfloat16
    worker.use_batch_runtime = False
    worker.tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/LocateAnything-3B", trust_remote_code=True, revision=revision
    )
    worker.processor = AutoProcessor.from_pretrained(
        "nvidia/LocateAnything-3B", trust_remote_code=True, revision=revision
    )
    worker.model = (
        AutoModel.from_pretrained(
            "nvidia/LocateAnything-3B",
            revision=revision,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    print(f"[load] revision={revision} tok_len={len(worker.tokenizer)}")
    return worker


def attach_adapter(worker, adapter_dir: Path, load_embeddings: bool = True):
    from peft import PeftModel
    from safetensors.torch import load_file

    lm = worker.model.language_model
    if hasattr(lm, "peft_config"):
        raise RuntimeError("adapter already attached")

    if load_embeddings:
        worker.model.language_model = PeftModel.from_pretrained(lm, str(adapter_dir))
    else:
        # Load LoRA-only tensors; keep base embeddings/lm_head untouched.
        worker.model.language_model = PeftModel.from_pretrained(lm, str(adapter_dir))
        # Restore base embeds if PEFT overwrote them — compare by reloading base keys skip.
        # Safer path: reload state and zero out non-lora after from_pretrained by
        # re-copying base embeds from a snapshot taken before load.
        pass
    worker.model.use_llm_lora = True
    worker.model.eval()
    return worker


def attach_adapter_lora_only(worker, adapter_dir: Path, base_embed, base_lm_head):
    """Attach adapter then restore base embedding / lm_head weights."""
    from peft import PeftModel

    lm = worker.model.language_model
    worker.model.language_model = PeftModel.from_pretrained(lm, str(adapter_dir))
    peft_lm = worker.model.language_model
    # Restore frozen base embeddings that PEFT overwrote from adapter file.
    with torch.no_grad():
        peft_lm.get_input_embeddings().weight.copy_(base_embed)
        out_emb = peft_lm.get_output_embeddings()
        if out_emb is not None and base_lm_head is not None:
            out_emb.weight.copy_(base_lm_head)
    worker.model.use_llm_lora = True
    worker.model.eval()
    return worker


def inspect_adapter(worker, adapter_dir: Path) -> Dict[str, Any]:
    from safetensors import safe_open

    lm = worker.model.language_model
    info: Dict[str, Any] = {"adapter_dir": str(adapter_dir)}
    if hasattr(lm, "peft_config"):
        info["active_adapters"] = list(getattr(lm, "active_adapters", []) or [])
        info["peft_config_keys"] = list(lm.peft_config.keys())
        cfg = lm.peft_config[info["peft_config_keys"][0]]
        info["target_modules"] = sorted(list(cfg.target_modules))
        info["r"] = cfg.r
        info["lora_alpha"] = cfg.lora_alpha
        info["modules_to_save"] = cfg.modules_to_save
    lora_params = [(n, p) for n, p in lm.named_parameters() if "lora_" in n.lower()]
    info["n_lora_param_tensors"] = len(lora_params)
    info["lora_param_norm_sum"] = float(
        sum(p.detach().float().norm().item() for _, p in lora_params)
    )
    info["lora_tensor_checksums"] = {n: tensor_checksum(p) for n, p in lora_params[:8]}
    # Adapter enabled?
    disabled = getattr(lm, "_disable_adapters", False) or (
        hasattr(lm, "peft_config")
        and all(getattr(c, "inference_mode", False) for c in lm.peft_config.values())
    )
    info["adapters_disabled_flag"] = bool(getattr(lm, "_disable_adapters", False))
    info["inference_mode"] = disabled
    weight_path = adapter_dir / "adapter_model.safetensors"
    with safe_open(str(weight_path), framework="pt") as f:
        keys = list(f.keys())
        other = [k for k in keys if "lora_" not in k]
        info["n_file_tensors"] = len(keys)
        info["n_file_lora"] = len(keys) - len(other)
        info["n_file_non_lora"] = len(other)
        info["non_lora_keys"] = other
        for k in other:
            info[f"file::{k}"] = tensor_checksum(f.get_tensor(k))
    info["live_embed_checksum"] = tensor_checksum(lm.get_input_embeddings().weight)
    tok = worker.tokenizer
    info["tokenizer_vocab_size"] = len(tok)
    specials = ["<ref>", "</ref>", "<box>", "</box>", "<0>", "<1>", "<999>", "<1000>", "<|im_end|>"]
    info["special_token_ids"] = {s: tok.encode(s, add_special_tokens=False) for s in specials}
    return info


def generate_one(worker, pair: Dict[str, Any], max_new_tokens: int = 128) -> str:
    image = Image.open(pair["image_path"]).convert("RGB")
    result = worker.ground_multi(
        image,
        pair["prompt_phrase"],
        generation_mode="hybrid",
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        verbose=False,
    )
    answer = result.get("answer", "")
    if isinstance(answer, tuple):
        answer = answer[0]
    return str(answer)


def analyze_example(pair: Dict[str, Any], tokenizer, processor) -> Dict[str, Any]:
    messages = messages_from_pair(pair)
    rendered = processor.py_apply_chat_template(messages, tokenize=False)
    inputs = tokenize_messages(processor, messages)
    input_ids = inputs["input_ids"][0]
    labels = build_assistant_only_labels(input_ids, tokenizer)
    supervised = (labels != IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist()
    img_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    image_pos = (input_ids == img_token_id).nonzero(as_tuple=False).flatten().tolist()
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_pos = (input_ids == im_end_id).nonzero(as_tuple=False).flatten().tolist()
    box_ids = {tokenizer.convert_tokens_to_ids(f"<{i}>") for i in range(1001)}
    first_sup = supervised[0] if supervised else None
    user_all_masked = (
        bool((labels[:first_sup] == IGNORE_INDEX).all().item()) if first_sup is not None else False
    )
    image_masked = all(int(labels[i]) == IGNORE_INDEX for i in image_pos) if image_pos else True
    pad_id = tokenizer.pad_token_id
    pad_ok = True
    if pad_id is not None:
        pad_pos = (input_ids == pad_id).nonzero(as_tuple=False).flatten().tolist()
        pad_ok = all(int(labels[i]) == IGNORE_INDEX for i in pad_pos)

    # Alignment check: labels[i] should equal input_ids[i] on supervised positions
    # (causal LM will shift internally).
    aligned = all(int(labels[i]) == int(input_ids[i]) for i in supervised)

    decoded_sup = tokenizer.decode([int(labels[i]) for i in supervised], skip_special_tokens=False)
    # First supervised token should be start of assistant target (usually <ref>)
    first_tok = tokenizer.decode([int(input_ids[first_sup])]) if first_sup is not None else None

    return {
        "image_index": pair["image_index"],
        "disease": pair["disease"],
        "user_prompt": pair["user_query"],
        "assistant_target": pair["assistant_target"],
        "rendered_chat_template": rendered,
        "n_tokens": int(input_ids.numel()),
        "n_supervised": len(supervised),
        "supervised_indices": supervised,
        "first_supervised_index": first_sup,
        "first_supervised_token": first_tok,
        "decoded_supervised": decoded_sup,
        "decoded_input_ids": tokenizer.decode(input_ids.tolist(), skip_special_tokens=False),
        "image_token_positions_n": len(image_pos),
        "image_tokens_masked": image_masked,
        "user_prefix_all_masked": user_all_masked,
        "pad_tokens_masked": pad_ok,
        "labels_equal_input_on_supervised": aligned,
        "im_end_positions": eos_pos,
        "n_coord_tokens_in_supervised": sum(1 for i in supervised if int(input_ids[i]) in box_ids),
        "input_ids": input_ids.tolist(),
        "labels": labels.tolist(),
    }


def special_token_report(tokenizer) -> Dict[str, Any]:
    tests = ["<ref>", "</ref>", "<box>", "</box>", "<0>", "<1>", "<999>", "<1000>", "<|im_end|>"]
    out: Dict[str, Any] = {}
    vocab = tokenizer.get_vocab()
    for t in tests:
        ids = tokenizer.encode(t, add_special_tokens=False)
        kind = (
            "one_token"
            if len(ids) == 1 and t in vocab
            else ("multiple_tokens" if len(ids) > 1 else "unknown_or_absent")
        )
        out[t] = {
            "ids": ids,
            "n_tokens": len(ids),
            "in_vocab": t in vocab,
            "kind": kind,
            "decoded": tokenizer.decode(ids, skip_special_tokens=False),
        }
    n_added = tokenizer.add_tokens(special_tokens_list + number_tokens_list, special_tokens=True)
    out["_add_tokens_would_add"] = int(n_added)
    out["vocab_size"] = len(tokenizer)
    return out


def teacher_force_stats(model, processor, tokenizer, pair, device, dtype) -> Dict[str, Any]:
    messages = messages_from_pair(pair)
    inputs = tokenize_messages(processor, messages)
    input_ids = inputs["input_ids"].to(device)
    labels = build_assistant_only_labels(input_ids[0], tokenizer).unsqueeze(0).to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_hws = torch.as_tensor(inputs["image_grid_hws"], device=device)
    image_flags = torch.tensor([len(inputs["image_grid_hws"])], device=device)
    attention_mask = torch.ones_like(input_ids)

    model.eval()
    patch_lora_embedding_inplace_fix(model)
    patch_qwen2_pos_loss_list_bug(model)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_hws=image_grid_hws,
            image_flags=image_flags,
            labels=None,
            use_cache=False,
        )
        # LocateAnything may return (CausalLMOutput, pos_loss_list)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    pred = shift_logits.argmax(dim=-1)
    mask = shift_labels != IGNORE_INDEX
    n = int(mask.sum().item())
    acc = float(((pred == shift_labels) & mask).sum().item()) / n if n else 0.0

    # CE loss on supervised tokens only
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loss = loss_fct(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
    )

    ref_id = tokenizer.convert_tokens_to_ids("<ref>")
    ref_end = tokenizer.convert_tokens_to_ids("</ref>")
    box_id = tokenizer.convert_tokens_to_ids("<box>")
    box_end = tokenizer.convert_tokens_to_ids("</box>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    coord_ids = {tokenizer.convert_tokens_to_ids(f"<{i}>") for i in range(1001)}
    structural = {ref_id, ref_end, box_id, box_end}

    def cat_acc(id_set):
        m = mask.clone()
        m_ids = torch.zeros_like(mask)
        for tid in id_set:
            m_ids |= shift_labels == tid
        m = mask & m_ids
        nn = int(m.sum().item())
        if nn == 0:
            return None, 0
        return float(((pred == shift_labels) & m).sum().item()) / nn, nn

    # reference-text = supervised non-structural non-coord non-eos
    special_all = structural | coord_ids | {im_end}
    m_text = mask.clone()
    for tid in special_all:
        m_text &= shift_labels != tid
    n_text = int(m_text.sum().item())
    text_acc = (
        float(((pred == shift_labels) & m_text).sum().item()) / n_text if n_text else None
    )

    report: Dict[str, Any] = {
        "image_index": pair["image_index"],
        "disease": pair["disease"],
        "loss": float(loss.detach().cpu()),
        "token_exact_acc": acc,
        "n_supervised_shifted": n,
        "structural_acc": cat_acc(structural)[0],
        "n_structural": cat_acc(structural)[1],
        "coord_acc": cat_acc(coord_ids)[0],
        "n_coord": cat_acc(coord_ids)[1],
        "closing_acc": cat_acc({im_end, box_end, ref_end})[0],
        "ref_text_acc": text_acc,
        "n_ref_text": n_text,
    }
    supervised_pos = (labels[0] != IGNORE_INDEX).nonzero(as_tuple=False).flatten()
    if len(supervised_pos):
        sp = int(supervised_pos[0].item())
        pred_pos = sp - 1
        if pred_pos >= 0:
            probs = torch.softmax(shift_logits[0, pred_pos].float(), dim=-1)
            topk = torch.topk(probs, k=5)
            report["first_target_token_str"] = tokenizer.decode([int(labels[0, sp].item())])
            report["top5_at_first_target"] = [
                {"id": int(i), "str": tokenizer.decode([int(i)]), "prob": float(p)}
                for i, p in zip(topk.indices.tolist(), topk.values.tolist())
            ]
    return report


def eval_small(worker, pairs: Sequence[Dict[str, Any]], tag: str) -> Dict[str, Any]:
    rows = []
    n_valid = 0
    n_parsed = 0
    ious: List[float] = []
    for p in pairs:
        raw = generate_one(worker, p)
        boxes_norm = parse_normalized_boxes(raw)
        boxes_px = convert_boxes_to_pixels(boxes_norm, p["image_width"], p["image_height"])
        metrics = compute_pair_metrics(p["gt_boxes_xyxy_px"], boxes_px)
        valid = bool(BOX_RE.search(raw)) or ("<box>" in raw and "</box>" in raw)
        if valid:
            n_valid += 1
        if boxes_norm:
            n_parsed += 1
        ious.extend(metrics["gt_ious"])
        rows.append(
            {
                "tag": tag,
                "image_index": p["image_index"],
                "disease": p["disease"],
                "raw": raw,
                "n_parsed_boxes": len(boxes_norm),
                "valid_structure": valid,
                "mean_matched_iou": metrics["mean_matched_iou"],
            }
        )
    return {
        "tag": tag,
        "n": len(pairs),
        "valid_output_rate": n_valid / len(pairs) if pairs else 0.0,
        "parsed_box_rate": n_parsed / len(pairs) if pairs else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "samples": rows,
    }


def summarize_loss_history(path: Path) -> Dict[str, Any]:
    hist = json.loads(path.read_text())
    losses = [h["loss"] for h in hist]
    amin = int(np.argmin(losses))
    save_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450]
    save_losses = {}
    for s in save_steps:
        rows = [h for h in hist if h["step"] == s]
        if rows:
            save_losses[s] = rows[0]["loss"]
    by_epoch: Dict[Any, List[float]] = defaultdict(list)
    for h in hist:
        by_epoch[h["epoch"]].append(h["loss"])
    # Detect collapse
    collapse_step = None
    for i in range(1, len(hist)):
        if hist[i]["loss"] > 10 and hist[i - 1]["loss"] < 2:
            collapse_step = hist[i]["step"]
            break
    return {
        "n_steps": len(hist),
        "min_loss": float(min(losses)),
        "min_loss_step": hist[amin]["step"],
        "final_loss": float(losses[-1]),
        "avg_loss_per_epoch": {str(k): float(np.mean(v)) for k, v in sorted(by_epoch.items())},
        "loss_at_save_steps": save_losses,
        "collapse_step": collapse_step,
        "best_save_step_by_train_loss": int(min(save_losses, key=save_losses.get))
        if save_losses
        else None,
        "validation_evaluated": False,
        "checkpoint_450_label": "final checkpoint (selected among save-step train losses; NO validation)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", default=PINNED_MODEL_REVISION)
    parser.add_argument(
        "--adapter",
        default=str(REPO_ROOT / "results/finetuning/lora/checkpoint-450/adapter"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoints", default="50,100,200,300,450")
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()
    set_global_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs = read_jsonl(CHEST_DIR / "splits" / "train_pairs_seed42.jsonl")
    test_pairs = read_jsonl(CHEST_DIR / "splits" / "test_pairs_seed42.jsonl")
    train5 = train_pairs[:5]
    test5 = test_pairs[:5]
    report: Dict[str, Any] = {"revision": args.revision, "adapter": args.adapter}

    report["loss_history"] = summarize_loss_history(
        REPO_ROOT / "results/finetuning/lora/loss_history.json"
    )
    print("=== LOSS HISTORY ===")
    print(json.dumps(report["loss_history"], indent=2))

    tokenizer, processor = load_tokenizer_and_processor(
        "nvidia/LocateAnything-3B", revision=args.revision, max_seq_length=4096
    )
    report["special_token_report"] = special_token_report(tokenizer)
    print("\n=== SPECIAL TOKENS ===")
    print(json.dumps(report["special_token_report"], indent=2))

    example_reports = [analyze_example(p, tokenizer, processor) for p in train5]
    # Compact print; full dump to disk
    compact = []
    for er in example_reports:
        compact.append(
            {
                k: er[k]
                for k in (
                    "image_index",
                    "disease",
                    "user_prompt",
                    "assistant_target",
                    "n_tokens",
                    "n_supervised",
                    "first_supervised_index",
                    "first_supervised_token",
                    "decoded_supervised",
                    "image_token_positions_n",
                    "image_tokens_masked",
                    "user_prefix_all_masked",
                    "pad_tokens_masked",
                    "labels_equal_input_on_supervised",
                    "im_end_positions",
                    "n_coord_tokens_in_supervised",
                )
            }
        )
        compact[-1]["rendered_preview"] = er["rendered_chat_template"][:400]
    report["training_examples_compact"] = compact
    (OUT_DIR / "training_examples_full.json").write_text(
        json.dumps(example_reports, indent=2, ensure_ascii=False) + "\n"
    )
    print("\n=== MASKING / TARGET FORMAT ===")
    for c in compact:
        print(
            f"{c['image_index']} {c['disease']}: supervised={c['n_supervised']} "
            f"user_masked={c['user_prefix_all_masked']} img_masked={c['image_tokens_masked']} "
            f"aligned={c['labels_equal_input_on_supervised']} first={c['first_supervised_token']!r}"
        )
        print(f"  target={c['assistant_target']!r}")
        print(f"  decoded_sup={c['decoded_supervised']!r}")

    print("\n=== LOAD BASE ===")
    worker = load_base_worker(args.revision, args.device)
    # Snapshot base embeds on CPU before adapter load (avoid GPU OOM)
    base_embed = (
        worker.model.language_model.get_input_embeddings().weight.detach().float().cpu().clone()
    )
    base_lm_head = (
        worker.model.language_model.get_output_embeddings().weight.detach().float().cpu().clone()
    )
    report["base_embed_checksum"] = tensor_checksum(base_embed)
    report["base_tokenizer_vocab_size"] = len(worker.tokenizer)
    report["train_tokenizer_vocab_size"] = len(tokenizer)
    report["generation_config"] = {
        "generation_mode": "hybrid",
        "max_new_tokens": 128,
        "temperature": 0.0,
        "same_for_base_and_lora": True,
    }

    probe = test5[0]
    out_A = generate_one(worker, probe)
    print("\n[A] base no-adapter:", repr(out_A[:300]))

    # Attach with embedding overwrite (eval path as-is)
    from peft import PeftModel

    worker.model.language_model = PeftModel.from_pretrained(
        worker.model.language_model, str(args.adapter)
    )
    worker.model.use_llm_lora = True
    worker.model.eval()
    adapter_info = inspect_adapter(worker, Path(args.adapter))
    report["adapter_inspect"] = {
        k: v for k, v in adapter_info.items() if not str(k).startswith("file::")
    }
    report["adapter_file_checksums"] = {
        k: v for k, v in adapter_info.items() if str(k).startswith("file::")
    }
    print("\n=== ADAPTER ===")
    print(json.dumps(report["adapter_inspect"], indent=2, default=str)[:2500])

    # Compare overwritten embeds vs base (CPU to avoid OOM on A4000)
    live_embed = worker.model.language_model.get_input_embeddings().weight.detach().float().cpu()
    embed_delta = float((live_embed - base_embed).norm().item())
    report["embed_delta_norm_vs_base"] = embed_delta
    print(f"embed delta norm vs base after adapter load: {embed_delta:.4f}")
    del live_embed
    torch.cuda.empty_cache()

    out_B = generate_one(worker, probe)
    print("\n[B] adapter ON (with file embeds):", repr(out_B[:300]))

    # Disable adapter layers
    lm = worker.model.language_model
    disable_mode = "none"
    if hasattr(lm, "disable_adapter_layers"):
        lm.disable_adapter_layers()
        disable_mode = "disable_adapter_layers"
    out_C = generate_one(worker, probe)
    print(f"\n[C] adapter OFF via {disable_mode}:", repr(out_C[:300]))

    if hasattr(lm, "enable_adapter_layers"):
        lm.enable_adapter_layers()

    # Also test LoRA-only (restore base embeds)
    with torch.no_grad():
        lm.get_input_embeddings().weight.copy_(
            base_embed.to(device=lm.get_input_embeddings().weight.device, dtype=lm.get_input_embeddings().weight.dtype)
        )
        if lm.get_output_embeddings() is not None:
            lm.get_output_embeddings().weight.copy_(
                base_lm_head.to(
                    device=lm.get_output_embeddings().weight.device,
                    dtype=lm.get_output_embeddings().weight.dtype,
                )
            )
    torch.cuda.empty_cache()
    out_B2 = generate_one(worker, probe)
    print("\n[B2] adapter ON + restored base embeds:", repr(out_B2[:300]))

    report["adapter_on_off"] = {
        "A_base": out_A,
        "B_adapter_on_with_file_embeds": out_B,
        "B2_adapter_on_base_embeds": out_B2,
        "C_adapter_off": out_C,
        "disable_mode": disable_mode,
        "A_C_prefix_match": out_A[:80] == out_C[:80],
        "embed_delta_norm_vs_base": embed_delta,
    }

    print("\n=== TEACHER FORCE train5 @450 ===")
    dtype = next(worker.model.parameters()).dtype
    device = next(worker.model.parameters()).device
    tf_reports = []
    for p in train5:
        try:
            tf_reports.append(
                teacher_force_stats(worker.model, processor, tokenizer, p, device, dtype)
            )
        except Exception as e:
            tf_reports.append({"image_index": p["image_index"], "error": repr(e)})
    report["teacher_force_train5"] = tf_reports
    print(json.dumps(tf_reports, indent=2, default=str)[:3000])

    print("\n=== GEN train5/test5 @450 ===")
    report["gen_train5_ckpt450"] = eval_small(worker, train5, "train5@450")
    report["gen_test5_ckpt450"] = eval_small(worker, test5, "test5@450")
    for s in report["gen_train5_ckpt450"]["samples"]:
        print("TRAIN", s["image_index"], repr(s["raw"][:140]))
    for s in report["gen_test5_ckpt450"]["samples"]:
        print("TEST", s["image_index"], repr(s["raw"][:140]))

    if not args.skip_sweep:
        print("\n=== CHECKPOINT SWEEP (in-place LoRA weight swap) ===")
        from safetensors.torch import load_file

        steps = [int(x) for x in args.checkpoints.split(",") if x.strip()]
        sweep = []
        if hasattr(worker.model.language_model, "enable_adapter_layers"):
            worker.model.language_model.enable_adapter_layers()
        name_to_param = {
            n: p for n, p in worker.model.language_model.named_parameters() if "lora_" in n
        }
        for step in steps:
            adapter = REPO_ROOT / f"results/finetuning/lora/checkpoint-{step}/adapter"
            weight_path = adapter / "adapter_model.safetensors"
            if not weight_path.is_file():
                continue
            print(f"\n--- checkpoint-{step} ---")
            state = load_file(str(weight_path), device="cpu")
            n_loaded = 0
            with torch.no_grad():
                for k, v in state.items():
                    if "lora_" not in k:
                        continue
                    p = name_to_param.get(k)
                    if p is None and k.endswith(".weight"):
                        # File keys omit PEFT adapter name: lora_A.weight vs lora_A.default.weight
                        alt = k[: -len(".weight")] + ".default.weight"
                        p = name_to_param.get(alt)
                    if p is None:
                        continue
                    p.copy_(v.to(device=p.device, dtype=p.dtype))
                    n_loaded += 1
                emb_key = "base_model.model.model.embed_tokens.weight"
                lm_key = "base_model.model.lm_head.weight"
                if emb_key in state:
                    emb = worker.model.language_model.get_input_embeddings().weight
                    emb.copy_(state[emb_key].to(device=emb.device, dtype=emb.dtype))
                if lm_key in state:
                    out = worker.model.language_model.get_output_embeddings()
                    if out is not None:
                        out.weight.copy_(
                            state[lm_key].to(device=out.weight.device, dtype=out.weight.dtype)
                        )
            del state
            torch.cuda.empty_cache()
            print(f"  loaded {n_loaded} LoRA tensors")
            tr = eval_small(worker, train5, f"train5@{step}")
            te = eval_small(worker, test5, f"test5@{step}")
            entry = {
                "step": step,
                "n_lora_loaded": n_loaded,
                "train5": {
                    k: tr[k] for k in ("valid_output_rate", "parsed_box_rate", "mean_iou")
                },
                "test5": {
                    k: te[k] for k in ("valid_output_rate", "parsed_box_rate", "mean_iou")
                },
                "train_raw": [s["raw"][:200] for s in tr["samples"]],
                "test_raw": [s["raw"][:200] for s in te["samples"]],
            }
            sweep.append(entry)
            print("train5", entry["train5"], "test5", entry["test5"])
            print(" train0:", repr(entry["train_raw"][0]))
            print(" test0:", repr(entry["test_raw"][0]))
        report["checkpoint_sweep"] = sweep

    out_path = OUT_DIR / "diagnostic_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
