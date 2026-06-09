#!/usr/bin/env python3
"""Train object-token aggregation on real Qwen3-VL visual tokens.

The script loads only Qwen3-VL's vision tower, extracts merged visual tokens
(`pooler_output`, the 4096-d tokens fed to the language model), and trains a
small supervised head from detection boxes. It does not use tracking IDs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "sha_ave_nwp"
DEFAULT_MODEL = (
    Path(__file__).resolve().parents[3]
    / "yong_task"
    / "token_pruner_merge"
    / "models"
    / "Qwen3-VL-8B-Instruct"
)
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "outputs" / "qwen3vl_visual_token_binding"

PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


@dataclass
class FrameRecord:
    dataset: str
    video_id: str
    frame_idx: int
    image_path: Path
    width: int
    height: int
    boxes: list[dict]


class TokenHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.obj_head = nn.Linear(256, 1)
        self.cls_head = nn.Linear(256, num_classes)
        self.offset_head = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.obj_head(h).squeeze(-1), self.cls_head(h), self.offset_head(h)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_frame_path(root: Path, item: dict) -> Path:
    dataset, video_id, frame_idx = item["dataset"], str(item["video_id"]), int(item["frame_idx"])
    frame_dir = root / dataset / "frames" / video_id
    for name in (f"{frame_idx:03d}.jpg", f"{frame_idx:04d}.jpg", f"{frame_idx}.jpg"):
        path = frame_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"cannot resolve frame path for {dataset}/{video_id}/{frame_idx}")


def detection_files(root: Path, datasets: list[str]) -> list[Path]:
    files = []
    for dataset in datasets:
        files.extend(sorted((root / dataset / "object_detection" / "yolo26x" / "detections").glob("*.jsonl")))
    return files


def load_records(args) -> list[FrameRecord]:
    rng = random.Random(args.seed)
    by_dataset: dict[str, list[FrameRecord]] = defaultdict(list)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for path in detection_files(args.root, datasets):
        dataset = path.parents[3].name
        if len(by_dataset[dataset]) >= args.max_frames_per_dataset * 3:
            continue
        for item in iter_jsonl(path):
            frame_idx = int(item["frame_idx"])
            if args.frame_stride > 1 and frame_idx % args.frame_stride != 0:
                continue
            boxes = [dict(b) for b in item.get("boxes", []) if float(b.get("confidence", 0.0)) >= args.min_conf]
            if not boxes:
                continue
            by_dataset[item["dataset"]].append(
                FrameRecord(
                    dataset=item["dataset"],
                    video_id=str(item["video_id"]),
                    frame_idx=frame_idx,
                    image_path=resolve_frame_path(args.root, item),
                    width=int(item["width"]),
                    height=int(item["height"]),
                    boxes=boxes,
                )
            )
            if len(by_dataset[item["dataset"]]) >= args.max_frames_per_dataset * 3:
                break

    records = []
    for dataset in datasets:
        items = by_dataset.get(dataset, [])
        rng.shuffle(items)
        records.extend(items[: args.max_frames_per_dataset])
    rng.shuffle(records)
    return records


def build_class_map(records: list[FrameRecord], min_class_count: int) -> dict[str, int]:
    counts = Counter()
    for rec in records:
        counts.update(str(b["label"]) for b in rec.boxes)
    labels = [label for label, count in sorted(counts.items()) if count >= min_class_count]
    return {"background": 0, **{label: i + 1 for i, label in enumerate(labels)}}


def split_records(records: list[FrameRecord], val_ratio: float, seed: int):
    rng = random.Random(seed)
    items = records[:]
    rng.shuffle(items)
    val_n = max(1, int(len(items) * val_ratio))
    return items[val_n:], items[:val_n]


def class_color(idx: int):
    if idx <= 0:
        return (30, 41, 59)
    return PALETTE[(idx - 1) % len(PALETTE)]


def intersection(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def assign_targets(
    rec: FrameRecord,
    class_map: dict[str, int],
    processed_w: int,
    processed_h: int,
    grid_w: int,
    grid_h: int,
    token_size: int,
    min_overlap: float,
):
    sx, sy = processed_w / rec.width, processed_h / rec.height
    scaled_boxes = []
    for box in rec.boxes:
        label = str(box["label"])
        if label not in class_map:
            continue
        x1 = max(0.0, min(processed_w, float(box["x1"]) * sx))
        y1 = max(0.0, min(processed_h, float(box["y1"]) * sy))
        x2 = max(0.0, min(processed_w, float(box["x2"]) * sx))
        y2 = max(0.0, min(processed_h, float(box["y2"]) * sy))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 > x1 and y2 > y1:
            item = dict(box)
            item.update({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            scaled_boxes.append(item)

    obj = np.zeros(grid_w * grid_h, dtype=np.float32)
    cls = np.zeros(grid_w * grid_h, dtype=np.int64)
    off = np.zeros((grid_w * grid_h, 2), dtype=np.float32)
    for ty in range(grid_h):
        for tx in range(grid_w):
            idx = ty * grid_w + tx
            x1, y1 = tx * token_size, ty * token_size
            x2, y2 = min(processed_w, x1 + token_size), min(processed_h, y1 + token_size)
            cell = (x1, y1, x2, y2)
            cell_area = max(1.0, (x2 - x1) * (y2 - y1))
            best, best_overlap = None, 0.0
            for box in scaled_boxes:
                overlap = intersection(cell, (box["x1"], box["y1"], box["x2"], box["y2"])) / cell_area
                if overlap > best_overlap:
                    best, best_overlap = box, overlap
            if best is not None and best_overlap >= min_overlap:
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                bc_x, bc_y = (best["x1"] + best["x2"]) * 0.5, (best["y1"] + best["y2"]) * 0.5
                obj[idx] = 1.0
                cls[idx] = class_map[str(best["label"])]
                off[idx] = [(bc_x - cx) / processed_w, (bc_y - cy) / processed_h]
    return obj, cls, off, scaled_boxes


def load_vision_model(model_dir: Path, device: torch.device, dtype: torch.dtype):
    config = AutoConfig.from_pretrained(model_dir).vision_config
    model = Qwen3VLVisionModel(config)
    state = {}
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual.") :]] = f.get_tensor(key)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype).eval()
    return model, config


def prepare_image(rec: FrameRecord, resize_width: int, resize_height: int) -> Image.Image:
    image = Image.open(rec.image_path).convert("RGB")
    if resize_width and resize_height:
        image = image.resize((resize_width, resize_height), Image.Resampling.BICUBIC)
    return image


@torch.inference_mode()
def extract_features(
    records: list[FrameRecord],
    processor,
    vision_model,
    vision_config,
    class_map: dict[str, int],
    args,
    keep_visuals: bool,
):
    rng = random.Random(args.seed)
    token_size = vision_config.patch_size * vision_config.spatial_merge_size
    features, obj_targets, cls_targets, off_targets = [], [], [], []
    stats = Counter()
    visual_payloads = []
    device = next(vision_model.parameters()).device

    for start in range(0, len(records), args.extract_batch_size):
        batch_records = records[start : start + args.extract_batch_size]
        images = [prepare_image(rec, args.resize_width, args.resize_height) for rec in batch_records]
        inputs = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        grid_thw = inputs["image_grid_thw"].to(device)
        out = vision_model(pixel_values, grid_thw=grid_thw, return_dict=True)
        split_sizes = (grid_thw.prod(dim=1) // (vision_config.spatial_merge_size**2)).tolist()
        embeds = torch.split(out.pooler_output.detach().cpu().to(torch.float16), split_sizes)

        for rec, image, grid, emb in zip(batch_records, images, grid_thw.cpu().tolist(), embeds):
            _, patch_h, patch_w = [int(x) for x in grid]
            grid_w = patch_w // vision_config.spatial_merge_size
            grid_h = patch_h // vision_config.spatial_merge_size
            processed_w = patch_w * vision_config.patch_size
            processed_h = patch_h * vision_config.patch_size
            obj, cls, off, scaled_boxes = assign_targets(
                rec,
                class_map,
                processed_w,
                processed_h,
                grid_w,
                grid_h,
                token_size,
                args.min_overlap,
            )
            pos = np.flatnonzero(obj > 0.5).tolist()
            neg = np.flatnonzero(obj <= 0.5).tolist()
            rng.shuffle(neg)
            if pos:
                keep_neg = neg[: min(len(neg), max(len(pos) * args.neg_ratio, args.min_neg_tokens))]
                idx = pos + keep_neg
            else:
                idx = neg[: args.min_neg_tokens]
            rng.shuffle(idx)
            idx_np = np.asarray(idx, dtype=np.int64)
            features.append(emb[idx_np])
            obj_targets.append(torch.from_numpy(obj[idx_np]))
            cls_targets.append(torch.from_numpy(cls[idx_np]))
            off_targets.append(torch.from_numpy(off[idx_np]))
            stats["frames"] += 1
            stats["dense_tokens"] += int(len(obj))
            stats["dense_fg_tokens"] += int((obj > 0.5).sum())
            stats["sampled_tokens"] += int(len(idx_np))
            stats["sampled_fg_tokens"] += int((obj[idx_np] > 0.5).sum())
            for c in cls[idx_np]:
                stats[f"class_{int(c)}"] += 1

            if keep_visuals and len(visual_payloads) < args.visualize_frames:
                display = image.resize((processed_w, processed_h), Image.Resampling.BICUBIC)
                visual_payloads.append(
                    {
                        "record": rec,
                        "image": display,
                        "boxes": scaled_boxes,
                        "features": emb,
                        "obj": obj,
                        "cls": cls,
                        "grid": (grid_w, grid_h),
                        "processed_size": (processed_w, processed_h),
                    }
                )
        print(f"extracted {min(start + args.extract_batch_size, len(records))}/{len(records)} frames", flush=True)

    dataset = TensorDataset(
        torch.cat(features, dim=0),
        torch.cat(obj_targets, dim=0),
        torch.cat(cls_targets, dim=0),
        torch.cat(off_targets, dim=0),
    )
    return dataset, dict(stats), visual_payloads


def train_one_epoch(model, loader, optimizer, device, pos_weight: float):
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    ce = nn.CrossEntropyLoss()
    smooth = nn.SmoothL1Loss()
    totals = Counter()
    for x, obj, cls, off in loader:
        x = x.to(device).float()
        obj = obj.to(device).float()
        cls = cls.to(device)
        off = off.to(device).float()
        optimizer.zero_grad(set_to_none=True)
        obj_logits, cls_logits, pred_off = model(x)
        obj_loss = bce(obj_logits, obj)
        cls_loss = ce(cls_logits, cls)
        fg = obj > 0.5
        off_loss = smooth(pred_off[fg], off[fg]) if fg.any() else pred_off.sum() * 0.0
        loss = obj_loss + cls_loss + off_loss
        loss.backward()
        optimizer.step()
        n = len(x)
        totals["loss"] += float(loss.item()) * n
        totals["obj_loss"] += float(obj_loss.item()) * n
        totals["cls_loss"] += float(cls_loss.item()) * n
        totals["off_loss"] += float(off_loss.item()) * n
        totals["n"] += n
    return {k: totals[k] / max(1, totals["n"]) for k in ("loss", "obj_loss", "cls_loss", "off_loss")}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c = Counter()
    off_abs = 0.0
    for x, obj, cls, off in loader:
        x = x.to(device).float()
        obj = obj.to(device).float()
        cls = cls.to(device)
        off = off.to(device).float()
        obj_logits, cls_logits, pred_off = model(x)
        pred_obj = torch.sigmoid(obj_logits) >= 0.5
        true_obj = obj > 0.5
        pred_cls = cls_logits.argmax(-1)
        c["tp"] += int((pred_obj & true_obj).sum())
        c["fp"] += int((pred_obj & ~true_obj).sum())
        c["fn"] += int((~pred_obj & true_obj).sum())
        c["tn"] += int((~pred_obj & ~true_obj).sum())
        c["class_correct"] += int(((pred_cls == cls) & true_obj).sum())
        c["class_total"] += int(true_obj.sum())
        if true_obj.any():
            off_abs += float((pred_off[true_obj] - off[true_obj]).abs().mean()) * int(true_obj.sum())
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_acc_fg": c["class_correct"] / max(1, c["class_total"]),
        "offset_l1_fg": off_abs / max(1, c["class_total"]),
        **dict(c),
    }


def draw_boxes(image, boxes, class_map):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    font = load_font(15)
    for box in boxes:
        label = str(box["label"])
        if label not in class_map:
            continue
        color = class_color(class_map[label])
        xy = (box["x1"], box["y1"], box["x2"], box["y2"])
        draw.rectangle(xy, outline=color, width=3)
        text = f"{label} {float(box.get('confidence', 0.0)):.2f}"
        tw = draw.textlength(text, font=font)
        y = max(0, box["y1"] - 18)
        draw.rectangle((box["x1"], y, box["x1"] + tw + 6, y + 18), fill=color)
        draw.text((box["x1"] + 3, y), text, fill=(255, 255, 255), font=font)
    return out


def draw_token_map(image, grid, classes, objectness, token_size, title):
    width, height = image.size
    grid_w, grid_h = grid
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ty in range(grid_h):
        for tx in range(grid_w):
            idx = ty * grid_w + tx
            x1, y1 = tx * token_size, ty * token_size
            x2, y2 = min(width, x1 + token_size), min(height, y1 + token_size)
            cls_idx = int(classes[idx])
            if objectness[idx] >= 0.5 and cls_idx > 0:
                color = class_color(cls_idx)
                draw.rectangle((x1, y1, x2, y2), fill=(*color, 110), outline=(*color, 180))
            else:
                draw.rectangle((x1, y1, x2, y2), fill=(15, 23, 42, 16), outline=(148, 163, 184, 40))
    out = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(21)
    draw.rectangle((0, 0, width, 32), fill=(15, 23, 42))
    draw.text((10, 3), title, fill=(255, 255, 255), font=font)
    return out


@torch.no_grad()
def make_visualizations(model, payloads, class_map, vision_config, device, out_dir: Path):
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    token_size = vision_config.patch_size * vision_config.spatial_merge_size
    panels = []
    model.eval()
    for payload in payloads:
        rec = payload["record"]
        x = payload["features"].to(device).float()
        obj_logits, cls_logits, _ = model(x)
        pred_obj = torch.sigmoid(obj_logits).cpu().numpy()
        pred_cls = cls_logits.argmax(-1).cpu().numpy()
        bbox = draw_boxes(payload["image"], payload["boxes"], class_map)
        target = draw_token_map(payload["image"], payload["grid"], payload["cls"], payload["obj"], token_size, "target tokens from detections")
        pred = draw_token_map(payload["image"], payload["grid"], pred_cls, pred_obj, token_size, "predicted qwen visual tokens")
        w, h = payload["image"].size
        sheet = Image.new("RGB", (w * 3, h + 54), (255, 255, 255))
        sheet.paste(bbox, (0, 0))
        sheet.paste(target, (w, 0))
        sheet.paste(pred, (w * 2, 0))
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, h + 12),
            f"{rec.dataset}/{rec.video_id} frame {rec.frame_idx} | Qwen visual token grid {payload['grid'][0]}x{payload['grid'][1]}",
            fill=(15, 23, 42),
            font=load_font(21),
        )
        out_path = vis_dir / f"{rec.dataset}_{rec.video_id}_{rec.frame_idx:04d}.jpg"
        sheet.save(out_path, quality=92)
        scale = min(1800 / sheet.width, 1.0)
        panels.append(sheet.resize((int(sheet.width * scale), int(sheet.height * scale))))
    if panels:
        width = max(p.width for p in panels)
        height = sum(p.height for p in panels)
        contact = Image.new("RGB", (width, height), (248, 250, 252))
        y = 0
        for p in panels:
            contact.paste(p, (0, y))
            y += p.height
        contact.save(out_dir / "overview_qwen3vl_visual_token_binding.jpg", quality=92)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--datasets", default="avenue_test,shanghaitech_test,nwpu_test")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-frames-per-dataset", type=int, default=40)
    p.add_argument("--frame-stride", type=int, default=24)
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument("--min-overlap", type=float, default=0.10)
    p.add_argument("--resize-width", type=int, default=1280)
    p.add_argument("--resize-height", type=int, default=720)
    p.add_argument("--extract-batch-size", type=int, default=2)
    p.add_argument("--neg-ratio", type=int, default=4)
    p.add_argument("--min-neg-tokens", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260609)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vision-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--visualize-frames", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(args)
    if not records:
        raise RuntimeError("no detection records found")
    class_map = build_class_map(records, args.min_class_count)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    print(f"records train={len(train_records)} val={len(val_records)}", flush=True)
    print(f"class_map={class_map}", flush=True)

    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vision_dtype]
    processor = AutoProcessor.from_pretrained(args.model_dir)
    vision_model, vision_config = load_vision_model(args.model_dir, device, dtype)

    train_ds, train_stats, _ = extract_features(train_records, processor, vision_model, vision_config, class_map, args, False)
    val_ds, val_stats, payloads = extract_features(val_records, processor, vision_model, vision_config, class_map, args, True)
    del vision_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    head = TokenHead(train_ds.tensors[0].shape[1], len(class_map)).to(device)
    pos = float(train_ds.tensors[1].sum())
    neg = float(len(train_ds) - pos)
    pos_weight = max(1.0, min(20.0, neg / max(1.0, pos)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(head, train_loader, optimizer, device, pos_weight)
        val_m = evaluate(head, val_loader, device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(
            f"epoch {epoch:02d} loss={train_m['loss']:.4f} "
            f"val_f1={val_m['f1']:.3f} val_p={val_m['precision']:.3f} "
            f"val_r={val_m['recall']:.3f} class_acc_fg={val_m['class_acc_fg']:.3f}",
            flush=True,
        )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(
                {
                    "model": head.state_dict(),
                    "class_map": class_map,
                    "vision_config": {
                        "patch_size": vision_config.patch_size,
                        "spatial_merge_size": vision_config.spatial_merge_size,
                        "temporal_patch_size": vision_config.temporal_patch_size,
                    },
                    "args": vars(args),
                    "metrics": row,
                },
                args.out_dir / "best_qwen3vl_visual_token_head.pt",
            )

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "class_map": class_map,
                "train_stats": train_stats,
                "val_stats": val_stats,
                "history": history,
                "best_f1": best_f1,
                "note": "Features are Qwen3-VL vision pooler_output tokens, supervised only by detection boxes.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    make_visualizations(head, payloads, class_map, vision_config, device, args.out_dir)
    print(f"saved results to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
