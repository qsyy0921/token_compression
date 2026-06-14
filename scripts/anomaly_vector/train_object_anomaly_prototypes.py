#!/usr/bin/env python3
"""Train object-level anomaly prototypes on frozen Qwen3-VL visual tokens.

This is the 20260613 small-data, full-pipeline validation script:

1. read portable package data from 20260613_data
2. build positive/negative object-track and relation-track samples
3. extract real Qwen3-VL visual tokens with the frozen vision tower
4. bind bbox tracks to token grids with a fixed geometric rule
5. train normal/T01-T05 prototypes with coarse/binary/separation losses
6. evaluate visual-only and visual+motion ablations

No script is written into the data repository. Results are written to data
repository and then copied to the code repository backup path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


DEFAULT_DATA_ROOT = Path("/home/expand_disk/data_repository/mfl/token_compression/20260613_data")
DEFAULT_CODE_ROOT = Path("/home/expand_disk/code_repository/mfl/token_compression")
DEFAULT_MODEL_DIR = (
    Path("/home/expand_disk/code_repository/mfl")
    / "yong_task"
    / "token_pruner_merge"
    / "models"
    / "Qwen3-VL-8B-Instruct"
)

LABELS = ["normal", "T01", "T02", "T03", "T04", "T05"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
TRAIN_CATEGORIES = set(LABELS[1:])

DEFAULT_TRAIN_PACKAGES = [
    "Avenue_07",
    "Avenue_18",
    "NWPU_D054_02",
    "Avenue_16",
    "NWPU_D047_06",
    "ShanghaiTech_01_0135",
    "ShanghaiTech_03_0033",
    "NWPU_D068_01",
    "Avenue_11",
    "NWPU_D047_03",
]
DEFAULT_VAL_PACKAGES = [
    "Avenue_21",
    "NWPU_D149_04",
    "NWPU_D031_09",
    "UCF-Crime_Assault_Assault020_x264",
    "ShanghaiTech_03_0039",
]
DEFAULT_OPENSET_PACKAGES = ["NWPU_D003_05"]


@dataclass
class SampleSpec:
    sample_id: str
    split: str
    package_id: str
    package_dir: str
    dataset: str
    sample_type: str
    track_ids: list[int]
    object_classes: list[str]
    time_range: list[int]
    label: str
    fine_subtype: str | None
    event_id: str | None
    is_positive: bool
    negative_strength: str | None = None


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def find_package_dirs(data_root: Path) -> dict[str, Path]:
    package_dirs = {}
    for path in sorted((data_root / "packages").glob("*/*")):
        if path.is_dir() and (path / "annotation.json").exists():
            package_dirs[path.name] = path
    return package_dirs


def read_tracks(package_dir: Path) -> tuple[dict[int, list[dict]], dict[int, dict[int, dict]], int]:
    frames: dict[int, list[dict]] = {}
    by_track: dict[int, dict[int, dict]] = defaultdict(dict)
    max_frame = 0
    for row in iter_jsonl(package_dir / "tracks.jsonl"):
        frame = int(row["frame_index"])
        tracks = []
        for tr in row.get("tracks", []):
            tid = int(tr["track_id"])
            item = {
                "track_id": tid,
                "class_name": str(tr.get("class_name", "")),
                "confidence": float(tr.get("confidence", 0.0)),
                "bbox_xyxy": [float(x) for x in tr["bbox_xyxy"]],
                "bbox_xywh": [float(x) for x in tr.get("bbox_xywh", [])],
            }
            tracks.append(item)
            by_track[tid][frame] = item
        frames[frame] = tracks
        max_frame = max(max_frame, frame)
    return frames, by_track, max_frame + 1


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def event_intervals(annotation: dict, categories: set[str] | None = None) -> list[tuple[int, int]]:
    spans = []
    for ev in annotation.get("events", []):
        cat = ev.get("train_category")
        if categories is not None and cat not in categories:
            continue
        tr = ev.get("time_range", {})
        spans.append((int(tr.get("start_frame", 0)), int(tr.get("end_frame", 0))))
    return spans


def sample_window(start: int, end: int, total_frames: int, window_frames: int, offset: int = 0) -> list[int]:
    """Return all frames in one short contiguous segment.

    The preliminary experiment uses fewer/shorter segments, but it should not
    sparsely sample frames inside a segment. A sample unit is therefore:

        one track_id + one short contiguous time segment + all object tokens
        from every frame in that segment.
    """

    start = max(0, min(total_frames - 1, start))
    end = max(start, min(total_frames - 1, end))
    if end - start + 1 <= window_frames:
        frames = list(range(start, end + 1))
    else:
        max_start = end - window_frames + 1
        # Use several deterministic positions if multiple windows are requested.
        frac = (offset % 4) / 3.0 if offset > 0 else 0.5
        lo = int(round(start + (max_start - start) * frac))
        lo = max(start, min(max_start, lo))
        frames = list(range(lo, lo + window_frames))
    return frames


def event_cover_windows(start: int, end: int, total_frames: int, window_frames: int) -> list[list[int]]:
    """Split an event span into consecutive windows that cover every event frame."""

    start = max(0, min(total_frames - 1, start))
    end = max(start, min(total_frames - 1, end))
    step = max(1, int(window_frames))
    windows = []
    cursor = start
    while cursor <= end:
        hi = min(end, cursor + step - 1)
        windows.append(list(range(cursor, hi + 1)))
        cursor = hi + 1
    return windows


def has_track_frames(by_track: dict[int, dict[int, dict]], tid: int, frames: list[int]) -> bool:
    return any(frame in by_track.get(tid, {}) for frame in frames)


def choose_negative_tracks(
    frames: dict[int, list[dict]],
    by_track: dict[int, dict[int, dict]],
    event_span: tuple[int, int],
    related_ids: set[int],
    event_spans: list[tuple[int, int]],
    total_frames: int,
    rng: random.Random,
    max_items: int,
    window_frames: int,
    cover_event_windows: bool = False,
) -> list[tuple[list[int], list[int], str]]:
    all_ids = [tid for tid in by_track if tid not in related_ids]
    rng.shuffle(all_ids)
    negatives: list[tuple[list[int], list[int], str]] = []

    # Strong negatives: windows outside all event intervals.
    candidate_spans = []
    if event_spans:
        merged = sorted(event_spans)
        cursor = 0
        for s, e in merged:
            if cursor < s:
                candidate_spans.append((cursor, s - 1))
            cursor = max(cursor, e + 1)
        if cursor < total_frames:
            candidate_spans.append((cursor, total_frames - 1))
    else:
        candidate_spans.append((0, total_frames - 1))
    candidate_spans = [span for span in candidate_spans if span[1] - span[0] + 1 >= max(2, window_frames // 2)]
    rng.shuffle(candidate_spans)

    for tid in all_ids:
        if len(negatives) >= max_items:
            break
        for span in candidate_spans:
            wins = (
                event_cover_windows(span[0], span[1], total_frames, window_frames)
                if cover_event_windows
                else [sample_window(span[0], span[1], total_frames, window_frames)]
            )
            added = False
            for win in wins:
                if len(negatives) >= max_items:
                    break
                if has_track_frames(by_track, tid, win):
                    negatives.append(([tid], win, "strong"))
                    added = True
            if added:
                break

    # Medium negatives: inside the event interval, but not annotated as related.
    for tid in all_ids:
        if len(negatives) >= max_items:
            break
        wins = (
            event_cover_windows(event_span[0], event_span[1], total_frames, window_frames)
            if cover_event_windows
            else [sample_window(event_span[0], event_span[1], total_frames, window_frames)]
        )
        for win in wins:
            if len(negatives) >= max_items:
                break
            if has_track_frames(by_track, tid, win):
                negatives.append(([tid], win, "medium"))

    return negatives[:max_items]


def build_samples(
    data_root: Path,
    train_packages: list[str],
    val_packages: list[str],
    openset_packages: list[str],
    window_frames: int,
    positives_per_event_object: int,
    neg_per_pos: int,
    include_relation_samples: bool,
    seed: int,
    cover_event_windows: bool = False,
) -> tuple[list[SampleSpec], dict]:
    rng = random.Random(seed)
    package_dirs = find_package_dirs(data_root)
    samples: list[SampleSpec] = []
    summary = {
        "missing_packages": [],
        "package_splits": {"train": train_packages, "val": val_packages, "openset": openset_packages},
    }

    def add_sample(**kwargs):
        sample_id = f"s{len(samples):06d}_{kwargs['package_id']}_{kwargs['sample_type']}_{kwargs['label']}"
        samples.append(SampleSpec(sample_id=sample_id, **kwargs))

    for split, package_names in (("train", train_packages), ("val", val_packages), ("openset", openset_packages)):
        for package_id in package_names:
            package_dir = package_dirs.get(package_id)
            if package_dir is None:
                summary["missing_packages"].append(package_id)
                continue
            annotation = load_json(package_dir / "annotation.json")
            frames, by_track, total_frames = read_tracks(package_dir)
            all_event_spans = event_intervals(annotation, TRAIN_CATEGORIES)

            for ev in annotation.get("events", []):
                label = str(ev.get("train_category"))
                is_openset = label == "R06" or split == "openset"
                if not is_openset and label not in TRAIN_CATEGORIES:
                    continue
                if is_openset and label != "R06":
                    continue
                tr = ev.get("time_range", {})
                start = int(tr.get("start_frame", 0))
                end = int(tr.get("end_frame", start))
                event_span = (start, end)
                fine = ev.get("fine_subtype")
                related = [
                    obj
                    for obj in ev.get("related_objects", [])
                    if obj.get("tracking_id") is not None and obj.get("is_anomalous", True)
                ]
                related_ids = {int(obj["tracking_id"]) for obj in related}
                event_label = "R06" if is_openset else label

                pos_count = 0
                if cover_event_windows:
                    positive_windows = event_cover_windows(start, end, total_frames, window_frames)
                else:
                    positive_windows = [
                        sample_window(start, end, total_frames, window_frames, offset=offset)
                        for offset in range(max(1, positives_per_event_object))
                    ]
                for obj in related:
                    tid = int(obj["tracking_id"])
                    for win in positive_windows:
                        if not has_track_frames(by_track, tid, win):
                            continue
                        add_sample(
                            split=split,
                            package_id=package_id,
                            package_dir=str(package_dir),
                            dataset=package_dir.parent.name,
                            sample_type="object_track",
                            track_ids=[tid],
                            object_classes=[str(obj.get("object_class", ""))],
                            time_range=[min(win), max(win)],
                            label=event_label,
                            fine_subtype=fine,
                            event_id=ev.get("event_id"),
                            is_positive=True,
                        )
                        pos_count += 1

                # Relation positives are optional. The preliminary experiment
                # uses a single object track as the basic unit.
                if include_relation_samples and len(related) >= 2:
                    pair = related[:2]
                    tids = [int(pair[0]["tracking_id"]), int(pair[1]["tracking_id"])]
                    relation_windows = (
                        positive_windows
                        if cover_event_windows
                        else [sample_window(start, end, total_frames, window_frames)]
                    )
                    for win in relation_windows:
                        if all(has_track_frames(by_track, tid, win) for tid in tids):
                            add_sample(
                                split=split,
                                package_id=package_id,
                                package_dir=str(package_dir),
                                dataset=package_dir.parent.name,
                                sample_type="relation_track",
                                track_ids=tids,
                                object_classes=[
                                    str(pair[0].get("object_class", "")),
                                    str(pair[1].get("object_class", "")),
                                ],
                                time_range=[min(win), max(win)],
                                label=event_label,
                                fine_subtype=fine,
                                event_id=ev.get("event_id"),
                                is_positive=True,
                            )
                            pos_count += 1

                if split == "openset":
                    continue
                negs = choose_negative_tracks(
                    frames=frames,
                    by_track=by_track,
                    event_span=event_span,
                    related_ids=related_ids,
                    event_spans=all_event_spans,
                    total_frames=total_frames,
                    rng=rng,
                    max_items=max(pos_count * neg_per_pos, neg_per_pos),
                    window_frames=window_frames,
                    cover_event_windows=cover_event_windows,
                )
                for tids, win, strength in negs:
                    cls_names = []
                    for tid in tids:
                        rows = by_track.get(tid, {})
                        name = next((row.get("class_name", "") for row in rows.values()), "")
                        cls_names.append(str(name))
                    add_sample(
                        split=split,
                        package_id=package_id,
                        package_dir=str(package_dir),
                        dataset=package_dir.parent.name,
                        sample_type="object_track",
                        track_ids=tids,
                        object_classes=cls_names,
                        time_range=[min(win), max(win)],
                        label="normal",
                        fine_subtype=None,
                        event_id=ev.get("event_id"),
                        is_positive=False,
                        negative_strength=strength,
                    )

    counts = Counter()
    for s in samples:
        counts[f"split_{s.split}"] += 1
        counts[f"label_{s.label}"] += 1
        counts[f"type_{s.sample_type}"] += 1
        counts[f"positive_{s.is_positive}"] += 1
    summary["counts"] = dict(counts)
    summary["windowing"] = {
        "window_frames": int(window_frames),
        "cover_event_windows": bool(cover_event_windows),
    }
    return samples, summary


def apply_initial_sample_caps(
    samples: list[SampleSpec],
    max_train_positive_per_label: int,
    max_val_positive_per_label: int,
    neg_per_positive_keep: int,
    max_openset_samples: int,
    seed: int,
) -> tuple[list[SampleSpec], dict]:
    """Keep a small but complete validation set.

    This limits only data volume. It keeps positives, negatives, object samples,
    relation samples, train/val split, and R06 open-set observations.
    """

    if (
        max_train_positive_per_label <= 0
        and max_val_positive_per_label <= 0
        and max_openset_samples <= 0
    ):
        return samples, {"enabled": False}

    rng = random.Random(seed)
    kept: list[SampleSpec] = []
    selected_positive_keys: dict[tuple[str, str], list[SampleSpec]] = defaultdict(list)

    for split, cap in (("train", max_train_positive_per_label), ("val", max_val_positive_per_label)):
        if cap <= 0:
            split_pos = [s for s in samples if s.split == split and s.is_positive and s.label in TRAIN_CATEGORIES]
            kept.extend(split_pos)
            for s in split_pos:
                selected_positive_keys[(split, s.event_id or s.package_id)].append(s)
            continue
        for label in sorted(TRAIN_CATEGORIES):
            candidates = [s for s in samples if s.split == split and s.is_positive and s.label == label]
            # Prefer at least one relation sample when available, then fill with object tracks.
            relation = [s for s in candidates if s.sample_type == "relation_track"]
            object_track = [s for s in candidates if s.sample_type == "object_track"]
            rng.shuffle(relation)
            rng.shuffle(object_track)
            selected = (relation[:1] + object_track)[:cap]
            kept.extend(selected)
            for s in selected:
                selected_positive_keys[(split, s.event_id or s.package_id)].append(s)

    # Keep negatives tied to selected events. If an event lacks enough negatives,
    # fill from the same split to preserve normal supervision.
    kept_ids = {s.sample_id for s in kept}
    for split in ("train", "val"):
        selected_pos = [s for s in kept if s.split == split and s.is_positive and s.label in TRAIN_CATEGORIES]
        target_neg = max(1, len(selected_pos) * neg_per_positive_keep)
        event_ids = {s.event_id for s in selected_pos if s.event_id}
        neg_candidates = [
            s
            for s in samples
            if s.split == split and not s.is_positive and s.label == "normal" and s.sample_id not in kept_ids
        ]
        tied = [s for s in neg_candidates if s.event_id in event_ids]
        other = [s for s in neg_candidates if s.event_id not in event_ids]
        rng.shuffle(tied)
        rng.shuffle(other)
        for s in (tied + other)[:target_neg]:
            kept.append(s)
            kept_ids.add(s.sample_id)

    openset = [s for s in samples if s.split == "openset" or s.label == "R06"]
    rng.shuffle(openset)
    for s in openset[:max(0, max_openset_samples)]:
        if s.sample_id not in kept_ids:
            kept.append(s)
            kept_ids.add(s.sample_id)

    # Preserve deterministic order for reproducibility.
    kept = sorted(kept, key=lambda s: s.sample_id)
    counts = Counter()
    for s in kept:
        counts[f"split_{s.split}"] += 1
        counts[f"label_{s.label}"] += 1
        counts[f"type_{s.sample_type}"] += 1
        counts[f"positive_{s.is_positive}"] += 1
    return kept, {
        "enabled": True,
        "max_train_positive_per_label": max_train_positive_per_label,
        "max_val_positive_per_label": max_val_positive_per_label,
        "neg_per_positive_keep": neg_per_positive_keep,
        "max_openset_samples": max_openset_samples,
        "counts_after_cap": dict(counts),
    }


def load_vision_model(
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "flash_attention_2",
):
    config = AutoConfig.from_pretrained(model_dir, attn_implementation=attn_implementation).vision_config
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


def read_video_frame(cap: cv2.VideoCapture, frame_idx: int) -> Image.Image | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def resize_for_qwen(image: Image.Image, max_long_edge: int) -> Image.Image:
    if max_long_edge <= 0:
        return image
    w, h = image.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def scale_bbox_to_processed(bbox: list[float], orig_w: int, orig_h: int, processed_w: int, processed_h: int, expand: float):
    x1, y1, x2, y2 = bbox
    sx, sy = processed_w / max(1, orig_w), processed_h / max(1, orig_h)
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 = max(0.0, cx - bw * (1.0 + expand) * 0.5)
    x2 = min(float(processed_w), cx + bw * (1.0 + expand) * 0.5)
    y1 = max(0.0, cy - bh * (1.0 + expand) * 0.5)
    y2 = min(float(processed_h), cy + bh * (1.0 + expand) * 0.5)
    return [x1, y1, x2, y2]


def token_indices_for_bbox(
    bbox: list[float],
    grid_w: int,
    grid_h: int,
    token_size: int,
    min_tokens: int,
):
    x1, y1, x2, y2 = bbox
    hits = []
    scored = []
    for ty in range(grid_h):
        for tx in range(grid_w):
            idx = ty * grid_w + tx
            cx = (tx + 0.5) * token_size
            cy = (ty + 0.5) * token_size
            in_box = x1 <= cx <= x2 and y1 <= cy <= y2
            if in_box:
                hits.append(idx)
            cell = [tx * token_size, ty * token_size, (tx + 1) * token_size, (ty + 1) * token_size]
            ix1, iy1 = max(x1, cell[0]), max(y1, cell[1])
            ix2, iy2 = min(x2, cell[2]), min(y2, cell[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0:
                scored.append((inter, idx))
    if len(hits) >= min_tokens:
        return sorted(set(hits))
    scored = sorted(scored, reverse=True)
    fallback = [idx for _, idx in scored[:max(min_tokens, len(hits))]]
    return sorted(set(hits + fallback))


def bbox_motion_features(rows: list[dict], video_w: int, video_h: int) -> np.ndarray:
    if not rows:
        return np.zeros(8, dtype=np.float32)
    centers = []
    sizes = []
    confs = []
    for row in rows:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        cx = ((x1 + x2) * 0.5) / max(1, video_w)
        cy = ((y1 + y2) * 0.5) / max(1, video_h)
        bw = max(0.0, x2 - x1) / max(1, video_w)
        bh = max(0.0, y2 - y1) / max(1, video_h)
        centers.append([cx, cy])
        sizes.append([bw, bh])
        confs.append(float(row.get("confidence", 0.0)))
    centers_np = np.asarray(centers, dtype=np.float32)
    sizes_np = np.asarray(sizes, dtype=np.float32)
    deltas = np.diff(centers_np, axis=0) if len(centers_np) > 1 else np.zeros((1, 2), dtype=np.float32)
    speed = np.linalg.norm(deltas, axis=1)
    return np.asarray(
        [
            centers_np[:, 0].mean(),
            centers_np[:, 1].mean(),
            sizes_np[:, 0].mean(),
            sizes_np[:, 1].mean(),
            np.abs(deltas[:, 0]).mean(),
            np.abs(deltas[:, 1]).mean(),
            speed.mean(),
            np.mean(confs),
        ],
        dtype=np.float32,
    )


def frame_token_cache_file(cache_dir: Path, package_id: str, frame_idx: int) -> Path:
    return cache_dir / package_id / f"{int(frame_idx):06d}.pt"


def load_frame_token_cache(
    cache_dir: Path | None,
    package_id: str,
    frame_idx: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], tuple[int, int], int] | None:
    if cache_dir is None:
        return None
    path = frame_token_cache_file(cache_dir, package_id, frame_idx)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return (
        payload["emb"].to(torch.float16),
        tuple(payload["grid"]),
        tuple(payload["processed_size"]),
        tuple(payload["orig_size"]),
        int(payload["token_size"]),
    )


def save_frame_token_cache(
    cache_dir: Path | None,
    package_id: str,
    frame_idx: int,
    emb: torch.Tensor,
    grid: tuple[int, int],
    processed_size: tuple[int, int],
    orig_size: tuple[int, int],
    token_size: int,
    model_dir: Path,
    resize_long_edge: int,
) -> None:
    if cache_dir is None:
        return
    path = frame_token_cache_file(cache_dir, package_id, frame_idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "emb": emb.detach().cpu().to(torch.float16),
        "grid": tuple(int(x) for x in grid),
        "processed_size": tuple(int(x) for x in processed_size),
        "orig_size": tuple(int(x) for x in orig_size),
        "token_size": int(token_size),
        "package_id": package_id,
        "frame_idx": int(frame_idx),
        "model_dir": str(model_dir),
        "resize_long_edge": int(resize_long_edge),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@torch.inference_mode()
def extract_sample_features(
    samples: list[SampleSpec],
    model_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
    extract_batch_size: int,
    bbox_expand: float,
    min_tokens: int,
    resize_long_edge: int,
    frame_token_cache_dir: Path | None,
    max_samples: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[dict]]:
    if max_samples:
        samples = samples[:max_samples]

    package_cache: dict[str, tuple[dict[int, list[dict]], dict[int, dict[int, dict]], int]] = {}
    feature_rows: list[torch.Tensor] = []
    motion_rows: list[np.ndarray] = []
    token_set_rows: list[torch.Tensor] = []
    meta_rows: list[dict] = []

    # Group work by package/frame so each frame is encoded once.
    frame_jobs: dict[tuple[str, int], list[SampleSpec]] = defaultdict(list)
    for sample in samples:
        start, end = sample.time_range
        frames = list(range(int(start), int(end) + 1))
        for frame in frames:
            frame_jobs[(sample.package_id, int(frame))].append(sample)

    frame_feature_cache: dict[tuple[str, int], tuple[torch.Tensor, tuple[int, int], tuple[int, int], tuple[int, int]]] = {}
    package_dirs = {s.package_id: Path(s.package_dir) for s in samples}

    grouped_frames = list(frame_jobs)
    print(f"loading/extracting Qwen visual tokens for {len(grouped_frames)} unique frames", flush=True)
    if frame_token_cache_dir is not None:
        frame_token_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"frame token cache: {frame_token_cache_dir}", flush=True)

    missing_frames: list[tuple[str, int]] = []
    cache_hits = 0
    token_size: int | None = None
    for package_id, frame_idx in grouped_frames:
        cached = load_frame_token_cache(frame_token_cache_dir, package_id, frame_idx)
        if cached is None:
            missing_frames.append((package_id, frame_idx))
            continue
        emb, grid, processed_size, orig_size, cached_token_size = cached
        frame_feature_cache[(package_id, frame_idx)] = (emb, grid, processed_size, orig_size)
        token_size = cached_token_size
        cache_hits += 1

    print(
        f"frame token cache hits {cache_hits}/{len(grouped_frames)}; missing {len(missing_frames)}",
        flush=True,
    )

    vision_model = None
    vision_config = None
    processor = None
    if missing_frames:
        processor = AutoProcessor.from_pretrained(model_dir)
        vision_model, vision_config = load_vision_model(model_dir, device, dtype)
        token_size = int(vision_config.patch_size * vision_config.spatial_merge_size)

    for start in range(0, len(missing_frames), extract_batch_size):
        keys = missing_frames[start : start + extract_batch_size]
        images = []
        valid_keys = []
        orig_sizes = []
        for package_id, frame_idx in keys:
            pkg = package_dirs[package_id]
            cap = cv2.VideoCapture(str(pkg / "video.mp4"))
            if not cap.isOpened():
                continue
            image = read_video_frame(cap, frame_idx)
            cap.release()
            if image is None:
                continue
            orig_sizes.append(image.size)
            images.append(resize_for_qwen(image, resize_long_edge))
            valid_keys.append((package_id, frame_idx))
        if not images:
            continue
        assert processor is not None and vision_model is not None and vision_config is not None
        inputs = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        grid_thw = inputs["image_grid_thw"].to(device)
        out = vision_model(pixel_values, grid_thw=grid_thw, return_dict=True)
        split_sizes = (grid_thw.prod(dim=1) // (vision_config.spatial_merge_size**2)).tolist()
        embeds = torch.split(out.pooler_output.detach().cpu().to(torch.float16), split_sizes)
        for key, orig_size, grid, emb in zip(valid_keys, orig_sizes, grid_thw.cpu().tolist(), embeds):
            _, patch_h, patch_w = [int(x) for x in grid]
            grid_w = patch_w // int(vision_config.spatial_merge_size)
            grid_h = patch_h // int(vision_config.spatial_merge_size)
            processed_w = patch_w * int(vision_config.patch_size)
            processed_h = patch_h * int(vision_config.patch_size)
            frame_feature_cache[key] = (emb, (grid_w, grid_h), (processed_w, processed_h), orig_size)
            save_frame_token_cache(
                frame_token_cache_dir,
                key[0],
                key[1],
                emb,
                (grid_w, grid_h),
                (processed_w, processed_h),
                orig_size,
                int(token_size),
                model_dir,
                resize_long_edge,
            )
        print(
            f"encoded missing frames {min(start + extract_batch_size, len(missing_frames))}/{len(missing_frames)} "
            f"(total cached now {cache_hits + min(start + extract_batch_size, len(missing_frames))}/{len(grouped_frames)})",
            flush=True,
        )

    if token_size is None:
        raise RuntimeError("No frame tokens were loaded or extracted.")

    for sample in samples:
        pkg = Path(sample.package_dir)
        if sample.package_id not in package_cache:
            package_cache[sample.package_id] = read_tracks(pkg)
        _, by_track, _ = package_cache[sample.package_id]
        cap = cv2.VideoCapture(str(pkg / "video.mp4"))
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
        cap.release()

        track_vectors = []
        track_motion = []
        sample_token_counts = []
        sample_frame_counts = []
        sample_token_chunks = []
        start, end = sample.time_range
        frames = list(range(int(start), int(end) + 1))
        for tid in sample.track_ids:
            token_sum = None
            token_count = 0
            used_frames = 0
            tokens_per_frame = []
            rows_for_motion = []
            for frame in frames:
                row = by_track.get(int(tid), {}).get(int(frame))
                cached = frame_feature_cache.get((sample.package_id, int(frame)))
                if row is None or cached is None:
                    continue
                emb, grid, processed_size, orig_size = cached
                bbox = scale_bbox_to_processed(
                    row["bbox_xyxy"],
                    orig_size[0],
                    orig_size[1],
                    processed_size[0],
                    processed_size[1],
                    bbox_expand,
                )
                idx = token_indices_for_bbox(bbox, grid[0], grid[1], token_size, min_tokens=min_tokens)
                if idx:
                    token_tensor = emb[torch.tensor(idx, dtype=torch.long)].float()
                    sample_token_chunks.append(token_tensor.cpu().to(torch.float16))
                    cur_sum = token_tensor.sum(dim=0)
                    token_sum = cur_sum if token_sum is None else token_sum + cur_sum
                    token_count += int(token_tensor.shape[0])
                    used_frames += 1
                    tokens_per_frame.append(int(token_tensor.shape[0]))
                    rows_for_motion.append(row)
            if token_count > 0 and token_sum is not None:
                # A sample unit is one object over a short continuous segment.
                # Objects naturally have variable numbers of tokens, so we pool
                # all object tokens in the segment into one fixed-size vector.
                track_vectors.append(token_sum / float(token_count))
                track_motion.append(bbox_motion_features(rows_for_motion, video_w, video_h))
                sample_token_counts.append(
                    {
                        "track_id": int(tid),
                        "total_object_tokens": int(token_count),
                        "used_frames": int(used_frames),
                        "tokens_per_frame": tokens_per_frame,
                    }
                )
                sample_frame_counts.append(int(used_frames))

        if track_vectors:
            visual = torch.stack(track_vectors, dim=0).mean(dim=0)
            motion = np.stack(track_motion, axis=0).mean(axis=0) if track_motion else np.zeros(8, dtype=np.float32)
            feature_rows.append(visual)
            motion_rows.append(motion.astype(np.float32))
            token_set_rows.append(torch.cat(sample_token_chunks, dim=0).to(torch.float16))
            meta = asdict(sample)
            meta["used_track_count"] = len(track_vectors)
            meta["total_object_tokens"] = int(sum(x["total_object_tokens"] for x in sample_token_counts))
            meta["used_frames"] = int(max(sample_frame_counts) if sample_frame_counts else 0)
            meta["token_count_by_track"] = sample_token_counts
            meta_rows.append(meta)

    del vision_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    visual_features = torch.stack(feature_rows, dim=0).to(torch.float16)
    motion_features = torch.from_numpy(np.stack(motion_rows, axis=0)).to(torch.float32)
    torch.save(
        {
            "visual": visual_features,
            "motion": motion_features,
            "token_sets": token_set_rows,
            "meta": meta_rows,
        },
        out_dir / "feature_cache.pt",
    )
    return visual_features, motion_features, token_set_rows, meta_rows


class PrototypeScorer(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, normal_k: int, anomaly_m: int, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )
        self.normal = nn.Parameter(torch.randn(normal_k, embed_dim) * 0.02)
        self.anomaly = nn.Parameter(torch.randn(5, anomaly_m, embed_dim) * 0.02)

    def all_prototypes(self) -> torch.Tensor:
        return torch.cat([self.normal, self.anomaly.reshape(-1, self.anomaly.shape[-1])], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.proj(x), dim=-1)
        normal = F.normalize(self.normal, dim=-1)
        anomaly = F.normalize(self.anomaly, dim=-1)
        logit_normal = (q @ normal.t()).max(dim=1).values
        logits_anomaly = torch.einsum("bd,cmd->bcm", q, anomaly).max(dim=2).values
        return torch.cat([logit_normal[:, None], logits_anomaly], dim=1) * self.tau


class TokenEvidencePrototypeScorer(nn.Module):
    """Prototype scorer for variable-length object token sets.

    Each object segment can contain a different number of tokens. The model
    first compares every token with normal/T01-T05 prototypes, then aggregates
    token evidence into object-level logits with a token-count-normalized rule.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        normal_k: int,
        anomaly_m: int,
        tau: float,
        pooling: str,
        topk_ratio: float,
        lme_alpha: float,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.pooling = pooling
        self.topk_ratio = topk_ratio
        self.lme_alpha = lme_alpha
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )
        self.normal = nn.Parameter(torch.randn(normal_k, embed_dim) * 0.02)
        self.anomaly = nn.Parameter(torch.randn(5, anomaly_m, embed_dim) * 0.02)

    def all_prototypes(self) -> torch.Tensor:
        return torch.cat([self.normal, self.anomaly.reshape(-1, self.anomaly.shape[-1])], dim=0)

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return scores.mean(dim=0)
        if self.pooling == "topk":
            k = max(1, int(math.ceil(float(scores.shape[0]) * self.topk_ratio)))
            k = min(k, int(scores.shape[0]))
            return scores.topk(k, dim=0).values.mean(dim=0)
        if self.pooling == "logmeanexp":
            alpha = float(self.lme_alpha)
            return torch.logsumexp(alpha * scores, dim=0) / alpha - math.log(scores.shape[0]) / alpha
        raise ValueError(f"unknown token evidence pooling: {self.pooling}")

    def forward_one(self, tokens: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.proj(tokens.float()), dim=-1)
        normal = F.normalize(self.normal, dim=-1)
        anomaly = F.normalize(self.anomaly, dim=-1)
        token_normal = (q @ normal.t()).max(dim=1).values
        token_anomaly = torch.einsum("nd,cmd->ncm", q, anomaly).max(dim=2).values
        # Normal evidence should describe the whole object; anomaly evidence can
        # be local, so use the configured evidence pooling for anomaly logits.
        normal_logit = token_normal.mean()
        anomaly_logits = self.aggregate(token_anomaly)
        return torch.cat([normal_logit[None], anomaly_logits], dim=0) * self.tau


def prototype_separation_loss(model: PrototypeScorer, margin: float) -> torch.Tensor:
    p = F.normalize(model.all_prototypes(), dim=-1)
    sim = p @ p.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return torch.relu(sim[mask] - margin).pow(2).mean()


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, metas: list[dict]) -> dict:
    pred = prob.argmax(axis=1)
    is_anom = y_true != 0
    pred_anom = pred != 0
    category_mask = is_anom
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float((pred == y_true).mean()) if len(y_true) else 0.0,
        "anomaly_recall": float(((pred_anom & is_anom).sum()) / max(1, is_anom.sum())),
        "normal_false_positive_rate": float(((pred_anom & ~is_anom).sum()) / max(1, (~is_anom).sum())),
        "category_accuracy_positive": float((pred[category_mask] == y_true[category_mask]).mean()) if category_mask.any() else 0.0,
    }
    per_class = {}
    for label, idx in LABEL_TO_ID.items():
        mask = y_true == idx
        if mask.any():
            per_class[label] = {
                "count": int(mask.sum()),
                "accuracy": float((pred[mask] == idx).mean()),
                "mean_score": float((1.0 - prob[mask, 0]).mean()),
            }
    metrics["per_class"] = per_class

    # Event top-k recall: among samples tied to an event, do positive samples rank high?
    by_event = defaultdict(list)
    for i, meta in enumerate(metas):
        if meta.get("event_id"):
            by_event[(meta["package_id"], meta["event_id"])].append(i)
    top1 = top3 = total = 0
    scores = 1.0 - prob[:, 0]
    for _, idxs in by_event.items():
        pos = [i for i in idxs if metas[i].get("is_positive")]
        if not pos or len(idxs) <= 1:
            continue
        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)
        total += 1
        top1 += int(ranked[0] in pos)
        top3 += int(any(i in pos for i in ranked[: min(3, len(ranked))]))
    metrics["event_top1_recall"] = top1 / max(1, total)
    metrics["event_top3_recall"] = top3 / max(1, total)
    metrics["event_ranking_count"] = total
    return metrics


def train_one_ablation(
    name: str,
    visual: torch.Tensor,
    motion: torch.Tensor,
    metas: list[dict],
    out_dir: Path,
    args,
) -> dict:
    if name == "visual_only":
        x = visual.float()
    elif name == "visual_motion":
        x = torch.cat([visual.float(), motion.float()], dim=1)
    else:
        raise ValueError(name)
    y_all = torch.tensor([LABEL_TO_ID.get(m["label"], -1) for m in metas], dtype=torch.long)
    supervised = y_all >= 0
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]

    device = torch.device(args.train_device)
    model = PrototypeScorer(
        in_dim=x.shape[1],
        embed_dim=args.embed_dim,
        normal_k=args.normal_prototypes,
        anomaly_m=args.anomaly_prototypes,
        tau=args.tau,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = TensorDataset(x[train_idx], y_all[train_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    history = []
    best = {"val_accuracy": -1.0, "epoch": -1}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            coarse = F.cross_entropy(logits, yb)
            prob = logits.softmax(dim=1)
            binary_target = (yb != 0).float()
            binary = F.binary_cross_entropy(1.0 - prob[:, 0], binary_target)
            sep = prototype_separation_loss(model, args.sep_margin)
            loss = coarse + args.lambda_binary * binary + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item()) * len(xb)
            totals["coarse"] += float(coarse.item()) * len(xb)
            totals["binary"] += float(binary.item()) * len(xb)
            totals["sep"] += float(sep.item()) * len(xb)
            totals["n"] += len(xb)
        with torch.no_grad():
            model.eval()
            val_logits = model(x[val_idx].to(device)).softmax(dim=1).cpu().numpy() if val_idx else np.zeros((0, 6))
            val_y = y_all[val_idx].cpu().numpy() if val_idx else np.zeros((0,), dtype=np.int64)
            val_metas = [metas[i] for i in val_idx]
            val_metrics = compute_metrics(val_y, val_logits, val_metas) if val_idx else {}
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "coarse": totals["coarse"] / max(1, totals["n"]),
            "binary": totals["binary"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
            "val_anomaly_recall": val_metrics.get("anomaly_recall", 0.0),
            "val_normal_fpr": val_metrics.get("normal_false_positive_rate", 0.0),
            "val_category_acc_pos": val_metrics.get("category_accuracy_positive", 0.0),
            "val_event_top3_recall": val_metrics.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if row["val_accuracy"] > best["val_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob = model(x.to(device)).softmax(dim=1).cpu().numpy()
    y_np = y_all.cpu().numpy()
    train_metrics = compute_metrics(y_np[train_idx], prob[train_idx], [metas[i] for i in train_idx])
    val_metrics = compute_metrics(y_np[val_idx], prob[val_idx], [metas[i] for i in val_idx])
    openset_metrics = {}
    if openset_idx:
        openset_scores = 1.0 - prob[openset_idx, 0]
        openset_metrics = {
            "n": len(openset_idx),
            "mean_anomaly_score": float(openset_scores.mean()),
            "max_anomaly_score": float(openset_scores.max()),
            "min_anomaly_score": float(openset_scores.min()),
            "predicted_categories": dict(Counter(LABELS[1 + int(prob[i, 1:].argmax())] for i in openset_idx)),
        }

    ab_dir = out_dir / name
    ab_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": LABELS, "args": vars(args)}, ab_dir / "best_model.pt")
    with (ab_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    predictions_path = ab_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as f:
        for meta, true_id, p in zip(metas, y_np, prob):
            pred_id = int(p.argmax())
            row = {
                **meta,
                "true_label": LABELS[int(true_id)] if int(true_id) >= 0 else str(meta["label"]),
                "pred_label": LABELS[pred_id],
                "object_anomaly_score": float(1.0 - p[0]),
                "category": LABELS[1 + int(p[1:].argmax())],
                "probabilities": {label: float(p[i]) for i, label in enumerate(LABELS)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "ablation": name,
        "best_epoch": int(best.get("epoch", -1)),
        "train": train_metrics,
        "val": val_metrics,
        "openset": openset_metrics,
        "best_row": best,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_openset": len(openset_idx),
    }
    (ab_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def train_token_evidence_ablation(
    name: str,
    token_sets: list[torch.Tensor],
    metas: list[dict],
    out_dir: Path,
    args,
) -> dict:
    y_all = torch.tensor([LABEL_TO_ID.get(m["label"], -1) for m in metas], dtype=torch.long)
    supervised = y_all >= 0
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]

    device = torch.device(args.train_device)
    model = TokenEvidencePrototypeScorer(
        in_dim=token_sets[0].shape[1],
        embed_dim=args.embed_dim,
        normal_k=args.normal_prototypes,
        anomaly_m=args.anomaly_prototypes,
        tau=args.tau,
        pooling=args.token_pooling,
        topk_ratio=args.token_topk_ratio,
        lme_alpha=args.token_lme_alpha,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def predict(indices: list[int]) -> np.ndarray:
        if not indices:
            return np.zeros((0, len(LABELS)), dtype=np.float32)
        rows = []
        model.eval()
        with torch.no_grad():
            for i in indices:
                logits = model.forward_one(token_sets[i].to(device))
                rows.append(logits.softmax(dim=0).detach().cpu())
        return torch.stack(rows, dim=0).numpy()

    history = []
    best = {"val_accuracy": -1.0, "epoch": -1}
    best_state = None
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_idx[:]
        rng.shuffle(order)
        totals = Counter()
        for i in order:
            tokens = token_sets[i].to(device)
            yb = y_all[i].to(device)
            logits = model.forward_one(tokens)
            coarse = F.cross_entropy(logits[None, :], yb[None])
            prob = logits.softmax(dim=0)
            binary_target = (yb != 0).float()
            binary = F.binary_cross_entropy(1.0 - prob[0], binary_target)
            sep = prototype_separation_loss(model, args.sep_margin)
            loss = coarse + args.lambda_binary * binary + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item())
            totals["coarse"] += float(coarse.item())
            totals["binary"] += float(binary.item())
            totals["sep"] += float(sep.item())
            totals["n"] += 1
        val_prob = predict(val_idx)
        val_y = y_all[val_idx].cpu().numpy() if val_idx else np.zeros((0,), dtype=np.int64)
        val_metas = [metas[i] for i in val_idx]
        val_metrics = compute_metrics(val_y, val_prob, val_metas) if val_idx else {}
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "coarse": totals["coarse"] / max(1, totals["n"]),
            "binary": totals["binary"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_accuracy": val_metrics.get("accuracy", 0.0),
            "val_anomaly_recall": val_metrics.get("anomaly_recall", 0.0),
            "val_normal_fpr": val_metrics.get("normal_false_positive_rate", 0.0),
            "val_category_acc_pos": val_metrics.get("category_accuracy_positive", 0.0),
            "val_event_top3_recall": val_metrics.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if row["val_accuracy"] > best["val_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    all_idx = list(range(len(metas)))
    prob = predict(all_idx)
    y_np = y_all.cpu().numpy()
    train_metrics = compute_metrics(y_np[train_idx], prob[train_idx], [metas[i] for i in train_idx])
    val_metrics = compute_metrics(y_np[val_idx], prob[val_idx], [metas[i] for i in val_idx])
    openset_metrics = {}
    if openset_idx:
        openset_scores = 1.0 - prob[openset_idx, 0]
        openset_metrics = {
            "n": len(openset_idx),
            "mean_anomaly_score": float(openset_scores.mean()),
            "max_anomaly_score": float(openset_scores.max()),
            "min_anomaly_score": float(openset_scores.min()),
            "predicted_categories": dict(Counter(LABELS[1 + int(prob[i, 1:].argmax())] for i in openset_idx)),
        }

    ab_dir = out_dir / name
    ab_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "labels": LABELS, "args": vars(args)}, ab_dir / "best_model.pt")
    with (ab_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    with (ab_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for meta, true_id, p, tokens in zip(metas, y_np, prob, token_sets):
            pred_id = int(p.argmax())
            row = {
                **meta,
                "pooling": args.token_pooling,
                "token_set_size": int(tokens.shape[0]),
                "true_label": LABELS[int(true_id)] if int(true_id) >= 0 else str(meta["label"]),
                "pred_label": LABELS[pred_id],
                "object_anomaly_score": float(1.0 - p[0]),
                "category": LABELS[1 + int(p[1:].argmax())],
                "probabilities": {label: float(p[i]) for i, label in enumerate(LABELS)},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "ablation": name,
        "token_evidence_pooling": args.token_pooling,
        "token_topk_ratio": args.token_topk_ratio,
        "token_lme_alpha": args.token_lme_alpha,
        "best_epoch": int(best.get("epoch", -1)),
        "train": train_metrics,
        "val": val_metrics,
        "openset": openset_metrics,
        "best_row": best,
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_openset": len(openset_idx),
    }
    (ab_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def draw_sample_visualizations(prediction_file: Path, out_dir: Path, max_images: int = 24):
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted(rows, key=lambda r: r.get("object_anomaly_score", 0.0), reverse=True)
    font = load_font(22)
    small = load_font(18)
    for row in rows[:max_images]:
        pkg = Path(row["package_dir"])
        cap = cv2.VideoCapture(str(pkg / "video.mp4"))
        frame_idx = int(sum(row["time_range"]) / 2)
        image = read_video_frame(cap, frame_idx)
        cap.release()
        if image is None:
            continue
        frames, by_track, _ = read_tracks(pkg)
        draw = ImageDraw.Draw(image)
        for tid in row["track_ids"]:
            tr = by_track.get(int(tid), {}).get(frame_idx)
            if tr is None:
                # find nearest available frame in the window
                candidates = [f for f in by_track.get(int(tid), {}) if row["time_range"][0] <= f <= row["time_range"][1]]
                if candidates:
                    nearest = min(candidates, key=lambda f: abs(f - frame_idx))
                    tr = by_track[int(tid)][nearest]
            if tr is None:
                continue
            x1, y1, x2, y2 = tr["bbox_xyxy"]
            color = (220, 38, 38) if row["object_anomaly_score"] >= 0.5 else (22, 163, 74)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            text = f"id={tid} score={row['object_anomaly_score']:.2f} cat={row['category']}"
            tw = draw.textlength(text, font=small)
            yy = max(0, y1 - 26)
            draw.rectangle((x1, yy, x1 + tw + 8, yy + 26), fill=color)
            draw.text((x1 + 4, yy + 1), text, fill=(255, 255, 255), font=small)
        header = f"{row['package_id']} | true={row['true_label']} pred={row['pred_label']} | {row['sample_type']}"
        draw.rectangle((0, 0, image.width, 34), fill=(15, 23, 42))
        draw.text((10, 3), header, fill=(255, 255, 255), font=font)
        image.save(vis_dir / f"{row['sample_id']}.jpg", quality=92)


def write_report(out_dir: Path, sample_summary: dict, metrics: list[dict], args):
    lines = [
        "# 20260613 对象级异常向量小样本完整链路训练结果",
        "",
        "本实验使用少量 package，但完整保留 object-to-token、正负样本构造、Projection/Temporal、normal/T01-T05 prototypes、三项 loss 与对象级验证。",
        "",
        "## 数据与样本",
        "",
        "```json",
        json.dumps(sample_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 训练设置",
        "",
        "```json",
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 指标摘要",
        "",
    ]
    for item in metrics:
        lines.extend(
            [
                f"### {item['ablation']}",
                "",
                "```json",
                json.dumps(item, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 说明",
            "",
            "- `object_anomaly_score = 1 - P(normal)`。",
            "- `category = argmax P(T01:T05)`。",
            "- R06 不参与 prototype 训练，只在后续 open-set evaluation 中观察。",
            "- Token compression 不参与当前训练 loss，只作为后续验证模块。",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--exp-name", default="")
    p.add_argument("--train-packages", default=",".join(DEFAULT_TRAIN_PACKAGES))
    p.add_argument("--val-packages", default=",".join(DEFAULT_VAL_PACKAGES))
    p.add_argument("--openset-packages", default=",".join(DEFAULT_OPENSET_PACKAGES))
    p.add_argument("--window-frames", type=int, default=8)
    p.add_argument("--positives-per-event-object", type=int, default=1)
    p.add_argument("--cover-event-windows", action="store_true")
    p.add_argument("--neg-per-pos", type=int, default=2)
    p.add_argument("--include-relation-samples", action="store_true")
    p.add_argument("--max-train-positive-per-label", type=int, default=1)
    p.add_argument("--max-val-positive-per-label", type=int, default=1)
    p.add_argument("--cap-neg-per-positive", type=int, default=1)
    p.add_argument("--max-openset-samples", type=int, default=1)
    p.add_argument("--bbox-expand", type=float, default=0.08)
    p.add_argument("--min-tokens", type=int, default=1)
    p.add_argument("--resize-long-edge", type=int, default=1280)
    p.add_argument("--frame-token-cache-dir", type=Path, default=None)
    p.add_argument("--extract-batch-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--normal-prototypes", type=int, default=2)
    p.add_argument("--anomaly-prototypes", type=int, default=1)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--run-token-evidence", action="store_true")
    p.add_argument("--token-pooling", choices=["mean", "topk", "logmeanexp"], default="topk")
    p.add_argument("--token-topk-ratio", type=float, default=0.2)
    p.add_argument("--token-lme-alpha", type=float, default=10.0)
    p.add_argument("--lambda-binary", type=float, default=0.5)
    p.add_argument("--lambda-sep", type=float, default=0.1)
    p.add_argument("--sep-margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vision-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--max-samples", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vision_dtype]
    device = torch.device(args.device)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"exp_{timestamp}_object_anomaly_prototype_small"
    data_out = args.data_root / "results" / exp_name
    code_out = args.code_root / "outputs" / "20260613" / "results" / exp_name
    data_out.mkdir(parents=True, exist_ok=True)
    code_out.parent.mkdir(parents=True, exist_ok=True)
    if args.frame_token_cache_dir is None:
        args.frame_token_cache_dir = args.data_root.parent / "cache" / "20260613_data_token_cache"

    train_packages = parse_list(args.train_packages)
    val_packages = parse_list(args.val_packages)
    openset_packages = parse_list(args.openset_packages)
    samples, sample_summary = build_samples(
        data_root=args.data_root,
        train_packages=train_packages,
        val_packages=val_packages,
        openset_packages=openset_packages,
        window_frames=args.window_frames,
        positives_per_event_object=args.positives_per_event_object,
        neg_per_pos=args.neg_per_pos,
        include_relation_samples=args.include_relation_samples,
        seed=args.seed,
        cover_event_windows=args.cover_event_windows,
    )
    raw_sample_count = len(samples)
    samples, cap_summary = apply_initial_sample_caps(
        samples,
        max_train_positive_per_label=args.max_train_positive_per_label,
        max_val_positive_per_label=args.max_val_positive_per_label,
        neg_per_positive_keep=args.cap_neg_per_positive,
        max_openset_samples=args.max_openset_samples,
        seed=args.seed,
    )
    sample_summary["raw_sample_count_before_cap"] = raw_sample_count
    sample_summary["initial_validation_cap"] = cap_summary
    (data_out / "sample_index.jsonl").write_text(
        "\n".join(json.dumps(asdict(s), ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )
    (data_out / "sample_index_summary.json").write_text(
        json.dumps(sample_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (data_out / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"built {len(samples)} sample specs", flush=True)
    print(json.dumps(sample_summary, ensure_ascii=False, indent=2), flush=True)

    visual, motion, token_sets, metas = extract_sample_features(
        samples=samples,
        model_dir=args.model_dir,
        device=device,
        dtype=dtype,
        out_dir=data_out,
        extract_batch_size=args.extract_batch_size,
        bbox_expand=args.bbox_expand,
        min_tokens=args.min_tokens,
        resize_long_edge=args.resize_long_edge,
        frame_token_cache_dir=args.frame_token_cache_dir,
        max_samples=args.max_samples or None,
    )
    with (data_out / "feature_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in metas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = []
    for ablation in ("visual_only", "visual_motion"):
        item = train_one_ablation(ablation, visual, motion, metas, data_out, args)
        metrics.append(item)
        draw_sample_visualizations(data_out / ablation / "predictions.jsonl", data_out / ablation, max_images=24)

    if args.run_token_evidence:
        token_name = f"token_{args.token_pooling}"
        item = train_token_evidence_ablation(token_name, token_sets, metas, data_out, args)
        metrics.append(item)
        draw_sample_visualizations(data_out / token_name / "predictions.jsonl", data_out / token_name, max_images=24)

    # Keep a top-level visualization folder for quick browsing.
    draw_sample_visualizations(data_out / "visual_motion" / "predictions.jsonl", data_out, max_images=24)
    (data_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(data_out, sample_summary, metrics, args)

    if code_out.exists():
        shutil.rmtree(code_out)
    shutil.copytree(data_out, code_out)
    print(f"saved data results: {data_out}", flush=True)
    print(f"saved code backup: {code_out}", flush=True)


if __name__ == "__main__":
    main()
