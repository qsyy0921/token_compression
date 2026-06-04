#!/usr/bin/env python3
"""Prepare frame-level anomaly ground truth for the local datasets."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

SHANGHAI_MASKS = Path(
    "/home/expand_disk/code_repository/mfl/yong_task/20260602/data/shanghai/data/testframemask"
)
AVENUE_ANNOTATIONS = Path(
    "/home/expand_disk/code_repository/mfl/VAD2/avenue_dataset/all_in_one/avenue_annotations.json"
)
NWPU_GT = Path("/home/expand_disk/data_repository/mfl/NWPU/NWPU_Campus_gt.npz")


def frame_count(frame_dir: Path) -> int:
    if not frame_dir.exists():
        return 0
    return len(list(frame_dir.glob("*.jpg")))


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "num_frames",
                "num_abnormal_frames",
                "num_normal_frames",
                "has_anomaly",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(out_dir: Path, dataset: str, source: str, rows: list[dict[str, object]]) -> None:
    total_frames = sum(int(r["num_frames"]) for r in rows)
    abnormal_frames = sum(int(r["num_abnormal_frames"]) for r in rows)
    metadata = {
        "dataset": dataset,
        "format": "one .npy file per video; 0=normal, 1=abnormal",
        "source": source,
        "num_videos": len(rows),
        "num_frames": total_frames,
        "num_abnormal_frames": abnormal_frames,
        "num_normal_frames": total_frames - abnormal_frames,
        "num_anomalous_videos": sum(1 for r in rows if int(r["num_abnormal_frames"]) > 0),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "README.md").write_text(
        "# Frame-Level Ground Truth\n\n"
        "Each `.npy` file contains one frame-level anomaly label array for a video.\n"
        "`0` means normal and `1` means abnormal.\n",
        encoding="utf-8",
    )


def save_label(out_dir: Path, video: str, labels: np.ndarray) -> dict[str, object]:
    labels = labels.astype(np.uint8)
    np.save(out_dir / f"{video}.npy", labels)
    abnormal = int(labels.sum())
    total = int(labels.shape[0])
    return {
        "video": video,
        "num_frames": total,
        "num_abnormal_frames": abnormal,
        "num_normal_frames": total - abnormal,
        "has_anomaly": int(abnormal > 0),
    }


def prepare_shanghaitech() -> None:
    out_dir = DATASETS_ROOT / "shanghaitech_test" / "frames_GT"
    reset_dir(out_dir)

    rows: list[dict[str, object]] = []
    for npy in sorted(SHANGHAI_MASKS.glob("*.npy")):
        labels = np.load(npy, allow_pickle=False)
        rows.append(save_label(out_dir, npy.stem, labels))

    write_summary(out_dir, rows)
    write_metadata(out_dir, "shanghaitech_test", str(SHANGHAI_MASKS), rows)


def prepare_avenue() -> None:
    out_dir = DATASETS_ROOT / "avenue_test" / "frames_GT"
    reset_dir(out_dir)

    frame_root = DATASETS_ROOT / "avenue_test" / "frames"
    annotations = json.loads(AVENUE_ANNOTATIONS.read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    segments_out: dict[str, list[dict[str, object]]] = {}
    for seq_num in range(1, 22):
        video = f"{seq_num:02d}"
        ann_key = f"01_{seq_num:04d}"
        n_frames = frame_count(frame_root / video)
        labels = np.zeros(n_frames, dtype=np.uint8)
        segments_out[video] = []

        for segment in annotations.get(ann_key, {}).get("segments", []):
            start = int(segment["start"])
            end = int(segment["end"])
            start_idx = max(0, start - 1)
            end_idx = min(n_frames, end)
            if start_idx < end_idx:
                labels[start_idx:end_idx] = 1
            segments_out[video].append(
                {
                    "start_frame_1based": start,
                    "end_frame_1based_inclusive": end,
                    "events": segment.get("events", []),
                    "objects": segment.get("objects", []),
                }
            )

        rows.append(save_label(out_dir, video, labels))

    (out_dir / "segments.json").write_text(
        json.dumps(segments_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(out_dir, rows)
    write_metadata(out_dir, "avenue_test", str(AVENUE_ANNOTATIONS), rows)


def prepare_nwpu() -> None:
    out_dir = DATASETS_ROOT / "nwpu_test" / "frames_GT"
    reset_dir(out_dir)

    gt = np.load(NWPU_GT, allow_pickle=False)
    rows: list[dict[str, object]] = []
    for video in sorted(gt.files):
        rows.append(save_label(out_dir, video, gt[video]))

    write_summary(out_dir, rows)
    write_metadata(out_dir, "nwpu_test", str(NWPU_GT), rows)


def main() -> None:
    prepare_shanghaitech()
    prepare_avenue()
    prepare_nwpu()
    print("Prepared frames_GT for shanghaitech_test, avenue_test, and nwpu_test.")


if __name__ == "__main__":
    main()
