#!/usr/bin/env python3
"""Render tracking JSONL outputs as MP4 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


PALETTE = [
    (64, 160, 255),
    (80, 220, 120),
    (220, 120, 80),
    (220, 180, 60),
    (180, 120, 240),
    (80, 220, 220),
    (255, 120, 180),
    (160, 220, 80),
]


def color_for(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]


def read_track_frames(track_file: Path) -> list[dict]:
    rows = []
    with track_file.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def frame_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def render_video(
    video_id: str,
    frames_root: Path,
    tracks_file: Path,
    output_path: Path,
    fps: float,
    draw_interpolated: bool,
) -> None:
    rows = read_track_frames(tracks_file)
    frame_dir = frames_root / video_id
    frame_paths = sorted(
        [path for path in frame_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=frame_sort_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    rows_by_idx = {int(row["frame_idx"]): row for row in rows}
    for frame_idx, frame_path in enumerate(frame_paths):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        row = rows_by_idx.get(frame_idx, {"tracks": []})
        for obj in row.get("tracks", []):
            if obj.get("interpolated") and not draw_interpolated:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in obj["bbox"]]
            track_id = int(obj["track_id"])
            label = str(obj["label"])
            color = color_for(track_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            tag = f"{track_id}:{label}"
            if obj.get("interpolated"):
                tag += "*"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y1 - th - 6)
            cv2.rectangle(image, (x1, y_text), (x1 + tw + 4, y_text + th + 6), color, -1)
            cv2.putText(
                image,
                tag,
                (x1 + 2, y_text + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"{video_id} frame {frame_idx}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(image)
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--all", action="store_true", help="Render every tracking JSONL file.")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--draw-interpolated", action="store_true")
    return parser.parse_args()


def pick_videos(tracks_root: Path, videos: list[str] | None, top_k: int, render_all: bool) -> list[str]:
    frames_dir = tracks_root / "frames"
    if videos:
        return videos
    if render_all:
        return [path.stem for path in sorted(frames_dir.glob("*.jsonl"))]
    counts = []
    for p in sorted(frames_dir.glob("*.jsonl")):
        count = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += len(row.get("tracks", []))
        counts.append((count, p.stem))
    counts.sort(reverse=True)
    return [video for _, video in counts[:top_k]]


def main() -> None:
    args = parse_args()
    output_root = args.output_root or args.tracks_root / "visualizations"
    videos = pick_videos(args.tracks_root, args.videos, args.top_k, args.all)
    for video_id in videos:
        tracks_file = args.tracks_root / "frames" / f"{video_id}.jsonl"
        output_path = output_root / video_id / f"{video_id}_tracking.mp4"
        render_video(
            video_id,
            args.dataset_root / "frames",
            tracks_file,
            output_path,
            args.fps,
            args.draw_interpolated,
        )
        print(output_path)


if __name__ == "__main__":
    main()
