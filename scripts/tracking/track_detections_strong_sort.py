#!/usr/bin/env python3
"""Offline object tracking for detection JSONL outputs.

The tracker is designed for already-computed dataset detections. It combines:

- label normalization and per-frame NMS
- constant-velocity Kalman filtering
- Hungarian assignment
- IoU, normalized center distance, and HSV color-histogram appearance cost
- short-gap interpolation for smoother downstream trajectories
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PERSON_ALIASES = {"person", "pedestrian", "runner", "running person"}
LABEL_ALIASES = {
    "person": "person",
    "pedestrian": "person",
    "runner": "person",
    "running person": "person",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "scooter": "motorcycle",
    "car": "car",
    "vehicle": "vehicle",
    "skateboard": "skateboard",
    "stroller": "stroller",
    "cart": "cart",
    "trolley": "cart",
    "bag": "bag",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "luggage": "suitcase",
}


@dataclass
class Detection:
    video_id: str
    frame_idx: int
    label: str
    source_label: str
    bbox: np.ndarray
    score: float
    appearance: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    label: str
    state: np.ndarray
    covariance: np.ndarray
    last_bbox: np.ndarray
    last_frame: int
    start_frame: int
    hits: int = 1
    age: int = 1
    missed: int = 0
    appearance: np.ndarray | None = None
    observations: dict[int, dict[str, Any]] = field(default_factory=dict)
    labels: Counter = field(default_factory=Counter)

    def predict(self) -> np.ndarray:
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = 1.0
        transition[1, 5] = 1.0
        transition[2, 6] = 1.0
        transition[3, 7] = 1.0
        process_noise = np.diag([4.0, 4.0, 1.0, 1.0, 25.0, 25.0, 4.0, 4.0])
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise
        self.age += 1
        self.missed += 1
        self.last_bbox = state_to_bbox(self.state)
        return self.last_bbox

    def update(self, det: Detection, appearance_alpha: float) -> None:
        measurement = bbox_to_measurement(det.bbox)
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        observation[2, 2] = 1.0
        observation[3, 3] = 1.0
        measurement_noise = np.diag([8.0, 8.0, 4.0, 4.0])
        innovation = measurement - observation @ self.state
        s_mat = observation @ self.covariance @ observation.T + measurement_noise
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(s_mat)
        self.state = self.state + kalman_gain @ innovation
        self.covariance = (np.eye(8) - kalman_gain @ observation) @ self.covariance

        self.last_bbox = det.bbox.astype(np.float64)
        self.last_frame = det.frame_idx
        self.hits += 1
        self.missed = 0
        self.labels[det.label] += 1
        if det.appearance is not None:
            if self.appearance is None:
                self.appearance = det.appearance.copy()
            else:
                self.appearance = (
                    appearance_alpha * self.appearance
                    + (1.0 - appearance_alpha) * det.appearance
                )
                norm = np.linalg.norm(self.appearance)
                if norm > 0:
                    self.appearance = self.appearance / norm
        self.observations[det.frame_idx] = {
            "frame_idx": det.frame_idx,
            "track_id": self.track_id,
            "label": self.label,
            "source_label": det.source_label,
            "bbox": det.bbox.round(3).tolist(),
            "score": float(det.score),
            "interpolated": False,
        }


def bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox.astype(np.float64)
    return np.array(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)],
        dtype=np.float64,
    )


def measurement_to_state(measurement: np.ndarray) -> np.ndarray:
    state = np.zeros(8, dtype=np.float64)
    state[:4] = measurement
    return state


def state_to_bbox(state: np.ndarray) -> np.ndarray:
    cx, cy, w, h = state[:4]
    w = max(1.0, float(w))
    h = max(1.0, float(h))
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: np.ndarray, b: np.ndarray, width: int, height: int) -> float:
    acx = (a[0] + a[2]) / 2.0
    acy = (a[1] + a[3]) / 2.0
    bcx = (b[0] + b[2]) / 2.0
    bcy = (b[1] + b[3]) / 2.0
    diag = math.hypot(width, height)
    return min(1.0, math.hypot(float(acx - bcx), float(acy - bcy)) / max(1.0, diag))


def appearance_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    return float(np.clip(1.0 - float(np.dot(a, b)), 0.0, 1.0))


def normalize_label(label: str, allowed: set[str]) -> str | None:
    raw = label.strip().lower()
    label_norm = LABEL_ALIASES.get(raw)
    if label_norm is None:
        if raw.startswith("motor"):
            label_norm = "motorcycle"
        elif raw.startswith("bike") or "bicycle" in raw:
            label_norm = "bicycle"
        elif raw in PERSON_ALIASES:
            label_norm = "person"
    if label_norm in allowed:
        return label_norm
    return None


def labels_compatible(track_label: str, det_label: str) -> bool:
    if track_label == det_label:
        return True
    vehicle_group = {"car", "vehicle"}
    bag_group = {"bag", "backpack", "handbag", "suitcase"}
    return (
        track_label in vehicle_group
        and det_label in vehicle_group
        or track_label in bag_group
        and det_label in bag_group
    )


def nms_detections(detections: list[Detection], threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = defaultdict(list)
    for det in detections:
        by_label[det.label].append(det)
    for group in by_label.values():
        group = sorted(group, key=lambda d: d.score, reverse=True)
        while group:
            best = group.pop(0)
            kept.append(best)
            group = [det for det in group if iou(best.bbox, det.bbox) < threshold]
    return kept


def crop_histogram(image: np.ndarray | None, bbox: np.ndarray) -> np.ndarray | None:
    if image is None:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    if norm <= 0:
        return None
    return (hist / norm).astype(np.float32)


def load_schema_targets(schema_path: Path) -> set[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    policy = schema["policy"]
    targets = policy.get("tracking_targets") or policy["annotation_targets"]
    return {str(label).lower() for label in targets}


def resolve_frame_path(frames_root: Path, video_id: str, rec: dict[str, Any]) -> Path:
    frame_idx = int(rec["frame_idx"])
    candidates = [
        frames_root / video_id / f"{frame_idx:03d}.jpg",
        frames_root / video_id / f"{frame_idx:04d}.jpg",
        frames_root / video_id / Path(str(rec.get("frame_path", ""))).name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_video_detections(
    det_path: Path,
    frames_root: Path,
    allowed_labels: set[str],
    min_area: float,
    nms_threshold: float,
    use_appearance: bool,
) -> tuple[list[list[Detection]], int, int]:
    frames: dict[int, list[Detection]] = defaultdict(list)
    width = 0
    height = 0
    image_cache: dict[int, np.ndarray | None] = {}
    video_id = det_path.stem

    with det_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            frame_idx = int(rec["frame_idx"])
            width = int(rec.get("width") or width or 0)
            height = int(rec.get("height") or height or 0)
            image = None
            if use_appearance:
                if frame_idx not in image_cache:
                    frame_path = resolve_frame_path(frames_root, video_id, rec)
                    image_cache[frame_idx] = cv2.imread(str(frame_path))
                image = image_cache[frame_idx]
            for box in rec.get("boxes", []):
                label = normalize_label(str(box.get("label", "")), allowed_labels)
                if label is None:
                    continue
                bbox = np.array(
                    [
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                    ],
                    dtype=np.float64,
                )
                area = max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
                if area < min_area:
                    continue
                appearance = crop_histogram(image, bbox) if use_appearance else None
                frames[frame_idx].append(
                    Detection(
                        video_id=video_id,
                        frame_idx=frame_idx,
                        label=label,
                        source_label=str(box.get("label", "")),
                        bbox=bbox,
                        score=float(box.get("score", box.get("confidence", 1.0))),
                        appearance=appearance,
                    )
                )
    frame_dir = frames_root / video_id
    frame_count = len(list(frame_dir.glob("*.jpg"))) if frame_dir.exists() else 0
    if frame_count > 0:
        max_frame = frame_count - 1
    elif frames:
        max_frame = max(frames)
    else:
        max_frame = -1
    ordered: list[list[Detection]] = []
    for frame_idx in range(max_frame + 1):
        ordered.append(nms_detections(frames.get(frame_idx, []), nms_threshold))
    return ordered, width, height


def assignment_cost(
    track: Track,
    det: Detection,
    width: int,
    height: int,
    iou_weight: float,
    center_weight: float,
    appearance_weight: float,
) -> float:
    if not labels_compatible(track.label, det.label):
        return 1e6
    overlap = iou(track.last_bbox, det.bbox)
    dist = center_distance(track.last_bbox, det.bbox, width, height)
    app = appearance_distance(track.appearance, det.appearance)
    label_penalty = 0.03 if track.label != det.label else 0.0
    return iou_weight * (1.0 - overlap) + center_weight * dist + appearance_weight * app + label_penalty


def interpolate_track(track: Track, max_gap: int) -> None:
    frames = sorted(track.observations)
    for prev_frame, next_frame in zip(frames, frames[1:]):
        gap = next_frame - prev_frame
        if gap <= 1 or gap > max_gap + 1:
            continue
        prev_obs = track.observations[prev_frame]
        next_obs = track.observations[next_frame]
        prev_box = np.array(prev_obs["bbox"], dtype=np.float64)
        next_box = np.array(next_obs["bbox"], dtype=np.float64)
        for frame_idx in range(prev_frame + 1, next_frame):
            alpha = (frame_idx - prev_frame) / gap
            bbox = (1.0 - alpha) * prev_box + alpha * next_box
            track.observations[frame_idx] = {
                "frame_idx": frame_idx,
                "track_id": track.track_id,
                "label": track.label,
                "source_label": "interpolated",
                "bbox": bbox.round(3).tolist(),
                "score": 0.0,
                "interpolated": True,
            }


def run_video(
    video_id: str,
    frame_detections: list[list[Detection]],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> tuple[list[Track], dict[str, Any]]:
    active: list[Track] = []
    finished: list[Track] = []
    next_track_id = 1
    raw_detections = 0

    for frame_idx, detections in enumerate(frame_detections):
        raw_detections += len(detections)
        for track in active:
            track.predict()

        if active and detections:
            costs = np.zeros((len(active), len(detections)), dtype=np.float64)
            for i, track in enumerate(active):
                for j, det in enumerate(detections):
                    costs[i, j] = assignment_cost(
                        track,
                        det,
                        width,
                        height,
                        args.iou_weight,
                        args.center_weight,
                        args.appearance_weight,
                    )
            row_idx, col_idx = linear_sum_assignment(costs)
        else:
            row_idx = np.array([], dtype=int)
            col_idx = np.array([], dtype=int)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for i, j in zip(row_idx.tolist(), col_idx.tolist()):
            det = detections[j]
            track = active[i]
            overlap = iou(track.last_bbox, det.bbox)
            dist = center_distance(track.last_bbox, det.bbox, width, height)
            if costs[i, j] <= args.max_cost and (overlap >= args.min_iou or dist <= args.max_center_distance):
                track.update(det, args.appearance_alpha)
                matched_tracks.add(i)
                matched_dets.add(j)

        for j, det in enumerate(detections):
            if j in matched_dets:
                continue
            measurement = bbox_to_measurement(det.bbox)
            track = Track(
                track_id=next_track_id,
                label=det.label,
                state=measurement_to_state(measurement),
                covariance=np.diag([20.0, 20.0, 10.0, 10.0, 100.0, 100.0, 25.0, 25.0]),
                last_bbox=det.bbox.copy(),
                last_frame=frame_idx,
                start_frame=frame_idx,
                appearance=det.appearance.copy() if det.appearance is not None else None,
            )
            track.labels[det.label] += 1
            track.observations[frame_idx] = {
                "frame_idx": frame_idx,
                "track_id": track.track_id,
                "label": track.label,
                "source_label": det.source_label,
                "bbox": det.bbox.round(3).tolist(),
                "score": float(det.score),
                "interpolated": False,
            }
            next_track_id += 1
            active.append(track)

        survivors: list[Track] = []
        for track in active:
            if track.missed > args.max_missed:
                finished.append(track)
            else:
                survivors.append(track)
        active = survivors

    finished.extend(active)

    kept_tracks: list[Track] = []
    for track in finished:
        if track.hits >= args.min_hits and len(track.observations) >= args.min_track_length:
            interpolate_track(track, args.interpolate_gap)
            kept_tracks.append(track)

    stats = {
        "video_id": video_id,
        "num_frames": len(frame_detections),
        "raw_detections_after_filter_nms": raw_detections,
        "tracks_total": len(finished),
        "tracks_kept": len(kept_tracks),
        "track_labels": Counter(t.label for t in kept_tracks),
    }
    return kept_tracks, stats


def write_video_outputs(out_dir: Path, video_id: str, tracks: list[Track], stats: dict[str, Any]) -> None:
    frames_out = out_dir / "frames"
    tracks_out = out_dir / "tracks"
    frames_out.mkdir(parents=True, exist_ok=True)
    tracks_out.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    summary_rows: list[dict[str, Any]] = []
    for track in tracks:
        for obs in track.observations.values():
            by_frame[int(obs["frame_idx"])].append(obs)
        observed_frames = sorted(track.observations)
        summary_rows.append(
            {
                "track_id": track.track_id,
                "label": track.label,
                "start_frame": min(observed_frames),
                "end_frame": max(observed_frames),
                "num_frames": len(observed_frames),
                "hits": track.hits,
                "label_votes": dict(track.labels),
            }
        )

    with (frames_out / f"{video_id}.jsonl").open("w", encoding="utf-8") as f:
        for frame_idx in range(int(stats["num_frames"])):
            objects = sorted(by_frame.get(frame_idx, []), key=lambda x: (x["track_id"], x["label"]))
            f.write(json.dumps({"video_id": video_id, "frame_idx": frame_idx, "tracks": objects}) + "\n")

    with (tracks_out / f"{video_id}.json").open("w", encoding="utf-8") as f:
        json.dump({"video_id": video_id, "tracks": summary_rows, "stats": stats}, f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detections-dir", type=Path, default=None)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--min-area", type=float, default=16.0)
    parser.add_argument("--nms-threshold", type=float, default=0.86)
    parser.add_argument("--max-missed", type=int, default=18)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--interpolate-gap", type=int, default=12)
    parser.add_argument("--min-iou", type=float, default=0.03)
    parser.add_argument("--max-center-distance", type=float, default=0.12)
    parser.add_argument("--max-cost", type=float, default=0.82)
    parser.add_argument("--iou-weight", type=float, default=0.58)
    parser.add_argument("--center-weight", type=float, default=0.24)
    parser.add_argument("--appearance-weight", type=float, default=0.18)
    parser.add_argument("--appearance-alpha", type=float, default=0.82)
    parser.add_argument("--no-appearance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    detections_dir = args.detections_dir or dataset_root / "detections"
    frames_root = args.frames_root or dataset_root / "frames"
    schema_path = args.schema or dataset_root / "annotation_schema.json"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = load_schema_targets(schema_path)
    if args.video_list:
        wanted = {line.strip() for line in args.video_list.read_text().splitlines() if line.strip()}
        det_files = [detections_dir / f"{video}.jsonl" for video in sorted(wanted)]
    else:
        det_files = sorted(detections_dir.glob("*.jsonl"))

    start = time.time()
    all_stats: list[dict[str, Any]] = []
    label_counts = Counter()
    for idx, det_path in enumerate(det_files, start=1):
        if not det_path.exists():
            print(f"[skip] missing {det_path}")
            continue
        video_id = det_path.stem
        frame_dets, width, height = load_video_detections(
            det_path,
            frames_root,
            allowed_labels,
            args.min_area,
            args.nms_threshold,
            not args.no_appearance,
        )
        tracks, stats = run_video(video_id, frame_dets, width, height, args)
        write_video_outputs(out_dir, video_id, tracks, stats)
        all_stats.append(stats)
        label_counts.update(stats["track_labels"])
        print(
            f"[{idx}/{len(det_files)}] {video_id}: "
            f"frames={stats['num_frames']} detections={stats['raw_detections_after_filter_nms']} "
            f"tracks={stats['tracks_kept']}"
        )

    summary = {
        "dataset_root": str(dataset_root),
        "detections_dir": str(detections_dir),
        "frames_root": str(frames_root),
        "schema": str(schema_path),
        "output_dir": str(out_dir),
        "allowed_annotation_targets": sorted(allowed_labels),
        "num_videos": len(all_stats),
        "num_frames": sum(int(s["num_frames"]) for s in all_stats),
        "num_detections_after_filter_nms": sum(int(s["raw_detections_after_filter_nms"]) for s in all_stats),
        "num_tracks_kept": sum(int(s["tracks_kept"]) for s in all_stats),
        "track_label_counts": dict(label_counts),
        "seconds": round(time.time() - start, 3),
        "tracker": {
            "name": "strong_sort_lite",
            "kalman": "constant_velocity_cxcywh",
            "assignment": "hungarian",
            "cost": "iou + center_distance + hsv_histogram_appearance + label_compatibility",
            "interpolation_max_gap": args.interpolate_gap,
            "max_missed": args.max_missed,
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
