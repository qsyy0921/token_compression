import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


def iou(a: dict, b: dict) -> float:
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def canonical_label(label: str | None) -> str:
    if not label:
        return "unknown"
    label = label.lower().strip()
    aliases = {
        "pedestrian": "person",
        "people": "person",
        "bike": "bicycle",
        "motorbike": "motorcycle",
        "electric scooter": "scooter",
        "kick scooter": "scooter",
        "vehicle": "car",
        "handbag": "bag",
        "backpack": "bag",
        "luggage": "suitcase",
        "package": "box",
        "parcel": "box",
        "trolley": "cart",
        "handcart": "cart",
        "stroller": "cart",
    }
    return aliases.get(label, label)


@dataclass
class Track:
    track_id: int
    label: str
    last_frame: int
    box: dict
    hits: int = 1
    missed: int = 0
    history: list[dict] = field(default_factory=list)


def load_detection_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda rec: int(rec["frame_idx"]))


def track_video(records: list[dict], min_iou: float, max_age: int) -> tuple[list[dict], list[dict]]:
    next_id = 1
    active: list[Track] = []
    rows: list[dict] = []

    for rec in records:
        frame_idx = int(rec["frame_idx"])
        detections = []
        for det_idx, box in enumerate(rec.get("boxes", [])):
            label = canonical_label(box.get("label"))
            detections.append(
                {
                    "det_idx": det_idx,
                    "label": label,
                    "x1": float(box["x1"]),
                    "y1": float(box["y1"]),
                    "x2": float(box["x2"]),
                    "y2": float(box["y2"]),
                    "raw_label": box.get("label"),
                }
            )

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        candidates = []
        for ti, track in enumerate(active):
            for di, det in enumerate(detections):
                if track.label != det["label"]:
                    continue
                candidates.append((iou(track.box, det), ti, di))
        candidates.sort(reverse=True, key=lambda item: item[0])

        for score, ti, di in candidates:
            if score < min_iou or ti in assigned_tracks or di in assigned_dets:
                continue
            track = active[ti]
            det = detections[di]
            track.last_frame = frame_idx
            track.box = det
            track.hits += 1
            track.missed = 0
            assigned_tracks.add(ti)
            assigned_dets.add(di)
            rows.append(track_row(rec["video_id"], frame_idx, track, det, score))

        for ti, track in enumerate(active):
            if ti not in assigned_tracks:
                track.missed += 1

        for di, det in enumerate(detections):
            if di in assigned_dets:
                continue
            track = Track(track_id=next_id, label=det["label"], last_frame=frame_idx, box=det)
            next_id += 1
            active.append(track)
            rows.append(track_row(rec["video_id"], frame_idx, track, det, 1.0))

        active = [track for track in active if track.missed <= max_age]

    summaries = []
    by_id: dict[int, list[dict]] = {}
    for row in rows:
        by_id.setdefault(row["track_id"], []).append(row)
    for track_id, items in sorted(by_id.items()):
        labels = {item["label"] for item in items}
        summaries.append(
            {
                "track_id": track_id,
                "label": sorted(labels)[0] if labels else "unknown",
                "start_frame": min(item["frame_idx"] for item in items),
                "end_frame": max(item["frame_idx"] for item in items),
                "length": len(items),
            }
        )
    return rows, summaries


def track_row(video_id: str, frame_idx: int, track: Track, det: dict, match_iou: float) -> dict:
    return {
        "video_id": video_id,
        "frame_idx": frame_idx,
        "track_id": track.track_id,
        "label": track.label,
        "raw_label": det.get("raw_label"),
        "x1": det["x1"],
        "y1": det["y1"],
        "x2": det["x2"],
        "y2": det["y2"],
        "match_iou": match_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple label-aware IoU tracking for LocateAnything outputs.")
    parser.add_argument("--detections-root", default="outputs/locateanything_shanghai_test")
    parser.add_argument("--output-root", default="outputs/shanghai_iou_tracks")
    parser.add_argument("--video", default=None)
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=5)
    args = parser.parse_args()

    det_root = Path(args.detections_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    files = sorted(det_root.glob("*.jsonl"))
    if args.video:
        files = [p for p in files if p.stem == args.video]

    run_summary = []
    for path in files:
        records = load_detection_records(path)
        if not records:
            continue
        rows, summaries = track_video(records, args.min_iou, args.max_age)
        rows_path = out_root / f"{path.stem}.tracks.jsonl"
        summary_path = out_root / f"{path.stem}.track_summary.json"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        run_summary.append(
            {
                "video_id": path.stem,
                "frames": len(records),
                "track_rows": len(rows),
                "tracks": len(summaries),
                "tracks_output": str(rows_path.as_posix()),
                "summary_output": str(summary_path.as_posix()),
            }
        )

    (out_root / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
