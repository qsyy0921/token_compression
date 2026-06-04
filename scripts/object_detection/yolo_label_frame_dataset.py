import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO


DEFAULT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "backpack",
    "handbag",
    "suitcase",
    "skateboard",
    "bottle",
    "cell phone",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def numeric_frame_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 0, path.name


def parse_class_filter(value: str | None) -> set[str] | None:
    if value is None or value.strip().lower() == "default":
        return DEFAULT_CLASSES
    if value.strip().lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def load_done_frames(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "frame_idx" in record:
                done.add(int(record["frame_idx"]))
    return done


def iter_video_dirs(frames_root: Path) -> list[Path]:
    return sorted([path for path in frames_root.iterdir() if path.is_dir()], key=lambda path: path.name)


def iter_frames(video_dir: Path) -> list[Path]:
    return sorted(
        [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=numeric_frame_key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO on per-video frame folders with resume support.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--classes", default="default", help="'default', 'all', or comma-separated class names.")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    class_filter = parse_class_filter(args.classes)

    model = YOLO(args.weights)
    name_to_id = {name: idx for idx, name in model.names.items()}
    class_ids = None if class_filter is None else [name_to_id[name] for name in class_filter if name in name_to_id]
    used_classes = "all" if class_filter is None else sorted(name for name in class_filter if name in name_to_id)
    missing_classes = [] if class_filter is None else sorted(class_filter - set(name_to_id))

    videos = iter_video_dirs(frames_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    summary = {
        "dataset": args.dataset,
        "frames_root": args.frames_root,
        "weights": args.weights,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "classes": used_classes,
        "missing_classes": missing_classes,
        "videos": [],
    }

    for video_dir in videos:
        frames = iter_frames(video_dir)
        out_path = output_root / f"{video_dir.name}.jsonl"
        done = load_done_frames(out_path)
        processed = 0
        boxes_total = 0
        start = time.time()
        with out_path.open("a", encoding="utf-8") as handle:
            for frame_path in frames:
                frame_idx = numeric_frame_key(frame_path)[0]
                if frame_idx in done:
                    continue
                try:
                    result = model.predict(
                        source=str(frame_path),
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        classes=class_ids,
                        device=args.device,
                        verbose=False,
                    )[0]
                except Exception as exc:
                    print(f"{video_dir.name}: skip {frame_path.name} after inference error: {exc}", flush=True)
                    continue
                boxes = []
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    label = model.names[cls_id]
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    boxes.append(
                        {
                            "label": label,
                            "class_id": cls_id,
                            "confidence": float(box.conf.item()),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )
                record = {
                    "dataset": args.dataset,
                    "video_id": video_dir.name,
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path.as_posix()),
                    "width": int(result.orig_shape[1]),
                    "height": int(result.orig_shape[0]),
                    "boxes": boxes,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                boxes_total += len(boxes)
                if processed % 200 == 0:
                    elapsed = time.time() - start
                    print(
                        f"{video_dir.name}: processed {processed} new frames "
                        f"({processed / elapsed:.2f} fps), boxes={boxes_total}",
                        flush=True,
                    )
                if args.sleep > 0:
                    time.sleep(args.sleep)
        elapsed = time.time() - start
        summary["videos"].append(
            {
                "video_id": video_dir.name,
                "frames_seen": len(frames),
                "new_frames_processed": processed,
                "boxes": boxes_total,
                "seconds": elapsed,
                "fps": processed / elapsed if elapsed > 0 else 0.0,
                "output": str(out_path.as_posix()),
            }
        )
        print(
            f"{video_dir.name}: done new_frames={processed} boxes={boxes_total} "
            f"seconds={elapsed:.1f} fps={processed / elapsed if elapsed > 0 else 0.0:.2f}",
            flush=True,
        )

    (output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
