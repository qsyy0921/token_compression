import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


DEFAULT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "truck",
    "bench",
    "backpack",
    "umbrella",
    "handbag",
    "suitcase",
    "skateboard",
    "bottle",
    "chair",
}


PALETTE = {
    "person": (0, 220, 120),
    "bicycle": (255, 170, 0),
    "motorcycle": (255, 80, 0),
    "car": (255, 40, 40),
    "truck": (255, 40, 40),
    "skateboard": (190, 80, 255),
    "backpack": (80, 160, 255),
    "handbag": (80, 160, 255),
    "suitcase": (80, 160, 255),
    "bottle": (255, 220, 80),
    "umbrella": (80, 220, 255),
    "bench": (220, 220, 220),
    "chair": (220, 220, 220),
}


def numeric_frame_key(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def parse_class_filter(value: str | None) -> set[str] | None:
    if value is None or value.strip().lower() == "default":
        return DEFAULT_CLASSES
    if value.strip().lower() == "all":
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def draw_detections(image_path: Path, record: dict, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for det in record["boxes"]:
        label = det["label"]
        color = PALETTE.get(label, (255, 255, 255))
        xy = [det["x1"], det["y1"], det["x2"], det["y2"]]
        draw.rectangle(xy, outline=color, width=3)
        text = f"{label} {det['confidence']:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = xy[0], max(0, xy[1] - th - 4)
        draw.rectangle([x, y, x + tw + 6, y + th + 4], fill=color)
        draw.text((x + 3, y + 2), text, fill=(0, 0, 0), font=font)

    title = f"{record['video_id']} frame {record['frame_idx']} boxes={len(record['boxes'])}"
    draw.rectangle([0, 0, image.width, 24], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO on ShanghaiTech test frames.")
    parser.add_argument("--frames-root", default="data/shanghai/data/testing/frames")
    parser.add_argument("--weights", default="models/yolo26x.pt")
    parser.add_argument("--output-root", default="outputs/yolo26x_shanghai_test")
    parser.add_argument("--visual-root", default="outputs/visualizations/yolo26x_test")
    parser.add_argument("--video", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--classes", default="default", help="'default', 'all', or comma-separated class names.")
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    class_filter = parse_class_filter(args.classes)
    model = YOLO(args.weights)
    frames_dir = Path(args.frames_root) / args.video
    output_root = Path(args.output_root)
    visual_dir = Path(args.visual_root) / args.video
    output_root.mkdir(parents=True, exist_ok=True)
    out_path = output_root / f"{args.video}.jsonl"
    frames = sorted(frames_dir.glob("*.jpg"), key=numeric_frame_key)
    name_to_id = {name: idx for idx, name in model.names.items()}
    class_ids = None if class_filter is None else [name_to_id[name] for name in class_filter if name in name_to_id]

    summary = {
        "video_id": args.video,
        "weights": args.weights,
        "frames": len(frames),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "classes": sorted(class_filter) if class_filter is not None else "all",
        "output": str(out_path.as_posix()),
        "visual_root": str(visual_dir.as_posix()) if args.save_vis else None,
    }

    with out_path.open("w", encoding="utf-8") as handle:
        for frame_path in frames:
            result = model.predict(
                source=str(frame_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                classes=class_ids,
                verbose=False,
            )[0]
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
                "video_id": args.video,
                "frame_idx": numeric_frame_key(frame_path),
                "frame_path": str(frame_path.as_posix()),
                "width": int(result.orig_shape[1]),
                "height": int(result.orig_shape[0]),
                "boxes": boxes,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.save_vis:
                draw_detections(frame_path, record, visual_dir / f"frame_{record['frame_idx']:04d}.jpg")

    (output_root / f"{args.video}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
