import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "person": (0, 220, 120),
    "pedestrian": (0, 220, 120),
    "bicycle": (255, 170, 0),
    "bike": (255, 170, 0),
    "motorcycle": (255, 80, 0),
    "motorbike": (255, 80, 0),
    "scooter": (255, 120, 0),
    "car": (255, 40, 40),
    "vehicle": (255, 40, 40),
    "skateboard": (190, 80, 255),
    "bag": (80, 160, 255),
    "backpack": (80, 160, 255),
    "box": (255, 220, 80),
    "cart": (255, 220, 80),
}


ALIASES = {
    "pedestrian": "person",
    "people": "person",
    "bike": "bicycle",
    "motorbike": "motorcycle",
    "vehicle": "car",
    "backpack": "bag",
    "handbag": "bag",
    "package": "box",
    "trolley": "cart",
    "stroller": "cart",
}


def canonical_label(label: str | None) -> str:
    if not label:
        return "unknown"
    label = label.lower().strip()
    return ALIASES.get(label, label)


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return sorted(records, key=lambda rec: int(rec["frame_idx"]))


def choose_records(records: list[dict], frames: str, max_images: int) -> list[dict]:
    if frames == "all":
        return records
    if frames == "sample":
        if len(records) <= max_images:
            return records
        idxs = sorted({round(i * (len(records) - 1) / (max_images - 1)) for i in range(max_images)})
        return [records[i] for i in idxs]
    requested = {int(part.strip()) for part in frames.split(",") if part.strip()}
    return [rec for rec in records if int(rec["frame_idx"]) in requested]


def draw_record(record: dict, output_path: Path) -> None:
    image = Image.open(record["frame_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for box in record.get("boxes", []):
        label = canonical_label(box.get("label"))
        color = PALETTE.get(label, (255, 255, 255))
        xy = [box["x1"], box["y1"], box["x2"], box["y2"]]
        draw.rectangle(xy, outline=color, width=3)
        text = label
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = xy[0], max(0, xy[1] - th - 4)
        draw.rectangle([x, y, x + tw + 6, y + th + 4], fill=color)
        draw.text((x + 3, y + 2), text, fill=(0, 0, 0), font=font)

    title = f"{record['video_id']} frame {record['frame_idx']} boxes={len(record.get('boxes', []))}"
    draw.rectangle([0, 0, image.width, 24], fill=(0, 0, 0))
    draw.text((6, 4), title, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def make_grid(paths: list[Path], output_path: Path, cols: int) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((320, 180))
        canvas = Image.new("RGB", (320, 180), (20, 20, 20))
        canvas.paste(img, ((320 - img.width) // 2, (180 - img.height) // 2))
        thumbs.append(canvas)
    rows = (len(thumbs) + cols - 1) // cols
    grid = Image.new("RGB", (cols * 320, rows * 180), (20, 20, 20))
    for idx, img in enumerate(thumbs):
        grid.paste(img, ((idx % cols) * 320, (idx // cols) * 180))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def make_gif(paths: list[Path], output_path: Path, duration: int) -> None:
    if not paths:
        return
    frames = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((640, 360))
        canvas = Image.new("RGB", (640, 360), (20, 20, 20))
        canvas.paste(img, ((640 - img.width) // 2, (360 - img.height) // 2))
        frames.append(canvas)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LocateAnything detection JSONL.")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output-root", default="outputs/visualizations")
    parser.add_argument("--frames", default="sample", help="'sample', 'all', or comma-separated frame indices.")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--gif-duration", type=int, default=300)
    args = parser.parse_args()

    det_path = Path(args.detections)
    records = load_records(det_path)
    selected = choose_records(records, args.frames, args.max_images)
    out_dir = Path(args.output_root) / det_path.stem
    drawn_paths = []
    for record in selected:
        out_path = out_dir / f"frame_{int(record['frame_idx']):04d}.jpg"
        draw_record(record, out_path)
        drawn_paths.append(out_path)

    make_grid(drawn_paths, out_dir / "grid.jpg", args.grid_cols)
    if args.gif:
        make_gif(drawn_paths, out_dir / "sample.gif", args.gif_duration)

    summary = {
        "detections": str(det_path.as_posix()),
        "output_dir": str(out_dir.as_posix()),
        "frames_visualized": [int(rec["frame_idx"]) for rec in selected],
        "images": [str(path.as_posix()) for path in drawn_paths],
        "grid": str((out_dir / "grid.jpg").as_posix()),
        "gif": str((out_dir / "sample.gif").as_posix()) if args.gif else None,
    }
    (out_dir / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
