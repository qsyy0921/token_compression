import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


DEFAULT_CATEGORIES = [
    "person",
    "pedestrian",
    "bicycle",
    "bike",
    "motorcycle",
    "motorbike",
    "scooter",
    "car",
    "vehicle",
    "skateboard",
    "bag",
    "backpack",
    "handbag",
    "suitcase",
    "luggage",
    "box",
    "package",
    "cart",
    "trolley",
    "stroller",
]


REF_RE = re.compile(r"<ref>(.*?)</ref>", re.DOTALL)
BOX4_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
BOX2_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
EVENT_RE = re.compile(
    r"(<ref>.*?</ref>)|(<box><\d+><\d+><\d+><\d+></box>)|(<box><\d+><\d+></box>)",
    re.DOTALL,
)


def numeric_frame_key(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def load_done_frames(output_path: Path) -> set[int]:
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "frame_idx" in rec:
                done.add(int(rec["frame_idx"]))
    return done


def norm_label(label: str | None) -> str | None:
    if label is None:
        return None
    label = re.sub(r"\s+", " ", label)
    label = label.replace("</c>", " ").strip().lower()
    return label or None


def parse_answer(answer: str, image_width: int, image_height: int) -> tuple[list[dict], list[dict]]:
    boxes: list[dict] = []
    points: list[dict] = []
    current_label: str | None = None

    for event in EVENT_RE.finditer(answer):
        text = event.group(0)
        ref = REF_RE.fullmatch(text)
        if ref:
            current_label = norm_label(ref.group(1))
            continue

        box4 = BOX4_RE.fullmatch(text)
        if box4:
            x1, y1, x2, y2 = [int(v) for v in box4.groups()]
            boxes.append(
                {
                    "label": current_label,
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                    "normalized": [x1, y1, x2, y2],
                }
            )
            continue

        box2 = BOX2_RE.fullmatch(text)
        if box2:
            x, y = [int(v) for v in box2.groups()]
            points.append(
                {
                    "label": current_label,
                    "x": x / 1000.0 * image_width,
                    "y": y / 1000.0 * image_height,
                    "normalized": [x, y],
                }
            )

    if not boxes:
        for match in BOX4_RE.finditer(answer):
            x1, y1, x2, y2 = [int(v) for v in match.groups()]
            boxes.append(
                {
                    "label": None,
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                    "normalized": [x1, y1, x2, y2],
                }
            )

    return boxes, points


class LocateAnythingWorker:
    def __init__(self, model_path: Path, device: str, dtype: torch.dtype, attn_implementation: str):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)
        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )
        if isinstance(response, tuple):
            return response[0]
        if isinstance(response, list):
            return response[0]
        return str(response)


def build_prompt(categories: list[str]) -> str:
    cats = "</c>".join(categories)
    return f"Locate all the instances that matches the following description: {cats}."


def load_categories(args: argparse.Namespace) -> list[str]:
    if args.categories_file:
        lines = Path(args.categories_file).read_text(encoding="utf-8").splitlines()
        categories = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    elif args.categories:
        categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    else:
        categories = DEFAULT_CATEGORIES
    if not categories:
        raise ValueError("No categories configured")
    return categories


def load_video_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def iter_videos(frames_root: Path, only_video: str | None, video_list: Path | None) -> list[Path]:
    videos = sorted([p for p in frames_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if only_video:
        videos = [p for p in videos if p.name == only_video]
    if video_list:
        allowed = load_video_list(video_list)
        videos = [p for p in videos if p.name in allowed]
    return videos


def iter_frames(video_dir: Path, image_extensions: set[str]) -> list[Path]:
    return sorted(
        [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in image_extensions],
        key=numeric_frame_key,
    )


def parse_image_extensions(raw: str) -> set[str]:
    extensions = set()
    for part in raw.split(","):
        ext = part.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        extensions.add(ext)
    return extensions


def main() -> None:
    parser = argparse.ArgumentParser(description="Label frame folders with LocateAnything.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--categories", default=None, help="Comma-separated categories.")
    parser.add_argument("--categories-file", default=None, help="One category per line.")
    parser.add_argument("--video", default=None)
    parser.add_argument("--video-list", default=None, help="Optional text file with one video id per line.")
    parser.add_argument("--image-extensions", default=".jpg,.jpeg,.png")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--generation-mode", choices=["fast", "slow", "hybrid"], default="hybrid")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", choices=["sdpa", "magi"], default="sdpa")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    frames_root = Path(args.frames_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    categories = load_categories(args)
    prompt = build_prompt(categories)
    image_extensions = parse_image_extensions(args.image_extensions)

    worker = LocateAnythingWorker(Path(args.model_path), args.device, dtype, args.attn_implementation)
    videos = iter_videos(frames_root, args.video, Path(args.video_list) if args.video_list else None)
    summary = {
        "dataset_name": args.dataset_name,
        "model_path": args.model_path,
        "frames_root": args.frames_root,
        "prompt": prompt,
        "categories": categories,
        "generation_mode": args.generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "videos": [],
    }

    for video_dir in videos:
        out_path = output_root / f"{video_dir.name}.jsonl"
        if args.overwrite and out_path.exists():
            out_path.unlink()
        done = load_done_frames(out_path)
        frames = iter_frames(video_dir, image_extensions)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        start = time.time()
        processed = 0
        total_time = 0.0
        total_tokens = 0
        with out_path.open("a", encoding="utf-8") as handle:
            window_time = 0.0
            window_tokens = 0
            for frame_path in frames:
                frame_idx = numeric_frame_key(frame_path)
                if frame_idx in done:
                    continue
                image = Image.open(frame_path).convert("RGB")
                frame_start = time.time()
                answer = worker.predict(
                    image,
                    prompt,
                    generation_mode=args.generation_mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                generation_seconds = time.time() - frame_start
                output_tokens = len(worker.tokenizer.encode(answer, add_special_tokens=False))
                window_time += generation_seconds
                window_tokens += output_tokens
                total_time += generation_seconds
                total_tokens += output_tokens
                boxes, points = parse_answer(answer, image.width, image.height)
                record = {
                    "dataset_name": args.dataset_name,
                    "video_id": video_dir.name,
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path.as_posix()),
                    "width": image.width,
                    "height": image.height,
                    "prompt": prompt,
                    "answer": answer,
                    "generation_seconds": generation_seconds,
                    "output_tokens": output_tokens,
                    "boxes": boxes,
                    "points": points,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                processed += 1
                if processed % 10 == 0:
                    elapsed = time.time() - start
                    window_tps = window_tokens / window_time if window_time > 0 else 0.0
                    avg_frame_time = window_time / 10.0
                    total_tps = total_tokens / total_time if total_time > 0 else 0.0
                    print(
                        f"{video_dir.name}: processed {processed} new frames in {elapsed:.1f}s; "
                        f"last10_time={window_time:.2f}s avg_frame={avg_frame_time:.2f}s "
                        f"last10_output_tokens={window_tokens} last10_tok_s={window_tps:.2f} "
                        f"total_tok_s={total_tps:.2f}",
                        flush=True,
                    )
                    window_time = 0.0
                    window_tokens = 0
        summary["videos"].append(
            {
                "video_id": video_dir.name,
                "frames_considered": len(frames),
                "new_frames_processed": processed,
                "generation_seconds": total_time,
                "output_tokens": total_tokens,
                "output_tokens_per_second": total_tokens / total_time if total_time > 0 else 0.0,
                "output": str(out_path.as_posix()),
            }
        )

    summary_path = output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
