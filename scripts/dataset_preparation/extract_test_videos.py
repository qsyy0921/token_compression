import argparse
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv")


def load_video_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def discover_videos(videos_root: Path, allowed: set[str] | None) -> list[Path]:
    videos = [path for path in videos_root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    if allowed is not None:
        videos = [path for path in videos if path.stem in allowed]
    return sorted(videos, key=lambda path: path.stem)


def extract_video(video_path: Path, output_dir: Path, overwrite: bool, width: int | None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        return len(existing)

    if overwrite:
        for path in existing:
            path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if width is not None and frame.shape[1] != width:
                scale = width / float(frame.shape[1])
                height = max(1, round(frame.shape[0] * scale))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out_path = output_dir / f"{frame_idx:04d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            frame_idx += 1
    finally:
        cap.release()
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract test videos into per-video frame folders.")
    parser.add_argument("--videos-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--video-list", default=None, help="Optional text file with one video id per line.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--width", type=int, default=None, help="Optional resize width while preserving aspect ratio.")
    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    frames_root = Path(args.frames_root)
    allowed = load_video_list(Path(args.video_list) if args.video_list else None)
    videos = discover_videos(videos_root, allowed)

    total = 0
    for video_path in videos:
        count = extract_video(video_path, frames_root / video_path.stem, args.overwrite, args.width)
        total += count
        print(f"{video_path.stem}: {count} frames", flush=True)

    print(f"done: videos={len(videos)} frames={total}", flush=True)


if __name__ == "__main__":
    main()
