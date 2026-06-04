import argparse
from pathlib import Path

import cv2


def extract_video(video_path: Path, output_dir: Path, overwrite: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.jpg"))
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
            out_path = output_dir / f"{frame_idx:03d}.jpg"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            frame_idx += 1
    finally:
        cap.release()
    return frame_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ShanghaiTech test videos into frame folders.")
    parser.add_argument("--videos-root", default="data/shanghai/data/testing/videos")
    parser.add_argument("--frames-root", default="data/shanghai/data/testing/frames")
    parser.add_argument("--video-list", default=None, help="Optional text file with one video id per line.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    frames_root = Path(args.frames_root)
    allowed = None
    if args.video_list:
        allowed = {line.strip() for line in Path(args.video_list).read_text(encoding="utf-8").splitlines() if line.strip()}

    videos = sorted(videos_root.glob("*.mp4"), key=lambda p: p.stem)
    if allowed is not None:
        videos = [path for path in videos if path.stem in allowed]

    total = 0
    for video_path in videos:
        count = extract_video(video_path, frames_root / video_path.stem, args.overwrite)
        total += count
        print(f"{video_path.stem}: {count} frames", flush=True)

    print(f"done: videos={len(videos)} frames={total}", flush=True)


if __name__ == "__main__":
    main()
