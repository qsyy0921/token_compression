import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tracking after all ShanghaiTech detections are available.")
    parser.add_argument("--detections-root", default="outputs/locateanything_shanghai_test")
    parser.add_argument("--output-root", default="outputs/shanghai_iou_tracks")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--max-age", type=int, default=5)
    args = parser.parse_args()

    det_root = Path(args.detections_root)
    videos = sorted(path.stem for path in det_root.glob("*.jsonl"))
    for video in videos:
        cmd = [
            args.python,
            "scripts/track_shanghai_objects_iou.py",
            "--detections-root",
            args.detections_root,
            "--output-root",
            args.output_root,
            "--video",
            video,
            "--min-iou",
            str(args.min_iou),
            "--max-age",
            str(args.max_age),
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
