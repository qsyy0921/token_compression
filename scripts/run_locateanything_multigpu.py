import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def discover_videos(videos_root: Path, frames_root: Path) -> list[str]:
    if videos_root.exists():
        videos = [path.stem for path in videos_root.glob("*.mp4")]
    else:
        videos = [path.name for path in frames_root.iterdir() if path.is_dir()]
    return sorted(videos)


def write_list(path: Path, videos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(videos) + ("\n" if videos else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LocateAnything ShanghaiTech labeling on multiple GPUs.")
    parser.add_argument("--frames-root", default="data/shanghai/data/testing/frames")
    parser.add_argument("--videos-root", default="data/shanghai/data/testing/videos")
    parser.add_argument("--model-path", default="models/LocateAnything-3B")
    parser.add_argument("--output-root", default="outputs/locateanything_shanghai_test")
    parser.add_argument("--work-root", default="outputs/locateanything_multigpu_work")
    parser.add_argument("--gpus", default="0,2", help="Comma-separated physical GPU ids.")
    parser.add_argument("--split-counts", default="55,52", help="Comma-separated video counts per GPU.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extract-frames", action="store_true", help="Extract frames from mp4 before labeling.")
    parser.add_argument("--overwrite-frames", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    split_counts = [int(part.strip()) for part in args.split_counts.split(",") if part.strip()]
    if len(gpus) != len(split_counts):
        raise ValueError("--gpus and --split-counts must have the same length")

    frames_root = Path(args.frames_root)
    videos_root = Path(args.videos_root)
    work_root = Path(args.work_root)
    videos = discover_videos(videos_root, frames_root)
    if sum(split_counts) != len(videos):
        raise ValueError(f"split counts sum to {sum(split_counts)}, but found {len(videos)} videos")

    manifest = {
        "videos": videos,
        "gpus": gpus,
        "split_counts": split_counts,
        "jobs": [],
    }

    if args.extract_frames:
        extract_cmd = [
            args.python,
            "scripts/extract_shanghai_test_frames.py",
            "--videos-root",
            args.videos_root,
            "--frames-root",
            args.frames_root,
        ]
        if args.overwrite_frames:
            extract_cmd.append("--overwrite")
        print("extract:", " ".join(extract_cmd), flush=True)
        if not args.dry_run:
            subprocess.run(extract_cmd, check=True)

    processes = []
    start = 0
    for gpu, count in zip(gpus, split_counts):
        subset = videos[start : start + count]
        start += count
        list_path = work_root / f"gpu{gpu}_videos.txt"
        log_path = work_root / f"gpu{gpu}.log"
        write_list(list_path, subset)
        cmd = [
            args.python,
            "scripts/locateanything_label_shanghai_test.py",
            "--frames-root",
            args.frames_root,
            "--model-path",
            args.model_path,
            "--output-root",
            args.output_root,
            "--video-list",
            str(list_path),
            "--device",
            "cuda:0",
            "--dtype",
            args.dtype,
            "--generation-mode",
            args.generation_mode,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--temperature",
            str(args.temperature),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        manifest["jobs"].append({"gpu": gpu, "videos": subset, "list": str(list_path), "log": str(log_path), "cmd": cmd})
        print(f"gpu {gpu}: {len(subset)} videos -> {list_path}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        return

    failed = 0
    for proc, log in processes:
        code = proc.wait()
        log.close()
        if code != 0:
            failed += 1
    if failed:
        raise SystemExit(f"{failed} worker(s) failed")


if __name__ == "__main__":
    main()
