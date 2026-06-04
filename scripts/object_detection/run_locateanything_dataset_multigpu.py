import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def discover_videos(frames_root: Path) -> list[str]:
    return sorted([path.name for path in frames_root.iterdir() if path.is_dir()])


def split_evenly(videos: list[str], parts: int) -> list[list[str]]:
    splits = [[] for _ in range(parts)]
    for idx, video in enumerate(videos):
        splits[idx % parts].append(video)
    return splits


def write_list(path: Path, videos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(videos) + ("\n" if videos else ""), encoding="utf-8")


def free_memory_mb(gpu: str) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            gpu,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return int(result.stdout.strip().splitlines()[0])


def wait_for_memory(gpu: str, min_free_mb: int, poll_seconds: int, log_path: Path) -> None:
    if min_free_mb <= 0:
        return
    while True:
        try:
            free_mb = free_memory_mb(gpu)
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"memory check failed for gpu {gpu}: {exc}\n")
            time.sleep(poll_seconds)
            continue
        if free_mb >= min_free_mb:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"gpu {gpu} free memory {free_mb} MiB >= {min_free_mb} MiB; starting worker\n")
            return
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"gpu {gpu} free memory {free_mb} MiB < {min_free_mb} MiB; waiting {poll_seconds}s\n")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LocateAnything labeling over frame folders on multiple GPUs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--categories-file", required=True)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU ids. Repeat ids for multiple workers.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-free-mb", type=int, default=22000)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = [part.strip() for part in args.gpus.split(",") if part.strip()]
    frames_root = Path(args.frames_root)
    work_root = Path(args.work_root)
    output_root = Path(args.output_root)
    videos = discover_videos(frames_root)
    splits = split_evenly(videos, len(gpus))

    manifest = {
        "dataset": args.dataset,
        "frames_root": args.frames_root,
        "model_path": args.model_path,
        "output_root": args.output_root,
        "categories_file": args.categories_file,
        "gpus": gpus,
        "jobs": [],
    }
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    processes = []
    for job_idx, (gpu, subset) in enumerate(zip(gpus, splits)):
        job_name = f"job{job_idx}_gpu{gpu}"
        list_path = work_root / f"{job_name}_videos.txt"
        log_path = work_root / f"{job_name}.log"
        write_list(list_path, subset)
        cmd = [
            args.python,
            "scripts/locateanything_label_frames.py",
            "--dataset",
            args.dataset,
            "--frames-root",
            args.frames_root,
            "--model-path",
            args.model_path,
            "--output-root",
            args.output_root,
            "--categories-file",
            args.categories_file,
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
        manifest["jobs"].append(
            {
                "job_idx": job_idx,
                "gpu": gpu,
                "videos": subset,
                "list": str(list_path),
                "log": str(log_path),
                "cmd": cmd,
            }
        )
        print(f"job {job_idx} gpu {gpu}: {len(subset)} videos -> {list_path}", flush=True)
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            wait_for_memory(gpu, args.min_free_mb, args.poll_seconds, log_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log = log_path.open("a", encoding="utf-8")
            processes.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))

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
