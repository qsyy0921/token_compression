import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def discover_videos(frames_root: Path) -> list[str]:
    return sorted([path.name for path in frames_root.iterdir() if path.is_dir()])


def split_evenly(items: list[str], parts: int) -> list[list[str]]:
    buckets: list[list[str]] = []
    start = 0
    for idx in range(parts):
        count = len(items) // parts + (1 if idx < len(items) % parts else 0)
        buckets.append(items[start : start + count])
        start += count
    return buckets


def split_by_counts(items: list[str], counts: list[int]) -> list[list[str]]:
    if sum(counts) != len(items):
        raise ValueError(f"split counts sum to {sum(counts)}, but found {len(items)} videos")
    buckets = []
    start = 0
    for count in counts:
        buckets.append(items[start : start + count])
        start += count
    return buckets


def write_list(path: Path, videos: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(videos) + ("\n" if videos else ""), encoding="utf-8")


def wait_for_gpu_memory(gpu: str, min_free_mib: int, poll_seconds: int) -> None:
    if min_free_mib <= 0:
        return
    query = [
        "nvidia-smi",
        "--id",
        gpu,
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    while True:
        try:
            out = subprocess.check_output(query, text=True).strip().splitlines()
            free_mib = int(out[0].strip())
        except Exception as exc:
            print(f"GPU {gpu}: failed to query memory ({exc}); retrying", flush=True)
            time.sleep(poll_seconds)
            continue
        if free_mib >= min_free_mib:
            print(f"GPU {gpu}: free memory {free_mib} MiB >= {min_free_mib} MiB", flush=True)
            return
        print(f"GPU {gpu}: free memory {free_mib} MiB < {min_free_mib} MiB; waiting", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LocateAnything labeling on multiple GPU workers.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--categories-file", required=True)
    parser.add_argument("--gpus", default="0,0,1,1", help="Comma-separated physical GPU ids; repeat ids for multiple workers.")
    parser.add_argument("--split-counts", default=None, help="Optional comma-separated video counts per worker.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--generation-mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--min-free-mib", type=int, default=22000)
    parser.add_argument("--memory-poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpus = parse_csv(args.gpus)
    if not gpus:
        raise ValueError("No GPUs configured")

    frames_root = Path(args.frames_root)
    work_root = Path(args.work_root)
    videos = discover_videos(frames_root)
    if args.split_counts:
        buckets = split_by_counts(videos, [int(part) for part in parse_csv(args.split_counts)])
        if len(buckets) != len(gpus):
            raise ValueError("--split-counts and --gpus must have the same length")
    else:
        buckets = split_evenly(videos, len(gpus))

    manifest = {
        "dataset_name": args.dataset_name,
        "videos": videos,
        "gpus": gpus,
        "jobs": [],
        "min_free_mib": args.min_free_mib,
    }

    processes = []
    for job_idx, (gpu, subset) in enumerate(zip(gpus, buckets)):
        job_name = f"job{job_idx}_gpu{gpu}"
        list_path = work_root / f"{job_name}_videos.txt"
        log_path = work_root / f"{job_name}.log"
        write_list(list_path, subset)
        cmd = [
            args.python,
            "scripts/object_detection/locateanything_label_dataset.py",
            "--dataset-name",
            args.dataset_name,
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
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
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
            wait_for_gpu_memory(gpu, args.min_free_mib, args.memory_poll_seconds)
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
