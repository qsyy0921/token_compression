#!/usr/bin/env python3
"""Precompute frozen Qwen3-VL frame visual tokens for an experiment sample index.

This is a cache-only helper. It does not train anything and does not change the
object anomaly prototype method. Its only job is to make the expensive ViT pass
restartable and shardable across GPUs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

from train_object_anomaly_prototypes import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_DIR,
    load_json,
    load_frame_token_cache,
    load_vision_model,
    read_video_frame,
    resize_for_qwen,
    save_frame_token_cache,
)


DEFAULT_EXP = "exp_20260613_initial_object_allframes_shortseg"


def log_stage(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def load_frame_jobs(sample_index: Path) -> list[tuple[str, int, Path]]:
    jobs: dict[tuple[str, int], Path] = {}
    with sample_index.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            package_id = str(row["package_id"])
            package_dir = Path(row["package_dir"])
            start, end = [int(x) for x in row["time_range"]]
            for frame_idx in range(start, end + 1):
                jobs[(package_id, frame_idx)] = package_dir
    return [(pkg, frame, pkg_dir) for (pkg, frame), pkg_dir in sorted(jobs.items())]


def read_package_frame_count(package_dir: Path) -> int:
    meta_path = package_dir / "tracking_metadata.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        frames = int(meta.get("summary", {}).get("frames", 0))
        if frames > 0:
            return frames
    cap = cv2.VideoCapture(str(package_dir / "video.mp4"))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return frames


def load_all_package_frame_jobs(
    data_root: Path,
    frame_stride: int,
    shard_index: int,
    shard_count: int,
    shard_mode: str,
) -> tuple[int, list[tuple[str, int, Path]]]:
    jobs: list[tuple[str, int, Path]] = []
    total = 0
    stride = max(1, int(frame_stride))
    package_dirs = [
        path
        for path in sorted((data_root / "packages").glob("*/*"))
        if path.is_dir() and (path / "video.mp4").exists()
    ]
    for package_pos, package_dir in enumerate(package_dirs):
        if not package_dir.is_dir() or not (package_dir / "video.mp4").exists():
            continue
        frame_count = read_package_frame_count(package_dir)
        package_jobs = [(package_dir.name, frame_idx, package_dir) for frame_idx in range(0, frame_count, stride)]
        total += len(package_jobs)
        if shard_mode == "package":
            if package_pos % shard_count == shard_index:
                jobs.extend(package_jobs)
        else:
            jobs.extend(
                job
                for frame_pos, job in enumerate(package_jobs)
                if (total - len(package_jobs) + frame_pos) % shard_count == shard_index
            )
    return total, jobs


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exp-name", default=DEFAULT_EXP)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--resize-long-edge", type=int, default=1280)
    parser.add_argument("--all-package-frames", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-mode", choices=["frame", "package"], default="package")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vision-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = args.data_root.parent / "cache" / "20260613_data_token_cache"
    sample_index = args.data_root / "results" / args.exp_name / "sample_index.jsonl"
    log_stage("building frame job list")
    if args.all_package_frames:
        total_frames, jobs = load_all_package_frame_jobs(
            args.data_root,
            args.frame_stride,
            args.shard_index,
            args.shard_count,
            args.shard_mode,
        )
        source = f"all package frames under {args.data_root / 'packages'}"
    else:
        all_jobs = load_frame_jobs(sample_index)
        total_frames = len(all_jobs)
        jobs = [job for i, job in enumerate(all_jobs) if i % args.shard_count == args.shard_index]
        source = str(sample_index)
    log_stage(f"built job list: shard_jobs={len(jobs)}")
    log_stage("checking existing frame token cache")
    missing = [job for job in jobs if load_frame_token_cache(args.cache_dir, job[0], job[1]) is None]
    log_stage(f"cache check done: missing={len(missing)}")

    print(
        json.dumps(
            {
                "source": source,
                "cache_dir": str(args.cache_dir),
                "total_unique_frames": total_frames,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "shard_frames": len(jobs),
                "missing_on_this_shard": len(missing),
                "device": args.device,
                "resize_long_edge": args.resize_long_edge,
                "all_package_frames": bool(args.all_package_frames),
                "frame_stride": int(args.frame_stride),
                "batch_size": int(args.batch_size),
                "shard_mode": args.shard_mode if args.all_package_frames else "frame",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not missing:
        print("nothing to cache for this shard", flush=True)
        return

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vision_dtype]
    device = torch.device(args.device)
    log_stage(f"loading processor from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)
    log_stage("processor loaded")
    log_stage(f"loading Qwen3-VL vision model to {device} with dtype={dtype}")
    vision_model, vision_config = load_vision_model(args.model_dir, device, dtype)
    log_stage("vision model loaded")
    token_size = int(vision_config.patch_size * vision_config.spatial_merge_size)
    log_stage(
        "vision config: "
        f"patch={vision_config.patch_size}, merge={vision_config.spatial_merge_size}, token_size={token_size}"
    )

    saved = 0
    skipped = 0
    failed = 0
    cap = None
    cap_package: str | None = None
    cap_next_frame: int | None = None
    try:
        batch_size = max(1, int(args.batch_size))
        log_stage(f"start caching frames: missing={len(missing)}, batch_size={batch_size}")
        for batch_start in range(0, len(missing), batch_size):
            batch_jobs = missing[batch_start : batch_start + batch_size]
            images = []
            valid_jobs: list[tuple[str, int, Path, tuple[int, int], int]] = []
            for batch_pos, (package_id, frame_idx, package_dir) in enumerate(batch_jobs, start=1):
                if load_frame_token_cache(args.cache_dir, package_id, frame_idx) is not None:
                    skipped += 1
                    continue
                if cap_package != package_id:
                    if cap is not None:
                        cap.release()
                    log_stage(f"opening video package={package_id} path={package_dir / 'video.mp4'}")
                    cap = cv2.VideoCapture(str(package_dir / "video.mp4"))
                    cap_package = package_id
                    cap_next_frame = 0
                if cap is None or not cap.isOpened():
                    failed += 1
                    print(f"[warn] failed to open {package_id}", flush=True)
                    continue
                if cap_next_frame is not None and int(frame_idx) == cap_next_frame:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(frame)
                        cap_next_frame += 1
                    else:
                        image = None
                else:
                    image = read_video_frame(cap, frame_idx)
                    cap_next_frame = int(frame_idx) + 1 if image is not None else None
                if image is None:
                    failed += 1
                    print(f"[warn] failed to read {package_id} frame {frame_idx}", flush=True)
                    continue

                orig_size = image.size
                images.append(resize_for_qwen(image, args.resize_long_edge))
                valid_jobs.append((package_id, frame_idx, package_dir, orig_size, batch_start + batch_pos))
            if not images:
                continue

            if saved == 0 and skipped == 0:
                log_stage(
                    f"first batch ready: frames={len(images)}, "
                    f"first={valid_jobs[0][0]}/{valid_jobs[0][1]:06d}, "
                    f"last={valid_jobs[-1][0]}/{valid_jobs[-1][1]:06d}"
                )
            inputs = processor.image_processor(images=images, return_tensors="pt")
            if saved == 0 and skipped == 0:
                log_stage(
                    f"first batch preprocessed: pixel_values_shape={tuple(inputs['pixel_values'].shape)}, "
                    f"grid_thw={inputs['image_grid_thw'].tolist()}"
                )
            pixel_values = inputs["pixel_values"].to(device)
            grid_thw = inputs["image_grid_thw"].to(device)
            if saved == 0 and skipped == 0:
                log_stage("first batch moved to device; running vision model")
            out = vision_model(pixel_values, grid_thw=grid_thw, return_dict=True)
            if saved == 0 and skipped == 0:
                log_stage(f"first batch vision done: pooler_output_shape={tuple(out.pooler_output.shape)}")
            split_sizes = (grid_thw.prod(dim=1) // (vision_config.spatial_merge_size**2)).tolist()
            embeds = torch.split(out.pooler_output.detach().cpu().to(torch.float16), split_sizes)
            for (package_id, frame_idx, _package_dir, orig_size, progress_n), grid, emb in zip(
                valid_jobs,
                grid_thw.detach().cpu().tolist(),
                embeds,
            ):
                _, patch_h, patch_w = [int(x) for x in grid]
                grid_w = patch_w // int(vision_config.spatial_merge_size)
                grid_h = patch_h // int(vision_config.spatial_merge_size)
                processed_w = patch_w * int(vision_config.patch_size)
                processed_h = patch_h * int(vision_config.patch_size)
                save_frame_token_cache(
                    args.cache_dir,
                    package_id,
                    frame_idx,
                    emb,
                    (grid_w, grid_h),
                    (processed_w, processed_h),
                    orig_size,
                    token_size,
                    args.model_dir,
                    args.resize_long_edge,
                )
                saved += 1
                print(
                    f"shard {args.shard_index}/{args.shard_count} cached {progress_n}/{len(missing)} "
                    f"saved={saved} skipped={skipped} failed={failed}: {package_id}/{frame_idx:06d}",
                    flush=True,
                )
    finally:
        if cap is not None:
            cap.release()

    print(
        json.dumps({"saved": saved, "skipped": skipped, "failed": failed}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
