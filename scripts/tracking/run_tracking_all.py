#!/usr/bin/env python3
"""Run strong tracking for one or more prepared datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_CONFIGS = {
    "shanghaitech_test": {
        "dataset_root": "datasets/shanghaitech_test",
        "detections_dir": "datasets/shanghaitech_test/detections",
        "output_dir": "datasets/shanghaitech_test/tracks_strong_sort",
    },
    "avenue_test": {
        "dataset_root": "datasets/avenue_test",
        "detections_dir": "datasets/avenue_test/detections",
        "output_dir": "datasets/avenue_test/tracks_strong_sort",
    },
    "nwpu_test": {
        "dataset_root": "datasets/nwpu_test",
        "detections_dir": "datasets/nwpu_test/yolo26x_detections",
        "output_dir": "datasets/nwpu_test/tracks_strong_sort",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["shanghaitech_test", "avenue_test", "nwpu_test"],
        choices=sorted(DATASET_CONFIGS),
    )
    parser.add_argument(
        "--python",
        default="/home/lcwt/.conda/envs/token_pruner_merge/bin/python",
    )
    parser.add_argument("--no-appearance", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    tracker = root / "scripts" / "tracking" / "track_detections_strong_sort.py"
    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        cmd = [
            args.python,
            str(tracker),
            "--dataset-root",
            cfg["dataset_root"],
            "--detections-dir",
            cfg["detections_dir"],
            "--output-dir",
            cfg["output_dir"],
        ]
        if args.no_appearance:
            cmd.append("--no-appearance")
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
