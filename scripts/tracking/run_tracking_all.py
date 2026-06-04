#!/usr/bin/env python3
"""Run strong tracking for one or more prepared datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_CONFIGS = {
    "shanghaitech_test": {
        "dataset_root": "datasets/sha_ave_nwp/shanghaitech_test",
        "detections_dir": "datasets/sha_ave_nwp/shanghaitech_test/object_detection/detections",
    },
    "avenue_test": {
        "dataset_root": "datasets/sha_ave_nwp/avenue_test",
        "detections_dir": "datasets/sha_ave_nwp/avenue_test/object_detection/detections",
    },
    "nwpu_test": {
        "dataset_root": "datasets/sha_ave_nwp/nwpu_test",
        "detections_dir": "datasets/sha_ave_nwp/nwpu_test/object_detection/detections",
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
    parser.add_argument("--scheme-config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    tracker = root / "scripts" / "tracking" / "track_detections_strong_sort.py"
    scheme_name = "default_schema"
    if args.scheme_config:
        import json

        scheme_path = root / args.scheme_config
        if not scheme_path.exists():
            scheme_path = Path(args.scheme_config)
        scheme_name = json.loads(scheme_path.read_text(encoding="utf-8"))["scheme"]
    for dataset in args.datasets:
        cfg = DATASET_CONFIGS[dataset]
        output_dir = f"datasets/sha_ave_nwp/{dataset}/tracking/{scheme_name}"
        cmd = [
            args.python,
            str(tracker),
            "--dataset-root",
            cfg["dataset_root"],
            "--detections-dir",
            cfg["detections_dir"],
            "--output-dir",
            output_dir,
        ]
        if args.no_appearance:
            cmd.append("--no-appearance")
        if args.scheme_config:
            cmd.extend(["--scheme-config", args.scheme_config])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
