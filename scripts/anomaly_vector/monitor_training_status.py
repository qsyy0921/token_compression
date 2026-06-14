#!/usr/bin/env python3
"""Summarize the 20260613 anomaly prototype training run."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/expand_disk/data_repository/mfl/token_compression/20260613_data")
DEFAULT_EXP = "exp_20260613_initial_object_allframes_shortseg"
DEFAULT_CACHE = DEFAULT_DATA_ROOT.parent / "cache" / "20260613_data_token_cache"


def run_text(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        out = exc.output
    except FileNotFoundError:
        out = ""
    return out.strip()


def tail(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-n:]


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_predictions(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    pred = Counter(r.get("pred_label") for r in rows)
    true = Counter(r.get("true_label") for r in rows)
    scores = [float(r.get("object_anomaly_score", 0.0)) for r in rows]
    return {
        "exists": True,
        "n": len(rows),
        "true": dict(true),
        "pred": dict(pred),
        "score_min": min(scores) if scores else None,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_max": max(scores) if scores else None,
    }


def summarize_feature_meta(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    totals = [int(r.get("total_object_tokens", 0)) for r in rows]
    frames = [int(r.get("used_frames", 0)) for r in rows]
    return {
        "exists": True,
        "n": len(rows),
        "total_object_tokens_min": min(totals) if totals else None,
        "total_object_tokens_mean": sum(totals) / len(totals) if totals else None,
        "total_object_tokens_max": max(totals) if totals else None,
        "used_frames_min": min(frames) if frames else None,
        "used_frames_mean": sum(frames) / len(frames) if frames else None,
        "used_frames_max": max(frames) if frames else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--exp-name", default=DEFAULT_EXP)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tmux", default="token_anomaly_20260613")
    parser.add_argument("--expected-frames", type=int, default=144)
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()

    exp_dir = args.data_root / "results" / args.exp_name
    log_path = exp_dir / "tmux_train.log"
    cache_count = len(list(args.cache_dir.rglob("*.pt"))) if args.cache_dir.exists() else 0
    artifacts = {
        "feature_cache.pt": (exp_dir / "feature_cache.pt").exists(),
        "feature_meta.jsonl": (exp_dir / "feature_meta.jsonl").exists(),
        "metrics.json": (exp_dir / "metrics.json").exists(),
        "report.md": (exp_dir / "report.md").exists(),
        "visualizations": (exp_dir / "visualizations").exists(),
        "code_backup": (
            Path("/home/expand_disk/code_repository/mfl/token_compression")
            / "outputs"
            / "20260613"
            / "results"
            / args.exp_name
        ).exists(),
    }
    ablations = {}
    for name in ("visual_only", "visual_motion", "token_topk", "token_logmeanexp"):
        ab_dir = exp_dir / name
        if ab_dir.exists():
            ablations[name] = {
                "metrics": load_json(ab_dir / "metrics.json"),
                "predictions": summarize_predictions(ab_dir / "predictions.jsonl"),
            }

    status = {
        "tmux": run_text(["tmux", "ls"]),
        "process": run_text(
            [
                "bash",
                "-lc",
                "ps -eo pid,stat,etime,pcpu,pmem,args | "
                "rg 'train_object_anomaly_prototypes|precache_qwen_frame_tokens|run_20260613_precache' | rg -v rg || true",
            ]
        ),
        "exp_dir": str(exp_dir),
        "frame_token_cache": {
            "dir": str(args.cache_dir),
            "count": cache_count,
            "expected": args.expected_frames,
            "progress": cache_count / args.expected_frames if args.expected_frames else None,
        },
        "artifacts": artifacts,
        "feature_meta": summarize_feature_meta(exp_dir / "feature_meta.jsonl"),
        "root_metrics": load_json(exp_dir / "metrics.json"),
        "ablations": ablations,
        "log_tail": tail(log_path, args.tail),
        "precache_log_tail": {
            "shard0_gpu0": tail(exp_dir / "precache_shard0_gpu0.log", args.tail),
            "shard1_gpu1": tail(exp_dir / "precache_shard1_gpu1.log", args.tail),
        },
    }
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
