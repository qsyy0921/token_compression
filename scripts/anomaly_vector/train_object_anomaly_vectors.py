#!/usr/bin/env python3
"""Train object-level anomaly vectors on short-video packages.

Strategy:
- freeze Qwen3-VL visual tokens and bbox-to-token binding
- train only anomaly vectors/projection heads
- normal is not a prototype class; low anomaly evidence means normal
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_object_anomaly_prototypes import (
    DEFAULT_CODE_ROOT,
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL_DIR,
    LABELS,
    TRAIN_CATEGORIES,
    SampleSpec,
    build_samples,
    draw_sample_visualizations,
    extract_sample_features,
    load_json,
)


ANOMALY_LABELS = LABELS[1:]


def read_frame_count(package_dir: Path) -> int:
    meta = load_json(package_dir / "tracking_metadata.json")
    return int(meta.get("summary", {}).get("frames", 10**9))


def package_event_labels(package_dir: Path) -> set[str]:
    ann = load_json(package_dir / "annotation.json")
    labels = set()
    for ev in ann.get("events", []):
        label = str(ev.get("train_category"))
        if label == "R06" or (ev.get("trainable", True) and label):
            labels.add(label)
    return labels


def find_package_dirs(data_root: Path) -> dict[str, Path]:
    out = {}
    for path in sorted((data_root / "packages").glob("*/*")):
        if path.is_dir() and (path / "annotation.json").exists() and (path / "tracks.jsonl").exists():
            out[path.name] = path
    return out


def select_short_packages(
    data_root: Path,
    train_count: int,
    val_count: int,
    openset_count: int,
    max_frames: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    package_dirs = find_package_dirs(data_root)
    rows = []
    for package_id, path in package_dirs.items():
        labels = package_event_labels(path)
        if not labels:
            continue
        frames = read_frame_count(path)
        if frames > max_frames and "R06" not in labels:
            continue
        rows.append(
            {
                "package_id": package_id,
                "dataset": path.parent.name,
                "frames": frames,
                "labels": sorted(labels),
                "path": str(path),
            }
        )
    rows = sorted(rows, key=lambda r: (r["frames"], r["package_id"]))

    selected_train: list[dict] = []
    selected_val: list[dict] = []
    used = set()

    def add_round_robin(target: list[dict], target_count: int, labels: list[str]) -> None:
        while len(target) < target_count:
            before = len(target)
            for label in labels:
                if len(target) >= target_count:
                    break
                candidates = [
                    r for r in rows
                    if r["package_id"] not in used and label in r["labels"] and label in TRAIN_CATEGORIES
                ]
                if not candidates:
                    continue
                # Small deterministic jitter among similarly short videos.
                head = candidates[: min(6, len(candidates))]
                chosen = rng.choice(head)
                target.append(chosen)
                used.add(chosen["package_id"])
            if len(target) == before:
                break

    add_round_robin(selected_train, train_count, ANOMALY_LABELS)
    add_round_robin(selected_val, val_count, ANOMALY_LABELS)

    # Fill if some labels are scarce under max_frames.
    for target, target_count in ((selected_train, train_count), (selected_val, val_count)):
        for row in rows:
            if len(target) >= target_count:
                break
            if row["package_id"] not in used and any(label in TRAIN_CATEGORIES for label in row["labels"]):
                target.append(row)
                used.add(row["package_id"])

    openset = []
    preferred_openset = ["NWPU_D003_05", "NWPU_D038_02", "NWPU_D013_01"]
    by_id = {row["package_id"]: row for row in rows}
    for package_id in preferred_openset:
        row = by_id.get(package_id)
        if row is not None and row["package_id"] not in used and "R06" in row["labels"]:
            openset.append(row)
            used.add(row["package_id"])
            if len(openset) >= openset_count:
                break
    for row in rows:
        if len(openset) >= openset_count:
            break
        if row["package_id"] not in used and "R06" in row["labels"]:
            openset.append(row)
            used.add(row["package_id"])

    return {
        "train": selected_train,
        "val": selected_val,
        "openset": openset,
        "selection_params": {
            "train_count": train_count,
            "val_count": val_count,
            "openset_count": openset_count,
            "max_frames": max_frames,
            "seed": seed,
        },
    }


class AnomalyVectorScorer(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, anomaly_vectors: int, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )
        self.anomaly = nn.Parameter(torch.randn(anomaly_vectors, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.proj(x), dim=-1)
        a = F.normalize(self.anomaly, dim=-1)
        logits = q @ a.t() * self.tau
        evidence = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
        return evidence, logits


class TokenAnomalyVectorScorer(AnomalyVectorScorer):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        anomaly_vectors: int,
        tau: float,
        pooling: str,
        topk_ratio: float,
        lme_alpha: float,
    ) -> None:
        super().__init__(in_dim, embed_dim, anomaly_vectors, tau)
        self.pooling = pooling
        self.topk_ratio = topk_ratio
        self.lme_alpha = lme_alpha

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return scores.mean()
        if self.pooling == "topk":
            k = max(1, int(math.ceil(scores.shape[0] * self.topk_ratio)))
            return scores.topk(min(k, scores.shape[0])).values.mean()
        if self.pooling == "logmeanexp":
            alpha = float(self.lme_alpha)
            return torch.logsumexp(alpha * scores, dim=0) / alpha - math.log(scores.shape[0]) / alpha
        raise ValueError(self.pooling)

    def forward_one(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, logits = super().forward(tokens.float())
        token_evidence = torch.logsumexp(logits, dim=1) - math.log(logits.shape[1])
        evidence = self.aggregate(token_evidence)
        return evidence, logits


class PromptAlignmentScorer(nn.Module):
    """AnomalyCLIP-style normal/anomaly prompt alignment for object embeddings."""

    def __init__(self, in_dim: int, embed_dim: int, tau: float) -> None:
        super().__init__()
        self.tau = tau
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )
        self.normal_prompt = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.anomaly_prompt = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def prompts(self) -> torch.Tensor:
        return torch.stack([self.normal_prompt, self.anomaly_prompt], dim=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.proj(x), dim=-1)
        p = F.normalize(self.prompts(), dim=-1)
        logits = q @ p.t() * self.tau
        probs = F.softmax(logits, dim=1)
        return probs[:, 1], logits


class TokenPromptAlignmentScorer(PromptAlignmentScorer):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        tau: float,
        pooling: str,
        topk_ratio: float,
        lme_alpha: float,
    ) -> None:
        super().__init__(in_dim, embed_dim, tau)
        self.pooling = pooling
        self.topk_ratio = topk_ratio
        self.lme_alpha = lme_alpha

    def aggregate(self, scores: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return scores.mean()
        if self.pooling == "topk":
            k = max(1, int(math.ceil(scores.shape[0] * self.topk_ratio)))
            return scores.topk(min(k, scores.shape[0])).values.mean()
        if self.pooling == "logmeanexp":
            alpha = float(self.lme_alpha)
            return torch.logsumexp(alpha * scores, dim=0) / alpha - math.log(scores.shape[0]) / alpha
        raise ValueError(self.pooling)

    def forward_one(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, logits = super().forward(tokens.float())
        token_scores = F.softmax(logits, dim=1)[:, 1]
        score = self.aggregate(token_scores)
        return score, logits


def vector_separation_loss(vectors: torch.Tensor, margin: float) -> torch.Tensor:
    p = F.normalize(vectors, dim=-1)
    sim = p @ p.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return torch.relu(sim[mask] - margin).pow(2).mean()


def prompt_separation_loss(model: PromptAlignmentScorer, margin: float) -> torch.Tensor:
    p = F.normalize(model.prompts(), dim=-1)
    sim = p[0] @ p[1]
    return torch.relu(sim - margin).pow(2)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    total = 0
    for p in pos:
        wins += float((p > neg).sum()) + 0.5 * float((p == neg).sum())
        total += len(neg)
    return wins / max(1, total)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, metas: list[dict], threshold: float) -> dict:
    pred = scores >= threshold
    is_anom = y_true == 1
    normal = ~is_anom
    metrics = {
        "n": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float((pred == is_anom).mean()) if len(y_true) else 0.0,
        "anomaly_recall": float((pred & is_anom).sum() / max(1, is_anom.sum())),
        "normal_false_positive_rate": float((pred & normal).sum() / max(1, normal.sum())),
        "normal_true_negative_rate": float(((~pred) & normal).sum() / max(1, normal.sum())),
        "auroc": auroc(y_true, scores),
    }
    by_label = {}
    for label in LABELS + ["R06"]:
        mask = np.array([m.get("label") == label for m in metas], dtype=bool)
        if mask.any():
            by_label[label] = {
                "count": int(mask.sum()),
                "mean_score": float(scores[mask].mean()),
                "pred_anomaly_rate": float(pred[mask].mean()),
            }
    metrics["per_label"] = by_label

    by_event = defaultdict(list)
    for i, meta in enumerate(metas):
        if meta.get("event_id"):
            by_event[(meta["package_id"], meta["event_id"])].append(i)
    top1 = top3 = total = 0
    for _, idxs in by_event.items():
        pos = [i for i in idxs if metas[i].get("is_positive")]
        if not pos or len(idxs) <= 1:
            continue
        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)
        total += 1
        top1 += int(ranked[0] in pos)
        top3 += int(any(i in pos for i in ranked[: min(3, len(ranked))]))
    metrics["event_top1_recall"] = top1 / max(1, total)
    metrics["event_top3_recall"] = top3 / max(1, total)
    metrics["event_ranking_count"] = total
    return metrics


def train_feature_ablation(name: str, x: torch.Tensor, metas: list[dict], out_dir: Path, args) -> dict:
    y_all = torch.tensor([1 if m["label"] in TRAIN_CATEGORIES else 0 for m in metas], dtype=torch.float32)
    supervised = torch.tensor([m["label"] in TRAIN_CATEGORIES or m["label"] == "normal" for m in metas], dtype=torch.bool)
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]

    device = torch.device(args.train_device)
    model = AnomalyVectorScorer(x.shape[1], args.embed_dim, args.anomaly_vectors, args.tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(x[train_idx].float(), y_all[train_idx]), batch_size=args.batch_size, shuffle=True)
    history = []
    best = {"val_balanced_accuracy": -1.0, "epoch": -1}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            evidence, _ = model(xb)
            bce = F.binary_cross_entropy_with_logits(evidence, yb)
            sep = vector_separation_loss(model.anomaly, args.sep_margin)
            loss = bce + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item()) * len(xb)
            totals["bce"] += float(bce.item()) * len(xb)
            totals["sep"] += float(sep.item()) * len(xb)
            totals["n"] += len(xb)
        model.eval()
        with torch.no_grad():
            val_e, _ = model(x[val_idx].float().to(device))
            val_scores = torch.sigmoid(val_e).cpu().numpy()
        val_y = y_all[val_idx].numpy().astype(np.int64)
        val_metas = [metas[i] for i in val_idx]
        vm = binary_metrics(val_y, val_scores, val_metas, args.threshold) if val_idx else {}
        bal = 0.5 * (vm.get("anomaly_recall", 0.0) + vm.get("normal_true_negative_rate", 0.0))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "bce": totals["bce"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_balanced_accuracy": bal,
            "val_anomaly_recall": vm.get("anomaly_recall", 0.0),
            "val_normal_fpr": vm.get("normal_false_positive_rate", 0.0),
            "val_auroc": vm.get("auroc"),
            "val_event_top3_recall": vm.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if bal > best["val_balanced_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        evidence, logits = model(x.float().to(device))
        scores = torch.sigmoid(evidence).cpu().numpy()
        top_vec = logits.argmax(dim=1).cpu().numpy()
    return write_outputs(name, model, history, best, scores, top_vec, y_all.numpy().astype(np.int64), metas, out_dir, args, train_idx, val_idx, openset_idx)


def train_token_ablation(name: str, token_sets: list[torch.Tensor], metas: list[dict], out_dir: Path, args) -> dict:
    y_all = torch.tensor([1 if m["label"] in TRAIN_CATEGORIES else 0 for m in metas], dtype=torch.float32)
    supervised = torch.tensor([m["label"] in TRAIN_CATEGORIES or m["label"] == "normal" for m in metas], dtype=torch.bool)
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]
    device = torch.device(args.train_device)
    model = TokenAnomalyVectorScorer(
        token_sets[0].shape[1],
        args.embed_dim,
        args.anomaly_vectors,
        args.tau,
        args.token_pooling,
        args.token_topk_ratio,
        args.token_lme_alpha,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    def predict(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        scores = []
        top_vec = []
        model.eval()
        with torch.no_grad():
            for i in indices:
                e, logits = model.forward_one(token_sets[i].to(device))
                scores.append(torch.sigmoid(e).detach().cpu())
                top_vec.append(int(logits.max(dim=0).values.argmax().detach().cpu()))
        return torch.stack(scores).numpy() if scores else np.zeros((0,), dtype=np.float32), np.asarray(top_vec)

    history = []
    best = {"val_balanced_accuracy": -1.0, "epoch": -1}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        order = train_idx[:]
        rng.shuffle(order)
        model.train()
        totals = Counter()
        for i in order:
            yb = y_all[i].to(device)
            e, _ = model.forward_one(token_sets[i].to(device))
            bce = F.binary_cross_entropy_with_logits(e[None], yb[None])
            sep = vector_separation_loss(model.anomaly, args.sep_margin)
            loss = bce + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item())
            totals["bce"] += float(bce.item())
            totals["sep"] += float(sep.item())
            totals["n"] += 1
        val_scores, _ = predict(val_idx)
        val_y = y_all[val_idx].numpy().astype(np.int64)
        val_metas = [metas[i] for i in val_idx]
        vm = binary_metrics(val_y, val_scores, val_metas, args.threshold) if val_idx else {}
        bal = 0.5 * (vm.get("anomaly_recall", 0.0) + vm.get("normal_true_negative_rate", 0.0))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "bce": totals["bce"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_balanced_accuracy": bal,
            "val_anomaly_recall": vm.get("anomaly_recall", 0.0),
            "val_normal_fpr": vm.get("normal_false_positive_rate", 0.0),
            "val_auroc": vm.get("auroc"),
            "val_event_top3_recall": vm.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if bal > best["val_balanced_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    all_idx = list(range(len(metas)))
    scores, top_vec = predict(all_idx)
    return write_outputs(name, model, history, best, scores, top_vec, y_all.numpy().astype(np.int64), metas, out_dir, args, train_idx, val_idx, openset_idx)


def train_prompt_alignment(name: str, x: torch.Tensor, metas: list[dict], out_dir: Path, args) -> dict:
    y_all = torch.tensor([1 if m["label"] in TRAIN_CATEGORIES else 0 for m in metas], dtype=torch.long)
    supervised = torch.tensor([m["label"] in TRAIN_CATEGORIES or m["label"] == "normal" for m in metas], dtype=torch.bool)
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]

    device = torch.device(args.train_device)
    model = PromptAlignmentScorer(x.shape[1], args.embed_dim, args.tau).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(x[train_idx].float(), y_all[train_idx]), batch_size=args.batch_size, shuffle=True)
    history = []
    best = {"val_balanced_accuracy": -1.0, "epoch": -1}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            sep = prompt_separation_loss(model, args.sep_margin)
            loss = ce + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item()) * len(xb)
            totals["ce"] += float(ce.item()) * len(xb)
            totals["sep"] += float(sep.item()) * len(xb)
            totals["n"] += len(xb)
        model.eval()
        with torch.no_grad():
            val_scores, _ = model(x[val_idx].float().to(device))
            val_scores = val_scores.cpu().numpy()
        val_y = y_all[val_idx].numpy().astype(np.int64)
        val_metas = [metas[i] for i in val_idx]
        vm = binary_metrics(val_y, val_scores, val_metas, args.threshold) if val_idx else {}
        bal = 0.5 * (vm.get("anomaly_recall", 0.0) + vm.get("normal_true_negative_rate", 0.0))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "ce": totals["ce"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_balanced_accuracy": bal,
            "val_anomaly_recall": vm.get("anomaly_recall", 0.0),
            "val_normal_fpr": vm.get("normal_false_positive_rate", 0.0),
            "val_auroc": vm.get("auroc"),
            "val_event_top3_recall": vm.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if bal > best["val_balanced_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores, logits = model(x.float().to(device))
        scores = scores.cpu().numpy()
        top_prompt = logits.argmax(dim=1).cpu().numpy()
    return write_outputs(
        name,
        model,
        history,
        best,
        scores,
        top_prompt,
        y_all.numpy().astype(np.int64),
        metas,
        out_dir,
        args,
        train_idx,
        val_idx,
        openset_idx,
        strategy="prompt_alignment_normal_anomaly",
        category_prefix="prompt",
        top_key="top_prompt",
    )


def train_token_prompt_alignment(name: str, token_sets: list[torch.Tensor], metas: list[dict], out_dir: Path, args) -> dict:
    y_all = torch.tensor([1 if m["label"] in TRAIN_CATEGORIES else 0 for m in metas], dtype=torch.long)
    supervised = torch.tensor([m["label"] in TRAIN_CATEGORIES or m["label"] == "normal" for m in metas], dtype=torch.bool)
    train_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "train"]
    val_idx = [i for i, m in enumerate(metas) if supervised[i] and m["split"] == "val"]
    openset_idx = [i for i, m in enumerate(metas) if m["label"] == "R06" or m["split"] == "openset"]
    device = torch.device(args.train_device)
    model = TokenPromptAlignmentScorer(
        token_sets[0].shape[1],
        args.embed_dim,
        args.tau,
        args.token_pooling,
        args.token_topk_ratio,
        args.token_lme_alpha,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = random.Random(args.seed)

    def predict(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        scores = []
        top_prompt = []
        model.eval()
        with torch.no_grad():
            for i in indices:
                s, logits = model.forward_one(token_sets[i].to(device))
                scores.append(s.detach().cpu())
                top_prompt.append(int(logits.mean(dim=0).argmax().detach().cpu()))
        return torch.stack(scores).numpy() if scores else np.zeros((0,), dtype=np.float32), np.asarray(top_prompt)

    history = []
    best = {"val_balanced_accuracy": -1.0, "epoch": -1}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        order = train_idx[:]
        rng.shuffle(order)
        model.train()
        totals = Counter()
        for i in order:
            yb = y_all[i].to(device)
            score, _ = model.forward_one(token_sets[i].to(device))
            bce = F.binary_cross_entropy(score.clamp(1e-6, 1 - 1e-6)[None], yb.float()[None])
            sep = prompt_separation_loss(model, args.sep_margin)
            loss = bce + args.lambda_sep * sep
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals["loss"] += float(loss.item())
            totals["bce"] += float(bce.item())
            totals["sep"] += float(sep.item())
            totals["n"] += 1
        val_scores, _ = predict(val_idx)
        val_y = y_all[val_idx].numpy().astype(np.int64)
        val_metas = [metas[i] for i in val_idx]
        vm = binary_metrics(val_y, val_scores, val_metas, args.threshold) if val_idx else {}
        bal = 0.5 * (vm.get("anomaly_recall", 0.0) + vm.get("normal_true_negative_rate", 0.0))
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, totals["n"]),
            "bce": totals["bce"] / max(1, totals["n"]),
            "sep": totals["sep"] / max(1, totals["n"]),
            "val_balanced_accuracy": bal,
            "val_anomaly_recall": vm.get("anomaly_recall", 0.0),
            "val_normal_fpr": vm.get("normal_false_positive_rate", 0.0),
            "val_auroc": vm.get("auroc"),
            "val_event_top3_recall": vm.get("event_top3_recall", 0.0),
        }
        history.append(row)
        if bal > best["val_balanced_accuracy"]:
            best = dict(row)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"{name} epoch {epoch:03d}: {row}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    all_idx = list(range(len(metas)))
    scores, top_prompt = predict(all_idx)
    return write_outputs(
        name,
        model,
        history,
        best,
        scores,
        top_prompt,
        y_all.numpy().astype(np.int64),
        metas,
        out_dir,
        args,
        train_idx,
        val_idx,
        openset_idx,
        strategy="token_prompt_alignment_normal_anomaly",
        category_prefix="prompt",
        top_key="top_prompt",
    )


def write_outputs(
    name,
    model,
    history,
    best,
    scores,
    top_vec,
    y_np,
    metas,
    out_dir,
    args,
    train_idx,
    val_idx,
    openset_idx,
    strategy="anomaly_vectors_only",
    category_prefix="anomaly_vector",
    top_key="top_anomaly_vector",
):
    ab_dir = out_dir / name
    ab_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "strategy": strategy}, ab_dir / "best_model.pt")
    with (ab_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (ab_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for meta, y, s, v in zip(metas, y_np, scores, top_vec):
            pred = "anomaly" if float(s) >= args.threshold else "normal"
            row = {
                **meta,
                "true_binary": int(y),
                "true_label": str(meta["label"]),
                "pred_label": pred,
                "object_anomaly_score": float(s),
                "category": f"{category_prefix}_{int(v)}",
                top_key: int(v),
                "threshold": args.threshold,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {
        "ablation": name,
        "strategy": f"{strategy}_low_score_is_normal",
        "anomaly_vectors": args.anomaly_vectors,
        "best_row": best,
        "train": binary_metrics(y_np[train_idx], scores[train_idx], [metas[i] for i in train_idx], args.threshold),
        "val": binary_metrics(y_np[val_idx], scores[val_idx], [metas[i] for i in val_idx], args.threshold),
        "openset": {
            "n": len(openset_idx),
            "mean_anomaly_score": float(scores[openset_idx].mean()) if openset_idx else None,
            "max_anomaly_score": float(scores[openset_idx].max()) if openset_idx else None,
            "min_anomaly_score": float(scores[openset_idx].min()) if openset_idx else None,
        },
        "num_train": len(train_idx),
        "num_val": len(val_idx),
        "num_openset": len(openset_idx),
    }
    (ab_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def write_report(out_dir: Path, selection: dict, sample_summary: dict, metrics: list[dict], args) -> None:
    lines = [
        "# Object-Level Anomaly Vector Training",
        "",
        "本实验冻结 Qwen3-VL 视觉 token 与 bbox-to-token 绑定规则，只训练对象级异常打分头。",
        "",
        "当前报告同时比较两条路线：",
        "",
        "1. anomaly vector bank：object embedding 与多个可学习 anomaly vectors 对齐；normal 不作为显式 prototype，低异常证据即判为 normal。",
        "2. prompt alignment：object embedding 与一个 normal prompt、一个 anomaly prompt 对齐，直接用 P(anomaly) 作为 object anomaly score。",
        "",
        f"- anomaly_vectors: `{args.anomaly_vectors}`",
        "- prompt_alignment_prompts: `normal/anomaly`",
        f"- threshold: `{args.threshold}`",
        f"- selected train/val/openset packages: `{len(selection['train'])}/{len(selection['val'])}/{len(selection['openset'])}`",
        "",
        "## Package Selection",
        "",
        "```json",
        json.dumps(selection, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Sample Summary",
        "",
        "```json",
        json.dumps(sample_summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--exp-name", default="")
    p.add_argument("--train-count", type=int, default=32)
    p.add_argument("--val-count", type=int, default=10)
    p.add_argument("--openset-count", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=1600)
    p.add_argument("--window-frames", type=int, default=8)
    p.add_argument("--positives-per-event-object", type=int, default=1)
    p.add_argument("--cover-event-windows", action="store_true")
    p.add_argument("--neg-per-pos", type=int, default=1)
    p.add_argument("--include-relation-samples", action="store_true")
    p.add_argument("--bbox-expand", type=float, default=0.08)
    p.add_argument("--min-tokens", type=int, default=1)
    p.add_argument("--resize-long-edge", type=int, default=1280)
    p.add_argument("--frame-token-cache-dir", type=Path, default=None)
    p.add_argument("--extract-batch-size", type=int, default=1)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--anomaly-vectors", type=int, default=8)
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--lambda-sep", type=float, default=0.1)
    p.add_argument("--sep-margin", type=float, default=0.2)
    p.add_argument("--run-token-evidence", action="store_true")
    p.add_argument("--token-pooling", choices=["mean", "topk", "logmeanexp"], default="topk")
    p.add_argument("--token-topk-ratio", type=float, default=0.2)
    p.add_argument("--token-lme-alpha", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--vision-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vision_dtype]
    exp_name = args.exp_name or f"exp_{time.strftime('%Y%m%d_%H%M%S')}_anomaly_vectors"
    data_out = args.data_root / "results" / exp_name
    code_out = args.code_root / "outputs" / "20260614" / "results" / exp_name
    data_out.mkdir(parents=True, exist_ok=True)
    if args.frame_token_cache_dir is None:
        args.frame_token_cache_dir = args.data_root.parent / "cache" / "20260613_data_token_cache"

    selection = select_short_packages(args.data_root, args.train_count, args.val_count, args.openset_count, args.max_frames, args.seed)
    train_packages = [x["package_id"] for x in selection["train"]]
    val_packages = [x["package_id"] for x in selection["val"]]
    openset_packages = [x["package_id"] for x in selection["openset"]]
    samples, sample_summary = build_samples(
        args.data_root,
        train_packages,
        val_packages,
        openset_packages,
        args.window_frames,
        args.positives_per_event_object,
        args.neg_per_pos,
        args.include_relation_samples,
        args.seed,
        args.cover_event_windows,
    )
    (data_out / "package_selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_out / "sample_index.jsonl").write_text(
        "\n".join(json.dumps(asdict(s), ensure_ascii=False) for s in samples) + "\n",
        encoding="utf-8",
    )
    (data_out / "sample_index_summary.json").write_text(json.dumps(sample_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_out / "config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"selected packages train/val/openset: {len(train_packages)}/{len(val_packages)}/{len(openset_packages)}", flush=True)
    print(f"built {len(samples)} sample specs", flush=True)
    print(json.dumps(sample_summary, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        print(f"prepare-only done: {data_out}", flush=True)
        return

    visual, motion, token_sets, metas = extract_sample_features(
        samples,
        args.model_dir,
        torch.device(args.device),
        dtype,
        data_out,
        args.extract_batch_size,
        args.bbox_expand,
        args.min_tokens,
        args.resize_long_edge,
        args.frame_token_cache_dir,
    )
    with (data_out / "feature_meta.jsonl").open("w", encoding="utf-8") as f:
        for row in metas:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = []
    metrics.append(train_feature_ablation("anomaly_vector_visual_only", visual.float(), metas, data_out, args))
    metrics.append(train_feature_ablation("anomaly_vector_visual_motion", torch.cat([visual.float(), motion.float()], dim=1), metas, data_out, args))
    metrics.append(train_prompt_alignment("prompt_alignment_visual_only", visual.float(), metas, data_out, args))
    metrics.append(train_prompt_alignment("prompt_alignment_visual_motion", torch.cat([visual.float(), motion.float()], dim=1), metas, data_out, args))
    if args.run_token_evidence:
        metrics.append(train_token_ablation(f"anomaly_vector_token_{args.token_pooling}", token_sets, metas, data_out, args))
        metrics.append(train_token_prompt_alignment(f"prompt_alignment_token_{args.token_pooling}", token_sets, metas, data_out, args))

    (data_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in metrics:
        draw_sample_visualizations(data_out / item["ablation"] / "predictions.jsonl", data_out / item["ablation"], max_images=32)
    draw_sample_visualizations(data_out / "anomaly_vector_visual_motion" / "predictions.jsonl", data_out, max_images=32)
    write_report(data_out, selection, sample_summary, metrics, args)
    if code_out.exists():
        shutil.rmtree(code_out)
    shutil.copytree(data_out, code_out)
    print(f"saved data results: {data_out}", flush=True)
    print(f"saved code backup: {code_out}", flush=True)


if __name__ == "__main__":
    main()
