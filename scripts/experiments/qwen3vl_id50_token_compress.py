#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path
from types import MethodType

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_UTILS = REPO_ROOT / "baseline" / "LAVIDA" / "qwen_vl_utils" / "src"
if QWEN_UTILS.exists() and str(QWEN_UTILS) not in sys.path:
    sys.path.insert(0, str(QWEN_UTILS))

from qwen_vl_utils import process_vision_info


DEFAULT_MODEL = "/home/expand_disk/model_repository/Models/Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VIDEO = (
    "/home/expand_disk/code_repository/mfl/token_compression/datasets/sha_ave_nwp/"
    "shanghaitech_test/tracking/scheme1_dataset_specific/visualizations/08_0044/"
    "08_0044_tracking.mp4"
)
DEFAULT_FRAMES = (
    "/home/expand_disk/code_repository/mfl/token_compression/datasets/sha_ave_nwp/"
    "shanghaitech_test/tracking/scheme1_dataset_specific/frames/08_0044.jsonl"
)


PROMPT = (
    "Focus on tracking ID 50 in the video. Classify the motion of tracking ID 50 as one of: "
    "running, jogging, fast walking, walking, or uncertain. Running/jogging means fast gait "
    "with rapid stride, strong arm swing, or airborne/near-airborne steps. Ignore other people "
    "unless they help compare speed. Return exactly three lines: label, confidence, evidence."
)


def load_frame_tracks(path):
    by_frame = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            by_frame[int(item["frame_idx"])] = item["tracks"]
    return by_frame


def expanded_bbox(box, factor, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = (x2 - x1) * factor
    bh = (y2 - y1) * factor
    return (
        max(0.0, cx - bw * 0.5),
        max(0.0, cy - bh * 0.5),
        min(float(width), cx + bw * 0.5),
        min(float(height), cy + bh * 0.5),
    )


def point_in_box(x, y, box):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def map_frame_for_t(t, grid_t, start_frame, end_frame):
    count = end_frame - start_frame + 1
    rel = (t + 0.5) * count / max(1, grid_t) - 0.5
    return int(round(start_frame + rel))


def build_groups(
    mode,
    grid_t,
    grid_h,
    grid_w,
    frame_tracks,
    frame_start,
    frame_end,
    src_w,
    src_h,
    resized_w,
    resized_h,
    target_id,
    target_expand,
    other_expand,
    focus_start,
    focus_end,
):
    n = grid_t * grid_h * grid_w
    if mode == "baseline":
        return None, {"visual_original": n, "visual_kept": n}

    groups = []
    stats = {
        "target_tokens": 0,
        "other_pruned": 0,
        "bg_groups": 0,
        "visual_original": n,
    }

    if mode == "global_merge":
        for t in range(grid_t):
            for y in range(0, grid_h, 2):
                for x in range(0, grid_w, 2):
                    members = []
                    for dy in range(2):
                        for dx in range(2):
                            yy, xx = y + dy, x + dx
                            if yy < grid_h and xx < grid_w:
                                members.append(t * grid_h * grid_w + yy * grid_w + xx)
                    groups.append((members, (t, y, x), "global"))
        stats["visual_kept"] = len(groups)
        return groups, stats

    if mode not in {"roi_aware", "motion_focus"}:
        raise ValueError(f"unknown mode: {mode}")

    sx = resized_w / src_w
    sy = resized_h / src_h
    cell_w = resized_w / grid_w
    cell_h = resized_h / grid_h

    for t in range(grid_t):
        frame_idx = map_frame_for_t(t, grid_t, frame_start, frame_end)
        if mode == "motion_focus" and not (focus_start <= frame_idx <= focus_end):
            start = t * grid_h * grid_w
            end = start + grid_h * grid_w
            groups.append((list(range(start, end)), (t, 0, 0), "temporal_compressed"))
            stats["bg_groups"] += 1
            continue

        tracks = frame_tracks.get(frame_idx, [])

        target_boxes = []
        other_boxes = []
        for tr in tracks:
            if tr.get("label") != "person":
                continue
            x1, y1, x2, y2 = tr["bbox"]
            box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
            if int(tr["track_id"]) == target_id:
                target_boxes.append(expanded_bbox(box, target_expand, resized_w, resized_h))
            else:
                other_boxes.append(expanded_bbox(box, other_expand, resized_w, resized_h))

        state = [[0 for _ in range(grid_w)] for _ in range(grid_h)]
        for y in range(grid_h):
            cy = (y + 0.5) * cell_h
            for x in range(grid_w):
                cx = (x + 0.5) * cell_w
                if any(point_in_box(cx, cy, b) for b in target_boxes):
                    state[y][x] = 1
                elif any(point_in_box(cx, cy, b) for b in other_boxes):
                    state[y][x] = 2

        for y in range(0, grid_h, 2):
            for x in range(0, grid_w, 2):
                bg = []
                for dy in range(2):
                    for dx in range(2):
                        yy, xx = y + dy, x + dx
                        if yy >= grid_h or xx >= grid_w:
                            continue
                        idx = t * grid_h * grid_w + yy * grid_w + xx
                        if state[yy][xx] == 1:
                            groups.append(([idx], (t, yy, xx), "target"))
                            stats["target_tokens"] += 1
                        elif state[yy][xx] == 2:
                            stats["other_pruned"] += 1
                        else:
                            bg.append(idx)
                if bg:
                    yy = (bg[0] // grid_w) % grid_h
                    xx = bg[0] % grid_w
                    groups.append((bg, (t, yy, xx), "background"))
                    stats["bg_groups"] += 1

    groups.sort(key=lambda item: item[0][0])
    stats["visual_kept"] = len(groups)
    return groups, stats


def video_runs(input_ids, video_token_id):
    pos = (input_ids == video_token_id).nonzero().flatten().tolist()
    runs = []
    if not pos:
        return runs
    start = prev = pos[0]
    for p in pos[1:]:
        if p == prev + 1:
            prev = p
        else:
            runs.append((start, prev))
            start = prev = p
    runs.append((start, prev))
    return runs


def compress_inputs(input_ids, attention_mask, runs, groups_by_t, coords_by_t, video_token_id):
    pieces = []
    pos_chunks = []
    old_cursor = 0
    pos_cursor = 0

    for t, (start, end) in enumerate(runs):
        if start > old_cursor:
            text = input_ids[old_cursor:start]
            pieces.append(text)
            l = text.numel()
            ids = torch.arange(l, device=input_ids.device, dtype=torch.long) + pos_cursor
            pos_chunks.append(ids.view(1, -1).expand(3, -1))
            pos_cursor += l

        keep_count = len(groups_by_t[t])
        pieces.append(torch.full((keep_count,), video_token_id, device=input_ids.device, dtype=input_ids.dtype))
        if keep_count:
            coords = torch.tensor(coords_by_t[t], device=input_ids.device, dtype=torch.long).T
            coords[0].zero_()
            pos_chunks.append(coords + pos_cursor)
            pos_cursor = int((coords + pos_cursor).max().item()) + 1
        old_cursor = end + 1

    if old_cursor < input_ids.numel():
        text = input_ids[old_cursor:]
        pieces.append(text)
        l = text.numel()
        ids = torch.arange(l, device=input_ids.device, dtype=torch.long) + pos_cursor
        pos_chunks.append(ids.view(1, -1).expand(3, -1))

    new_input_ids = torch.cat(pieces).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    position_ids = torch.cat(pos_chunks, dim=1).unsqueeze(1)
    rope_delta = position_ids.max().reshape(1, 1) + 1 - new_input_ids.shape[1]
    return new_input_ids, new_attention_mask, position_ids, rope_delta


def group_by_temporal(groups, grid_t, grid_h, grid_w):
    groups_by_t = [[] for _ in range(grid_t)]
    coords_by_t = [[] for _ in range(grid_t)]
    for members, coord, _kind in groups:
        t = members[0] // (grid_h * grid_w)
        groups_by_t[t].append(members)
        coords_by_t[t].append(coord)
    return groups_by_t, coords_by_t


def reduce_features(features, group_tensors):
    return torch.stack([features[g].mean(dim=0) for g in group_tensors], dim=0)


def patch_compression(model, groups, position_ids, rope_delta):
    group_tensors = None
    original_get_video_features = model.model.get_video_features
    original_get_rope_index = model.model.get_rope_index

    def compressed_get_video_features(self, pixel_values_videos, video_grid_thw=None):
        nonlocal group_tensors
        embeds, deepstack = original_get_video_features(pixel_values_videos, video_grid_thw)
        final = embeds[0]
        if group_tensors is None or group_tensors[0].device != final.device:
            group_tensors = [torch.tensor(g[0], dtype=torch.long, device=final.device) for g in groups]
        final_new = reduce_features(final, group_tensors)
        deep_new = [reduce_features(layer, group_tensors) for layer in deepstack]
        return (final_new,), deep_new

    def compressed_get_rope_index(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        if input_ids is not None and input_ids.shape[1] == position_ids.shape[-1]:
            return position_ids.to(input_ids.device), rope_delta.to(input_ids.device)
        return original_get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask)

    model.model.get_video_features = MethodType(compressed_get_video_features, model.model)
    model.model.get_rope_index = MethodType(compressed_get_rope_index, model.model)


def run(args):
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    content = [
        {
            "type": "video",
            "video": args.video,
            "nframes": args.nframes,
            "resized_height": args.height,
            "resized_width": args.width,
        },
        {"type": "text", "text": args.prompt},
    ]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    raw_grid = inputs["video_grid_thw"][0].tolist()
    merge = int(model.config.vision_config.spatial_merge_size)
    grid_t, grid_h, grid_w = int(raw_grid[0]), int(raw_grid[1] // merge), int(raw_grid[2] // merge)
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    runs = video_runs(inputs["input_ids"][0], video_token_id)
    original_visual = sum(end - start + 1 for start, end in runs)
    if len(runs) != grid_t:
        raise RuntimeError(f"expected {grid_t} video token runs, got {len(runs)}")

    stats = {
        "mode": args.mode,
        "grid_raw": raw_grid,
        "grid_llm": [grid_t, grid_h, grid_w],
        "input_original": int(inputs["input_ids"].shape[1]),
        "visual_original": int(original_visual),
        "height": args.height,
        "width": args.width,
        "nframes_requested": args.nframes,
    }

    generate_inputs = inputs
    if args.mode != "baseline":
        frame_tracks = load_frame_tracks(args.frames_jsonl)
        groups, group_stats = build_groups(
            args.mode,
            grid_t,
            grid_h,
            grid_w,
            frame_tracks,
            args.frame_start,
            args.frame_end,
            args.src_width,
            args.src_height,
            args.width,
            args.height,
            args.target_id,
            args.target_expand,
            args.other_expand,
            args.focus_start,
            args.focus_end,
        )
        groups_by_t, coords_by_t = group_by_temporal(groups, grid_t, grid_h, grid_w)
        new_ids, new_mask, pos_ids, rope_delta = compress_inputs(
            inputs["input_ids"][0],
            inputs["attention_mask"][0],
            runs,
            groups_by_t,
            coords_by_t,
            video_token_id,
        )
        patch_compression(model, groups, pos_ids, rope_delta)
        generate_inputs = dict(inputs)
        generate_inputs["input_ids"] = new_ids.to(model.device)
        generate_inputs["attention_mask"] = new_mask.to(model.device)
        stats.update(group_stats)
        stats["input_kept"] = int(new_ids.shape[1])
    else:
        stats["visual_kept"] = stats["visual_original"]
        stats["input_kept"] = stats["input_original"]

    print("EXPERIMENT_STATS", json.dumps(stats, ensure_ascii=False), flush=True)
    start = time.time()
    generated = model.generate(
        **generate_inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    elapsed = time.time() - start
    new_tokens = generated[:, generate_inputs["input_ids"].shape[1] :]
    answer = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"EXPERIMENT_SEC {elapsed:.3f}", flush=True)
    print("ANSWER_START")
    print(answer.strip())
    print("ANSWER_END")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--frames-jsonl", default=DEFAULT_FRAMES)
    parser.add_argument("--mode", choices=["baseline", "global_merge", "roi_aware", "motion_focus"], default="baseline")
    parser.add_argument("--height", type=int, default=728)
    parser.add_argument("--width", type=int, default=1288)
    parser.add_argument("--nframes", type=int, default=312)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=312)
    parser.add_argument("--src-width", type=int, default=856)
    parser.add_argument("--src-height", type=int, default=480)
    parser.add_argument("--target-id", type=int, default=50)
    parser.add_argument("--target-expand", type=float, default=3.0)
    parser.add_argument("--other-expand", type=float, default=1.0)
    parser.add_argument("--focus-start", type=int, default=136)
    parser.add_argument("--focus-end", type=int, default=166)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
