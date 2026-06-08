#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_FRAMES_DIR = "datasets/sha_ave_nwp/shanghaitech_test/frames/08_0044"
DEFAULT_TRACKS = "datasets/sha_ave_nwp/shanghaitech_test/tracking/scheme1_dataset_specific/frames/08_0044.jsonl"
DEFAULT_OUT_DIR = "figures/id50_720p_token_compression"


COLORS = {
    "target": (48, 220, 115),
    "other": (240, 82, 82),
    "background": (70, 130, 230),
    "focus": (255, 205, 70),
    "text": (245, 245, 245),
    "black": (20, 20, 20),
    "white": (255, 255, 255),
}


RESULTS = [
    ("Baseline full 720p", "11960 -> 11960", "walking / high", (82, 82, 82)),
    ("Spatial ROI-aware", "11960 -> 3178", "walking / high", (66, 135, 245)),
    ("Motion-focus 136-166", "11960 -> 353", "running / self-report 0.98", (34, 178, 104)),
    ("Negative focus 220-260", "11960 -> 497", "walking / self-report 0.98", (205, 139, 45)),
]


def font(size=18, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_14 = font(14)
FONT_16 = font(16)
FONT_18 = font(18)
FONT_20 = font(20)
FONT_22 = font(22, bold=True)
FONT_28 = font(28, bold=True)


def load_tracks(path):
    frames = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            frames[int(item["frame_idx"])] = item["tracks"]
    return frames


def read_frame(frames_dir, idx):
    path = Path(frames_dir) / f"{idx:03d}.jpg"
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def track_for_id(tracks, frame_idx, target_id=50):
    for tr in tracks.get(frame_idx, []):
        if int(tr.get("track_id", -1)) == target_id:
            return tr
    return None


def expand_box(box, factor, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * factor
    bh = (y2 - y1) * factor
    return [
        max(0, cx - bw / 2),
        max(0, cy - bh / 2),
        min(width - 1, cx + bw / 2),
        min(height - 1, cy + bh / 2),
    ]


def draw_label(draw, xy, text, fill, text_fill=(255, 255, 255)):
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=FONT_16)
    pad = 4
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=fill)
    draw.text((x, y), text, fill=text_fill, font=FONT_16)


def draw_wrapped(draw, xy, text, max_width, fill, font_obj=FONT_16, line_gap=4):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        cand = word if not current else current + " " + word
        if draw.textbbox((0, 0), cand, font=font_obj)[2] <= max_width:
            current = cand
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    line_h = draw.textbbox((0, 0), "Ag", font=font_obj)[3] + line_gap
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, fill=fill, font=font_obj)


def draw_frame_panel(frame, tracks, frame_idx, title, focus_range=None, target_id=50, target_expand=3.0):
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    if focus_range and focus_range[0] <= frame_idx <= focus_range[1]:
        draw.rectangle((0, 0, width - 1, height - 1), outline=COLORS["focus"] + (255,), width=8)

    for tr in tracks.get(frame_idx, []):
        if tr.get("label") != "person":
            continue
        box = tr["bbox"]
        is_target = int(tr["track_id"]) == target_id
        color = COLORS["target"] if is_target else COLORS["other"]
        line_w = 4 if is_target else 1
        draw.rectangle(tuple(box), outline=color + (230,), width=line_w)
        if is_target:
            exp = expand_box(box, target_expand, width, height)
            draw.rectangle(tuple(exp), outline=COLORS["focus"] + (220,), width=3)
            draw_label(draw, (box[0], max(0, box[1] - 24)), f"ID {target_id}", color)

    draw.rectangle((0, 0, width, 38), fill=(0, 0, 0, 170))
    draw.text((10, 8), title, fill=COLORS["white"], font=FONT_18)
    return img


def token_mask(frame, tracks, frame_idx, grid_h=23, grid_w=40, focus_range=None, target_id=50, target_expand=3.0):
    h, w = frame.shape[:2]
    state = np.zeros((grid_h, grid_w), dtype=np.uint8)
    target = track_for_id(tracks, frame_idx, target_id)
    target_boxes = []
    if target:
        target_boxes.append(expand_box(target["bbox"], target_expand, w, h))

    other_boxes = []
    for tr in tracks.get(frame_idx, []):
        if tr.get("label") == "person" and int(tr.get("track_id", -1)) != target_id:
            other_boxes.append(tr["bbox"])

    cell_w = w / grid_w
    cell_h = h / grid_h
    for y in range(grid_h):
        cy = (y + 0.5) * cell_h
        for x in range(grid_w):
            cx = (x + 0.5) * cell_w
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in target_boxes):
                state[y, x] = 1
            elif any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in other_boxes):
                state[y, x] = 2

    img = Image.fromarray(frame).convert("RGB")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    in_focus = focus_range is None or focus_range[0] <= frame_idx <= focus_range[1]

    if not in_focus:
        draw.rectangle((0, 0, w, h), fill=(70, 130, 230, 150))
        draw.text((12, 12), "compressed temporal block", fill=COLORS["white"], font=FONT_22)
    else:
        for y in range(grid_h):
            for x in range(grid_w):
                x1 = int(x * cell_w)
                y1 = int(y * cell_h)
                x2 = int((x + 1) * cell_w)
                y2 = int((y + 1) * cell_h)
                if state[y, x] == 1:
                    color = COLORS["target"] + (95,)
                elif state[y, x] == 2:
                    color = COLORS["other"] + (80,)
                else:
                    color = COLORS["background"] + (34,)
                draw.rectangle((x1, y1, x2, y2), fill=color)
                if x % 2 == 0 and y % 2 == 0:
                    draw.rectangle((x1, y1, int((x + 2) * cell_w), int((y + 2) * cell_h)), outline=(255, 255, 255, 35), width=1)

    img = Image.alpha_composite(img.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle((0, h - 34, w, h), fill=(0, 0, 0, 170))
    draw.text((10, h - 27), "green: ID50 kept  red: other people pruned  blue: background merged", fill=COLORS["white"], font=FONT_16)
    return img.convert("RGB")


def make_sheet(frames_dir, tracks, frame_ids, out_path, title, subtitle, focus_range):
    panels = []
    for idx in frame_ids:
        frame = read_frame(frames_dir, idx)
        left = draw_frame_panel(frame, tracks, idx, f"Frame {idx}: original + ID50 ROI", focus_range)
        right = token_mask(frame, tracks, idx, focus_range=focus_range)
        row = Image.new("RGB", (left.width + right.width, left.height), COLORS["black"])
        row.paste(left, (0, 0))
        row.paste(right, (left.width, 0))
        panels.append(row.resize((856 * 2, 480), Image.Resampling.LANCZOS))

    header_h = 92
    sheet = Image.new("RGB", (panels[0].width, header_h + len(panels) * panels[0].height), (28, 30, 34))
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 16), title, fill=COLORS["white"], font=FONT_28)
    draw.text((24, 54), subtitle, fill=(210, 215, 220), font=FONT_18)
    y = header_h
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(out_path, quality=94)


def make_timeline(out_path):
    w, h = 1400, 420
    img = Image.new("RGB", (w, h), (250, 252, 255))
    draw = ImageDraw.Draw(img)
    draw.text((36, 26), "ID 50 Full-Video 720p Token Compression Controls", fill=(20, 24, 28), font=FONT_28)
    draw.text((36, 66), "Complete video, same Qwen3-VL-8B prompt. Only the visual token sequence changes.", fill=(78, 84, 92), font=FONT_18)

    x0, x1 = 120, 1320
    y = 140
    draw.line((x0, y, x1, y), fill=(120, 130, 140), width=4)
    for f in [0, 136, 166, 220, 260, 312]:
        x = x0 + (x1 - x0) * f / 312
        draw.line((x, y - 14, x, y + 14), fill=(90, 95, 102), width=2)
        draw.text((x - 18, y + 24), str(f), fill=(60, 65, 70), font=FONT_14)

    def span(a, b, color, label, yy):
        xa = x0 + (x1 - x0) * a / 312
        xb = x0 + (x1 - x0) * b / 312
        draw.rounded_rectangle((xa, yy, xb, yy + 34), radius=8, fill=color)
        draw.text((xa + 8, yy + 7), label, fill=COLORS["white"], font=FONT_16)

    span(136, 166, COLORS["target"], "running-positive focus window", 190)
    span(220, 260, (205, 139, 45), "walking negative-control window", 238)

    table_x, table_y = 64, 304
    col_w = [390, 220, 220, 280]
    headers = ["Setting", "Visual tokens", "Input tokens", "Qwen output"]
    row_h = 42
    draw.rounded_rectangle((table_x - 10, table_y - 10, table_x + sum(col_w) + 10, table_y + row_h * 5 + 10), radius=10, fill=(238, 242, 248))
    x = table_x
    for i, head in enumerate(headers):
        draw.text((x + 8, table_y), head, fill=(30, 36, 42), font=FONT_16)
        x += col_w[i]
    for r, (setting, visual, output, color) in enumerate(RESULTS, start=1):
        yrow = table_y + r * row_h
        x = table_x
        draw.rectangle((table_x, yrow - 6, table_x + sum(col_w), yrow + row_h - 8), fill=(255, 255, 255))
        draw.rectangle((table_x, yrow - 6, table_x + 6, yrow + row_h - 8), fill=color)
        vals = [setting, visual, "12156 -> " + ("12156" if "Baseline" in setting else "3374" if "Spatial" in setting else "549" if "136" in setting else "693"), output]
        for i, val in enumerate(vals):
            draw.text((x + 10, yrow + 4), val, fill=(28, 32, 36), font=FONT_16)
            x += col_w[i]
    img.save(out_path, quality=94)


def make_chinese_summary(out_path):
    w, h = 1600, 980
    img = Image.new("RGB", (w, h), (248, 250, 253))
    draw = ImageDraw.Draw(img)
    draw.text((56, 42), "ID50 Running 识别：运动聚焦 Token 压缩验证", fill=(24, 30, 38), font=FONT_28)
    draw.text((56, 88), "同一完整视频、同一 720p 输入、同一 prompt；只改变送入 LLM 的视觉 token 序列。", fill=(78, 86, 96), font=FONT_18)

    def box(x, y, ww, hh, title, color):
        draw.rounded_rectangle((x, y, x + ww, y + hh), radius=18, fill=(255, 255, 255), outline=(214, 222, 232), width=2)
        draw.rectangle((x, y, x + 12, y + hh), fill=color)
        draw.text((x + 30, y + 22), title, fill=(26, 32, 40), font=FONT_22)

    box(56, 150, 460, 230, "Baseline：完整视频直接推理", (100, 108, 118))
    draw.text((94, 220), "视觉 token：11960 -> 11960", fill=(58, 66, 76), font=FONT_18)
    draw.text((94, 260), "输出：walking / high", fill=(46, 52, 60), font=FONT_22)
    draw.text((94, 310), "问题：其他人、背景、长时间正常片段稀释 ID50 的关键步态证据。", fill=(78, 86, 96), font=FONT_16)

    box(560, 150, 460, 230, "仅空间 ROI 压缩", COLORS["background"])
    draw.text((598, 220), "视觉 token：11960 -> 3178", fill=(58, 66, 76), font=FONT_18)
    draw.text((598, 260), "输出：walking / high", fill=(46, 52, 60), font=FONT_22)
    draw.text((598, 310), "结论：只压其他人和背景还不够，时间维度的 walking 片段仍会稀释判断。", fill=(78, 86, 96), font=FONT_16)

    box(1064, 150, 460, 230, "运动聚焦 Token 压缩", COLORS["target"])
    draw.text((1102, 220), "视觉 token：11960 -> 353", fill=(58, 66, 76), font=FONT_18)
    draw.text((1102, 260), "输出：running，自报告0.98", fill=(22, 116, 66), font=FONT_22)
    draw.text((1102, 310), "保留 frames 136-166 的 ID50 关键运动 token，压缩非关键 token。", fill=(78, 86, 96), font=FONT_16)

    draw.line((520, 265, 555, 265), fill=(120, 130, 142), width=4)
    draw.polygon([(555, 265), (540, 255), (540, 275)], fill=(120, 130, 142))
    draw.line((1024, 265, 1059, 265), fill=(120, 130, 142), width=4)
    draw.polygon([(1059, 265), (1044, 255), (1044, 275)], fill=(120, 130, 142))

    draw.text((56, 450), "Token 压缩策略", fill=(26, 32, 40), font=FONT_28)
    strategy = [
        ("绿色", "保留 ID50 关键运动 token", COLORS["target"]),
        ("红色", "删除其他行人 token", COLORS["other"]),
        ("蓝色", "背景 token 2x2 平均池化", COLORS["background"]),
        ("灰色", "非关键时间块压缩为 summary token", (198, 207, 220)),
    ]
    y = 510
    for name, text, color in strategy:
        draw.rounded_rectangle((70, y, 106, y + 36), radius=8, fill=color)
        draw.text((124, y + 4), f"{name}：{text}", fill=(44, 52, 62), font=FONT_20 if "FONT_20" in globals() else FONT_18)
        y += 58

    draw.text((760, 450), "关键对照", fill=(26, 32, 40), font=FONT_28)
    rows = [
        ("完整视频 baseline", "walking / high", (100, 108, 118)),
        ("运动聚焦 136-166", "running，自报告0.98", COLORS["target"]),
        ("负对照 220-260", "walking，自报告0.98", (205, 139, 45)),
    ]
    y = 512
    for label, result, color in rows:
        draw.rounded_rectangle((770, y, 1470, y + 48), radius=12, fill=(255, 255, 255), outline=(214, 222, 232))
        draw.rectangle((770, y, 784, y + 48), fill=color)
        draw.text((802, y + 12), label, fill=(44, 52, 62), font=FONT_18)
        draw.text((1210, y + 12), result, fill=(30, 38, 48), font=FONT_18)
        y += 70

    draw.rounded_rectangle((56, 800, 1524, 910), radius=18, fill=(232, 247, 238), outline=COLORS["target"], width=2)
    draw.text((86, 828), "结论", fill=(22, 116, 66), font=FONT_22)
    draw.text((86, 868), "该案例证明：在多对象长视频中，目标感知 + 运动感知的 token 压缩可以放大 ID50 的 running 证据，使同一模型从 walking 翻转为 running。", fill=(30, 72, 50), font=FONT_18)
    img.save(out_path, quality=94)


def make_token_compression_mechanism(out_path):
    w, h = 1600, 920
    img = Image.new("RGB", (w, h), (248, 250, 253))
    draw = ImageDraw.Draw(img)
    draw.text((44, 34), "Token Compression Visualization for ID 50 Running Detection", fill=(22, 28, 34), font=FONT_28)
    draw.text(
        (44, 78),
        "The video pixels are unchanged. Only the visual token sequence is pruned or averaged before Qwen3-VL decoding.",
        fill=(76, 84, 94),
        font=FONT_18,
    )

    def panel(x, y, ww, hh, title, caption):
        draw.rounded_rectangle((x, y, x + ww, y + hh), radius=16, fill=(255, 255, 255), outline=(215, 222, 232), width=2)
        draw.text((x + 22, y + 18), title, fill=(22, 28, 34), font=FONT_22)
        draw_wrapped(draw, (x + 22, y + hh - 58), caption, ww - 44, fill=(76, 84, 94), font_obj=FONT_14)

    def token_grid(x, y, rows, cols, cell, states, alpha=False):
        for r in range(rows):
            for c in range(cols):
                state = states(r, c)
                if state == "target":
                    fill = COLORS["target"]
                elif state == "other":
                    fill = COLORS["other"]
                elif state == "merge":
                    fill = COLORS["background"]
                elif state == "drop":
                    fill = (218, 224, 232)
                elif state == "focus":
                    fill = COLORS["focus"]
                else:
                    fill = (235, 240, 246)
                draw.rounded_rectangle(
                    (x + c * cell, y + r * cell, x + c * cell + cell - 4, y + r * cell + cell - 4),
                    radius=4,
                    fill=fill,
                    outline=(255, 255, 255),
                )

    panel(44, 138, 456, 300, "1. Full visual token grid", "All people and background compete for attention.")
    top_cell = 22
    token_grid(
        92,
        198,
        8,
        14,
        top_cell,
        lambda r, c: "target" if (3 <= r <= 5 and 7 <= c <= 8) else "other" if (1 <= r <= 6 and c in {1, 2, 11, 12}) else "base",
    )
    draw_label(draw, (268, 278), "ID 50", COLORS["target"])
    draw_label(draw, (96, 188), "other people", COLORS["other"])

    panel(572, 138, 456, 300, "2. Object-aware pruning", "Other-person tokens are deleted, ID 50 is preserved.")
    token_grid(
        620,
        198,
        8,
        14,
        top_cell,
        lambda r, c: "target" if (3 <= r <= 5 and 7 <= c <= 8) else "drop" if (1 <= r <= 6 and c in {1, 2, 11, 12}) else "base",
    )
    for x in [646, 906]:
        draw.line((x, 222, x + 42, 264), fill=COLORS["other"], width=4)
        draw.line((x + 42, 222, x, 264), fill=COLORS["other"], width=4)
    draw_label(draw, (792, 278), "ID 50 kept", COLORS["target"])

    panel(1100, 138, 456, 300, "3. Background 2x2 merge", "Four nearby background tokens become one averaged token.")
    grid3_x, grid3_y = 1148, 198
    token_grid(
        grid3_x,
        grid3_y,
        8,
        14,
        top_cell,
        lambda r, c: "target" if (3 <= r <= 5 and 7 <= c <= 8) else "merge" if (r % 2 == 0 and c % 2 == 0) else "drop",
    )
    for r in range(0, 8, 2):
        for c in range(0, 14, 2):
            x = grid3_x + c * top_cell
            y = grid3_y + r * top_cell
            draw.rectangle((x - 2, y - 2, x + 42, y + 42), outline=(35, 90, 180), width=2)
    draw_label(draw, (1226, 278), "avg pool 2x2", COLORS["background"])

    panel(44, 506, 456, 314, "4. Temporal compression", "Non-key time blocks are compressed to compact summary tokens.")
    tx0, ty = 94, 606
    for i in range(13):
        x = tx0 + i * 29
        color = COLORS["target"] if i in {5, 6} else (205, 214, 226)
        draw.rounded_rectangle((x, ty, x + 22, ty + 90), radius=5, fill=color)
        if i not in {5, 6}:
            draw.text((x + 6, ty + 32), "1", fill=(80, 88, 98), font=FONT_14)
    draw.text((tx0, ty - 34), "13 Qwen temporal blocks", fill=(40, 46, 54), font=FONT_16)
    draw_label(draw, (tx0 + 5 * 29 - 8, ty + 104), "frames 136-166", COLORS["target"])

    panel(572, 506, 456, 314, "5. Final compressed sequence", "11960 visual tokens shrink to 353 visual tokens.")
    x, y = 630, 604
    for i, (label, count, color) in enumerate(
        [
            ("ID50 key motion", "162", COLORS["target"]),
            ("background groups", "191", COLORS["background"]),
            ("other people", "0", COLORS["other"]),
        ]
    ):
        yy = y + i * 58
        draw.rounded_rectangle((x, yy, x + 260, yy + 38), radius=8, fill=color)
        draw.text((x + 14, yy + 9), label, fill=COLORS["white"], font=FONT_16)
        draw.text((x + 296, yy + 9), count, fill=(25, 30, 36), font=FONT_18)
    draw.text((x, y + 182), "Input tokens: 12156 -> 549", fill=(34, 42, 50), font=FONT_18)

    panel(1100, 506, 456, 314, "6. Controlled result", "Same video + same prompt; only token sequence changes.")
    draw.rounded_rectangle((1160, 598, 1490, 660), radius=14, fill=(238, 241, 246), outline=(205, 212, 222))
    draw.text((1188, 616), "baseline: walking / high", fill=(42, 48, 56), font=FONT_22)
    draw.rounded_rectangle((1160, 690, 1490, 752), radius=14, fill=(223, 250, 233), outline=COLORS["target"])
    draw.text((1188, 708), "compressed: running / self-report 0.98", fill=(24, 104, 60), font=FONT_22)

    # Legend
    legend_y = 858
    items = [
        ("ID50 / key motion preserved", COLORS["target"]),
        ("other-person tokens pruned", COLORS["other"]),
        ("background tokens averaged", COLORS["background"]),
        ("non-key temporal block compressed", (205, 214, 226)),
    ]
    x = 54
    for label, color in items:
        draw.rounded_rectangle((x, legend_y, x + 24, legend_y + 24), radius=4, fill=color)
        draw.text((x + 34, legend_y + 2), label, fill=(54, 60, 68), font=FONT_16)
        x += 355

    img.save(out_path, quality=94)


def make_video(frames_dir, tracks, out_path, start, end, focus_range, fps=12):
    sample = read_frame(frames_dir, start)
    h, w = sample.shape[:2]
    out_w, out_h = w * 2, h
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    for idx in range(start, end + 1):
        frame = read_frame(frames_dir, idx)
        left = draw_frame_panel(frame, tracks, idx, f"Frame {idx}", focus_range)
        right = token_mask(frame, tracks, idx, focus_range=focus_range)
        canvas = Image.new("RGB", (out_w, out_h), COLORS["black"])
        canvas.paste(left, (0, 0))
        canvas.paste(right, (w, 0))
        writer.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--tracks", default=DEFAULT_TRACKS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(args.tracks)

    make_sheet(
        args.frames_dir,
        tracks,
        [136, 146, 156, 166],
        out_dir / "id50_running_focus_sheet.jpg",
        "Positive Case: Motion-Focused Token Compression Detects Running",
        "Frames 136-166 are preserved around ID 50; other objects, background, and non-key time blocks are compressed.",
        (136, 166),
    )
    make_sheet(
        args.frames_dir,
        tracks,
        [220, 232, 246, 260],
        out_dir / "id50_walking_negative_control_sheet.jpg",
        "Negative Control: Focusing a Slow Segment Does Not Produce Running",
        "The same compression mechanism focused on frames 220-260 yields walking, not a false running label.",
        (220, 260),
    )
    make_timeline(out_dir / "id50_720p_experiment_timeline.jpg")
    make_chinese_summary(out_dir / "id50_chinese_summary.jpg")
    make_token_compression_mechanism(out_dir / "id50_token_compression_mechanism.jpg")
    make_video(args.frames_dir, tracks, out_dir / "id50_running_focus_overlay.mp4", 136, 166, (136, 166))
    make_video(args.frames_dir, tracks, out_dir / "id50_walking_negative_control_overlay.mp4", 220, 260, (220, 260))
    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
