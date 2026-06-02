# ShanghaiTech LocateAnything 服务器标注包说明

这个包用于在服务器上对 ShanghaiTech test 集 107 个视频做离线对象标注，并在标注完成后做 tracking。LocateAnything 只作为离线 pseudo-label 工具使用，后续 token 压缩实验不需要在线调用它。

## 包内容

- `models/LocateAnything-3B/`：LocateAnything 模型权重与 remote code。
- `data/shanghai/data/testing/videos/`：ShanghaiTech test 集 107 个 mp4 视频。
- `data/shanghai/data/testframemask/`：每个 test 视频对应的帧级异常 mask 标注。
- `scripts/extract_shanghai_test_frames.py`：从 mp4 自动抽帧。
- `scripts/locateanything_label_shanghai_test.py`：逐帧运行 LocateAnything 并输出 JSONL 检测结果。
- `scripts/run_locateanything_multigpu.py`：多 GPU 分片运行入口。
- `scripts/track_shanghai_objects_iou.py`：label-aware IoU tracking。
- `scripts/run_tracking_all.py`：对所有检测结果批量 tracking。

## 环境建议

本地验证环境是：

```bash
python -m pip install torch transformers==4.57.1 pillow opencv-python-headless lmdb decord peft
```

如果服务器上 transformers 版本太新导致 remote code 加载失败，优先使用 `transformers==4.57.1`。
如果服务器没有 `magi_attention`，脚本默认使用 `sdpa` attention 后端。

## 两张 GPU 分片运行

推荐用 GPU0 处理前 55 个视频，GPU2 处理剩下 52 个视频：

```bash
python scripts/run_locateanything_multigpu.py \
  --gpus 0,2 \
  --split-counts 55,52 \
  --extract-frames \
  --frames-root data/shanghai/data/testing/frames \
  --videos-root data/shanghai/data/testing/videos \
  --model-path models/LocateAnything-3B \
  --output-root outputs/locateanything_shanghai_test
```

这个命令会先从 `testing/videos` 抽帧到 `testing/frames`，然后启动两个 worker。每个 worker 只加载一次模型，并按自己的 video list 顺序处理。

GPU 分片文件和日志会写到：

```text
outputs/locateanything_multigpu_work/gpu0_videos.txt
outputs/locateanything_multigpu_work/gpu2_videos.txt
outputs/locateanything_multigpu_work/gpu0.log
outputs/locateanything_multigpu_work/gpu2.log
```

## 输出格式

每个视频输出一个 JSONL：

```text
outputs/locateanything_shanghai_test/<video_id>.jsonl
```

每一行是一帧：

```json
{
  "video_id": "01_0014",
  "frame_idx": 0,
  "frame_path": "data/shanghai/data/testing/frames/01_0014/000.jpg",
  "width": 856,
  "height": 480,
  "prompt": "...",
  "answer": "...",
  "boxes": [
    {"label": "pedestrian", "x1": 688.0, "y1": 142.0, "x2": 718.0, "y2": 246.0}
  ],
  "points": []
}
```

脚本支持断点续跑：如果某个 JSONL 已经有部分帧，默认会跳过已有 `frame_idx`，继续处理剩余帧。

日志每 10 帧会输出一次速度统计：

```text
last10_time=最近10帧总耗时
avg_frame=最近10帧平均每帧耗时
last10_output_tokens=最近10帧生成的输出 token 数
last10_tok_s=最近10帧输出 token/s
total_tok_s=当前视频累计输出 token/s
```

每帧 JSONL 也会记录：

```json
{
  "generation_seconds": 3.21,
  "output_tokens": 128
}
```

## Tracking

标注完成后运行：

```bash
python scripts/run_tracking_all.py \
  --detections-root outputs/locateanything_shanghai_test \
  --output-root outputs/shanghai_iou_tracks \
  --min-iou 0.25 \
  --max-age 5
```

输出：

```text
outputs/shanghai_iou_tracks/<video_id>.tracks.jsonl
outputs/shanghai_iou_tracks/<video_id>.track_summary.json
```

当前 tracking 是简单的 label-aware IoU tracker。它的优点是可复现、易检查；如果服务器算力允许，后续可以替换成 BoT-SORT、ByteTrack 或 StrongSORT，以追求更好的轨迹质量。

## 干跑检查

不真正启动模型，只检查分片：

```bash
python scripts/run_locateanything_multigpu.py --gpus 0,2 --split-counts 55,52 --dry-run
```
