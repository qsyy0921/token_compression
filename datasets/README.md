# Dataset Object Labels

This directory organizes LocateAnything pseudo labels for the object-centric token compression experiments.

## Layout

- `shanghaitech_test/`: existing ShanghaiTech test labels from the 2026-06-02 labeling package.
- `avenue_test/`: CUHK Avenue testing frames and LocateAnything labels.
- `nwpu_test/`: NWPU Campus testing videos, extracted frames, tracking references, and LocateAnything labels.

Each dataset directory contains:

- `categories.txt`: object prompts chosen from the dataset anomaly definitions.
- `detections/`: one JSONL per test video, one record per frame.
- `work/`: multi-worker manifests and logs.
- `metadata.json`: source paths and anomaly/object rationale.
