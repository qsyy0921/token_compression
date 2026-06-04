# SHA-AVE-NWP Dataset Collection

This collection contains the current three video anomaly datasets used by the project:

- `shanghaitech_test`
- `avenue_test`
- `nwpu_test`

The directory is designed to be extensible. Future datasets should be added as sibling dataset folders under this collection and registered in one or more scheme files.

This is the active local data workspace. Concrete payloads should live inside this collection rather than being required through external dataset paths. Git tracks the metadata, scheme files, scripts, and documentation; large payload directories are intentionally ignored.

## Directory Layout

```text
datasets/sha_ave_nwp/
  README.md
  collection.json
  shanghaitech_test/
    README.md
    dataset.json
  avenue_test/
    README.md
    dataset.json
  nwpu_test/
    README.md
    dataset.json
  schemes/
    README.md
    scheme1_dataset_specific/
      README.md
      scheme.json
    scheme2_unified_common/
      README.md
      scheme.json
```

## Dataset Folder Contract

Each dataset folder should contain:

- `dataset.json`: dataset metadata, detector, data roots, detectable objects, evidence-only objects, and supported schemes.
- `README.md`: human-readable notes.
- `frames/`: local test frames.
- `detections/` or detector-specific detection folders: local object detection JSONL files.
- `tracking/`: per-scheme tracking outputs.
- `anomaly_types/`: dataset-local scheme definitions.
- `metadata/`: categories and annotation notes.

Large generated payloads should stay local and ignored by Git:

- frames
- raw videos
- detection JSONL files
- tracking outputs
- visualization videos
- old frame-level GT backups

## Scheme Contract

Each scheme folder should contain:

- `scheme.json`: machine-readable tracking targets and anomaly types.
- `README.md`: human-readable explanation.

Schemes are independent from datasets. A new dataset can join an existing scheme if it supports the required tracking targets, or it can define a new scheme.

## Current Design Choice

We keep two schemes:

- Scheme 1: dataset-specific targets and anomaly types.
- Scheme 2: unified targets and anomaly types shared by all datasets.

The old frame-level anomaly annotations are not active supervision for these schemes. If needed, they should remain as backups only.

## Current Local Status

- `shanghaitech_test`: frames, masks, LocateAnything detections, metadata, old frame GT backup, and scheme definitions are materialized in this collection.
- `nwpu_test`: extracted frames, raw videos, tracking reference files, YOLO26x detections, metadata, old frame GT backup, and scheme definitions are materialized in this collection.
- `avenue_test`: frames, partial LocateAnything detections, metadata, old frame GT backup, and scheme definitions are in this collection. A repair run is filling the remaining detection frames.

## Scheme Summary

Scheme 1 keeps dataset-specific targets and anomaly labels for best per-dataset performance.

Scheme 2 uses the shared tracking targets `person` and `bicycle`, and the shared anomaly labels:

- `person_fast_motion`
- `person_chasing_or_pursuit`
- `person_fight_or_collision`
- `person_fall_or_lying`
- `person_jump_or_climb`
- `person_loitering_or_wrong_direction`
- `bicycle_or_rideable_anomaly`
- `person_object_interaction`
