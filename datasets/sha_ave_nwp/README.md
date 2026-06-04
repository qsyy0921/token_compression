# SHA-AVE-NWP Dataset Collection

This collection contains the current three video anomaly datasets used by the project:

- `shanghaitech_test`
- `avenue_test`
- `nwpu_test`

The directory is designed to be extensible. Future datasets should be added as sibling dataset folders under this collection and registered in one or more scheme files.

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

Large generated payloads should stay outside Git or be ignored:

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

