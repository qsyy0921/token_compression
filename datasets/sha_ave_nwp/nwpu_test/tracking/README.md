# Tracking Results

Tracking outputs for `nwpu_test` are organized by scheme.

Only objects listed by the selected scheme should be tracked.

Scheme 1 targets: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `dog`.

Scheme 2 targets: `person`, `bicycle`.

Expected layout:

```text
tracking/
  scheme1_dataset_specific/
    frames/
    tracks/
    visualizations/
    run_summary.json
  scheme2_unified_common/
    frames/
    tracks/
    visualizations/
    run_summary.json
```

Generated tracking payloads are local outputs and are ignored by Git.
