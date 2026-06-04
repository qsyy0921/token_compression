# Tracking And Anomaly Schemes

Schemes define which objects should be tracked and which anomaly types should be modeled.

## Scheme 1

`scheme1_dataset_specific`

Each dataset keeps its own tracking targets and anomaly types.

## Scheme 2

`scheme2_unified_common`

All datasets use the same tracking targets and anomaly types.

## Future Extension

When adding a new dataset:

1. Add a dataset folder under `datasets/sha_ave_nwp/`.
2. Register the dataset in `collection.json`.
3. Add the dataset policy to each scheme that should support it.
4. If neither existing scheme fits, create a new scheme folder.

