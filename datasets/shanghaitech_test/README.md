# ShanghaiTech Test Dataset

## Current Annotation Scope

Only annotate anomaly carriers that our current LocateAnything pipeline can localize.

Object targets:

- `person`
- `bicycle`
- `motorcycle`
- `car`
- `vehicle`
- `skateboard`
- `stroller`
- `cart`
- `bag`
- `backpack`
- `handbag`
- `suitcase`

Current shared tracking targets:

- `person`
- `bicycle`

The previous frame-level anomaly annotations are kept only as backup in `frames_GT.bak/` and are not used by the current redesigned tracking/anomaly setup.

Person-track state labels:

- `running`
- `chasing`
- `fighting`
- `brawling`
- `falling`
- `stealing`
- `snatching`
- `robbery`

Current anomaly design is track-centric and no longer treats the previous frame-level labels as active supervision.
