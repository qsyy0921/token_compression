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

Person-track state labels:

- `running`
- `chasing`
- `fighting`
- `brawling`
- `falling`
- `stealing`
- `snatching`
- `robbery`

Use `frames_GT/` as the authoritative frame-level abnormal label. These targets are detection/tracking annotations only, not official ShanghaiTech multi-class GT.
