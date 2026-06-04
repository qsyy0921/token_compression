# Avenue Test Dataset

## Current Annotation Scope

Only annotate anomaly carriers that our current LocateAnything pipeline can localize.

Object targets:

- `person`
- `bicycle`
- `bag`
- `backpack`
- `handbag`
- `suitcase`
- `paper`
- `box`
- `package`
- `thrown object`
- `abnormal object`

Current shared tracking targets:

- `person`
- `bicycle`

The previous frame-level anomaly annotations and segment labels are kept only as backup in `frames_GT.bak/` and are not used by the current redesigned tracking/anomaly setup.

Track state labels:

- `running`
- `throwing`
- `loitering`
- `wrong direction`
- `too close`
- `bicycle`

Current anomaly design is track-centric. Query aliases such as `running person` are not standalone annotation classes.
