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

Track state labels:

- `running`
- `throwing`
- `loitering`
- `wrong direction`
- `too close`
- `bicycle`

Use `frames_GT/` as frame-level abnormal GT. Use `frames_GT/segments.json` for Avenue segment labels. Query aliases such as `running person` are not standalone annotation classes.
