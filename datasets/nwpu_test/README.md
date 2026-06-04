# NWPU Test Dataset

## Current Annotation Scope

Only annotate anomaly carriers that our current YOLO26x pipeline can localize.

Object targets:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`
- `dog`
- `backpack`
- `handbag`
- `suitcase`
- `cell phone`

Track state/event labels:

- `jaywalking`
- `cycling on footpath`
- `cycling on square`
- `suddenly stopping cycling`
- `car crossing square`
- `u-turn`
- `driving on wrong side`
- `illegal parking`
- `trucks`
- `chasing`
- `loitering`
- `scuffle`
- `battering`
- `falling`
- `group conflict`
- `stealing`
- `snatching bag`
- `forgetting backpack`
- `protest`
- `photographing in restricted area`
- `dogs`

Do not annotate current YOLO-missing objects or regions such as `trash can`, `fence`, `tree`, `lawn`, `grass`, `litter`, `water`, `skateboard`, or `bottle`. Use `frames_GT/` as the authoritative frame-level abnormal label.
