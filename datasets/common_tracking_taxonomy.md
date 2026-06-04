# Common Tracking Taxonomy

This file defines the redesigned cross-dataset tracking setup for ShanghaiTech, Avenue, and NWPU.

## Design Goal

Track only object categories that are detectable across all three datasets and are stable enough for useful trajectories.

## Default Tracking Objects

Use only:

- `person`
- `bicycle`

These are the safest shared tracking objects:

- They are detectable by LocateAnything on ShanghaiTech and Avenue.
- They are detectable by YOLO26x on NWPU.
- They are directly related to many abnormal events.
- Their boxes are large and stable enough for tracking.

## Evidence-Only Objects

Do not track these by default:

- `bag`
- `backpack`
- `handbag`
- `suitcase`
- `thrown object`
- `paper`
- `box`
- `package`
- `cell phone`

They can be used as evidence attributes attached to a nearby `person` track, but they should not create independent tracks unless a later experiment specifically studies abandoned or thrown objects.

## Common Anomaly Types

Because the old frame-level abnormal labels are no longer treated as fixed supervision, we can redesign anomaly types around common track carriers.

### person_behavior_anomaly

Person-centered abnormal behavior.

Examples:

- running
- chasing
- fighting
- falling
- loitering
- wrong direction
- too close
- jaywalking
- stealing or snatching
- protest or group conflict

Carrier:

- `person`

Evidence-only:

- `bag`, `backpack`, `handbag`, `suitcase`, `cell phone`

### bicycle_anomaly

Bicycle or rider-centered abnormal behavior.

Examples:

- bicycle in a pedestrian area
- cycling on footpath
- cycling on square
- suddenly stopping cycling
- bicycle as abnormal object

Carrier:

- `bicycle`
- optionally nearby `person`

### person_object_interaction

Object interaction where the stable carrier is usually a person.

Examples:

- throwing
- abandoned bag
- forgetting backpack
- snatching bag
- littering when visible

Carrier:

- `person`

Evidence-only:

- `bag`, `backpack`, `handbag`, `suitcase`, `thrown object`, `paper`, `box`, `package`

## Dataset-Specific Notes

### ShanghaiTech

Track:

- `person`
- `bicycle`

Anomaly types:

- `person_behavior_anomaly`
- `bicycle_anomaly`
- `person_object_interaction`

### Avenue

Track:

- `person`
- `bicycle`

Anomaly types:

- `person_behavior_anomaly`
- `bicycle_anomaly`
- `person_object_interaction`

### NWPU

Track:

- `person`
- `bicycle`

Anomaly types:

- `person_behavior_anomaly`
- `bicycle_anomaly`
- `person_object_interaction`

Vehicle, dog, and scene-region anomalies are intentionally excluded from the common default tracking setup because they do not appear as reliable shared tracking targets across all three datasets.

