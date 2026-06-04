# Scheme 1: Dataset-Specific

This scheme is more expressive. Each dataset keeps YOLO26x-detectable tracking targets and anomaly types matched to its own scene semantics.

Use this scheme when the goal is best per-dataset performance.

## ShanghaiTech Test

Tracking targets:

- `person`
- `bicycle`
- `motorcycle`
- `car`
- `skateboard`

Anomaly types:

- `person_running`
- `person_chasing`
- `person_fighting`
- `person_falling_or_lying`
- `person_jumping_or_climbing`
- `person_loitering_or_reversal`
- `person_interaction`
- `vehicle_or_rideable_intrusion`

## Avenue Test

Tracking targets:

- `person`
- `bicycle`
- `frisbee`
- `sports ball`
- `bottle`
- `backpack`
- `handbag`
- `suitcase`
- `cell phone`

Anomaly types:

- `person_running`
- `person_jumping`
- `person_falling_or_lying`
- `person_loitering_or_wrong_direction`
- `person_object_throwing`
- `person_abnormal_object_interaction`
- `bicycle_intrusion_or_abnormal_ride`
- `object_throwing`
- `abnormal_object`

## NWPU Test

Tracking targets:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`
- `dog`

Anomaly types:

- `person_running`
- `person_chasing`
- `person_fighting`
- `person_falling_or_lying`
- `person_jumping_or_climbing`
- `person_loitering_or_reversal`
- `person_conflict`
- `bag_interaction`
- `cycling_violation`
- `vehicle_violation`
- `animal_intrusion`
