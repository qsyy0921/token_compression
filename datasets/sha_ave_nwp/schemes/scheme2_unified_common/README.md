# Scheme 2: Unified Common

This scheme is more comparable. All datasets use the same tracking targets and anomaly types.

Use this scheme when the goal is cross-dataset consistency.

Tracking targets:

- `person`
- `bicycle`

Unified anomaly types:

- `person_fast_motion`
- `person_chasing_or_pursuit`
- `person_fight_or_collision`
- `person_fall_or_lying`
- `person_jump_or_climb`
- `person_loitering_or_wrong_direction`
- `bicycle_or_rideable_anomaly`
- `person_object_interaction`

The labels are intentionally behavior-level rather than dataset-name-level. They cover Avenue-style running, throwing, loitering, jumping, and falling; ShanghaiTech-style pedestrian abnormal behavior and rideable intrusion; and NWPU-style campus conflict, cycling, and abnormal pedestrian motion.
