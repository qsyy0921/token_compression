# NWPU Test

NWPU uses YOLO26x detections in this project.

Local payload status:

- `frames/`: materialized extracted frames.
- `videos/`: materialized raw test videos.
- `tracking_reference/`: materialized reference tracking files.
- `yolo26x_detections/`: complete YOLO26x JSONL files, one per test video.
- `frames_GT.bak/`: old frame-level labels kept only as backup.

This dataset supports both current schemes:

- Scheme 1: dataset-specific tracking.
- Scheme 2: unified common tracking.

Scheme 1 tracking targets:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`
- `dog`

Scheme 1 anomaly types:

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

Old frame-level anomaly labels are not active supervision in the redesigned setup.
