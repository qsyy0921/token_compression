# ShanghaiTech Test

ShanghaiTech uses LocateAnything detections in this project.

Local payload status:

- `frames/`: materialized in this folder.
- `masks/`: materialized in this folder.
- `detections/`: materialized LocateAnything JSONL files.
- `frames_GT.bak/`: old frame-level labels kept only as backup.

This dataset supports both current schemes:

- Scheme 1: dataset-specific tracking.
- Scheme 2: unified common tracking.

Scheme 1 tracking targets:

- `person`
- `bicycle`
- `motorcycle`
- `car`
- `vehicle`
- `skateboard`
- `stroller`
- `cart`

Scheme 1 anomaly types:

- `person_running`
- `person_chasing`
- `person_fighting`
- `person_falling_or_lying`
- `person_jumping_or_climbing`
- `person_loitering_or_reversal`
- `person_interaction`
- `vehicle_or_rideable_intrusion`
- `cart_or_stroller_intrusion`

Old frame-level anomaly labels are not active supervision in the redesigned setup.
