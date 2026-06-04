# Avenue Test

Avenue uses LocateAnything detections in this project.

Local payload status:

- `frames/`: materialized in this folder.
- `detections/`: LocateAnything JSONL files; a repair run is filling missing frames for videos `15`, `20`, and `21`.
- `frames_GT.bak/`: old frame-level and segment labels kept only as backup.

This dataset supports both current schemes:

- Scheme 1: dataset-specific tracking.
- Scheme 2: unified common tracking.

Scheme 1 tracking targets:

- `person`
- `bicycle`
- `thrown object`

Scheme 1 anomaly types:

- `person_running`
- `person_jumping`
- `person_falling_or_lying`
- `person_loitering_or_wrong_direction`
- `person_object_throwing`
- `person_abnormal_object_interaction`
- `bicycle_intrusion_or_abnormal_ride`
- `object_throwing`
- `abnormal_object`

Old frame-level and segment anomaly labels are not active supervision in the redesigned setup.
