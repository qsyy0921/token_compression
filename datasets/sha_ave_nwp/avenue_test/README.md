# Avenue Test

Avenue uses YOLO26x detections in the current project pipeline.

Local payload status:

- `frames/`: materialized in this folder.
- `object_detection/yolo26x/detections/`: YOLO26x JSONL files; detection is running.
- `frames_GT.bak/`: old frame-level and segment labels kept only as backup.

This dataset supports both current schemes:

- Scheme 1: dataset-specific tracking.
- Scheme 2: unified common tracking.

Scheme 1 tracking targets:

- `person`
- `bicycle`
- `frisbee`
- `sports ball`
- `bottle`
- `backpack`
- `handbag`
- `suitcase`
- `cell phone`

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
