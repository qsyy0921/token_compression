# Avenue Annotation Design

## Goal

Use LocateAnything detections only for Avenue anomaly carriers that can be localized. Avenue abnormal timing comes from `frames_GT/`, and segment details come from `frames_GT/segments.json`.

## Official Anomaly Definition

The official Avenue page defines three broad anomaly groups:

- Strange action
- Wrong direction
- Abnormal object

The official page also provides rectangle-based spatial ground truth. Common examples in papers include running, throwing objects, loitering, walking in the wrong direction, and abnormal objects such as bicycles.

Our local `avenue_annotations.json` further splits segments into `running`, `throwing`, `loitering`, `too_close`, and `bicycle`. `too_close` is a local segment label, not one of the three official coarse names.

## Annotation Targets

Only annotate the following detectable anomaly carriers:

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

Attach these detectable event/state labels to `person` or object tracks:

- `running`
- `throwing`
- `loitering`
- `wrong direction`
- `too close`
- `bicycle`

## Not Annotation Targets

Do not create separate object categories for these prompts. They are only query aliases and should be normalized to `person`:

- `running person`
- `loitering person`
- `person walking in wrong direction`
- `person too close to camera`

## Alignment Notes

- For `running`, `loitering`, `wrong direction`, and `too_close`, the box should be on the person.
- For `throwing`, keep the person box as primary; keep a thrown-object box only when the object is visible and stable.
- Small paper objects may be missed. Do not require a thrown-object box if the actor/person is clear.
