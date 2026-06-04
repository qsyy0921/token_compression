# ShanghaiTech Annotation Design

## Goal

Use LocateAnything detections only for anomaly carriers that this detector can localize. Official frame-level/pixel-level GT remains the source of abnormal time supervision.

## Official Anomaly Definition

ShanghaiTech contains 13 campus scenes and 130 abnormal events. The official page highlights sudden-motion anomalies such as chasing and brawling, and provides pixel-level abnormal-region ground truth. It does not publish a complete official per-event class list in the dataset page.

The concrete event names below are annotation targets derived from the official examples plus commonly cited ShanghaiTech descriptions in related papers. They must not be treated as official multi-class ground truth.

## Annotation Targets

Only annotate the following detectable anomaly carriers:

- `person`
- `bicycle`
- `motorcycle`
- `car`
- `vehicle`
- `skateboard`
- `stroller`
- `cart`
- `bag`
- `backpack`
- `handbag`
- `suitcase`

For person-centric anomalies, attach one of these detectable state labels to the `person` track when the visual evidence is clear:

- `running`
- `chasing`
- `fighting`
- `brawling`
- `falling`
- `stealing`
- `snatching`
- `robbery`

## Not Annotation Targets

Do not create separate object categories for these prompts. They are only query aliases and should be normalized to `person`:

- `running person`
- `chasing person`
- `fighting person`
- `falling person`
- `person lying on ground`

## Alignment Notes

- For action anomalies, the primary bounding box should be the actor/person.
- For stealing/snatching/robbery, object detection alone is insufficient; keep `person` and `bag` tracks as evidence and let temporal modeling handle the interaction.
