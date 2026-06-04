# NWPU Annotation Design

## Goal

Use YOLO26x detections only for anomaly carriers that YOLO can reliably localize. Do not force scene objects or action semantics into YOLO labels.

## Official NWPU Anomaly Classes

- Climbing fence
- Car crossing square
- Cycling on footpath
- Kicking trash can
- Jaywalking
- Snatching bag
- Crossing lawn
- Wrong turn
- Cycling on square
- Chasing
- Loitering
- Scuffle
- Littering
- Forgetting backpack
- U-turn
- Battering
- Driving on wrong side
- Falling
- Suddenly stopping cycling in the middle of the road
- Group conflict
- Climbing tree
- Stealing
- Illegal parking
- Trucks
- Protest
- Playing with water
- Photographing in restricted area
- Dogs

## Annotation Targets

Only annotate the following detectable anomaly carriers:

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

`cell phone` is included only as evidence for photographing in restricted areas.

## Track State/Event Mapping

- Person track: jaywalking, climbing fence/tree, kicking trash can, crossing lawn, chasing, loitering, scuffle, battering, falling, group conflict, stealing, protest, photographing.
- Bicycle/person track: cycling on footpath, cycling on square, suddenly stopping cycling.
- Car/truck track: car crossing square, U-turn, driving on wrong side, illegal parking, trucks.
- Bag/person track: snatching bag, forgetting backpack, stealing.
- Dog track: dogs.

## Not Annotation Targets

Do not annotate these in the current YOLO-only version, because the object/region is not covered reliably by our current detector:

- `trash can`
- `fence`
- `tree`
- `lawn`
- `grass`
- `litter`
- `water`
- `water area`
- `wrong turn` as pure route semantics
- `playing with water`
- `littering` when the litter object is tiny or invisible
- `kicking trash can` if no trash can detector is available
- `crossing lawn` if no lawn/grass region is available
- `skateboard`
- `bottle`

## Alignment Notes

- Current NWPU labels should be object-track labels, not full anomaly-class labels.
- Use frame-level GT from `frames_GT/` to identify anomalous intervals, then associate detections/tracks inside those intervals.
- For action and location anomalies, the detected object is only the carrier; the final anomaly decision must use temporal/location context.
