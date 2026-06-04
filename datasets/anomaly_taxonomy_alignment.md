# Anomaly Taxonomy And Detection Alignment

This note summarizes anomaly categories found from dataset pages and papers, then aligns them with the current object detection setup.

## Sources

- ShanghaiTech official page: https://svip-lab.github.io/dataset/campus_dataset.html
- Avenue official page: https://www.cse.cuhk.edu.hk/~leojia/projects/detectabnormal/dataset.html
- NWPU Campus official page: https://campusvad.github.io/
- Online VAD survey table for common dataset anomaly types: https://pmc.ncbi.nlm.nih.gov/articles/PMC10490792/
- ShanghaiTech object-centric discussion: https://pmc.ncbi.nlm.nih.gov/articles/PMC10385872/

## Current Detection Setup

- ShanghaiTech: LocateAnything open-vocabulary detections.
- Avenue: LocateAnything open-vocabulary detections.
- NWPU: YOLO26x with selected COCO classes:
  `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `dog`, `backpack`, `handbag`, `suitcase`, `skateboard`, `bottle`, `cell phone`.

## ShanghaiTech

Official resources describe ShanghaiTech as a 13-scene campus dataset with 130 abnormal events and pixel-level ground truth. The official page explicitly highlights sudden-motion anomalies such as chasing and brawling. Other papers and summaries commonly describe ShanghaiTech anomalies as pedestrian-zone anomalies including running, fighting/brawling, chasing, stealing/snatching/robbery, falling, bicycles, motorcycles, cars/vehicles, skateboards, strollers/prams/carts, and other unusual objects.

Recommended annotation objects:

- Strongly keep: `person`, `running person`, `chasing person`, `fighting person`, `falling person`, `person lying on ground`, `cyclist`, `bicycle`, `motorcycle`, `car`, `vehicle`, `skateboard`, `stroller`, `cart`, `bag`, `backpack`, `handbag`, `suitcase`.
- Detection alignment: LocateAnything can describe most of these, including action-state prompts, but action boxes should be treated as person boxes with event labels rather than true object categories.
- Weak/ambiguous: stealing/snatching/robbery depends on interaction and temporal context; object detection alone can only provide `person` and `bag` evidence.

## Avenue

The official Avenue page defines three coarse anomaly groups: strange action, wrong direction, and abnormal object. It also provides rectangle-based spatial ground truth. Commonly cited concrete events include running, throwing objects, loitering, walking in the wrong direction, and abnormal objects such as bicycles. The local `avenue_annotations.json` currently contains segment labels: `running`, `throwing`, `loitering`, `too_close`, and `bicycle`.

Recommended annotation objects:

- Strongly keep: `person`, `running person`, `loitering person`, `person walking in wrong direction`, `person too close to camera`, `bicycle`, `bike`, `bag`, `backpack`, `paper`, `box`, `package`, `thrown object`, `abnormal object`.
- Detection alignment: LocateAnything can cover these with prompts, but `wrong direction`, `loitering`, `too_close`, and `throwing` are temporal/action labels. The box should usually be on the person and/or thrown object.
- Weak/ambiguous: small thrown paper is easy to miss; if object box is unstable, keep the actor/person box as primary.

## NWPU Campus

The official NWPU page lists 28 anomaly classes:

- Climbing fence
- Car crossing square
- Cycling on footpath (scene-dependent)
- Kicking trash can
- Jaywalking
- Snatching bag
- Crossing lawn
- Wrong turn (scene-dependent)
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
- Trucks (scene-dependent)
- Protest
- Playing with water
- Photographing in restricted area (scene-dependent)
- Dogs

Current YOLO26x coverage:

- Good direct object coverage: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `dog`, `backpack`, `handbag`, `suitcase`, `bottle`, `cell phone`.
- Partially covered by object boxes plus tracking/motion: climbing fence, car crossing square, cycling on footpath, jaywalking, snatching bag, cycling on square, chasing, loitering, forgetting backpack, U-turn, driving on wrong side, falling, stopping cycling, group conflict, stealing, illegal parking, trucks, protest, photographing, dogs.
- Not covered or weak with YOLO COCO classes: trash can, fence, lawn/grass, tree, water area, litter/trash, wrong turn semantics, playing with water, scuffle/battering as action semantics.

Recommended NWPU annotation policy with YOLO26x:

- Keep high-value object categories: `person`, `group/crowd`, `bicycle/cyclist`, `car`, `motorcycle`, `bus`, `truck`, `dog`, `bag/backpack/handbag/suitcase`, `bottle`, `cell phone`.
- For action/location anomalies, use the detected object box as the carrier:
  person-box for jaywalking, climbing, chasing, loitering, falling, stealing, protest, photographing;
  bicycle-box/person-box for cycling anomalies and stopping cycling;
  car/truck-box for illegal parking, car crossing square, wrong-side driving, U-turn, trucks.
- Ignore or handle with a future open-vocabulary pass: `trash can`, `fence`, `tree`, `lawn/grass`, `litter`, `water`, because YOLO26x cannot reliably detect these with the current COCO class subset.

## Practical Recommendation

- For ShanghaiTech and Avenue, continue using LocateAnything with action/object prompts. Keep both object labels and event labels when possible.
- For NWPU, do not try to force all 28 anomaly classes into YOLO labels. Store YOLO object tracks first, then map frame-level anomaly intervals to object tracks where the object exists.
- If we need full NWPU semantic coverage later, add a second open-vocabulary pass only for missed scene objects: `trash can`, `fence`, `tree`, `lawn/grass`, `litter`, `water area`.
