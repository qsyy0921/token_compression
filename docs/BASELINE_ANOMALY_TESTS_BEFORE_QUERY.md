# Object Query 之前的异常测试方案

## 当前决策

先暂停 object query 的实现，不直接从“对象框内视觉 token / ROI token”开始建模。当前阶段先做更可控的对象级 anomaly signal 测试：

```text
离线对象检测 -> tracking -> 简单异常规则/轻量模型 -> 对象级异常分数 -> 帧级异常分数
```

目标是验证：

- ShanghaiTech 的异常是否能被对象框和轨迹解释；
- 哪些对象类别最有用；
- tracking 是否足够稳定；
- 哪些异常类型需要更强的视觉 token 或 query 模型。

如果这些基础信号有效，再继续设计 object-conditioned anomaly query。

## 为什么先做这个

直接做 object query 风险较高：

- 需要对象 token 聚合模块；
- 需要对象级伪标签；
- 需要 token retention head；
- 训练和 ablation 成本高；
- 如果对象框和轨迹本身没有信号，query 设计会变成盲目调参。

先做 baseline anomaly tests 可以快速判断研究方向是否成立。

## 可先测试的 Anomaly 方法

## 方法 1：异常对象出现规则

直接把某些对象类别在步行校园场景中的出现视为高风险。

高风险对象：

```text
bicycle
motorcycle
motorbike
scooter
car
vehicle
skateboard
```

对象分数：

```text
s_i(t) = class_prior(label_i)
```

帧分数：

```text
S_frame(t) = max_i s_i(t)
```

用途：

- 测试非行人交通工具异常；
- 验证 LocateAnything 离线标注的 object recall；
- 作为最简单的 object-level baseline。

局限：

- 无法检测 running、fighting、loitering；
- 可能把静止停放物体误报为异常。

## 方法 2：基于速度的轨迹异常

对每条 track 计算中心点速度。

```text
v_i(t) = ||center_i(t) - center_i(t-1)||
```

用训练集正常视频估计每个场景的正常速度分布：

```text
mu_scene, sigma_scene
```

异常分数：

```text
s_i(t) = zscore(v_i(t), mu_scene, sigma_scene)
```

用途：

- running；
- sudden motion；
- bicycle/skateboard 快速穿越；
- camera-shake 需要额外过滤。

建议：

- 每个 scene 单独建正常速度分布；
- 用 track 长度过滤短轨迹；
- 坐标最好按图像宽高归一化。

## 方法 3：方向异常

每个 scene 学习正常人流方向。对 person track 或 vehicle track 计算运动方向：

```text
d_i(t) = normalize(center_i(t) - center_i(t-k))
```

与场景主方向比较：

```text
s_i(t) = 1 - cosine(d_i(t), d_normal_scene)
```

用途：

- wrong direction；
- restricted path；
- 逆行或反常穿越。

局限：

- 多方向场景需要聚类多个 normal direction；
- 对短轨迹不稳定。

## 方法 4：人-人交互异常

对 person tracks 计算 pairwise distance、relative velocity 和持续接近关系。

高风险模式：

- 两人快速接近；
- 多人距离很近且速度变化剧烈；
- person tracks 长时间重叠或纠缠；
- 局部人群突然聚集。

简单分数：

```text
s_i(t) = max_j InteractionRisk(track_i, track_j)
```

用途：

- chasing；
- fighting；
- pushing；
- collision。

局限：

- 单靠 box 可能无法区分正常同行和 fighting；
- 以后可能需要局部视觉 token 或 pose。

## 方法 5：人-物交互异常

关注 person 与 bag、box、cart、suitcase 等对象的关系。

高风险模式：

- object 长时间离开所有 person，疑似 abandoned；
- person 与 cart/box/suitcase 持续共同移动；
- object 突然高速运动，疑似 throwing；
- person 与 vehicle/skateboard 近距离共同移动。

对象分数：

```text
s_object(t) = RelationRisk(object_track, nearest_person_track)
```

用途：

- abandoned bag；
- pushing cart；
- carrying object；
- throwing object 的弱信号。

## 方法 6：异常区域覆盖的 Oracle 测试

用 `testframemask/*.npy` 的异常 mask 检查对象框是否覆盖异常区域。

```text
cover_i(t) = area(box_i(t) ∩ mask_t) / area(box_i(t))
```

这个方法不是最终模型，因为它使用了测试标注。但它非常重要：

- 验证我们检测的对象是否覆盖真实异常区域；
- 判断哪些异常无法被当前 object list 捕捉；
- 为后续对象级伪标签提供依据。

推荐指标：

- abnormal frame object recall；
- abnormal mask coverage；
- abnormal object category distribution；
- normal object false positive count。

## 方法 7：轻量异常模型

在规则特征上训练轻量模型，而不是直接上 object query。

输入特征：

```text
label embedding
box center/size
velocity
acceleration
direction
track duration
nearest person distance
nearest suspicious object distance
scene id
```

候选模型：

- Isolation Forest；
- One-Class SVM；
- Logistic Regression with pseudo labels；
- Gradient Boosting / XGBoost；
- small MLP。

建议优先：

```text
Isolation Forest -> Logistic Regression -> small MLP
```

原因：

- 可解释；
- 实现快；
- 方便定位 object/query 是否真的必要。

## 推荐实验顺序

1. 离线检测最小对象集合：

```text
person
bicycle
motorcycle
motorbike
scooter
car
vehicle
skateboard
bag
backpack
suitcase
box
cart
trolley
```

2. 对检测框做 tracking。

3. 先跑 oracle coverage：

```text
检测框是否覆盖异常 mask？
```

4. 跑异常对象出现规则。

5. 跑速度异常和方向异常。

6. 跑人-人、人-物关系异常。

7. 汇总 frame-level score：

```text
S_frame(t) = max_i s_i(t)
```

8. 计算 frame-level AUC 和 object coverage。

9. 再决定是否恢复 object query。

## 第一阶段输出文件

建议输出：

```text
outputs/shanghai_object_labels/{video_id}.json
outputs/shanghai_tracks/{video_id}.json
outputs/shanghai_baselines/{method_name}/{video_id}.json
outputs/shanghai_baselines/summary.csv
```

对象检测结果格式：

```text
video_id, frame_idx, label, x1, y1, x2, y2, confidence, prompt_group
```

轨迹结果格式：

```text
video_id, frame_idx, track_id, label, x1, y1, x2, y2, score
```

异常分数格式：

```text
video_id, frame_idx, track_id, label, object_score, frame_score, reason
```

## 判断是否值得继续 Object Query

继续 object query 的条件：

- 异常 mask 大多能被对象框覆盖；
- person/vehicle/object tracks 能解释主要异常；
- 简单规则已有一定 frame-level AUC；
- 误检主要来自复杂动作或交互，而不是对象漏检。

暂缓 object query 的条件：

- 异常区域经常没有被任何对象框覆盖；
- tracking 大量断裂；
- 异常主要是姿态或细粒度动作，box 级特征不够；
- LocateAnything 离线标注召回不足。

## 当前建议

当前应优先实现：

```text
LocateAnything 离线检测 + tracking + oracle coverage + speed/object-presence baseline
```

这一步可以快速验证对象级路线是否成立。object query 放到第二阶段，等确认对象框和轨迹能覆盖足够多异常后再做。

