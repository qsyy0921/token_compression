# 对象级异常向量训练方案：当前验证阶段

版本：2026-06-13

## 1. 当前阶段目标

当前阶段不训练完整 token compression 系统，而是先验证一个核心假设：

> 从冻结的 Qwen3-VL visual tokens 中抽取 object-track / relation-track 特征后，训练 normal/T01-T05 anomaly prototypes，是否能够给对象产生合理的异常分数。

当前主训练链路停在 object anomaly score：

```text
video
-> frozen Qwen3-VL visual tokens
-> frozen YOLO/HybridSORT tracks
-> fixed bbox-to-token binding
-> object-track / relation-track samples
-> Projection Head + Temporal Pooling / lightweight Temporal Encoder
-> normal + T01-T05 anomaly prototypes
-> object anomaly score
```

Token compression 和 Qwen3-VL LLM 解释只作为后续验证模块，不进入当前 anomaly vector 训练 loss。

这里的“简单验证”只表示使用的数据量少，不表示训练流程被简化。即使只挑少量 package，也必须完整保留：

- object-to-token binding
- object-track / relation-track 样本构造
- positive / negative 样本定义
- `label: normal / T01-T05`
- Projection Head
- Temporal Pooling / lightweight Temporal Encoder
- normal/T01-T05 prototype bank
- `L_coarse + L_binary + L_sep`
- object anomaly score 与 category 输出
- 对象级指标、开放集观察和后续 token compression 对照验证

因此当前阶段是“小数据完整链路验证”，不是“删掉关键模块的简化版”。

## 1.1 数据与代码组织

数据仓库路径：

```text
/home/expand_disk/data_repository/mfl/token_compression/20260613_data
```

该路径只放数据、训练方法文档和实验结果产物，不放训练脚本。建议组织为：

```text
20260613_data/
  packages/                 # 已传入的数据包
  training_method/          # 训练方法文档
  results/                  # 实验结果、指标、可视化、日志摘要
```

代码仓库路径：

```text
/home/expand_disk/code_repository/mfl/token_compression
```

该路径放脚本、配置和方法/结果备份：

```text
token_compression/
  scripts/                  # 训练、抽特征、构建索引、可视化脚本
  docs/design/20260613/     # 方法文档备份
  outputs/20260613/         # 实验结果备份
```

后续原则：

- 脚本只放在 code repository。
- data repository 里不放 Python 训练脚本。
- 方法文档与结果在 data repository 保存一份，同时在 code repository 备份一份。
- 原始 `packages/` 不移动、不删除、不混入脚本。

## 2. 标签空间

当前训练只使用 6 类：

```text
normal
T01 individual_human_behavior_anomaly
T02 light_mobility_rule_violation
T03 motor_vehicle_rule_violation
T04 aggression_or_crowd_disorder
T05 object_property_interaction_anomaly
```

暂时不训练 R06 prototype。R06 样本少且视觉模式混杂，当前只用于 rare/open-set evaluation。

S01-S22 细分类暂时只作为 metadata 保存，不作为当前主监督标签。

## 3. 冻结、固定与训练模块

冻结模块：

- Qwen3-VL Vision Encoder：只抽取 visual tokens，不微调。
- Qwen3-VL LLM：当前训练阶段不使用 LLM loss，不微调。
- YOLO 检测器与 HybridSORT tracker：使用已有 bbox / track_id。
- 原始 video、tracks、annotations：只作为数据和监督来源。

固定规则：

- BBox-to-Token alignment：第一版使用几何规则，不训练。
- Negative sampling rule：第一版按时间窗口和空间距离规则采样。

当前训练模块：

- Projection Head：把 object-track token 投影到 anomaly embedding space。
- Temporal Pooling / lightweight Temporal Encoder：聚合同一对象在时间窗口内的特征。
- Normal prototypes。
- T01-T05 anomaly vectors / prototypes。
- 可选 Relation Encoder：第一版可以先不开，第二版再加入。

不训练模块：

- Token compression policy。
- Qwen3-VL LLM。
- R06 prototype。
- S01-S22 fine-grained classifier。

## 4. 样本构造

每个训练样本必须显式包含 label：

```json
{
  "sample_type": "object_track",
  "package_id": "Avenue_07",
  "dataset": "avenue",
  "track_ids": [40],
  "object_classes": ["person"],
  "time_range": [start_frame, end_frame],
  "label": "T01",
  "fine_subtype": "S01",
  "is_positive": true
}
```

relation 样本示例：

```json
{
  "sample_type": "relation_track",
  "package_id": "NWPU_D054_02",
  "dataset": "nwpu",
  "track_ids": [1, 5],
  "object_classes": ["person", "skateboard"],
  "time_range": [start_frame, end_frame],
  "label": "T02",
  "fine_subtype": "S06",
  "is_positive": true
}
```

normal 样本示例：

```json
{
  "sample_type": "object_track",
  "package_id": "Avenue_07",
  "dataset": "avenue",
  "track_ids": [12],
  "object_classes": ["person"],
  "time_range": [start_frame, end_frame],
  "label": "normal",
  "negative_strength": "strong",
  "is_positive": false
}
```

正样本来源：

- `annotation.json` 中事件时间窗内的 `related_objects`。
- label 使用事件的 `train_category`，只保留 T01-T05。
- 若一个事件包含多个 related object，先构造单对象样本；对 T02/T04/T05 再构造 relation 样本。

负样本来源：

- strong negative：非事件时间段内的 track。
- medium negative：事件时间段内、但空间上远离异常对象的无关 track。
- weak negative：事件时间段内未标注为 anomalous 的 track；第一版可少用或不用。

建议第一版正负比例：

```text
positive : negative = 1 : 2
```

每个正样本至少配一个同视频 negative，再配一个跨视频 normal negative。

## 5. Object-to-Token Binding

给定第 t 帧的 Qwen visual tokens：

```text
X_t = {x_t,i}
```

和 tracker bbox：

```text
B_o,t
```

用几何规则选择属于对象的 token：

```text
X_o,t = {x_t,i | token_i center inside expanded_bbox(B_o,t)}
```

第一版规则：

- bbox 向外扩 5%-10%，避免边界 token 丢失。
- token center 落在 bbox 内即归属该对象。
- 如果 bbox 内 token 数太少，至少取与 bbox IoU 最大的 top-k tokens。
- 对重叠 bbox，可以允许 token 属于多个对象；relation 阶段再处理冲突。

每帧 object token：

```text
z_o,t = MeanPool(X_o,t)
```

第二版可以替换为 attention pooling：

```text
z_o,t = AttentionPool(X_o,t)
```

## 6. Object-Track Feature

一个时间窗口内，同一 track 有一组 object tokens：

```text
z_o,t1, z_o,t2, ..., z_o,tL
```

第一版使用 Temporal Pooling：

```text
h_o,w = MeanPool_t(z_o,t)
```

更强版本使用 lightweight Temporal Encoder：

```text
h_o,w = TemporalEncoder(z_o,t1 ... z_o,tL)
```

对于 running、逆行、打架等强时序异常，可以额外拼接轻量运动特征：

```text
g_o,t = [cx, cy, w, h, delta_cx, delta_cy, scale_change, confidence]
```

建议做两个 ablation：

```text
visual-only
visual + motion
```

这样可以区分模型到底是从 object visual tokens 中学到异常，还是只靠速度规则完成判断。

## 7. Relation-Track Feature

当前数据中大量事件是 `object_relation`，因此 relation feature 后续很重要。

第一版可以先不训练 relation encoder，只做 object-track prototype 验证。

第二版加入 relation-track sample：

```text
r_i,j,w = RelationEncoder(h_i,w, h_j,w, spatial_relation_i,j,w)
```

其中 spatial relation 包括：

```text
relative_dx, relative_dy, bbox_iou, distance, size_ratio, temporal_overlap
```

不同类别建议：

- T01：优先单 person object-track。
- T02：person + bicycle/skateboard relation。
- T03：vehicle object-track + scene/context。
- T04：person-person relation 或 group feature。
- T05：person-object relation。

## 8. Prototype Bank

训练一个 prototype bank：

```text
P_normal = {n_1 ... n_K}
P_T01 = {a_1,1 ... a_1,M}
...
P_T05 = {a_5,1 ... a_5,M}
```

小样本第一版建议：

```text
K = 2 或 4
M = 1 或 2
```

不要一开始为每个类别放太多 prototype，否则小样本下容易过拟合。

Projection 后归一化：

```text
q_i = normalize(MLP(h_i))
p_j = normalize(p_j)
```

每类 logit：

```text
logit_c = tau * max_m cosine(q_i, p_c,m)
```

也可以用更平滑的 logsumexp：

```text
logit_c = tau * logsumexp_m(alpha * cosine(q_i, p_c,m)) / alpha
```

第一版用 max 即可，后续再换 logsumexp。

## 9. 输出定义

模型输出：

```text
P(normal), P(T01), P(T02), P(T03), P(T04), P(T05)
```

对象异常分数：

```text
score = 1 - P(normal)
```

对象异常类别：

```text
category = argmax P(T01:T05)
```

最终对象级结果：

```json
{
  "track_id": 40,
  "object_class": "person",
  "score": 0.87,
  "category": "T01",
  "time_range": [120, 180]
}
```

注意：score 是 softmax 概率派生的对象异常分数，不应写成无来源的置信度。

## 10. Loss 设计

当前只使用三个 loss：

```text
L = L_coarse + lambda_b * L_binary + lambda_s * L_sep
```

### L_coarse

normal/T01/T02/T03/T04/T05 多类分类损失：

```text
L_coarse = CE([logit_normal, logit_T01, ..., logit_T05], y)
```

### L_binary

normal vs anomaly 辅助二分类：

```text
p_anomaly = 1 - P(normal)
y_binary = 0 if y == normal else 1
L_binary = BCE(p_anomaly, y_binary)
```

### L_sep

prototype separation，防止 prototypes 塌缩：

```text
L_sep = mean_{i != j} max(0, cosine(p_i, p_j) - margin)^2
```

建议初始超参：

```text
lambda_b = 0.5
lambda_s = 0.1
margin = 0.2
tau = 10
```

暂时不加入：

- token keep loss
- LLM answer loss
- temporal smooth loss
- S01-S22 fine-grained loss

这些等 anomaly vectors 可用之后再加。

## 11. 小样本验证数据建议

当前只挑少量 package 验证，不使用全部 217 个。

候选包：

| 类别 | package | fine subtype | 说明 |
|---|---|---|---|
| T01 | Avenue_07 | S01 | running，person track |
| T01 | Avenue_18 | S02 | fall/posture，person tracks |
| T02 | NWPU_D054_02 | S06 | person + skateboard |
| T02 | Avenue_16 | S05 | bicycle + person |
| T03 | NWPU_D047_06 | S09 | car related violation |
| T03 | ShanghaiTech_01_0135 | S09 | car related violation |
| T04 | ShanghaiTech_03_0033 | S13 | person-person aggression |
| T04 | NWPU_D068_01 | S13 | person-person aggression |
| T05 | Avenue_11 | S15 | person + suitcase/object |
| T05 | NWPU_D047_03 | S17 | person + bicycle/object manipulation |
| R06 eval only | NWPU_D003_05 | S21 | dog intrusion，open-set eval only |

建议最小训练组合：

```text
T01: Avenue_07, Avenue_18
T02: NWPU_D054_02, Avenue_16
T03: NWPU_D047_06, ShanghaiTech_01_0135
T04: ShanghaiTech_03_0033, NWPU_D068_01
T05: Avenue_11, NWPU_D047_03
```

R06 只用于观察：

```text
NWPU_D003_05
```

## 12. 验证指标

当前验证不追求 SOTA，优先验证 object anomaly score 是否合理。

对象级指标：

- positive object recall：异常对象是否排在高分。
- top-k recall：每个事件 top-k object 是否包含 related_objects。
- normal false positive rate：普通对象是否被大量误判异常。
- category accuracy：T01-T05 大类是否预测正确。

排序指标：

- event 内异常对象分数是否高于无关对象。
- 同一视频内异常时间窗分数是否高于非事件时间窗。

开放集观察：

- R06 样本是否得到高 anomaly score。
- R06 不要求分类正确，只观察是否被 normal prototype 排斥。

后续 token compression 验证：

- baseline：不压缩，完整 tokens 输入 Qwen3-VL。
- rule-based compression：高分对象保留，低分对象/背景 merge 或 prune。
- 比较 VLM 输出是否更关注异常对象。
- 记录压缩率、输出差异、异常对象可视化。

## 13. 实施顺序

第一步：生成样本索引。

```text
event_object_index.jsonl
```

每行包含：

```text
package_id, dataset, sample_type, track_ids, object_classes,
time_range, label, fine_subtype, is_positive, negative_strength
```

第二步：抽取并缓存冻结特征。

```text
Qwen3-VL visual tokens
tracks / bbox alignment
object tokens
```

第三步：训练 scorer。

```text
Projection Head
Temporal Pooling / lightweight Temporal Encoder
normal + T01-T05 prototypes
```

第四步：对象级验证。

```text
object score ranking
positive object recall
category accuracy
normal false positive rate
```

第五步：后续 token compression 验证。

```text
score-driven merge/prune
Qwen3-VL reasoning comparison
token keep/delete visualization
```

## 14. 当前结论

当前方法应被表述为：

> Object-level anomaly prototype learning for later token compression.

而不是：

> End-to-end token compression training.

当前要先证明：

```text
object-track token -> anomaly prototype similarity -> object anomaly score
```

这条链路是有效的。只有当对象异常分数可靠之后，才有必要进一步验证基于该分数的 token compression 是否能改善 VLM 推理。
