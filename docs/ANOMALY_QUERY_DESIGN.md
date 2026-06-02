# 对象条件化 Anomaly Query 设计

## 核心想法

我们要设计的 anomaly query 不应该是一个全局帧级 query，例如“这帧是否异常”。ShanghaiTech 的异常大多发生在具体对象或对象交互上，因此 query 应该是对象条件化的：

```text
给定对象 o_i 的视觉 token、轨迹 token、场景上下文和邻近对象关系，判断该对象在时间 t 是否异常。
```

对应分数定义：

```text
s_i(t) = AnomalyQuery(o_i, track_i, context_t, neighbors_t)
S_frame(t) = max_i s_i(t)
```

这样可以同时满足两个目标：

- 做对象级异常检测；
- 用对象风险分数指导 token 压缩，保留异常相关对象 token，压缩低风险正常对象 token。

## 为什么不用单一帧级 Query

帧级 query 有三个问题：

- 它只能输出整帧异常分数，不能解释异常来自哪个对象；
- 它无法自然决定哪些对象 token 应该被压缩；
- 它容易把背景变化、光照和拥挤程度当成异常证据。

对象条件化 query 更适合 ShanghaiTech，因为异常通常可以落到具体对象轨迹：

| 异常 | 应询问的对象 |
| --- | --- |
| running | person track |
| chasing / fighting | 多个 person tracks |
| bicycle / car / skateboard in pedestrian area | vehicle/board track and nearby person track |
| loitering / wrong direction | person track with scene context |
| abandoned bag / moving cart | object track and person-object relation |

## Query 输入

每个对象 query 的输入建议包含四部分。

### 1. 对象视觉 Token

来自对象框内的视觉 token 或 ROI token：

```text
z_obj_i(t)
```

用途：

- 判断对象类别和外观；
- 识别是否是 person、bicycle、car、bag 等异常相关对象；
- 估计姿态、遮挡和局部外观变化。

### 2. 轨迹 Token

来自同一对象过去 K 帧的轨迹特征：

```text
z_track_i(t-K:t)
```

建议包含：

- box center；
- box size；
- velocity；
- acceleration；
- direction；
- track duration；
- missing/interpolation flags。

用途：

- 判断 running、sudden motion、wrong direction、loitering；
- 区分静止对象和移动对象；
- 发现轨迹突然变化。

### 3. 场景上下文 Token

来自全局帧或低分辨率背景 token：

```text
z_scene(t)
```

用途：

- 判断当前对象是否出现在不合理区域；
- 学习每个场景的正常人流方向；
- 解释入口、台阶、栏杆、道路等区域规则。

这部分不应太重。它可以是压缩后的全局 token，避免背景 token 反过来主导异常判断。

### 4. 邻近对象关系 Token

来自当前对象附近的其他对象：

```text
z_rel_i(t) = Relation(o_i, neighbors_i)
```

建议关系：

- person-person distance；
- person-object distance；
- IoU / overlap；
- relative velocity；
- whether approaching；
- whether moving together；
- whether object becomes unattended。

用途：

- chasing；
- fighting；
- pushing/collision；
- carrying/abandoned object。

## Query 形式

建议使用可学习 query，而不是纯文本 prompt。文本可以作为初始化或解释，不应作为最终推理依赖。

### 方案 A：单个通用 Anomaly Query

使用一个共享 query：

```text
q_anom
```

它对每个对象重复使用：

```text
h_i = CrossAttention(q_anom, [z_obj_i, z_track_i, z_scene, z_rel_i])
s_i = MLP(h_i)
r_i = Sigmoid(MLP_retention(h_i))
```

其中：

- `s_i` 是对象异常分数；
- `r_i` 是对象 token 保留概率。

优点：

- 简单；
- 参数少；
- 适合第一版 baseline。

缺点：

- 不同异常机制混在同一个 query 中，解释性较弱。

### 方案 B：多类型 Anomaly Queries

定义多个可学习 query，每个 query 对应一种异常机制：

```text
q_motion
q_interaction
q_vehicle
q_region
q_object_relation
q_unknown
```

输出：

```text
s_i^motion
s_i^interaction
s_i^vehicle
s_i^region
s_i^object_relation
s_i^unknown
```

最终对象分数：

```text
s_i = max_k s_i^k
```

或者：

```text
s_i = WeightedSum_k s_i^k
```

推荐 query 对应关系：

| Query | 负责异常 |
| --- | --- |
| `q_motion` | running, sudden motion, wrong direction |
| `q_interaction` | chasing, fighting, pushing, collision |
| `q_vehicle` | bicycle, motorcycle, car, skateboard, scooter in pedestrian scene |
| `q_region` | restricted-zone entry, climbing, jumping |
| `q_object_relation` | abandoned bag, cart movement, carrying suspicious object |
| `q_unknown` | mask 覆盖到运动对象但不确定类别 |

优点：

- 更符合 ShanghaiTech 异常结构；
- 可解释性强；
- 后续 ablation 清晰。

缺点：

- 需要更谨慎的伪标签或弱监督设计。

第一版建议从方案 A 开始，随后扩展到方案 B。

## 与 Token 压缩的连接

Anomaly query 不只是输出异常分数，还应该输出 token 保留策略。

每个对象得到两个值：

```text
s_i(t): anomaly risk
r_i(t): token retention probability
```

推荐保留规则：

```text
r_i(t) = clamp(alpha * s_i(t) + beta * uncertainty_i(t) + gamma * class_prior_i)
```

其中：

- `s_i(t)`：对象异常风险；
- `uncertainty_i(t)`：检测/轨迹/分类不确定性；
- `class_prior_i`：对象类别先验，例如 person、vehicle、skateboard 更重要。

对象 token 压缩策略：

| 对象状态 | Token 策略 |
| --- | --- |
| 高异常分数 | 保留完整对象 token |
| 高不确定性 | 暂时保留，等待后续帧消歧 |
| 正常 person | 中等压缩 |
| 正常背景/上下文对象 | 强压缩 |
| 静态低风险对象 | 只保留低维摘要 |

这使模型可以形成明确主张：

```text
不是压缩整帧 token，而是根据对象异常风险自适应压缩对象 token。
```

## 监督信号设计

ShanghaiTech 主要提供 frame-level / pixel-level anomaly mask，不直接提供对象级异常类别。因此对象 query 的监督应来自伪标签。

### 对象级伪标签

对每个 track 在每一帧计算：

```text
y_i(t) = 1 if IoU(box_i(t), anomaly_mask_t) > tau
```

或使用覆盖率：

```text
cover_i(t) = area(box_i(t) ∩ mask_t) / area(box_i(t))
```

建议：

- `cover_i(t) >= 0.3` 标为异常对象；
- `cover_i(t) == 0` 且该帧正常，标为正常对象；
- 中间区域标为 uncertain，不参与强监督 loss。

### Frame-level MIL 约束

因为最终评估仍常用 frame-level AUC，可以使用 MIL：

```text
S_frame(t) = max_i s_i(t)
L_frame = BCE(S_frame(t), y_frame(t))
```

### Ranking 约束

同一帧内异常对象应高于正常对象：

```text
L_rank = max(0, margin - s_pos + s_neg)
```

这适合对象级 token 压缩，因为它直接学习“哪些对象更值得保留”。

### Retention 约束

让 token 保留率和异常风险一致：

```text
L_retention = BCE(r_i(t), y_i(t)) + lambda * mean(r_i(t))
```

第二项鼓励压缩，避免全部保留。

## 第一版模型结构

建议第一版保持简单：

```text
Object tokens from frame/video model
        +
Track features from offline labels
        +
Scene summary token
        +
Relation features
        ↓
Object-conditioned anomaly query
        ↓
object anomaly score s_i(t)
object retention score r_i(t)
        ↓
frame score max_i s_i(t)
token compression by r_i(t)
```

模块：

- `ObjectTokenEncoder`：聚合对象框内视觉 token；
- `TrackEncoder`：编码轨迹速度、方向、持续时间；
- `RelationEncoder`：编码邻近对象关系；
- `SceneContextEncoder`：低成本场景摘要；
- `AnomalyQueryHead`：输出 `s_i(t)` 和 `r_i(t)`。

## Ablation 设计

建议至少做以下 ablation：

| 变体 | 目的 |
| --- | --- |
| A0 frame query only | 证明全局帧 query 不足以解释对象异常 |
| A1 object query without tracking | 证明单帧对象框不够 |
| A2 object query + tracking | 主要模型 |
| A3 object query + tracking + relation | 验证人-人/人-物交互 |
| A4 multi-type anomaly queries | 验证多 query 是否提升解释性和性能 |
| A5 retention without anomaly supervision | 验证异常监督对 token 压缩的必要性 |

## 推荐实现顺序

1. 用 LocateAnything 离线生成对象框。
2. 用 tracking 生成对象轨迹。
3. 用 anomaly mask 给对象轨迹生成伪标签。
4. 实现单个 `q_anom` 的对象 query baseline。
5. 加入 token retention head。
6. 加入 relation encoder。
7. 扩展为多类型 anomaly queries。

## 关键风险

| 风险 | 处理方式 |
| --- | --- |
| LocateAnything 漏检异常对象 | 高召回 prompt、抽帧密度提高、用 YOLO/BoT-SORT 补充 person |
| 伪标签噪声高 | 使用 mask 覆盖率阈值和 uncertain 区间 |
| tracking 断裂 | 使用 tracklet 合并，不把短断裂当作新对象异常 |
| 模型退化为 frame classifier | 强制输出对象分数，使用 object ranking loss |
| token 压缩全部保留 | 加 retention budget loss |
| token 压缩误删异常对象 | 高风险和高不确定对象默认保留 |

## 推荐结论

第一版最合理的设计是：

```text
单个对象条件化 anomaly query + tracking features + object-level pseudo labels + retention head
```

不要一开始就做复杂文本 query 或大语言 prompt。我们需要的是能训练、能 ablation、能和 token 压缩绑定的可学习 query。

后续可以扩展到多类型 anomaly queries，让每个 query 对应一种异常机制，但第一版应先验证对象条件化 scoring 和 token retention 是否成立。

