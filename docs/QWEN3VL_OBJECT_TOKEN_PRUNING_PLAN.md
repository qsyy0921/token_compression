# 基于 Qwen3-VL-8B 的对象风险 Token 删除方案

## 当前研究目标

最终模型依靠 `Qwen3-VL-8B-Instruct` 做 ShanghaiTech 异常检测。我们的核心假设是：

```text
在异常检测中，并不是所有视觉 token 都有意义。大量正常背景对象、低风险正常对象和无关区域 token 可以被删除或强压缩。
```

目标不是让 LocateAnything 在线参与推理。LocateAnything 只用于离线给数据集生成对象伪标签。最终方法应在 Qwen3-VL 的视觉 token 中，根据对象异常风险选择保留或删除 token。

## Qwen3-VL 本地模型相关配置

本地模型：

```text
models/Qwen3-VL-8B-Instruct
```

关键配置：

| 项 | 值 |
| --- | --- |
| 架构 | `Qwen3VLForConditionalGeneration` |
| 文本 hidden size | 4096 |
| 视觉 hidden size | 1152 |
| 视觉输出 hidden size | 4096 |
| vision patch size | 16 |
| spatial merge size | 2 |
| temporal patch size | 2 |
| image token id | 151655 |
| video token id | 151656 |
| vision start/end token | 151652 / 151653 |

这意味着单个视觉 token 大致对应合并后的空间 patch 单元。对视频来说，还涉及 temporal patch。对象框可以被映射到一组视觉 token index，再按对象风险决定保留或删除。

## 方法定位

我们要做的是：

```text
Object-risk-guided visual token pruning for Qwen3-VL anomaly detection
```

不是：

```text
Use LocateAnything at inference time for anomaly detection
```

也不是：

```text
Train a large video anomaly model end-to-end
```

LocateAnything 的作用：

```text
离线对象标注 -> 对象框/类别/轨迹 -> 伪标签/规则分数 -> 指导 Qwen3-VL token pruning 设计
```

Qwen3-VL 的作用：

```text
接收被裁剪后的视觉 token -> 判断视频/帧是否异常 -> 输出异常分数或解释
```

## 总体流程

```text
ShanghaiTech frames/videos
        ↓
LocateAnything 离线对象标注
        ↓
Tracking 得到对象轨迹
        ↓
Training-free / 轻量规则估计对象风险
        ↓
把对象框映射到 Qwen3-VL 视觉 token
        ↓
删除低风险对象 token，保留高风险/不确定对象 token
        ↓
Qwen3-VL 做异常检测
```

## 对象风险定义

每个对象轨迹在时间 `t` 有一个风险分数：

```text
r_i(t) ∈ [0, 1]
```

它不是由大模型训练出来的第一阶段结果，而是由低算力方法产生：

```text
r_i(t) = f(class_prior, track_motion, relation_score, clip_text_score, uncertainty)
```

推荐组成：

| 分量 | 来源 | 作用 |
| --- | --- | --- |
| `class_prior` | 对象类别 | person、vehicle、skateboard 等更重要 |
| `track_motion` | tracking | running、wrong direction、sudden motion |
| `relation_score` | 对象关系 | fighting、chasing、人-物交互 |
| `clip_text_score` | CLIP 文本相似度 | training-free 语义异常先验 |
| `uncertainty` | 检测/轨迹不确定性 | 不确定对象先保留，避免误删 |

第一版可手工融合：

```text
r_i(t) = 0.25 * class_prior
       + 0.30 * track_motion
       + 0.20 * relation_score
       + 0.15 * clip_text_score
       + 0.10 * uncertainty
```

如果允许轻量训练，可以用 logistic regression 学融合权重。

## Token 删除策略

## 1. 对象框到视觉 Token 的映射

对每一帧，已知对象框：

```text
box_i = (x1, y1, x2, y2)
```

Qwen3-VL 图像 patch size 为 16，spatial merge size 为 2，因此合并后的 token 网格近似为：

```text
grid_h = ceil(H / (16 * 2))
grid_w = ceil(W / (16 * 2))
```

将对象框映射到 token 网格：

```text
token_x1 = floor(x1 / 32)
token_y1 = floor(y1 / 32)
token_x2 = ceil(x2 / 32)
token_y2 = ceil(y2 / 32)
```

对象 token 集合：

```text
T_i = tokens inside mapped box_i
```

对视频 token，还要乘上 temporal patch 对应的 frame group。

## 2. 风险控制的删除规则

对每个对象 token 集合 `T_i`：

| 对象风险 | Token 处理 |
| --- | --- |
| `r_i >= 0.7` | 完整保留 |
| `0.4 <= r_i < 0.7` | 保留中心 token + 边界 token，删除部分内部冗余 token |
| `0.2 <= r_i < 0.4` | 只保留对象摘要 token 或少量 pooled token |
| `r_i < 0.2` | 删除对象 token，或只保留低维 summary |

背景 token：

```text
如果不属于任何对象，默认强压缩或按固定比例采样。
```

高不确定对象：

```text
即使风险分数不高，也先保留。
```

原因：异常检测里漏掉异常对象比多保留几个 token 更危险。

## 3. Token 冲突处理

多个对象框重叠时，token 风险取最大值：

```text
r_token = max_i r_i for token ∈ T_i
```

保留规则：

```text
keep token if r_token >= threshold
```

这样可以避免 person 与 bicycle、person 与 bag 交互时误删关系区域。

## Qwen3-VL 推理策略

第一版不改模型权重，只改输入视觉 token 或视觉 token mask。

可做三种实验：

| 方案 | 描述 | 目的 |
| --- | --- | --- |
| Full tokens | 不删除 token | 上限 baseline |
| Random pruning | 随机删除同等比例 token | 证明不是简单 token 数量减少带来的效果 |
| Object-risk pruning | 按对象风险删除 token | 主方法 |

评价：

```text
frame-level AUC
token retention ratio
abnormal object token recall
normal object token compression ratio
```

## 为什么先 Training-Free

当前算力限制下，第一版不建议训练复杂模块。可以使用：

- LocateAnything 离线标注；
- tracking；
- CLIP text similarity；
- class prior；
- speed/direction/relation rules；
- logistic regression 或 KMeans/kNN 作为轻量可选项。

这些方法足够生成对象风险分数和 token 删除 mask。

## Prompt 设计

Qwen3-VL 最终异常检测 prompt 应保持简单，避免把方法变成 prompt engineering。

建议中文/英文都可测试，第一版用英文减少模型不确定性：

```text
You are given a surveillance video from a pedestrian campus scene.
Detect whether there is any abnormal event.
Focus on pedestrians, vehicles, skateboards, unusual motion, and person-object interactions.
Return an anomaly score from 0 to 1 and a short reason.
```

如果按帧/短 clip 推理：

```text
Analyze this short surveillance clip.
Is any object behaving abnormally in this pedestrian scene?
Return: anomaly_score, abnormal_object, reason.
```

注意：prompt 不负责 token 删除。token 删除在视觉 token 输入阶段完成。

## 与 Object Query 的关系

Object query 可以后置。

当前阶段：

```text
使用 training-free 对象风险 -> 生成 token pruning mask -> 测 Qwen3-VL 异常检测效果
```

后续阶段：

```text
如果规则风险有效，再训练轻量 object risk predictor 或 object query
```

因此 object query 不是第一阶段必需组件。

## 第一批实验建议

1. 用 LocateAnything 离线检测最小对象集合。
2. 做 tracking。
3. 用规则产生对象风险：

```text
class_prior + speed + direction + relation + uncertainty
```

4. 把对象框映射到 Qwen3-VL token 网格。
5. 生成三种 token mask：

```text
full
random
object-risk
```

6. 用 Qwen3-VL 在同样 prompt 下跑异常检测。
7. 比较：

```text
AUC vs token retention ratio
abnormal frame recall
false positive cases
which objects were removed
```

## 最重要的 Claim

如果实验成立，我们的核心 claim 是：

```text
Qwen3-VL 在 ShanghaiTech 异常检测中不需要保留所有视觉 token。基于对象风险的 token 删除可以显著减少正常对象和背景 token，同时保留异常相关对象 token，并尽量维持异常检测能力。
```

这比“用外部检测器做异常检测”更合理。外部检测器只用于构造对象风险先验；真正评估的是 Qwen3-VL 在 token 被删除后的异常检测能力。

