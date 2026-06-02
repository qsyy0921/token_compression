# 没有对象级 Label 时怎么办

## 当前问题

我们现在没有人工对象级 label：

```text
每个对象是不是异常？
每个对象属于哪种异常？
哪些对象 token 应该删？
```

这不是 blocker。ShanghaiTech 本身的设定就是训练集正常、测试集有异常标注。我们可以用下面三类信号构造对象级监督：

- 训练集正常视频；
- 测试集 frame/mask 异常标注；
- LocateAnything 离线对象框和 tracking。

## 可以使用的监督来源

| 来源 | 是否已有 | 用途 |
| --- | --- | --- |
| 训练集正常视频 | 有 | 学正常对象、正常轨迹、正常位置和速度分布 |
| 测试集 frame-level 异常 | 可从 mask 得到 | 评估 frame-level AUC |
| 测试集 pixel-level mask | 有 `testframemask/*.npy` | 生成对象级伪标签、做 oracle coverage |
| LocateAnything 对象框 | 需要离线生成 | 对象 crop、对象类别、token 区域 |
| Tracking 轨迹 | 需要生成 | 运动异常、关系异常、对象级时间一致性 |

## 方案 1：完全不需要异常对象 Label

只用训练集正常视频做 normality modeling。

流程：

```text
训练集正常视频
-> LocateAnything / YOLO 离线检测
-> tracking
-> 提取对象轨迹特征
-> 建正常对象/轨迹分布
-> 测试对象离正常分布越远，异常分数越高
```

对象特征：

```text
label
box center
box size
velocity
acceleration
direction
track duration
nearest object distance
scene id
```

可用模型：

- kNN distance；
- KMeans prototype distance；
- Isolation Forest；
- One-Class SVM；
- Gaussian / Mahalanobis distance；
- scene-specific histogram。

优点：

- 不需要异常对象 label；
- 符合 ShanghaiTech 训练集只有正常视频的设定；
- 低算力可跑。

缺点：

- 复杂语义异常，如 fighting/stealing，可能只靠轨迹不够；
- 依赖检测和 tracking 质量。

## 方案 2：用异常 Mask 自动生成对象级伪标签

测试集有 `testframemask/*.npy`。我们可以把对象框和异常 mask 对齐，生成伪对象标签。

对对象框：

```text
box_i(t)
```

对异常 mask：

```text
mask_t
```

计算覆盖率：

```text
cover_i(t) = area(box_i(t) ∩ mask_t) / area(box_i(t))
```

伪标签规则：

```text
y_i(t) = 1      if cover_i(t) >= 0.30
y_i(t) = 0      if frame is normal and cover_i(t) == 0
y_i(t) = ignore otherwise
```

也可以使用 mask 被对象框覆盖的比例：

```text
mask_cover_i(t) = area(box_i(t) ∩ mask_t) / area(mask_t)
```

如果异常区域主要被某个对象框覆盖，这个对象就是异常候选。

优点：

- 自动得到对象级伪标签；
- 可以训练 logistic regression / 小 MLP；
- 可以评估异常对象 token 是否被保留。

缺点：

- 只能用于测试集或 development split，不能把结果当作无泄漏最终训练；
- mask 与对象框不一定精确对齐；
- 多对象交互异常中，mask 可能覆盖多个人，需要多对象共同标异常。

## 方案 3：弱标签 / MIL

如果只有 frame-level 异常标签，也可以用 MIL：

```text
S_frame(t) = max_i a_i(t)
```

训练或校准时：

```text
frame abnormal -> 至少一个对象异常
frame normal -> 所有对象正常
```

loss：

```text
L = BCE(max_i a_i(t), y_frame(t))
```

优点：

- 不需要对象级 label；
- 和 frame-level AUC 对齐。

缺点：

- 容易把异常分数分配给错误对象；
- 最好配合 mask 伪标签或对象先验。

## 方案 4：文本/类别先验作为零样本标签

不需要人工 label，只用 ShanghaiTech 的场景定义：

```text
campus pedestrian scene
```

高风险类别：

```text
bicycle
motorcycle
car
skateboard
scooter
```

高风险动作文本：

```text
running
fighting
chasing
throwing objects
abandoned bag
```

可以用：

- 类别先验；
- CLIP object-text similarity；
- Qwen3-VL 对短 clip 的零样本描述；
- 手工规则。

优点：

- training-free；
- 适合低算力；
- 可解释。

缺点：

- 语义先验不等于数据集真实标注；
- 需要用 mask 和 AUC 验证是否有效。

## 推荐路线

当前最稳路线是：

```text
1. 训练集正常视频 -> 建正常对象/轨迹分布
2. 测试集 mask -> 生成对象级伪标签，只用于验证和轻量校准
3. 类别/文本先验 -> 提供 training-free 初始异常分数
4. 融合得到对象异常分数 a_i(t)
5. 用 a_i(t) 决定 Qwen3-VL token 删除
```

不要一开始人工标对象 label。

## 具体执行顺序

### Step 1：对象框和轨迹

用 LocateAnything 离线检测对象：

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

然后做 tracking，得到：

```text
track_id, label, frame_idx, box
```

### Step 2：正常分布

只用 training split：

```text
data/shanghai/data/training
```

建每个 scene、每个 label 的：

- 速度分布；
- 方向分布；
- 位置热力图；
- box size 分布；
- person-object 距离分布。

### Step 3：测试集伪标签

用：

```text
data/shanghai/data/testframemask/*.npy
```

对测试对象框生成：

```text
object_pseudo_label = abnormal / normal / ignore
```

这一步用于：

- oracle coverage；
- 轻量 fusion 权重校准；
- token 删除策略验证。

### Step 4：对象分数

无标签版本：

```text
a_i(t) = normality_distance + class_prior + uncertainty
```

伪标签校准版本：

```text
a_i(t) = LogisticRegression(features_i(t))
```

其中 features 包括：

```text
A_class
A_motion
A_relation
A_scene
A_semantic
U
```

### Step 5：Token 删除标签

用对象异常分数生成 token retention pseudo label：

```text
retain = 1 if a_i(t) >= tau_high
retain = 1 if uncertainty high
retain = 0 if a_i(t) <= tau_low and frame normal
ignore otherwise
```

推荐：

```text
tau_high = 0.6
tau_low = 0.2
```

## 避免数据泄漏

如果使用测试集 mask 生成对象伪标签，要注意：

- 不能把测试集 mask 训练出的结果当作官方最终泛化成绩；
- 可以把它标为 development / oracle / pseudo-label analysis；
- 最终无泄漏结果应只用训练集正常分布和 training-free 规则；
- 或者从测试集划分 dev/test，dev 用于校准，test 只用于评估。

推荐文档措辞：

```text
Mask-derived object labels are used for development analysis and oracle coverage validation, not as final benchmark supervision unless a held-out split is used.
```

## 结论

没有对象级 label 时，第一阶段不需要人工标注。我们可以用：

- 训练集正常对象分布；
- 测试集 mask 自动伪标签；
- 类别和文本先验；
- tracking 统计；
- 轻量 one-class / logistic 模型；

来得到对象异常分数，并进一步生成 Qwen3-VL token 删除策略。

