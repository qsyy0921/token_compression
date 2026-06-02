# 对象异常分数与 Token 删除策略

## 核心问题

我们真正需要解决的是：

```text
如何给每个对象 o_i 在时间 t 计算异常分数 a_i(t)，并用这个分数决定该对象对应的 Qwen3-VL 视觉 token 是否删除？
```

最终 token 删除不是按类别硬删，而是按对象异常风险删：

```text
高风险对象 token 保留
低风险对象 token 删除或强压缩
不确定对象 token 暂时保留
```

## 对象异常分数定义

对每个对象轨迹 `o_i`，定义对象异常分数：

```text
a_i(t) ∈ [0, 1]
```

它由五类信号组成：

```text
a_i(t) = Fusion(
    A_class_i,
    A_motion_i(t),
    A_relation_i(t),
    A_scene_i(t),
    A_semantic_i(t),
    U_i(t)
)
```

其中：

| 符号 | 含义 | 是否需要训练 |
| --- | --- | --- |
| `A_class_i` | 类别先验异常分数 | 不需要 |
| `A_motion_i(t)` | 运动/轨迹异常分数 | 不需要或轻量 |
| `A_relation_i(t)` | 人-人、人-物关系异常分数 | 不需要 |
| `A_scene_i(t)` | 场景区域/方向异常分数 | 不需要或轻量 |
| `A_semantic_i(t)` | CLIP/文本语义异常分数 | 不需要 |
| `U_i(t)` | 不确定性分数 | 不需要 |

第一版建议全部使用 training-free 分数，最多用 logistic regression 学一个融合权重。

## 1. 类别先验异常分数

ShanghaiTech 是校园步行场景，因此对象类别本身就有异常先验。

定义：

```text
A_class_i = prior(label_i)
```

推荐先验：

| 类别 | 先验分数 |
| --- | --- |
| person | 0.35 |
| bicycle / bike | 0.90 |
| motorcycle / motorbike | 0.95 |
| scooter | 0.85 |
| car / vehicle / truck | 0.95 |
| skateboard | 0.90 |
| cart / trolley / stroller | 0.55 |
| bag / backpack / suitcase | 0.45 |
| box / package | 0.45 |
| umbrella | 0.25 |
| bench / chair / stairs / door / gate / fence | 0.10 |
| unknown moving object | 0.60 |

解释：

- `person` 本身不是异常，但几乎所有动作异常都依赖 person，所以不能给太低；
- 交通工具和滑行对象在 pedestrian campus 场景中高风险；
- 包、箱子、推车是交互风险，不宜直接设成高异常；
- 静态场景对象只作为上下文，默认低风险。

## 2. 运动异常分数

运动异常主要覆盖：

- running；
- sudden motion；
- wrong direction；
- fast bicycle/skateboard；
- abnormal stop/start。

### 速度分数

对 track center：

```text
c_i(t) = ((x1+x2)/2, (y1+y2)/2)
v_i(t) = ||c_i(t) - c_i(t-k)|| / k
```

用训练集正常视频建立 scene-level 正常速度分布：

```text
mu_v(scene, label), sigma_v(scene, label)
```

速度异常：

```text
z_v = (v_i(t) - mu_v) / (sigma_v + eps)
A_speed_i(t) = sigmoid((z_v - tau_v) / temp_v)
```

推荐初始值：

```text
tau_v = 2.0
temp_v = 1.0
```

### 加速度分数

```text
acc_i(t) = ||v_i(t) - v_i(t-k)||
```

同样用正常分布转成 z-score。

### 方向异常分数

每个 scene 用训练集 person tracks 学正常方向原型：

```text
D_scene = {d_1, d_2, ..., d_K}
```

测试方向：

```text
d_i(t) = normalize(c_i(t) - c_i(t-k))
```

方向异常：

```text
A_dir_i(t) = 1 - max_j cosine(d_i(t), d_j)
```

如果场景有多个正常方向，取最接近的方向原型。

### 运动总分

```text
A_motion_i(t) = max(A_speed_i(t), A_acc_i(t), A_dir_i(t))
```

## 3. 关系异常分数

关系异常主要覆盖：

- chasing；
- fighting；
- pushing；
- collision；
- person-object interaction；
- abandoned object。

### 人-人关系

对 person track `i` 和 person track `j`：

```text
dist_ij(t) = ||c_i(t) - c_j(t)||
rel_v_ij(t) = ||v_i(t) - v_j(t)||
approach_ij(t) = dist_ij(t-k) - dist_ij(t)
```

风险模式：

- 距离很近；
- 相对速度高；
- 持续接近；
- 多帧重叠或纠缠。

分数：

```text
A_pp_i(t) = max_j [
    w_d * close_score(dist_ij)
  + w_r * rel_velocity_score(rel_v_ij)
  + w_a * approach_score(approach_ij)
]
```

### 人-物关系

对 person 和 object：

```text
dist_po(t)
same_motion_po(t) = cosine(v_person, v_object)
```

风险：

- person 与 bicycle/skateboard/scooter 共同移动；
- person 与 cart/box/suitcase 持续接近并共同移动；
- bag/box 离开所有 person 后长时间静止。

Abandoned object 分数：

```text
A_abandon_o(t) =
    static_score(o)
  * no_near_person_score(o)
  * duration_score(o)
```

### 关系总分

```text
A_relation_i(t) = max(A_pp_i(t), A_po_i(t), A_abandon_i(t))
```

## 4. 场景异常分数

场景异常主要覆盖：

- restricted-zone entry；
- wrong region；
- climbing / crossing fence；
- object appears in unusual location。

第一版不做人工 scene mask 时，可以用训练集正常对象位置分布。

对每个 scene 和 label 建位置热力图：

```text
P_normal(x, y | scene, label)
```

对象位置：

```text
c_i(t)
```

场景异常分数：

```text
A_scene_i(t) = 1 - P_normal(c_i(t) | scene, label)
```

实现上可以用 2D histogram 或 KDE：

```text
grid_h = 24
grid_w = 43
```

低频位置风险更高。

注意：

- 不能只靠位置判断异常；
- 位置分数适合辅助判断 `vehicle in pedestrian area` 和 `restricted region`。

## 5. 语义异常分数

使用 CLIP 或类似预训练模型，不训练。

对象 crop 得到图像特征：

```text
f_img(o_i(t))
```

正常文本集合：

```text
T_normal = {
  "a person walking",
  "a person standing",
  "normal pedestrian traffic",
  "a person sitting",
  "a person carrying a bag normally"
}
```

异常文本集合：

```text
T_abnormal = {
  "a person running",
  "a person fighting",
  "a person chasing another person",
  "a bicycle in a pedestrian area",
  "a motorcycle in a pedestrian area",
  "a car in a pedestrian area",
  "a person riding a skateboard",
  "an abandoned bag",
  "an object being thrown"
}
```

分数：

```text
sim_abn = max cosine(f_img, f_text in T_abnormal)
sim_norm = max cosine(f_img, f_text in T_normal)
A_semantic_i(t) = sigmoid((sim_abn - sim_norm) / temp_clip)
```

注意：

- 对单个 crop，CLIP 不一定能识别 `running/fighting`；
- 对交通工具和明显对象类别有用；
- 更适合补充类别先验和对象外观。

## 6. 不确定性分数

异常检测中，误删异常对象很危险。所以不确定对象应倾向保留。

不确定性来源：

| 来源 | 例子 |
| --- | --- |
| 检测不确定 | confidence 低、框抖动 |
| 分类不确定 | 同一 track label 多次变化 |
| tracking 不确定 | track 很短、频繁断裂 |
| 运动不确定 | 缺帧、遮挡、速度不稳定 |
| mask 不确定 | 训练阶段伪标签覆盖率处于中间区间 |

定义：

```text
U_i(t) = max(
    low_conf_score,
    label_instability_score,
    short_track_score,
    missing_score
)
```

保留策略：

```text
如果 U_i(t) 高，即使 a_i(t) 不高，也降低删除强度。
```

## 融合方式

## Training-Free 手工融合

第一版推荐：

```text
a_i(t) =
  0.20 * A_class_i
+ 0.30 * A_motion_i(t)
+ 0.20 * A_relation_i(t)
+ 0.10 * A_scene_i(t)
+ 0.10 * A_semantic_i(t)
+ 0.10 * U_i(t)
```

再做视频内归一化：

```text
a_i_norm(t) = normalize_per_video(a_i(t))
```

frame score：

```text
S_frame(t) = max_i a_i_norm(t)
```

## 轻量训练融合

如果允许一点训练，使用 logistic regression：

```text
input_i(t) = [
  A_class,
  A_motion,
  A_relation,
  A_scene,
  A_semantic,
  U
]
```

标签来自异常 mask 覆盖：

```text
y_i(t) = 1 if cover(box_i(t), mask_t) >= 0.3
y_i(t) = 0 if frame normal and cover = 0
ignore otherwise
```

优点：

- 训练极轻；
- 权重可解释；
- 比手工权重更稳。

## 从对象分数到 Token 删除

对象 token 集合：

```text
T_i(t) = Qwen3VL_tokens_inside_box(o_i(t))
```

每个视觉 token 的风险：

```text
a_token = max_i a_i(t), for token ∈ T_i(t)
```

如果 token 不属于任何对象：

```text
a_token = A_background
```

默认：

```text
A_background = 0.05
```

## 删除等级

| 对象异常分数 `a_i(t)` | 不确定性 `U_i(t)` | Token 策略 |
| --- | --- | --- |
| `a_i >= 0.75` | 任意 | 完整保留 |
| `0.50 <= a_i < 0.75` | 任意 | 保留 70% 对象 token，优先中心和运动方向区域 |
| `0.30 <= a_i < 0.50` | `U_i >= 0.5` | 保留 50% 对象 token |
| `0.30 <= a_i < 0.50` | `U_i < 0.5` | 保留 25% 或 pooled summary |
| `a_i < 0.30` | `U_i >= 0.5` | 保留 25%，避免误删 |
| `a_i < 0.30` | `U_i < 0.5` | 删除或只保留 summary |

## 固定 Token Budget 版本

如果需要控制总 token 数，可以使用排序：

```text
sort objects by a_i(t) + lambda * U_i(t)
keep tokens until budget B
```

推荐 budget：

| 实验 | 保留比例 |
| --- | --- |
| mild pruning | 75% |
| medium pruning | 50% |
| aggressive pruning | 25% |

必须和 random pruning 对比：

```text
object-risk pruning vs random pruning at same token budget
```

## 应先做的实验

1. 只用 `A_class`。
2. `A_class + A_motion`。
3. `A_class + A_motion + A_relation`。
4. 加入 `A_scene`。
5. 加入 `A_semantic`。
6. 加入 `U` 作为保守保留项。
7. 手工融合 vs logistic regression 融合。

每一步都测：

```text
frame-level AUC
token retention ratio
abnormal object token recall
normal object token deletion ratio
random pruning 对照
```

## 判断分数是否有效

对象异常分数有效的标准：

- 异常帧中的对象 `a_i(t)` 明显高于正常帧对象；
- `S_frame(t)=max_i a_i(t)` 有非随机 AUC；
- 高分对象与 `testframemask` 异常区域有较高覆盖；
- 在同等 token budget 下，object-risk pruning 比 random pruning 保留更多异常区域 token；
- 删除低分对象 token 后，Qwen3-VL 异常判断下降小于 random pruning。

## 最推荐的第一版

先不要做复杂模型，第一版使用：

```text
A_class + A_motion + A_relation + U
```

原因：

- 不需要训练；
- 与 ShanghaiTech 异常定义最直接相关；
- 对 token 删除足够可解释；
- 后续可以自然加入 `A_semantic` 和轻量 logistic regression。

第一版对象分数：

```text
a_i(t) =
  0.25 * A_class_i
+ 0.35 * A_motion_i(t)
+ 0.25 * A_relation_i(t)
+ 0.15 * U_i(t)
```

token 保留：

```text
keep_score_i(t) = a_i(t) + 0.5 * U_i(t)
```

排序保留 top-B token，和 random pruning 做同 budget 对比。

