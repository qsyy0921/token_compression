# ShanghaiTech 对象级 Token 压缩方案

## 目标

本项目面向 ShanghaiTech Campus 数据集做对象级异常检测，并把它和对象 token 压缩结合起来：

- 检测能够解释异常事件的对象；
- 保留异常相关对象的 token；
- 压缩或丢弃背景对象、低风险正常对象的 token；
- 当单帧检测框不足以判断异常时，引入 tracking 建模对象的时间行为。

当前工作目录中的目标检测/grounding 模型是 `models/LocateAnything-3B`。运行环境固定使用：

```text
C:\Users\10495\anaconda3\envs\token_pruner_merge
```

## 数据集事实

ShanghaiTech Campus 是 ICCV 2017 论文 `A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework` 中提出的视频异常检测数据集。

标准描述如下：

- 13 个校园监控场景；
- 330 个训练视频；
- 107 个测试视频；
- 训练视频只包含正常事件；
- 测试视频包含正常事件和异常事件；
- 共 130 个异常事件；
- 测试视频提供异常区域的像素级标注。

本地目录结构和标准划分一致：

```text
data/shanghai/data/training/videos   330 videos
data/shanghai/data/testing/videos    107 videos
data/shanghai/data/testframemask     107 npy files
```

参考资料：

- 原始论文：https://openaccess.thecvf.com/content_ICCV_2017/papers/Luo_A_Revisit_of_ICCV_2017_paper.pdf
- 官方数据页：https://svip-lab.github.io/dataset/campus_dataset.html

## 数据集定义了哪些异常

ShanghaiTech 没有公开一个严格封闭的类别表，例如“类别 1 = 自行车、类别 2 = 打架”。它定义的是事件级和区域级异常：只要行为或对象出现方式违反当前校园监控场景的正常模式，就可能被标为异常。

因此本文档使用两层定义：

- 数据集事实：来自原始论文、官方数据页和使用该数据集的论文。
- 项目工作定义：为了做对象级异常检测和 token 压缩，将异常整理成可检测、可跟踪、可打分的类别。

对本项目最有用的异常分组如下：

| 异常分组 | 典型例子 | 需要的对象证据 |
| --- | --- | --- |
| 非行人物体进入步行区域 | 自行车、摩托车、汽车、滑板、滑板车 | 车辆/滑板对象，人与物体关系 |
| 人的快速或异常运动 | 奔跑、追逐、突然加速 | 人的轨迹速度和方向 |
| 人与人的交互异常 | 打架、推搡、碰撞 | 多个人的轨迹、距离和运动关系 |
| 场景规则违背 | 逆行、进入限制区域、攀爬 | 人的轨迹、场景区域，可选静态上下文 |
| 可疑人-物交互 | 搬运、推动、拖拽、遗留物体 | 人与包、箱子、推车等对象关系 |
| 群体行为异常 | 人群聚集、群体异常运动 | 多个人的轨迹和群体运动模式 |

关键结论：ShanghaiTech 的异常不只是“画面里出现了哪个对象”，很多异常是“对象在一段时间内做了什么”。

## 完整异常种类：项目工作定义

下面是本项目建议采用的完整 anomaly taxonomy。它不是官方封闭标签表，而是面向对象级检测、tracking 和 token 压缩的操作性分类。

| ID | 异常种类 | 典型表现 | 需要检测的对象 | 是否必须 tracking |
| --- | --- | --- | --- | --- |
| A01 | 非行人交通工具进入步行区域 | 自行车、摩托车、汽车、滑板车出现在校园步行场景 | person, bicycle, motorcycle, motorbike, scooter, car | 建议 |
| A02 | 滑板/滑行类异常 | 滑板、滑行、快速穿越人群 | person, skateboard, scooter | 必须 |
| A03 | 奔跑或突然加速 | 行人速度明显高于该场景正常行走速度 | person | 必须 |
| A04 | 追逐 | 一人或多人持续高速接近另一人 | person | 必须 |
| A05 | 打架/推搡/冲突 | 多人距离很近，动作剧烈，轨迹相互纠缠 | person | 必须 |
| A06 | 碰撞/跌倒 | 人与人或人与物体发生异常接触，随后姿态/轨迹突变 | person, bicycle, car, cart | 必须 |
| A07 | 攀爬/翻越 | 人靠近栏杆、围栏、台阶等并出现越界动作 | person, railing, fence, stairs | 必须 |
| A08 | 跳跃 | 人体中心高度或框变化异常，或在不该跳跃的位置跳跃 | person | 必须 |
| A09 | 逆行/异常方向 | 人或车辆运动方向违背场景主流方向 | person, bicycle, motorcycle, scooter, car | 必须 |
| A10 | 进入限制区域/越界 | 对象进入通常不应进入的区域 | person, bicycle, car, scene context | 必须 |
| A11 | 徘徊/异常停留 | 对象在局部区域停留时间异常长 | person, bag, box | 必须 |
| A12 | 推/拉/拖拽物体 | 人持续推动或拖拽推车、箱子、行李等 | person, cart, trolley, suitcase, box | 必须 |
| A13 | 搬运异常物体 | 人携带大件或异常形态物体 | person, bag, backpack, suitcase, box, package | 建议 |
| A14 | 遗留/无人看管物体 | 包、箱子等离开人体后长时间静止 | bag, backpack, suitcase, box, package | 必须 |
| A15 | 抛掷物体 | 物体短时高速运动，可能伴随人手部动作 | person, ball, bottle, bag, generic object | 必须 |
| A16 | 人群异常聚集/分散 | 多人局部密集聚集或突然散开 | person | 必须 |
| A17 | 非典型静态姿态 | 躺倒、蹲坐、异常停在路面中间 | person, bench, chair | 建议 |
| A18 | 低置信未知异常 | 不在上述类别中，但异常 mask 覆盖到运动对象 | person, unknown moving object | 必须 |

这些类别可以进一步合并成训练/评估时的 5 个大类：

| 大类 | 包含 ID | 建模重点 |
| --- | --- | --- |
| 异常对象出现 | A01, A02 | 检测非行人物体，保留其对象 token |
| 人体运动异常 | A03, A08, A09, A10, A17 | person track 的速度、方向、区域和姿态 |
| 人-人交互异常 | A04, A05, A06 | 多人轨迹关系 |
| 人-物交互异常 | A12, A13, A14, A15 | person 与物体轨迹关系 |
| 群体/未知异常 | A16, A18 | 多对象聚合和不确定性保留 |

## 相关论文依据

下面这些论文可以作为本项目设计依据。它们要么定义了数据集，要么在 ShanghaiTech 上评测过，要么支持对象中心异常检测的思路。

| 论文 | 对本项目的意义 |
| --- | --- |
| Luo et al., ICCV 2017, `A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework` | ShanghaiTech Campus 原始数据集论文，给出数据集划分和基本异常定义。 |
| Liu et al., CVPR 2018, `Future Frame Prediction for Anomaly Detection` | ShanghaiTech 经典 baseline，用未来帧预测误差做异常检测。 |
| Ionescu et al., CVPR 2019, `Object-Centric Auto-Encoders and Dummy Anomalies for Abnormal Event Detection in Video` | 明确支持 object-centric 异常检测，说明先检测对象再建模异常是合理方向。 |
| Tian et al., ICCV 2021, `Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning` | ShanghaiTech 上常用的弱监督时序特征 baseline。 |
| 近期 object-centric / track-level VAD 论文 | 支持使用对象框和对象轨迹，而不是只依赖整帧 token 或全局特征。 |

链接：

- CVPR 2018 Future Frame Prediction：https://openaccess.thecvf.com/content_cvpr_2018/html/Liu_Future_Frame_Prediction_CVPR_2018_paper.html
- CVPR 2019 Object-Centric Auto-Encoders：https://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf
- ICCV 2021 RTFM：https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf

## 需要检测哪些对象

不要一开始就使用很宽的开放词表。开放词表太大时，LocateAnything 会产生很多和异常无关的背景框，浪费对象 token，也会削弱“压缩非异常对象 token”的研究主张。

对象集合应该直接服务于 ShanghaiTech 的异常机制。

## 完整对象种类：项目工作定义

本项目采用下面的完整 object taxonomy。它覆盖 ShanghaiTech 异常检测需要的主体对象、异常物体、交互物体和少量场景上下文。

| ID | 对象种类 | LocateAnything prompt / 同义词 | 作用 | 默认 token 策略 |
| --- | --- | --- | --- | --- |
| O01 | 人 | `person`, `pedestrian`, `people` | 所有行为异常的主体 | 保留 |
| O02 | 自行车 | `bicycle`, `bike` | 非行人交通工具、骑行异常 | 保留 |
| O03 | 摩托车 | `motorcycle`, `motorbike` | 非行人交通工具 | 保留 |
| O04 | 滑板车 | `scooter`, `kick scooter`, `electric scooter` | 非行人交通工具、快速穿越 | 保留 |
| O05 | 汽车 | `car`, `vehicle` | 非行人交通工具 | 保留 |
| O06 | 滑板 | `skateboard` | 滑行类异常 | 保留 |
| O07 | 推车 | `cart`, `trolley`, `handcart`, `stroller`, `pram` | 推/拉/拖拽交互 | 条件保留 |
| O08 | 包 | `bag`, `backpack`, `handbag` | 携带、遗留、遮挡 | 条件保留 |
| O09 | 行李箱 | `suitcase`, `luggage` | 拖拽、携带、遗留 | 条件保留 |
| O10 | 箱子/包裹 | `box`, `package`, `parcel` | 搬运、推动、遗留 | 条件保留 |
| O11 | 雨伞 | `umbrella` | 外观/交互上下文，雨天或遮挡 | 中等压缩 |
| O12 | 球/可抛物 | `ball`, `bottle`, `thrown object` | 抛掷物体、短时高速小物体 | 条件保留 |
| O13 | 长椅/椅子 | `bench`, `chair` | 徘徊、坐卧、静态姿态上下文 | 默认压缩 |
| O14 | 台阶/楼梯 | `stairs`, `steps` | 攀爬、区域上下文 | 默认压缩 |
| O15 | 门/出入口 | `door`, `gate`, `entrance` | 进入/离开区域上下文 | 默认压缩 |
| O16 | 栏杆/围栏 | `railing`, `fence`, `barrier` | 攀爬、翻越、限制区域上下文 | 默认压缩 |
| O17 | 未知移动物体 | `moving object`, `unusual object` | 兜底覆盖未知异常物体 | 不确定时保留 |

主实验必须检测 O01 到 O10。O11 到 O17 用于提升召回、交互解释或场景规则 ablation，不建议在第一版中全部高保留。

## 第一优先级：必须检测

这些对象要么是异常主体，要么是最直接的异常物体证据。

| 对象 | 原因 | Token 策略 |
| --- | --- | --- |
| person | 大多数异常都是人的行为异常。 | 保留 |
| bicycle | 校园步行场景中的典型异常交通对象。 | 保留 |
| motorcycle / motorbike | 非行人交通对象。 | 保留 |
| car | 非行人交通对象。 | 保留 |
| skateboard | 和校园异常例子高度相关，常体现步行区域违规。 | 保留 |
| scooter | 和自行车、摩托车类似，可提升开放世界鲁棒性。 | 保留 |

推荐 LocateAnything prompt：

```text
person</c>bicycle</c>motorcycle</c>motorbike</c>car</c>skateboard</c>scooter
```

## 第二优先级：用于交互证据

这些对象不一定单独异常，但能解释可疑的人-物交互。

| 对象 | 原因 | Token 策略 |
| --- | --- | --- |
| bag / backpack | 携带、遗留物体、人体附近遮挡。 | 中等保留 |
| suitcase | 拖拽、携带交互。 | 中等保留 |
| box / package | 搬运、推动、遗留物体。 | 中等保留 |
| cart / trolley | 推动物体或非标准移动对象。 | 中等保留 |
| umbrella | 外观和上下文辅助，不是主要异常对象。 | 中等保留或压缩 |

推荐 prompt：

```text
bag</c>backpack</c>suitcase</c>box</c>package</c>cart</c>trolley</c>umbrella
```

## 第三优先级：可选场景上下文

这些对象不是异常主体。只有当模型显式使用场景规则时，才建议加入。

| 对象 | 原因 | Token 策略 |
| --- | --- | --- |
| bench / chair | 有助于解释徘徊、坐卧等上下文。 | 默认压缩 |
| stairs | 有助于攀爬、异常区域判断。 | 默认压缩 |
| door / gate | 有助于入口、出口、限制区域判断。 | 默认压缩 |
| railing / fence | 有助于攀爬、跨越区域判断。 | 默认压缩 |

只在上下文实验中使用：

```text
bench</c>chair</c>stairs</c>door</c>gate</c>railing</c>fence
```

## 主实验中不建议检测的对象

除非做专门 ablation，否则不要默认检测：

```text
tree, building, wall, window, road, pavement, lamp, sign, pole, sky, grass
```

这些大多是背景对象。它们会增加 token 数量，但通常不能直接解释异常标签。

## 推荐对象集合

第一版最小可行集合：

```text
person
bicycle
motorcycle
car
skateboard
scooter
bag
box
cart
```

高召回版本：

```text
person
bicycle
motorcycle
motorbike
car
skateboard
scooter
bag
backpack
suitcase
box
package
cart
trolley
umbrella
```

场景上下文 ablation：

```text
bench
chair
stairs
door
gate
railing
fence
```

## 是否需要 Tracking

需要。只要目标是对象级异常检测，而不是单纯目标检测，就应该做 tracking。

单帧 LocateAnything 检测框可以回答：

- 画面里有没有人；
- 画面里有没有自行车、汽车、滑板等可疑对象；
- 可疑对象大概在哪里。

单帧检测框无法可靠回答：

- 这个人是否在奔跑；
- 这个人是否在追逐另一个人；
- 这个人是否逆行；
- 这个对象是否正在进入限制区域；
- 这辆自行车是异常移动，还是只是静止停放；
- 这个人是否正在和包、箱子、推车发生交互。

因此 tracking 应该进入主流程。既然当前目标优先看效果而不是速度，tracking 阶段应优先追求召回率、轨迹稳定性和身份一致性。

## Tracking 方案建议

推荐两阶段流程：

1. 用 LocateAnything 在采样帧或全帧上检测推荐对象。
2. 使用 ByteTrack、BoT-SORT 或 OC-SORT 将检测框关联成轨迹。
3. 对每条对象轨迹计算时序特征。
4. 给每条对象轨迹分配异常风险分数。
5. 对低风险正常轨迹压缩 token，对高风险或不确定轨迹保留 token。

tracking 目标优先级：

| 轨迹类型 | 是否保留 | 原因 |
| --- | --- | --- |
| person 轨迹 | 是 | 大多数行为异常都依赖人轨迹。 |
| 车辆/滑板类轨迹 | 是 | 直接提供异常物体证据。 |
| bag/box/cart 轨迹 | 条件保留 | 靠近人或发生移动时有交互价值。 |
| 静态背景/上下文轨迹 | 通常不保留 | 更适合作为场景先验或压缩上下文。 |

建议计算的 tracking 特征：

- 轨迹持续时间；
- 中心点速度；
- 加速度；
- 相对当前场景正常人流方向的运动方向；
- 框面积变化；
- 人与人之间的距离；
- 人与车辆、滑板、包、箱子、推车之间的距离；
- 持续接近、重叠或交互关系；
- 如果有场景区域 mask，则判断是否进入异常区域。

## Token 压缩策略

token 压缩策略应该是风险感知的，而不是对所有对象平均压缩。

| 对象/轨迹状态 | Token 策略 |
| --- | --- |
| 速度或方向异常的 person | 保留对象 token |
| 靠近车辆、滑板、推车的 person | 保留对象 token 和关系 token |
| bicycle / motorcycle / car / skateboard / scooter | 保留或轻度压缩 |
| 靠近人或正在移动的 bag / box / cart | 保留或轻度压缩 |
| 孤立静止的 bag / box / cart | 中等压缩，只保留低维摘要 |
| 正常慢速行人轨迹 | 中等压缩 |
| 静态背景对象 | 重度压缩或丢弃 |
| 不确定检测或断裂轨迹 | 暂时保留，等待后续帧消歧 |

帧级异常分数可由对象级分数聚合：

```text
S_frame(t) = max_i S_object(track_i, t)
```

这样既能对齐 ShanghaiTech 常用的 frame-level evaluation，又能保留对象级解释能力。

## 第一批实验建议

先从测试视频开始，因为测试集有异常标注。

1. 选择一小组代表性测试场景：

```text
01_0014
01_0025
03_0031
04_0001
05_0017
08_0077
12_0142
```

2. 用 LocateAnything 跑最小对象集合。

3. 保存每帧检测结果为 JSON 或 CSV：

```text
video_id, frame_idx, object_label, x1, y1, x2, y2, score_or_confidence, prompt
```

4. 对 `person`、`bicycle`、`motorcycle`、`car`、`skateboard`、`scooter`、`bag`、`box`、`cart` 做 tracking。

5. 将对象轨迹与 `testframemask/*.npy` 的异常区域对齐。

6. 观察异常帧中高风险对象 token 是否被保留，正常对象 token 是否被压缩。

第一阶段不应直接追求最终 AUC，而应先验证：

- 异常帧对象召回率；
- 检测/轨迹框对异常区域的覆盖率；
- 正常对象和异常相关对象的 token 保留比例；
- 拥挤场景中的 tracking 稳定性。

## 工作假设

对 ShanghaiTech 来说，一个合理的对象 token 压缩方法应该优先保留 `person` 和非行人移动对象，然后通过 tracking 判断哪些对象轨迹在时间上变得异常。大部分背景对象可以激进压缩，因为它们不能直接解释数据集中的异常事件。

可以形成的论文式主张：

```text
ShanghaiTech 的异常证据是对象条件化的：同一帧中的视觉 token 是否应该被压缩，取决于其所属对象轨迹是否在行为或场景上下文中可疑。
```
