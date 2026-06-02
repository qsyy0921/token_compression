# LocateAnything 对 ShanghaiTech Label 的支持矩阵

## 结论

LocateAnything 可以作为 ShanghaiTech 的离线对象标注器，但它不能直接检测所有异常 label。

它适合检测：

```text
可见实体对象
```

例如 `person`、`bicycle`、`car`、`skateboard`、`bag`、`cart` 等。

它不适合直接检测：

```text
动作、关系、时序事件、场景规则异常
```

例如 `running`、`chasing`、`fighting`、`loitering`、`wrong direction` 等。这些需要由对象框、tracking、速度/方向/关系特征来推断。

## 模型说明依据

根据本地 `models/LocateAnything-3B/README.md`，LocateAnything 的定位是：

- 开放词表目标检测；
- 密集多目标检测；
- referring expression grounding；
- 自动数据标注；
- GUI grounding；
- OCR/layout localization；
- point-based localization。

模型输入是图片和自然语言 prompt，输出是结构化 box 或 point：

```text
<ref>label</ref><box><x1><y1><x2><y2></box>
```

因此，它可以做“这个对象在哪里”，不能单独做“这个行为是否异常”。

## 支持等级定义

| 等级 | 含义 |
| --- | --- |
| S | 强支持，适合直接作为检测 prompt |
| A | 可支持，但建议加入同义词或做召回验证 |
| B | 可尝试，但小目标/监控视角/语义泛化会影响稳定性 |
| C | 不建议作为主检测 prompt，只能作为兜底 |
| N | 不是对象检测 label，不能直接交给 LocateAnything 判断 |

## 实体对象 Label 支持矩阵

| Label | 支持等级 | 推荐 prompt | 说明 |
| --- | --- | --- | --- |
| person | S | `person`, `pedestrian`, `people` | 必须检测。smoke 中模型更倾向输出 `pedestrian`。 |
| pedestrian | S | `pedestrian`, `person` | 与 person 合并为 person track。 |
| bicycle | S | `bicycle`, `bike` | 交通工具异常核心对象。 |
| bike | S | `bike`, `bicycle` | 同 bicycle。 |
| motorcycle | S | `motorcycle`, `motorbike` | 交通工具异常核心对象。 |
| motorbike | S | `motorbike`, `motorcycle` | 同 motorcycle。 |
| car | S | `car`, `vehicle` | 交通工具异常核心对象。 |
| vehicle | A | `vehicle`, `car`, `truck` | 泛化 prompt，建议和具体类别一起用。 |
| truck | A | `truck`, `large vehicle`, `vehicle` | ShanghaiTech 中不一定常见，但模型应能定位。 |
| skateboard | A | `skateboard` | 支持开放词表检测，但小目标时要验证召回。 |
| scooter | A | `scooter`, `electric scooter`, `kick scooter` | 可尝试，建议同义词扩展。 |
| cart | A | `cart`, `trolley`, `handcart` | 人-物交互对象，外观差异大。 |
| trolley | A | `trolley`, `cart`, `handcart` | 与 cart 合并。 |
| stroller / pram | A | `stroller`, `pram`, `baby stroller` | 可尝试，常与 cart 类合并。 |
| bag | S | `bag`, `backpack`, `handbag` | 可见实体对象，但小目标/遮挡会漏。 |
| backpack | S | `backpack`, `bag` | 与 bag 合并。 |
| handbag | A | `handbag`, `bag` | 与 bag 合并。 |
| suitcase | S | `suitcase`, `luggage` | 人-物交互/遗留物体相关。 |
| luggage | A | `luggage`, `suitcase` | 与 suitcase 合并。 |
| box | A | `box`, `package`, `parcel` | 可检测，但小包裹可能漏。 |
| package | A | `package`, `box`, `parcel` | 与 box 合并。 |
| umbrella | A | `umbrella` | 可检测，但不是核心异常对象。 |
| ball | B | `ball` | 小目标，监控视角下召回不稳定。 |
| bottle | B | `bottle` | 小目标，召回不稳定。 |
| bench | A | `bench` | 场景上下文对象，默认不作为高风险对象。 |
| chair | A | `chair` | 场景上下文对象。 |
| stairs | A | `stairs`, `steps` | 场景上下文对象。 |
| door | A | `door`, `entrance` | 场景上下文对象。 |
| gate | A | `gate`, `entrance` | 场景上下文对象。 |
| railing | B | `railing`, `fence`, `barrier` | 可尝试，细长结构可能不稳定。 |
| fence | A | `fence`, `railing`, `barrier` | 场景上下文对象。 |
| unknown moving object | C | `moving object` | prompt 太泛，容易不稳定，不建议主实验使用。 |
| unusual object | C | 不建议 | “unusual” 是异常语义，不是稳定视觉类别。 |
| thrown object | C | `ball`, `bottle`, `bag` 更好 | 应检测具体物体，再用 tracking/光流判断 throwing。 |

## 动作/事件 Label 支持矩阵

下面这些不是实体对象，不应直接作为 LocateAnything 的 object detection prompt。

| Label | 支持等级 | 原因 | 应如何得到 |
| --- | --- | --- | --- |
| running | N | 动作状态，不是物体 | person tracking 的速度/加速度 |
| sudden motion | N | 运动属性 | track speed / acceleration |
| fast-moving | N | 运动属性 | track speed |
| chasing | N | 多人关系事件 | 多 person tracks 的接近方向和速度 |
| brawling | N | 人-人交互事件 | person tracks + 局部视觉/姿态 |
| fighting | N | 人-人交互事件 | person tracks + 局部视觉/姿态 |
| quarrel | N | 语义事件，单帧检测不可靠 | 不建议第一阶段处理 |
| pushing | N | 交互事件 | person-person 或 person-object relation |
| collision | N | 接触/轨迹突变事件 | tracking overlap + velocity change |
| falling down | N | 姿态/时序事件 | person track + pose/box aspect change |
| stealing | N | 长时语义事件 | person-bag/suitcase relation |
| robbing | N | 长时语义事件 | person-person/person-object relation |
| throwing objects | N | 动作 + 小物体运动 | 具体物体检测 + tracking/光流 |
| loitering | N | 长时间停留行为 | person track duration + location |
| wrong direction | N | 场景规则 + 方向 | track direction vs normal direction |
| restricted-zone entry | N | 场景区域规则 | track position vs ROI/normal heatmap |
| climbing | N | 姿态/区域事件 | person track + railing/fence/stairs context |
| jumping | N | 姿态/运动事件 | person track + vertical/box change |

## 推荐标注 Prompt 分组

不要一次性把所有 label 都塞进一个 prompt。推荐分组标注，后续合并。

主体和交通工具：

```text
person</c>pedestrian</c>bicycle</c>bike</c>motorcycle</c>motorbike</c>scooter</c>car</c>vehicle</c>truck</c>skateboard
```

人-物交互对象：

```text
bag</c>backpack</c>handbag</c>suitcase</c>luggage</c>box</c>package</c>cart</c>trolley</c>stroller
```

场景上下文：

```text
bench</c>chair</c>stairs</c>door</c>gate</c>railing</c>fence
```

第一版全测试集标注可以先只跑前两组；如果后续要分析 climbing、restricted-zone entry、loitering，再补场景上下文组。

## 当前 Smoke 观察

在 `01_0014` 前几帧的 smoke 中，LocateAnything 返回了：

```text
<ref>pedestrian</ref><box>...</box>
```

而 `person` 返回 `<box>None</box>`。这说明：

- 同义词非常重要；
- `pedestrian` 应归一化到 `person`；
- 后处理必须做 label canonicalization；
- 不能只用单个 `person` prompt。

当前 tracking 脚本已经把：

```text
pedestrian -> person
```

合并为统一轨迹 label。

## 结论

LocateAnything 可以覆盖我们需要的大部分“对象种类”，但不能覆盖“完整异常种类”。

可直接用于离线标注的对象：

```text
person/pedestrian
bicycle/bike
motorcycle/motorbike
scooter
car/vehicle/truck
skateboard
bag/backpack/handbag
suitcase/luggage
box/package
cart/trolley/stroller
umbrella
bench/chair/stairs/door/gate/fence/railing
```

不能直接标注、必须由 tracking/规则/轻量模型推断的异常：

```text
running
chasing
fighting
pushing
collision
falling
stealing
loitering
wrong direction
restricted-zone entry
climbing
jumping
throwing objects
```

因此，LocateAnything 在本项目中的正确角色是：

```text
离线实体对象标注器，而不是异常事件分类器。
```

