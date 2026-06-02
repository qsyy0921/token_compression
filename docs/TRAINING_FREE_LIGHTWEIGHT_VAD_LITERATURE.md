# 低算力 / Training-Free VAD 相关论文与可行路线

## 当前约束

本项目算力只能支持：

- training-free；
- 轻量训练；
- 简单特征模型；
- 小规模 ablation；
- 离线检测和离线特征提取。

因此不应优先做大规模端到端视频模型训练，也不应把最终方法设计成依赖大模型在线推理。更合适的路线是：

```text
离线对象标注 + tracking + 预训练特征 + training-free / 轻量异常模型
```

## 论文路线总览

| 路线 | 代表论文 | 是否适合当前项目 |
| --- | --- | --- |
| Training-free VLM/LLM 推理 | LAVAD, MM-VAD, RAG4VAD, VADTree, EventVAD | 适合作为思想参考和弱 baseline，但在线调用大模型可能慢。 |
| CLIP 文本描述相似度 | Unsupervised VAD Based on Similarity with Predefined Text Descriptions | 很适合当前项目，可直接复现对象级 CLIP/text baseline。 |
| 对象中心轻量建模 | Object-Centric Auto-Encoders, object-centric adversarial learning, memory-guided normality reconstruction | 和本项目最接近，但部分方法仍需要训练 AE/GAN。可简化成预训练特征 + SVM/IForest。 |
| Training-less motion/statistics | Adaptive training-less framework for anomaly detection in crowd scenes | 适合做轨迹/光流规则 baseline。 |
| 轻量 one-class / weak training | DSVDD, One-Class SVM, dummy anomaly SVM | 可在对象特征上做，不需要大模型训练。 |

## Training-Free / VLM 相关论文

### LAVAD: Harnessing Large Language Models for Training-free Video Anomaly Detection

链接：

- arXiv: https://arxiv.org/abs/2404.01014
- CVPR OpenAccess: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf

核心思想：

- 使用 VLM 给每帧生成 caption；
- 使用 LLM 对 caption 序列做异常判断；
- 不训练模型；
- 用跨模态相似度清理 noisy captions，并平滑 anomaly score。

对本项目的启发：

- 可以把对象轨迹转成结构化文本，而不是整帧 caption；
- 例如：

```text
frame 120: person track 3 moves fast to the left; bicycle appears near person track 5.
```

- 再用规则或轻量 LLM prompt 输出 anomaly score。

局限：

- 依赖 caption/LLM，速度慢；
- 容易变成外部大模型推理，不适合作为最终 token 压缩方法；
- 可以作为 training-free reference baseline。

### Unsupervised VAD Based on Similarity with Predefined Text Descriptions

链接：

- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10385872/
- MDPI: https://www.mdpi.com/1424-8220/23/14/6256

核心思想：

- 用 ChatGPT/人工定义 normal 和 abnormal 文本描述；
- 用 CLIP 计算图像/对象 crop 与文本描述的相似度；
- 对 ShanghaiTech，因为对象很小，论文使用 object detector 先裁剪对象；
- 论文明确指出 ShanghaiTech 是 walking-only zone，因此 `cars`, `bicycles`, `motorbikes`, `running` 等在该数据集中应被视为 abnormal；
- 还把 COCO 中 transportation / large animals 等类别划为 abnormal，其他类别划为 normal。

对本项目的启发：

- 非常适合当前算力；
- 我们可以用 LocateAnything 离线生成对象 crop；
- 用 CLIP image encoder 提取 crop 特征；
- 用文本描述做 training-free object anomaly score。

推荐实现：

```text
object_crop -> CLIP image feature
normal_texts / abnormal_texts -> CLIP text features
score = max_sim(abnormal_texts) - max_sim(normal_texts)
frame_score = max_object_score
```

推荐 abnormal texts：

```text
a person running
a person fighting
a person chasing another person
a bicycle in a pedestrian area
a motorcycle in a pedestrian area
a car in a pedestrian area
a person riding a skateboard
a person pushing a cart
an abandoned bag
an object being thrown
```

推荐 normal texts：

```text
a person walking
people walking normally
a person standing
a person sitting on a bench
a person carrying a bag normally
normal pedestrian traffic
```

### MM-VAD: Geometry-Aware Semantic Reasoning for Training Free VAD

链接：

- CatalyzeX summary: https://www.catalyzex.com/paper/geometry-aware-semantic-reasoning-for

核心思想：

- training-free；
- 使用 frozen foundation models；
- 将场景/事件表示放到更适合层次语义的空间；
- 用 test-time prompt / confidence-sparsity 目标做自适应校准；
- 报告了 ShanghaiTech 上较高的 AUC。

对本项目的启发：

- 可以借鉴“test-time calibration”，但不要照搬复杂超曲空间和大 LLM；
- 简化为：

```text
每个视频内 score normalization + sparsity prior + temporal smoothing
```

也就是假设异常稀疏，用视频内排序和稀疏正则校准对象分数。

### RAG4VAD

链接：

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0020025526005207

核心思想：

- training-free；
- 先提取 scene-aware structured visual representations；
- 检索正常样例；
- 再做 anomaly scoring 和语言解释。

对本项目的启发：

- 很适合对象轨迹检索 baseline；
- 从训练集正常视频建立正常对象轨迹库；
- 测试对象轨迹检索最近正常轨迹，距离越大越异常。

简化实现：

```text
normal track feature bank from training videos
test track feature
score = distance_to_k_nearest_normal_tracks
```

这比训练模型更适合当前算力。

### VADTree / EventVAD

链接：

- VADTree: https://arxiv.org/abs/2510.22693
- EventVAD: https://arxiv.org/abs/2504.13092

核心思想：

- training-free；
- 不用固定长度窗口；
- 根据事件边界或层次结构进行采样；
- 再用 VLM/LLM 做异常推理。

对本项目的启发：

- 我们可以不对每帧都跑重模型；
- 先用 tracking 找到事件片段：

```text
track appears
track disappears
speed changes
object enters scene
person-object distance changes
```

- 再对这些片段做异常打分。

## 对象中心 VAD 相关论文

### Object-Centric Auto-Encoders and Dummy Anomalies

链接：

- CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/papers/Ionescu_Object-Centric_Auto-Encoders_and_Dummy_Anomalies_for_Abnormal_Event_Detection_in_CVPR_2019_paper.pdf
- arXiv: https://arxiv.org/abs/1812.04960

核心思想：

- 先用 detector 检测对象；
- 对 object crops 学 appearance/motion autoencoder；
- 对正常样本聚类；
- 用 one-vs-rest SVM / dummy anomalies 方式判断异常；
- 在 ShanghaiTech 上 object-centric 思路有明显收益。

对本项目的启发：

- 非常支持我们的对象级路线；
- 但训练多个 autoencoder 仍有成本；
- 可简化为：

```text
object crop -> frozen CLIP/DINO feature
normal feature clustering
one-class SVM / kNN distance / Isolation Forest
```

### Local Anomaly Detection in Videos using Object-Centric Adversarial Learning

链接：

- arXiv: https://arxiv.org/abs/2011.06722

核心思想：

- 只需要 object regions；
- 学当前 appearance 与过去 gradient/motion 之间的对应；
- reconstruction/adversarial score 作为 region-level anomaly score；
- 在 ShanghaiTech 等数据集上测试。

对本项目的启发：

- object region 是合理最小单元；
- 但 adversarial training 不适合当前算力；
- 可用简单 motion feature 代替 gradient generation。

### Object-centric and memory-guided normality reconstruction

链接：

- arXiv: https://arxiv.org/abs/2203.03677

核心思想：

- 只学习 normal patterns；
- 使用 object-level appearance/motion features；
- 用 memory prototypes 和 cosine distance 估计异常。

对本项目的启发：

- 适合低算力简化：

```text
训练集正常 object/track features -> KMeans prototypes
测试 object/track feature -> nearest prototype cosine distance
```

无需训练深网络。

## Training-Less / 统计运动方法

### Adaptive training-less framework for anomaly detection in crowd scenes

链接：

- ScienceDirect: https://www.sciencedirect.com/science/article/pii/S0925231220311668

核心思想：

- training-less；
- 多对象检测和关联；
- 局部 motion descriptor；
- 使用 Earth Mover's Distance 判断异常；
- 在 UCSD、UMN、Avenue、ShanghaiTech 上测试。

对本项目的启发：

- 可以先做完全不训练的 motion/trajectory baseline；
- 不必训练视频模型；
- 对 running、sudden motion、wrong direction 很有用。

简化实现：

```text
track velocity histogram
scene normal velocity histogram
score = EMD(test_hist, normal_hist)
```

## 最适合当前项目的方案

## 方案 A：完全 Training-Free 对象文本相似度

输入：

- LocateAnything 离线对象框；
- object crop；
- CLIP image encoder；
- normal/abnormal text bank。

输出：

- object anomaly score；
- frame anomaly score。

优点：

- 不需要训练；
- 可解释；
- 和 LocateAnything 离线标注天然兼容；
- 可以快速跑 ShanghaiTech 测试集。

缺点：

- 对 running/fighting 这种动作不一定强；
- 小目标 crop 质量影响大；
- CLIP 对监控视角可能弱。

## 方案 B：Training-Free 正常轨迹检索

输入：

- training videos 的正常 tracks；
- testing videos 的 tracks；
- track feature。

track feature：

```text
label
box center
box size
velocity
acceleration
direction
duration
nearest object distance
```

分数：

```text
score = kNN_distance(test_track, normal_track_bank)
```

优点：

- 不训练；
- 适合 ShanghaiTech training set 只有正常视频的设定；
- 对 running、wrong direction、vehicle movement 更有用。

缺点：

- tracking 质量决定上限；
- 需要 scene-specific normalization。

## 方案 C：轻量 One-Class 模型

输入同方案 B，但训练一个轻量模型：

- Isolation Forest；
- One-Class SVM；
- KMeans prototype；
- Logistic Regression with pseudo labels；
- 小 MLP。

推荐顺序：

```text
KMeans/kNN -> Isolation Forest -> One-Class SVM -> Logistic Regression -> small MLP
```

优点：

- 训练成本低；
- 结果可解释；
- 方便 ablation。

缺点：

- 依赖手工特征；
- 不一定能处理复杂交互。

## 方案 D：对象级 CLIP + 轨迹融合

这是当前最推荐的低算力方案：

```text
object crop CLIP score
        +
track abnormality score
        +
object class prior
        ↓
object anomaly score
```

融合：

```text
s_i(t) = w1 * s_clip_i(t) + w2 * s_track_i(t) + w3 * class_prior_i
S_frame(t) = max_i s_i(t)
```

权重可以先手动设定，不训练：

```text
w1 = 0.4
w2 = 0.4
w3 = 0.2
```

后续如果允许轻量训练，可以在 validation split 上用 logistic regression 学权重。

## 推荐第一批实验

优先级：

1. `oracle coverage`：对象框是否覆盖异常 mask。
2. `class prior baseline`：交通工具/滑板出现即异常。
3. `track speed baseline`：速度 z-score。
4. `CLIP text similarity baseline`：对象 crop 与异常文本相似度。
5. `normal track retrieval baseline`：测试轨迹到训练正常轨迹库的距离。
6. `score fusion baseline`：CLIP + track + class prior。

评价：

- frame-level AUC；
- abnormal object coverage；
- token retention ratio；
- 正常帧误报来源；
- 哪些异常类型失败。

## 对 Token 压缩的意义

这些低算力方法不是最终 token 压缩模型，但可以提供三个关键产物：

- 对象级伪异常分数；
- 对象级 token 保留/压缩标签；
- 失败案例分析，决定后续是否需要 object query。

推荐先形成：

```text
retain_label_i(t) = 1 if object_score_i(t) is high or uncertain
retain_label_i(t) = 0 if object_score_i(t) is low and normal
```

然后再用这些标签训练轻量 token retention head。

## 当前建议

最适合当前算力的路线是：

```text
LocateAnything 离线对象标注
    -> tracking
    -> CLIP object-text similarity
    -> normal track retrieval / KMeans prototypes
    -> 手工或 logistic regression 融合
    -> 生成对象级 retain/compress 伪标签
```

这条路线训练成本最低，也最贴合 ShanghaiTech 的对象中心异常特点。

