# 多对象视频中 ID50 Running 识别的 Token 压缩实验验证报告

## 1. 实验目的

本实验用于验证一个问题动机：

> 多对象、长视频会稀释目标异常对象的关键证据，导致大模型在完整视频上判断失败；如果在 token 层保留目标对象的关键运动证据，同时压缩其他对象、背景和非关键时间段，就可能让模型重新识别出异常行为。

本案例关注 ShanghaiTech 测试视频 `08_0044` 中 tracking ID `50` 的行人是否在 running。

这个实验不是大规模 benchmark，而是一个用于论文或汇报中说明问题动机的案例验证。

## 2. 中文可视化总览

![中文实验总览](figures/id50_720p_token_compression/id50_chinese_summary.jpg)

这张图概括了完整对照：完整视频 baseline 输出 `walking`，仅空间 ROI 压缩仍为 `walking`，而运动聚焦 token 压缩将同一模型的输出翻转为 `running`。

## 3. 实验对象与设置

视频：

```text
datasets/sha_ave_nwp/shanghaitech_test/tracking/scheme1_dataset_specific/visualizations/08_0044/08_0044_tracking.mp4
```

模型：

```text
Qwen3-VL-8B-Instruct
```

本地权重路径：

```text
/home/expand_disk/model_repository/Models/Qwen/Qwen3-VL-8B-Instruct
```

环境：

```text
lavida
```

输入分辨率：

```text
728 x 1288
```

说明：原视频分辨率为 `856 x 480`。为了增强远处行人的步态细节，实验使用 720p 级别输入，即 Qwen 处理后的尺寸为 `728 x 1288`。

Qwen3-VL 在完整视频 720p 输入下的视觉网格：

```text
raw grid: [13, 46, 80]
LLM grid: [13, 23, 40]
visual tokens: 11960
```

其中 `13` 是视频时间块数量，`23 x 40` 是每个时间块送入 LLM 的空间视觉 token 网格。

## 4. Running 判断规则

本实验中的 `running` 不是根据 `0.98` 判断出来的，而是根据预先写入 prompt 和报告的步态规则判断。规则如下：

如果 ID50 在连续关键帧中同时满足以下若干视觉证据，则判断为 `running` 或 `jogging`：

1. **快速步幅变化**：短时间内腿部前后跨度明显变大，步幅比普通 walking 更快、更大。
2. **快速位移**：ID50 在 frames `136-166` 中相对自身人体尺度发生明显位移，速度高于普通行走段。
3. **手臂强摆动**：上肢摆动幅度明显，和快速步态同步。
4. **离地或近似离地瞬间**：部分帧中双脚支撑不稳定，出现跑步/小跑常见的离地或近似离地姿态。
5. **与负对照区间区分**：后段 frames `220-260` 中 ID50 步态更平稳、步幅更小、没有明显摆臂和离地证据，因此判断为 walking。

也就是说，判断依据是：

```text
关键帧段中的步态证据 + 与慢速负对照段的差异
```

而不是：

```text
模型输出的自报告 confidence 数字
```

## 5. Prompt

所有实验使用同一个 prompt：

```text
Focus on tracking ID 50 in the video. Classify the motion of tracking ID 50 as one of: running, jogging, fast walking, walking, or uncertain. Running/jogging means fast gait with rapid stride, strong arm swing, or airborne/near-airborne steps. Ignore other people unless they help compare speed. Return the label and concise visual evidence.
```

这样可以保证对照实验只改变视觉 token 序列，不改变文本提示。

## 6. Token 压缩方法

本实验没有修改原视频像素，也没有裁剪视频画面。压缩发生在 Qwen3-VL 视觉编码器输出之后、送入语言模型之前。

压缩策略分为三类：

1. **目标对象保留**  
   对 tracking ID `50` 在关键运动窗口中的 token 不做平均池化，尽量保留其腿部、脚部、身体边缘和手臂摆动等细节。

2. **其他对象删除**  
   对非目标人的 token 直接删除，降低多对象干扰。

3. **背景与非关键时间压缩**  
   背景 token 做局部平均池化；非关键时间块被压缩为少量 summary token，减少长视频中正常行走片段对最终判断的稀释。

机制图如下：

![token 压缩机制](figures/id50_720p_token_compression/id50_token_compression_mechanism.jpg)

图中绿色表示保留的 ID50 / 关键运动 token，红色表示删除的其他人 token，蓝色表示背景平均池化 token，灰色表示非关键时间块压缩。

## 7. 实验结果

所有行都使用完整视频、同样 720p 输入和同一个 prompt。唯一变化是视觉 token 序列。

| 实验设置 | 视觉 token 数 | 输入 token 数 | Qwen3-VL 输出 | 判断依据 |
| --- | ---: | ---: | --- |
| 完整视频 baseline，不做 token 压缩 | `11960 -> 11960` | `12156 -> 12156` | `walking` | 模型没有稳定捕捉到快速步幅、强摆臂和离地证据 |
| 仅空间 ROI-aware 压缩 | `11960 -> 3178` | `12156 -> 3374` | `walking` | 目标空间区域保留，但完整时间轴中的 walking 片段仍然稀释判断 |
| 运动时段聚焦压缩，保留 frames `136-166` | `11960 -> 353` | `12156 -> 549` | `running` | 关键窗口中可见快速步幅、明显摆臂和离地/近似离地姿态 |
| 负对照：聚焦后段慢速 frames `220-260` | `11960 -> 497` | `12156 -> 693` | `walking` | 后段步态平稳，缺少 running 所需的快速步幅和离地证据 |

## 8. 正例可视化：保留关键 running 时间窗

frames `136-166` 是 ID50 出现快速步态的关键窗口。对这个窗口保留 ID50 token，并压缩其他对象、背景和非关键时间段后，模型输出从 baseline 的 `walking` 翻转为 `running`。

该判断主要依赖以下视觉证据：

- ID50 在连续帧中腿部跨度变化大，步幅明显快于普通行走；
- 上肢摆动更明显，和快速步态同步；
- 若干帧中出现离地或近似离地的跑步姿态；
- 与后段负对照窗口相比，位移速度和步态幅度更强。

![正例：运动聚焦 token 压缩](figures/id50_720p_token_compression/id50_running_focus_sheet.jpg)

对应视频：

```text
figures/id50_720p_token_compression/id50_running_focus_overlay.mp4
```

图中：

- 绿色：ID50 token 保留
- 红色：其他人 token 删除
- 蓝色：背景 token 平均池化
- 黄色框：ID50 扩张 ROI 区域

## 9. 负对照可视化：聚焦慢速时间窗不会诱导 running

为了排除“只要强压缩就会让模型胡乱输出 running”的可能性，我们把同样的 token 压缩机制应用到后段慢速窗口 frames `220-260`。

结果仍然是：

```text
walking
```

![负对照：慢速时间窗](figures/id50_720p_token_compression/id50_walking_negative_control_sheet.jpg)

对应视频：

```text
figures/id50_720p_token_compression/id50_walking_negative_control_overlay.mp4
```

这个负对照说明：正例中的 `running` 不是压缩本身造成的伪影，而是因为压缩后模型看到了并利用了 ID50 在关键时间窗中的运动证据。

## 10. 时间线可视化

![实验时间线](figures/id50_720p_token_compression/id50_720p_experiment_timeline.jpg)

绿色窗口是正例 running focus 区间 `136-166`，黄色窗口是后段 walking 负对照区间 `220-260`。

## 11. 可以证明什么

这个案例可以支持以下结论：

1. **完整视频 baseline 会被多对象和长时间上下文稀释。**  
   即使提升到 720p，完整视频 baseline 仍然输出 `walking`。

2. **仅做空间 ROI 压缩不够。**  
   只压缩其他人和背景，但保留完整时间轴，模型仍然输出 `walking`。这说明长视频中的正常时间段也会稀释关键 running 证据。

3. **运动感知 token 压缩可以改变模型判断。**  
   当保留 ID50 在 frames `136-166` 的关键运动 token，并强压缩非关键 token 后，模型输出 `running`。判断依据是关键窗口中的快速步幅、明显摆臂和离地/近似离地证据。

4. **负对照排除了简单压缩伪影。**  
   同样压缩机制聚焦后段慢速窗口时，模型输出 `walking`，说明不是“压缩必然导致 running”。

因此，更准确的表述是：

> 在该多对象长视频中，Qwen3-VL baseline 无法在完整 720p 输入上识别 ID50 的 running。通过保留目标对象关键运动时段 token，并压缩其他对象、背景和非关键时间 token，可以让同一模型在同一完整视频上输出 running。

## 12. 不能过度声称什么

这个实验目前不能直接证明：

- 方法在所有视频上都有效；
- 方法已经超过 VAD SOTA；
- 任意 token 压缩都会提升异常检测；
- 只靠空间压缩就一定能解决问题。

它更适合作为 problem motivation 或 case study，用于说明为什么需要目标感知、运动感知的 token 压缩。

## 13. 复现实验

注意：当前 GitHub 仓库按要求只保留报告、可视化和脚本，不再上传数据集、baseline 工程或模型权重。复现需要本地具备：

- Qwen3-VL-8B-Instruct 权重；
- ShanghaiTech `08_0044` 视频、帧图和 tracking jsonl；
- `lavida` 环境；
- `qwen_vl_utils`，可以来自本地 LAVIDA，也可以单独安装。

生成可视化：

```bash
cd /home/expand_disk/code_repository/mfl/token_compression
/home/expand_disk/code_repository/miniconda3/envs/lavida/bin/python \
  scripts/visualize_id50_token_compression.py
```

运行完整视频 baseline：

```bash
/home/expand_disk/code_repository/miniconda3/envs/lavida/bin/python \
  scripts/qwen3vl_id50_token_compress.py \
  --mode baseline \
  --nframes 312 \
  --height 728 \
  --width 1288
```

运行正例 motion-focus token 压缩：

```bash
/home/expand_disk/code_repository/miniconda3/envs/lavida/bin/python \
  scripts/qwen3vl_id50_token_compress.py \
  --mode motion_focus \
  --focus-start 136 \
  --focus-end 166 \
  --target-expand 3.0 \
  --other-expand 1.0 \
  --nframes 312 \
  --height 728 \
  --width 1288
```

运行后段慢速负对照：

```bash
/home/expand_disk/code_repository/miniconda3/envs/lavida/bin/python \
  scripts/qwen3vl_id50_token_compress.py \
  --mode motion_focus \
  --focus-start 220 \
  --focus-end 260 \
  --target-expand 3.0 \
  --other-expand 1.0 \
  --nframes 312 \
  --height 728 \
  --width 1288
```

## 14. 文件说明

```text
README.md
REPORT_CN.md
figures/id50_720p_token_compression/
scripts/qwen3vl_id50_token_compress.py
scripts/visualize_id50_token_compression.py
```

其中 `figures/id50_720p_token_compression/` 包含所有报告可视化图和 overlay 视频。
