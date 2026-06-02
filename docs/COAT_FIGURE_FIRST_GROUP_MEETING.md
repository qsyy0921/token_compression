# COAT 组会汇报：上下文感知对象异常 Token 压缩

> 目标：用对象异常分数作为 token 压缩的主要参考指标，在保留异常相关对象 token 的同时压缩正常对象和无关背景 token。

## 图 1：不同语义信息如何从 token 中得到

![从 ViT 视觉 token 得到四类对象上下文语义](figures/coat_semantic_sources.png)

图 1 说明：当前对象语义、历史轨迹语义、局部背景语义和对象交互语义都来自 ViT visual tokens 与离线对象/轨迹结构。这里不是把 box 坐标表格直接喂给 LLM，而是把对象相关 token 组织成可学习的语义表示。

## 图 2：对象异常分数如何得到

![对象异常分数计算](figures/coat_score_derivation.png)

图 2 说明：对象异常分数 `s_i(t)` 由对象上下文表示和异常语义 query 共同得到。训练时使用帧级 MIL、mask-overlap 伪监督、ranking loss 和 token budget 约束，使 `s_i(t)` 表示对象 token group 对异常判断的语义贡献。

## 图 3：异常分数如何指导 token 压缩

![根据对象异常分数压缩 token 并构造 LLM 输入](figures/coat_llm_input.png)

图 3 说明：高异常分数对象保留完整 visual tokens，中低分对象压缩成 summary tokens 或删除，同时保留轨迹、场景和异常查询等辅助上下文 token。最终喂给 LLM 的是压缩后的视觉 token 序列和少量上下文 token，而不是完整原始 token 或对象框表格。

## 汇报主线

```text
1. 先说明：异常检测中 token 的价值不是视觉显著性，而是异常语义价值。
2. 再说明：对象异常语义来自当前视觉、历史轨迹、局部背景和稀疏交互。
3. 重点说明：对象异常分数 s_i(t) 是 token 压缩的主参考指标。
4. 最后说明：根据 s_i(t) 动态决定保留、汇聚或删除对象 token。
```

