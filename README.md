# token_compression

对象级异常检测与对象 token 压缩实验仓库。

当前本机数据工作目录：

- `datasets/sha_ave_nwp/`

这个目录是当前 ShanghaiTech、Avenue、NWPU 三个测试集的统一工作入口。后续数据集也应扩展到该目录下，而不是继续使用旧的顶层 `datasets/*_test` 入口。

当前设计文档：

- [ShanghaiTech 对象级 Token 压缩方案](docs/SHANGHAITECH_OBJECT_TOKEN_COMPRESSION.md)
- [对象条件化 Anomaly Query 设计](docs/ANOMALY_QUERY_DESIGN.md)
- [Object Query 之前的异常测试方案](docs/BASELINE_ANOMALY_TESTS_BEFORE_QUERY.md)
- [低算力 / Training-Free VAD 相关论文与可行路线](docs/TRAINING_FREE_LIGHTWEIGHT_VAD_LITERATURE.md)
- [基于 Qwen3-VL-8B 的对象风险 Token 删除方案](docs/QWEN3VL_OBJECT_TOKEN_PRUNING_PLAN.md)
- [对象异常分数与 Token 删除策略](docs/OBJECT_ANOMALY_SCORE_FOR_TOKEN_PRUNING.md)
- [没有对象级 Label 时的监督构造策略](docs/NO_OBJECT_LABEL_STRATEGY.md)
- [LocateAnything 对 ShanghaiTech Label 的支持矩阵](docs/LOCATEANYTHING_LABEL_SUPPORT_MATRIX.md)
