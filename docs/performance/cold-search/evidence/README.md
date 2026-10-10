# 证据索引

这些是 2026-10-09—10 的实验导出。正式比较以 `formal-stages.json` 中明确列出的各 cohort 为准；`latest` 表示冻结的 v83，`previous` 表示 v28。所有耗时字段的单位为 ms，count 字段为观测次数。

| 文件 | 内容与使用边界 |
|---|---|
| `formal-stages.json` | 原始 NVMe 基线，以及 v28/v41/v45/v49/v57/v64/v70/v73/v80/v83 的逐条阶段数据。原始每场景一次，其余各三次 |
| `full-load-stages.json` | 基线两条与 v83 六条的完整阶段差分，含 17 个 Go Load 阶段与 native 14 阶段 + total |
| `formal-phase-comparison.csv` | 上述版本的阶段均值矩阵；嵌套/并行阶段不能累加回总耗时 |
| `experiment-configurations.json` | 历史 profile overrides 和 runtime 记录，可核对每轮线程/分配器/开关等实际输入；存在配置文件不等于参与某次测量 |
| `experiment-ledger.csv` | 冻结的 840 条历史记录，551 个 label；包括早期诊断、两场景筛选、控制、启动首搜与确认，不能混算总体分位数 |
| `all-measured-runs.json` | 同一批 840 条记录的原始请求字段和能关联到的阶段摘要；失败与异常仍保留 |
| `request-timelines.json` | 原始基线两条、v83 六条、v81 三 GET 异常一条；保留时间/调用结构，删除认证 headers |
| `metadata-rpc-overlap.json` | QN 元数据调用真实 RPC 区间；同名调用可能来自不同调用者，需结合源代码解释 |
| `pk-matched-control.json` | 旧 native core / 新 native core、其余相同的主键回填对照 |
| `prefetch-worker-observations.jsonl` | worker 的 TID/start_ticks 和空闲后身份；最初错误过滤产生的空列表不能算验证成功 |
| `measured-build-provenance.json` | v83 的 Go/core/storage、依赖 patch、测试日志 SHA256 与运行配置。路径只保留文件名或占位符，hash 对应原始未修改文件 |
| `historical-test-results.json` | 已执行的 21 份历史测试日志摘要、结果行、原始 hash；含明确标出的旧库负对照失败 |
| `publication-validation.json` | 本次干净提交/可移植工具的追加验证；与 v83 正式性能样本分开 |
| `publication-harness-smoke.json` | 打包后 harness 的两场景正常 lease 功能复测，全部样本另列，不并入 v83 六次均值 |
| `audits/` | 各阶段当时写下的 correctness、拒绝方案和因果审计 |

注意口径：

- `classification` 是按原始字段/标签生成的导航分类，不是物理冷态的替代证明。真正纳入正式结论的样本同时检查 Release 等待、NotLoad→Loaded、Go/native load count 和新 raw GET。
- 没有阶段摘要的早期诊断保留空项；不补造缺失指标。旧 SN `catalog_save` 的 0 表示当时无埋点，不表示没有持久化。
- `minio_waves` 是请求区间连通分量。v81 迟到 fallback 可被长请求连接成一波，因果上仍有后续发起阶段；必须看原始时间线。
- `audits/` 为历史时点记录，其中“未提交”“goal active”等句子描述文件写作当时，不描述这份已发布分支的当前状态。最终状态以两份主文档和提交历史为准。
- 原始大体积 before/after metrics、完整 OTLP/MinIO trace、服务日志、binary 和 native build 保留在实验工作区；本分支不包含这些二进制/大型 capture。hash 用于溯源，不代表相关原始文件都在 Git 中。
