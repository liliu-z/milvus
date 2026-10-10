# 从源码编译到六次冷搜低于 10 ms：实验与工作过程

这份文档记录可核对的工作过程：输入如何固定，harness 如何搭建，做过哪些候选、怎么比较、哪些结果推翻了原来的判断，以及最后如何整理交付。它不把事后推测写成实验已经证明的因果。性能结论和每项实现细节见 [完整报告](REPORT.zh-CN.md)。

## 1. 先把任务纠偏和测量对象固定下来

最初收到的地址是 `codex/master-ack-callback-registration`。后来明确改为同一 sunby 仓库的 `codex/load-1m-segments-pr-stack-rebased-qv-work`。我在独立 worktree 重新编译和部署，用 `2dd24a2189a11c7f61652eb5a4b1c17921348325` 作基线，保留旧目录，不拿旧分支数字补新分支结果。

用户后续把输出要求收窄成两个场景：只返回 ID、返回 ID + vector。此前做过的标量输出诊断留在历史记录里，但最终对比只用这两个场景。表仍保留五个标量，这符合最初的数据准备要求；最终没有第三个“返回标量”场景来稀释比较。

“目标 10 ms”也需要有操作性口径：测的是服务已启动、collection Release 后，下一次 Search 自己触发 Load 并返回的 SDK 时间。进程启动首搜、热搜、显式 Load、short-lease screening、normal-lease confirmation 分开记录。最终达成的是固定六次 Release 冷重载样本全部 <10 ms，没有把它改写成 P99 保证。

## 2. 构建和部署中实际遇到的问题

这不是直接拉镜像跑 SDK。需要编译该分支的 Go、C++、Rust，以及匹配的 storage/Knowhere 依赖。

| 问题 | 采取的动作 | 留下的约束 |
|---|---|---|
| OpenTelemetry C++ ABI 不匹配 | Conan profile 显式 C++20，WITH_STL=CXX17，与 Milvus/Knowhere 对齐；把额外配置纳入 package ID | 不能“库名一样就继续链接”，也不能复用错误 ABI 的旧缓存 |
| Rust/Corrosion 的 CC/linker 环境变化让缓存失效 | 固定 native 编译环境；归档局部 Corrosion workaround；native 增量构建复用已产出的 Rust archive | 这是构建缓存措施，不是查询优化 |
| native 和 Go 头文件/库不配套 | 新 API 的 header 与 core/storage 一起安装；每个候选记录 native SHA256 | 单独换 Go binary 不能证明新 C++ 已生效 |
| scoped prefetch 同时改了 storage | 重编并部署 core 和 storage 两个 DSO，harness 同时快照 | 只换 core 会留下旧实现/ABI；这是实际复现风险 |
| 根盘接近满 | 大构建、历史产物放 NVMe；同时指定 TMPDIR、GOTMPDIR；迁移旧产物后核对 hash | 只设 GOTMPDIR 仍可能让 native linker 在 /tmp 失败 |
| 本地 MinIO 初始化 | 显式 useIAM=false，补 glog.conf，确认 bucket/root/endpoint | 配置错误不能算性能问题 |
| Woodpecker 磁盘水位 | 用专用低使用率 ext4 loop 放 WAL，后续开启 loop direct IO | 没有通过关闭 fsync 绕过持久化 |
| 服务关闭较慢 | 等旧进程真正退出再启动，保留失败启动日志 | 一次过早启动失败不纳入成功测量；正式轮次确保只有一个健康实例 |
| 本机 native check 找不到 ld.lld | 使用正确 linker PATH 重新编译；失败记录保留 | 构建助手的依赖也要验证，不能只提供脚本文件 |

后面候选越来越多，我把“源码状态”“实际运行 binary”“两个实际映射的 native 库”“配置”和“结果标签”绑成一套 provenance。v57 是一个具体教训：工作树当时已经开始改 parallel JSON stats，但运行 binary 仍是 v55 的代码；必须按 binary/hash 报告，不能看当前 git diff 就断言它参与了 v57。

最终打包时又做了一次反向检查：从各历史冻结 patch 生成临时 Git tree，按优化边界重建提交；最终 production tree 与 v83 冻结 tree 逐文件一致。这样整理 commit 不会顺手引入未经测量的新实现。

## 3. Harness 为什么这样搭

### 3.1 输入、标签、控制和正确性

数据由固定 seed 生成；向量保存 float32 并按行归一化，query 固定取第 123 行。NumPy 全量 cosine 作为独立参考，不拿 Milvus 自己的前一次结果当正确答案。ID、距离、返回向量都在计时结束后检查。

每次候选使用新 label，脚本发现已有 JSONL 就拒绝覆盖。一个 label 对应配置、日志、原始 request record、metrics before/after、OTLP、MinIO 和派生阶段结果。保留所有正式轮次，失败也写入原始记录，不“取最快一次”。

`--setup` 只负责建表/插入/封存。数据准备的 flush 和 timed Search 明确分开。正式 loop 不做手动 Load、不在计时前提前取 vector 数据。后来 prefetched GET 是在这一次共享自动 Load 内发生的，仍被 SDK 总时间覆盖。

### 3.2 NotLoad 不是足够的冷态证据

Release 会改状态，但 QueryView lease 可能继续保护旧 segment。一开始容易把“API 已返回 NotLoad”误当“物理数据必已释放”。为解决这个问题，正常确认统一使用 lease 60 s、等待 62 s，再验证：

- Go segment_load_attempt total count 增加 1。
- C++ segment Load total count 增加 1，并检查细分阶段完成/成功。
- 目标 segment 的 MinIO raw GET 真正重新发生。
- Search 后状态为 Loaded，结果仍与独立参考一致。

为了让尝试不每次花八分钟，另设 short-lease screening。这个区别一直保留在结果里：v62 短测六次 <10 ms 后，v64 正常 lease 仍出现 >10 ms，直接说明不能把快速筛选当最终验收。

### 3.3 三种观测交叉校验

第一层是 SDK 单调时钟：用户实际等待多久。第二层是 Prometheus sum/count 的单次差分：哪些 load 阶段真实发生，完成次数是多少。第三层是 trace 与 MinIO 逐请求时间戳：谁在等待谁、哪些 IO 真并发、同名 RPC 是否来自同一条因果链。

我没有把 histogram 的 p99 或 rate() 当某一次搜索的耗时，也没有把并行 span 的 duration 相加说成总延迟。Ready、execution、SDK 其余是互斥分区；WAL/native/parquet/chunk 是内部解释。

MinIO 的 trace 特别关键。关闭了某个配置、增加了两个 worker，只能说明“可能并发”；两个 GET 的时间区间有交集才是“这次真的并发”。后来的 v81 三 GET 还暴露了一个分析陷阱：区间连通会把迟到 fallback 包在一个长请求里，得出一波，但它们不是一次同时发起。

### 3.4 控制观察工具本身的扰动

完整 `/metrics` 文本曾达到约 15–19 MB，抓取和解析本身有分配成本。确认 cohort 中仅保留固定 before/after 抓取，不临时多抓一次“看看有没有激活”，也不并行编译、race test、哈希整个库或 CPU profile。

tracing 也做过关闭对照，没有稳定改善，所以正式实验保持 100% trace，让阶段归因可核验。先观察，再决定要不要改变测量环境；不能一见长尾就猜 GC。某次 Go3 尾延迟里 GC count 没变，也就不能写成 GC 根因。

## 4. 第一阶段：先看访问链，不急着优化算距

早期看到整体几十毫秒，而纯 BF 只有约 0.3 ms。对象读取却是 footer、PK、Bloom、vector 串起来；WAL append 还有接近 batching timer 的等待。于是实验先围绕依赖链展开。

部署从 EBS 迁到 NVMe 后，原配置约 47.851/47.551 ms 变成 33.495/35.572 ms。这先建立了一个更可控的本机基线，而不是把不同磁盘条件下的数字混成“代码快了多少”。

保留的第一批手段是：单条 WAL 即进入同步持久化、lazy group 不打开不需要的标量、关闭当前搜索不需要的 Bloom 加载、小 Parquet 整读、向量直接返回而不是再 QueryOnView。整体确认逐步进入 15 ms 左右，随后 v28 约 14.156/14.742 ms。

这里同时有反例：

- 最早 async-open 只是让 Load 提前结束，Search 等 vector 又多约 2 ms，整体 15.470/16.522 ms；撤回。
- 单线程减少了调度资源，却让两个 GET 串行，整体升到 20 ms 以上；不是“线程越少越快”。
- IVF_FLAT nlist=1 能得到正确答案，但这个数据量下引入索引文件和 HEAD，3 次 S3 / 2 步，没有胜过直接 BF；不保留。
- 调 jemalloc decay 单独只有很小、不稳定差异；最终保留了运行配置以复现，但没有把它列成确定的毫秒级收益。

这阶段也澄清了用户追问的 flush：自动 Load 的 WAL 消息是 AlterLoadConfig，coord/flush 指元数据保存；不是搜索把 collection 的数据重新 flush 一遍。

## 5. 第二阶段：小数据搬运、元数据和 rollover 长尾

v28 后的读 IO 已经缩短，开始能看到 chunk 构造、短 RPC 和 catalog 写入的比例。

我把小 vector 的 Arrow buffer 留给 chunk 持有，避免再复制一遍；实际 chunk build 从约 0.74 ms 到 0.05 ms。边界要求完整 buffer、布局可直接解释、owner 活着，不能为了省拷贝留下悬空指针。

LoadConfig 的两个小 etcd 写合为一个有界 MultiSave，保留相同键/序列化与大请求回退；QN DescribeCollection 和 GetQueryViewLoadInfo 改为并发。后者不只验证结果成功，还用 barrier 保证两次调用都启动、错误时 sibling 能取消并 join，防止一个“并发函数”实际仍串行或泄漏。

WAL rollover 则是另一类问题：正常搜索十几毫秒，不代表不会撞到固定 5 秒 Fence。强制 rollover 的对照把问题触发出来，再确认 own flock 的 writer 已经 drain 且拒绝新写，这时不必等待“其他 writer”观察 fence。recovery handle 不拥有锁，必须保留 5 秒等待。另把 25 ms drain poll 改为完成通知。这两项单列，不能算作正常均值优化。

v41 正式确认约 13.274/13.902 ms。这个版本包含多个代码和部署变化，只有阶段和控制实验支持的部分才能分摊收益，不能把每项都报“快了 1 ms”。

## 6. 第三阶段：重新做真正的异步 reader 与调度对照

第一版 async-open 被否定后，后续重做的是有明确生命周期的实现：单 FLOAT_VECTOR reader 的初始化本身在已有 async CacheSlot 工作中执行，持有 operation context 和 group owner，Release 能取消排队任务，前台失败可以重试。17 个 group 测试覆盖并发初始化、Release、失败、OOM 等场景。

v44 短测约 11.51/11.61 ms，但 v45 正常 lease 为 12.47/13.55 ms，并且一轮 GET 仍不重叠。由此继续检查服务调度和 IO，而不是只盯代码里已经存在的 async 标志。

MinIO 改 CPU 5–6 / Go2，etcd 改 host 网络 CPU7 / Go1，WAL loop 开 direct IO；v49 正式约 12.009/12.212 ms。这是组合部署收益。设备 fsync 的局部下降有独立诊断，但不是完整搜索的纯单变量 A/B。

期间试过 AWS 接收 buffer 和 checksum 选择缓存，没有证实收益；最终只保留“checksum header 不用 stringstream”的等价简化。功能测试通过，是允许继续比较性能的前提，不是保留更复杂代码的理由。

## 7. 第四阶段：gRPC 与 C++ future 进入 11 ms 左右

本机 standalone 仍有许多内部 RPC。增加精确地址的 bufconn byte transport，保留整套 gRPC 协议而不是直接函数调用，避免靠跳过 TLS/interceptor/序列化换一个不可比较的系统。

但仅做 in-process、再把 Go 并发度压到 2，v52 反而出现 19–28 ms 等待。这说明平均小开销下降后，future callback 与调度仍会形成明显停顿。

下一步引入 native Baton 等待，同时把等待 slot 和普通 CGO/cancel slot 分开。容量满时走原 callback，不能让等待者把 future_cancel 的入口也占满。测试刻意做 saturation/cancel、非协作任务、双重消费和 Release 竞态。

v55 短测约 10.52/10.60 ms，v57 正式约 11.695/10.967 ms。每次真实请求 native wait count = 2，证明新路径参与了测量。正式继续保留 trace；“关 tracing 应该会更快”的猜测未获得稳定数据支持。

## 8. 第五阶段：最有效的后段改动是把 GET 与 WAL 重叠

v57 的控制路径里，先完成 WAL，再准备 view，再读对象，形成清楚的串行依赖。数据是 immutable 的，读取本身不会发布 load state；因此在同一次共享 Load 里提前做只读预取，有机会重叠 WAL 耗时。

实现时没有直接开一个永久 raw cache。scope 只属于这一次 Load，按 fs 身份/key/size 匹配，一次消费，结束必清理；文件、entry、进程 bytes 都有上限。排队预取可以被前台接手，避免同池等待死锁；已经执行的同步 IO 仍持预算直到真的退出。

第一个 v60 候选“看起来打开了”却没有变快。计数 reserved/consumed 都是 0，追到数据才发现：FLAT-NO_TRAIN 有 IndexInfo，却没有实际索引文件。准入检查错用了“有无索引元数据”，导致目标 workload 根本没进新路径。修正成检查真实 file keys 后，v61 才命中。

随后把准备 metadata 本身也移到共享 loadCtx 下后台执行，退出时 cancel + join，避免只是把启动 IO 前的同步工作留在关键路径。v62 ON 六次全部 <10 ms；同二进制关开关的 v63 回到 10.905/10.813 ms。这个约 1.2 ms 的 E2E 差异，比跨版本总均值更适合归因。

但 v64 的 normal-lease 正式确认仍有超标：9.849/10.122 ms。目标没有在那时宣布完成。短周期过线只能支持继续做正式确认。

## 9. 第六阶段：剩余小成本和不能贸然删掉的语义

继续压 0.1–0.3 ms 时，很容易把成本“挪走”或者删掉看似冗余的正确性约束。

- Proxy 的 Strong query 已由 WAL 给 QueryPlan MVCC，不代表原始 TSO 就无用。审计看到 TTL、iterator SessionTs、requery 仍消费它，所以没有直接跳过 AllocTimestamp。
- DataView 能提供 LoadInfo，但当前 GetQueryViewSegmentLoadInfos 已是本地调用，不能宣称删掉一个远程 RTT。
- Parallel JSON stats v58 对比控制 v59 没变快，撤回。
- Go3、Go1、WAL idle 100 ms 都进行了控制，没有稳定改善，恢复最终 Go2 / idle10。

确实还能省掉重复元数据转换：不可变、显式版本的 manifest stats 描述可以做有界缓存。两次重复 FFI 变成两个 cache hit；同一 binary ON/OFF 的 E2E 有约 0.11–0.32 ms 差异。v70 两场景均值都小于 10 ms，但 ID 仍有 10.381176 ms，所以没有把“均值过线”偷换成“每次过线”。

删除路径描述缓存进一步降低 delta_logs 阶段，但 v71 出现 20.614160 ms，InitializeRecovery 就占约 10.972615 ms。它没有被删掉，也没有因为 delta 子阶段变快就宣布整个候选更快。v73 又出现向量 14.973801 ms，sync_up 等待约 5.0226 ms。

为此增加 SN catalog timer，不改执行逻辑。v74 + v76 共 32 次诊断未复现 5 ms sync_up；正常 SN save 约 0.24–0.48 ms。能说的是“异常落在确认阶段、这次没有复现”，不能说“已经证明 etcd 慢并修好”。

## 10. 第七阶段：PK 的负对照真正抓住了“成本搬家”

PK fill 原来约 0.15 ms。第一版 v77 直接读已有 raw PK column 后，前段指标下降，可整段 reduce 没有相应下降；后段 output fill 多出相近成本。

沿真实调用往后追，看到 `bulk_subscript_from_state` 在判断请求的是哪个字段之前就 PinPkIndex。也就是说，哪怕输出 vector，也可能把刚绕过的 PK index 又建回来。

v78 同时修前后两处：只在需要 PK index 的路径 pin，尊重已有 raw-field preference；保持外部集合和 index-only fallback，不删 lazy index 槽，delete 后续仍能建。

测试不只比输出：真正检查 CacheSlot IsCached。乱序、重复 offset、空结果、取消、外部虚拟 PK 都覆盖。把**同一套测试换回旧 native 库**，三个不该建索引的断言失败。这比只测新 helper 更能证明优化目标已达成。

相邻旧 core v79 → 新 core v78，reduce 下降约 0.127 / 0.154 ms，SDK 小幅下降约 0.107 / 0.096 ms。v80 正式均值 9.515/9.915 ms，但仍有一条向量 10.204165 ms。这个异常同样留下。

## 11. 最后一轮：拒绝排序尝试，核对固定 worker 的真实行为

试过把大文件优先提交。v81 和恢复原序的 v82 各有 12 条短测记录；v81 没有更好，且出现三个 GET 的 fallback，原始记录已分析：一个预取读没被消费，scope 结束后普通 loader 发起新读。修改提交顺序并不能保证对象并发，更不能解决所有尾延迟。

因此撤回 longest-first，只保留专用两 worker 的受限 IO executor。为什么不是用旧池线程名数量证明原因？因为 `/proc` 线程名会继承/截断；旧池数量从 4→3 只能说明某种变化，不能对应到那次慢请求。perf 工具也没有匹配当前 kernel 的可用版本。

新的验证用 `autoload_io0/1` 的 TID + start_ticks，跨空闲/Release 比较身份。第一次用全名相等过滤而遗漏了数字后缀，得到空列表，这个观察无效；改为前缀并核对 TID 后才有证据。

v83 的同两个 worker 跨多个空闲间隔存活，六次 GET 均真重叠，SDK 为：

- ID：9.691247、9.608509、9.378333 ms。
- ID + vector：9.954684、9.836449、9.640484 ms。

于是对**这六次固定 workload**完成 10 ms 目标核对。没有把池的作用夸大：与 v80 比，ID 均值还略慢 0.044 ms，向量略快 0.104 ms，单项没有显著均值收益证明。最后过线和“已经根治旧长尾”是两件需要不同证据的事。

## 12. 交付阶段怎么保持提交干净

用户要求每个优化一个 commit，并推到 liliu-z 的一个分支。我没有直接把实验工作树一次性 add/commit：那会混入 rejected patch、机器绝对路径、native 输出 symlink、多个候选二进制与复用缓存。

具体整理方式：

1. 新建出版用 worktree，从正确 sunby base 开始。NVMe 顶层不可写时只给这个新目录设置所有者，没有改整个盘权限。
2. 对 v28/v41/v45/v55/v62/v70/v71/v74/v78/v83 的冻结 patch 用临时 Git index 生成 tree，避免手工猜回某个中间实现。
3. 按文件/完整 hunk 拆开相互覆盖的配置变化，分开 gRPC 和 native future、owner Fence 和 flush notification；不保留 longest-first、parallel JSON 等被否定代码。
4. 外部依赖不能遗漏：固定 storage、gRPC、Woodpecker、AWS 版本，各自 patch 跟对应优化在同一个 commit。构建助手复制到私有 overlay，不污染共享缓存。
5. 最终 Milvus production tree 与 v83 字节一致，storage/gRPC/Woodpecker 修改文件也逐字节一致。新 Go build 从该 worktree 与私有 module overlay 编译通过。
6. 重跑取消/缓存/etcd/prefetch 等选定 Go race、整个 qvresource、gRPC overlay race、原生 scoped check；用旧原始 capture 重放新的可移植分析脚本，核对每个 summary 字段。
7. 生成历史 run ledger、正式阶段数据、依赖与测试 hash。大型 metrics、完整服务日志、binary、构建树保留原工作区，不进入仓库。
8. 两份中文文档一起做最终文档 commit；全部提交带 Li 的 DCO sign-off；只推用户要求的 fork 分支，不创建未请求的 PR。

过程中的一些小问题也记录下来：首次依赖 patch 的 Git whitespace check 把 unified diff 的空上下文当成尾随空格，使用 patch 专用 attributes 处理，没有剥掉会破坏补丁的 context；一次测试 regex 显示 `[no tests to run]`，随后针对实际函数名补跑，不能把空跑当覆盖。渲染运行配置时补上 glog.conf，避免只复制 YAML 而漏掉已知的启动依赖。


交付阶段另做两次正常 lease / Release 等 62 s 的 harness 功能复测：ID **9.748019 ms**、ID + vector **9.454053 ms**；结果正确，Go/native Load 各一次，两 GET 真正重叠。运行的仍是冻结 v83 服务，这两次单列在 [publication-harness-smoke.json](evidence/publication-harness-smoke.json)，不重算原六次均值。

## 13. 这次工作留下的可复用方法

- 优化入口必须证明命中：开关、代码存在、单测通过都替代不了 reserved/consumed、cache hit、实际调用计数。
- 先看端到端，再看嵌套阶段。阶段降低只能解释一部分；v77 是成本搬家的直接反例。
- 冷态要有物理证据。NotLoad、Release 返回、等待时间三者都不足以单独证明。
- 并发要看区间和因果链。两个 request 重叠是一回事，是否发生后续 fallback 是另一回事。
- 构建 provenance 是性能 harness 的一部分。源代码、Go binary、native 两个 DSO、配置和数据要成套。
- 负对照很值钱：旧库跑同一测试应当在优化目标断言上失败；同 binary 关开关应当回到原路径。
- 失败样本不是噪音垃圾。20 ms、15 ms、三个 GET 都改变了后续检查方式，必须保留。
- 接近目标时要缩小结论，不扩大口号。六次 <10 ms 可以完成本轮特定目标，不能代表生产尾延迟承诺。

## 附录：已导出 cohort 的完整版本索引

以下按版本号排列所有存在 summary 的 nvme-v* cohort。n 为各场景样本数，均值/最大值单位 ms；screening、首搜和正常 lease 不能混在一起估算总体分位数。没有 summary 的版本号不补造数据。每个单次样本和更多早期诊断见 [experiment-ledger.csv](evidence/experiment-ledger.csv)。各轮 profile 与 runtime 参数另见 [experiment-configurations.json](evidence/experiment-configurations.json)，包括早期线程池、allocator、Go 调度和开关组合；文件存在本身不代表它曾参与某个结果，仍须对照 label 与 provenance。

| cohort | ID n；均值 / 最大，ms | vector n；均值 / 最大，ms |
|---|---:|---:|
| nvme-v2 | 1；17.607407 / 17.607407 | 1；17.884992 / 17.884992 |
| nvme-v3-first | 1；30.618955 / 30.618955 | — |
| nvme-v3 | 1；16.370830 / 16.370830 | 1；16.874219 / 16.874219 |
| nvme-v4 | 1；15.838533 / 15.838533 | 1；16.046415 / 16.046415 |
| nvme-v5 | 1；14.895720 / 14.895720 | 1；15.144338 / 15.144338 |
| nvme-v6 | 1；15.469696 / 15.469696 | 1；16.522403 / 16.522403 |
| nvme-v7 | 1；21.217579 / 21.217579 | 1；20.627365 / 20.627365 |
| nvme-v8 | 1；29.914140 / 29.914140 | 1；29.609110 / 29.609110 |
| nvme-v9 | 2；15.265434 / 15.604908 | 2；15.194285 / 15.222068 |
| nvme-v10 | 1；20.558191 / 20.558191 | 1；15.182754 / 15.182754 |
| nvme-v11 | 1；15.934435 / 15.934435 | 1；16.233211 / 16.233211 |
| nvme-v12 | 1；15.288301 / 15.288301 | 1；16.041938 / 16.041938 |
| nvme-v14-fast | 2；14.672242 / 15.136398 | 2；14.670786 / 14.670860 |
| nvme-v14-long | 1；15.059814 / 15.059814 | 1；15.133761 / 15.133761 |
| nvme-v14-tight-0 | 1；16.060355 / 16.060355 | 1；14.693481 / 14.693481 |
| nvme-v14-tight-1 | 1；14.266277 / 14.266277 | 1；14.659836 / 14.659836 |
| nvme-v14-tight-2 | 1；13.666331 / 13.666331 | 1；15.350851 / 15.350851 |
| nvme-v15-fast | 2；14.735912 / 14.752674 | 2；15.375453 / 15.698383 |
| nvme-v19 | 2；14.567441 / 14.602664 | 2；15.856293 / 16.837038 |
| nvme-v20 | 2；14.119817 / 14.177543 | 2；14.479996 / 14.682727 |
| nvme-v21 | 2；15.090411 / 15.895605 | 2；14.675281 / 15.019576 |
| nvme-v22 | 2；14.244988 / 14.300388 | 2；14.162793 / 14.368261 |
| nvme-v23 | 3；26.401176 / 50.938595 | 3；15.402499 / 16.541312 |
| nvme-v24 | 3；17.560392 / 23.481336 | 3；14.266313 / 14.464050 |
| nvme-v25 | 3；28.951918 / 59.413360 | 3；14.485774 / 14.981922 |
| nvme-v26 | 3；21.881015 / 28.462983 | 3；14.045623 / 14.223348 |
| nvme-v27 | 3；17.102768 / 24.146449 | 3；15.343141 / 17.427491 |
| nvme-v28-confirm | 3；14.155954 / 14.228072 | 3；14.741890 / 14.915642 |
| nvme-v29 | 3；14.849228 / 15.093645 | 3；14.621073 / 14.943007 |
| nvme-v30-steady | 3；13.753067 / 13.938525 | 3；13.715462 / 13.886086 |
| nvme-v30 | 3；14.168173 / 14.326333 | 3；14.162204 / 14.718675 |
| nvme-v31 | 3；17.438872 / 24.474245 | 3；13.942668 / 14.490138 |
| nvme-v34-confirm | 3；14.445075 / 14.947979 | 3；14.639614 / 15.060233 |
| nvme-v35-combined | 3；13.845711 / 14.320250 | 3；13.795301 / 14.000371 |
| nvme-v36-immutable | 3；13.916301 / 14.293784 | 3；13.835351 / 14.105710 |
| nvme-v38-immutable | 3；13.264127 / 13.375295 | 3；13.564765 / 13.980961 |
| nvme-v39-host | 3；13.126689 / 13.439104 | 3；13.517164 / 13.780996 |
| nvme-v40-affinity | 3；12.840430 / 13.195617 | 3；12.989973 / 13.220083 |
| nvme-v41-confirm | 3；13.274345 / 13.432719 | 3；13.902495 / 15.130270 |
| nvme-v42-aws-buffer | 3；12.818300 / 13.139894 | 3；13.133718 / 13.298466 |
| nvme-v43-aws-cache | 3；12.985678 / 13.551416 | 3；13.475267 / 14.314136 |
| nvme-v44-async-open | 3；11.509939 / 11.684412 | 3；11.606442 / 11.669666 |
| nvme-v45-confirm | 3；12.473029 / 13.786024 | 3；13.554026 / 14.283611 |
| nvme-v46-minio-gomax2 | 3；13.087000 / 15.480366 | 3；11.618454 / 11.867350 |
| nvme-v47-etcd-host | 3；11.312513 / 11.719063 | 3；11.538933 / 11.805380 |
| nvme-v48-loop-direct | 3；11.037256 / 11.120333 | 3；11.618819 / 11.706508 |
| nvme-v49-confirm | 3；12.009415 / 12.963169 | 3；12.212438 / 12.608165 |
| nvme-v50-local-grpc | 3；11.219479 / 11.546128 | 3；10.789824 / 11.003017 |
| nvme-v51-local-go3 | 3；10.864715 / 11.257151 | 3；11.216280 / 11.763935 |
| nvme-v52-local-go2 | 3；21.871886 / 27.661286 | 3；18.835831 / 25.236363 |
| nvme-v53-unsampled | 3；10.932980 / 11.181482 | 3；10.626575 / 10.951219 |
| nvme-v54-native-wait | 3；11.663038 / 13.723267 | 3；10.804672 / 10.924367 |
| nvme-v55-native-wait-go2 | 3；10.524239 / 10.616989 | 3；10.602597 / 10.871062 |
| nvme-v56-native-wait-unsampled | 3；10.669304 / 10.900624 | 3；10.672661 / 10.809289 |
| nvme-v57-confirm | 3；11.695200 / 12.509459 | 3；10.966598 / 11.124561 |
| nvme-v58-parallel-json | 3；11.760045 / 13.727578 | 3；11.452022 / 12.066091 |
| nvme-v59-native-wait-control | 3；10.691323 / 11.199672 | 3；10.612190 / 10.798865 |
| nvme-v60-scoped-prefetch | 3；10.808492 / 11.341723 | 3；11.054463 / 11.493417 |
| nvme-v61-scoped-prefetch | 3；9.731793 / 9.763445 | 3；10.239638 / 11.223266 |
| nvme-v62-scoped-prefetch-async | 3；9.649140 / 9.873835 | 3；9.631154 / 9.930678 |
| nvme-v63-prefetch-off-control | 3；10.904931 / 11.622872 | 3；10.813152 / 11.046306 |
| nvme-v64-confirm | 3；9.848825 / 10.251394 | 3；10.122087 / 10.395277 |
| nvme-v65-prefetch-go3 | 3；9.699001 / 10.205674 | 3；10.070773 / 11.461713 |
| nvme-v66-wal-idle100 | 3；10.397952 / 12.090434 | 3；9.509769 / 9.885742 |
| nvme-v67-wal-idle10-control | 3；9.510521 / 9.675050 | 3；9.593511 / 9.857074 |
| nvme-v68-manifest-stats | 3；9.161611 / 9.308487 | 3；9.595470 / 9.887581 |
| nvme-v69-manifest-off | 3；9.483227 / 9.598913 | 3；9.701790 / 9.889499 |
| nvme-v70-confirm | 3；9.851744 / 10.381176 | 3；9.818669 / 9.976608 |
| nvme-v71-manifest-delta | 3；13.165634 / 20.614160 | 3；10.372790 / 11.925068 |
| nvme-v72-stats-only-control | 3；9.231267 / 9.428868 | 3；9.771007 / 10.028664 |
| nvme-v73-confirm | 3；9.771481 / 10.392536 | 3；11.387437 / 14.973801 |
| nvme-v74-sn-persist-probe | 6；9.871943 / 10.613089 | 6；9.714580 / 9.977276 |
| nvme-v75-go1 | 6；9.966492 / 10.458139 | 6；10.436971 / 11.744629 |
| nvme-v76-sn-persist-tail | 10；9.351041 / 9.943954 | 10；9.348657 / 9.613364 |
| nvme-v77-int64-pk | 3；9.386537 / 9.451365 | 3；9.693101 / 9.842341 |
| nvme-v78-int64-pk-output | 3；9.545944 / 9.965181 | 3；9.360813 / 9.696799 |
| nvme-v79-pk-control | 3；9.653174 / 9.717013 | 3；9.456868 / 9.808849 |
| nvme-v80-confirm | 3；9.515349 / 9.668970 | 3；9.914880 / 10.204165 |
| nvme-v81-prefetch-longest-first | 6；9.296848 / 10.217963 | 6；10.488446 / 13.225701 |
| nvme-v82-prefetch-control | 6；9.005287 / 9.160596 | 6；9.369039 / 9.809411 |
| nvme-v83-confirm | 3；9.559363 / 9.691247 | 3；9.810539 / 9.954684 |
