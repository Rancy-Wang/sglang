# Drop / Reposition 迁移执行账本

批准计划：`PLAN-CS-20260917-R2`，实施轮次 1。本文记录已批准的目标、证据和剩余工作，
不是功能已经实现或性能已经达标的声明。

## 基线与完成状态

| 项目 | 固定版本 / 状态 |
| --- | --- |
| SGLang 官方基线 | `9d0a8d75364ea4571e05ba2e37227ec2579324f2` |
| Rancy-Wang/sglang main | 已从 `b3570a453` fast-forward 到上述基线并推送；未改写历史 |
| 实施分支 | `system`，从上述基线创建并推送 |
| mini-sglang System | `89d8a9fd22a2784a8989bdd7b80e7a232d5e877e` |
| mini-sglang main | `9a91cfafe754aa85daee49998176275667eb58f2` |
| mini-sglang System-test | `2966eb49a522041f9c42bce7dca07119ef6929de`；只读参考 |
| 历史清单 | 52 份清单内历史会话已逐条通读；阅读覆盖与运行时验证分开记录 |
| System 相对 main | 75 个文件的完整 diff 已通读；逐文件范围见 `source_inventory.json` |
| 生产功能 / GPU 验证 | 尚未完成，不能据此部署或宣称等价 |

基线冻结以后不持续追逐上游移动的 main。后续实验记录实际 `system` commit、模型配置、
权重版本、依赖版本、GPU 映射和后端选择。审计时使用过的 `923e4a56d` 与最终冻结点相比，
本次迁移涉及的 SRT 源码没有变化；kernel 变化仅为 CPU 构建依赖。

2026-09-18 用户修订性能验收：同模型、GPU、并发和工作负载下，启用 Drop/Reposition
后的输出吞吐须达到原生 SGLang 的 **95% 以上**，即不得下降超过 5%。原生 SGLang
无功能时慢于 mini-sglang 可以接受；mini 不再是吞吐硬门槛，仍是功能、计算和 SWA
机制的对照。该修订替代此前约 3% 的目标。保留 TTFT、TPOT、实际计算量和输入差异，
避免把少算 token、额外 JIT 或模板差异误归为实现开销；本轮不测 SLO。

## 追加验收与验收后的实验顺序（2026-09-18）

用户在监控任务 `01a0b06e-a973-77e0-9306-1f70ba537820` 中追加并授权：R2 除上述
95% 原生吞吐门槛外，原要求 PD C2 Drop+Repos 的 TTFT 与 TPOT 都优于普通调度
C2 Drop+Repos。用户随后明确修订：**如果指标劣势主要来自大量 KV 传输，而 D 端没有
普通调度那样的长时间等待和低效，允许 PD 的 TTFT/TPOT 不达该比较目标**。
必须结合传输量、P/D 排队时间和长 decode 间隔给出证据，不能仅因使用 PD 就豁免。
这不放宽相对同资源原生 SGLang 的 95% 吞吐门槛。普通调度和 PD 必须各自在同模型、同 task/turn 下通过 mini 数值
误差校准。不能用修改版 no_drop 替代原生吞吐对照，也不能仅凭退出码或部分 smoke 验收。

后续 PD 最小实验的原生及修改版均开启 `--enable-request-time-stats-logging`，每个完成
请求各记录一条原生阶段摘要。注意 `ReqTimeStats.convert_to_duration` 的 D
`transfer_duration` 从进入接收队列到 KV 就绪，也包含等待 P 计算，不能当作纯网络
耗时；Mooncake 未提供 `transfer_latency_s` 时，`compute_and_observe_kv_transfer_metrics`
只使用最后 chunk 的队列耗时，报告的总字节/速度不能当作全程有效带宽。
证据不足时标记尚未归因，不宣称满足这一条件豁免。

只有 R2 全部通过后，才开始以下独立实验阶段；当前尚未满足启动条件：

- 两组均为 GPT-OSS-120B、PD 四卡，P=GPU0/1 TP2、D=GPU2/3 TP2。
  A=drop-aware eviction + Drop+Repos；B=普通 eviction + no_drop；资源不足以并行时顺序跑。
- 沿参考任务 `01a0ab12-b9ee-7751-bee0-6a78dcac9c1f` 的最终测速约定，依次
  C=1/2/4/8/16/32，每组3轮，即3/6/12/24/48/96条不同首遍 task；每完成一个 C
  立即输出 A/B 全部与逐轮指标。保留完整轨迹、seed、实际生成历史、源输出长度、
  ignore_eos、恒定并发、filler 截止、Rolling Drop keep12、累计96Ki Reposition。
- 报告实际 Prefill/Decode/All 吞吐、mean/P90 TTFT/TPOT/TBT/E2E、成功/失败/Abort/filler、
  实际并发、cache/eviction 证据、长 TBT 次数及等待、原始 JSON。精确计算计数缺失时
  标为缺失，不用逻辑 tokens 代替。标准 TPOT、排除首 decode gap 的自定义 TPOT
  与 TBT 分列；客户端 SSE ITL 不冒充服务端 TBT。
- 全部矩阵结束后才做 C1、96 tasks 的 A/B 同配置 SLO 基线，再补算 SLO；前期报告
  标记待基线。不得使用本次 C1/N2 或旧跨主机结果代替。该阶段的 SLO 不改变 R2
  小量验证不测 SLO 的范围。
- 本任务保持唯一修改和实验启动者；不影响 InfiniAI-BUS GPU0/1 的参考实验。

## 必验模型数值证据汇总（2026-09-18）

BUS `r2-numeric-evidence-final-audit-v1.json` 离线读取已保存的 comparison JSON，
没有重新执行模型。Qwen3-0.6B、AgenticQwen-8B、GPT-OSS-20B/120B 的普通与PD共8组，
每组Drop冷算、Repos冷算、Repos热命中、Retry、连续Repos共5条路径，40条均满足
原冻结BF16门限。逐路径结果、实际usage、原始文件来源和生成观察都保留在该报告。

| 模型 | max / mean / p99 允许绝对误差 | 普通 / PD |
| --- | --- | --- |
| Qwen3-0.6B | 2.25 / 0.108541 / 0.4375 | 5/5、5/5 |
| AgenticQwen-8B | 2.0 / 0.109375 / 0.625 | 5/5、5/5 |
| GPT-OSS-20B | 5.421875 / 0.139224 / 0.875 | 5/5、5/5 |
| GPT-OSS-120B | 12.1875 / 0.448795 / 2.6796875 | 5/5、5/5 |

门限逐项取 `max(固定floor, 2×同模型无功能误差)`；floor依次为0.125/0.02/0.0625，
没有按失败路径重新校准。Agentic连续Repos普通路径使用已明确记录的同缓存路径mini
参考，保留初始冷算参考失败；不是放宽门限。实际生成输出仍按下文的原生语法/采样差异
单独报告，数值通过不代表自由生成逐token完全相同。模型范围、BF16、page1、后端及
共享Full/SWA设置之外不外推。PD长轨迹性能与等待归因尚未完成，整体R2仍待验收。

## 原生 PD C1 基线与阶段分析（2026-09-18）

BUS `minimal-native-pd120-c1-none-parser-v1` 使用原生兼容基线 `d7a3df66a`，
P=GPU0/1 TP2、D=GPU2/3 TP2、GPT-OSS-120B、默认 Triton、page1、chunk8192、
KV262144、CUDA Graph。2/2 首遍 task 完成，84 请求成功、0 失败、无 filler，
27390 输出 tokens / 1086.423733s = **25.211158 token/s**。
TTFT mean/P90 为 `7479.460/8867.849 ms`，TPOT 为 `15.453/21.380 ms`，
E2E 为 `12618.205/18973.562 ms`。原生实际 Prefill/Decode 计算计数缺失，记为 null。

离线 `benchmark/context_system/analyze_pd_timing.py` 通过 bootstrap room 关联
84/84 请求与 P/D 完成日志，无缺失。它不运行模型、不增加推理 hook；在完整 run
目录上执行 `python benchmark/context_system/analyze_pd_timing.py RUN --output NEW.json`。
BUS 已保存 `pd-native-c1-timing-analyzer-final.json`；本地 Ruff 与 py_compile 通过，
对完整原始日志重新分析后的逐请求阶段值与先前独立分析一致。未完成 filler 的截止取消
单列，不计为失败或缺失关联。请求数、完整性检查仍由原始 benchmark result 验收。

| CPU 观测阶段 | mean / P90 / max（秒） |
| --- | --- |
| P forward，包含 chunk 间等待 | 7.135255 / 8.469192 / 10.879159 |
| P 最后传输尾段 | 0.079292 / 0.089187 / 0.537503 |
| P 完成到 D KV 就绪 | 0.081125 / 0.091506 / 0.538803 |
| D 接收等待，包含 P 计算 | 7.225027 / 8.571852 / 10.988711 |
| D KV 就绪后排队 | 0.000141 / 0.000221 / 0.000328 |

日志只报告 TP0 的传输量，mean/P90/max 为 `2178.71/3895.37/4603.46 MiB`；
不能把它当作两卡实测总量。现有 Decode 日志每40次 forward 记录一次，时间戳精度1秒；
用请求完成点校准后，在正式 D forward 区间内的采样空隙最大约1.23秒。
客户端最长 SSE 间隔约5秒，其间 D 仍有进度日志，不能把它写成5秒 GPU 停顿；
这种粗粒度证据也不能排除较短停顿。默认间隔来自
`python/sglang/srt/arg_groups/fields/observability.py:143`，记录条件位于
`python/sglang/srt/managers/scheduler_components/metrics_reporter.py:856`。
上述是原生 C1 结果，尚不能替代修改版 C1/C2 的效率和等待归因验收。

## PD 长历史 TCP 队列满与分批修复（2026-09-18）

修改版 `d4b598572` 的 `minimal-pd120-c1-none-parser-v1` 已完成84请求、0失败，
27390 / 1095.360421s = **25.005468 token/s**，比上述原生 PD C1 低 **0.816%**。
全部84个 assistant 对象在仅剔除随机 tool-call ID 后相同；原始对象76/84相同，
不能把随机 ID 导致的后续历史哈希变化当作生成内容分歧。阶段分析中 D KV 就绪后排队
mean/max 为0.000134/0.000312秒，粗粒度 D 进度空隙最大约1.31秒。
原始结果与对照分别保存为 `pd-modified-c1-none-timing-final.json`、
`pd-c1-native-modified-content-comparison.json`。

`minimal-pd120-c1-drop-parser-v1` 在第二个 task case786 的 turn31 失败，
75成功/1传输错误，整组无效，不计作吞吐通过。P 的两个 TP rank 均记录
`TCP lane queue-full rejection`，继而 P HTTP500 / D 接收失败；该请求 active39431、
raw89930、cached86982、PF2948。前一 task 的 Reposition 与热缓存延续已经执行，
但不能用这些部分成功替代全轨迹验收。

BUS 安装 `mooncake-transfer-engine-cuda13==0.3.13`，无 HCA 时走 TCP。
上游同版本 [`tcp_transport.cpp`](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport.cpp#L302)
默认每 peer 排队1024项、pending admission1024项；
[`enqueuePooledTransfer`](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13/mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport_lane_impl.h#L580)
在两者均满时拒绝。Drop 后不连续的 page1 KV 区间跨所有 K/V 层展开，native 单次
`batch_transfer_sync` 可以超出这个数量。原有按 token indices 分批的选项没有直接约束
“区间数 × 层数”，且默认关闭。

R2 修复限定 `MooncakeKVManager._transfer_data`：每次最多提交1024个地址区间，
同步完成后才提交下一批；1024项以内保持一次调用，连续大 KV 不按字节拆分。当前支持的
同构 TP、无 custom memory pool 路径把相同 peer 的 chunk 放在同一 worker 中串行发送。
保留既有源页租约、最终元数据发布时机、失败处理，不添加部分写入后的盲目重试。
本次边界按上述默认 TCP 队列配置验证；人为缩小底层队列或未支持的 custom pool 并行
不是本次验证范围。测试 `test_mooncake_fragmented_transfer.py` 覆盖地址/长度与顺序、
边界批次、失败后停止，以及独立GPU进程 TCP 字节级校验。BUS 在 `862c7af37`
执行 `pytest -q test/registered/context_system/test_mooncake_fragmented_transfer.py -k "not real_tcp" test/registered/unit/disaggregation/test_mooncake_transfer_batching.py`
为13 passed、1 deselected、3 subtests；设置 `CUDA_VISIBLE_DEVICES=0,1` 与
`CONTEXT_TEST_MOONCAKE_TCP_GPU=1` 后执行同文件 `-k real_tcp` 为1 passed、8 deselected。
两个独立进程传输49,545,216字节，三批1024/1024/976，payload与未写入保护区逐字节正确。
完整 C1/C2 Drop 性能仍待完成，R2尚未通过。

原生 `d7a3df66a` 的 `minimal-native-pd120-c2-none-parser-v1` 同样失败：44成功、4失败，
整组无效。首次错误为case786 turn21、case787 turn23，P传输返回非零后封禁session；
D当时仍运行。该日志没有queue-full、CUDA异常或显式传输timeout证据，不能断言与上述
Drop失败同因。原生对照后续仅应用相同的1024-descriptor传输修复，模型、scheduler与
Radix保持原生；后续C2基线须标为“原生+传输兼容修复”，不冒充未修改的上游。
同一benchmark launcher为两边增加失败时的descriptor数、字节数与返回码日志，不记录
地址、不增加逐token hook；成功调用只多一层Python转发。

## PD C1 Drop+Repos 完整轨迹通过（2026-09-18）

BUS `minimal-pd120-c1-drop-parser-v2` 在`862c7af37`完成2/2 task、84成功、0失败，
无filler，越过原case786 turn31传输失败点，后续Repos与热命中均完成。
27390输出tokens / 1081.968956s = **25.314959 token/s**，相对原生PD C1的
25.211158提高**0.412%**，通过95%门槛。TTFT mean/P90为8597.794/10107.549ms，
TPOT为12.147/13.321ms，E2E为12547.833/17327.174ms。
实际PF245126、D27306，对应226.555484/25.237323/251.792806 token/s（PF/D/总计算）。
usage累计cached2671500、repos74157、drop-skipped1385529；最多一次R。
三组C1原始assistant均无空输出、非法tool arguments JSON或替换字符。

`pd-modified-c1-drop-timing-final.json`关联84/84请求、无缺失。P forward含chunk间隙
mean7.270984s，P最终传输尾段mean0.805331/P901.057909/max1.143720s；
P结束至D就绪mean0.820047s，D接收等待（含P计算）mean8.110500s。
D就绪后排队mean0.000606/max0.001100s。可见TP0传输量mean1186.34/max1683.95MiB，
不能当作所有rank实测总量。按现有每40次forward日志与完成点对齐，正式D计算区间的
粗粒度进度空隙最大约1秒；最长客户端SSE间隔3.327549s内有3条D进度采样，
不能认定为D停顿，也不能用这些采样排除亚秒停顿。传输分批修复解决了完整轨迹失败，
但C2尚未完成，不能据C1单独宣称R2验收通过。

## PD 容量异常路径补修（2026-09-18）

`capacity-pd-qwen-v2` 在 P=GPU0/110 KV、D=GPU1/640 KV、Qwen3-0.6B、chunk32、
原生 D 512-token reserve 下失败：P 正确拒绝自锁住 108 KV、再需 3 的请求，D 却在
60 秒 HTTP 超时。`test_serving_capacity_pd.py` **1 failed in 279.95s**，不能算通过。
根因是原生 sender `abort()` 只标记 P 本地 Failed，没有向 D 发失败消息。

此次补修限定 Context 容量错误：P 先标记 transport Failed 阻止新写入，保存 D endpoint，
等待每个 TP rank 的已开始传输计数归零，再发 Failed、清理 transport 状态并释放 KV/metadata。
等待时保留 chunk 和资源，停止新 chunk admission；普通成功路径不新增 collective。
Mooncake 跳过已取消 chunk 时只减去该 chunk 的计数，不能清掉其他 worker 的在途计数。
Context PD 此阶段明确限制为 Mooncake、无 staging；其他 transport 不冒充已验证支持。
数值对照无需因仅异常清理路径的改动重复跑全模型，但必须复测上述 GPU 失败/恢复用例。
`659611bc8` 在 BUS 的 `capacity-pd-qwen-v3` 完成真实模型复测：
`CONTEXT_PD_CAPACITY_PRESSURE=1 python -m pytest -s -q test/registered/context_system/test_serving_capacity_pd.py`
为 **1 passed in 225.89s**。P 在 2.50699s 返回 503，D 在 2.50727s 返回传输失败 500；
随后清空缓存的 90-token 普通请求两端均为 200，D 在 0.05068s 完成 8-token 输出。
这证明该容量拒绝路径能结束两端请求并恢复服务，不是所有传输取消竞态的普遍证明。
同提交的 CPU 定向清理用例 **6 passed, 7 deselected in 16.34s**，包含等待其他 TP rank
仍在传输时不得通知 D、不得释放页的检查。

## 多请求高压 BCP 续接冲突（2026-09-18）

`75a1eb49c` 的 GPT-OSS-120B C2 Drop+R 完整轨迹
`minimal-sg120-c2-drop-parser-v2` 在 135 个成功请求后触发
`Scheduler._get_new_batch_prefill_raw` 的 `assert self.chunked_req is None`；
最终 135 成功、2 transport error，只有 2/4 task 完成。该运行无效，不计为吞吐达标。
缓存洞恢复产生短 query 区间，或者容量限制缩短 chunk 后，旧请求仍需要续接，却留下
可用 prefill 预算；新候选因此可能再次占用 scheduler 唯一的 chunk 续接位置。

`1478d3a05` 在 `PrefillAdder.add_one_req` 的实际 admission 判定之后拒绝第二个
需要续接的请求；仍允许剩余预算容纳完整 prefill。若旧 chunk 因容量不足未能发起，
则将控制权交还 decode，避免给没有发起的 chunk 增加 in-flight 计数。
BUS CPU 验证命令：
`CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_admission.py -k 'short_repair_interval or parked_chunk'`，
结果 **4 passed, 13 deselected in 16.34s**。覆盖旧/本轮新 chunk、允许完整短请求，
以及暂时/不可恢复容量不足。`minimal-sg120-c2-drop-parser-v3` 已按相同完整轨迹、
GPU2/3 TP2、page1、默认 Triton、KV262144、chunk8192 重跑，最终仍失败：134 个成功请求、2 个错误。case864 turn28 的长 Prefill 持续占用 KV，case228 turn22 的 Decode 分配失败。此运行无效，不能作为吞吐达标证据。

## 延迟副本、发布边界与稀疏所有权（2026-09-18）

`5b31f7aeb` 将终态 Reposition 副本推迟到最终 Prefill，并在 admission 中预留后续
阶段及现有 Decode 的推进空间。`32ba491c5` 的 BUS CPU admission 检查为
`test_context_admission.py` **17 passed in 16.56s**。先前失败请求的 CPU 所有权回放
峰值从保留全部终态副本时超过 302K 页降到 256969 页；这只是 CPU 容量证据。
该提交的真实 GPT-OSS-20B BCP `bcp778-sg-gpt20-capacity-lifetime-v1` 仍失败：
首个 Drop 冷算在 chunk 发布时触发 `Context insert hole has no Drop proof on the inserted path`。

后续修复保留严格的 Radix 空洞校验。`request_storage.py` 的 `context_publish_length`
（`python/sglang/srt/context_system/request_storage.py:11`）用 CPU 前缀最大值计算可证明的发布边界：边界 b 的 Drop 记录只有包含 token b 的
key 才可使用；没有终态副本的 active token 不能冒充已 Drop 空洞。普通结束、未完成
chunk 和 prompt/output 分节点都遵守同一边界。如果旧稀疏匹配依赖更后面的 Drop，
则暂缓发布、保留原租约及请求的私有 KV，继续推进 raw cursor。

真实 allocator 检查随后发现恢复路径会将私有副本与借用页交错放在所谓保护前缀内，
旧的单个 `cache_protected_len` 无法表达所有权，表现为少回收 4 页。
`0427574cd` 增加 Context CPU 所有权 mask，插入时只释放私有重复页，结束/中止时
只释放请求自己的页；保留原生无 Context 的前缀规则。SWA 组件沿同一所有权边界
处理替换，避免把借用页当成新 SWA 分配。该变更没有修改 attention 的数值计算。
所有权入口为 `python/sglang/srt/mem_cache/unified_radix_cache.py:1348` 的
`_context_cache_ownership`；暂缓发布在同文件 `:1216` 的 `cache_unfinished_req`。

- 本地纯 CPU 边界/生命周期定向检查：**9 passed, 17 deselected**。
- BUS `test_context_unified_cache.py test_drop_eviction.py -k 'occurrence_native or swa_holes or native_chunk'`：
  **12 passed, 1 skipped, 32 deselected in 14.86s**，涵盖冷算、重复、Retry、缺页恢复、
  Drop 尾边界、禁用缓存，以及暂缓发布时中止后的完整页回收。
- `e6b219f0d` 修复自己刚插入的短 Context 前缀在 rematch 时误用新 Decode 请求的
  SWA window 门槛。`context_cache_publication` 仅用于 chunk 重新绑定；仍保留实际
  SWA residency，新请求匹配不变（`unified_radix_cache.py:1523` 和
  `unified_cache/unified_tree_core.py:1053`，均在 `python/sglang/srt/mem_cache/`）。
  BUS `test_drop_eviction.py -k swa_req_recovery` 为 **4 passed, 4 skipped, 22 deselected in 14.77s**。
- `bcp778-sg-gpt20-capacity-lifetime-v2` 在 BUS GPU0、TP1、默认 GPT-OSS backend、
  page1、chunk512、KV24576、生产提交 `0427574cd` 完成：
  `python -m pytest -s -q test/registered/context_system/test_bcp_numeric.py`
  **1 passed in 316.43s**。无功能、Drop、Drop+R 的实际生成均与 mini 64/64 token 相同。
  fixed 的 max/mean/p99：无功能 `2.71094/0.069612/0.4375`，Drop `4.875/0.101741/0.875`，
  Drop+R 冷 `2.27930/0.064363/0.412109`、热 `1.734375/0.062314/0.390625`、
  Retry `1.265625/0.057951/0.34375`，均通过相同无功能校准门限，raw argmax 均64/64。
  Retry 的 cached/repos/drop_skipped/actual_prefill 为 `432/3929/2820/1746`；
  热命中为 `3384/0/0/1`。热请求的 Drop 页在初始匹配时已是物理空洞，故
  `drop_skipped=0`；该计数只统计初始物理驻留却未读的 Drop KV，不把已释放空洞计入
  （`python/sglang/srt/context_system/usage.py:103`，`ContextUsage.snapshot`）。
- `minimal-sg120-c2-drop-parser-v4` 在 GPU2/3、`e6b219f0d` 完成 133 个请求后停滞；
  本次运行已中断并标记无效，具体诊断与修复见下节，不能计入吞吐达标。
- 旧提交 `1478d3a05` 的普通 C1 no-drop 完整轨迹完成：84 成功、0 失败，
  27390 输出 tokens / 1046.433687s = **26.174616 token/s**。这是对照结果，
  不代替修复后 Drop C2 与原生的 95% 性能门槛。

## 空闲容量重试被 full 标志阻塞（2026-09-18）

上述 C2 v4 在 case864 turn27 / case228 turn20 之后停滞。BUS 03:49 的 `/v1/loads`
显示 running=0、waiting=2、GPU2/3 利用率为0；3秒 `py-spy` 采样显示 scheduler
仍在接收/广播循环，客户端在等待响应。保存 `stall-evidence.json` 与 `stall-stack.txt`。
本次请求实际生成历史不同于旧 v3；待发 case864 turn28 的 raw 长度为133364，
不能沿用旧请求132996的容量数值作为这次的精确证据。

`PrefillAdder._fit_context_admission` 可设置 `context_force_miss`，要求下一轮放弃
自锁定的缓存匹配冷算；原生 scheduler 却对该 NO_TOKEN 无条件设置 `batch_is_full`。
没有 running 请求时，也就没有 decode 完成来清除该标志，下一轮直接跳过队列。
`182bdc5cc` 在 `Scheduler._get_new_batch_prefill_raw`
（`python/sglang/srt/managers/scheduler.py:4060`）处理准入新产生的 Context 容量错误，
并且仅在仍有可运行请求时对 Context 的 NO_TOKEN 设置 full。普通请求规则不变。

BUS CPU 命令 `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q
 test/registered/context_system/test_context_admission.py -k idle_rejected`
为 **3 passed, 17 deselected in 15.51s**：空闲冷算请求能再次尝试、不可恢复容量错误
从队列结束、普通原生请求行为不变。下一项是失败附近两条 BCP 请求的缓存预热/并发
重放；这是定向功能诊断，不是完整轨迹吞吐。R2仍未完成。

## 连续 Reposition 的缓存保留与定向复测（2026-09-18）

`182bdc5cc` 的 `bcp-c2-idle-pressure-v1` 用失败 C2 v4 保存的实际生成历史重建四条
请求，逐轮核对 history hash，全部 HTTP200 且输出长度正确。两条预热 raw130697/55923
耗时334.116/334.431s，后续 raw133364/58907耗时229.932/339.150s。这证明空闲调度
可以继续推进，但 case864 turn28 的 cached/repos/drop-skipped仍均为0，实际重算
133364 tokens；该诊断不是吞吐测试，也不算效率通过。

`acc6222cc` 在 `Req.prepare_context_recovery`
（`python/sglang/srt/managers/schedule_batch.py:1650`）对齐 mini 的两阶段恢复判断：
先只根据缺页与真实依赖计算恢复范围；只有确需执行历史 query 时，才将不兼容位置的
来源加入重建集合，避免低精度逆旋转污染恢复结果。仅追加下一次 Reposition、没有
必要历史缺页时，保留缓存并用独立目标页旋转，不能因最终位置变化就冷算整个前缀。
SWA residency、query窗口和终态需求在两次计划中均保留。

BUS 隔离 checkout `sglang-r2-capacity-check` 已通过 Git bundle 从已推送提交快进到
`d4b598572`（GitHub HTTPS 当次 TLS 中断，未直接修改远端源码）。CPU 命令
`CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_admission.py`
为 **23 passed in 16.99s**，新增完整缓存、不必恢复的 Drop 空洞、必须恢复的空洞三种检查。

`d4b598572` 增加真实 BCP 连续 R 数值路径：先仅带首个 R 预热，再带两个 R 请求固定
64-token路径，对照已保存的 mini logits；同时检查确有缓存复用和 Prefill减少。
Qwen/Agentic/GPT20仅补该新增路径并复用旧无功能误差校准，120B普通/PD最终检查包含它。
这些 GPU结果尚待完成，不以CPU计划一致代替模型输出验证。

同一时段完成的 `minimal-sg120-c1-drop-final-v1` 使用 `d2ef1d0f1`、GPU0/1 TP2、
page1、GPT-OSS默认Triton、共享Full/SWA池、KV262144、chunk8192：2/2 task完成，
84成功、0失败，27390输出tokens / 1000.032115s = **27.389120 token/s**。
原生C1为25.628504，提升 **6.8698%**，达到95%吞吐门槛；实际PF235943/D27306。
17个请求含R，但本组没有连续两次R的长请求，不能替代C2复测。
TTFT mean/P90为7619.376/9057.320ms，标准TPOT为12.118/13.356ms。
原始证据为实验目录下`workload/result.json`和`events.jsonl`，不测SLO。

### 连续 R 的真实长请求与数值结果

`d4b598572` 的 `bcp-c2-idle-pressure-v2` 重放相同实际历史，四请求均HTTP200、完整
输出长度。冷预热两条为331.075/331.391s；case864 turn28为20.7476s，
actual_prefill2551、cached7920/repos81644；case228 turn21为21.4710s，
actual_prefill2983、cached34099/repos0。相对v1减少的是不必要重算，不能把这组诊断
耗时直接作为完整轨迹吞吐。`minimal-sg120-c2-drop-parser-v5`随后按完整方法启动。

120B普通数值 `bcp778-sg-gpt120-capacity-lifetime-final-v1`：BUS GPU0/1 TP2、
默认Triton、page1、共享Full/SWA池，`test_bcp_numeric.py` **1 passed in 269.82s**。
连续R的max/mean/p99相对mini冷算为 `10.125/0.119956/0.90625`，
cached470/repos2913/actual_prefill2/D63。无功能、Drop、R冷、热、Retry均通过相同
无功能校准；自由生成与mini的逐位置匹配数仍为43/62/64，不冒充完全相同。

新增矩阵 `bcp-consecutive-models-v1`：Qwen普通 **1 passed in 128.80s**，
连续R max/mean/p99为`1.0625/0.064690/0.28125`，actual_prefill3、
cached1638/repos3237；GPT20普通 **1 passed in 184.71s**，actual_prefill2、
cached470/repos2913，64/64 raw argmax与参考相同。

Agentic普通首次对mini冷算参考 **1 failed in 134.03s**：max2.359375超过原校准
上限2.0；mean0.075452/p990.390625在界内。没有放宽门限或修改生产计算。
`763a65af1`补充相同缓存路径的mini参考：同一fixture、首个R预热1token，第二个R
固定64tokens；mini原生mask/page-occurrence、默认FA、CUDA Graph保持不变。
`bcp778-mini-agentic-consecutive.json`记录mini HEAD2966eb49a和全部输入/阶段。
核实raw索引对应的active token、预热及目标forced token一致，目标有63次graph replay。

离线复用上述已保存SGLang logits，与同路径mini比较为
`1.8125/0.069156/0.34375`，64/64 raw argmax一致，**通过原门限**。
保留初次冷算参考失败结果及`agentic-normal/matched-consecutive-comparison.json`，
不把原pytest失败改写为通过。

`763a65af1` 的连续R PD补验：Qwen **1 passed in 228.25s**，Agentic
**1 passed in 246.15s**，P=GPU0/D=GPU1、各TP1；Qwen使用原参考，Agentic使用
上述同缓存路径参考。相对mini的max/mean/p99分别为
`1.0625/0.064690/0.28125`、`1.8125/0.069156/0.34375`，沿用原门限。
两者实际PF3/D63、cached1638/repos3237，无retract。
离线拼接P首步与D后63步，与普通调度已保存张量比较：两个模型均逐元素完全一致、
64/64 raw argmax一致；结果保存在各PD目录的`normal-pd-comparison.json`。
GPT20 PD随后 **1 passed in 301.14s**，默认Triton、共享Full/SWA池，
max/mean/p99为`1.375/0.062624/0.371094`，PF2/D63、cached470/repos2913，
无retract；普通与PD的64步logits也逐元素完全一致。六项连续R普通/PD补验完成，
矩阵`result.json`为valid；Agentic普通仍明确使用离线同路径参考的通过记录。
尚待120B PD和完整小量吞吐矩阵，R2未完成。

### 120B PD 最终数值验收

`d4b598572` 的 `bcp778-sg-pd-gpt120-capacity-lifetime-v1` 在BUS完成
`test_bcp_pd_numeric.py`：**1 passed in 434.94s**。P=GPU0/1 TP2、D=GPU2/3 TP2，
默认Triton、BF16、page1、共享Full/SWA池、CUDA Graph，P24576/D16384 KV、
context16384、chunk512；D Radix关闭，无D输出KV回传。
相对mini参考`bcp778-mini-gpt120-tp2-store.json`，max/mean/p99依次为：

| 路径 | max | mean | p99 | actual PF / D |
| --- | ---: | ---: | ---: | ---: |
| 无功能 | 6.09375 | 0.224397 | 1.339844 | 原生未报告 / 未报告 |
| Drop | 6.375 | 0.214703 | 1.292969 | 8927 / 63 |
| Repos冷 | 10.625 | 0.119297 | 0.988281 | 8927 / 63 |
| Repos热 | 10.625 | 0.121178 | 1.0 | 1 / 63 |
| Retry | 11.3203125 | 0.135114 | 1.09375 | 1746 / 63 |
| 连续Repos | 10.125 | 0.119956 | 0.90625 | 2 / 63 |

各项沿用无功能校准门限，未调整生产计算或容差。热命中cached3384；Retry为
cached432/repos3929/drop_skipped2820；连续R为cached470/repos2913，无retract。
离线拼接P首步与D后63步，对照普通调度最终运行：无功能、Drop、Repos冷、Retry、
连续R的logits均逐元素完全相同。热路径max/mean/p99为9.0/0.098700/1.03125，
raw argmax相同63/64；两端热路径各自均通过mini门限，不声称所有缓存路径逐位相同。
实际生成三种路径的普通/PD输出均完全一致；相对mini匹配数仍为43/62/64。
证据为本目录`comparison.json`、`normal-pd-comparison.json`及原始logits/响应。
至此必验模型普通/PD数值矩阵完成，PD完整轨迹性能配对与等待归因仍待验收。

### 普通调度 C2 最终完整轨迹

`d4b598572` 的 `minimal-sg120-c2-drop-parser-v5` 在 BUS GPU2/3 TP2完成：
4/4 task、150成功、0失败，其中143首遍、7 filler；47377输出tokens /
1469.288711s = **32.244854 token/s**。同配置原生
`minimal-native120-c2-none-parser-v1`为31.164463，提升 **3.4667%**，通过95%门槛。
结合上述C1提升6.8698%，普通调度两组性能门槛均通过；PD仍须单独验证。

沿mini原测速函数、以task完成时刻划分两轮，修改版每轮输出吞吐为
33.643632 / 30.151513，原生为31.831804 / 30.110349 token/s。
第二轮各含一个跨轮请求，保留原方法归属，不将各轮当作独立冷启动重复实验。
修改版实际PF454635/D47227，实际PF/D/All吞吐为
309.425232 / 32.142764 / 341.567996 token/s；原生缺少实际计算计数，保持缺失。
TTFT mean/P90为10351.176/14932.022ms；标准TPOT为27.887/46.365ms；
E2E为19221.116/29620.200ms，最大SSE chunk间隔16419.511ms，后者不是服务端TBT。

本次真实历史最大包含两个R。case864 turn28完成耗时16.5s、actual_prefill2587、
cached7733/repos81502/drop_skipped10942，未复现旧的空running队列停滞或整段冷重算。
turn24确有新的长tool输入，实际PF53075，不能将该次123.466s耗时归为同一重算缺陷。
测速中发生的约1s `_topk_forward` 编译时间保留在测量内，未事后扣除。

离线核对原生与修改版首遍case、源输出长度和数据集一致；实际生成历史按各自输出推进。
两组均无空assistant、非法工具参数JSON或替代字符。固定输出长度、ignore_eos下的
连续重复文本启发式分别标记47/47项，不作为答案质量判定；两组各一条截止取消filler
单列，不计150成功或0失败。原始结果在各运行`workload/result.json`、`events.jsonl`；
逐轮汇总为`r2-small-summary-20260918-0449.json`，输出审查为
`offline-output-audit-20260918-045133.json`。这些是BUS实验根下的文件，不随仓库提交。

## 延迟终态副本的 PD 传输边界（2026-09-18）

审计发现上述副本延迟生成后，旧 PD 仍按已计算 raw 长度发送，可能读取尚未生成的
终态 active KV。另一个异步风险是，尚未交给 Radix 的私有副本可能在后续发布时被
去重释放，不能在此之前作为未完成传输的源。

`d2ef1d0f1` 的 `ContextTransferPlan.full_chunk`
（`python/sglang/srt/disaggregation/context_transfer.py:54`）仅允许非末块发送连续、
已生成且已交给缓存持有的终态页；遇到延迟副本或私有副本时保留 raw 发送游标。
末块确认所有终态 active 页已存在，由原生 inflight 队列保护到发送完成。
`SchedulerDisaggregationPrefillMixin._send_kv_chunk` 的接线位于 `python/sglang/srt/disaggregation/prefill.py:1361`；
判断只读取 CPU occurrence 所有权，不增加 GPU 同步，也不改变普通请求传输。

本地 `python -m pytest -q test/registered/context_system/test_context_transfer.py`
**1 passed in 0.58s**，涵盖空洞、延迟副本、未发布私有页与最终发送不重不漏。
首次 `bcp778-sg-pd-gpt20-capacity-lifetime-v1` 在启动时遇到 ZMQ socket 路径超过107字节，
未进入模型请求；保留失败日志，以独立短临时目录重新启动。
真实 `bcp778-sg-pd-gpt20-capacity-lifetime-v2` 在 `d2ef1d0f1`、P GPU0、D GPU1、
独立 TP1 进程完成：`python -m pytest -s -q test/registered/context_system/test_bcp_pd_numeric.py`
为 **1 passed in 452.08s**（含启动，不是吞吐）。配置为 BF16、page1、默认 Triton、
共享 Full/SWA 物理池、P24576/D16384 KV、context16384、chunk512；D Radix 关闭。
复用 `bcp778-mini-gpt20-native-input.json` 的相同 token 路径与保存的 mini logits。

| PD 路径 | max / mean / p99 绝对误差（相对 mini） | 实际 Prefill / Decode tokens |
| --- | --- | --- |
| 无功能 | 2.710938 / 0.069612 / 0.437500 | 原生计数未提供 |
| Drop 冷算 | 4.875000 / 0.101741 / 0.875000 | 8927 / 63 |
| Drop+R 冷算 | 2.279297 / 0.064363 / 0.412109 | 8927 / 63 |
| Drop+R 热命中 | 2.279297 / 0.064380 / 0.410156 | 1 / 63 |
| Drop+R Retry | 1.265625 / 0.057951 / 0.343750 | 1746 / 63 |

三条自由生成路径均为 64/64 tokens 与 mini 相同。热命中 cached3384、repos0、
drop-skipped0；Retry cached432、repos3929、drop-skipped2820。Drop-skipped0 的
物理驻留口径同前述普通调度说明；不能用逻辑空洞增加该计数。

读取已保存的普通/PD logits 做离线直接对照，无额外模型 forward：无功能、Drop 冷算、
Drop+R 冷算及 Retry 的张量逐元素完全一致；热命中 max/mean/p99 为
0.9375/0.030352/0.21875，64/64 argmax 一致。原始结果为该 PD 目录下
`comparison.json`、`normal-pd-comparison.json` 及逐路径 `.pt`。
本次覆盖稳定终态传输、冷算、热命中和 Retry，不替代 120B PD 与完整轨迹吞吐门槛。

## 最终 Qwen / Agentic BCP 三方对照（2026-09-18）

`d2ef1d0f1` 在 BUS 的 `final-qwen-bcp-lifetime-v1` 使用相同 BCP 输入和已保存的 mini
默认 logits；未额外运行 mini forward。BF16、page1、Triton、context16384、chunk512，
普通 KV24576；PD P24576/D16384、P GPU0 / D GPU1，D Radix关闭。普通 Qwen GPU0、
Agentic GPU1；PD在普通检查结束后顺序执行，CUDA Graph保持开启。

命令分别为 `python -m pytest -s -q test/registered/context_system/test_bcp_numeric.py`
及 `test_bcp_pd_numeric.py`，通过时间包含启动：Qwen普通 **1 passed in 133.58s**、
PD **1 passed in 238.89s**；Agentic普通 **1 passed in 157.65s**、
PD **1 passed in 270.71s**。Agentic Retry沿用已保存的匹配状态 mini Retry参考，
来源记录在 `reference-sources.json`。

| 模型 / 路径 | 普通及 PD 相对 mini 的 max / mean / p99 |
| --- | --- |
| Qwen 无功能 | 1.125 / 0.054271 / 0.218750 |
| Qwen Drop | 0.5625 / 0.049975 / 0.187500 |
| Qwen Drop+R 冷 | 1.125 / 0.065057 / 0.312500 |
| Qwen Retry | 0.625 / 0.050695 / 0.187500 |
| Agentic 无功能 | 1.000 / 0.054687 / 0.312500 |
| Agentic Drop | 0.968750 / 0.045616 / 0.218750 |
| Agentic Drop+R 冷 | 1.578125 / 0.068580 / 0.312500 |
| Agentic Retry | 1.339844 / 0.068650 / 0.359375 |

热命中相对 mini：Qwen普通 `1.046875/0.064866/0.296875`、
PD `1.125/0.065100/0.312500`；Agentic普通 `1.101562/0.059618/0.271484`、
PD `1.578125/0.068670/0.312500`。全部通过既有同模型无功能误差校准。
两模型的热命中均实际PF1/D63、cached4877/repos0/drop-skipped0；
Retry均PF2002/D63、cached1508/repos4122/drop-skipped2850。

离线读取已保存张量直接比较普通与PD：两模型无功能、Drop冷、R冷、Retry均逐元素
一致；热命中Qwen `0.5625/0.043355/0.15625`，Agentic `0.65625/0.049008/0.1875`，
均64/64 raw argmax一致。各三条实际生成路径普通与PD均64/64 tokens相同。
与mini的自由生成仍有此前已定位的原生语法/采样行为差异：Qwen逐位置相同数为
1/1/5，Agentic为1/2/1（无功能/Drop/R，各64 tokens）。不能将固定token通过或
普通与PD一致改写成自由生成与mini完全相同；原始消息、首次分歧与数值证据保留。
各目录 `comparison.json`、`normal-pd-comparison.json` 为本次结果。

## 当前实测状态（2026-09-17）

2026-09-18 追加检查点（整体 R2 仍未完成）：

- `4c3993038` 将 Context attention 的打包元数据字段按 16 字节对齐，避免相同语义
  因指针余数变化产生多组 Triton 编译。BUS GPU0 执行
  `RUN_CONTEXT_GPU=1 python -m pytest -q test/registered/context_system/test_segmented_attention.py`：
  **4 passed in 23.86s**，覆盖 FP16/BF16、full/SWA+sinks 和混合请求独立对照。
- `44ed3e215` 修复初始 Retry 匹配锁住自己的来源页、导致最小 chunk 永久无法进入的问题。
  临时锁释放后下一次匹配退回 root；外部请求造成的暂时压力继续等待；冷匹配仍无法容纳
  时通过现有 503 清理路径拒绝。本地 raw-storage 的容量子集 **2 passed**；BUS CPU 执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_admission.py`：
  **10 passed in 17.18s**。编号 9 为不可见设备，未使用 GPU4+。首次设置空的可见设备字符串
  触发原生 test_utils 端口读取错误（3 passed / 7 setup errors），原始日志保留。
  后续 chunk 的完整峰值容量保证仍待审计，不能据此宣称所有容量边界通过。
- GPT-OSS-120B、TP2、完整 BCP：原生 C1 无功能 84/0 成功/失败，1068.732s，
  输出 27390 tokens，25.6285 token/s；修改版 `1929c115c` C1 Drop+R 84/0，1030.028s，
  同样输出 27390，26.5915 token/s。该单样本满足 95% 门槛，尚不是最终性能结论。
  原生 C2 无功能 4/4 task，147/0 请求，其中首次轨迹 143、filler 4，1497.176s，
  输出 46597 tokens，31.1233 token/s；原生没有精确物理计算计数，未估造。
- 修改版 C2 无功能 `minimal-sg120-c2-none-native-ar-v1` 在原生 Harmony parser 卡住。
  CPU 栈显示 scheduler、detokenizer 已空闲，HTTP 主线程停在 `HarmonyParser.parse` 的
  无锚点空白正则；1596 输出 tokens 时的 delta 为连续换行。现场记录
  `sg-c2-none-parser-stall-20260918.txt`，该运行主动停止并排除吞吐验收。
  修复仅将 header 搜索改为等价的非空白边界，并消除 fallback 的重叠空白量词；保留原生
  解析事件与采样行为。新测试覆盖 120000 字符空白、跨 chunk header、1500 个原生匹配
  对照；另有 10000 个分块输入与修改前解析事件和剩余 buffer 完全一致。新测试
  **2 passed**。完整 parser Ruff 的 20 项既有诊断无新增，新测试 Ruff 通过。

追加核验（2026-09-18，以下均不表示整体 R2 已验收）：

- Harmony parser 修复 `652810595` 在 BUS 执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/unit/parser/test_harmony_parser.py test/registered/context_system/test_harmony_whitespace.py`：
  **45 passed in 16.63s**。原生对照 `d7a3df66a` 应用同一修复；新 C2 运行使用此配对口径。
- 首次 GPT-OSS-20B shared-KV 普通调度复测失败是输入日期不同：mini oracle 为 9 月 17 日，
  SGLang 模板为 9 月 18 日，首个差异的索引为 35。`441a55e07` 的 fixture
  从 oracle 提取 `Current date`，仅固定测试模板的日期，再逐请求检查
  input token IDs。没有放宽数值阈值，也没有改生产模板。PD fixture 在 `f63091f59`
  支持独立长历史测试不提供 oracle 的情况。
- 修正日期后，`f63091f59`、GPT-OSS-20B、GPU0/TP1、原生默认 Triton、共享全层 KV、
  page1、chunk512、CUDA Graph 的 `test_bcp_numeric.py` 为 **1 passed in 346.67s**。
  目录 `bcp778-sg-gpt20-shared-oracle-date-v1`。无功能、Drop、Drop+R 实际生成均与
  mini 的 64 个 token 完全一致，输入也一致；固定路径 max/mean/p99 分别为：
  无功能 `2.7109375/0.06961208/0.4375`，Drop `4.875/0.10174099/0.875`，
  R 冷 `2.279296875/0.06436337/0.412109375`，
  R 热 `1.734375/0.06231369/0.390625`，Retry `1.265625/0.05795066/0.34375`。
  各路径 argmax 均为 64/64；均通过同模型无功能误差校准。
  R 热 cached/repos/drop-skipped 为 `3384/0/5542`，实际 prefill/decode 为 `1/63`；
  Retry 为 `432/3929/2820` 与 `1746/63`。
- `b9d295063` 补齐已开始的 chunk 被自己持有的页永久阻塞的处理。真实 CPU occurrence
  规划反例：raw100、Drop80→[0,40)、R98、pool110、chunk32，在 query74 已持有108页，
  下一步至少再需3页。现在仅在容量失败分支按 CPU 所有权映射去重物理页，确认单请求
  无法推进时走原生 pending-chunked-abort，普通与 PD 均保留 503 原因并释放 KV、锁、
  sender 与传输元数据。外部请求造成的暂时压力继续等待。
  本地 `test_context_raw_storage.py -k 'capacity or self_pin'`：**3 passed**；BUS 执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_admission.py test/registered/context_system/test_context_raw_storage.py -k 'not cuda'`：
  **17 passed, 1 skipped in 16.48s**。GPU 行存储用例本轮跳过；503 清理使用定向 mock，
  尚未新增真实模型的容量拒绝实验。该计数是自有映射的下界，不证明所有独立 source lease
  的容量边界都已覆盖。不会把这个低频防停滞修复的 CPU 结果当作 GPU 吞吐结果。
- 已保存输出离线审计 `offline-output-audit-20260918-010858.json`。固定长输出预算下，
  原生和修改版均存在重复内容；简单重复检测不等于模型质量验收，也不能将每次重复
  直接归因为 Drop/Reposition。必须结合原始输出、首次分歧、数值对照及 finish reason。

追加检查点 `9746a7881`：

- 对 Retry 独立 source lease 补充容量计数。真实 CPU UnifiedRadix/allocator 反例：
  source70、target50、共享24、私有30、pool128，实际保留126页、空闲2页、最小 forward
  还需3页；旧计数只看到当前状态的30页，未拒绝。现在
  `request_storage.py:11` 的 `handle_prefill_capacity_pressure` 在已有 source lease 时调用
  `unified_radix_cache.py:965` 的 `context_leased_page_count`，仅沿 target/source 两条
  CPU 树路径去重，尊重 Drop path-only receipt 与已淘汰的 hole，再加私有页；不做
  GPU→CPU 页号读取，不扫描整棵树，也不进入正常成功调度的热路径。
- BUS capacity-check 已通过 Git fast-forward 到该提交。执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_drop_eviction.py -k retry_source_capacity`：
  **1 passed, 29 deselected in 14.53s**。用例还验证共享祖先不重复计数、外部读者占用不误拒绝、
  Drop path-only 不占本请求页额度、hole 不计数以及最后128页全部归还。这个测试使用真实
  CPU 树与分配器，但 occurrence state 为构造数据，不是实际模型的 GPU 停滞复现。
- 同提交执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_admission.py test/registered/context_system/test_context_raw_storage.py -k 'capacity or self_pin'`：
  **10 passed, 8 deselected in 16.39s**。本机 Anaconda 环境同类运行为1 passed/2 setup errors，
  原因是缺少 `tvm_ffi`；BUS 完整环境的结果如上。未为这个只改失败分支的计数再跑完整 GPU 矩阵。
- 普通调度修改版 `652810595` 的 C2 无功能配对运行
  `minimal-sg120-c2-none-parser-v1` 已完成：4/4首次 task、150成功/0失败，47377输出 tokens，
  1516.924459s，**31.232274 token/s**；逻辑输入8137716 tokens。
  精确物理 prefill/decode 计数缺失，保持 null。该项是修改版无功能对照；对应原生与
  Drop+R 配对运行仍在执行，暂不据此判断5%门槛。

追加检查点 `75a1eb49c`：

- C2 Drop 的 `minimal-sg120-c2-drop-parser-v1` 在 warmup 混合 prefill/decode 时失败，
  没有进入正式 BCP 工作负载。堆栈定位到 `ContextPrefillInput.concatenate` 仍读取旧的
  `field_offsets`；16字节对齐后的实际数据结构已经改为不含 padding 的 `field_ranges`。
  现使用真实字段区间拼接，保留对齐与原生混合调度；不关闭 mixed chunk 或 CUDA Graph。
- BUS capacity-check 修复前执行已有
  `test_context_decode.py -k mixed_extend`：**1 failed, 5 deselected in 2.94s**，
  与真实启动相同的 AttributeError。Git fast-forward 后执行
  `CUDA_VISIBLE_DEVICES=9 PYTHONPATH=python python -m pytest -q test/registered/context_system/test_context_decode.py -k 'not gather'`：
  **4 passed, 2 deselected in 2.07s**。这组检查验证 mixed query positions、prefix positions、
  物理 KV 索引、query/KV indptr 和 decode 工作量计数；GPU graph gather 未在本轮重复运行。
  本地默认 Anaconda 无 torch，收集失败；本地 py_compile 与 diff-check 通过。
- 恢复 C2 Drop 为 `minimal-sg120-c2-drop-parser-v2`，后续 C1 无功能及 PD 队列继续。
  已完成 C2 无功能不重复跑；失败的 warmup 不进入任何吞吐平均值。

原生配对 C2 `minimal-native120-c2-none-parser-v1` 已完成（`d7a3df66a`，共享同一
Harmony parser 兼容修复）：4/4首次 task，150成功/0失败，1520.225151s，47377输出 tokens，
**31.164463 token/s**。TTFT mean/P90 为8953.958/13943.135 ms；标准 TPOT mean/P90
为31.6765/58.0744 ms；实际物理计算计数缺失。对应修改版无功能为31.232274 token/s，
相差+0.22%，这只验证普通无功能对照，不能据此判断 Drop+Repos 的5%门槛。
原始结果在 BUS 实验根的相应 `workload/result.json`；C2 Drop+Repos 仍在执行。

追加真实容量拒绝验证 `0a005af39`（只新增测试，生产代码仍为 `75a1eb49c`）：

- BUS 独立 `sglang-r2-pressure-check` checkout、GPU0、Qwen3-0.6B BF16/Triton、
  page1、KV110、chunk32、CUDA Graph 与 mixed chunk。执行
  `CONTEXT_CAPACITY_PRESSURE=1 CONTEXT_KV_CAPACITY=110 CONTEXT_CHUNK_SIZE=32 CONTEXT_MAX_LENGTH=2048 python -m pytest -s -q test/registered/context_system/test_serving_runtime.py -k capacity_rejection`，
  同时指定 `CONTEXT_SERVER_MODEL`、端口29901、独立日志/cache 与 CUDA 兼容库环境。
  结果 **1 passed, 2 deselected in 119.32s**（包含模型启动）。
- raw100、Drop80→[0,40)、R98 在模型实际执行中触发 continuation 容量边界；
  **2.261432s** 返回 HTTP503，原因是保留108个KV、至少还需3个、总池110。
  随后 `/flush_cache` 成功，普通90-token输入完成8-token生成，HTTP200，health正常。
  容量失败前后都实际 replay prefill CUDA Graph。此用例不重复完整数值矩阵。
- 原始证据：`capacity-http-qwen-v1/{pytest.log,server.log,result.json}`；测试结束后
  其自有服务进程退出，GPU0回到36MiB。该证据覆盖普通调度真实拒绝与恢复，不自动
  扩展为PD容量拒绝或所有source lease压力场景已验证。

mini System/main 的75份逐文件 diff 与清单内52份历史会话均已通读；这不等于
所有迁移功能已通过。补充差异与效率约束：

- mini `scheduler/prefill.py` 的 occurrence 规划先试最大 chunk，再用预计算容量曲线
  O(1) 判断候选；包括临时页、持久页与未来预留。同位置 Retry 的终态副本仍需独立所有权。
  容量失败须区分自身页使推进不可能与外部读者暂时占用；chunk 完成只归还已无后续读者
  的临时/birth 页。mini 部分 chunk 会阻止启动第二个 partial occurrence；不能照搬为
  SGLang 的全局串行化约束。
- mini 的旧 staged warmup、0.95 reuse 阈值不是 mask/page-occurrence 的生产设置。
  本次迁移只做一次正式请求；SGLang 保留原生 tokenizer/model/MoE/parser。
- mini `drop_rules.py` 保留部分匹配跨边界 token；整消息删除包含该消息模板归属。
  KeepText 使用协议元数据与最右有序匹配；不得退化为只按字符串搜索。
- mini System 的 Harmony 模板会处理旧 analysis；最终 System-test `2966eb49a` 的
  stable history 是显式 opt-in。benchmark 与数值对照必须记录实际模板和输入 IDs，
  不以旧分支默认值替代最后批准的 setting。
- mini SWA 的位置空洞路径依据绝对位置，可见 KV 可能少于 compact 序列末128项；
  连续尾部覆盖窗口后可恢复原生快速路径。已有 attention dense 对照只验证 kernel，
  全模型数值依据仍来自真实 mini 固定路径。

完整读取的 mini System/main diff 已逐项标在 `source_inventory.json`，与“端到端迁移验收”
分开计数。attention adapter 的额外差异：mini FA/FI 在带位置空洞的 SWA decode 上避开
普通 CUDA Graph，并禁止混合 masked/普通请求；迁移保留可见集与 shared all-layer KV
语义，不能把这两个调度限制搬入 SGLang。mini FA3/FI FA2 的 sinks 通过 LSE 补偿实现，
SGLang 使用原生默认 attention 的 sinks；不能叠加第二次补偿。mini 的 pinned staging
与按消费 stream event 管理复用，是防止 H2D CPU 停顿和异步覆写的重要参考；SGLang
需按其实际 tensor 生命周期验证，不能只凭 `non_blocking=True` 宣称无阻塞。

以上日志均位于前述 BUS 实验根目录；双并发候选与 PD 小量吞吐仍需补齐。对比时必须
注明 parser 等兼容修复是否同时应用于原生对照，不把修复原生卡顿带来的收益宣称为
Drop/Reposition 本身的性能收益。

普通调度的终态发布修复为 `feff22efd`，首轮真实 PD 接线为 `feb289a24`。
`760592534` 进一步修复了 Context 的原生 host-pool 备份读集、SWA 回收和 PD 双池容量
估算。`ec321ae8f` 的真实并发 retract 验证已通过；`83cfdf638` 恢复 P 分块传输后，
同一 BCP 的六条冷/热/Retry 固定路径也通过。以下结果只覆盖 Qwen3-0.6B，不能推断
AgenticQwen、GPT-OSS 或吞吐验收已经通过。

BCP case 778 使用同一规范化原生工具描述、原始 10482 tokens、终态 active 4878 tokens。
mini 固定版本默认 mask/page-occurrence、BF16、FA3；SGLang 使用 BF16/Triton，保留
原生 overlap、CUDA Graph 和 512-token chunk。对齐的是同一模型、权重、输入和固定
64-token 路径；后端不同引入的误差由无功能运行校准。

| 路径 | 普通 SGLang max / mean / p99 | P/D max / mean / p99 |
| --- | --- | --- |
| 无功能 | 1.125 / 0.054271 / 0.21875 | 1.125 / 0.054271 / 0.21875 |
| Drop | 0.5625 / 0.049975 / 0.1875 | 0.5625 / 0.049975 / 0.1875 |
| Drop+R 冷 | 1.125 / 0.065057 / 0.3125 | 1.125 / 0.065057 / 0.3125 |
| Drop+R 热 | 1.046875 / 0.064866 / 0.296875 | 1.125 / 0.065100 / 0.3125 |
| Drop+R Retry | 0.625 / 0.050695 / 0.1875 | 0.625 / 0.050695 / 0.1875 |

每项门限是 `max(绝对下限, 2 × 同模型无功能误差)`；max/mean/p99 下限分别为
0.125/0.02/0.0625。这是本次跨后端 BF16 的诊断标准，不是所有模型的通用误差标准。
R 热命中 cached=4877、repos=0、drop_skipped=5604、实际 prefill=1；Retry 分别为
1508、4122、2850、2002。两种调度均为 63 次实际 decode 加 P/普通 prefill 的首 token。

普通调度先前存在首 decode 使用旧位置 KV 的真实错误（R cold max=16.6875、Retry
max=17.9375）。`ScheduleBatch._prepare_context_occurrences`
（`python/sglang/srt/managers/schedule_batch.py:3232`）在原生 overlap 消费前发布最终
raw→terminal 映射，保留独立 birth 来源，修复后的结果如上。

P/D 的 P、D 分别位于 BUS 物理 GPU 0、1，独立进程，D Radix 关闭，无 D→P 输出传输。
`ContextTransferPlan` 对最终 active 行、位置和事件身份做一致性校验；D 只分配实际
active Full 和必要 SWA 页（`disaggregation/context_transfer.py:124`
`allocate_context_destination`、`:181` `commit_context_metadata`，路径前缀为
`python/sglang/srt/`）。`83cfdf638` 的 P 按完成的 raw chunk 发送对应的终态 active
Full KV（`disaggregation/context_transfer.py:53` `full_chunk`、
`disaggregation/prefill.py:1325` `_send_kv_chunk`），跨 Drop holes 后保持连续目标次序。
请求的原生锁保护已发布终态，未来 birth 页独立写入；保留最后一页随最终元数据/SWA
一同发送，避免传输层凭累计页数提前宣布完成。该版本 BCP 数值为表中 PD 结果，
`1 passed in 144.06s`，含启动时间，不能用作吞吐或传输重叠收益测量。

原生 host-pool retract 验证为 `1 passed in 158.31s`：两个并发 BCP 中一条实际发生
4 次回收恢复，64 个固定输出 token 与 mini 一致；该条 max/mean/p99 为
1.0625/0.068561/0.3125，落在同一无功能基线门限内。实际 decode=67，包含原生 overlap
被丢弃后重放的 4 次 forward，不能误记成 63。初次实验因网络到达错开未触发 retract，
后续仅在测试 probe 中等两条传输就绪；第二次暴露 probe 未回退被丢弃采样的问题，
`ec321ae8f` 令 probe 随原生回收回退到已提交 output 长度，未替换生产 sampler。

BUS 安装的是当前 SGLang Docker 配置对应的 Mooncake CUDA13 0.3.13，实际日志显示
未发现 HCA，使用 TCP 传输。本次证明真实跨 GPU KV 传输的功能，不能视为 RDMA 性能。
首轮 PD 测试为 `1 passed in 190.49s`（含启动，不是吞吐）；普通数值测试为
`1 passed in 85.80s`。结果目录在 BUS 的
`/mnt/public/wangruoxi/local/sglang-system-20260917/`：

- `bcp778-mini-aligned-qwen.json` 和同名 `.json.pt`：复用的 mini 数值参考。
- `bcp778-sg-aligned-qwen/`：普通调度固定路径结果；旧实际生成记录注明为复用。
- `bcp778-sg-pd-qwen/`：P/D 原始响应、分端 logits、comparison.json 和启动日志。
- `bcp778-sg-pd-retract-qwen/`：首次未触发回收的失败证据，保留。
- `bcp778-sg-pd-retract-barrier-qwen/`：probe 未回退导致固定路径跳 token 的失败证据。
- `bcp778-sg-pd-retract-rewind-qwen/`：`ec321ae8f`，4 次真实回收恢复通过。
- `bcp778-sg-pd-chunked-qwen/`：`83cfdf638`，分块传输后的完整 PD BCP 数值矩阵。

可复现的 PD 测试入口：

```bash
CONTEXT_PD_BCP_ORACLE="$EXP/bcp778-mini-aligned-qwen.json" \
CONTEXT_TRACE_DIR="$EXP/bcp778-sg-pd-qwen" \
CONTEXT_SERVER_MODEL="$MODEL_PATH" CONTEXT_TEST_ATTENTION_BACKEND=triton \
CONTEXT_P_GPU=0 CONTEXT_D_GPU=1 \
python -m pytest -s -q test/registered/context_system/test_bcp_pd_numeric.py
```

这里的 EXP 是上述 BUS 实验目录，MODEL_PATH 是 Qwen3-0.6B 的本地权重路径。
需使用已记录的隔离 CUDA13 环境及 compat 库路径；不可用同名输出目录覆盖既有结果。
retract 实验额外设置 `CONTEXT_PD_RETRACT=1` 和指向首轮 comparison.json 的
`CONTEXT_PD_CALIBRATION`，复用无功能校准，只新增两个并发 BCP 请求的抢占恢复检查。
P/D 的 cache、临时目录、端口、日志和 GPU 分开。

实际生成尚未验收：此 BCP 下 mini Qwen3-0.6B 默认工具语法首先输出终止符，但
`ignore_eos` 继续排满 64 tokens；SGLang 的同描述原生结构化语法生成两个 tokens 后
正常终止。不能为凑齐长度替换 SGLang sampler，也不能把固定路径误差通过当作实际
Agentic 回答通过。`9a2bfc811` 将实际生成与固定路径验证拆开，支持
`CONTEXT_BCP_MODE=actual|fixed|all`，自由生成不注入 logit processor，也不读取旧输出
充作本次结果。该版本对三种功能各做一次新实际生成，均得到相同的两个 tokens；
`bcp778-sg-native-actual-qwen/` 保存原始响应与 mini 文本/结束原因对照，执行测试为
`1 passed in 70.77s`，其含义是观察记录完整，不是回答质量或 R2 验收通过。
源码确认 SGLang `managers/schedule_batch.py:2186` 对已终止 grammar 单独结束请求，
不受 `ignore_eos` 控制；mini `engine/tool_grammar.py:180` `_accept` 则在终止后保留
上一份 mask。迁移没有为了凑长度修改这两个原生语法行为。需在必验大模型上完成
实际生成对照，并据其原生协议记录真实终止原因。

GPT-OSS 的模型配置都把 window=128 转为左距离 127，但这不足以证明完整
SWA 路径一致。继续追踪发现冻结版原生 Triton decode 的
`layers/attention/triton_backend.py:2660` `update_sliding_window_buffer` 使用
`min(seq_len, sliding_window_size)`，长序列实际取 127 个 KV；mini 的
`attention/fa.py:137` 则将 127 作为左距离，包含当前 token 时共 128 个。
SGLang 引用前缀为 `python/sglang/srt/`，mini 为 `python/minisgl/`。
此前仅根据模型 getter 认定无窗口差异的结论已更正。

迁移的 Context decode 使用 `python/sglang/kernels/ops/attention/context_page_table.py:8`
`context_window_lengths`，按 `key_position >= query_position - 127` 搜索，
与 mini 一样最多保留 128 个连续位置；Drop 空洞不能用更早 KV 补齐。Context prefill
在原生 extend kernel 内同样按真实位置施加窗口。普通无功能 lane 保留原生行为，
不会因同批存在 Context 请求而改变窗口。当前没有修改原生模型或普通 attention 的
窗口定义；这是跨系统无功能校准需要记录的计算差异，不能全部归为浮点舍入。

GPU 资源阻塞已解除：按用户随后授予的 GPU 0–3 全部任务释放权限，核实宿主进程
归属后停止了前四卡的占用任务。GPU 4–7 的其他训练保留；特权会话已关闭。当前必验
模型实验显式分配 0–3，记录独立进程、输出、端口及实际 HEAD。后续只对新修改和未覆盖
功能运行必要测试，不重复完整旧测试集。

本次分块区间 CPU 测试 `python -m pytest -q
test/registered/context_system/test_context_transfer.py` 为 1 passed。新增 helper 和
BCP 测试的 Ruff 通过；完整 `prefill.py` 的 Ruff 在修改前后均有 29 项相同既有
诊断，无新增项，未为此格式化无关原生代码。此处早期状态只完成了 52 条会话索引。后续已补齐清单内 52/52 会话通读和
System 相对 main 的 75/75 文件完整 diff 阅读；System-test 额外差异仍按清单逐项核对。
阅读完成不等于功能、数值或效率验收通过，本文继续保留每项实际验证与未决风险。

## 追加实测与 SWA 运行设置（2026-09-17）

AgenticQwen-8B 的普通与 PD 固定路径均已执行。Retry 必须与 mini 的同源 Retry 对照：
mini 自身 Retry 相对冷算已有差异，不能将两种不同计算路径混为误差基线。使用已保存的
mini Retry logits 复核后，普通及 PD 的 max/mean/p99 为
1.33984375/0.068649568/0.359375，低于同模型无功能校准门限；没有重跑 GPU 来生成
重复证据。原始与匹配对照保存在 `bcp778-mini-agentic-retry.json`、
`bcp778-agentic-matched-retry-comparison.json`。实际生成的原生语法结束行为仍单独记录。

GPT-OSS-20B 普通调度在 `4a726e827` 的 BCP 验证通过，测试耗时 361.42 秒（含启动和
logits 诊断，不是吞吐）。none/Drop/Drop+R 冷算的实际输出各 64 tokens 均与 mini 相同。
输入先对齐到 SGLang 原生 Harmony 模板的 8927 tokens，mini 仅在测试 renderer 中
使用同一输入；这证明计算对齐，不证明两个服务端的默认模板相同。

| GPT-OSS-20B 路径 | max / mean / p99 |
| --- | --- |
| none | 2.7109375 / 0.069612078 / 0.4375 |
| Drop | 4.875 / 0.101740994 / 0.875 |
| Drop+R 冷 | 2.279296875 / 0.064363368 / 0.412109375 |
| Drop+R 热 | 1.734375 / 0.062313691 / 0.390625 |
| 同源 Retry | 1.8125 / 0.060070530 / 0.359375 |

各路径 raw argmax 均 64/64 一致。热路径 cached/repos/skipped/prefill 为
3384/0/5542/1，同源 Retry 为 3252/3916/0/1759。结果位于
`bcp778-sg-gpt20-aligned/`，参考为 `bcp778-mini-gpt20-native-input.json`。
以上是独立 SWA 池旧配置的证据；下述新配置正在补齐验证，不能直接沿用为其验收结果。

用户要求 SWA+Drop+R 使用 mini 机制。mini 的 `python/minisgl/kvcache/mha_pool.py`
`MHAKVCache` 将所有层绑定在同一 token page；SGLang 的独立 SWA pool 会回收窗口外
SWA 同伴页，未来 Retry 可能多做修复。R2 运行配置从 `342daa299` 开始对 GPT-OSS
普通、P、D 显式传入原生 `--disable-hybrid-swa-memory`。它通过
`configs/model_config.py` 的 `ModelConfig._derive_hybrid_model` 关闭独立 SWA pool，
**不关闭 SWA attention**：原生 GPT-OSS 层的窗口、sinks、MoE 和默认 Triton backend
全部保留。mask/page-occurrence 按 query 所在阶段的 true position 选择可见版本，
Retry 与正常 Drop 共用 Full/SWA 页生命周期，来源页只读、目标 RoPE 旋转使用独立页。
生产无功能请求仍走原生计算分支。只开 `--context-drop-aware-eviction` 不会隐式改变
SGLang 的物理池；部署时必须同时采用这里的 SWA 配置。

这一配置使 P/D 均使用同构全层 KV 页，传输最终 active 页；D Radix 仍关闭，无 D→P
输出 KV。它增加了相对独立 SWA 尾窗口传输的字节数，不能宣称已保留该优化的带宽收益。
最小吞吐的原版 SGLang 对照也采用相同物理池设置，避免把不同 KV 容量当成补丁开销。
原生独立 SWA 池仍保留，但不作为本次 mini 等价路径的完成依据。

PD 旧双池测试暴露两个问题：page_size=1 的普通长 prompt 错受 D 的 SWA 小池限制
（8927 > 2052），`ebbd491d9` 增加页大小为一的尾窗口分配；随后已完成请求在 overlap
队列中尚未过滤、Context layout 已释放，admission 访问空 layout 导致 D 退出。
`2c24bf6b9` 将无 KV 所有权的请求排除出回收预算，测试增加不注入 probe 的 PD 实际
生成对照。失败日志 `bcp778-sg-pd-gpt20-tail/` 保留，不能算作 PD 完整通过。

GPT-OSS-120B 的 mini TP2 固定参考已生成，路径为
`bcp778-mini-gpt120-tp2-store.json`；测试 helper 让 mini 自己持有 rendezvous store，
避免 torchrun 的 agent-store 环境令所有 rank 同时等待一个不存在的服务端。
共享 SWA 配置的 GPT-OSS-120B 普通/PD 与 GPT-OSS-20B PD 结果如下；吞吐尚未验收。

BCP 吞吐 helper `benchmark/context_system/run_minimal.py` 保留完整任务、原输出长度、
seed 42、C1/N2 与 C2/N4、K12/96Ki，不测 SLO。`a9e7970d9` 将原生 SG client 的
journal 改为与 mini 相同的异步 FIFO 写入；计时前用无关合成输入单独预热，记录在
`warmup.json`，不预热被测任务。mini 没有 HTTP cache flush，因此各引擎均保留这段
无关前缀，并如实记录，不能声称测试从物理空缓存开始。仍需检查测量区间有无新 JIT。

## 共享 SWA、长历史和思考历史追加证据

以下均使用 BUS 的原生 GPT-OSS 默认 Triton attention、page_size=1 和
`--disable-hybrid-swa-memory`。普通 120B 为 TP2；PD 120B 为 P GPU0/1 TP2、
D GPU2/3 TP2，D Radix 关闭，无 D→P 输出 KV。固定路径仍与同输入 8927 tokens
的 mini 参考比较，门限沿用上文同模型无功能校准。

| 路径 | 120B 普通 max / mean / p99 | 120B PD max / mean / p99 | 20B PD max / mean / p99 |
| --- | --- | --- | --- |
| none | 6.09375 / 0.224397 / 1.339844 | 6.09375 / 0.224397 / 1.339844 | 2.710938 / 0.069612 / 0.4375 |
| Drop | 6.375 / 0.214703 / 1.292969 | 6.375 / 0.214703 / 1.292969 | 4.875 / 0.101741 / 0.875 |
| Drop+R 冷 | 10.625 / 0.119297 / 0.988281 | 10.625 / 0.119297 / 0.988281 | 2.279297 / 0.064363 / 0.412109 |
| Drop+R 热 | 6.625 / 0.129831 / 1.0625 | 10.625 / 0.121178 / 1 | 2.279297 / 0.064380 / 0.410156 |
| 同源 Retry | 11.320313 / 0.135114 / 1.09375 | 11.320313 / 0.135114 / 1.09375 | 1.265625 / 0.057951 / 0.34375 |

三个测试分别为 `1 passed in 446.83s`、`1 passed in 683.43s`、
`1 passed in 428.48s`。这是含启动和 logits 采集的测试耗时，不能当作吞吐。
结果目录依次为 `bcp778-sg-gpt120-tp2-shared-v3/`（`b6e3679d6`）、
`bcp778-sg-pd-gpt120-shared-v1/`（`c773554a0`）、
`bcp778-sg-pd-gpt20-shared-v3/`（`b6e3679d6`）。各目录的 `comparison.json`
保留逐 token 误差、实际输出和 usage。热命中 cached/repos/skipped/prefill 为
3384/0/5542/1；同源 Retry 为 432/3929/2820/1746；实际 decode 均 63。
这些 PD 请求没有触发 retract，不代替前述专门的回收验证。

20B 三条实际生成与 mini 的 64 tokens 全部相同。120B 普通与 PD 实际输出彼此相同，
对 mini 的 none/Drop/Drop+R 分别有 43/62/64 tokens 相同；无功能路径已有生成分歧。
固定路径误差通过不能证明自由生成逐字相同。这些 64-token 结果只是 reasoning 片段，
完整 Agentic 输出质量仍由完整 BCP 轨迹另行检查。

`dacf9ceaa` 修复 raw 历史超过原生请求表宽度时的存储：
`python/sglang/srt/context_system/request_storage.py:18-54`
`prepare_request_row` 为溢出的 Context 请求分配 int32 页号行，并更新稳定的设备行指针；
原生 Graph 捕获的表不扩容。`write_request_slots`（同文件 `:57-80`）批量写混合行；
`validate_positions`（`:83-105`）分别检查 birth、迁移来源/目标、终态位置，
不再把 raw 长度误当作模型位置。这里新增的是索引存储，不是全历史 KV 保留。

BUS GPU3 的 `test_context_long_history.py` 在 GPT-OSS-20B 共享 SWA 下通过
`1 passed in 315.94s`：raw 4668，五次 Drop+R，原生表宽 8192 与 2048 的冷、热
路径各 8 个固定 tokens/logits 完全一致，热路径 Drop-skipped 超过 4000。
证据为 `long-raw-gpt20-dacf9ceaa/comparison.json` 与两份服务器日志。
另有针对行指针、CUDA Graph 和 admission 的 13 项 GPU 检查通过；这不等同于
raw>128K 的完整 PD 压力已经通过。

`a515fdaa3` / `f45b968b2` 补齐思考历史保留。原生 Qwen/GPT 模板会省略一部分历史
reasoning；请求要求 ThinkingDrop 时，必须先确实保留对应文本，才能按准确 token
来源删除。`context_system/thinking_template.py:14-59` 的 `retained_template`
仅调整已识别模板的省略条件，`:62-80` 的 `prepare_thinking_history` 使用请求私有视图，
不修改 tokenizer 全局状态。GPT 同时接受 OpenAI `reasoning_content`；冲突字段报错。
未请求保留且没有 ThinkingDrop 的请求保持原渲染路径。所有路径仍使用原生工具、
reasoning parser、模型与 sampler。

BUS 使用四个必验模型的真实 tokenizer 检查保留、准确 Drop 来源、无功能不变、
模板幂等性和原生 Jinja 独立使用，最终 `4 passed in 24.68s`。
`benchmark/context_system/run_minimal.py:122-131` 为修改版与冻结原版 SGLang
生成同一保留历史模板，启动时通过原生 `--chat-template` 传入，并记录 SHA256。
服务端和客户端使用同一模板与 `preserve_thinking_history=true`，避免省略历史带来
虚假的性能提升；mini 对照使用其已验证的 `MINISGL_PRESERVE_HARMONY_HISTORY=1`。
原生模板间仍存在格式差异，不能仅据此宣称完整轨迹 input IDs 相同。

### 已完成的最小吞吐子集

以下来自完整轨迹最终 JSON，非中途估计；GPT-OSS-120B TP2、BUS GPU0/1，
mini `2966eb4`。单位 tokens/s，实际吞吐排除 cache hit。

| 配置 | 首次任务 | 成功 turn / 失败 | 测量秒数 | 实际 Prefill / Decode / All |
| --- | --- | --- | --- | --- |
| C1 no_drop | 2/2 | 84 / 0 | 850.903 | 252.298 / 32.091 / 284.389 |
| C1 Drop+R | 2/2 | 84 / 0 | 824.251 | 260.506 / 33.128 / 293.634 |
| C2 no_drop | 4/4 | 150 / 0 | 1321.638 | 324.724 / 35.734 / 360.457 |
| C2 Drop+R | 4/4 | 150 / 0 | 1416.758 | 358.832 / 33.335 / 392.167 |

相对上述 BUS 实验根目录，证据分别为：

- `minimal-mini120-c1-none-v1/workload/20260917_223811_238167_no_drop_C1.json`
- `minimal-mini120-c1-drop-v1/workload/20260917_225617_541427_drop_C1.json`
- `minimal-mini120-c2-none-v1/workload/20260917_231258_426732_no_drop_C2.json`
- `minimal-mini120-c2-drop-v1/workload/20260917_233903_368300_drop_C2.json`

C2 成功 turn 含 143 次 first-pass 和 7 次 filler；截止取消单列，不计失败或成功分子。
所有返回长度均符合源任务要求。mini 四组已完成，修改版/原生 SGLang 及 PD 矩阵
仍未全部完成，不能依据此表宣称 SGLang 达到修订后的 5% 原生对照门槛。并行实验分别占用
0/1 和 2/3，但共享主机 CPU；小样本、首遇形状 JIT 和执行时段差异均须列为比较限制。

2026-09-18 补充：`minimal-sg120-c1-none-v5` 完成 84/84、零失败，但测量跨越午夜，
GPT-OSS 模板日期改变使已缓存长前缀重新 prefill。其 1228.124 秒不纳入性能对照。
`7dab80625` 将实验模板日期固定在启动时；生产模板保持原生行为。
`minimal-sg120-c1-drop-native-ar-v1` 在接收任何请求前因 Unix socket 路径超过
107 字节启动失败。`1929c115c` 让每个 server 使用 `/tmp/sg-bcp-*` 独立短目录，
缓存和实验输出仍隔离存放；实际目录记录在 `launch.json`。v2 使用此修复重启。

### 中间 prefill 容量边界（2026-09-18）

`7390089b8` 补充匹配后的最低 KV 需求检查：最终 active 长度可装入池，并不保证
Drop 生效前的待算 query 可以装入。对 raw 历史超过整个池容量的请求，根据
`RecoveryPlan.intervals` 计算实际待算 query 的最大可见 KV 数；跳过热命中区间，
不按冷历史峰值误拒绝热请求。相同 program 和区间复用 CPU 检查结果；raw 长度不超过
池容量时直接返回。拒绝发生在请求取得 KV 前，返回 503，不缩短输入或执行 dummy
forward；P 通知 sender 并释放 metadata，D 不执行历史 prefill 容量检查。

实现：`python/sglang/srt/context_system/request_storage.py:11`
`prefill_capacity_error`；`python/sglang/srt/managers/scheduler.py:3319`
`_reject_context_prefill_capacity`。这是不可能读集的下界检查，额外 COW 分配仍由原有
admission 预算计算；不能声称覆盖了所有资源不足场景。

本地 `pytest test/registered/context_system/test_context_raw_storage.py -q`：
3 passed、1 CUDA skipped；py_compile、Ruff format、diff 检查通过。
BUS 独立 worktree `sglang-r2-capacity-check@7390089b8`，不暴露 GPU：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=python python -m pytest \
  test/registered/context_system/test_context_raw_storage.py \
  test/registered/context_system/test_context_admission.py \
  -k 'prefill_capacity or capacity_rejection' -q
```

结果 4 passed、9 deselected，17.25 秒；日志 `capacity-focused-7390089b8.log`。
先前较宽的 `-k capacity` 同样通过这四项，但另选中一项既有 fixture，该 fixture 对
空 `CUDA_VISIBLE_DEVICES` 取首字符而报 IndexError；保留该失败日志，不将其计作通过。
计时实验工作树保持 `1929c115c`，没有在运行中更新源码。

### Native allreduce 与长 raw P/D 补充

GPT-OSS-120B 在恢复原生 custom allreduce 后，普通调度六条 BCP 固定路径复核为
`1 passed in 288.45s`，测试 HEAD `30bdff1b6`，GPU2/3 TP2。
`bcp778-sg-gpt120-native-ar-v1/` 中的逐项误差与上述普通 120B 表相同；不能将
此前关闭 custom allreduce 的未完成吞吐运行当作有效基线。原生 IPC 的两处构建兼容
修复只补声明和限定 `tvm::ffi::get` 名字查找；独立 allreduce 检查两 rank 各 180 种
dtype/size/algorithm/eager/graph 组合通过，日志 `allreduce-ipc-v1.log`。
吞吐原生对照使用 `b21177f2e`，即冻结官方基线加同样的 SM80 MoE 调优和这两处构建
修复，不称为未经改动的官方版本。

长 raw PD 证据 `long-raw-pd-gpt20-30bdff1b6-v1/` 使用 GPT-OSS-20B、P GPU2、
D GPU3，raw=4668 超过原生行宽 2048，五次 Drop+R，8 个固定输出。
原始 pytest 因热路径过严对照失败（304.40s）；保留失败，不改称 pytest 通过。
定位到原生 P 在传输前进行 unfinished-cache 去重：重算的末 prompt KV 被冷缓存
canonical 页替代，而普通热路径继续使用重算页。因此 P 首 logits 与普通热算完全
相同，D 后七步与普通冷算完全相同。保存的 GPU 数据按这一原生交接路径复核为逐元素
完全一致，有限值、input/output IDs、usage 均通过；`2dcec1f19` 的测试 oracle 按
此路径修正，没有为改断言重复运行模型。普通热路径诊断 max/mean/p99 为
0.4375/0.024850/0.15625，八步 argmax 相同；热缓存 cached=142、repos=0、
Drop-skipped=4525、实际 prefill=1、decode=7。后验核验文件为
`posthoc-handoff-validation.json`，与原始失败报告一并保留。

## 最终设置

1. 第一阶段必验 Qwen3、AgenticQwen、GPT-OSS。保留 SGLang 原生模型、MoE、
   tool/reasoning parser 和调度器，不移植 mini 的替代实现。
2. 生产路径只支持 mask prefill 与 paged-occurrence。mini staged 用作计算语义对照，
   不作为 SGLang 生产路径。一次正式请求直接保留 prefill 产生的首 token；不增加请求级 warmup。
3. 消息 ID 使用用户 `messages` 数组的下标。`drop_message[n]` 在消息 n 算完后生效，
   消息 n 仍可读被选中的历史 KV。同时保留结构化 DropRule 的语义。
4. Drop 不改变幸存 token 的位置；Reposition 才压缩位置。同一边界先 Drop 后 Reposition。
   原始 token 下标、当前 RoPE position、active 下标、物理 KV slot 分别维护。
5. 最终 Radix key 包含 token / Drop / Reposition 历史以及最终位置。保留最终位置 KV，
   兼容 Retry 生成目标位置的新 KV；不原地旋转其他请求可共享的来源 KV。
6. 普通调度与 P 端支持 drop-aware eviction；先淘汰无引用叶节点，再淘汰符合条件的内部
   Drop 区间。拓扑引用与实际 KV 读引用分开，不能释放仍有 GPU 读者的页面。
7. PD 使用不同 GPU 上独立 P/D 进程。D 端默认关闭 Radix；不将 D 生成的 KV 回传 P。
   下一轮 prompt 中包含上轮输出时，P 缺少的部分正常重算，并计入实际计算量。
8. GPT-OSS 使用 SGLang 在实验硬件上自动选择的默认 attention backend，不强制切换 FI/FA。
9. 无功能请求走原生快速路径；原生 chunking、overlap、mixed batch、retract 保持有效。
   按用户后续修订，首阶段仅支持 `page_size=1`；撤回本次新增的大页 Reposition 分支与
   多页功能验证矩阵。Context cache/Retry/attention 入口显式拒绝其他页大小；
   SGLang 原生无功能请求的大页能力不变。

## 已确认的实现差异与接入要求

以下 SGLang 引用均以冻结基线为准；mini 引用以 System-test 固定版本为准。

| 差异 | 源码证据 | 迁移要求 |
| --- | --- | --- |
| 默认缓存是 UnifiedRadixCache | `python/sglang/srt/mem_cache/registry.py` 的 `default_radix_cache_factory`；`kv_cache_builder.py` | 接入 unified tree/components 的插入、分裂、锁和回收，不能只补旧 RadixCache |
| 两者都缓存已计算的输出 KV | SGLang `unified_radix_cache.py:959` `cache_finished_req` 使用 prompt+output；mini `core.py:764` `append_host` | 普通 scheduler 保留输出缓存；未做 forward 的最终采样 token 不能冒充 KV |
| SGLang 会缓存未完成 chunk | `unified_radix_cache.py:1105` `cache_unfinished_req`；mini `scheduler/cache.py:422` 对 contextual/delta unfinished 返回 | chunk cache 必须带阶段/版本语义；不能把临时 occurrence 作为最终 cache 提交 |
| Unified insert 自行释放重复输入 KV | `unified_tree_core.py` 的 `_insert_walk_step` 生成 `FreeDeviceKV`；`unified_radix_cache.py` 的 `insert` 执行动作 | 调用方不得再次释放 `prefix_len` 对应输入；本次 Retry 测试的重复释放已据此修正 |
| unified SWA 有独立有效范围与锁 | `unified_cache/components/{full,swa}.py`、`unified_tree_core.py` | full 与 SWA 有效页不同，Drop-skipped 不能因 SWA 窗口裁剪而增加 |
| 原生支持真实 retract | `managers/schedule_batch.py` 的 `reset_for_retract`、`retract_decode` | 释放 KV 后保留可重建的 Context 状态，恢复不能退化为裸 token 普通 prefill |
| mini occurrence 某些混合 batch 限制 | mini `attention/{fi,fa}.py` | 不迁移其 mixed-batch 限制到 SGLang；请求间 metadata 必须独立 |
| 模型默认后端依硬件而定 | `arg_groups/model_overrides/gpt_oss.py` | 记录解析后后端；A800 的当前默认选择为 Triton，保留 sinks、SWA、原生 MoE |
| 原始长度可以超过 RoPE 上限 | mini 历史 128K 越界修复 | 按 raw 长度给 raw 表定界；模型边界按实际计算 position 与合法 active 容量检查 |

## PD 原生流程与迁移边界

P 的 bootstrap 队列完成连接与 D 容量协商，再进入正常 prefill 等待与计算；已可传输的
KV 进入传输队列，GPU stream event 保证写完成后发送。P 的最终 prefill 还产生首个输出
token。D 先分配自己的目标页，接收 KV 和首 token 等元数据，构造接续 decode 的 batch，
通常不再执行一次完整 prompt forward。

关键入口：`disaggregation/prefill.py` 的 `finalize_bootstrap`、`process_batch_result`、
`_send_kv_chunk`、`optimistic_release_and_requeue`；`disaggregation/decode.py` 的
`_match_prefix_and_lock` 与预分配路径。连接接口位于 `base/conn.py`、`common/conn.py`、
`mooncake/conn.py`。配置 `arg_groups/fields/disagg.py` 的 D Radix 默认值为关闭。

迁移传输的是最终 active 的 K/V、相应位置及可验证的 Context 元数据。P 上用于早期 query
的 birth/临时 occurrence 不是 D 所需的终态 KV；P 必须保留这些页直到最后一个 GPU 读者
完成。D 按 active 长度而非 raw 历史长度预分配，并沿最终位置继续 decode。P/D 页号各自
独立，传输协议不能把 P 页号当成 D 页号。

在存在最终位置变化的区间，不能直接复用原生的“按 raw prompt 长度发送全部页”假设。
`83cfdf638` 已恢复完成 chunk 中稳定终态 Full 页的传输，并保留末页至最终元数据
就绪；仍需在隔离 GPU 上测量终态整理与传输等待的代价。
连接失败、取消、重复完成通知、重试时复用原生连接状态机制，同时隔离请求尝试的 KV 所有权。

本轮不增加 D Radix 重试矩阵，也不增加 D→P 输出 KV 传输。P Radix 命中和 Context Retry
仍须验证；这与网络传输 retry 是不同的机制。

## 必须重点防止的效率回退

| 风险 | 最终约束 / 需要观察的证据 |
| --- | --- |
| 每 token 整树 DFS 或完整 integrity 检查 | 生产热路径只做增量维护；完整核验在测试/诊断开启 |
| 每条消息重新渲染整个前缀 | 一次规范模板渲染的 provenance；不按反复 tokenize/LCS 猜 owner |
| 全历史 stage tensor clone/IPC/各 TP 重复编译 | 传输紧凑不可变 IR；当前 chunk 按需编译；避免累计 tensor 的重复序列化 |
| SWA 按 query 在 Python 展开大量 segments | 保留精确位置语义，使用批量/native 编译；测 compile 与 kernel launch 时间 |
| 容量探测反复物化整个 occurrence 计划 | 先最大终点快路径，复用预处理；只物化获选 chunk |
| `tolist/item/isin` 等引入 GPU↔CPU 同步 | 优先 CPU 元数据和批量 H2D；逐处记录 stream 与生命周期 |
| 运行请求时首次 JIT/AOT 编译 | 启动时准备所需 kernel；startup 时间单列，不藏入首请求 |
| 过早压缩或释放造成 overlap 阻断 | GPU event 驱动回收；CPU 不全局同步等待无关 batch |
| 所有历史 occurrence 永久预留 | 只保留当前读集、必要 birth 来源和待传输终态；及时归还临时页 |
| 内存不足直接清空前缀重算 | 先 evict、等待/调度其他请求或原生 retract；最小可推进 chunk 检查 |
| Retry 扫全树、选错分支、原地 RoPE | 长度界限剪枝、最长兼容匹配、只物化胜者；新目标页与源页隔离 |
| 重复逆旋转累积误差 | 保留正确 birth 来源；同位置直接复制；RoPE 参数来自原生模型 |
| Drop-skipped 按终态 inactive 计算 | 必须覆盖本次所有实际 query 与所有 chunk 的真实读集 |
| 用逻辑 prompt tokens 冒充计算吞吐 | cache hit、Drop skip、RoPE copy、graph padding 不计新 forward tokens |
| 页配置或后端不同导致性能结论失真 | 按批准范围，功能开关均使用 page_size=1 和默认后端；结论限于该配置，不外推到原生默认页配置 |

历史决策链重点：`01a05dc4…`（key、CPU 热点、重复模板），`01a077b2…`（一次正式
prefill、usage、staged oracle），`01a07c59…`（occurrence/临时 KV/chunk），`01a086ac…`
（SWA/AOT/raw 越界），`01a08e0b…`（最终位置、Retry、overlap），`01a099fc…`
（drop-aware、holes、高压调度），`01a0ab12…`（最新测速方法）。早期“禁用 Retry”、
“遇任意 R 就停止”、旧 staged 默认、旧 drop-aware 优先级均不能作为最终要求。

缓存 usage 以真实 raw token 集合计数：R 为本次匹配驻留集合，U 为本次所有计算 query
在 full attention 上读取的缓存集合，T 为其中经过位置变换的集合。
`cached=|U\T|`，`repos=|U∩T|`，`drop_skipped=|R\U|`；应满足三者之和等于 |R|。
虚拟标记、空洞、重复 occurrence 不增加 R；仅为 cache-back 而复制、未被 query 使用的
KV 不增加 repos。SWA 自身窗口淘汰不当作 Drop-skipped。

## 验证矩阵与通过依据

| 层级 | 场景 | 通过依据 |
| --- | --- | --- |
| 规则与边界 | legacy/structured Drop、重复文本、思考历史、tool、多 Drop/R 同边界 | 与固定 mini 版本的 token IDs、owner、raw 区间、位置、key 对照 |
| 计划与缓存 | 冷/热/部分命中、相同/不同 R、最长 Retry、跨分支、cache-back | 逐 query 可见集相同；目标 K/V 与来源所有权独立 |
| 生命周期 | chunk/mixed/overlap/abort/retract、叶/内部 evict、hole recovery | 无悬空/重复释放/泄漏；恢复后输出与无洞参考在允许误差内 |
| 数值 | 三个必验模型，固定 token teacher forcing 与实际生成 | 按模型/dtype/backend 校准 logits 误差；定位首次分歧，报告原始数据 |
| GPU 内核 | 同位置 copy、多次 R、SWA/sinks、page_size=1 的非连续物理 slot | 对照实际 mini staged/occurrence 路径；不使用简化 dense 模型冒充 oracle |
| PD | 普通 SGLang↔PD↔mini，首 token/终态/取消/传输重试 | D Radix 关闭，无输出回传；元数据和 active KV 对齐，无多算首 token |
| 压力 | raw>128K 但 position 合法、低 KV、多请求长期运行 | 原生调度可推进，drop-aware 无 -1 进入 attention，池容量可回收 |
| 性能 | 原生无功能↔带功能；mini 作为参考；PD 小矩阵 | 相对原生输出吞吐下降不超过 5%；实际计算量、TTFT/TPOT/E2E 和比较限制单列，统计不足不得宣称达标 |

最小吞吐矩阵：GPT-OSS-120B；普通 mini/SGLang TP2；PD P=TP2 GPU0/1、D=TP2
GPU2/3。所有对照固定 page_size=1；每种系统做 C1/N2、C2/N4，各含 no_drop+普通 eviction、Drop+drop-aware
（PD 仅 P 开启）。同一固定 seed 的任务子集，完整轨迹与源输出长度，不截断 turn 或 output。

测速沿 `01a0ab12-b9ee-7751-bee0-6a78dcac9c1f` 的 `test_serving.py`：同一 task 串行
推进，把上轮实际生成重建的消息加入下轮；持续维持 C 个不同 task；长尾填充成功请求计入
总体并单列，所有首次任务终结时截止。消息重建有损，这是已接受的客户端口径，不能声称
精确还原 sampled token IDs。固定 token 数值测试另做。Rolling Drop 保留 12 条 tool
response，TR13 后丢 TR1；按当前位置到 96 Ki 后在合法边界 Reposition，保留历史事件。

同时报告逻辑吞吐与实际 prefill/decode 吞吐；缺少实际计算计数时报告缺失，不能估造精确值。
标准 TPOT=(E2E−TTFT)/(输出数−1)，首 decode gap 单列。SSE chunk 间隔不能当作 GPU
token 间隔；server token timing 也不是 GPU 硬件计时。本轮不测 SLO。

## 历史检查点（以下描述其当时状态）

### 初始基础编译器检查点

新增 `context_system/ir.py` 与 CPU `kernels/ops/attention/context_plan.py`：保持
mini 的两遍 native 编译、最终位置 key、Drop→R 顺序、尾部 Drop 后无效 R 的语义。
整型元数据在转 int32 前检查范围，避免截断后错误寻址。`occurrence.py` 只展开当前
query window；`recovery.py` 用可见性依赖闭包恢复洞，并保留可复用后缀。

本地 macOS / Python 3.13 CPU 检查：

```bash
PATH=/tmp/sglang-context-cpu-20260917/bin:$PATH \
  /tmp/sglang-context-cpu-20260917/bin/python -m pytest -q \
  test/registered/context_system -k 'not fixed_mini'
```

结果：321 passed，1 deselected。包含 304 个事件程序、逐 query/window 可见集对照、
1000 个随机缓存洞依赖图和 raw 140000 的位置压缩场景。测试独立加载 CPU 模块，
没有初始化 SRT；这不覆盖 scheduler、KV 分配器、attention、PD 或模型输出。
mini 原始 native 编译器的可选直接对照因其 Ninja 输入路径未转义空格在本机失败，
归类为 `ENVIRONMENT_BLOCKER`，待远端无空格路径复核，不能将 deselected 算通过。

Zhangyudong-BUS：隔离 checkout 已通过 HTTPS Git clone 到
`/mnt/public/wangruoxi/repo/sglang-system`，初始 HEAD 为冻结官方基线；原有 detached
SGLang checkout 保持不变。现有 Torch 2.11 cu130 原先无法初始化 GPU；将 NVIDIA
`cuda-compat-13-0 580.178.04` 解包到独立实验目录并仅设置该进程 `LD_LIBRARY_PATH`
后，GPU0 上设备识别和 `torch.arange(100, device='cuda').sum()` 通过（4950）。
没有更新系统驱动。最新上游所需的隔离 Python 环境依赖安装仍在进行，尚无模型实验结果。

兼容包 SHA256：`14a3d14373f882297f368d6282fc7fba85e46682f34166291d61df1913a59c8f`。
配置依据：[NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。

- 完成 52 会话的决策/失败证据核对与 75 文件、System-test 增量的迁移分类。
- 实现紧凑 Context IR、native template 接入及规则对齐。
- 实现普通 scheduler/cache/attention，再接 PD，按上述门禁验证。
- 所有实验在 Zhangyudong-BUS 的隔离 checkout/environment 上运行；通过 Git 同步代码。
- 每个实现检查点精确提交到 `system`；测试失败按证据修复，不回写 mini，不覆盖远端已有修改。

## 2026-09-17 页大小范围收缩

用户明确要求先只支持 `page_size=1`。保留 paged-occurrence 的 birth/终态版本、
Drop-skipped、最长兼容 Retry 和 COW RoPE 设计；这里的 occurrence 机制不要求一页
包含多个 token。撤回新增 Reposition HND 四维大页寻址；Context Radix 的结构化
key 仍是一项真实 token 对应一项 KV，保留 Drop/R 事件和最终位置的身份区分。

已推送历史不做 reset 或重写，通过后续提交撤回相关实现。旧 HEAD `5e2f78326` 的
多页 GPU 回归已按此范围修订主动中断（31 passed 后 KeyboardInterrupt），不作为
最终验收。后续只运行 page_size=1 的正向功能测试；大页仅保留拒绝测试及原生能力
未被改变的回归测试。模型和 PD 验证仍待完成。

## 历史：Native chat / IPC 初始接线

OpenAI chat 请求现在显式解析 DropRule、legacy Drop 和严格整数 Reposition。
Context 请求在 native Jinja 规范化之后用一次完整渲染取得 token provenance；
`continue_final_message` 保留原生独立编码 assistant prefix、去除开头 BOS 的行为，
不把拼接后的文本重新 tokenize。KeepText 使用完整历史进行同一原生模板处理。
未带功能的请求继续使用原有 render/encode 路径。

编译结果使用现有 tensor-buffer MessagePack IPC，避免转换为逐 token Python 列表；
接收方检查 schema、dtype、原始 token IDs、最终 key positions 和 visibility。
此检查点仍保留 tokenizer admission guard：scheduler 的 Context KV 生命周期尚未
接通时明确拒绝正式生成，防止把 Drop/Reposition 请求静默算成普通 attention。
这不是生产功能通过的声明。native chat、实际模型 tokenizer 与 IPC 的 Linux 测试
在 Zhangyudong-BUS 执行；完整模型、PD 与吞吐仍待后续验证。
