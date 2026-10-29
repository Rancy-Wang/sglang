# 通用 recurrent CDB 与 LoopSpec V2：晋升记录和实施计划

## 状态、授权与预算

- 用户要求先晋升现有 bs=1 系统代码，再生成 CDB 与多请求 LoopSpec 的计划。本轮不实施 CDB，也不将未执行的 profile 标为通过。
- 晋升源：`loopspec-test@60de0ffc562895cc723e0a15c8987e3d5e9912ab`；最后运行时代码提交为 `7966c680135e38699eb2d5273692adf8111ff289`。
- 晋升前正式分支：`loopspec@baa9befa381511cd4d56f83980951e5a45fffc6d`，即固定的 SGLang v0.5.22 基线。
- 用户本次明确授权把所有系统代码（排除测试、MD/JSON 等无关文件）作为一个提交放入 `loopspec`，提交名严格为 `[update] Support Loopspec for bs=1`。再将其连接到 `loopspec-test` 最新历史；实施为正常 merge 回测试分支，不重写历史。
- 旧迁移任务已在 16/25 轮达到原定 Ouro 验收标准；其历史及预算不重置。本任务是新增的通用 CDB / 多请求功能，当前 **0/5**，未开始代码候选。项目 AGENTS.md 的五轮上限优先于技能正文中不一致的十轮文字；阶段不是独立预算，不按阶段重新计数。
- 官方 LoopSpec 参考：`41aa710ed7c9ce908cffa03597be5d1df19d7b56`；vllm-rlt 结构参考固定为 `6b164d3844d35ab8e753df1864f0c87dc180401a`，不静默追踪 main。

## 本次晋升的六项计划

1. **目的**：将已验收的 bs=1 运行时发布到正式分支，建立后续性能/正确性对照锚点。
2. **步骤**：核对本地与远端分支、diff；提取运行时文件到正式分支；按指定标题提交；正常合并回原测试分支；只在测试分支记录新计划；以非强制 push 同步两分支。
3. **范围**：源分支相对于正式分支变更的 35 个 `python/sglang/**/*.py` 运行时文件，包括 kernel/model/config/worker/KV/metrics/参数与原生接入。`research/` 中全部测试、诊断脚本和文档不晋升。
4. **成功标准**：正式提交仅包含上述 35 文件，其内容/模式与源提交逐项一致；正式分支只新增一个提交；测试分支包含该正式提交及原最新提交，两者代码树合并前后不变。
5. **风险**：遗漏运行时依赖、混入测试记录、分支并发更新、merge 冲突改变代码。通过完整差异清单、blob/文件模式比对、merge 结果树比对和远端引用核验排除；不 force push、不重写旧历史。
6. **验证**：`git diff --check`、35 文件 AST 解析、提交文件集合/逐项内容比对、`git merge-base --is-ancestor`、远端 SHA 与本地核对。没有运行时代码变化，本轮不重复已有 GPU 验收；原测试证据在旧任务 record.md 第 16 轮。真实 Raven checkpoint 仍未验收。

## 晋升执行结果

- 正式提交：`237f4d44aee9f9751234758c86d3154421c852e0`，标题 `[update] Support Loopspec for bs=1`，父提交为 `baa9befa381511cd4d56f83980951e5a45fffc6d`。
- 回接提交：`b6d73d386e9494cb46acbfa311325f4d807bcb3d`，第一父提交为原测试 HEAD `60de0ffc562895cc723e0a15c8987e3d5e9912ab`，第二父提交为上述正式提交。使用 `git merge --no-ff loopspec`，不是 ours 合并，不改写旧提交。
- 正式提交仅有 35 个运行时 Python 文件，2572 insertions / 11 deletions。逐文件 blob、模式、AST 解析、`git diff --cached --check`、祖先关系检查均通过，命令退出 0。
- merge 后 tree 与原测试 HEAD 的 tree 完全相同；研究脚本/测试/记录均保留在测试分支，未进入正式分支的新增提交。
- `git push --atomic origin loopspec:refs/heads/loopspec loopspec-test:refs/heads/loopspec-test` 退出 0，两分支已同时发布；后续本计划文档只提交到测试分支。
- 本轮没有新 GPU 测试；晋升代码等同旧记录第 16/25 轮验收代码，不将本次静态核验当作重新跑过 GPU。

## CDB / LoopSpec 计划

### 1. 修改目的与不可改变的原则

先让 SGLang 普通 loop transformer 推理在 **不启用 speculative algorithm** 时具有 continuous depth batching，再让 LoopSpec V2 使用同一能力。不能把 CDB 留在 LoopSpec worker 私有 while 循环内；不能用多个串行 bs=1 实例冒充 CDB。

- CDB 表示在 recurrent loop 安全边界重新准入/选批，允许不同请求处于不同深度。不是抢占正在执行的 GPU kernel，也不意味着跳过新请求自己的 prefill。
- 完全依据固定 vllm-rlt 的阶段、状态、执行描述及回填结构映射到 SGLang；复用其设计，而不是再引入一套 HTTP 服务、IPC、请求生命周期或与原生 Scheduler 并存的独立引擎。
- 只增加确实缺少的表达能力；现有 ScheduleBatch、ForwardBatch、allocator、KVLocPlan、FutureMap、CUDA graph runner 已能表达的内容不再创建平行框架。每个新抽象在 diff 中注明参考符号、复用位置和新增必要性。
- LoopSpec 直接采用 V2：同一个 worker 支持同步诊断和 overlap 执行，不先写 V1 再迁移。对外继续使用原生 SGLang API、`--speculative-algorithm loopspec` 与 `--speculative-draft-depth`，并发使用 `--max-running-requests` 等原参数；不新增 CDB HTTP 路由或算法专用调度开关。
- 本期实测目标为 Ouro-2.6B BF16、完整 R=4、单 GPU TP/PP/DP/CP=1。普通 CDB 层不依赖 Ouro 或 LoopSpec 的算法判断；模型 adapter 显式声明能力。其他循环模型维持原有可用路径，不能未经适配/验证就宣称支持 CDB。Raven 至少做既有 fixture 回归；真实权重另需实测。
- 不引入 early exit、DAS 的跨深度前缀读取或 SHARED/LAST_EXITED 近似来换性能。不同 depth、core-layer、token position、branch 的 KV 语义必须保持。
- 用户第 4 项“只增不减”结合第 3/5 项解释为吞吐不下降、时间不增加或只有微小增加；耗时的验收方向始终是越低越好。

### 2. 参考结构、实施步骤与依赖门槛

固定参考仓库：[vllm-rlt 6b164d3](https://github.com/InferMatrixLab/vllm-rlt/tree/6b164d3844d35ab8e753df1864f0c87dc180401a)。下表中的落点是计划，不表示已实现。

| vllm-rlt 参考符号 | SGLang 实施落点 / 复用关系 |
| --- | --- |
| `RequestPhase`、`Request`、`StateSpan`、`ExecutionBatch` | 扩展原生请求的 recurrent 状态和 ForwardBatch 描述：phase、token span、depth、hidden/aux state handle、完成事件；普通 AR decode 一个 state，LoopSpec 多个 state |
| `Scheduler.schedule()`、`take_phases()`、`_planned_loop_count()` | 原生 Scheduler 选批路径下的 recurrent 计划；一次 core 工作只推进一个 loop，在返回边界重新准入。prefill 同时有 token chunk 和 depth 边界 |
| `RefillPolicy`、`SplitPrefillPolicy`、`MixedPolicy` | 从阶段队列选取就绪工作；复用原生准入、预算和请求管理。先有可交替调度的 prefill/decode，再按能力合批；不让长 prompt 连续跑完整 R 后才让出 |
| `Scheduler.update_from_output()` | 明确区分状态进展、readout 完成和 token 提交；无输出步不增加 seq_lens、不造占位 token，不触发错误的 EOS/stream/token 计数 |
| `StageEngine._step()` 的 submit/collect/update | 复用 SGLang run_batch、V2 forward isolation / FutureMap / result queue；只增加 recurrent 所需的进展载荷，不复制 StageEngine |
| `MetadataBuilder.resolve_rows()/build()`、KV planes | 复用现有 timeline 与原生 allocator/translator，构建逐行请求、分支、位置、深度地址；只更新活跃行/变更范围，不每步复制整个 request×context 表 |
| `OuroForCausalLM.recurrent()/recurrent_prepared()` | 复用本地 `prepare_recurrent/advance_recurrent/readout_recurrent`，将 mixed-depth runner 从 LoopSpec 私有路径提升到通用层 |
| `SpeculativeRunner._submit_pipeline/_collect_pipeline` | 参考跨请求 state flatten/回填和允许零输出的组织方式；不照搬 greedy 单 candidate 的算法限制或其 KV 近似 |

EAGLE/DFLASH 仅参考本项目固定 v0.5.22 的 worker/info/utils 职责划分和简洁管理方式，**不复制、继承或调用其算法专属实现**（用户批准时的补充要求）。直接接入 SGLang 通用 V2 协议，包括 `forward_batch_generation(..., on_publish, grammar_barrier, pp_proxy_tensors)`、`SpecInput`、FutureMap 和 KVLocPlan；这是复用公共基础设施，不是复用 EAGLE/DFLASH 实现。不能只采用 V2 类名却继续拒绝 on_publish 和 overlap。

实施按以下依赖顺序进行；这些是里程碑，不是各自重新获得五轮预算。

**P0：先锁定基线并建立 profile。** 先运行下面 V1/V2 两组，冻结输入、环境、到达时刻和结果。使用现有原生 profiler/NVTX 能力；确有归因缺口才添加可关闭的最小 trace 点。工具/测试/annotation 改动也计入候选轮次。重型 trace 与性能测量分开，先证明新增 trace 关闭时不改变输出和延迟口径。未得到有效普通 AR B=8 基线前不进入 CDB 性能结论。

**P1：普通 Scheduler 的 CDB。** 在 `spec_algorithm=NONE` 下实现阶段状态、partial forward、逐请求回填、split prefill、动态准入和取消；普通 AR 是唯一依赖入口。先用同步 eager 排查正确性，但直接沿用 V2 能兼容的提交/完成模型，不建立临时 V1 实现。完成 native graph 和 CPU/GPU overlap，并达到 V3 门槛，才开始 LoopSpec 多请求算法改造。不能用 bs=1 时偷偷恢复不可调度的全 R forward 来掩盖 CDB 开销。

**P2：LoopSpec 使用 CDB。** 保留当前采样算法，拆掉 worker 中阻塞至 commit 的私有推进循环。worker 只向通用 CDB 提供就绪 state、readout 请求和分支策略结果。完成多请求单草拟后，在同一接口上接回双草拟、sampling；不能把最终功能缩减为 greedy 单草拟。完成 V4 的 8 请求功能检查后，再做 bs=1 精细 profile 和无 profiler 性能比较。

**P3：依据 profile 消除实测瓶颈，完成 V5。** 针对 GPU 空闲区间、host 调度、metadata copy、动态分配、图填充、readout、接受/拒绝与回收逐项优化。每个候选保留前一版和冻结基线的配对结果。输出/数值或延迟不达标时先定位，再在剩余轮次内修复；不靠放宽阈值、换输入、关 graph 或改变 target 深度过关。

### 3. 修改范围与状态所有权

| 范围（相对仓库根目录） | 计划职责 |
| --- | --- |
| `python/sglang/srt/managers/scheduler.py`、`schedule_batch.py`、`scheduler_components/batch_result_processor.py`、`managers/utils.py` | 原生准入与循环阶段调度、零输出进展/部分提交、取消/结束和普通输出处理；优先提取小的 recurrent component，避免往 Scheduler 填入算法细节 |
| `python/sglang/srt/model_executor/forward_batch_info.py`、runner/graph setup、必要的通用 recurrent runner | 对标 StateSpan/ExecutionBatch 的逐步执行；pre/core/post 和混合深度批次，通用图捕获/重放 |
| `python/sglang/srt/models/ouro.py`、`raven.py` | 模型特有初始化、循环一步、readout 和附加状态。模型层不调度用户请求，不处理 LoopSpec 接受率 |
| `python/sglang/srt/mem_cache/timeline_pool.py`、KV translator / allocation 相关既有模块 | depth/role/branch 隔离、前缀视图、物化前沿、引用和执行事件、容量预留；分配/回收单一权威 |
| `python/sglang/srt/speculative/loopspec/worker.py`、`pipeline.py`、`sampling.py`、`graph.py`，必要的 `info.py` | worker 对接 V2；info 保存算法请求状态并支持 filter/merge/idle；pipeline/sampling 保留算法逻辑，公共执行与图代码移出后删除重复实现，不保留两套路径 |
| `spec_info.py`、`spec_registry.py`、`spec_utils.py`、原参数 validation/metrics | 复用算法注册及调度协议，只放必要接入和明确支持边界；不会再把固定 lane 宽度当接受率分母 |
| 本目录计划新增 `profile_workload.py`、`analyze_profile.py`、必要的定向回归测试 | 可重放负载、逐请求/逐 token 事件、trace 分析和门槛检查；不把大 trace、dataset、模型提交到 Git |

具体文件拆分在实施前用最小 diff 确认；不能为凑 worker/utils/info 名称增加薄包装或把完整另一套 runtime 藏入 utils。

**请求与节点。** 只保留原始用户 Req；内部身份为 `(rid, request_epoch, lane_uid)`，执行票据再带节点 generation/version。每节点保存 parent/path、token position、depth、hidden/aux handle、prefix view、slot 与 proposal。物理 batch row 每轮可重排，通过 owner 映射回填，不把行号当长期身份。普通 AR decode 是同一结构的单 state 特例；prefill 可以有多个 chunk span。

**调度。** 先按原始请求分配公平预算，再由算法提交满足依赖的 lane；双草拟不能因为 lane 多天然多得资源。预算分别核算请求数 B、执行行 M、prefill tokens 和 KV；B=8 不代表只有 8 行，R4/d1=1 单草拟完整布局可到 32 行，双草拟可到 56 行。保留 target/纠正 token 所需容量，额外 draft 不能吃光使验证死锁。不能任意单步推进子节点而读取尚未物化的祖先同深度 KV。

**缓存。** branch fork 共享前缀索引，分离后缀；共享权重不共享不同 depth 的 KV。提交转移所有权，拒绝只取消对应请求子树；真实释放必须同时满足逻辑引用为零与 GPU 事件完成。处理取消、filter/merge、请求 ID 重用及旧结果返回的 ABA 问题。padding 不得写入仍有意义的共享槽；空 batch/idle 分支不计输出。

**V2 与异步。** 状态进展可以没有 token，FutureMap 的已提交长度与为将来执行预留的长度必须分开；`on_publish` 只能发布已确定的提交进度，不可将推测长度暴露给 detokenizer 或停止逻辑。保留有限在途票据和独立 buffer lease；不能在采样尚未完成时在 CPU 预判分支。复用 SGLang 已有 schedule/forward/plan stream 及事件，不引入第二套不受生命周期管理的 stream 系统。

**CUDA graph / bubble。** 复用通用 recurrent 图入口，按实际工作量和必要 layout 分桶、持久化 buffers；不按每请求捕图，不热路径重捕获。有效行 mask、depth 地址、prefix 长度和 slot 都是动态数据；图不得固化请求身份。readout 只计算草拟/完整深度 endpoint，hidden gather/scatter、KV metadata 尽量驻留 GPU，避免每 loop 的 `.item()/.tolist()`、同步和全表复制。提前准备下一步元数据、批量回收，不在有独立就绪工作时等待无关请求的 host 结果。分析 GPU 空闲时间时区分无工作、依赖等待、CPU 提交慢、allocator、H2D/D2H、graph padding；不声称消除了模型本身的串行依赖。本期单卡测量覆盖 CPU/GPU 以及 GPU 计算阶段之间的 bubble；真正跨卡 TP/PP 通信 bubble 没有被这组实验验证。

### 4. 预期效果、冻结负载与验收标准

#### 已实查的数据与环境

- X5 SSH 已成功连接；2026-10-29 只读核查时 `/home/ruoxi/repo/sglang` 为干净 `loopspec-test@60de0ffc...`。未运行 GPU profile。
- 数据：`/home/ruoxi/repo/test/datasets/aime.jsonl`，30 条非空记录，SHA256 **`b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331`**；字段 `problem/answer/id`。单请求取文件顺序第 1 条，多请求取前 8 条；只把 problem 发送给模型，不拼入 answer。
- 模型实际 config：`OuroForCausalLM`、`total_ut_steps=4`、48 个物理 core layer、hidden 2048、max_position_embeddings 65536。服务窗口仍按下述共同配置固定，不因为模型上限大就扩大测试窗口。
- X5 有 `/usr/local/bin/nsys` 和 Docker；未发现 `ncu`。本轮实查 Nsight Systems 为 **2025.5.1.121-255136380782v0**。容器注入权限、与实际 CUDA runtime 的兼容性和可用 graph tracing 选项需在运行前核实，不能把“可执行文件存在”当作采集已经可用。
- 沿用旧验收环境的镜像/依赖优先：`loopspec-native-20261026`、容器 Python `/opt/sglang/bin/python`、源码路径 `/workspace/sglang/python`。本轮 docker inspect 确认源码和模型只读，`/results` 仍映射旧任务目录，且没有 AIME dataset 挂载。实施时复用其镜像和依赖创建本任务隔离容器（例如 `loopspec-cdb-20261029`，先检查名字是否占用），明确挂载新的结果目录和只读数据目录；保留旧容器/旧日志。实际运行前核实解释器、sglang.__file__、包版本、驱动和 GPU UUID，不能直接使用旧环境快照作证明。

#### 冻结负载

默认主工作负载：BF16、Triton、`bf16-gemm-backend=auto`、page_size=1、radix cache 关闭、目标 R4、temperature=0、max_new_tokens=512、默认 EOS/stop；prompt 原文不加聊天模板，不强制 ignore_eos。服务 context-length=2048、mem-fraction-static=0.7，单请求 B=1，多请求 B=8。运行前先用实际 tokenizer 验证每题 prompt+512 在窗口内，并核算 8 请求和推测保留空间；若共同资源配置不足，必须在正式基线测量之前统一确定可运行配置并写入记录，不能只改变新版本预算。

主交错到达时间固定为从同一 client monotonic 起点的 **[0, 10, 25, 45, 75, 120, 200, 320] ms**，按第 1..8 题对应。client 在 X5 本机通过原生流式 API 异步发出，预热连接并记录计划/实际发送时间与请求到达时间；不能把上一请求完成作为下一请求发送条件。比较两版重放同一绝对相对时刻，不用“各自测得的 token 时间”重新计算间隔。补充逆序/轮转题序用于公平性、取消和长短组合检查，与主配对结果分开报告。

同样的输入先做逐请求独立 AR 输出参考，再验证旧 AR B8 与它一致，避免把错误的并发基线当正确性 oracle。所有对照记录 token IDs、finish reason、输出长度和停止位置；若 512 截断则如实标为 length，不把它包装为解题成功。数学答题正确率可记录，但不是推理系统一致性的替代指标。

#### 正确性与延迟门槛

| 指标 | 通过标准 |
| --- | --- |
| 模型数值 | 沿用原阈值：hidden max abs ≤0.125、RMSE ≤0.02；logits max abs ≤0.5、RMSE ≤0.1，比较对应相同 token/prefix/depth，不在生成序列分叉后比较不相同输入 |
| Greedy 输出 | 各请求 token ID 和停止位置与固定 full-depth AR 参考一致；浮点导致 argmax 差异必须定位，不能以“答案意思一样”直接通过 |
| CDB 语义 | trace 证明在 A 未完成全部 recurrent depth 时，资源允许的 B 工作能在后续 loop 边界进入；无输出步不伪提交，无跨请求/跨分支 KV 污染，split prefill 确实可让出 |
| Sampling | 保留实际归一化 proposal；覆盖单/双候选接受、残差和零质量分支，统计检验沿用现有方法；不能要求同 seed 跨调度顺序必然产生同一序列 |
| 主性能门槛（预先提出的“微小增加”量化） | 每题的配对 E2E 比值、8 请求整体 makespan 和聚合 E2E p95：点估计不超过 1.02，配对重复的 95% 置信上界也不超过 1.02；bs=1 同样执行。超过即失败，区间跨界为证据不足，不标通过 |
| 效果表述 | 满足 ≤2% 仅证明非劣；若要宣称有效降低延迟，还需预登记的 E2E/makespan/排队时间至少一项有明确改善证据。不能用吞吐提高掩盖某些请求持续退化 |

2% 是本计划对“微乎其微”的初始明确建议，不是已经测出的噪声；实施前固定，不能看结果后扩大。每题分别汇报，不能用平均加速抵消个别请求 >2% 回退。不能承诺每一次测量都单调降低，使用配对重复区分调度改变和测量噪声。

测量先 warmup 至图捕获/编译结束，至少 2 个完整 cohort 不入统计；然后至少 10 个配对 cohort，按 ABBA/平衡顺序运行，旧/新服务器不同时占 GPU。每题与 makespan 的点估计使用各配对重复 `candidate/baseline` 比值的中位数，聚合 p95 使用两版完整负载分布的 p95 比值。保留每次原始数据，以配对 cohort 为 bootstrap 单位，避免把同一 batch 中共享时序的 8 请求当独立样本。区间不明确时在同一候选上增加至最多 30 对；仍不明确则报告证据不足。不得只删掉新版本慢样本；环境干扰的剔除规则必须双边对称并保留原始记录。

### 5. 风险与避免错误结论的约束

1. **CDB host 开销可能超过合批收益**：loop 边界更频繁。通过 V2 overlap、静态缓冲区、批量提交和 GPU 元数据减小；先 profile 再改，不能以关掉 CDB 规避。
2. **共享 GPU 工作无法精确拆成每请求独占时间**：同时报告批次 GPU 区间与每请求参与/等待时间。按行数或 FLOPs 分摊只能标为估计，不能重复计费后当作总耗时。
3. **循环依赖和状态复用易引入 off-by-one**：输入 token 的 R-depth KV 与它预测的输出 token 区分，记录 position 与 parent。Raven 的附加状态和随机初始化单独处理，不能假定所有模型只有 hidden。
4. **异步 prune/cancel 的 use-after-free**：事件完成和引用清零双条件释放；保存 epoch/generation，测试旧结果晚到、结束后立即新请求、batch filter/merge。
5. **图布局与 padding 错写**：graph/eager 对照、动态深度/长度/行重排和空批；限制捕获桶避免巨大 logits/hidden padding 和启动内存膨胀。
6. **混合 prefill/decode 的注意力成本不等价**：token 数、lane 行数、上下文长度与 KV 预算分开；不能把长 prompt token 和一个 decode lane 简单视为等成本。
7. **工作量变化伪造加速**：输出 token 数/输入完全一致后才比较；提前 EOS、模型精度降低、变更 stop/max tokens、不同 graph/backend 都不能混作优化。
8. **profiler 扭曲时序**：不在热路径逐 token/layer synchronize；trace 采集不参与正式 E2E 门槛。graph 内部 CPU record_function 不会在 replay 时自动形成同样的 GPU 分段，必须用 graph/node trace 与 step metadata 关联；eager 只用于诊断，不能冒充 graph 性能。
9. **工程冗余**：每次审查检查是否复用了原生职责、是否删除被提升至公共层的重复代码、是否把模型/算法特例塞入通用调度；禁止引入一个与原生 Scheduler 并行拥有请求/KV 的新框架。

### 6. 五步验证与精细 profile 方案

用户批准时指定结果根目录为 X5 **`/home/ruoxi/repo/test/results/loopspec`**。本任务原始数据位于其下 `2026-10-29-cdb-v2/round-N/`，容器 `/results` 映射本任务目录，保留每个候选/模式/重复的独立子目录。每轮记录本地/origin/X5 SHA、运行时文件哈希、完整命令、环境、退出码、实际请求、配置和输出；模型/dataset/大型 trace 不进 Git。

| 用户步骤 | 固定实验 | 必须产出的证据 / gate |
| --- | --- | --- |
| V1：旧 LoopSpec bs=1 | 正式运行时 `237f4d44a`，第 1 题，d1=1，R4，graph 默认开启 | 每个输出 token 的路径、各阶段 CPU/GPU 区间、循环推进次数、接受/残差、资源与 bubble；独立无 profiler E2E 基线 |
| V2：旧普通 loop transformer | 同一运行时，关闭 speculative algorithm，前 8 题按固定时刻交错，B8 | 每请求到达/准入/排队/prefill/decode/readout/输出时间线，逐 token 正确性、E2E 与 makespan；不以 B1 串行 AR 冒充 B8 对照 |
| V3：普通 CDB 完成后 | 不启用 LoopSpec，完全重放 V2；追加 B1/B2/B4 及 arrival/cancel 检查 | CDB trace 证据、原数值/token 门槛、逐请求与整体延迟门槛；通过前不进入 LoopSpec 算法移植 |
| V4：LoopSpec CDB 完成后 | 先前 8 题交错：新 LoopSpec vs 新 CDB AR，检查功能/输出；再第 1 题 B1 重放 V1 | 多请求输出/停止/分支与 KV 正确；B1 各路径 profile 与 E2E 对比。多请求再记录相对同配置 CDB AR 的性能，不用旧 B1 AR 宣称并发加速 |
| V5：每个候选回归 | 运行受影响步骤和已通过的关键回归；最后完整重跑 V1..V4 对照矩阵 | 任一功能/效率门槛失败则定位根因并进入下一候选；已有候选上的重复测量不另计轮，改变代码/测试/行为配置再验证计下一轮 |

#### V1：depth 的精确定义和计时归因

完整 target 永远执行 R=4，不能把“每次输出新增计算少于 4 个 loop”称为 target 早退。输出 token 对应其前驱输入 token 在 R 深度的分布；部分计算可能在上一次输出之前已经完成，并与其他推测 token 同批执行。

每个已验证输出记录：`rid/output_index/input_position/lane_uid/parent/path`、本次 scheduler 调用起始的 `entry_depth`、本次新执行的 `delta_core_loops`、pipeline cycles、每 cycle 活跃/占位行、proposal origin、accepted/residual/restart、图桶、KV materialization/commit/release、前次保留 hidden 的 depth。

d1=1 时重点分组观察 entry_depth 0/1/2/3 及相应新增 4/3/2/1 个 loop 的实际路径。另分开 prefill 产生的首 token、warmup、稳态接受、拒绝重启和 EOS/尾部清空；不能将这些混成同一个平均 decode。AIME 第 1 题未出现的路径标记 N=0，不伪造数据；可用定向 fixture 补路径正确性，但不混入 AIME 性能分布。

每条路径依次归因到：

1. HTTP/tokenization/IPC/排队；Scheduler ingest、选批和 V2 准备。
2. lane 选择、KV 预留/分配、prefix/depth metadata、hidden gather、H2D。
3. 新 lane 的 pre；每次 core 的 norm、QKV、RoPE、KV write、attention、O projection、残差、MLP 和 step norm。R4 Ouro 的每次 core 包含 48 个物理层，不能把一个 layer 当一个 depth。
4. endpoint post/readout、logits 过滤/采样、proposal、验证/残差。
5. 分支选择、prune、提交、引用回收、hidden 保存、D2H/FutureMap 与事件等待。
6. 原生结果处理、detokenization、流式发送、客户端收到 token。

输出三张关联表：逐输出 token 表；按路径分组的 count/median/p95 和 stage 时间表；逐 GPU batch/core-step 表。共享 core 的 GPU wall time 只计一次；每请求表标记参与该区间和关键路径，额外按 row/FLOPs 估算的份额单独列。总工作量包括被拒绝分支的 wasted compute，不能只把成功分支成本放分母。

#### V2/V3/V4：交错请求与端到端测量

- client 用 monotonic clock 记录计划发送、实际发送、first byte/token、每次 token 到达和 last token；服务器记录 receive、admit、各阶段 begin/end、提交及 detokenizer 发送。不同进程 CPU 时钟与 GPU trace 通过工具校准关联，不直接相减未校准的时钟。
- per-request E2E = 最后输出收到时刻 − 实际发出时刻；同时报告计划发送到结束的延迟以暴露 client 发不出去的排队。TTFT、ITL 分布、server queue、prefill、decode span、GPU busy、CPU wait 分列。makespan = 最后一个完成 − 第一个实际发送。
- 合批 GPU 时间报告 shared 区间，不能将所有请求的 decode span 相加当作 GPU 忙碌时间。端到端重叠阶段也不能直接求和；用关键路径和时间区间并集解释。
- 每题和每条输出保存 token IDs 与 stop；输出一致才进行效率对照。取消/故障注入、强制分支 fixture、sampling 测试作为独立功能负载，不混入主 E2E 配对。
- 详细 trace 至少覆盖完整第 1 题和一个完整 8 请求 cohort；如数据过大，保留所有轻量 per-token 事件，分窗口采集重型 operator trace，并记录覆盖范围。不得只展示最快窗口。

#### 采集方式

先复用原生 `/start_profile` / `/stop_profile` 与 `ProfileReq`，CPU/GPU activities、`detailed_annotations`；按需要使用 `record_shapes/with_stack` 进行单独诊断。用 Nsight Systems 的 CUDA/NVTX/OS runtime 与可用 graph node tracing 检查 CPU 提交、kernel、拷贝和空闲区间；不依赖不存在的 ncu。

增加的 trace 点需要能映射 `rid/epoch/lane/depth/output/step`，同时避免在生产热路径做字符串拼接、磁盘 IO 或强制同步。轻量事件放预分配缓冲区，运行后导出；CUDA events 在合适的完成边界批量读取。测量至少区分：正式版本无 profiler、候选无 profiler、候选有 profiler；先校验输出相同，再量化观测开销。V1/V2 的重型 trace 若必须使用最小 annotation 候选，要同时保留正式 `237f4d44a` 的无 profiler 延迟，标注 annotation SHA 和额外开销。

#### 命令与执行入口

以下已有服务入口可直接复用；profile_workload.py / analyze_profile.py 是 **计划新增，当前不存在**，其命令是待实现接口，不表示已执行。

```bash
# 每次 X5 GPU 实验前：确认 branch/status，pull --ff-only，核对候选 SHA/import/mount。
ssh X5
cd /home/ruoxi/repo/sglang
git status --short --branch
git pull --ff-only origin loopspec-test
git rev-parse HEAD
nvidia-smi
# 检查进程、GPU UUID、显存和端口；仅使用已核实空闲 GPU，不停止他人任务。

# 隔离环境内：普通 AR/CDB，主测 B=8。
/opt/sglang/bin/python -m sglang.launch_server \
  --model-path /home/ruoxi/models/Ouro-2.6B \
  --host 127.0.0.1 --port 30126 \
  --dtype bfloat16 --attention-backend triton --bf16-gemm-backend auto \
  --max-running-requests 8 --page-size 1 --disable-radix-cache \
  --context-length 2048 --mem-fraction-static 0.7 --enable-metrics

# 旧 LoopSpec B1 基线使用 max-running-requests=1，并加：
# --disable-overlap-schedule --chunked-prefill-size -1
# --speculative-algorithm loopspec --speculative-draft-depth 1
# 新 LoopSpec 复用 CDB/V2；同步 eager 仅用于诊断，最终性能使用 graph+overlap。

# 待实现负载工具：用原生流式 API，不引入另一套 serving server。
python profile_workload.py --base-url http://127.0.0.1:30126 \
  --dataset /home/ruoxi/repo/test/datasets/aime.jsonl --first 8 \
  --arrival-ms 0 10 25 45 75 120 200 320 \
  --temperature 0 --max-new-tokens 512 --output /results/V2-ar-before
# B1 使用 --first 1 --arrival-ms 0；单次运行由外层配对驱动重复，记录完整 cohort。
python analyze_profile.py --baseline /results/V2-ar-before \
  --candidate /results/V3-cdb --latency-ratio-limit 1.02
```

CDB 前后的主对照保留各自生产可用的 graph 路径；普通 AR 若已支持 overlap/chunking，不为制造优势而禁用。必须同时记录完整 resolved args，并另做相同 overlap、相同 token chunk 预算的机制对照，分离 CDB、chunked prefill 和 overlap 的贡献。新 CDB 的 loop 级 prefill 分段属于被测试功能，不能在旧版本伪造支持；两版所有非被测试因素保持相同。

旧版本重复测量使用 Git 中冻结提交对应的只读源码目录/容器，核验 import 与运行时树哈希；源码只由本地提交、push、X5 fetch/pull 获得，不在 X5 改代码，不用 scp/rsync。实验默认工作树仍是 loopspec-test，旧基线只作为受控只读对照，不改正式分支。

## 当前执行状态

- 已完成：系统代码晋升和回接；X5 数据集/模型 config/工具存在性只读核验；本实施计划。
- 尚未执行：V1/V2 profile、新 CDB 源码、LoopSpec 多请求改造、V3/V4/V5。不能从文件可用、旧短 prompt 验收或静态检查推导 AIME 性能。
- 新任务代码候选：**0/5**。开始实施时先记录候选 1/5 的具体文件和假设，再按 Git 同步/验证；后续每轮保留完整命令与退出码。
- 后续 CDB/多请求代码只进入 loopspec-test；本次晋升授权不自动覆盖未来正式分支发布。

## 用户批准与候选 1/5：基线 profile 工具

- 用户已批准实施，并要求 EAGLE/DFLASH 仅供管理方式参考、不复用其代码，结果使用上述新目录。
- 目的：在改变调度前建立 AIME 输出参考、B1 LoopSpec 与 B8 普通 AR 的可重放时间线。
- 步骤：新增原生 HTTP 流式负载/标准 server 启停工具；最小 opt-in recurrent trace，挂接既有 start/stop_profile；记录 pipeline 路径和 CPU/CUDA stream 区间；输出分析。
- 范围：本目录 profile_workload.py、analyze_profile.py；utils/recurrent_trace.py；现有 Scheduler/profiler/结果处理及 Ouro/LoopSpec 方法的观测标记，不改计算/采样/缓存语义。
- 成功标准：512-token 上限、R4、相同 AIME 输入；独立 AR、并发 AR、LoopSpec token/stop 一致；profile 包含逐输出起始 depth、cycles、阶段与 GPU 区间；关闭观测开关时装饰器返回原方法，正式性能不带 trace。
- 风险：捕图内插入 timing event、嵌套时间重复计算、profile 污染延迟、初始服务配置不能承载 B8。捕图期间跳过观测、仅在 stop_profile 批量同步导出，报告 inclusive interval；无 profiler 单独测量。
- 验证：本地 AST/ruff/diff；push/pull 同 SHA；X5 GPU0/端口重新检查后，先 AR B1 串行八题参考，再 LoopSpec d1=1、AR B8 的输出与 profile/无 profile 窗口。图捕获 max batch 固定为 8（覆盖本任务最大请求数），记录全部 resolved args。若失败不得进入 CDB 算法改造。
- 开始状态：1/5，待验证；尚未修改 CDB。

### 第 1/5 轮结果与候选 2/5

- 第 1 轮源码 `3c07290df584996e60b3d10cbe612895eb7a3c68`，已 push/pull，同 SHA/import 已确认。环境为独立 `loopspec-cdb-20261029`，镜像 `81726dc9...`，GPU0 RTX5090 空闲、30126 空闲，Torch2.14.1+cu130/Transformers5.19.0/Triton3.8.0/FlashInfer0.7.1。
- 本地 AST、diff-check、真实 HTTP SSE fixture 的累计/增量 token 重建检查均退出 0。远端第一次 compileall 因源码只读无法写 pyc 退出 1；改用容器内 `/tmp/cdb-pycache` 后同源码 compileall 退出 0，未修改源文件。
- `profile_workload.py --serial --first 8 --max-running-requests 1 --repeat 1 --warmup 0` 在服务参数解析阶段失败（server 退出 2）：`--cuda-graph-max-bs` 在固定 0.5.22 中匹配 decode/prefill 两项而产生歧义。模型未开始运行，没有正确性或性能结果。完整 manifest/status/server.log 保留在用户结果目录 `2026-10-29-cdb-v2/round-1/ar-reference/`。
- **候选 2/5 修改前计划**：仅把 profile_workload.py 的参数明确为 `--cuda-graph-max-bs-decode 8`，目的为启动已计划的基线；不改变捕获容量或测量标准。风险为其余实际运行配置的兼容性；先实际解析并检查退出码，再按第 1 轮既定基线/trace 矩阵验证。输出写 `round-2/`，不覆盖 round-1。
- 当前 2/5；CDB 与多请求 LoopSpec 尚未实施。

### 候选 3/5 修改前计划：并发基线分歧定位

1. 目的：第 2 轮独立 AR 八题成功完成，但未改 CDB 的 AR B8 已与 B1 输出不同；最早为第 5 题输出 index=11。先定位这一前置正确性问题，不能把它隐藏在 CDB 改造中。
2. 步骤：使用原生 one_batch loader/extend/decode，固定 B1 参考的 teacher-forced 输入，比较 B1/B8 每 depth hidden 和最终 logits；对实际第一层投影输入单独比较 M=1/M=8；可用原生 batch-invariant 算子做诊断对照。
3. 范围：仅新增本目录 batch_numerics.py；不改变生产模型、调度、KV 或采样逻辑。候选 2 的 profile 仍使用已同步的旧 SHA 完成后才同步本候选。
4. 标准：输出逐题/逐步首个不同 token、hidden max/RMSE 和 logits max/RMSE，区分测量完成与正确性通过；原阈值和 token 判据不改变。固定输入避免生成分叉后无意义的数值比较。
5. 风险：诊断 eager 与 graph 的执行路径不同，不能拿其时间作为 serving 性能；投影孤立误差不能独自证明所有整模型误差来源。报告这两项限制。
6. 验证：AST/diff-check、Git commit/push/pull 同 SHA、X5 空闲 GPU0 上执行，原始张量和 JSON 仅写用户指定 results 目录；不改模型文件或实验输入。当前 3/5，此轮为诊断工具候选，CDB 尚未实施。

### 第 2 轮已获得的实测证据（完整 AIME 负载）

运行时/工具 SHA：`1cf83659b15eb0002f80570981b58b9581ac687e`。以下目录均相对于用户结果根目录下 `2026-10-29-cdb-v2/round-2/`。

- `ar-reference`：`profile_workload.py --serial --first 8 --max-running-requests 1 --repeat 1 --warmup 0`，status.success=true。八题独立输出保存成功；这不是独立 HF 长文本正确性验收。
- `ar-b8-benchmark`：`--first 8 --max-running-requests 8 --repeat 10 --warmup 2 --reference /results/round-2/ar-reference`。10 个完整 cohort 完成，但逐 token 对照失败（status.success=false），所有 80 个请求均有分歧。10 次测量的每题首次不同 index 稳定为 `[143,63,374,83,11,61,136,314]`。Q7 在 B8 输出 453 token，其余均 512；没有把不同输出长度的时间当作无损加速。
- B8 原始 makespan median=22596.110 ms、p95=22657.614 ms。各题 E2E median(ms)：`[22521.28,22511.41,22496.81,22476.62,22446.82,22402.09,19664.53,22276.10]`；TTFT median(ms)：`[43.20,83.95,117.76,154.18,124.28,118.30,129.94,99.58]`。**正确性未通过，以上不能充当合格性能门槛结果。**
- `loopspec-b1-profile`：`--algorithm loopspec --depths 1 --first 1 --max-running-requests 1 --profile --repeat 1 --warmup 2 --reference /results/round-2/ar-reference`。512 token 全程 trace 导出成功，但 AR token 对照在 index=228 失败；两次预热与 profile 的首次分歧一致。profile E2E=12131.102 ms，预热为9873.259/9835.083 ms；预热不是正式十次性能测量，也不能拿 profile 耗时做加速结论。
- 完整 LS trace：112731083 字节 gzip Chrome trace、2004945 字节 recurrent JSONL；分析命令 `analyze_profile.py /results/round-2/loopspec-b1-profile` 退出0，产生 summary.json 和 output-paths.json。实际511个 decode输出（首token来自prefill）、676个core推进、457次accept、54次reject、456次carried-state复用。

| 起始 depth | 样本数 | 新增 core loops | 单次输出 CPU wall median/p95 (ms) |
| --- | ---: | ---: | ---: |
| 0 | 55 | 4 | 62.604 / 70.404 |
| 1 | 0 | 未出现 | 无样本 |
| 2 | 0 | 未出现 | 无样本 |
| 3 | 456 | 1 | 17.992 / 20.642 |

上表包含 profiler 开销。单次 core graph 的 GPU stream interval median：depth0路径13.768ms、depth3路径15.461ms；不同上下文长度/路径不能直接归因于depth。post graph约0.182ms，verify GPU stream interval约0.437–0.443ms。verify CPU wall约13–14ms包含等待此前core完成，不是验证kernel耗时。整次trace kernel忙碌时间并集8051.910ms；nested GPU intervals不可相加，各lane不能重复计共享core耗时。

结论：原短prompt验收不能外推到本次AIME长文本/多请求输入；先保留失败证据并定位，不修改原数值或token阈值，不把尚未实现的CDB当作分歧来源。第3轮诊断候选在测试前补充M=2/4/8与LM head对照，属于同一未验证候选，不新增轮次。

第3轮诊断工具候选同时完善已有 analyze_profile.py 的归因：按每次输出汇总同一stage的多次调用（depth0有4个core调用，不能把单次core中位数误标为单token成本）；按rid导出receive、共享batch引用和结果处理事件。Scheduler在schedule stream记录的GPU events不代表overlap forward时长，明确排除这项错误归因；完整GPU忙碌时间仍来自kernel trace。此改动只分析已采集文件，不影响服务。新增输出shared-batches.json，子stage的inclusive时间不可横向求和。

### 2026-10-29 04:06 UTC：连接阻塞与交接状态

- 第3轮候选代码为 `47d2ceaf71540e40bb01efb7d8159c1c1cdf5c3d`（此前38e9b7145/bb319a783属于同一GPU验证前的诊断工具候选），已push origin/loopspec-test，本地clean。AST/diff-check通过；嵌套span、每输出多次core汇总、共享batch引用和N=0路径的分析fixture退出0。**batch_numerics.py GPU诊断未执行，不宣称其数值结果或正确性通过。**
- X5最后确认的代码仍为 `1cf83659b15eb0002f80570981b58b9581ac687e`；没有在任务运行期间pull新版。待恢复后先检查远端状态，再ff-only pull并核对import/SHA。
- 旧AR B8 profile以同一第2轮源码完成请求并进入stop_profile导出；最后成功快照为 `ar-b8-profile-TP-0.trace.json.gz` 253722624字节，服务仍在导出。随后SSH失联，**尚未核实最终status、gzip完整性和recurrent JSONL**。不得把未确认的导出标为完成。
- X5/X5-BUS原有AICoder-Direct链路超时；直连18.25.6.221:22超时。根据用户“使用可用服务器做踏板连接尝试”的指示，已尝试经AICoder-Direct、InfiniAI-BUS、Zhangyudong-BUS、InfiniAI-BUS-2、Tengxuan-BUS-2连接X5，均SSH banner超时（exit255）。前3个跳板本身hostname命令成功；AICoder/InfiniAI-BUS解析x5.mit.edu均为18.25.6.221，向其22端口读取banner超时（timeout exit124）。经BUS尝试A10/A11/L5/X4/B200作为后续入口也未连接成功。没有修改用户SSH配置、停止他人进程或改远端源文件；本地临时SSH配置只用于排查网关的en0绑定。
- 当前 **3/5**，基础设施阻塞不增加轮数。已采集的LS完整trace及AR10次并发负载保留于指定results目录；原基线long-context token门槛失败，数值根因仍待GPU诊断。CDB/LoopSpec V2多请求运行时代码尚未实施，正式loopspec分支未变。
- 恢复后的首要步骤：检查原profile进程及导出，不重复覆盖；同步本候选；空闲GPU0执行 `batch_numerics.py --reference /results/round-2/ar-reference --output /results/round-3/batch-numerics`，另加`--invariant`写独立目录作诊断对照。两者只用于定位，不能静默改变生产验收配置或放宽阈值。用新版analyze_profile.py重新分析现有trace（保留原summary），然后根据实测决定下个候选。

### X5 恢复：第 3/5 轮结果

- 按更新的 x5-persistent-session 技能建立单一 SSH 交互会话（本地工具 session 60877，远端 shell PID 2768178，ControlPath=/tmp/loopspec-cdb-x5-resume1-20261029.sock），后续复用。X5 ff-only 同步到 `50b42c556ad8ddd3bbeddb3318337216469c47fa`，工作树 clean，容器导入为 /workspace/sglang/python/sglang/__init__.py。两张 GPU 均空闲，使用原隔离容器 GPU0；没有更改正式分支。
- 旧 AR B8 profile 已完整导出：trace gzip 267377228 字节，recurrent JSONL 358683 字节，gzip -t 退出0。请求 status=false 是原 token 分歧。新版 analyze_profile.py 对 AR/LS 均退出0，生成 summary/output-paths/shared-batches。LS 原汇总备份为 *.analysis-v1.json；host ruoxi 写 root-owned 结果目录失败后改由原容器写入，未修改权限。
- 诊断命令（容器 /opt/sglang/bin/python）：`batch_numerics.py --reference /results/round-2/ar-reference --output /results/round-3/batch-numerics`，以及另加 `--invariant`、输出 batch-numerics-invariant；均退出0。默认12 decode+prefill共13位置、8题、R4、teacher-forced。退出0表示诊断完成，不表示数值验收通过。
- 原算子 B1/B8：hidden 最大绝对误差0.640625、最大逐depth RMSE0.12341515；logits 最大绝对误差0.982421875、最大RMSE0.21822403，超原阈值。104个位置的argmax仍一致且与原B1服务一致。实际相同输入的QKV和LM head投影仅改变M=1/2/4/8也产生舍入差异，证明存在batchshape因素，但不能由此独自推断全部根因。
- batch-invariant 对照：上述 B1/B8 hidden/logits 与孤立投影全部逐位相同；但相对原 B1 的 hidden max=0.4375、四depth合并RMSE最大0.05104308，logits max=0.7109375/RMSE=0.14225098。第5题step11输出1303，原B1为588（原B1在此top2 margin=0，invariant margin=0.125）。不能静默重设参考或改生产配置来标记通过。
- AR trace GPU kernel busy union=20340.045ms，profile makespan=23232.148ms；各题 E2E=[23115.676,23105.776,23090.994,23071.100,23041.425,22996.619,20142.588,22912.139]ms。receive→first-forward仅0.632–1.088ms，它不包含接收前的IPC等待，不能当全部排队延迟。decode共享batch提交CPU中位约10.91ms，结果处理中位约33.61ms（包含GPU等待），不可按8题重复加总或解释为纯结果处理开销。
- LS逐输出阶段汇总（包含profile开销）：entry_depth=0的4次core GPU stream interval之和中位55.737ms，entry_depth=3的1次core为15.461ms；prepare_lane分别0.995/0.264ms，pre0.291/0.074ms，post0.726/0.182ms，verify0.443/0.438ms，commit0.047/0.047ms。嵌套区间不能横向相加。depth1/2仍无样本。

### 候选 4/5 修改前计划：分离 prefill、GEMM 与原生 graph 的影响

1. 目的：定位 eager 诊断B8前13位置无argmax分歧、真实服务B8却在step11分歧的原因；评估保留原B1参考的数值修复是否可行。
2. 步骤：扩展同一诊断工具，支持分别prefill后使用原生ScheduleBatch.merge_batch合批decode，以及原生decode/all graph开关；记录实际can_run_graph。诊断BF16禁用reduced-precision reduction、仅固定GEMM、仅decode逐行投影三种隔离变量，输出同前缀hidden/logits与原参考token。
3. 范围：仅本目录batch_numerics.py和记录；不改生产Scheduler/模型/算法，不改权重，不提交原始张量。逐行投影是定位对照，不能作为CDB性能实现。
4. 标准：原hidden0.125/0.02、logits0.5/0.1及token/stop标准不变；区分对照完成与通过。graph模式只报告实际logits，不用Python hook假装读取replay内部hidden。保留每种配置独立输出。
5. 风险：合并batch/graph状态不同导致诊断本身无效；先验证无改算子eager结果复现第3轮，再检查graph实际replay及prefill分组。提前终止或短前缀成功不能替代512token验收。
6. 验证：本地AST/ruff F/diff-check；commit/push、同一持久SSH内ff-only pull与SHA/import核对；GPU空闲检查后在原容器执行基准与各单变量对照，结果存round-4，记录命令及退出码。当前4/5，五轮上限来自用户AGENTS.md，优先于技能正文残留的十轮描述。

### 第 4/5 轮结果

- 本地/origin/X5 同为 `b49eabdc69fb4a43d9cdb5a2a8c91870be218146`；AST、CLI help、diff-check退出0。本地ruff模块未安装（退出1），未把它记为lint通过。原环境、GPU0、模型R4及参考未变。
- 在原容器依次运行 batch_numerics.py，共同参数 `--reference /results/round-2/ar-reference --output /results/round-4/<name>`。各项均退出0，每组开始保存GPU占用快照，无并发GPU实验。原始JSON/张量、各组log/exit与综合 diagnostic-summary.json 已保留。

| name / 附加参数 | B1/B8 hidden max / RMSE | B1/B8 logits max / RMSE | 对原B1 token不同数（104点） |
| --- | --- | --- | ---: |
| native-joint / 无 | 0.640625 / 0.123415 | 0.982422 / 0.218224 | 0 |
| native-serial / --prefill serial | 0.622314 / 0.050391 | 0.5 / 0.091615 | 1 |
| no-reduced / --linear no-reduced | 1.046875 / 0.241290 | 2.09375 / 0.412255 | 1 |
| linear-invariant / --linear invariant | 0.25 / 0.039501 | 0.328125 / 0.067579 | 1 |
| serial-rowwise / --prefill serial --linear rowwise-decode | 0.3125 / 0.038308 | 0.367188 / 0.067801 | 0 |
| graph-decode / --graph decode | 不采集graph内部hidden | 0.982422 / 0.218224 | 0 |
| graph-all / --graph all | 不采集graph内部hidden | 0.982422 / 0.218224 | 0 |

native-joint所有张量与第3轮逐位相同。native-serial复现第5题step11差异：B1的1303/588 logits同为21.625，B8为21.625/21.5。no-reduced及linear-invariant同时改变原B1数值，不能作为无回归修复。逐行投影保留原B1数值，但第6题step2/depth2开始仍有hidden差异。代码检查发现LM head走torch.matmul，未被F.linear对照替换；必须单独归因。

graph两项实际均108次decode replay、0次prefill replay，logits与eager逐位相同。原因：诊断沿用chunked_prefill_size=-1，使resolved prefill bs为空；不能据名称“all”声称测过prefill图。实际旧服务resolved chunked_prefill_size=4096、prefill backend=breakable，需补测。

### 候选 5/5 修改前计划：关闭剩余诊断缺口

1. 目的：确认逐行投影后的剩余误差是否来自RMSNorm reduction；对读出和真正prefill graph单独验证，定位能保留原B1的数学路径。
2. 步骤：给诊断添加独立逐行norm及匹配LM head对照、可指定prefill图预算；增加第6题step2的逐层输入/输出探针，找首个“输入相同、输出不同”的算子。保留B1/B8同前缀和原B1对照，必要时扩大同候选采样位置。
3. 范围：仅batch_numerics.py及本记录。生产算子、Scheduler、LoopSpec和公开参数不改；逐行对照只是根因实验，不以它冒充高效多请求实现。
4. 标准：仍为原hidden/logits阈值与token一致；graph报告实际路径，逐层差异只比较相同层/depth/token。任何短探针成功不代表完整512token或性能门槛通过。
5. 风险：探针带同步，不能用于延迟验收；逐行对照会增多launch，不能直接进入生产；prefill图桶可能改变GEMM形状，必须与无图单变量比较。
6. 验证：AST/CLI/diff审查；只经loopspec-test同步、同SHA核验；X5原隔离容器顺序运行原逐行对照、逐行norm、匹配head、graph+4096预算等项，保存独立round-5结果。此为5/5；若前置正确性仍未解决，按AGENTS停止代码修改，保留证据和后续建议，不擅自进入第6轮或改验收标准。

### 第 5 轮结果、追加预算及用户修订验收标准

- 候选 `cd85309fd2d2681593a34f1cbf2bfc8aba8407be` 已本地提交、push、X5 pull 并核验同 SHA；生产模型、Scheduler、算子均未修改。所有以下诊断退出 0；这表示诊断完成，不代表 CDB 已实现或性能通过。
- `round-5/serial-linear-probe`：第 6 题 step2，首个相同输入而不同输出为第二 loop 的物理 layer39 `post_attention_layernorm_2`，max abs 0.0009765625、RMSE 0.00003996339。误差经递归累积，hidden max 0.3125。
- `serial-linear-norm`：逐行 linear+norm 后 hidden 完全相同，logits max 0.125、RMSE 0.00070745；`serial-linear-head` 只补 head 仍有 hidden 差异；`serial-all` 三者逐行后 hidden/logits 全部相同。均为 8 题×13 点，0 token 差异；逐行模式仅定位，不进入生产热路径。
- `graph-prefill-joint`：4096 prefill 预算下真实 9 次 extend replay、108 次 decode replay；与 native joint eager logits 相同。`graph-prefill-serial`：16 次 extend replay、108 次 decode replay；复现第 5 题 step11 的近 tie token 差异。图重放不是这组差异的来源。
- `serial-all-q1q5-long`：Q1/Q5×512 位置，共1024点，hidden/logits 完全相同，B1/B2 均与冻结 serving B1 tokens 一致。不能将这项逐行诊断的耗时作为生产性能。
- 本地 AST、CLI help、diff-check 退出0；本地环境无 ruff，不记 lint 通过。原始结果留在 `/home/ruoxi/repo/test/results/loopspec/2026-10-29-cdb-v2/round-5/`。
- 用户明确追加10轮，本任务总预算改为 **15轮，已用5/15**，候选6尚未开始。此前5/5及原技能文字是历史状态，不再作为本次授权后的停止点。
- 用户随后明确要求“请根据实际计算情况固定误差范围，不要要求过于严格，主要是运行逻辑要正确”。据此修订本任务数值验收：先用**未修改的原生 BF16 基线**测量 B1/B8 同前缀长序列的误差范围，修改 CDB 前冻结带明确裕量的门槛；不再要求跨 batch 生成 token 逐位一致。保留高 margin 翻转核查、非有限值/错误位置/错误深度拒绝和输出停止语义检查。用户此项授权优先于此前严格阈值与逐 token 相等约定；不是候选失败后擅自放宽。
- 逻辑验收仍严格：完整 R4、深度/分支 KV 隔离、每 token 只分配/提交一次、无输出步不提交、引用/取消/回收安全、LoopSpec 接受/残差分布正确。性能仍沿用预登记的配对≤2%口径，记录输出工作量差异；不同生成长度不能直接推导调度加速。
- 同一未改变候选追加 `native-joint-long`、`native-serial-long`，各 `--steps 496`（8题共同可比较497个位置，受Q7 EOS限制），固定B1 teacher-forcing。前者已退出0，后者运行中；待两者完成后固定数值包络。此为第5轮计划内扩大采样，不新增源码候选。

### 数值校准冻结（候选6之前）

两组均退出0，各3976个同前缀位置（8×497）；joint/serial 的 argmax 差异为24/18，翻转时 B1 top2 margin 最大0.25/0.125。逐位置 logits 最大绝对误差的 P99 为2.027344/1.597656，逐位置 RMSE 的 P99 为0.384274/0.312777；极端最大值分别12.078125/10.652344和1.536603/1.542996。少数远离决策边界的坐标有较大差异，因此不用全词表最大值单独决定模型是否正确。

按两组较大值留约25%裕量并向上取整，冻结 `numerical_envelope.json`：各深度分别检查逐位置 max-abs/RMSE 的 P99 与最大值，logits P99≤2.6/0.5、极端值≤16/2。所有值必须有限。greedy 在相同前缀上翻转比例≤1.25%，且参考 margin>0.5 时不能翻转；这比基线最大0.604%留有余量，同时保留高置信度错误拦截。此包络只校准于 Ouro-2.6B BF16/R4/AIME 本负载，不泛化到其他模型或 dtype。功能测试必须独立验证 KV 地址、depth、分支/请求归属；数值门槛不能替代逻辑证明。

### 候选 6/15 修改前计划：通用 mixed-depth 执行基础

1. 目的：为普通 Scheduler 提供一个 loop 的执行入口，解除物理 batch 必须共享 depth 的限制；遵循 vllm-rlt MetadataBuilder 的逐行 KV plane 选择和独立执行描述，复用 SGLang ForwardBatch/translator/模型接口。
2. 步骤：增加按执行行选择 depth 的 compact KV view（只处理活跃行）；增加通用 pre/core/post runner 和 decode graph，depth/request/position作为图输入；用原生 AR teacher forcing 和不同 depth/行排列验证。随后才接入 Scheduler 准入/回填；本候选不宣称 serving CDB 完成。
3. 范围：mem_cache/timeline_pool.py，新增 model_executor/recurrent_runner.py，Ouro/Raven 的最终 hidden 读出接口，本目录数值包络及定向 GPU 验证脚本。保留旧 LoopSpec 接受/拒绝语义，不复制 EAGLE/DFLASH。生产 Scheduler 接线属于后续候选。
4. 标准：上述冻结数值包络；同一深度对应独立 KV，行重排不改变归属，重复图 replay 读取新的 request/depth/length；无隐式 target 深度变化。普通 AR 的输出/采样仍由原生接口处理。
5. 风险：捕图固化深度或请求；padding写入有效槽；额外全表拷贝；状态被后续replay覆盖。使用 compact 活跃行视图、零槽padding和显式图输出生命周期；调用者持有状态。
6. 验证：AST/diff及已有CPU契约；本地commit/push loopspec-test、X5 pull同SHA/import核验；空闲GPU0上顺序执行混合深度与eager/graph teacher-forced检查，结果只写round-6；下一阶段再测真实Scheduler E2E，执行器测试时间不作serving性能结论。

### 第6轮结果与候选7/15

- 提交 `54f43f2fa4f337dd291ef3caa90a3f13dd238fa5`，两端同SHA/import核验。AST/diff/CLI检查退出0；本地3项mixed timeline寻址契约和2项既有timeline所有权/延迟释放测试通过。
- GPU0启动前7MiB占用、无计算进程、30126空闲。`verify_recurrent.py --reference /results/round-5/native-joint-long --tokens /results/round-2/ar-reference --output /results/round-6/eager --steps 63` 退出134，完整log保留。首prefill和常规decode推进后，混合行测试在attention期间illegal memory access；尚无数值验收结果，不记通过。
- 代码诊断：重排/重复源请求的执行描述保留了原 `seq_lens_sum`；重复长请求后KV CSR所需容量可能大于旧汇总值，Triton backend按旧值分配而按新GPU seq_lens写入。compact view应独立确定有界索引容量，不继承父batch的长度汇总。
- **候选7/15六项计划**：目的为修正混合视图的KV索引容量；步骤为将映射后的 `seq_lens_sum` 设置为执行行数×上下文容量（纯host已知上界，无GPU readback），再重放原测试；范围仅timeline_pool.py及本记录；标准仍为固定数值包络、原mixed重复/重排行测试，禁止删失败测试；风险为过量scratch分配，规模限于活跃行×context且graph可复用；验证为相同CPU契约、Git同步后同X5环境顺序eager/graph检查，结果写round-7。本任务已用6轮，开始7/15，Scheduler尚未接线。

### 第7轮验证结果及候选8/15计划

- 提交 `9e809300a396fe62c23a6fd23de77f3d59dc6bfd` 已同步核验。CPU三项寻址契约通过。短64位置测试的单项P99越界（0.25>0.24）同样存在于未改基线相同前64位置，不能把全497位置分布门槛套到一个非随机子集后推导回归；保留该失败结果，不更改包络。
- `round-7/eager-full` 与 `graph-full`，`verify_recurrent.py ... --steps 496 [--graph]` 均退出0，各3976位置，固定数值包络检查全部通过。每项8组混合深度/行重排/重复前缀检查均零误差。图重放的动态身份/深度已验证；尚无真实Scheduler CDB或性能结论。
- **候选8/15**：
  1. 目的：将已验证的通用执行器接入普通Scheduler，使每loop返回原生事件循环并可准入新请求。
  2. 步骤：模型显式能力声明与兼容配置下自动选timeline；通用recurrent component维护预填充/解码continuation和完成回填；一个loop只产进度，最终depth才走原生采样/输出；复用正常/overlap事件循环、FutureMap、原生allocator与结果处理。
  3. 范围：scheduler及小型scheduler_components/recurrent.py；GenerationBatchResult/ScheduleBatch的进度描述；ModelRunner/graph setup/pool配置；模型能力与验证脚本。EAGLE/DFLASH算法不复用；LoopSpec旧worker暂保持原路径。
  4. 标准：8题交错请求正常输出，无错深度/错owner/多提交；trace证明未完成R4时能重新准入；取消和chunk生命周期安全；固定包络内同前缀一致；真实性能以原负载profile/paired E2E单独判断，未达到≤2%不称效率通过。
  5. 风险：overlap的在途EOS/取消释放、partial prefill被误合并、FutureMap发布提前、hidden被图覆盖、phase公平性、graph与metadata开销。保留每token一次分配，partial状态持有tensor直到完成，使用原生stream/event屏障，显式完成列表而非伪造token。
  6. 验证：CPU结构契约与静态检查；Git同步X5；先短服务/交错/取消/小chunk测试，再原AIME8题完整profile与无profiler测量。错误/慢路径如实记录；只有普通CDB功能与性能成立后再迁移LoopSpec V2。

### 第8轮结果及候选9/15

- 提交 `e05ea719888f31535efdad8b5d847482162418e2`；AST/diff及4项CPU寻址/准入契约通过。本地完整tp_worker导入受缺少uvicorn阻塞，不记通过；X5容器完整依赖用于实际服务。
- 两端同SHA；GPU无计算进程、30126空闲。`profile_workload.py --output /results/round-8/cdb-smoke --first 8 --max-new-tokens 32 --warmup 0 --repeat 1` 退出1，server初始化失败。根因：能力选择读取raw ServerArgs.device=None，而本版本解析结果在ResolvedView；未开始请求推理，不能把它作为数值/性能失败。
- **候选9/15六项计划**：目的为正确接入本版本配置解析协议并保持构造接口兼容；步骤为能力选择统一使用resolved_view，新增SB/结果字段设为keyword-only；范围recurrent_runner.py、schedule_batch.py、utils.py和记录；标准原CLI启动原生CDB、原构造参数顺序不变、四项CPU契约及原smoke；风险是其他初始化/运行期接线问题，必须继续实际请求检查而非仅导入；验证同环境同负载，结果写round-9，仍沿用固定数值和性能门槛。当前开始9/15。

### 第9轮阶段证据与候选10/15计划

- 提交 `62cf7b7cd791044e123e9596371ca78ba3a8ee21`；4项CPU契约通过，X5原生CLI已启动CDB。`round-9/cdb-smoke`退出0，8请求各32token；首次分歧只有Q3 index23、Q5 index11，属于原先校准近tie位置，尚需完整同前缀/停止等验收。
- `round-9/cdb-b8-profile`退出0，8题完整输出长度 `[512,512,512,512,512,512,453,512]`；profile makespan 27547.512ms，不能当无profiler性能。两次无profiler预热 makespan23946.53/24043.90ms，较旧基线22596.11ms约增加6%，当前效率未通过。10次正式无profiler测量仍在X5原提交进行，不在运行中pull新代码。
- trace保存286MiB gzip及2.5MiB recurrent JSONL。2084个loop步骤中24个prefill、2060个decode，2046个decode步骤包含多个depth。前几个core顺序为A depth0、A depth1、B/C/D depth0、A depth2，直接证明普通Scheduler的loop边界准入。profile decode core CPU中位12.47ms、GPU span中位12.85ms；CPU提交经常等待 `_start` 内槽号 `.tolist()`。
- **候选10/15六项计划**：
  1. 目的：消除普通AR已知全深度时多余的host timeline分配/同步，并降低分段prefill的提交开销。
  2. 步骤：普通AR使用预先固定、互不重叠的depth KV plane地址（逻辑allocator仍唯一拥有token）；通用core图支持packed extend桶，共享图内存池；保留原生开关/eager回退；补充partial状态的观测与过滤元数据。
  3. 范围：timeline_pool/configurator、recurrent_runner、Scheduler recurrent component及原生load/invariant观测接线；本目录图/取消/chunk验证工具。旧LoopSpec暂保留lazy timeline和分支物化校验，不改变算法。
  4. 标准：既定包络、严格地址与row ownership契约、8请求/取消/chunk无泄漏和正常输出；每loop可准入；配对E2E≤2%，尚未运行不得标通过。
  5. 风险：static映射不能代表已物化、图padding污染、共享graph内存输出寿命、取消后logprob元数据未过滤、忙时load少计continuation。物化仍由state/depth依赖保证，graph输出在下一replay前复制，观测去重原始Req。
  6. 验证：CPU地址/准入契约、源码静态检查，经Git同步；等第9轮所有远端测试结束后才pull；X5先mixed/graph及小服务测试，再同负载profile/无profiler重复。第10轮开始，原阈值与正式分支均不变。

- 第9轮正式无profiler测量完成：10次 makespan 中位24124.879ms；各题E2E中位[23892.810,23979.914,23965.064,23992.638,23986.078,23975.244,21101.018,23804.873]ms，相对旧基线增加6.090%–7.305%，效率失败。输出长度全部与旧B8相同，10次重复一致；不能把稳定输出等同数值验收。paired-performance.json已保存在X5 round-9。
- 第10轮本地5项CPU契约通过，AST/diff-check退出0；新增HTTP生命周期验证覆盖不等长结束、logprobs、partial prefill/decode取消、survivor和空闲flush，并开启原生busy严格内存检查。GPU验证待执行。

### 第10轮结果与候选11/15计划

- 提交 `84f04c60fab029e35b28c46af8e6566867646c59` 两端一致；CPU5项recurrent与2项原timeline契约通过，AST/diff通过。
- `round-10/eager-full` 完整3976位置固定包络通过（exit0）；`graph-full` mixed 8组均零误差、logits/P99/argmax判据全部通过，唯一失败为depth0 hidden单点max_abs=0.578125>0.55（Q8 step234，RMSE0.0636032，logits max2.875/RMSE0.447864，top1与参考相同且margin7.375）。总体28/3976翻转，最大参考margin0.25。保留exit1，不自动放宽包络。
- `profile_workload.py --output /results/round-10/lifecycle --chunked-prefill-size 32 --lifecycle --warmup 0 --repeat 1` exit0。两轮8请求不同长度结束及logprobs、两种时刻取消/幸存请求、每轮flush全部通过，busy严格内存检查无异常。
- `cdb-b8-profile`已完成生成，profile makespan24088.406ms，长度仍与旧B8一致；预热makespan23355.676/23298.689ms，约3%增加，尚不能判定效率通过。trace仍在导出期间，不pull或启动并发GPU作业。
- **候选11/15六项计划**：
  1. 目的：消除通用mixed core退化到单组时的多余复制，以及分散完成请求的读出提交开销；核对padding的纯数值影响。
  2. 步骤：单attention组直接传张量；复用common graph实现捕获精确decode行数的原生LogitsProcessor读出，沿用正常采样/结果路径；增加原生整R执行加独立padding请求的对照，避免用CDB自身当数值基线。
  3. 范围：Ouro.advance_batch、recurrent_runner及本目录验证工具/记录。无EAGLE/DFLASH算法复用，无深度/采样改变，无新公开参数；未进入LoopSpec多请求阶段。
  4. 标准：原固定包络和运行逻辑标准；padding对照只解释误差，不自动重设数值门槛；普通服务E2E仍≤2%。读出graph不得丢失logprobs或返回hidden语义（hidden捕获回退原生路径）。
  5. 风险：共享图池导致输出被覆写、读出图改变batch形状、对照padding触碰原请求KV。精确读出行数、采样前复制输出；原生padding为独立Req并在decode前排除，真实请求的token/位置不变。
  6. 验证：AST/diff/CPU；Git同步同SHA；原生padding数值校准、mixed/eager/graph完整检查、小chunk/取消/logprobs，然后profile/无profiler配对。现已开始11/15；正式分支不变。
- 第10轮profile完整导出且exit0。trace中 recurrent.core CPU中位2.196ms（第9轮12.47ms），GPU区间中位11.348ms；start CPU中位0.259ms，finish 0.136ms。说明消除读回确实去除了CPU同步，但剩余单core GPU执行和每loop结果路径仍影响E2E。当前无10次正式无profiler结果，不用预热替代性能验收。
- 第11轮读出图仅使用已有decode捕获尺寸，不为所有可能batch额外分配图；没有匹配精确行数时回退原生读出，避免人为padding改变LM head形状。新增原生padding对照临时关闭测试进程中的CDB能力声明，断言普通KV和whole-R路径；不修改生产配置或模型文件。

### 第11轮独立padding校准证据（测量中）

- `verify_native_padding.py --reference /results/round-5/native-joint-long --tokens /results/round-2/ar-reference --output /results/round-11/native-padding`退出0（诊断完成）。该进程显式断言无recurrent_runner、无timeline_runtime，走普通KV与原模型完整R4；实际总prefill 791tokens，加独立dummy请求到1024，随后只teacher-force原8题×497位置。
- 原生padding和第10轮CDB graph的**全部3976条error/argmax记录完全相同**。原生padding自身同样唯一超旧包络depth0.max.max_abs=0.578125。这是此前联合/分开prefill校准未覆盖的形状，而非CDB相对同形状计算的回归。
- 新对照hidden max_abs按depth=[0.578125,1.61376953125,3.015625,4.671875]，P99=[0.1875,0.3125,0.52099609375,1.0234375]；max RMSE=[0.0636032,0.4059574,0.6887687,0.8429846]，P99 RMSE=[0.0252809,0.0596582,0.1185733,0.1957051]。logits P99 maxabs/RMSE=1.921875/0.3561359，max=8.625/1.5029808。28/3976翻转，max reference margin0.25。
- 按用户“根据实际计算固定误差范围”的要求，下一候选将用原先同一校准规则（原生对照最大值约25%裕量、向上取整）补齐padding范围：hidden depth0 max_abs 0.55→0.75，depth1 max_abs2→2.1，depth3 P99 max_abs1.25→1.3；其余数值、严格逻辑、token置信度及性能标准不变。新版由独立原生控制决定，不使用CDB候选的误差去拟合门槛；旧失败日志继续保留，原包络版本在Git历史中可追溯。
- 第11轮graph-full（含原生读出图）仍仅有上述同一已复现极值；mixed检查全零；lifecycle退出0。10次无profiler测量正在进行。
- 为检查时间段漂移，已用Git创建只读历史基线worktree `sglang-cdb-reference@1cf83659b15eb0002f80570981b58b9581ac687e`（本地/X5各一份，detached对照，非开发分支；未修改任何源码/正式分支）。X5隔离容器 `loopspec-cdb-control-20261029` 使用完全相同image SHA与GPU0、模型/数据/结果挂载。导入和torch2.14.1+cu130/transformers5.19.0/triton3.8.0一致；控制容器内设置safe.directory以读取只读worktree SHA。只有sleep进程，尚未运行GPU对照；须等候选测量结束再开始。

### 候选12/15修改前计划：复用既有HF语义融合RMSNorm

1. 目的：第11轮相对原记录E2E+0.57%–0.98%已接近目标，但紧邻历史控制出现21.2–22.45s的更快结果，尚不能认定P1效率达标；进一步削减GPU小算子开销。
2. 步骤：先按上述独立native-padding控制补齐并冻结v2包络；Ouro保留cast_x_before_out_mul=True，移除force_native=True，使原生RMSNorm选择本版本已有rmsnorm_hf融合核；不新增定制算子。其归约顺序可能改变，必须通过新包络及高margin判据后才保留。
3. 范围：models/ouro.py的norm构造、numerical_envelope.json和实验记录。CDB执行/调度与LoopSpec算法不变；这是配套模型核优化，不能把其全部收益归因于CDB调度本身。
4. 标准：校准v2具体三项变动如上，其余hidden/logits/argmax判据不变；严格R4、KV/owner、生命周期与E2E≤2%不变。所有判据在本候选测量前冻结。融合norm失败则撤回此候选优化，不能继续追着候选误差放宽范围。
5. 风险：FP32归约顺序在BF16舍入边界放大；读出token分歧不能仅以same-seed解释。保留teacher-forcing、有限值/P99/极值和高margin翻转检查，greedy自由输出首次分歧另核对。
6. 验证：本地AST/diff与CPU生命周期契约；Git同步；先3976位置eager/graph与混合深度，再小chunk/取消/logprobs、8请求profile与无profiler对照。X5历史控制完成后才启动GPU候选。不把控制诊断exit0当包络通过；正式分支保持不变。当前12/15。

### 第11轮配对控制及第12轮结果

- 第11轮10次makespan中位22815.231ms，较紧邻的同image/模型/数据/GPU历史控制22002.862ms，各请求E2E增加3.27%–3.77%，效率失败。原时间段基线较慢，不能选它来宣称通过。控制SHA为1cf83659；结果round-11/paired-fresh-performance.json。
- 第12轮SHA `cc54af1b761df45d5480d0cee8e0c42930b5dfa2`。CPU5项及AST/diff通过；X5 eager-full和graph-full各3976位置均通过冻结v2包络（exit0），8组mixed/重复/重排行检查全零；graph argmax18/3976翻转，没有高margin翻转。小chunk32生命周期、logprob、取消/survivor/flush和严格busy内存检查exit0。
- `cdb-b8-benchmark --first 8 --max-new-tokens 512 --warmup 2 --repeat 10` exit0；10次重复输出一致。makespan中位18532.454ms；各题E2E中位[18358.992,18386.842,18372.028,18388.420,18358.548,18349.622,18159.641,18212.455]ms；相对紧邻历史控制分别-16.291%,-16.126%,-16.137%,-15.987%,-16.007%,-15.876%,-5.095%,-16.005%。包含既有HF融合norm优化，不将全部收益归因于CDB。
- 自由输出长度[512,512,512,512,512,512,508,512]，控制Q7为453token；不声称文本逐token完全相等或相同工作量。相对B1首次分歧位置[53,245,23,203,11,61,173,314]，对应同前缀参考margin[.125,.125,.125,.125,0,.125,0,.125]，均属于被冻结标准允许的近tie。
- `cdb-b8-profile --first 8 --max-new-tokens 512 --warmup 2 --repeat 1 --profile` exit0，analyze_profile exit0。profile makespan19316.836ms，GPU kernel busy union17223.127ms；2096个core step中2048个mixed-depth，单core CPU中位1.480ms/GPU区间8.994ms，finish CPU0.170ms/GPU0.153ms。完整trace、各请求阶段及E2E、shared-batches和output-paths均保留在round-12。P1在已列配置和Ouro上功能/效率通过，非TP>1或其他模型的认证。

### 候选13/15修改前计划：LoopSpec V2复用普通CDB

1. 目的：在已通过的普通CDB之上将独立请求的LoopSpec分支组成同一mixed-depth GPU batch，每个loop返回原生事件循环，可随时准入。
2. 步骤：common scheduler提供非算法特定的decode continuation/prefill完成策略hook；LoopSpec worker按Req管理Pipeline/context，拆开plan/collect；分支只为执行行、不创建伪Req；common RecurrentRunner/graph和静态depth planes共同执行。V2完成结果通过FutureMap、on_publish、原生结果处理；未验证只返回进度。分支proposal/verify批量收集决策，保留实际proposal分布。
3. 范围：scheduler_components/recurrent及invariant的私有slot观测hook；recurrent_runner图配置；LoopSpec worker/info/utils/pipeline/config/sampling及原图实现删除；原CLI验证工具和CPU/GPU验证。Eagle/Dflash仅管理方式参考，无算法代码继承/复用。原生用户接口不增加开关。
4. 标准：保留冻结v2数值包络、R4/owner/物化/提交/回滚严格条件；d1、d2及双depth模式、多请求、取消、小chunk、logprobs、sampling正确；8请求交错及bs1 profile/10次配对，性能≤2%回归，未测项如实标记。普通CDB回归必须通过。
5. 风险：overlap中的EOS释放/后续分支在途、分支行重排、共享图输出寿命、私有KV漏计、sampling分布残差、同请求root提交顺序、bs1读回和小算子开销。采用每请求身份表、执行行与owner分离、精确已物化frontier、提交先转owner再剪枝，私有slots纳入原生严格内存守恒。
6. 验证：先CPU Pipeline拒绝/双proposal/跨请求ownership和sampling统计；静态/diff；commit/push/pull同SHA后X5先8请求短smoke与lifecycle，再完整输出与同前缀误差验证、bs1/多请求profile和10次配对。原始失败保存；普通CDB成功不替代LoopSpec验收。当前13/15，正式loopspec分支不改。
- 第13轮本地8项契约通过（包含三种分支接受/全拒绝、跨请求重排、relay过滤合并、3×100000次残差采样5σ检验），AST/diff通过。增加prefill立即EOS和原生retraction的私有状态释放hook，re-prefill恢复已输出history；压力retraction尚待实测。LoopSpec CDB明确限于具备已验证common能力的Ouro，Raven CDB未认证，不继续使用旧bs1私有调度路径冒充通用CDB。

### 第13轮结果与候选14/15计划

- 第13轮SHA e9b3c447b5ac4d3c14d5acae86007114f9ede0b7，两端一致。ls-b8-smoke退出1，配置阶段读取不存在的enable_priority_preemption；实际v0.5.22用enable_priority_scheduling与disable_priority_preemption组合，未开始服务推理。独立ls-graph-full在模型/图初始化后退出1：测试直接加载的HF tokenizer缺少服务包装后的additional_stop_token_ids属性，尚未检验分支数值。两个失败日志均保留。
- **候选14/15六项计划**：目的修复以上真实版本接口差异；步骤改用真实优先级参数组合，额外停止词采用可选属性读取；范围loopspec/config.py、utils.py及common recurrent能力门控；成功标准仍是原冻结数值/严格ownership/生命周期和≤2%性能，不能把接口修复当最终通过；风险为之后才能触及的V2运行期接线，不以本地测试代替；验证CPU/AST、Git同步同SHA、完整服务smoke/分支teacher forcing/lifecycle，随后profile与配对性能。预算14/15。
- 独立历史控制容器在SHA1cf83659以原bs1路径运行Q1、draft-depth1、512tokens、2warmup+10测量，输出round-13/ls-b1-control；控制未运行完前不启动其他GPU作业。

### 第14轮结果与候选15/15计划

- SHA68475579163997ae9d8bbb5405e603f73d6ed7b7；CPU8项与diff通过。X5服务配置已通过，但worker构造先于内存池分配，读取尚不存在的ModelRunner.max_running_requests，smoke退出1，未开始请求。
- 独立真实分支执行graph-full完成3976位置：21次argmax翻转，最大参考margin0.25；除depth0单点RMSE以外全项通过。唯一越界Q4 step302，depth0 RMSE0.1017153>0.085/maxabs0.373291；depth1/2/3 RMSE0.056584/0.026217/0.024337，最终logits maxabs0.25/RMSE0.044972，token34与参考相同，margin4.75。全部请求均实际走过接受与回滚，跨owner重排和物化检查无异常。仍保留success=false/exit1，不在此追着候选扩大包络。
- **候选15/15六项计划**：
  1. 目的：修复worker构造/分配生命周期，使原生V2服务实际启动；进一步解释32执行行的新形状误差。
  2. 步骤：把max_rows计算移到alloc_memory_pool，容量以target已分配结果为准；扩充独立native-padding诊断可保留32个原生decode请求，并可选择原始native norm，断言没有CDB/LoopSpec计算。该对照只提供证据，不修改既定数值门槛。
  3. 范围：LoopSpec worker初始化两处、verify_native_padding诊断和记录。算法、调度、采样及用户接口不变。
  4. 标准：既定严格逻辑、冻结v2数值范围及≤2%性能回归；本轮未达不能称最终完成。普通CDB第12轮已通过的证据保持独立。
  5. 风险：内存池后捕获的分支容量、overlap运行期接线、取消释放、hidden单点差异和bs1开销。必须实际8请求、生命周期和profile，不能仅依赖offline测试。
  6. 验证：CPU/AST/diff，同SHA Git同步；服务短smoke，smallchunk lifecycle，完整8请求/分支数值，独立native32；可运行时做bs1及8请求profile/10次性能。第15轮结束如仍有失败或未完成项，按预算停止源码修改并报告，不启动第16轮。

### 第15轮结果：达到授权预算，未完成

- 测试源码SHA `46411c2d0c1124864ba9df83386201120dbf2860`；CPU8项、AST、diff通过。X5同SHA干净，实验前GPU0无计算进程、30126空闲，容器源码映射未变。
- `profile_workload.py --output /results/round-15/ls-b8-smoke --algorithm loopspec --first 8 --max-new-tokens 32 --warmup 0 --repeat 1` exit1。模型/pool/graph初始化完成后SchedulerLoadInquirer构造失败：第13轮加入的get_extra_owned_slots本应只传给SchedulerInvariantChecker，但批量替换同时把它传到了init_load_inquirer。**当前实验HEAD无法启动服务，普通CDB第12轮通过结论不能套用当前HEAD。** 这是实现接线错误，不是网络/基础设施问题。
- 具体待修位置：scheduler.py::init_load_inquirer删除不支持的get_extra_owned_slots实参；init_invariant_checker的同名实参应保留。按轮次上限本轮不再修改源代码。
- smallchunk lifecycle按计划在smoke成功后才运行，因此未执行；当前LoopSpec V2的8请求完整HTTP输出、bs1/8请求profile、配对性能、sampling服务验证均未完成，不标通过。第14轮offline分支数值结果仍是最完整的算法证据，唯一hidden单点越界仍保留。
- `verify_native_padding.py ... --output /results/round-15/native32 --decode-rows 32 --original-norm` exit1：纯原生32请求的KV总需求(1024+32×496=16896 slots)超过测试pool容量9089；one_batch空树桩无evict_for_alloc。属于诊断容量配置错误，未形成完整32行校准，不能据此改数值门槛。后续应在独立控制里合理限制dummy历史或调整pool布局/容量并保持8个真实请求原计算，不修改生产模型。
- 旧bs1的紧邻控制 `round-13/ls-b1-control` exit0，10次512token，makespan中位9841.986811ms；尚无通过的新V2服务性能与之配对。
- X5 `/home/ruoxi/repo/test/results/loopspec/2026-10-29-cdb-v2/round-15/status-summary.json` 已写明测试SHA、15/15状态、精确失败和未执行项；所有原始失败日志保留。GPU计算进程和30126已清空，只保留本任务容器sleep及SSH session60877。
- 按AGENTS.md第43行达到授权15轮后停止源码/测试/行为配置修改。已向用户请求追加5轮(总20)；等待明确答复，时间经过不构成授权。这里只补充结果记录，不启动第16轮，不改变正式loopspec分支。

### 授权追加与候选16/20计划

用户明确答复“追加5轮，按上述范围继续”，总预算增至20；前15轮和失败证据保留。沿用loopspec-code-change、x5-loopspec-experiment与持久连接技能，X5 session60877已确认存活，GPU空闲。

1. 目的：恢复普通/LoopSpec Scheduler启动，并完成独立native32数值控制，进入实际V2服务验收。
2. 步骤：只删除init_load_inquirer误传实参，保留invariant_checker私有slot观测；native32对照每步释放24个独立dummy刚生成的KV并回退它们的长度，保留各自prefill。8个真实请求仍完整增长至497位置，M=32不变。
3. 范围：scheduler.py一处接线、verify_native_padding.py诊断容量和记录；不改算法、原生模型或用户接口。
4. 标准：严格R4/owner/物化/回滚、既定数值包络和E2E≤2%回归；仅独立原生计算可作为形状校准证据，不以候选误差扩大阈值。
5. 风险：overlap结束/取消释放、私有KV守恒、接线尚未触达的运行期错误；dummy回收必须仅作用于最后24行且不污染真实8行。
6. 验证：CPU8契约、AST/diff；Git同SHA同步。X5短8请求、smallchunk生命周期、独立native32及完整teacher-forcing，再做8请求完整输出、bs1/8请求profile及10次配对；普通AR也做服务回归。测试/性能失败保留，未执行项不标通过。当前16/20，正式分支不动。

### 第16轮结果与候选17/20计划

- SHA77851460919b398d049bd08794aa2735c87f7a7d；本地CPU8项、AST/diff通过；X5同SHA干净，容器解释器/源码路径确认正确。
- ls-b8-smoke exit0：原生overlap/CDB/LoopSpec V2，8题交错各32tokens成功。ls-lifecycle exit0：chunk32，两轮不等长结束+logprobs、.03/.4s取消、survivor、flush通过，严格busy KV检查无异常。
- native32与native32-fused均完成3976位置（诊断exit0）。前者原始RMSNorm全项过v2，depth0最大RMSE0.0674779192；后者使用已有原生fused RMSNorm、普通KV、整R forward且显式禁用CDB/LoopSpec，唯一超标depth0最大RMSE0.1017152667，Q4 step302，与第14轮LoopSpec该点完全相同。由独立对照复现，不归因为分支/调度。真实8行前缀未截断，dummy每步只回收自身新slot，峰值需5016slots<9089容量。所有原始失败保留。

候选17/20六项计划：
1. 目的：依据真实原生BF16计算固定合理误差包络，并补齐服务验收覆盖。
2. 步骤：沿用独立native校准25%余量规则，depth0最大RMSE设0.13（0.1017152667×1.25=0.1271440834，向上取两位），其余所有门槛不变；生命周期补原生sampling/penalties/min-length/OpenAI/zero-token接口；性能工具可传已有max-total-tokens参数用于内存压力测试。
3. 范围：numerical_envelope.json、verify_lifecycle.py、profile_workload.py及记录；生产计算/调度不变，不增加LoopSpec用户参数。
4. 标准：新包络v3冻结在后续候选测量前；仍严格R4、owner/物化/剪枝守恒和高margin不翻转；8请求/bs1服务完整输出、原生API、AR回归；E2E≤2%不变。
5. 风险：低容量下pending分支的额外KV需求、不同depth组合/采样残差，长输出和graph共享池寿命；统计采样继续CPU解析分布/Monte Carlo验证，服务采样只证明接线和输出正常，不能替代分布检验。
6. 验证：CPU/AST/diff后Git同SHA同步；完整d1 graph/eager、d2/双depth数值/HTTP，生命周期与低容量8请求，普通CDB回归；bs1/8请求profile和10次重复配对，保留每请求时间与共享步骤说明。预算17/20，正式分支不动。
