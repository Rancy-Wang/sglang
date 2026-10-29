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
