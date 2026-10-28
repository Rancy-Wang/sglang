# LoopSpec → SGLang v0.5.22 迁移研究与实施计划

日期：2026-10-25（Asia/Shanghai）。最终状态（2026-10-28）：候选16/25源码7966c680135e38699eb2d5273692adf8111ff289已通过计划内Ouro-2.6B验收：native/compact原16点数值阈值、10组服务矩阵100次主请求、生命周期/采样/graph测试及两组配对性能检查全部通过。达到标准后停止迭代。代码在loopspec-test；正式loopspec分支未晋升。Raven接口与合成fixture有验证，真实Raven checkpoint仍未验收。历史段落保留各轮当时状态，以本行及末尾记录为准。

用户请求：判断 SGLang 对 loop transformer serving 的支持，分析官方 LoopSpec 如何接入 SGLang，设计完整迁移方案与 Ouro-2.6B 单请求验证方案。

## 1. 已核对的版本与证据范围

- 本地 SGLang：`loopspec-test`，HEAD `baa9befa381511cd4d56f83980951e5a45fffc6d`，即项目指定 v0.5.22；检查时工作树干净。origin 为 `git@github.com:Rancy-Wang/sglang.git`。
- X5 SGLang：同分支、同 SHA；只读检查时工作树干净。
- 官方 LoopSpec：`main`，HEAD `41aa710ed7c9ce908cffa03597be5d1df19d7b56`；检查时工作树干净；本次 `git ls-remote origin HEAD` 返回同一 SHA。
- 官方依赖：Python 3.11、sglang 0.5.14、torch 2.11.0、transformers 5.8.1、flashinfer-python 0.6.12。不能直接覆盖 v0.5.22 的开发环境。
- 本次读取本地源码、官方 GitHub 页面、项目已有论文文本，以及 X5 模型 config/模型源码；没有加载模型、安装推理环境或执行 GPU 测试。
- 本文对 SGLang 支持范围的结论限定在已核对的 v0.5.22，不将最新版 main 作为基线。

官方来源：

- [LoopSpec 官方仓库](https://github.com/kaist-flexml-lab/loopspec)
- [固定版本 SGLang IQuestLoopCoder](https://github.com/sgl-project/sglang/blob/v0.5.22/python/sglang/srt/models/iquest_loopcoder.py)
- [Ouro-2.6B 配置](https://huggingface.co/ByteDance/Ouro-2.6B/blob/main/config.json)
- LoopSpec 论文的流水线与第二候选说明见工作区 `references/papers/loopspec-2609.17184.txt` §4.2–4.3；本计划的实现判断以以上固定 SHA 源码为准。

## 2. SGLang 是否支持循环模型 serving

支持部分循环架构，但不是任何循环模型都能直接加载。

v0.5.22 的 `python/sglang/srt/models/iquest_loopcoder.py` 已实现循环 forward、共享层权重、每轮独立 attention/KV 索引；`models/nanbeige.py` 也有循环执行。`configs/model_config.py` 明确将这些架构的 attention 层数按循环次数展开。因此不能说 SGLang 完全不支持 loop transformer。

本次检索该版本 `models/`、`configs/`、`speculative/` 未发现 Ouro、Raven 或 LOOPSPEC 实现。`spec_info.py::SpeculativeAlgorithm` 也没有 LOOPSPEC 内置值。`trust_remote_code=True` 允许加载远程配置/代码，不等于已经实现 SGLang 原生的循环状态、KV 和推测调度支持。

结论分层：

| 问题 | 已核对结论 |
| --- | --- |
| SGLang 能否作为循环模型的 serving 框架 | 能，有原生架构先例 |
| v0.5.22 原生支持 Ouro/Raven | 未发现，需要移植 |
| v0.5.22 原生支持 LoopSpec | 没有，需要算法与运行时接入 |
| 官方 LoopSpec 能否通过 SGLang 提供 HTTP 服务 | 有完整实现入口，但为外部扩展和定制执行路径 |
| 本项目是否已经跑通 | 没有，本次仅研究和计划 |

## 3. 官方实现如何接入 SGLang

入口不是 `sglang.launch_server`，而是 `python -m sglang_recurrent.server`。

```text
sglang_recurrent.server.cli.main
  ├─ ServerArgs.add_cli_args + --loopspec/--first/--second
  ├─ 配置解析器选择内存中的 serving architecture
  └─ SGLang launch_server(..., run_scheduler_process_func=...)
       ├─ 原生 HTTP / TokenizerManager
       ├─ 自定义 scheduler 子进程
       │    ├─ baseline: 包装原生 Scheduler，执行完整深度模型
       │    └─ LoopSpec: 自有 ZMQ 接收循环 → generate_loopspec
       │          → PipelineSchedule / BranchManager / ModelRunner
       │          → pre / recurrent / post + 分角色 CUDA graphs
       └─ SGLang DetokenizerManager → HTTP/流式输出
```

### 3.1 注册与模型实现

- `integrations/sglang.py::install_sglang_config_compatibility` 注册两个 config parser：baseline 对应 `LoopedRecurrentServingModel`，LoopSpec 对应 `RecurrentServingModel`。复用 HF parser，仅在内存中覆写 Ouro/Raven 的 `architectures`，不改原 checkpoint。
- `register_recurrent_models()` 调用 `ModelRegistry.register("sglang_recurrent.modeling", overwrite=True, strict=True)`。`modeling/serving.py::EntryClass` 提供模型类。父进程与 scheduler 子进程在各自需要配置/模型前完成注册。
- `modeling/ouro.py` 使用 SGLang 的 embedding、LM head、Llama attention/MLP、RMSNorm，以及 Gemma2 的残差流程。它重新映射 Ouro 的四个 norm 和 packed QKV/MLP 权重，不是直接调用 HF generate，也不能当作普通 Llama 层。
- `OuroRecurrentExecutor` 每轮执行所有共享层和 step norm。early-exit gate 保留权重但固定深度 serving 不执行该 gate。
- `RecurrentState` 携带 hidden、position，以及 Raven 的 injected tensor；KV 在运行时另行管理。Ouro 的 injected 为 None。

### 3.2 baseline 与 LoopSpec 是两条路径

- baseline 使用 `LoopedRecurrentServingModel.forward`：pre → R 次 recurrent → post；共享权重但为每轮设置不同逻辑 attention layer ID。
- `baseline_scheduler.py` 在进程内把原生 `Scheduler` 替换成其子类，主要增加 tokenizer 修正与 decode 计时，仍委托原生调度循环。
- LoopSpec 使用 `RecurrentServingModel.forward`，一次调用只执行 pre/core/post 中一个角色。调度器负责持久化 lane 的 hidden、循环深度、token position、分支以及实际 proposal 分布。
- `loopspec_scheduler.py` 自己读取 SGLang ZMQ 请求并发送 `BatchTokenIDOutput`。即使收到 batch request，也逐个调用 `_run_request`。它不是原生 continuous batching。

### 3.3 流水线、验证与 KV

- `pipeline.py` 组织逻辑 lane；`executor.py` 将活跃 lane 合并执行共享 core。父 token 向完整深度推进的同时，较浅 readout 已启动后继 token。
- 接受候选：提交 token 并保留对应后代已做的循环计算；拒绝：剪去错误后代，提交校正 token 并重启相应分支。不能把它改成普通“先 draft 完一段再 verify 完一段”而仍称等价移植。
- 第一提议 `q1=p_d1`；第二提议是同一位置的备用候选，来自 `normalize(max(p_d2-q1,0))`。gate 在 deeper readout 降低第一候选的概率时开启。greedy 退化为深层 argmax 与首候选不同。
- full-depth 验证第一候选的接受率是 `min(1,p_R(x1)/q1(x1))`。拒绝后的目标是 `rho1=normalize(max(p_R-q1,0))`；第二候选必须用实际保存的 residual proposal 对 rho1 验证，不能直接对 p_R 验证。第二次拒绝再从 `normalize(max(rho1-q2,0))` 采样。
- temperature/top-k/top-p 产生的实际归一化 proposal 必须保存；residual 不再次 top-p 截断。kernel 排除首候选 token 与零质量浮点兜底也要保留并单独验证，不能将兜底称为精确分布证明。
- baseline 通过逻辑 layer ID 隔离循环 KV；LoopSpec 通过 `(branch, role, depth)` 的 timeline/request row 和 slot mapping 隔离，各 lane 可处于不同深度。两种布局不能混用。
- `branching/manager.py` 的 fork 复制前缀索引并借用前缀 KV；inactive ancestor 可能仍被后代引用，必须延迟回收或转移所有权。`decode_slots.py` 与 graph 侧共同管理可重用 slot。
- `request.py::prefill` 为各循环建立完整 prompt KV；当前实现会重置共享 pool，因此依赖外部单请求执行约束。

### 3.4 无文件修改仍存在运行时 patch

`server/loopspec_runner.py::make_loopspec_runner`：

1. 从 schedule 推导各角色 graph shape 与内部 request-row 容量；外部请求数仍为 1。
2. 暂时替换 `DecodeCudaGraphRunner` 为 `RecurrentCudaGraphRunner`。
3. 暂时替换 `bench_one_batch.get_tokenizer`，然后通过 `bench_one_batch.load_model` 初始化 runner、权重、pool、graph。
4. 给该 runner 的 eager `load_batch` 绑定 bridge，保留 `recurrent_role` 和 `injected`。
5. 初始化 sampler/拒绝采样 graph，最后恢复临时模块全局替换。

另外 `integrations/flashinfer.py` 对符合条件的 BF16 小 batch linear 安装进程级 TinyGEMM 路径；tokenizer hook 保留 Ouro 的 thinking delimiters，Raven hook 对齐特殊 token IDs。

准确描述：官方是基于 SGLang 的外部 serving 扩展，结合正式扩展点与运行时 monkey patch，并非已经 upstream merge，也并非仅调用 HTTP API。

### 3.5 当前官方能力边界

LoopSpec 模式限制：单个活跃请求、Triton attention、所有并行维度为 1、page size=1、禁用 overlap、stream interval=1；不支持 PD disaggregation/PDMux，不接受 SGLang 常规 speculative 参数。

支持 greedy、temperature/top-k/top-p sampling、标准停止检查和流式输出。显式拒绝包括多模态、LoRA、sessions、return_logprob、约束解码、penalties、min_p 等选项。其他 control request 在其循环中被忽略或只能等请求结束后处理；不能宣称有完整原生取消/管理语义。

## 4. 已确认的版本差异

| 位置 | 官方依赖/做法 | v0.5.22 的迁移工作 |
| --- | --- | --- |
| scheduler 入口 | 参数含 attn_cp/moe_dp/moe_ep ranks | 当前入口使用 tp/pp/dp 与 display ranks，且先 publish runtime context；不能按旧位置参数调用 |
| 模型加载 | `sglang.bench_one_batch.load_model` | 该模块已不存在；`bench_one_batch_server.py` 不是同一内部 API；改用 worker/ModelRunner 正式构造生命周期 |
| recurrent batch 字段 | 动态 setattr + eager bridge | 显式定义字段/状态传输；覆盖 buffer registry、replace/copy、eager 和 graph |
| 执行输出 | pre/core 输出 RecurrentState，post 输出 logits | 明确 recurrent runner 输出协议，不默认普通 logits 后处理能处理 hidden state |
| speculative 接入 | 完全绕开 SGLang spec worker | v0.5.22 有 `SpeculativeAlgorithm.register`，但统一为 V2 schema，禁用 overlap 也不能回到旧 V1 |
| worker 返回值 | 自己更新 Req/发 IPC | 适配 GenerationBatchResult 的 accept_lens/new_seq_lens/next_draft_input/on_publish 等语义，不虚构 bonus token |
| KV pool | baseline 与 pipeline 两种布局 | 分开描述物理层数、循环 timeline 数、slot 数、外部请求数、内部 lane 数，避免错误乘倍与容量估计 |
| graph 与 linear | 替换全局 class/method | 改为显式 recurrent graph/linear backend 选择；适配当前 runner/graph buffer 生命周期 |

## 5. 修改前六项计划

### 5.1 修改目的

使用户 fork 的固定 v0.5.22 通过标准 `python -m sglang.launch_server` 提供 Ouro full-depth AR 和 LoopSpec 单/双候选推理，保留官方 pipeline reuse、branch KV、greedy/sampling、streaming、CUDA graph 和指标能力，消除对旧 benchmark 加载入口和进程级临时 patch 的依赖。

首个可交付验收是 Ouro-2.6B 单请求与 AR 精确 token 对齐。完整功能验收还需要分支/采样/graph/生命周期测试，不能用单条请求代替。保留官方 Raven builder/config/tokenizer/injection 接口的迁移位置；未加载 Raven checkpoint 之前只标记“代码已迁入、模型未验收”，不能称全模型覆盖完成。

### 5.2 修改步骤与依赖

**A. 版本契约和 full-depth AR。** 对齐 loader、process/runtime context、model registry、配置与 tokenizer；新增 Ouro 原生模型，固定 R=4，完成 strict 权重覆盖检查、norm/RoPE/残差/LM head 对齐。AR 用 48 个共享参数层、192 个逻辑 attention 层；核实 pool 分配依据是逻辑 attention 数而非仅物理层数。第一步先 eager，禁用 prefix cache/chunked prefill/overlap，不引入 early exit。

**B. 显式 recurrent 执行协议。** 定义 model spec、role、hidden/injected、stage/position/branch 状态。提供 `prefill`、`advance`、`readout`、`fork`、`retain/prune`、`finish/abort`。原生 AR 的完整 forward 与 LoopSpec 的分角色 forward 共享同一套权重计算实现。对相同 prefix 比较每轮 hidden/logits，建立可解释的差分路径。

**C. 单候选流水线与私有 timeline pool。** 迁入 pipeline、分支管理、metadata、slot 管理，先 d1=1、greedy、eager。保留跨 token/跨 depth 的 lane batching。LoopSpec 模式采用官方的物理 core 层 + 每 depth 独立 slot/timeline 布局；AR 的展开布局用于配对基线，不能把两者未经转换直接绑定到同一 pool。通过显式 memory-pool config 核算有效容量，所有前缀共享有明确 owner/引用关系。

**D. 原生 scheduler/worker 接入。** 新增 `LoopSpecWorker`，使用 v0.5.22 自定义算法注册或等价内置注册。在正常启动进程和子进程初始化前完成注册；公开参数 `--speculative-algorithm LOOPSPEC` 与命名明确的 LoopSpec 深度参数。采用 V2 worker schema，初期 `supports_overlap=False`。

worker 持有每个 request 的 pipeline state，把整段阻塞的 `generate_loopspec` 拆为 `advance_until_commit`：每次返回已经验证的 token，预热/重启时可在有界的循环推进后返回，保留其余 lane 状态。不能要求原生 scheduler 等待整段生成完成。对齐 `forward_batch_generation(batch, on_publish, pp_proxy_tensors)`、sequence length 发布、结果处理、finish/abort、cache release；需要时新增小范围明确 hook，不冒用 EAGLE predicate。

先保证 active request=1，恢复原生 admission/HTTP/Tokenizer/Detokenizer/请求管理。原生接口通常假定已提交前缀 KV 齐备；必须明确最后输出 token 尚未作为输入物化的状态，以及不同深度 timeline 的长度。对接失败视为关键设计阻塞，不以串行自定义 server 冒充原生接入完成。

**E. 双候选与完整 sampling。** 移植 q2 gate、残差 proposal、cascade rejection、请求独立 RNG 和 temperature/top-k/top-p。Ouro 默认 (d1,d2)=(1,2)；约束 `d1 | R`、`d1 | d2`、`1 <= d1 < d2 < R`。分别覆盖 q1 接受、q1 拒绝/q2 接受、两者拒绝、gate 关闭和零 residual。

**F. CUDA graph、优化和功能收尾。** 迁入 pre/core/post 分角色 graph、state gather/scatter、metadata 与 mapping 更新、拒绝 graph；先证明与 eager 一致。R=4,d1=1 时单候选最大 lane 数 4，双候选按官方 schedule 上界为 7；request-row 容量另算，不能把 7 当外部 concurrency。TinyGEMM 作为显式且可关闭的 backend，在支持当前环境时保留官方功能，baseline 与候选使用相同设置。

完成 streaming/EOS/stop/max tokens/取消/异常清理/连续请求/指标与文档；移植 Raven 的 pre/core/post 与 injection 差异并标记模型验证状态。多请求 continuous batching、TP/PP、PD、LoRA、grammar、DAS/DAS-Wave 不属于“复现官方现有 LoopSpec”的必需项，后续扩展需独立方案。

以上为实现依赖阶段，不是六轮预算。整个已定义实施任务最多五次“候选修改＋验证”；不得用阶段名称重置计数。

### 5.3 具体修改范围（拟新增路径，尚不存在）

以下路径相对用户 SGLang 仓库：

| 路径 | 职责 |
| --- | --- |
| `python/sglang/srt/models/ouro.py` | Ouro 原生模型、权重映射、full-depth forward |
| `python/sglang/srt/models/recurrent/` | 通用 contracts/components、共享 builder、Raven 适配；配合顶层 EntryClass 被 registry 发现 |
| `python/sglang/srt/configs/ouro.py`、`configs/model_config.py` | config 注册与逻辑 attention 层数/物理 pool 布局 |
| `python/sglang/srt/speculative/loopspec/` | pipeline/executor/request/sampling/branching/worker/注册，尽量保持算法与服务层分离 |
| `python/sglang/srt/model_executor/forward_batch_info.py`、`cuda_graph_buffer_registry.py` | 显式 recurrent state 传输契约 |
| `python/sglang/srt/model_executor/runner/` | 分角色 eager/graph runner；不再借 benchmark 模块构造服务 |
| `python/sglang/srt/mem_cache/` 中必要的 pool config 接口 | per-depth timeline 的容量与独立所有权，分支管理主体放 loopspec 包 |
| `python/sglang/srt/managers/{scheduler,schedule_batch,tp_worker}.py`、`managers/utils.py` | 必要的 V2 生命周期/结果接口适配，优先复用现有 finish hook |
| `python/sglang/srt/server_args.py`、`arg_groups/`、`runtime_context.py` | 参数、验证、进程可见运行配置 |
| tokenizer/config helper 与 `layers/linear.py` 的必要扩展点 | 特殊 token 一致性和可选 linear backend，不全局劫持 |
| `test/registered/unit/`、`test/registered/spec/` 下 LoopSpec 测试 | 算法、slot 生命周期、V2 contract 与 GPU 差分 |
| `research/loopspec/experiments/2026-10-25-native-migration-plan/` | 后续验收 harness、轻量输出与本任务记录 |

仅更改实际需要的文件；这不是要求把表中全部文件无条件改一遍。迁入代码保留来源和许可证信息。

### 5.4 预期效果与成功标准

1. 普通 `sglang.launch_server` 能加载 Ouro-2.6B；全部预期权重都已加载，不能仅对 unmatched checkpoint keys 发 warning。
2. 固定完整深度 AR 的 prefill/逐 token decode 与可信 checkpoint forward 对齐；主请求所有 generated token IDs 和停止位置一致。记录 logits 最大误差/RMSE、top-1 margin；数值容差在验证脚本写入前依据 dtype/kernel 误差规范固定，不能在看到失败后放宽。若 BF16 argmax 分歧，定位首个不同循环/层，不能仅看文本判通过。
3. LoopSpec 单/双候选 greedy 与已通过的 AR token IDs、stop reason 完全相同；并记录真实 lane 推进和 accept/restart，排除“实际上退回 AR”的假通过。
4. fork/prune/reuse/finish/abort 后无 KV 污染、双重释放或活跃前缀误释放；独立请求资源计数回到基准。
5. sampling 以分布检验通过；同 seed 是否得到相同序列不是判据。
6. graph 与 eager 输出、状态与资源生命周期一致。
7. 当前官方支持的模型接口和功能逐项有兼容性状态；只测 Ouro 不宣称 Raven 已验收。
8. 首次请求验证不设加速倍数门槛。性能是后续配对测量结果；正确但没有加速也须如实报告。

### 5.5 风险

- 权重共享与循环 KV 共享混淆；layer 维/slot 维同时乘 R 导致过量显存；pipeline 混深度 lane 错用统一 layer ID。
- Ouro 四个 norm、step norm、残差、RoPE 及 packed weights 映射错误，AR/LoopSpec 同源 bug 互相掩盖。
- 悬挂分支仍引用前缀、异步 graph 尚未读完映射，过早复用 slot 或 state bank。
- V2 worker 默认 bonus/accept count/seq_lens 语义与流水线不同；重复发 token、跳 token、重复处理最后 anchor 或提前缓存未物化 token。
- parent/child process 注册顺序、runtime_context 与 eager/graph buffer 复制遗漏字段。
- sampling 保存错分布、二次过滤 residual、seed 跨请求串用；零 residual 浮点兜底引入未量化偏差。
- RTX 5090 的 Torch/CUDA/FlashInfer 兼容及 graph 资源；不保证官方锁定 CUDA 环境可直接在 X5 运行。
- 仅一条请求通常不覆盖拒绝分支、长上下文、取消和所有采样条件；必须分别报告覆盖率。

### 5.6 验证与发布流程

实施时本地先记录候选 n/5、检查 diff，仅提交本任务文件；`loopspec-test` commit → push → X5 `pull --ff-only`；核对本地/origin/X5 SHA 与实际导入路径。不得同步到正式 `loopspec` 分支。

计划内本地检查：目标文件语法、权重键映射与纯 CPU schedule/ownership 测试、V2 result contract 测试；本机不加载 GPU 模型。具体测试文件写出后在记录中固定 pytest 命令，不将下方拟议命令描述为已经可执行。

X5 分离三个环境：原 checkpoint 的兼容 Transformers 参考环境；官方 LoopSpec 的 0.5.14 锁定环境；用户 fork v0.5.22 的 editable 开发环境。不要在同一环境反复覆盖版本。参考 repo 只读。

每轮保存 SHA、config/tokenizer hash、实际 R/dtype/backend、启动命令、PID、端口、GPU UUID、版本、退出码、stderr、token IDs、首个差异及指标。原始日志在 `/home/ruoxi/experiments/loopspec/2026-10-25-native-migration-plan/round-<n>/`；小型摘要随实验分支版本化。

## 6. Ouro-2.6B 单请求验证计划

### 6.1 本次已读到的实际模型配置

X5 路径：`/home/ruoxi/models/Ouro-2.6B`。

| 参数 | 值 |
| --- | --- |
| architecture / model_type | OuroForCausalLM / ouro |
| num_hidden_layers / total_ut_steps | 48 / 4 |
| hidden_size / head_dim | 2048 / 128 |
| attention heads / KV heads | 16 / 16 |
| intermediate_size / vocab_size | 5632 / 49152 |
| dtype / max_position_embeddings | bfloat16 / 65536 |
| tie_word_embeddings | false |
| bos / eos | 0 / 0；实际 tokenizer IDs 还需启动前一并核对 |
| early_exit_threshold | 1.0；参考 forward 必须显式核对完整四轮 readout |

config SHA256：`23b5c6942a9f0ca8619fb15acfaca93e02c169cdc4e03dbceb4d8a2df145697a`。

tokenizer.json SHA256：`fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d`。

模型自带源码 `modeling_ouro.py` 使用 `current_ut * num_hidden_layers + layer_idx` 隔离 KV，在每轮末执行 `self.norm`。参考 harness 应直接取第 4 轮 logits，明确 early-exit readout 的选择，不能只相信配置里 R=4 就排除 readout 早退。

### 6.2 唯一主请求

```json
{
  "text": "The capital of France is",
  "sampling_params": {
    "temperature": 0,
    "max_new_tokens": 32,
    "ignore_eos": false
  },
  "stream": false
}
```

使用 raw completion，避免 chat template 引入另一个变量。固定 tokenizer 与 BOS 处理并记录实际 input_ids；程序差分优先将同一份 input_ids 送入各后端。不能预设输出一定是 Paris，也不能把“答案看起来合理”当正确性标准。

“1 个简单请求”指一个 workload，各对照执行该同一请求，并非只调一个待验证服务一次就证明无损。

| 顺序 | 执行路径 | 目的 |
| --- | --- | --- |
| 1 | 原 checkpoint forward，强制四轮 readout，greedy | 独立检查权重/模型语义，保存 reference token IDs |
| 2 | 官方 LoopSpec repo baseline | 核对官方 SGLang 模型重写的输出 |
| 3 | 新版原生 SGLang AR，eager | 先建立 v0.5.22 full-depth 基线 |
| 4 | 新版 LoopSpec 单候选 d1=1，eager | 验证第一条流水线链路 |
| 5 | 新版 LoopSpec 双候选 d1=1,d2=2，eager | 验证完整 Ouro 调度路径 |
| 6 | 同样候选开启 CUDA graph | 验证优化后仍一致；graph 阶段才运行 |

原 checkpoint/HF 与官方路径若受精度影响出现差异，须先定位再确定可用于回归的基线，不能忽略首个不一致继续宣称无损。需要时记录逐轮 hidden、每步 top-1/top-2 margin、full logits 误差；调试数据保留在 X5。

### 6.3 实验前命令（未来执行，本次未执行 GPU 部分）

```bash
ssh X5
cd /home/ruoxi/repo/sglang
git status --short --branch
git remote -v
git rev-parse HEAD
nvidia-smi
ss -ltnp
```

确认分支、工作树后才 `git pull --ff-only origin loopspec-test`，并与本地候选 SHA 比较。GPU/端口根据当时资源选择，不沿用旧快照；不终止别人进程。

在已选独立环境检查：

```python
import sys, importlib.metadata, sglang, torch
print(sys.executable, sglang.__file__)
for name in ("sglang", "torch", "transformers", "triton", "flashinfer-python"):
    print(name, importlib.metadata.version(name))
print(torch.cuda.is_available(), torch.version.cuda)
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
```

要求实际 `sglang.__file__` 指向 `/home/ruoxi/repo/sglang/python/sglang/`，或已核实绑定同一 SHA 的容器挂载。

### 6.4 启动与调用方案

官方入口已存在，但环境与 GPU 尚未实测：

```bash
cd /home/ruoxi/repo/loopspec
CUDA_VISIBLE_DEVICES="$GPU_ID" uv run --locked --no-dev python -m sglang_recurrent.server \
  --model /home/ruoxi/models/Ouro-2.6B \
  --host 127.0.0.1 --port "$PORT"
```

以上是官方 AR；官方单候选附加 `--loopspec --first 1`，双候选再附加 `--second 2`。不需要下载其他模型。

下列是拟议的原生接口验收形态，当前分支没有 Ouro/LOOPSPEC 实现，不能直接运行成功；新增参数最终需通过 `--help` 与参数验证测试确认：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python -m sglang.launch_server \
  --model-path /home/ruoxi/models/Ouro-2.6B \
  --host 127.0.0.1 --port "$PORT" \
  --trust-remote-code --dtype bfloat16 --attention-backend triton \
  --max-running-requests 1 --page-size 1 \
  --disable-overlap-schedule --disable-radix-cache \
  --chunked-prefill-size -1 --disable-cuda-graph
```

单候选拟附加 `--speculative-algorithm LOOPSPEC --speculative-draft-depth 1`；双候选改为 `--speculative-draft-depth 1 2`。这些 LoopSpec 参数均为待实现设计。graph 阶段撤去 `--disable-cuda-graph` 并验证实际进入 recurrent graph path。

待 `/health` 就绪后，每个模式调用同一 JSON：

```bash
curl --fail-with-body --max-time 120 "http://127.0.0.1:$PORT/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":32,"ignore_eos":false},"stream":false}'
```

用未来验收 harness 同时保存 committed token IDs、stop reason 和原始响应；若 HTTP 输出不含 token IDs，从 worker/测试调用直接捕获，不将响应文本重新 tokenize 作为证据，也不依赖官方不支持的 return_logprob。

### 6.5 本条请求的通过标准

- 启动成功、未静默改变模型/R/backend，健康检查不计入主请求数据。
- AR 对独立 reference 通过；单/双候选与 AR 完整 token IDs 和停止位置一致。
- 不输出未验证 draft；自然 EOS 或 32-token 长度限制准确；输出非空且请求正常结束。
- pipeline trace 显示浅层 draft、full-depth 验证、状态复用确实发生。
- 单请求结束资源正确回收；无 CUDA、IPC、权重缺失错误。
- 若自然请求没有触发某种拒绝分支，结果中标记未覆盖，另用固定合成分布/受控 draft 测试补齐，不能凭此一条请求说 rollback 已全面验证。

### 6.6 完整功能所需的补充验证

- schedule/branch 测试：全接受、首次拒绝、q2 恢复、双拒绝、gate 关闭、depth 边界、slot 复用、共享前缀所有权。
- stop/lifecycle：输出上限 0/1、EOS、stop string 跨 token 边界、streaming 拼接与非流一致、取消/异常/连续请求的资源回收。
- sampling：小词表 CPU 双精度 oracle 对照 proposal/gate/residual 与 GPU kernel；预先固定分布集合、seed 集和统计判据。例如每种分布 100,000 次样本、逐 token 误差阈值 `max(5*sqrt(p*(1-p)/N),5/N)`，并对多重比较记录约定；零概率支持必须不泄漏。另对真实模型固定 prefix logits 检验接受/残差流程。此测试不等价于用同 seed 比逐 token 序列。
- graph：所有计划内 role/shape 与 eager 差分，拒绝/回收时的异步生命周期检查。
- 功能矩阵：Ouro 当前 checkpoint、其他 Ouro/Thinking 和 Raven 分开记录验证覆盖；不能由一个 checkpoint 推断全系列通过。

### 6.7 性能口径

单次 32-token 请求只报告观察值，不作加速结论。记录 input/output token 数、prefill、decode、端到端、TTFT、accept/restart、pipeline cycles、峰值显存。

若后续要给速度结论，在正确性通过后对同一 workload 每模式预热 2 次、测量 10 次，保持 GPU、精度、backend、prompt、停止条件、graph/TinyGEMM 设置可比，尽量交替配对，不能让模型初始化混入 decode。资源不足无法同时容纳服务时顺序重启、分别预热并说明测量设计。

官方 decode 从第一个 completion token 处理完成后开始，计数为 `max(output_tokens-1,0)`；客户端 E2E 单独计时。加速比用相同口径 AR 耗时 / LoopSpec 耗时。若提前 EOS 导致过短，则该请求不适合稳定性能推断；不能擅改 stop 规则后沿用原结论。

## 7. 当前结果与预算

- 完成：版本/入口/接口/模型配置的只读核对，完整迁移与单请求验证计划。
- 未执行：源码修改、安装、commit/push/pull、模型加载、GPU correctness、sampling 统计测试、性能测试。
- 本次唯一新增文件：本计划记录。
- 已用代码候选轮次：0/5。后续若实施，整个任务沿用本记录；第 5 轮未达到标准则停止改代码并报告，不能用阶段切分自动取得新预算。
- 正式 `loopspec` 分支未晋升。没有宣称本机/X5 加速结果。

## 8. 2026-10-26 批准与接口补充

用户批准完整计划，并要求所有公开接口保持 SGLang 原样：使用 `--speculative-algorithm loopspec`，仅新增 `--speculative-draft-depth`，接受一或两个深度整数。不得增加自定义启动模块或请求字段。标准参数的既有含义不变。未支持的组合明确报错，不静默回退 AR。

### 候选 1/5：原生 Ouro 固定深度基线

1. 目的：先证明 native Ouro full-depth AR 的模型数学、权重和 KV 展开正确，为 LoopSpec 固定 target。
2. 步骤：注册配置与模型；按循环展开 attention/KV；严格校验 packed 权重；建立独立 reference 对比与标准 serving smoke harness；同步 X5 验证。
3. 文件范围：`configs/ouro.py`、`models/ouro.py`、`utils/hf_transformers/common.py`、`configs/model_config.py`、`model_executor/model_runner_components/layer_setup.py`、本任务测试/运行脚本及记录。
4. 成功标准：每个 checkpoint 权重和 packed shard 完整加载，48 共享层/192 KV 层；完整 R=4 AR 与独立 reference 的 greedy token IDs/停止位置一致，标准 HTTP 单请求完成。此轮通过不表示整个迁移完成。
5. 风险：HF 版本兼容、RMSNorm 精度、层范围统计、5090 依赖与 CUDA kernel 兼容；不修改模型原文件。
6. 验证：本地 compile/diff/针对性测试；X5 隔离环境核实 source SHA 与 import path；独立 reference 与 native AR 相同 prompt、BF16、32 token；保存完整日志。正确性后才讨论配对耗时。

起始 HEAD：baa9befa381511cd4d56f83980951e5a45fffc6d。当前无代码测试结果。整个迁移仍共用最多五轮，后续阶段不另起预算。

### 第 1 轮结果

候选 f28ab9520aee8042d7c2c6b59f8e747349592f71，已 push loopspec-test 并 X5 pull 核对。compileall/diff-check 退出 0；X5 test_ouro.py 四项通过。native harness 退出 0，11 token IDs `[7042,30,198,198,504,3403,2988,314,330,30,0]`，EOS stop。这是 AR 跑通，尚非独立正确性通过。

环境：容器 loopspec-native-20261026，镜像 v0.5.22-cu130 digest e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae；源码只读挂载 /workspace/sglang，PYTHONPATH=/workspace/sglang/python；Torch 2.14.1+cu130、Transformers 5.19.0、flashinfer-python0.7.1、sglang-kernel0.4.9。镜像 wheel metadata 为0.5.15.dev381+gc05c48b35，不作为源码版本证据；执行源码路径和 Git SHA 已核对。镜像默认 SGLANG_RUST_BUILD_MODE=never 导致第一次 native 失败(137)，改 auto 后构建当前源码 Rust扩展并成功。这是环境修复，无源码改动。

独立参考环境 /opt/ouro-reference：Transformers4.55.0、kernels0.9.0，复用同Torch；依次遇到Torch路径缺失、kernels/hub不兼容，环境修复后仍失败(1)：checkpoint UniversalTransformerCache 对 key_cache 赋值，与其父类只读属性冲突。失败日志及成功 native.json 在 X5 /results/round-1（宿主对应实验目录）。未作性能结论。

### 候选 2/5：独立 reference 的无缓存验证

目的：消除 checkpoint Cache API 版本耦合。步骤：只将 HF reference 改为 use_cache=False，每步完整前缀重算；不改权重/模型/R/停止条件。范围：verify_ouro.py 及记录。标准：reference 与 native 的 token IDs、stop 完全一致。风险：完整前缀重算浮点顺序可能和 cached decode 不同；若分歧需定位，不能弱化标准。验证：compile/diff-check、Git同步，X5 reference→compare；native 标准 HTTP smoke。无缓存 reference 耗时不参与性能对比。

### 第 2 轮结果

候选9a5cfe9aba995283376758d621a8262f489efaf0已同步 X5。HF reference(use_cache=False)退出0，token IDs与第1轮native完全相同，compare退出0；标准 `python -m sglang.launch_server` /health及 /generate 通过，HTTP结果再次与reference逐token/stop一致，compare退出0。原生 AR 的本条请求正确性基线通过。原始结果在round-2/reference.json、http.json、http-server.log。

### 候选 3/5：原生 LoopSpec worker 与流水线

1. 目的：标准 speculative-algorithm=loopspec，唯一新增draft-depth参数；完整深度target不变。
2. 步骤：接入参数/enum/V2worker；持久化跨token recurrence lane；按depth隔离KV，分支引用追踪/回收；实现单候选与门控residual第二候选；每次worker返回一个已验证token，让原scheduler处理stop/stream/cancel；复用原生attention、logprob。
3. 范围：arg_groups/fields/spec.py、field_order.py、speculative_hook.py；speculative/spec_info.py、spec_registry.py、spec_utils.py及新loopspec模块；必要的tokenizer/scheduler生命周期hooks；本任务tests/harness/记录。
4. 标准：相同单请求单/双候选与AR逐token/stop一致；trace证明早期proposal、full-depthverify和accepted-state复用；合成分布验证q2 residual而非原p_d2；资源回收及非法组合显式拒绝。
5. 风险：V2 scheduler水位、KV allocator ownership、未物化深度读取、top-k/p实际proposal、graph输入生命周期。首个集成候选先用eager single-request/TP1/Triton；不把eager通过声称完整graph/TinyGEMM/Raven移植。
6. 验证：本地语法/diff；X5测试包含权重覆盖、强制拒绝/双拒绝、sampling分布；HTTP相同prompt与reference比较；后续轮次修复失败并补齐graph/生命周期验收，整个任务仍最多5轮。

### 第 3 轮结果

候选e142f4928f080d21e2179d8afe8eb38618c869ae。compileall/diff-check退出0；X5 test_loopspec.py四项通过(11.102s)，覆盖首候选/第二候选/全部拒绝、共享前缀保留、深度校验及三组各100000次sampling统计。single/double serve_smoke均退出0，各两次标准HTTP请求，与round-2独立HF结果全token/stop一致(compare均退出0)。single trace cycles22,reused7,restarts3,accepted[7,0],rejected3；double cycles18,reused9,restarts1,accepted[7,2],rejected1。无运行期memory-leak告警。server退出日志的SIGTERM/SIGQUIT来自harness终止自己进程组，不是请求失败。

限制：当前核心按lane/depth依次调用共享权重，保留了计算状态但没有官方mixed-depth core合批；eager正确不代表速度提升。通用spec_accept_rate把存储预留宽度当proposal数，不适用，现不输出该错误指标；实际计数记日志。graph/Raven/TinyGEMM尚未完成。

### 候选 4/5：recurrent CUDA graph 与生命周期

目的：标准CUDA graph开关控制原生recurrent graph，补齐服务生命周期证据。步骤：模型runner对LoopSpec使用eager prefill；worker为每个depth捕获独立静态core/readout graph并clone持久化输出，重放前更新position/seq-len/KV-slot；扩充标准HTTP harness。范围：loopspec/config.py、worker.py、新graph.py、model_runner_components/cuda_graph_setup.py、测试/记录。成功：graph单/双候选逐token与eager/AR一致，实际replay计数>0；短输出、stream、stop、cancel后可继续请求及flush无泄漏；sampling真实请求可完成。风险：静态metadata和输出别名、graph输入生命周期、graph/eager浮点差异。验证：本地语法/diff，Git同步，X5单/双graph→AR compare，生命周期脚本，必要时eager对照定位；不作尚无配对数据的性能结论。

第4轮验证前范围补充：将共享层的QKV/o-proj/MLP按mixed-depth lanes合批，attention按depth分组执行；graph捕获整个合批core而非逐lane graph。新增范围 models/ouro.py、loopspec/pipeline.py。容量由原schedule的birth recurrence推导，padding不进入输出；验证增加batched eager与batched graph同AR对比。该修改仍在第4轮候选首次验证前，未消耗新轮次。

### 第 4 轮结果

候选 fb7e35ddc1c24a817f287da091da68c61234db4d 已同步 X5。模型四项、LoopSpec 四项测试通过；batched double eager、single graph、double graph 的 serve_smoke --lifecycle 均退出0。前两次请求全部与独立 HF R=4 token/stop一致。生命周期覆盖0/1输出上限、stop string、stream、取消后继续请求、OpenAI completions、真实模型sampling、flush_cache；无memory leak警告。single graph实际22次replay，double18次。depth=2 graph补充验证与官方0.5.14独立环境AR对照进行中。raw在round-4。未做稳定配对benchmark。

### 候选 5/5：显式 TinyGEMM backend 与最终回归

1. 目的：保留官方支持的BF16小batch线性优化，公开接口仍只有新增draft-depth；不得进程级替换linear方法。
2. 步骤：在原有bf16-gemm-backend中添加tinygemm选项；初始化时检查硬件及内核；按shape/dtype/device选择，其他shape用Torch；core readout也走同一backend；扩展harness选择既有backend并做数值/graph验证；补严格不支持配置检查。
3. 范围：arg_groups/fields/exec_.py、layers/quantization/unquant.py、新layers/quantization/tinygemm.py、loopspec/graph.py及config.py；本任务harness、GPU kernel验证脚本与记录。
4. 标准：SM120实测TinyGEMM调用>0；bias/shape边界与graph replay数值检查；Ouro AR及单/双候选与HF固定R4逐token/stop一致；标准生命周期继续通过。默认auto行为不变。
5. 风险：BF16累加顺序导致argmax变化、首次JIT/graphcapture兼容；选定但不可用时明确失败。不将TinyGEMM支持等同已获加速。
6. 验证：本地compile/ruff/diff审查；仅loopspec-test提交并X5核对SHA/importpath；GPU kernel float32 oracle误差，原生AR与单/双graph HTTP差分，统计采样与模型单测回归。第5轮若失败不再改源码。Raven独立pre/post/injection、拒绝采样graph、完整模型logit诊断仍须在最终结果明确标记，不将Ouro smoke当全部验收。


## 9. 当前原生接口与实现边界（2026-10-26）

公开入口保持SGLang标准入口；请求仍是原有GenerateReqInput / sampling_params，未增加请求字段。选择 `--speculative-algorithm loopspec`；唯一新参数名是 `--speculative-draft-depth`，值为一或两个整数，例如1、2、1 2。既有 `--bf16-gemm-backend` 新增tinygemm枚举值；auto/torch不变。

```bash
python -m sglang.launch_server \
  --model-path /home/ruoxi/models/Ouro-2.6B \
  --host 127.0.0.1 --port 30126 \
  --dtype bfloat16 --attention-backend triton \
  --context-length 2048 --mem-fraction-static 0.7 \
  --max-running-requests 1 --page-size 1 \
  --disable-overlap-schedule --disable-radix-cache \
  --chunked-prefill-size -1 \
  --speculative-algorithm loopspec --speculative-draft-depth 1 2
```

单候选改为draft-depth 1或2；需要eager时使用既有disable-cuda-graph；可选既有bf16-gemm-backend tinygemm。AR移除algorithm和draft-depth两个选项。该例没有使用early exit，也未修改checkpoint的R=4。原生Python Engine可用同名snake_case构造参数。标准HTTP请求：

```bash
curl --fail-with-body http://127.0.0.1:30126/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":32},"return_logprob":true}'
```

| 范围 | 当前状态 |
| --- | --- |
| Ouro-2.6B，BF16，完整4轮AR | HF独立参考、官方AR、nativeAR单请求token/stop对齐 |
| LoopSpec单候选d1=1/2、双候选1/2 | native V2 worker、mixed-depth batching、branch state reuse、残差验证已实现；eager/graph已验证（d1=2验证graph） |
| 原生HTTP /generate、/v1/completions | 相同参数格式；流式、max0/1、EOS、stop、取消/后续请求、flush已验证 |
| Sampling | 三组合成分布各100000次通过；真实模型请求通过；真实prefix分布补充统计见第5轮结果 |
| 图捕获 | mixed-depth recurrent core和readout；prefill使用eager；没有独立拒绝采样graph |
| 并行/缓存功能 | 当前单活跃请求、TP/PP/DP/CP=1、Triton、page1；禁用overlap/radix/chunkedprefill；不支持grammar/LoRA/PD等组合，显式校验 |
| 监控 | cycles、reuse、restart、first/second accept、reject、graph replay写日志；原生通用spec_accept_rate的分母不适用，因此不输出该字段；尚无完整Prometheus算法指标 |
| Raven / 其他Ouro checkpoint | Raven未迁入，其他Ouro/Thinking未加载验收；不能标记全模型完成 |
| 官方完整内存布局 | 当前共享native AR的192逻辑KV层，branch私有request映射；没有迁成官方48物理KV层加depth timeline的紧凑pool，容量效率有差异 |
| 性能 | 仅功能请求观察值；没有warmup后配对重复benchmark，不给加速结论 |

原计划仍未完成：Raven的config/builder/tokenizer/injected状态及pre/post attention；官方分角色pre/post与rejection graphs；紧凑timeline pool；完整模型逐深度hidden/logits误差诊断、数值阈值和更广模型/请求矩阵；完整指标与配对性能。不能把上表Ouro通过项替代完整迁移验收。

### 固定环境与只读参考

源码测试SHA：fff1dc3c924f64ce31a61417f6ce8387de6398a6；X5/source路径核对 `/opt/sglang/bin/python`、`/workspace/sglang/python/sglang/__init__.py`。GPU0 UUID GPU-449e057b-787e-b46a-7a4c-727bfe1bd515，RTX5090，driver580.178.04。模型config SHA256 `23b5c6942a9f0ca8619fb15acfaca93e02c169cdc4e03dbceb4d8a2df145697a`，tokenizer.json SHA256 `fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d`。权重未修改。

官方参考使用独立容器loopspec-official-20261026，uv sync --locked --no-dev --python3.11（源目录只读），官方依赖sglang0.5.14/torch2.11.0+cu130。官方AR的请求输出与HF/native精确一致（compare退出0），结果official-ar.json；完成后停止该容器。官方仓库SHA仍41aa710ed7c9ce908cffa03597be5d1df19d7b56、clean。未把官方定制scheduler用作本项目serving入口。

网络环境临时出现X5域名解析到198.18.0.204并关闭SSH；通过公共DNS核对18.25.6.221后使用 `ssh -o Hostname=18.25.6.221 -o HostKeyAlias=x5.mit.edu X5`，保留host-key校验，未改系统SSH配置。一次host侧重定向help日志因容器创建目录所有者不同而Permission denied，改在容器内写/results后成功；属于环境/记录重试，无源码变化。


### 第 5 轮结果与最终验收状态

测试源码 fff1dc3c924f64ce31a61417f6ce8387de6398a6（文档收尾提交不改源代码），本地compileall、ruff F、diff-check退出0。X5两端源码SHA与import路径一致，工作树干净。

| 验证 | 命令/结果 | 退出码 |
| --- | --- | --- |
| 严格加载/循环KV结构 | `python test_ouro.py`，4项通过 | 0 |
| 流水线/分支/采样 | `python test_loopspec.py`，4项通过；三组合成分布各100000次，11.206s | 0 |
| TinyGEMM数值与graph | `python verify_tinygemm.py`，三组支持shape实际kernel执行及17-row回退均通过；graph更改输入后与eager完全一致 | 0 |
| TinyGEMM native AR | `python serve_smoke.py --bf16-gemm-backend tinygemm --lifecycle --output /results/round-5/ar-tiny` | 0 |
| TinyGEMM单候选graph | 同上添加 `--depths 1 --cuda-graph`，输出目录single-tiny | 0 |
| TinyGEMM双候选graph | 同上添加 `--depths 1 2 --cuda-graph`，输出目录double-tiny | 0 |
| 默认后端回归 | `serve_smoke.py --depths 1 2 --cuda-graph --lifecycle --output /results/round-5/double-auto` | 0 |
| HF/官方/native token对照 | verify_ouro.py compare，官方与第4轮4份通过；第5轮HF+TinyGEMM三模式各两次共7份通过 | 0 |
| CLI兼容 | `python -m sglang.launch_server --help`，包含LOOPSPEC、draft-depth及既有backend的新枚举tinygemm | 0 |

上述脚本均位于 `/workspace/sglang/research/loopspec/experiments/2026-10-25-native-migration-plan/`，在loopspec-native-20261026容器内执行；serving使用 `SGLANG_RUST_BUILD_MODE=auto`。脚本路径的宿主源是经Git同步的只读源码。每种服务两次主要请求+生命周期，未启动通用大型benchmark。

所有主要请求输出11个token `[7042,30,198,198,504,3403,2988,314,330,30,0]`，EOS=0，文本 ` Paris.\n\nThe final answer is A.`；input5token。单候选主请求cycles22/reused7/restarts3/accepted[7,0]/rejected3/graphreplays22；双候选cycles18/reused9/restarts1/accepted[7,2]/rejected1/graphreplays18。不要取server warmup请求的trace冒充主请求。

TinyGEMM FP32 oracle对比：三组支持shape最大绝对误差分别0.00780272、0.01560545、0.01512432，RMSE分别0.00167549、0.00234346、0.00167909；固定容差rtol0.02/atol0.04。native三模式日志均记录SM120 TinyGEMM enabled，未发现Traceback、CUDA error或memory leak。该kernel数值检查不是全模型logits误差验收。

真实前缀sampling补充：在独立HF参考环境对相同prompt做use_cache=False完整R4，保存各depth logits到X5 round-5/hf-prefix-logits.pt；通过native sampling函数生成q1/p2/p4，压缩零支持坐标后CPU双精度100000次抽样，seed11907、逐bin阈值max(5sigma,5/N)、零概率支持无泄漏。temperature0.8/topk40/topp0.9时p4退化为单点，该项只覆盖路径、不可作为强统计证据。补充temperature1/topk40/topp1非退化分布：union67、target支持41（并列cutoff保留）、gate94126次、接受首候选18084/第二候选69756/回退12160，最大频率绝对误差0.00147601，全部阈值通过。原始summary分别real-prefix-sampling.json和real-prefix-sampling-nondegenerate.json。首前缀四个depth top1为260/7042/7042/7042，top1-top2 logit margin为0.4375/2.375/3.1875/3.125。

客户端E2E仅作原始观察值：tiny AR两次0.6361/0.6078s（eager），single graph0.2360/0.2240s，double graph0.2125/0.2003s；每次11输出token。AR与candidate graph条件不同、样本少且没有配对warmup重复测量，所以不能据此计算/宣称LoopSpec加速比。prefill/decode/峰值显存的完整配对指标仍未收集。

本轮支持的Ouro功能项通过，但完整计划尚未完成。预算已用5/5，根据AGENTS与loopspec-code-change停止进一步源码修改；不得以新阶段名继续第6轮。剩余位置：Raven模型/配置/tokenizer与显式injection协议；loopspec graph/sampling的独立拒绝图；mem_cache紧凑timeline布局与容量；metrics接入及全模型差分harness。正式loopspec仍停留基线baa9befa，不作merge/cherry-pick/push晋升。测试容器在验证后停止，保留隔离环境供后续复现。


## 10. 继续授权：2026-10-26

用户明确要求“继续修改直到全部完成”。保留累计5轮历史，不重置计数；工作区loopspec-code-change技能正文已更新为总10轮，因此本次继续第6/10轮，最多追加5轮。正式分支晋升仍未授权。范围仍为原计划，不把Raven模型未实测误写成模型验收。

### 候选6/10：Raven与显式pre/core/post状态协议

1. 目的：补齐官方Raven架构、tokenizer与injection语义，保持Ouro已验收路径和标准SGLang公开参数。
2. 步骤：建立原生Raven config/model/严格权重映射；pre初始化独立hidden和injected，core持久化injection，post按depth隔离KV并预填各endpoint；通用worker保存显式injected；图分为pre/core/post角色，尊重decode graph配置；补checkpoint tokenizer修正。
3. 范围：configs/raven.py、models/raven.py、models/recurrent/、Ouro协议方法、loopspec/worker.py与graph.py/config.py/pipeline.py、HF config/tokenizer与模型层数、research验收脚本。
4. 成功标准：Ouro原有请求/生命周期保持通过；Raven pre-norm和OLMo post-norm小型fixture检查层索引、完整权重覆盖、injection状态、AR/单/双eager/graph一致。Raven真实checkpoint若未存在，按原计划只声明代码迁入和fixture通过，不声明真实模型验收。
5. 风险：Raven随机hidden不能用相同seed保证与spec执行顺序一致；fixture显式使用零初始化以隔离数学与缓存验证，不改真实模型配置。post浅深前缀必须分别物化；不能复用R=4的指数存储上界到R=32。
6. 验证：本地compile/ruff/diff；Git同步X5后检查GPU/端口与import SHA；模型fixture GPU数值/缓存与标准server请求；Ouro主请求回归。后续候选补紧凑KV布局、拒绝采样graph、指标与数值/配对性能harness，仍累计至10轮。

第6轮本地compileall、ruff F、diff-check均退出0。X5域名解析198.18.0.204的SSH与公共DNS核实18.25.6.221的SSH均连续超时；只读DNS仍返回相同公网地址。已向用户询问网络/VPN状态，GPU验证明确未执行。源码候选保留并通过loopspec-test提交；下一候选可继续独立实现，但不把静态检查当GPU通过。

### 候选7/10：紧凑物理KV与角色/深度timeline

1. 目的：将LoopSpec的共享core KV存为物理层数个buffer，按(role,depth,logical token)映射物理槽；拒绝分支不预占未执行深度的槽，原生scheduler仍使用逻辑token ID。
2. 步骤：原生pool factory选择TimelineTokenToKVPool；pre/core/post分别按需要的物理容量分配；逻辑allocator回收时同步释放所有timeline条目；target prefill与worker eager/graph显式绑定各depth翻译器；保持baseline原布局作独立对照。
3. 范围：mem_cache/timeline_pool.py与kv_cache_configurator.py；loopspec worker/graph、Ouro/Raven prefill协议；allocator/model/tokenizer支持的参数约束；专用pool所有权测试。
4. 标准：Ouro pool物理48层，core槽容量R*N，native AR仍192层；Raven物理P+C+O层、pre容量N、core/post各R*N；相同KV字节预算，scheduler保留保守N full-depth token容量；浅层只分配已物化timeline；释放后物理槽无泄漏、无重复释放；Ouro/fixture输出与此前AR一致。
5. 风险：翻译器与graph静态buffer别名、slot0 padding、deferred free、prefix共享、同一role/depth重绑定。不开启不兼容HiCache/PD/量化KV；不将物理容量R*N冒称上下文容量扩大R倍。
6. 验证：本地compile/ruff/diff；pool在CPU/GPU上按角色深度读写隔离、延迟释放和重用；X5恢复后同SHA运行第6/7轮模型fixture与Ouro生命周期、graph/eager差分；记录physical/live槽数。

第7轮compileall、ruff F、diff-check均退出0。第6轮提交23a78bf67已push；X5 SSH重试仍超时，尚未pull或执行新增GPU测试。timeline物理容量与full-depth逻辑容量分开核算，保持每role独立free list；原生AR未切换pool。

### 候选8/10：拒绝图、指标与完整验收工具

1. 目的：补齐独立rejection CUDA graph、原生Prometheus观测和可复现数值/配对性能证据工具，公开参数名保持不变。
2. 步骤：增加纯张量残差验证与静态graph replay，eager作为独立oracle；worker记录proposal/accept/reject/reuse和各role replay；使用既有enable-metrics；harness收集warmup后重复请求、标准时序、token/stop及metrics；增加逐depth hidden/logits诊断。
3. 范围：loopspec sampling/worker/metrics模块及本任务research脚本，不改官方参考仓库。
4. 标准：固定uniforms分支oracle一致、100000次目标分布统计无零支持泄漏；graph改输入后不读旧buffer；指标分母为真实verify/proposal；所有模型回归通过后才计算相同条件配对性能。
5. 风险：零概率/残差下溢、随机数使用顺序变化、graph输出别名、指标进程注册、短请求计时噪声；随机模式不要求相同seed序列等同AR。全模型BF16误差记录max/RMSE/argmax与margin，不能仅以文本相同代替数值诊断。
6. 验证：本地隔离CPU环境执行采样与指标测试，compile/ruff/diff；X5恢复后同SHA跑CPU/GPU测试、Raven合成fixture、Ouro AR/单/双eager/graph、生命周期、独立HF数值对比；配对至少warmup2+测量10，每种模式同prompt/precision/graph/backend，保留raw时序。

第8轮本地隔离环境/tmp/loopspec-cpu-env（Python3.13.3、Torch2.14.1、transformers5.20.2）仅执行CPU小张量，不加载模型。test_rejection_graph.py退出0：3项CPU通过（固定uniform分支oracle；三组各100000次目标分布与零支持；真实verify/proposal指标分母），1项CUDA输入变更replay明确跳过。compileall、ruff F、git diff --check退出0。新增diagnose_ouro.py用于独立HF/native AR/compact-prefill四前缀×四depth hidden/logits误差与argmax；serve_smoke默认warmup2/测量10并采集原生metrics；compare_performance严格要求graph/backend等配置相同。GPU数值、服务和性能脚本尚未执行。TTFT明确包含调度，不能称纯prefill kernel时间。

第8轮后补充只读/CPU验证：补齐隔离环境的CPU导入依赖并固定transformers5.19.0/tokenizers0.23.2后，test_loopspec.py 4项通过(10.042s)，test_raven.py 4项通过(0.009s)，test_timeline.py 2项通过(0.005s)，均退出0。此前这些脚本因orjson/IPython/torchvision/dill缺失退出1，属于本地环境导入阻塞，未修改源码绕过测试。RavenConfig独立roundtrip返回hidden256/head128/logical9/R4。CPU tests不等于Raven真实权重或CUDA验证。

### 候选9/10：v0.5.22 residual构造与设备计时适配

1. 目的：修复审查发现的Raven pre-norm构造问题，补齐既有GPU阶段计时覆盖。
2. 步骤：Raven使用LlamaDecoderLayer时传stage_boundaries=False，由本模型显式执行residual；增加真实native构造回归；LoopSpec decode使用runner.device_timer的既有上下文，prefill保留原runner计时；性能harness启用既有环境开关、记录warmup前后stage指标和模型时间，继续区分TTFT与GPU区间。
3. 范围：models/raven.py、loopspec/worker.py、本任务test_raven.py及serve_smoke/compare_performance。
4. 标准：不依赖make_layers/layer_stack即可构造pre-norm Raven，post-norm行为保持；CPU既有tests通过；X5上AR和LoopSpec均暴露prefill/decode计时，计时开启条件相同、token/stop一致。
5. 风险：v0.5.22 layer-boundary语义与旧参考不同；DeviceTimer不可嵌套，故仅wrap worker自有decode、不wrap调用原runner的prefill。CUDA事件测量包含stream等待/host提交空隙，不称kernel独占时间。
6. 验证：本地CPU构造/权重/分配/采样及compile/ruff；X5恢复后完成累计6–9轮GPU矩阵与配对性能。累计预算9/10，保留最后一轮用于GPU实际失败修复；未执行项目不得标通过。


### 第9轮本地结果与继续条件

Raven真实原生模块构造CPU测试通过，pre-norm/post-norm各构造小模型、logical attention IDs均0..8、tied head共享同一Parameter，pre-norm不依赖layer_stack。test_raven.py 5项/0.030s，test_ouro.py 4项/0.002s，test_rejection_graph.py 3项CPU/0.038s+1项CUDA跳过；compileall、ruff F、git diff --check退出0。第8轮已通过且未变化的pipeline/sampling4项、timeline2项不重复无意义运行。

实际TimelineTokenToKVPool CPU实例检查（不是mock）：N=8/R4，Ouro C3有3物理/12逻辑层，三个buffer形状(33,2,64)，K/V各25344字节；Raven P1/C1/O1有3物理/9逻辑层，buffer形状(9,2,64)/(33,2,64)/(33,2,64)，K/V各19200字节。分别在4个core-depth写入1/2/3/4后读取无交叉污染，release后live_slots=0，退出0。该检查未调用CUDA scatter/attention，不替代GPU验证。

本轮唯一新增行为修复是Raven构造的stage_boundaries=False；性能工具启用既有SGLANG_ENABLE_METRICS_DEVICE_TIMER，LoopSpec decode接入同一runner.device_timer。保存metrics-before/after的sglang:forward_execution_seconds_total分阶段差值；先后0.2s idle让scheduler排空完成事件，GPU执行前须确认指标完整。CUDA stream区间可能包含host提交间隙；TTFT、service decode吞吐和client E2E分别报告，不混称纯kernel耗时。

源码提交链：第6轮23a78bf67、第7轮b9b0963f02092eb8629006772773e7141b710eab、第8轮2e06298ab9c41a50c576956f57fb7a4211dbf641；第9轮见本记录所在修复提交。均仅loopspec-test。正式loopspec仍为baa9befa381511cd4d56f83980951e5a45fffc6d。

| 验收项 | 当前代码/CPU状态 | 最新GPU证据 |
| --- | --- | --- |
| Ouro native AR、单/双LoopSpec、原生接口/生命周期、TinyGEMM | 保留并扩展 | 第5轮fff1dc3通过；不能代替新timeline/rejection图回归 |
| Raven config/权重映射/tokenizer/injection/pre-core-post | 已迁入，5项CPU通过 | 未执行合成fixture；真实Raven checkpoint未验收 |
| 紧凑KV timeline/所有权/延迟释放 | CPU所有权与实际buffer隔离通过 | 未执行CUDA attention/graph |
| 独立rejection graph | 张量eager oracle/分布通过 | graph输入更新测试跳过 |
| 既有metrics端点与算法指标 | 指标分母CPU通过，GPU区间接入 | 未执行server scrape验收 |
| 全模型逐depth hidden/logits | diagnose_ouro.py已准备，阈值预声明 | 未执行 |
| 同配置warmup2+测量10配对性能 | harness和严格配置比较已准备 | 未执行，不宣称加速 |

当前外部阻塞：X5域名/公共DNS核实IP18.25.6.221的SSH多次超时，未修改host-key校验或SSH配置。已询问用户VPN/连通性，尚无恢复信息。未pull到X5、未启动GPU任务或修改远端源码。保留累计9/10，GPU验收及可能的第10轮修复仍待完成，不能把CPU通过标作全部完成。

恢复后先执行（只读检查→确认无用户改动→ff-only pull）：

```bash
ssh -o Hostname=18.25.6.221 -o HostKeyAlias=x5.mit.edu X5
cd /home/ruoxi/repo/sglang
git status --short --branch
git remote -v
git rev-parse HEAD
nvidia-smi
ss -ltnp
# 确认loopspec-test工作树干净、GPU/30126端口空闲后：
git pull --ff-only origin loopspec-test
git rev-parse HEAD
# 启动并检查之前保留的loopspec-native-20261026容器及RO源码挂载，
# 核实/opt/sglang/bin/python与/workspace/sglang/python/sglang/__init__.py。
```

容器内待执行验收命令（输出在/results/round-9；执行前记录SHA，失败保留日志）：

```bash
cd /workspace/sglang/research/loopspec/experiments/2026-10-25-native-migration-plan
python test_ouro.py
python test_loopspec.py
python test_raven.py
python test_timeline.py
python test_rejection_graph.py
python verify_tinygemm.py
python raven_fixture.py --output /results/round-9/fixtures
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-9/ar-eager --lifecycle
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-9/ar-graph --cuda-graph --lifecycle
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-9/single-graph --cuda-graph --depths 1 --lifecycle
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-9/double-graph --cuda-graph --depths 1 2 --lifecycle
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-9/double-eager --depths 1 2 --lifecycle
# 补d1=2 graph；TinyGEMM下AR/单/双相同graph设置；两种Raven fixture各AR/单/双eager/graph。
# Raven fixture不使用硬编码Paris stop的--lifecycle；所有结果compare token/stop。
env -u PYTHONPATH /opt/ouro-reference/bin/python diagnose_ouro.py reference --output /results/round-9/hf-depths.pt
python diagnose_ouro.py native --reference /results/round-9/hf-depths.pt --output /results/round-9/native-depths.pt
python diagnose_ouro.py native --compact --reference /results/round-9/hf-depths.pt --output /results/round-9/compact-depths.pt
python diagnose_ouro.py compare --reference /results/round-9/hf-depths.pt --candidate /results/round-9/native-depths.pt --output /results/round-9/numeric-ar.json
python diagnose_ouro.py compare --reference /results/round-9/hf-depths.pt --candidate /results/round-9/compact-depths.pt --output /results/round-9/numeric-compact.json
python compare_performance.py /results/round-9/ar-graph/benchmark.json /results/round-9/single-graph/benchmark.json /results/round-9/double-graph/benchmark.json --output /results/round-9/performance.json
```

需保持原成功标准：greedy与独立HF/官方AR逐token和停止位置一致；sampling按分布验证；Raven非零随机初始hidden不要求同seed逐序列相同。最终GPU回归未通过前，不晋升正式分支、不输出性能胜出结论。


## 2026-10-27 恢复验证尝试与架构归属确认

用户告知X5可以重新连接并要求继续修改验证，同时询问调度实现来源。当前源码仍4d2f8d3c2b1e919f12eb068352f84a4bdee55934，loopspec-test clean且与origin一致，累计9/10；本次没有源码修改，不增加轮次。

实际连接结果：`ssh -o ConnectTimeout=15 X5` 返回Connection closed by 198.18.0.204 port22、退出255；`ssh -o Hostname=18.25.6.221 -o HostKeyAlias=x5.mit.edu -o ConnectTimeout=15 X5` 返回Operation timed out、退出255。公共DNS查询仍返回18.25.6.221。ssh -G确认ruoxi/x5.mit.edu/22，无ProxyJump/ProxyCommand；公网IP路由为en0默认网关。Tailscale正在运行，但没有hostname/DNSName含x5的peer；系统HTTP/SOCKS代理未开启。未改网络/host-key设置，未连接成功、未pull或运行GPU任务。已向用户询问刚才成功使用的SSH命令与VPN/跳板路径，等待补充连通方式。

### 架构归属（根据当前源码调用链）

- SGLang原生层：launch_server → tokenizer_manager → Scheduler → 原生speculative V2 worker接口 → BatchResultProcessor/detokenizer。请求接收、队列、逐次调度、stop/stream/cancel和结果处理仍归原生SGLang。
- 本次新增层：SpeculativeAlgorithm.LOOPSPEC在spec_info.py::create_worker返回LoopSpecWorker(BaseSpecWorker)。worker的prefill调用原target_worker；decode调用Pipeline.advance_until_commit，每次交回一个经过完整target验证的token及GenerationBatchResult。scheduler因而可以在输出间处理原生生命周期。
- LoopSpec算法层：persistent lane、跨token/recurrence流水线、浅depth proposal、门控第二residual proposal、完整depth验证、接受后复用/拒绝后剪枝是参考官方算法为原生worker接口重写/迁移的代码。紧凑timeline和pre/core/post/rejection图是相应原生实现，不是上游已有的通用loop-transformer调度器。
- 计算层：复用SGLang的attention、linear/norm、ModelRunner、ForwardBatch、token allocator协议；新增Ouro/Raven模型与timeline pool。上游IQuestLoopCoder等循环支持是在model.forward中遍历loop并区分逻辑KV层，不能等同已有LoopSpec跨token流水线。
- 官方对照：只读repo 41aa710ed7c9ce908cffa03597be5d1df19d7b56的sglang_recurrent/server/loopspec_scheduler.py::_run_scheduler_loop自己recv_pyobj，顺序调用_run_request运行整条请求；注释明确活跃请求完成前不读取其他控制消息。移植版不调用或import该scheduler，不依赖sglang_recurrent运行包。

因此准确表述为“原生SGLang服务/请求调度框架 + 移植的LoopSpec算法worker”，不能表述为直接使用上游现成LoopSpec调度，也不能表述为仍由官方自写服务scheduler接管。当前单活跃请求、TP/PP/DP/CP=1等支持限制仍有效，原生接口不意味着所有SGLang调度/缓存组合已经支持。

### X5-BUS 恢复与第9轮GPU结果（2026-10-27）

用户提供已有别名X5-BUS，实际连接成功；使用既有ProxyCommand/host-key配置，未修改SSH或网络设置。远端loopspec-test clean，经ff-only pull同步至3058b7767f1779857f3b11b7e333ec2e512b4fa1（源码4d2f8d3c2b1e919f12eb068352f84a4bdee55934）。实验前GPU0/1无计算任务，端口30126/30127空闲。启动本任务已有loopspec-native-20261026容器，GPU0、RO源码/workspace/sglang、RW /results；确认/opt/sglang/bin/python及/workspace/sglang/python/sglang/__init__.py。Torch2.14.1+cu130、Transformers5.19.0、Triton3.8.0、Flashinfer0.7.1。模型config/tokenizer哈希与第5轮相同，仍Ouro-2.6B BF16完整R4。

原始日志在X5 `/home/ruoxi/experiments/loopspec/2026-10-25-native-migration-plan/round-9/`。以下均通过既有脚本执行，没有远端源码修改：

- test_ouro.py 4项、test_loopspec.py 4项、test_raven.py 5项、test_timeline.py 2项、test_rejection_graph.py 4项全部退出0；最后一项本次实际执行CUDA输入更新replay，未跳过。采样仍包含三组各100000次分布检验。
- verify_tinygemm.py退出0，实际kernel三组shape与17-row回退、graph变更输入验证通过，误差与第5轮一致。
- serve_smoke.py的Ouro AR eager/graph、single eager/graph、double eager/graph、depth2 graph共7组退出0，各warmup2+测量10并执行lifecycle。所有70个主要请求与独立HF round-2/reference.json的11个token和stop精确一致。
- lifecycle覆盖max0/1、stop字符串、stream、abort后下一请求、OpenAI completions、sampling与flush。新compact pool日志为48物理层/192逻辑层，full-depth容量11190、core槽44760；同字节预算不宣称上下文容量扩大4倍。
- 单候选主请求cycles22/reused7/restarts3/accepted[7,0]/rejected3，双候选cycles18/reused9/restarts1/accepted[7,2]/rejected1，depth2 cycles12/reused9/restarts1/accepted[9,0]/rejected1。pre/core/post/rejection的Prometheus实际replay计数均存在。
- 同auto backend/graph的compare_performance.py退出0：E2E中位数AR237.262ms、single233.876ms、double208.418ms，对AR比值1.0145/1.1384。每请求GPU stream区间extend/decode分别49.734/180.803ms、51.079/175.090ms、51.482/149.872ms。单短请求、顺序测量块，不能外推吞吐或一般工作负载。

诊断工具失败保留：HF reference退出0；native/compact均在最终深度重新投影与服务logits的零容差断言处退出1，25/49152坐标不同，max0.03125。工具把4个depth拼为M=4的GEMM，而服务实际M=1；不同GEMM shape的BF16舍入不能要求逐位相同。下一候选将HF/native均逐depth做M=1投影，保留与服务实际logits的零容差断言，且不改变预声明hidden/logits阈值与all-depth argmax标准。另发现LoopSpec指标标签取raw ServerArgs得到None，以及通用spec日志错误将slot预留宽度作为候选数；均需修正。此处失败不是Ouro服务token失败，也不能标记全模型数值通过。

### 候选10/10：诊断shape对齐与监控语义修正

1. 目的：完成第9轮GPU发现的诊断与监控适配问题，不改target深度或放宽正确性标准。
2. 步骤：HF/native逐depth使用M=1 LM-head投影，对native实际服务logits继续rtol=atol=0；worker用get_serving解析后的模型名；通用spec块长/接受率对LoopSpec标NaN并停止错误日志和计数，真实统计仍由loopspec_events提供；harness核对标签和测量区间verify/accept/proposal事件。
3. 范围：research/.../diagnose_ouro.py与serve_smoke.py；speculative/loopspec/worker.py；managers/scheduler_components/metrics_reporter.py。
4. 标准：四prefix×四depth的HF/native与compact hidden/logits在原阈值内且argmax一致；native投影零容差断言通过；Ouro请求与HF token/stop一致；Raven两fixture各AR/单/双eager/graph一致；指标分母与实际验证数相符。
5. 风险：改变诊断GEMM shape不保证消除真实模型数值差异，若阈值仍失败必须保留失败；通用spec指标NaN表示不适用，不能解读成0接受率。Raven真实checkpoint仍未验证。
6. 验证：本地compile/ruff/diff，确认仅任务文件后commit/push loopspec-test，X5 clean/ff-only pull/同SHA/import核对；GPU数值、Ouro关键服务回归和TinyGEMM AR/单/双graph，各warmup2+测量10。外部nvidia-smi按100ms采样整段服务显存，报告采样最大值而非精确allocator峰值；auto/TinyGEMM分别配对，不混backend。第9轮未改变的Raven/所有权/采样通过项保留证据，修改涉及的服务指标再次验证。累计10/10，若失败停止继续源码修改。

第9轮Raven补充：raven_fixture.py生成pre/post两份合成checkpoint（零初始hidden），各AR/single/double×eager/graph共12个serve_smoke进程均退出0；每组10个主要请求均与同layout AR eager逐token/stop一致，共120次、每次32token。pre输出全为100；post包含103/104/100/106等token。仅为架构/缓存/graph fixture验收，不代表真实Raven权重质量。

第10轮发布前检查：compileall、/tmp/loopspec-formatter/bin/ruff check --select F、git diff --check退出0；误用不存在的/tmp/loopspec-cpu-env/bin/ruff曾返回127，改用既有formatter工具后通过，无源码绕过。改动仅4个Python文件和任务记录，不改变模型计算、pipeline或KV路径。

### 第10轮结果与追加授权

源码1557ec5106e7754a13fb32bc1e6e5f533a4037de已push并在X5 clean/ff-only pull，import路径一致。数值reference/native/compact生成均退出0，M=1与服务logits的零容差断言通过；compare-native/compact均退出1。两种native布局的全部hidden/logits逐位相同，排除了本组prefill的timeline布局差异。HF对照16个depth点argmax全部相同，logits最大误差0.197265625/RMSE最大0.04274631，符合原0.5/0.1阈值；hidden在(prefix,depth)=(0,3),(2,3),(2,4),(3,3),(3,4)超阈值，最大误差0.25、最大RMSE0.024934724，未通过原0.125/0.02标准。不能将token相同替代该失败。

auto/tinygemm分别AR/single/double graph共6个serve_smoke全部退出0，含lifecycle、新指标标签/NaN语义与真实计数检查；两组compare_performance退出0。auto测量区间10请求各100个verified token；single首候选接受70/残差30，double首接受70/第二接受20/残差10。原始证据保存在round-10，失败日志不删除。

用户随后明确批准“追加5轮，保持原验收阈值”，累计预算扩展至15，历史不重置。正式分支仍未授权晋升。

### 候选11/15：逐层数值定位

1. 目的：定位HF eager与native共同计算路径的最早数值差异，保留既定阈值。
2. 步骤：对相同固定prefix保存4个循环的层输入、四个norm、Q/K/V、attention输出、MLP gate/up/product及层输出；比较两端trace；使用相同输入检查norm、RoPE和SiLU乘法的舍入差异。
3. 范围：diagnose_ouro.py及新ouro_trace.py测试辅助，任务记录；不改变生产模型、pool或worker。
4. 标准：可追溯到depth/layer/operator的最早分歧，区分输入误差传播与同输入算子误差；不把诊断运行当模型验收通过。
5. 风险：hook同步及额外保存影响时序，只用于正确性定位；张量数据可能数十至数百MB，留X5，不入Git；参考和native trace形状须明确对齐。
6. 验证：本地compile/ruff/diff后Git同步同SHA；独立HF与native在相同prefix运行，输出层级JSON及算子误差，结合源码定位下一候选修复点。不得修改权重、target R4或阈值。

第11轮5ea2cf1b4def2fc79d77ef5a3968ece472922315：本地compile/ruff/diff通过，X5同SHA/clean/import路径确认；HF trace退出0，native trace退出1。原因是get_rope返回共享模块，每层附着的hook同时触发，尚未进入的layer1没有depth记录，报KeyError:1。属于诊断工具错误，生产路径未改变；原始日志round-11/native-trace.log保留。

### 候选12/15：共享RoPE hook归属

1. 目的：修复诊断工具对共享RoPE实例的错误归属。
2. 步骤：层入口记录当前物理层，RoPE hook仅记录对应当前层的输出。
3. 范围：ouro_trace.py和任务记录，不改生产代码。
4. 标准：HF/native trace完整生成，same-input算子对比可执行；原模型阈值保持。
5. 风险：hook影响时序，仍只用于正确性；不能把工具退出0误标为数值通过。
6. 验证：静态检查后本地commit/push、X5同SHA pull，重跑native trace与比较；第11轮HF数据由相同未改变的参考路径生成，可复用并核对prefix。

第12轮56005b4dca6e9ae453ff6992ea237db948e71207静态检查通过；X5 native-trace及trace-compare均退出0。prefix3第一层input/norm1/Q/K/V逐位相同，首个差异在attention输出(max0.03125/RMSE0.000654878)。相同输入算子对照192个循环层：norm2/3/4全部逐位相同，norm1仅1处不同(max0.00390625)；RoPE Q/K全部不同(max0.0625/0.03125)，SiLU product全部不同(max0.03125)，attention全部不同(max0.03125)。HF RoPE先将cos/sin及各乘积舍入到BF16再相加，原生融合RoPE保持FP32缓存与融合算术；HF SiLU先舍入再乘up，原生融合SiLU乘法省略该中间舍入。原始trace与JSON在round-11/round-12，未提交张量。

### 候选13/15：Ouro的HF舍入语义

1. 目的：对齐已由同输入实验确认的RoPE和SiLU中间舍入，保持完整R4及原数值阈值。
2. 步骤：Ouro专用RoPE包装器使用SGLang已有forward_native的dtype舍入路径；Ouro MLP保留原生packed gate/up及down投影，显式执行F.silu(gate)*up，避免融合激活或dual-GEMM省略中间舍入。
3. 范围：models/ouro.py与任务记录；不修改全局RoPE/activation、Raven、attention backend或KV/pipeline。
4. 标准：四prefix×四depth的native/compact hidden/logits通过原阈值、argmax一致；所有主要服务token/stop与参考一致，eager/graph生命周期通过。
5. 风险：增加kernel调用和graph节点，性能可能下降；残余attention softmax累加差异仍需原阈值检验，不能预先宣称已解决。复用shared RoPE对象不改变其全局forward方法。
6. 验证：本地compile/ruff/diff和既有权重测试；Git同步X5同SHA后先numeric native/compact，必要时读trace定位残差；通过后Ouro AR/单/双eager/graph、d1=2和TinyGEMM同配置配对、lifecycle/指标与显存采样。Raven未改，保留既有12组fixture证据。

### 第13轮本地结果与网络中断

候选ed8dd1c1032d8ef991a6b96fe75eb14b8110ef8b已push loopspec-test。本地compileall、ruff F、diff-check退出0，test_ouro.py 4项通过(0.013s)。额外CPU小张量检查实际OuroRotaryEmbedding(RotaryEmbedding(128,128,64,10000,True,bfloat16))，FP32 cos/sin cache、positions[0,1,7,31]、Q宽512/K宽256，输出与checkpoint BF16逐乘积舍入公式rtol=atol=0通过。CPU检查不替代GPU。

同步时X5-BUS连续SSH banner exchange超时(255)。直连X5仍被198.18.0.204关闭；已核实跳板InfiniAI-BUS可登录、getent解析x5.mit.edu为18.25.6.221，但从跳板socket.create_connection((18.25.6.221,22),10)也超时(1)。未修改网络、host-key或官方模型。已询问用户当前X5-BUS是否仍可登录。X5最后确认源码56005b4dca6e9ae453ff6992ea237db948e71207，第13轮尚未pull/验证；最后GPU快照均7/8MiB且无实验进程。native容器仍保留运行状态，因连接中断尚未停止；官方容器未启动。累计13/15，不因网络重试增加轮数，也不在无GPU证据下继续试改第14轮。

### 第10轮配对测量存档（不是第13轮性能）

来源round-10/performance-auto.json、performance-tinygemm.json，源码1557ec510；每组warmup2、10次相同5-token prompt/11-token completion，graph均开启，BF16/R4/Triton/page1/单请求，启用同一device timer。memory.csv为100ms采样的整个服务运行期GPU占用最大值，包含启动和lifecycle，不能称精确allocator峰值。该候选hidden阈值未通过，以下仅为该版本测量记录，不构成最新修复完成或一般性能保证。

| backend/mode | E2E中位数ms [min,max] | TTFT中位数ms | decode token/s | stream extend/decode均值ms | 采样显存最大MiB |
| --- | --- | --- | --- | --- | --- |
| auto AR | 236.949 [236.431,237.840] | 50.580 | 53.927 | 49.300/180.812 | 22883 |
| auto single | 235.459 [234.340,236.501] | 53.891 | 55.516 | 52.498/175.234 | 22947 |
| auto double | 209.579 [207.551,216.655] | 53.165 | 64.445 | 51.853/150.587 | 22969 |
| tinygemm AR | 267.492 [266.916,270.705] | 58.558 | 48.128 | 57.604/203.010 | 22883 |
| tinygemm single | 225.972 [224.942,231.276] | 59.957 | 60.638 | 59.265/160.049 | 22883 |
| tinygemm double | 199.438 [199.109,200.465] | 60.256 | 72.499 | 58.922/133.027 | 22905 |

同backend E2E比值auto single/double=1.0063/1.1306，tinygemm single/double=1.1837/1.3412。TinyGEMM AR本例比auto AR慢，不能只报告其相对比值而隐藏基线差异。时序块顺序采集、单短请求，不外推并发吞吐；stream计时可包含host提交间隙。

### 重连后的第13轮验证命令

先依运行约定确认X5 loopspec-test clean、GPU/端口空闲，ff-only pull，核对含ed8dd1c10源码的HEAD与本地一致、RO挂载/import路径；保留已有HF参考。容器内执行：

```bash
mkdir -p /results/round-13
cd /workspace/sglang/research/loopspec/experiments/2026-10-25-native-migration-plan
python diagnose_ouro.py native --reference /results/round-10/hf-depths.pt --output /results/round-13/native-depths.pt
python diagnose_ouro.py native --compact --reference /results/round-10/hf-depths.pt --output /results/round-13/compact-depths.pt
python diagnose_ouro.py compare --reference /results/round-10/hf-depths.pt --candidate /results/round-13/native-depths.pt --output /results/round-13/numeric-native.json
python diagnose_ouro.py compare --reference /results/round-10/hf-depths.pt --candidate /results/round-13/compact-depths.pt --output /results/round-13/numeric-compact.json
# 数值通过后，沿第9/10轮命令执行eager/graph、单/双、depth2、lifecycle，
# auto和TinyGEMM分别AR/单/双graph配对，外部100ms显存采样。
# 数值未通过则先读取最早差异，保留原阈值，再决定第14/15轮候选。
```


### 第13轮恢复后的GPU结果

直连X5恢复，clean ff-only pull至4cade66a1df44e3982ca6493b787c725d74112ee，与本地、容器import/mount核对一致。GPU0/1空闲，30126端口空闲。native/compact生成均退出0，比较均退出1；16个argmax相同，hidden最大误差均<=0.125，logits最大误差0.25、最大RMSE0.04670124通过。hidden RMSE仍有5处失败：(prefix,depth)=(0,3):0.02088499,(2,3):0.02016919,(2,4):0.02491019,(3,3):0.02471127,(3,4):0.02045178。原0.125/0.02阈值保持，原始文件在/results/round-13。服务与性能未重跑，先解决数值。

### 候选14/15：Ouro attention舍入

1. 目的：对齐HF eager的QK、缩放及归一化概率BF16舍入，保留原阈值。
2. 步骤：新增Ouro专用Triton两遍attention，第一遍归一化、第二遍将归一化概率舍入后乘V；backend保留原生KV写入、prefix/decode索引，Ouro单独选择此算子。
3. 范围：models/ouro.py、layers/attention/triton_backend.py、新kernels/ops/attention/ouro_attention.py和针对性test_ouro_attention.py。
4. 标准：四prefix四depth native/compact hidden max<=0.125/RMSE<=0.02、logits max<=0.5/RMSE<=0.1及argmax一致；服务token/stop与独立参考一致，graph/lifecycle通过。
5. 风险：两遍计算增加开销；不同归约实现仍可能产生误差；KV稀疏索引、GQA和graph动态长度必须验证。只作用Ouro，不替换请求Scheduler或官方runtime。
6. 验证：本地静态检查，Git commit/push/pull同SHA及import核对；GPU随机GQA/前缀/非连续KV/graph动态长度测试、16点数值；通过后Ouro eager/graph单/双/depth2、TinyGEMM配对和生命周期。性能报告实际结果，不预设加速结论。累计14/15。


### 第14轮结果

1095183bd9540f099711a065f507f58d735b68bc在本地/X5同SHA，import路径正确。静态检查通过；test_ouro_attention.py的随机GQA、非连续KV页、分块prefix、decode及改变query/索引/长度的graph replay全部退出0。native生成退出0，compare退出1：hidden最大误差0.171875、最大RMSE0.023990724，原阈值未通过；logits和argmax均通过。第13轮trace比较命令误加compare位置参数退出2，按原CLI去掉后退出0（未修改代码）；第14轮trace生成/比较均0。

第14轮prefix3逐层：第一层所有stage逐位相同，第二层Q/K/V、attention及o_proj也逐位相同，最早差异位于d00/l01/norm2.out，max0.00048828125/RMSE4.538e-6。192层same-input算子中RoPE和SiLU全部精确相同，attention只剩6处不同、最大0.001953125/RMSE2.3404e-5；norm1/2/4仍有1/3/2处不同。证明attention修复有效但不足以使递归误差达标，不能用小算子误差替代全模型验收。

### 候选15/15：Ouro RMSNorm归约顺序

1. 目的：修复trace已确认的最早RMSNorm归约舍入差异，不放宽阈值。
2. 步骤：Ouro的_norm使用原生RMSNorm已有force_native=True，使FP32 pow/mean/rsqrt及BF16 cast按checkpoint顺序执行。
3. 范围：models/ouro.py的norm构造与任务记录，不改变其他模型或全局norm。
4. 标准：原16点hidden/logits阈值及argmax通过；随后服务token/stop、eager/graph、生命周期、指标正确。
5. 风险：增加算子/graph节点和host调用开销；残余attention或GEMM舍入仍可能传播，不预设成功。
6. 验证：静态检查后commit/push，X5同SHA/import核对；native和compact诊断，通过后既定Ouro服务矩阵及auto/tinygemm配对性能、显存采样。累计15/15，若仍未达标不再修改源码或测试，保留证据报告剩余问题。


### 第15轮最终结果（未达到原数值验收标准）

候选0078e8a8ad501a987862d017885b3840cf63485b：本地compile/ruff F/diff-check退出0；Git push及X5 clean ff-only pull成功，HEAD/import与本地一致，模型仍原始Ouro-2.6B BF16 R4。GPU/30126端口在运行前检查空闲。所有原始文件位于X5 `/home/ruoxi/experiments/loopspec/2026-10-25-native-migration-plan/round-15/`。

- `diagnose_ouro.py native`及`native --compact`均退出0，两个compare均退出1；native/compact全部hidden/logits逐位相同。16点argmax均一致，logits最大误差0.25、最大RMSE0.041031711通过原0.5/0.1阈值。
- hidden三处失败：(prefix0,depth3) RMSE0.020862814；(prefix3,depth3) RMSE0.022747414；(prefix3,depth4) max0.25。其余13点通过。没有改动原max0.125/RMSE0.02阈值。
- native trace及ouro_trace.py比较均退出0。prefix3前29个物理层全部stage逐位相同，首次分歧为d00/l29/attention，max0.001953125/RMSE0.0000196331。192循环层same-input检查：所有四种norm、SiLU、Q/K RoPE均逐位相同，attention仅2处不同；该算子级改进不能替代整模型失败。
- `test_ouro.py`(4)、`test_loopspec.py`(4，包含三组各100000次采样分布)、`test_raven.py`(5)、`test_timeline.py`(2)、`test_rejection_graph.py`(4，实际CUDA replay)、`test_ouro_attention.py`(extend/decode/dynamic graph)、`verify_tinygemm.py`全部退出0。
- `SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-15/{ar,single,double}-graph --cuda-graph --lifecycle [--depths 1 / --depths 1 2]`三组退出0。每组warmup2+10主请求，30次output_ids全部与round-2独立HF reference一致，stop={type:stop,matched:0}，输出11token ` Paris.\n\nThe final answer is A.`。lifecycle的max0/1、stop、stream、cancel/after_cancel、OpenAI completions、sampling、flush均通过。
- single测量窗口100个verified：accept1=70/residual=30、cycles220/reused70/restart30；double：accept1=70/accept2=20/residual=10、cycles180/reused90/restart10。不是退回AR的假通过，metrics标签/通用NaN语义检查通过。
- 数值门槛未过，因此不做本轮配对性能比较、不发布加速结论。harness自然产生的延迟字段保留为原始服务日志；本轮未做新显存峰值采样。新的eager/depth2/TinyGEMM整模型矩阵尚未重新执行；不能用第9/10轮数据冒充当前版本结果。Raven真实checkpoint依然未验收，既有fixture证据及本轮结构单测不等同真实模型测试。

### 只读追加定位：softmax的BF16边界

没有修改源码或测试。在已保存的第15轮d00/l29张量上重算同输入HF eager：head12/query1只有两个可见key，缩放后BF16 scores=[0.11962890625,0.12353515625]。PyTorch FP32 softmax=[0.4990234673023224,0.5009765625]，转BF16=[0.5,0.5]；FP64概率=[0.4990234387417615,0.5009765612582384]，同样舍入到[0.5,0.5]。

把概率手工设为[0.498046875,0.5]再执行PV，128个输出坐标与当前Triton输出逐位相同；HF概率[0.5,0.5]则有49个坐标不同、max0.001953125。固定HF概率后FP64 PV与HF PV逐位相同，FP64 softmax再BF16也与HF输出相同。因此该首处分歧可以由第一概率向下错舍入一档完全解释，优先继续检查当前tl.exp/归一化的近似精度，而非再次改norm、权重或timeline布局。

建议下一候选（尚未实施，需新预算）：在ouro_attention.py对指数和归一化采用更准确且明确的浮点路径，加入上述半精度边界及多长度归约测试；仍保留两遍、分页KV、动态graph和原数值阈值。先检查此首处分歧，再完整16点native/compact，全部通过后补齐eager/depth2/TinyGEMM服务与配对性能。不得只针对固定prefix或某个head写特例。

累计15/15，停止源码及测试修改。已完成修改始终保留在loopspec-test；loopspec正式分支仍baa9befa381511cd4d56f83980951e5a45fffc6d，未晋升。


### 2026-10-28追加授权及候选16/25

用户明确追加10轮，累计预算25轮，不重置历史15轮，不改变原阈值，不授权晋升正式分支。
1. 目的：修复已复现的softmax概率在BF16边界向下错舍入问题。
2. 步骤：Ouro kernel采用libdevice.exp和tl.div_rn，避免近似exp2及近似除法改变舍入方向；增加可独立复现该概率边界的测试。
3. 范围：kernels/ops/attention/ouro_attention.py、test_ouro_attention.py、任务记录。
4. 标准：边界概率与HF逐位一致，16点hidden max<=0.125/RMSE<=0.02、logits max<=0.5/RMSE<=0.1及argmax一致，后续服务token/stop通过。
5. 风险：指数/除法开销增加；长序列归约及其他边界仍可能失败，不能预设成功。
6. 验证：本地静态检查、Git同步同SHA；GPU kernel/prefix/graph测试、native/compact数值；通过后既定服务矩阵与auto/TinyGEMM配对性能及显存采样。累计16/25。


### 标准服务入口与当前支持边界

在已安装本分支、依赖与X5验证环境一致的环境中，双草稿运行命令如下（相对AR只需增加算法及draft-depth选择；其余均为SGLang已有参数）：

```bash
python -m sglang.launch_server \
  --model-path /home/ruoxi/models/Ouro-2.6B \
  --host 127.0.0.1 --port 30126 \
  --dtype bfloat16 --attention-backend triton \
  --max-running-requests 1 --page-size 1 \
  --disable-overlap-schedule --disable-radix-cache \
  --chunked-prefill-size -1 --context-length 2048 \
  --mem-fraction-static 0.7 --enable-metrics \
  --speculative-algorithm loopspec --speculative-draft-depth 1 2

curl http://127.0.0.1:30126/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

- AR：去掉speculative-algorithm与speculative-draft-depth；单草稿改为`--speculative-draft-depth 1`，另验收depth2；Ouro target保持完整R4。
- CUDA graph默认开启；eager添加已有`--disable-cuda-graph`。TinyGEMM使用已有参数的扩展值`--bf16-gemm-backend tinygemm`；没有新增独立TinyGEMM开关。不得为本模式指定独立draft-model-path。
- HTTP仍为原生/generate、stream、/v1/completions及管理/取消入口，LoopSpec没有新HTTP路由或自定义服务启动器。
- 请求接收、队列、逐次调度及结果处理属于原生SGLang；LoopSpec worker迁入persistent lane/branch/rejection算法，使用原生ModelRunner/ForwardBatch/KV协议。Ouro新增checkpoint兼容的attention算术路径；不import官方sglang_recurrent runtime，不接管为官方自写的整请求scheduler。
- 本阶段支持单活跃请求、TP/PP/DP/CP=1、CUDA/Triton、page1，关闭overlap/radix/chunked-prefill。多请求continuous batching、分布式并行、PD、LoRA、grammar等未在本计划中承诺支持；非法组合显式拒绝。
- Raven的pre/core/post、injection、tokenizer和配置接口已迁入并通过结构及合成checkpoint测试。没有真实Raven checkpoint的验收结果，不宣称真实Raven模型质量或性能已验证。


### 第16/25轮最终验收：通过（2026-10-28）

**版本与环境。** 源码7966c680135e38699eb2d5273692adf8111ff289，本地loopspec-test提交/push、X5 clean ff-only pull、HEAD及/workspace/sglang/python/sglang/__init__.py一致。使用既有loopspec-native-20261026容器（源码/模型只读挂载，/results可写），/opt/sglang/bin/python，Torch2.14.1+cu130、Transformers5.19.0、Triton3.8.0，X5 GPU0 RTX5090、驱动580.178.04。运行前GPU0/1为7/8MiB且无计算进程，30126空闲；模型config/tokenizer SHA256与§6.1一致，Ouro-2.6B、BF16、完整R4未变。

**修复。** Ouro两遍attention改用libdevice.exp和tl.div_rn，避免近似指数/除法使已归一化概率跨越BF16舍入边界。复用原生KV索引及请求调度，没有对特定prompt/head写特例。新增10组正负/顺序/相邻间距边界测试，V直接读出概率，要求与HF逐位相同。

**数值验收。** 本地compileall/ruff F/diff-check均退出0。X5 test_ouro_attention.py退出0，覆盖边界概率、GQA、非连续分页、带前缀extend、decode及改变query/页索引/长度后的graph replay。diagnose_ouro.py native、native --compact及两次compare均退出0。两种布局所有hidden/logits逐位相同；16个depth点argmax全部相同。

| 指标 | 本轮最大值 | 原阈值 | 结果 |
| --- | --- | --- | --- |
| hidden最大绝对误差 | 0.109375 | 0.125 | 通过 |
| hidden RMSE | 0.0112857055 | 0.02 | 通过 |
| logits最大绝对误差 | 0.125 | 0.5 | 通过 |
| logits RMSE | 0.0268899128 | 0.1 | 通过 |

**功能回归。** test_ouro(4)、test_loopspec(4)、test_raven(5)、test_timeline(2)、test_rejection_graph(4，含实际CUDA replay及采样分布)和verify_tinygemm均退出0。服务矩阵如下，每组调用既有serve_smoke.py，warmup2+repeat10并带--lifecycle，全部退出0：

| 输出目录 | backend | graph | draft depth |
| --- | --- | --- | --- |
| ar-graph | auto | 开 | AR |
| single-graph | auto | 开 | 1 |
| double-graph | auto | 开 | 1 2 |
| ar-eager | auto | 关 | AR |
| single-eager | auto | 关 | 1 |
| double-eager | auto | 关 | 1 2 |
| depth2-graph | auto | 开 | 2 |
| tiny-ar-graph | tinygemm | 开 | AR |
| tiny-single-graph | tinygemm | 开 | 1 |
| tiny-double-graph | tinygemm | 开 | 1 2 |

所有100个主请求都逐token匹配独立HF round-2/reference.json及此前官方AR对照；5个prompt token、11个completion token `[7042,30,198,198,504,3403,2988,314,330,30,0]`，停止信息均`{type:stop,matched:0}`。服务显示` Paris.\n\nThe final answer is A.`（HF原始decode还包含EOS文本标记，比较使用token ID与停止位置）。每组9项lifecycle为max0/1、stop_string、stream、cancel、after_cancel、OpenAI completions、sampling、flush，全部通过。

单草稿主窗口100个verified：accept1=70、residual=30，cycles220/reused70/restart30；双草稿accept1=70/accept2=20/residual=10，cycles180/reused90/restart10；depth2 accept1=90/residual=10，cycles120/reused90/restart10。TinyGEMM的单/双接受及复用计数与auto相同。指标标签、计数分母和通用spec NaN语义断言通过，证明实际执行草拟/验证/拒绝及复用。

**配对性能。** 两次compare_performance.py均退出0，固定同一短prompt、R4/BF16/Triton/graph/内存预算/停止策略；仅在同backend内配对AR与单/双。10次测量前warmup2，串行测量块。以下为本轮真实数据，不外推并发吞吐或一般任务加速：

| backend/mode | E2E中位数ms [min,max] | TTFT中位数ms | decode token/s | stream extend/decode均值ms | 100ms采样显存最大MiB | 同backend E2E倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| auto AR | 391.440 [390.705,394.043] | 97.002 | 34.102 | 95.834/288.113 | 22913 | 1.000 |
| auto single | 354.791 [354.182,362.131] | 98.828 | 39.207 | 98.040/249.930 | 22953 | 1.103 |
| auto double | 318.209 [316.771,319.033] | 97.985 | 45.747 | 96.581/213.643 | 22975 | 1.230 |
| tinygemm AR | 436.240 [435.438,440.033] | 110.895 | 30.839 | 110.160/319.167 | 22913 | 1.000 |
| tinygemm single | 353.776 [353.160,354.739] | 112.540 | 41.622 | 110.893/235.249 | 22889 | 1.233 |
| tinygemm double | 314.881 [314.350,315.386] | 111.813 | 49.568 | 110.250/196.566 | 22911 | 1.385 |

TinyGEMM AR比auto AR慢，不能只用1.385×掩盖基线差异。checkpoint舍入对齐增加kernel/graph操作，本轮AR绝对延迟高于第10轮；第10轮hidden验收失败，其延迟不作为有效正确性基线。TTFT包括服务调度，stream计时可能包括host提交间隙；显存为nvidia-smi每100ms整个服务期间采样最大值（包括启动/lifecycle），不是精确allocator峰值。

**复现与证据。** 原始日志、数值PT/JSON、每次请求、command.json、lifecycle.json、events.json、metrics前后快照、memory.csv和性能JSON均在X5 `/home/ruoxi/experiments/loopspec/2026-10-25-native-migration-plan/round-16/`。代码/测试通过Git同步；大型张量与日志没有进入Git。命令采用既有脚本：

```bash
# 容器内，PYTHONPATH=/workspace/sglang/python
cd /workspace/sglang/research/loopspec/experiments/2026-10-25-native-migration-plan
python test_ouro_attention.py
python diagnose_ouro.py native --reference /results/round-10/hf-depths.pt --output /results/round-16/native-depths.pt
python diagnose_ouro.py native --compact --reference /results/round-10/hf-depths.pt --output /results/round-16/compact-depths.pt
# 分别compare native/compact，参数与前轮相同，原阈值未变。
SGLANG_RUST_BUILD_MODE=auto python serve_smoke.py --output /results/round-16/double-graph --cuda-graph --lifecycle --depths 1 2
# 其余9组按上表选择graph/depth/backend；每组实际完整命令在command.json。
python compare_performance.py /results/round-16/ar-graph/benchmark.json /results/round-16/single-graph/benchmark.json /results/round-16/double-graph/benchmark.json --output /results/round-16/performance-auto.json
python compare_performance.py /results/round-16/tiny-ar-graph/benchmark.json /results/round-16/tiny-single-graph/benchmark.json /results/round-16/tiny-double-graph/benchmark.json --output /results/round-16/performance-tinygemm.json
```

累计16/25，达到计划内Ouro验收标准，停止不必要的迭代。Raven真实checkpoint与多请求/并行扩展仍按既定边界标记未验收/未支持，不将Ouro成功扩大为全模型全配置保证。正式loopspec分支保持baa9befa381511cd4d56f83980951e5a45fffc6d，只有用户明确验收授权才晋升。
