# LoopSpec → SGLang v0.5.22 迁移研究与实施计划

日期：2026-10-25（Asia/Shanghai）。状态：2026-10-26 用户批准实施；第 2 轮进行中。代码修改与验证轮次：2/5。

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
