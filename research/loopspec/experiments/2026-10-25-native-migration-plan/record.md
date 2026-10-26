# LoopSpec → SGLang v0.5.22 迁移研究与实施计划

日期：2026-10-25（Asia/Shanghai）。状态：2026-10-26 用户批准实施；用户已授权继续，总预算10轮；第7/10轮本地静态检查通过，X5 SSH持续超时，第6/7轮GPU待验证。

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
