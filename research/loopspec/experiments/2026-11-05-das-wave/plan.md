> 历史归档（2026-11-06）：DAS-Wave 已按用户要求撤除；本文描述撤除前的实现和结果，不代表当前可用算法。DFlash/LoopSpec 公共工具与修复保留。见 [撤除记录](../2026-11-06-remove-das-wave/record.md)。

# DAS-Wave 原生实现与 AIME25 实验计划

日期：2026-11-05（Asia/Shanghai）。目标是在 SGLang v0.5.22 fork 中增加独立的 `DAS_WAVE` speculative algorithm，先验收单请求，再验收连续多请求，最后在 X5 上用 Ouro-2.6B-Thinking 完成 AIME25 全30题、bs=1 的 AR、DFlash、LoopSpec、DAS-Wave 对比和逐阶段 profiling。本文保留最初的设计与验收口径；实施后的最新结果见 [status.md](status.md) 和 [record.md](record.md)。已用10/10轮，原生V2与graph路径、DFlash导出/接入已验证；AIME25 25组全30题已完成，见最新报告。严格生成数值一致性仍未通过；下文的起始配置与预算为历史计划，不代表当前状态。

## 已确认的基线

| 对象 | 已确认内容 | 对实施的影响 |
| --- | --- | --- |
| 实验源码 | 起始 `2eaa51692ff3f6da104a7f1b7b5cc8996dbb2439`；v0.5.22 基线 `baa9befa381511cd4d56f83980951e5a45fffc6d` | 不引入其他 main/研究分支提交 |
| 正式晋升 | `82a851f6fe12c6e17354054fd7b160594f203b66`，25 个运行时文件 | 与起始实验 runtime tree 一致 |
| 历史连接 | `099cfd389c4c02e478d45b22c974ed658a7e874a` | 第一父为原实验 HEAD，第二父为正式晋升；merge 前后 tree 一致 |
| 官方 LoopSpec | `41aa710ed7c9ce908cffa03597be5d1df19d7b56` | 只读参考 |
| Target | `/home/ruoxi/models/Ouro-2.6B-Thinking`；R=4，48 core layers，hidden=2048，vocab=49152，BF16 | 固定完整 target，不改原始 config/权重 |
| Draft checkpoint | `/home/ruoxi/models/Ouro-2.6B-Thinking-DFlash/training_state.pt` | 训练态，须导出独立推理目录 |
| Draft 元数据 | step10000，5层，block size=16，mask token ID=16，feature layers=(1,12,23,34,45) | 不能套用其他模型配置或默认 feature 选择 |
| AIME 文件 | `/home/ruoxi/repo/test/datasets/aime.jsonl`，30条，ID 0–29，字段 problem/answer/id | 必须逐题核验 AIME2025 I/II 来源映射，不能只凭条数认定版本 |
| 数据 SHA256 | `b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331` | 全算法共享冻结数据 |
| 资源快照 | 2026-11-05 15:18 CST，两张 RTX5090 均无计算进程 | 真正启动实验前重新检查 GPU/端口，不视为资源预留 |

历史多 batch 工作包含逻辑、生命周期和性能验证，也明确记录 bs=1/2/4 的固定 argmax 翻转比例门槛失败。不能将本次晋升等同于严格逐 token 无损证明。新实现必须独立检查相同前缀的 target 条件分布与自由生成序列；保留所有分歧和原始门槛。

## 论文约束与 Ouro 适配

来源为 [DAS-Wave 论文 v1](https://arxiv.org/pdf/2609.34538)，重点 §3.2–3.4、§4.3–4.5、附录 A.2–A.3/B。论文实验是 Huginn/Raven 的 R=32，主实验为 greedy；没有 Ouro R=4 的实测保证。当前未取得作者官方实现，不把下面的自主移植称为官方代码复现。

论文算法要求：draft 在同一 core layer 读取已验证、已物化前缀的 full-depth K/V；当前位置 K/V 仍来自当前 draft，两者共享一个 causal attention softmax。窗口 `W_j=min(gamma,W_init+j)`，跨轮保留每个位置的 latent，每轮重建临时 draft KV。target 从普通初始化独立计算 `[anchor,y1,...,y_(w-1)]` 的完整 R 次循环，验证对应的 w 个 proposals。首个 mismatch 输出 target token，丢弃后缀；全接受且 m=w 时不额外生成 bonus。最后已输出但未物化的 token 成为新 anchor。这些约束分别对应 §3 和附录 B。

论文主配置为 gamma=32、J=24、W_init=2，Raven k=1、Huginn k=2；A 表示每次验证平均接受的 draft 数，不是接受概率。移植保持算法语义，但需要为 Ouro 单独冻结参数。[配置和指标出处](https://arxiv.org/pdf/2609.34538)

Ouro 的 `prepare_recurrent` 用 embedding 初始化 hidden，`advance_recurrent` 不接收有效的独立 input injection。因此主移植按 Ouro 原有递推保留 latent：已有位置即使 guess 变化也不重置、不人为加入新 embedding；新位置按当轮更新后的 guess 初始化。旧位置后续仍会通过 causal attention 接收活跃前驱的变化。它是论文过程在无 injection 模型上的明确实例化，其 proposal 质量是待验证假设；full-depth verifier 仍保证算法层面的 greedy 目标保持。

分别做 carry/no-carry、aligned/deep-KV 的 2×2 消融。无 carry 对照每轮按新 guess 重新初始化；不得把该对照命名为完整 DAS-Wave。若后续尝试 embedding-delta injection，应作为独立扩展，不能暗中代替主方案。

## 修改前六项计划

1. **目的**：增加与 `DFLASH`、`LOOPSPEC` 并列的原生算法；保持完整 R4 target；建立可审计的单请求和多请求 KV/状态生命周期；给出完整解题负载下的真实速度和开销。
2. **步骤**：冻结数据/模型清单 → bs=1 算法和 CPU 契约 → X5 AR/target/rollback 验证 → 多 batch 调度与所有权 → DFlash 导出及 Ouro feature 接入 → 独立开发集选参 → 全30题测量 → 单独 profile 和归因。
3. **范围**：下表列明 runtime 模块；测试、实验脚本及轻量报告放本目录。全部新代码只在 `loopspec-test` 开发和测试；当前正式授权仅覆盖已晋升的旧快照，新 DAS-Wave 不自动晋升。
4. **标准**：同前缀 verifier 与 full-R AR 契约正确；greedy token/stop 差分通过，或明确报告数值失败而不宣称严格无损；无跨请求污染、泄漏、重复释放；bs=1/2/4/8/16 生命周期验证；每算法30个独立题目完整记录；指标可由原始计数重算。速度提升不是正确性的替代指标，测得慢于 AR 也必须报告。
5. **风险**：Ouro carry 提议质量、R4 下 draft 开销、成熟 KV/anchor 错位、DFlash 训练 feature 语义、BF16 shape 差异、graph padding、长上下文显存、profile 扰动和重复计时。各自用独立验证或消融定位。
6. **验证**：本地 AST/CPU 状态机测试 → Git commit/push → X5 同 SHA pull/环境核对 → full-R AR → greedy 与 KV 生命周期 → 多 batch → checkpoint/export/reference parity → 无 profiler 配对计时 → 单独 profile。命令和矩阵见后文；未执行项不算通过。

DAS-Wave 及配套接入共用最多10轮修改＋验证预算，当前0/10。bs=1到多 batch、数值或环境相关代码修复均延续计数。每个候选开始前在 record.md 写 n/10、假设和文件，结束写 SHA、命令、退出码及结论；第10轮仍未通过则停止源码迭代。下面的里程碑不是额外预算。

## 原生接口和模块划分

| 文件或模块 | 计划变化 |
| --- | --- |
| `python/sglang/srt/speculative/spec_info.py` | 增加 `DAS_WAVE`、`is_das_wave()`、validator/worker 分发及独立状态类型；保持其他算法判断语义 |
| `python/sglang/srt/arg_groups/fields/spec.py`、`field_order.py`、`speculative_hook.py` | 注册/help/序列化新参数，不借用 LoopSpec depth 参数或伪装成 EAGLE |
| 新 `speculative/das_wave/config.py` | 验证 gamma/J/k/W_init 与支持范围；拒绝不支持的采样和运行模式 |
| 新 `speculative/das_wave/state.py`、`worker.py`、`info.py` | 每请求 wave 状态、draft refinement、target block verification、native V2 输出与 filter/merge |
| `mem_cache/timeline_pool.py` | 增加专用 draft 映射：prefix 映射 R-1 平面，active suffix 映射独立临时平面；target 映射保持逐深度 |
| `model_executor/recurrent_runner.py` | 复用 prepare/core/readout；提供显式的 draft/target block 执行接口，不能把 carried state 传入 target |
| `managers/scheduler_components/recurrent.py`、`tp_worker.py`、`model_runner.py`、KV configurator | 明确选择 DAS-Wave 执行和分配方式，接入原生调度、取消、retraction、结果发布 |
| `models/ouro.py` | 尽量保持 target 数学不变；为 DFlash 增加与训练契约一致的 feature capture |
| `model_runner_components/spec_aux_hidden_state.py` | 明确 Ouro 的循环 feature 索引语义，禁止按普通层数静默猜测 |
| `utils/recurrent_trace.py`、DAS-Wave worker、DFlash worker | 可关闭的循环/阶段统计及关联 ID，常规性能运行关闭重型 trace |

拟提供的启动方式（参数为待实现接口，不能现在直接运行）：

```bash
python -m sglang.launch_server \
  --model-path /home/ruoxi/models/Ouro-2.6B-Thinking \
  --speculative-algorithm DAS_WAVE \
  --speculative-das-wave-max-width 16 \
  --speculative-das-wave-rounds 4 \
  --speculative-das-wave-updates 1 \
  --speculative-das-wave-initial-width 2 \
  --attention-backend triton --page-size 1 --disable-radix-cache \
  --max-running-requests 1
```

gamma=16、J=4、k=1、W_init=2 仅作首个可运行候选，最后活跃宽度为5，不能把它报告成每次16个 proposal。通过后在独立开发集从 gamma={4,8,16,32}、J={2,4,8,24}、k={1,2} 中筛选预算可比配置；仅合并 k 和完整宽度序列都相同、且容量不改变执行形状的重复配置；不能因最终宽度相同就合并不同 J，额外 carry 轮数仍会改变提议和成本。记录真实宽度与计算量。保留论文 gamma32/J24 的对照，最终参数在 AIME25 测试前冻结，不能用30题中的最佳速度挑参数。

初期支持 Ouro/TP=PP=DP=CP=1、BF16、Triton/page1、无 radix cache、greedy。非零 temperature、grammar、penalties、LoRA、特殊缓存、分离式服务等未验证路径显式报错；后续扩展另定标准，不静默回退为另一算法。bs=1 先关闭 overlap、eager 验证，之后接入 native overlap/graph；比较时公开各路径和 graph coverage。

## 单请求状态与 KV 实现

每个原生 Req 保存 `rid/epoch`、已物化前缀长度、anchor、guess IDs、carried hidden、active width、round、临时 slot 列表、target block slot 列表和累计指标。状态归属 Req，不依赖可变 batch 行号；不创建伪造的用户子请求。

使用三种互不混淆的存储：

| 存储 | 可被谁读取 | 生命周期 |
| --- | --- | --- |
| 全深度已验证前缀 | target 按自身 depth；draft 只读最后 depth 的对应 core layer | 保留到请求完成或合法 retraction |
| 临时 draft K/V 与 latent | draft 当前轮可读 KV；下一轮只继承 latent | KV 每轮覆盖/清空有效位；不发布到原生前缀 |
| target speculative block | 独立 target 的当前 depth | 接受的已物化输入转移到前缀；拒绝后缀回收 |

`FixedDepthSlots` 有地址不代表 token 已物化。对 draft map 必须显式检查 `position < materialized_prefix_len`，仅对这些位置选择 depth R-1；临时 active suffix 只能读取本轮 causal 可见的位置。统一构建 prefix+suffix 索引后一次 attention normalization，不能把两次独立 softmax 的结果相加。最初复用 Triton 索引能力，不先写新的 attention 算子。

target verify 输入长 w，输出 w 个条件分布。若 a 个 proposal 接受后发生 mismatch，则输出 a+1 个 token，保留对应的 a+1 个 target 输入（anchor 加已接受候选）的完整 KV，纠正 token 成为未物化 anchor。若全接受，则输出 w 个 token，保留 w 个输入 KV，最后一个 proposal 为新 anchor。EOS/stop/长度上限可能进一步截断：只有最终实际保留的输入可转移所有权，预计算后缀必须释放。验收逐一覆盖 a=0、a=w-1、全接受、w=1、EOS位于接受前缀/纠正 token、remaining<w。

target 必须取得每个 verify 位置的 logits，不能误用普通 EXTEND 只返回最后一行的 readout；选择明确的 TARGET_VERIFY/logits capture 契约，并核对 native `accept_lens/new_seq_lens/speculative_output_stride`、logprob 行与输出 ID 的映射。prefill 输出首 token 作为首 anchor，首 token 不算 draft acceptance。

## 单请求验证矩阵

| 顺序 | 验证 | 成功判据 |
| --- | --- | --- |
| A | CPU 可控 toy target，枚举长度/拒绝位置/停止 | 与逐步 greedy AR 的 token 和停止位置完全一致；全接受不多发 bonus |
| B | KV sentinel/所有权测试 | 跨 depth/core layer/branch 的读取符合映射；draft 不写成熟前缀；target 不接触 carry |
| C | Ouro full-R AR 与 checkpoint 官方 HF 对照 | 相同输入、完整R4、正常循环归一化，记录各层/loop/logits；任何分歧定位并保留 |
| D | teacher forcing verifier 对 AR 同前缀 | 覆盖 w=1/2/4/8/16/32、prefix 1/128/512/2048/8192及短尾；检查 logits/argmax/target KV |
| E | 自由生成 | 独立 smoke/dev prompts 的完整 token IDs 和 stop 一致；首差异记录 prefix、margin、backend/shape |
| F | 请求生命周期 | 多次顺序请求、取消、EOS、max_new_tokens=0/1、chunked prefill、flush、强制显存压力/retraction后无泄漏 |
| G | graph/overlap | eager 与 graph、同步与 overlap 的同前缀和自由生成对照；padding 不进入 attention/计数 |

严格 token 一致是首要检验。若 BF16 shape 差异导致翻转，先用同 shape 串行 target、参考算子/FP32 诊断区分数值与算法；不能以重新校准阈值掩盖失败。可另报告“在已声明数值包络内”的研究结果，但它不替代严格无损验收；性能表同时保留输出差异和答案准确率。

本任务主方法只实现 greedy，不声称具有论文外的 sampling 保证。若未来支持 sampling，需保存实际 proposals 并实现正确接受/残差分布，做小词表精确枚举及统计检验，不能以同 seed 序列对齐代替。

## 多 batch 实现与验证

bs=1 通过后，将不同 Req 的 active positions 按 request×width 打包为 ragged rows，使用 prefix lengths、token-to-request、positions、round 和 materialization mask 绑定各自状态。初版同步推进同阶段 wave；随后按阶段/宽度分桶，允许 prefill 插入及请求独立结束。不能简单循环调用 bs=1 worker 并称作多 batch 实现。

target 独立以 ragged/padded block 验证，各请求接受长度不同；按每行 accept length 转移 KV、发布下一 anchor，再进入 native filter/merge。任何 batch 重排都必须按 Req 身份恢复 carry/slot，graph replay 输出在下次 replay 前拷贝到 owner buffer。资源预算计入 B×gamma 的临时 draft、B×gamma×R 的 target 物化空间及 prefix；临时空间 OOM 时走有记录的 retraction/重新 prefill，不能污染存活请求。

验证 bs=2/4/8/16，各100组 seed 固定的随机 CPU 状态机批次，加 X5 同时和交错到达请求；prompt 长度混合、remaining length混合、不同拒绝位置、不同停止轮次。检查：

- 同一个请求在单独执行和批量执行下的 token/stop；数值异常按照前述固定规程归因。
- filter/reorder/merge、取消首中尾请求、取消后新请求复用 slot、先完成请求不再推进 hidden。
- 小 KV 容量强制 retraction，恢复后的 target 前缀一致；严格 invariant checker 看到临时 owned slots。
- full accept/首拒/中拒混合；没有跨行 KV/guess/hidden 访问；反复请求完成后 allocator 回到初始可用量。
- eager/graph、overlap on/off、chunked prefill；同 shape 桶核验实际图命中和 padding。

多 batch 正确性与吞吐矩阵独立于用户要求的 bs=1 AIME25 主实验，不能用多并发吞吐代替单请求加速比。

## DFlash 推理前置工作

X5 训练文件 SHA256 在原目录 SHA256SUMS 中。`training_state.pt` 包含完整 `draft_state_dict` 和训练元数据，可用 CPU `torch.load(weights_only=True)` 读取；四份原文件保持原样，不把 optimizer 状态当推理权重。

1. 获取并核对训练配置原件：元数据记录路径 `/homes/ruitaol/loop/configs/experiments/specforge/ouro-2.6b-thinking-dflash.json`，内容 SHA256 `74cbfab3ab3b5567cc9dd6af20efe64707e118dff789d7d3b605b1fa2ca44ee4`。确认 attention heads/KV heads、QK norm、RoPE、feature normalization、mask embedding 和 tokenizer。tensor shape 只能确定部分配置，不能凭 shape 猜训练语义。
2. 获取训练期 target wrapper 的 feature 捕获定义：`(1,12,23,34,45)` 是哪一轮、layer 输入还是输出、循环结束 norm 前后。Ouro 配置用 `total_ut_steps=4`，现有通用 DFlash resolver 只识别 `num_loops`；既不能盲目乘4改成192层，也不能默认这5项来自最后一轮。
3. 编写本地版本化 exporter，经 Git 到 X5，在新目录 `/home/ruoxi/models/Ouro-2.6B-Thinking-DFlash-inference-step10000` 生成 config/safetensors/manifest，不覆盖原训练文件；保留 tensor 名称映射和 strict missing/unexpected keys 检查。
4. 原生 Ouro 实现 `set_dflash_layers_to_capture`、正确返回各位置 aux hidden，覆盖 prefill 和 verify、graph capture；target 正常 logits 不受是否开启 feature capture 影响。与训练 wrapper 对照选中 feature 和 draft logits 后才能开始性能比较。
5. 使用现有 `DFLASH` worker、block size=16，不替换成其他 draft。当前实现16包含 anchor，通常只有15个 draft candidates；统计从真实 verify mask/counters 提取，bonus 单独记录。

计划中的导出与验证文件为 `export_dflash_checkpoint.py`、`verify_dflash_features.py`、`verify_dflash_greedy.py`。若训练配置/feature 语义仍缺失，DFlash 项明确阻塞，不能随机初始化或替换 checkpoint 来补齐四算法表。

## AIME25 全30题实验

先将本地数据每题映射到2025 AIME I/II各15题及参考答案，保存来源/revision/hash和题号；第一题已与公开 [AIME2025 数据](https://huggingface.co/datasets/MathArena/aime_2025/viewer) 对照，尚不能据此认定其余29题全部核验完成。答案仅用于离线评测，不拼入输入。调参开发集必须与这30题不重叠。

统一 checkpoint 自带 chat template，显式 `enable_thinking=True, add_generation_prompt=True`，冻结完整渲染文本、prompt token IDs、tokenizer hash、EOS/stop IDs。所有算法 temperature=0、无额外 penalties，R=4、相同 BF16/硬件/attention backend，bs=1 串行发送。保留推理和最终回答的全部 tokens。最大新 token 初始8192，预留prompt与verify临时空间；实际容量核验后冻结全算法统一上限。Ouro未量化KV约每token `2×4×48×16×128×2 bytes = 1.5 MiB`，仅32768 token的KV就约48 GiB，单张32 GiB RTX5090不能据模型65536的配置值承诺该长度。若有题触顶，标为截断，在共同显存预算允许时提高上限并重跑对应的全部配对方法；仍放不下则如实报告资源限制，不用滑动窗口或削减target depth换取“完成”。不能沿用历史512 token截断结果称为完整解题。

主表采用 AR、DFlash-b16、LoopSpec(draft depth=1)、DAS-Wave(冻结参数)。为覆盖用户的 Depth 1/3 要求，另外完整运行 LoopSpec(draft depths=[1,3])，分别统计两个 proposal endpoint。R4 当前约束不允许把单独 depth=3 当作合法 first draft depth；不会悄悄改成R3。若 Depth 1/3 指一次token的浅层1轮+后续3轮，profile也提供这个分解，见下一节。

主表使用同一公开配置下的原生AR基线，并记录普通AR/CDB和各spec实际执行路径。当前 `use_recurrent_execution` 只为AR/LoopSpec启用CDB，而DFlash走普通target runner；需补充同一target执行方式的AR控制组来分离调度收益，不能把系统路径差异全归为算法收益。

所有算法至少2次独立非测试 prompt 预热；随后3个完整重复，每个重复均包含全部30题，按预先冻结的题目顺序/算法轮换顺序运行，同一 GPU 串行占用。每个重复内 AR 与各候选成对，固定运行环境/graph 配置并记录各自实际捕获形状；不让 profiler 与主计时并行争用 GPU。首轮30题完成且正确性核对后，才扩展重复数，不删困难题或失败题。

记录每题的 load/warmup 分离信息、prefill/TTFT、decode、E2E wall、输出token数、stop原因、答案、正确/错误/未解析、截断、accept计数和显存。decode 起点在首token后，分母为其后输出token数，避免把首token重复算入；另提供整个生成过程的 server device 时间。无Profiler计时以 perf_counter/已存在device timer为准，不用 profiler 总时长作为主加速比。

| 指标 | 计算与报告方式 |
| --- | --- |
| 总耗时 speedup | 同次重复 `sum(T_AR)/sum(T_method)`；仅在输出 token/stop一致时解释为同工作量耗时比 |
| 总吞吐 speedup | `(sum(N_method)/sum(T_method))/(sum(N_AR)/sum(T_AR))`，decode/E2E分开 |
| 逐题平均 speedup | 对30个 `(N_method/T_method)/(N_AR/T_AR)` 取均值，配对 bootstrap 10000次给95% CI；同时给中位数 |
| 平均接受概率 | 每题 `accepted_tested/tested_proposals` 后对30题宏平均；另给全局微平均 |
| 草稿利用率 | `retained_accepted/generated_proposals`；包含首拒后丢弃的未测试suffix，单独命名 |
| 平均接受长度 | 每题 `retained_accepted/verify_cycles`，再取题均值；另给每cycle实际输出数含fallback/bonus |
| LoopSpec | depth1、depth3分别记录generated/tested/accepted；depth3条件接受率分母仅为实际进入第二验证的候选；pruned proposals单列 |
| 准确率 | 30题同一答案抽取器，完整输出保留；失败/截断不从分母删掉 |

AR 的 acceptance 为 N/A。0 次验证或0个 tested proposal 的比例为 N/A，不填100%。acceptance、accepted length、output tokens/cycle 必须分别给出，不能混用论文术语。若存在token分歧，主表保留该标记，并增加固定AR轨迹的teacher-forced性能对照；后者是诊断，不能冒充自然生成E2E。

计划新增 `run_aime25.py`、`summarize_aime25.py`；运行时必须断言每个method/repeat恰好覆盖30个唯一ID。失败可同SHA同配置重试，但保留失败日志和重试次数。全部30题有终止状态与完整输出记录才算跑完，截断比例另报；不把仍在后台运行视为完成。

## 逐 token 和逐循环 Profile

主计时结束后单独采集 profile。先对所有30题收集轻量循环级计数/时间；按冻结规则覆盖所有题目的早/中/晚 decode 窗口，另对低/中/高接受率以及短/中/长上下文各选代表题做详细 kernel trace。选择依据、覆盖位置和未采样部分都写入 manifest；不通过对某个快题全程重型 profile 外推全套实验。

每条事件记录 `rid, request_epoch, algorithm, cycle_id, phase, round, depth, core_layer, position, actual/padded_rows, verify_width, accepted, emitted, prefix_len, graph_id`。CPU用单调时钟；GPU用stream CUDA events与profiler关联，结束后统一同步，禁止每层 `.synchronize()` 改变真实调度。

| 算法 | 报告的基本单位 | 必需拆分 |
| --- | --- | --- |
| AR | 一个完整R4 decode token；另列prefill首token | prepare、4次core、readout/sample、result/commit |
| DFlash | 一次真实 draft+target verify+commit cycle | draft prepare/feature cache、draft forward、target prepare/verify、accept/bonus、commit/rollback |
| DAS-Wave | 一次完整J轮refine+verify+commit | init、各j/k/width、deep-prefix map、draft attention/core/readout、independent target R4、accept、commit/rollback |
| LoopSpec | 一个共享core step，以及相邻commit间隔 | endpoint depth1、endpoint depth3、target depth4分开；无commit步、warmup/排空单列 |

LoopSpec是流水线，不能硬套“一个完整串行 draft+verify block”。提供三张互补表：

1. 配置为 depths=[1,3] 时，各endpoint的readout/decision时间、generated/tested/accepted计数；shared core不是任何单一endpoint专属，不能重复分摊。
2. 每token从普通初始化到depth1的1轮，以及depth1到target depth4的3轮：分别报告该token经历的wall span和loop数。由于有跨token并行，累计token span不等于总wall。
3. commit前继承的 `entry_depth=0/1/2/3` 分组：已做循环数、剩余循环数、经历shared steps、下一commit间隔、该组样本数。depth=3表示复用前三轮只剩一轮，不是独立draft-depth=3配置。

每个单位输出 N、mean/median/P90/P95、wall ms、GPU stream ms、exclusive GPU busy ms、kernel count、输出tokens与amortized ms/token。DAS/DFlash的ms/cycle除以该cycle实际输出tokens，另给总时间/总tokens；不能用 mean(time)/mean(accepted) 忽略fallback/bonus和尾部。

阶段结构采用两层，避免把 forward 和 attention 相加重复计时：

- 顶层互斥：prepare/metadata、draft forward、target forward、decision、commit/rollback/KV、result/transport、未归因。
- forward内部：attention kernels、QKV/O projections、MLP、norm/RoPE、LM head、其他模型算子；分别给占本forward和整个cycle百分比。

CPU线程耗时、GPU busy、GPU stream区间和wall elapsed各自展示。需要合计100%的wall分区时，明确采用“GPU执行优先，CPU只计未与GPU重叠部分，余量保留未归因”的占用规则；嵌套span扣除子span，重叠kernel取区间并集。该表描述时间占用，不声称是可回收的因果收益。不同资源的原始时长不能直接相加。

CUDA graph下通过launch/correlation/graph-node关联把kernel映射回phase；对应不上保留unknown，不靠kernel名称把全部GEMM误归模型forward（decision也可能含矩阵操作）。attention识别同时核对native backend符号和代表trace。每轮给graph hit、padding、clock alignment误差和插桩开销；profile与无profile输出一致性也要核对。旧 `partition_profile.py` 仅覆盖d1=1并有该断言，须扩展独立分析器，不能直接复用它生成depth1/3或DFlash结果。

计划新增 `profile_aime25.py`、`analyze_cycles.py`。最终交付逐题CSV、逐cycle/逐depth表、阶段堆叠图及kernel明细，所有数值能追溯到SHA、配置、prompt IDs、trace与event ID。

## 执行命令与记录

以下为待实现脚本的预期接口，实施时以 `--help` 和最终 manifest 冻结实际命令：

```bash
# 本地，每个候选提交前
python3 -m unittest discover -s research/loopspec/experiments/2026-11-05-das-wave -p 'test_*.py'
git diff --check
git push origin loopspec-test

# X5，在本对话持久 SSH 会话中顺序执行
git -C /home/ruoxi/repo/sglang pull --ff-only origin loopspec-test
git -C /home/ruoxi/repo/sglang rev-parse HEAD
nvidia-smi
ss -ltn
# 新隔离容器映射当前源码、target、已导出draft、只读dataset及本任务/results。
# 容器内先检查 /opt/sglang/bin/python、sglang.__file__、torch/CUDA/transformers。

/opt/sglang/bin/python research/loopspec/experiments/2026-11-05-das-wave/verify_das_wave.py --stage bs1 --output /results/round-N/bs1
/opt/sglang/bin/python research/loopspec/experiments/2026-11-05-das-wave/verify_das_wave.py --stage batched --batch-sizes 2 4 8 16 --output /results/round-N/batched
/opt/sglang/bin/python research/loopspec/experiments/2026-11-05-das-wave/run_aime25.py --manifest /results/frozen-manifest.json --expect-problems 30 --batch-size 1 --repeats 3 --output /results/aime25
/opt/sglang/bin/python research/loopspec/experiments/2026-11-05-das-wave/profile_aime25.py --manifest /results/frozen-manifest.json --performance /results/aime25 --output /results/profile
```

不指定目前尚未核定的GPU编号、端口或镜像tag。旧容器镜像ID已查到，但启动新容器前必须核对依赖和挂载，不能改旧任务容器。大型日志/完整输出/trace保留 `/home/ruoxi/experiments/loopspec/2026-11-05-das-wave/`，轻量汇总和执行记录经本地Git进入loopspec-test。

里程碑：M1 bs=1契约与GPU验证 → M2多batch与graph/overlap → M3 DFlash导出/feature一致性 → M4冻结参数和30题配对计时 → M5详细Profile。任何前置验收失败都记录为失败/阻塞，独立研究可继续，依赖它的性能结论不能提前成立。
