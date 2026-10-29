# LoopSpec endpoint 批量决策修复计划

## 身份与预算

- 用户请求：制定逐 endpoint 草拟／验证决策批量化修复计划，沿用前次 Profile 规则，最终输出与 AR 的新对比。
- 状态：生产候选及基础正确性验证完成，配对实验与归因工具进行中。
- 当前基线：`7df281a0e656e368c90f529d8d84305db99001f5`，分支 `loopspec-test`，制定计划前工作树干净。
- 前次测量 SHA：`da56f8103931e5b3a2451219b4e0c4b3dcb16286`；之后仅分析和结果记录改变。实施前重新确认生产源码 diff。
- 官方参考 SHA：`41aa710ed7c9ce908cffa03597be5d1df19d7b56`，只读。
- 新修复任务预算：7/10（用户随后明确将本任务上限提高到10轮；计数不重置）（按本对话用户提供的 AGENTS.md 五轮上限；磁盘技能仍写十轮，用户要求优先）。前次 benchmark 任务保持 6/10，不重置历史计数。每个源码／测试／行为配置候选的修改＋验证记一轮。
- 正式 `loopspec` 分支未获本次晋升授权。

## 1. 修改目的与证据

现有 worker 已将 core/readout 合批，但 `_advance_rows` 在 Python 循环中逐 endpoint 调用 `RequestContext.queue_decision`。greedy 仍构造整词表 one-hot，执行 multinomial 和通用 residual/CDF 验证；无惩罚时仍计算历史 bincount。所有 endpoint 最后集中 D2H，随后 CPU collect，形成下一次分支推进的依赖。

bs16 同时发送的 profile：决策 CPU 累计 10.897s，profile E2E 36.239s，无 profiler E2E 24.820s。决策 CPU 区间会与 GPU 执行重叠，profiler 对小操作有明显放大；不能直接减去 10.897s 预测收益。上一轮 18.85–21.09s 等数字仅为情景推算，不作为事实、验收目标或修改数值阈值的理由。

本次假设：批量 endpoint 数学计算、greedy 特化及减少不必要的 tensor/RNG 操作，可缩短每步暴露的决策尾部，尤其改善 bs8/16。通过独立旧 LoopSpec、候选 LoopSpec、AR 对照验证。

## 2. 修改步骤

### A. 固定输入与决策契约

- 冻结本次基线、前次 16 道题的 prompt IDs、模型 revision、环境和数值校准文件。先核对现有接口及 d1/d2 的 proposal/gating/verify 语义。
- 将 endpoint 收集为一个有明确行映射的批次：`(rid, lane_uid, depth, role)`；临时行号不能充当持久 owner。
- 返回批量 token、proposal enabled、accepted origin，保持 `Pipeline.plan/collect` 的语义和请求顺序。CPU 可遍历元数据，但主路径不再在循环内逐 endpoint 执行词表级 GPU 运算。

### B. 批量 greedy 决策

- 对 `[endpoint_count, vocab_size]` logits 一次完成行级处理和 argmax。先正确应用各请求的 logit_bias、惩罚、min_new_tokens/stop mask，再选 token；保持 tie-breaking。
- 无 frequency/presence/repetition penalty 时跳过历史计数，而非计算后乘零。非默认参数采用批量索引／mask，不静默忽略。
- proposal 保存 token 和必要元数据；greedy 不保存词表 one-hot、不调用 multinomial、不计算 residual CDF 或消耗无用随机数。
- d1=1 的 target 验证按 draft/target token 比较返回 origin；兼容现有两个 draft depth 的 gating 和优先接受顺序，不改变第二候选算法。

### C. 通用 sampling 的合批与兼容性

- greedy 与 sampling 按模式和 endpoint 角色分组，兼容混合参数请求；禁止为了合批要求所有请求使用相同参数或新增公开开关。
- sampling 对实际过滤／归一化后的 proposal 分布做批量保存、验证和残差采样，复用本项目 `verify_tensor` 已有的 batch 维能力；保留零质量、第二候选 gating 和支持集语义。
- 随机状态归 request 所有；允许少量逐 owner 随机数准备，重的概率／残差计算必须合批。不得把所有请求合成一个共享 seed 流；不要求改变随机数消费方式后同 seed 必须逐 token 一致。
- 不声称 sampling 已优化，除非它的路径和分布验证均完成。若仅 greedy 达标，明确记录部分完成，不能以 fallback 代替全部计划验收。

### D. 生命周期、CUDA graph 与 V2 接入

- 优先扩展现有 worker/utils/sampling，必要时使用小型 endpoint metadata 结构，避免建立第二套 scheduler 或完整 worker。
- core 和 finish graph 继续复用原生 recurrent executor。决策采用按 endpoint capacity 捕获的批量图或少量融合操作；预分配稳定 buffer、mask padding 行、在图外提供 RNG。
- 不逐 endpoint replay 旧 `RejectionGraph`。若批量图增加管理成本且无收益，可选择经测量验证的批量 eager／融合实现，并明确实际路径。
- 持久 proposal/logits 不能引用下次 replay 会覆盖的输出；按请求／lane 保持有效生命周期，避免一个小 view 长期保留已无效的整个大 batch buffer。
- 每步最多保留一次聚合决策回读；不在 endpoint 循环引入 item/tolist/cpu。若使用 pinned buffer 和 event，必须明确 wait 的最晚合法位置；不能在 CPU 真正需要结果前伪称已消除依赖。
- 保持 CDB、原生 overlap 和 V2 结果契约，不复用 Eagle/DFlash 代码。

## 3. 文件范围与非目标

生产文件（相对于 SGLang 根目录）：

- `python/sglang/srt/speculative/loopspec/worker.py`：收集／批量执行／回填 endpoint 决策。
- `python/sglang/srt/speculative/loopspec/utils.py`：请求采样元数据、proposal 与决策所有权。
- `python/sglang/srt/speculative/loopspec/sampling.py`：批量 logits 处理、greedy／sampling 决策、批量 graph buffer。
- `python/sglang/srt/speculative/loopspec/info.py`：仅在确需轻量 endpoint 元数据契约时扩展；原生 relay 语义不变。
- `pipeline.py` 仅在 proposal 表示适配必需时最小修改，不改推进／接受算法。
- 本任务目录：决策等价性／统计测试、paired matrix driver、分析与报告；复用前次 workload、numerical probe 和 profile analyzer。
- Profile 如需新增 span，仅用于 decision pack/compute/readback；旧阶段名称、覆盖和 shared ID 保持可比，禁用时无热路径插桩开销。

不改 core、Ouro attention、普通 CDB admission、KV prefix 管理策略、R、draft depth 或公共 API。CLI 仍仅使用 `--speculative-algorithm loopspec --speculative-draft-depth 1` 及原有 SGLang 参数。attention／prefix map 优化是独立任务，避免混淆归因。

## 4. 成功标准

- 逻辑：固定同一 logits/history/params 的 greedy 决策与独立标量参考逐 token、enabled、origin 完全一致；d1/d2、全接受／拒绝／第二候选恢复、EOS 等边界均覆盖。不能用模型数值容差掩盖离散分支决策错误。
- sampling：验证接受与 residual 分布及统计一致性；测试样本量与多重检验阈值在运行前固定。按相同外部随机数核对批量 verify 与逐行参考，同时不以自由采样相同 seed 序列作为唯一标准。
- ownership：跨请求无串扰、图输出不被覆写、拒绝分支不污染 target、finish/abort 后无泄漏或 double free。
- 模型：沿用独立 full-R 原生校准范围；仍报告 AR／新旧 LoopSpec 的完整序列匹配和首差异。前次 52/310 完全一致不是新测试的通过阈值，也不是严格无损证明。发现新差异先检查同前缀决策，再检查调度改变造成的计算形状；不得自行放宽误差范围。
- 结构：常见 greedy 主路径词表级运算不随 endpoint 数变成逐行独立提交；无逐 endpoint 回读；可观测决策 kernel/launch 数按批或固定角色分组，而非按请求线性重复。
- 性能：相对于同期旧 LoopSpec，各 bs／发送方式整批 E2E 中位数回归不超过 2%；bs8/16 应观察到超过重复间波动的改善，才能称本次修复有效。TTFT、单请求延迟和显存如有代价必须单列。只减少 profiler 时间而无无 profiler E2E 收益不算性能达标。
- 5 次重复不能区分噪声时，允许预先约定追加一组 5 次相反顺序配对；合并全部结果，仍不明确则结论为不确定，不选择性保留更快结果。
- 真实测量未达标时保留失败证据，在剩余轮次内调整；到 10/10 停止修改。不得将情景预估当成加速目标强行解释结果。

## 5. 风险

- 不同请求惩罚／stop mask、endpoint 角色／depth、d2 enabled 混合造成行对应错误。
- argmax 并列处理或采样过滤顺序变化导致算法语义偏移。
- RNG 与 graph padding 消耗随机数，或请求间共享 RNG，影响统计正确性。
- graph buffer 被复用时污染保存的 proposal/logits；整块 tensor view 导致额外显存滞留。
- 消除无用 greedy 操作可能改变主机时序和请求混合方式，从而触发既有 BF16 batch 形状差异。
- profiler 对旧路径的逐 endpoint 打点比新批量 span 更昂贵，不能仅凭前后 profile 差额宣称性能收益；需要无 profiler E2E 和低开销聚合核对。
- bs16 包含更长题目，不是纯粹固定长度的 batch 扩展实验；保持同题比较，不泛化成所有 workload 的硬件饱和结论。

## 6. 验证计划与命令框架

### 环境及 Git

本地只在 loopspec-test 编辑／提交／push；X5 复用已有长连接，pull 后逐一核对 SHA、sglang.__file__、容器映射、GPU UUID/占用、端口、依赖。源代码不通过 SCP/rsync 同步，不在 X5 修源码。

基线对照使用已同步 Git 的冻结版本，在独立只读副本／容器中运行；不是另建开发分支，不切换或覆盖活动 loopspec-test。只有已确认本任务拥有的进程可停止，同一 GPU 不并跑 serving 对照。

保留前次环境：RTX5090 单卡，SGLang v0.5.22-cu130 隔离镜像及记录的依赖，BF16/Triton，R=4，CUDA graph、原生 overlap、LoopSpec V2。实施前核对 image digest，不静默升级依赖。

本地已有检查示例（实施后运行）：

```bash
python -m compileall -q python/sglang/srt/speculative/loopspec
python -m pytest research/loopspec/experiments/2026-10-29-cdb-v2/test_loopspec_cdb.py
git diff --check
```

本任务将新增 `test_decisions.py`、`benchmark_decisions.py`、`paired_matrix.py`、`report.py`；尚未实现，以下是计划接口，不是已执行命令：

```bash
python -m pytest research/loopspec/experiments/2026-10-30-batched-decisions/test_decisions.py
python research/loopspec/experiments/2026-10-30-batched-decisions/benchmark_decisions.py --output /results/decision-bench
python research/loopspec/experiments/2026-10-30-batched-decisions/paired_matrix.py --phase performance --output /results/performance
python research/loopspec/experiments/2026-10-30-batched-decisions/paired_matrix.py --phase profile --output /results/profile
```

准确的 baseline/candidate 启动命令、只读快照映射及容器参数在第一轮写入记录后执行；不得仅根据上述接口宣称运行通过。

### 验证顺序

1. 决策单元／小词表解析与统计测试；相同 endpoint inputs、外部 uniforms 对照逐行与批量版本。覆盖非默认 penalties、bias、top-k/p、min-p、temperature、min-new-tokens、stop、并列 logits，以及 d1 和双候选路径。
2. GPU 决策微基准：使用固定 checkpoint endpoint logits 与实际角色/数量分布，比较旧与新；分别测同步端到端、CPU 提交、GPU kernel、D2H wait，不能只测异步 launch。graph/eager、变化 batch 和 padding 均覆盖。输入采集在计时外，数据不进入 Git。
3. full-R AR 检查与实际 LoopSpec teacher-forced 数值／KV 诊断；复用前次独立参考、固定 envelope，覆盖全部 bs 和混合深度。新增 greedy 离散决策必须严格等价。
4. HTTP 严格内存 smoke：bs1/16、两种到达方式、短输出、EOS、取消／中途加入、连续两轮请求，以及 greedy/sampling 混合；检查生命周期而非只看 HTTP 200。
5. 正确性通过后，执行完整无 profiler 性能矩阵与独立精细 Profile；每轮候选先做必要小范围回归，最终候选才跑完整矩阵。若候选改变结果则重跑其受影响验证，不复用旧提交的测试结论。

### 保持前次 workload

- 模型 `/home/ruoxi/models/Ouro-2.6B-Thinking`，revision `f1edd81e7ac41355db670500ceaf204e0f73af68`，原文件不变。
- AIME `/home/ruoxi/repo/test/datasets/aime.jsonl` 的前 bs 道题；bs=1/2/4/8/16。
- 原 checkpoint template：enable_thinking=True、add_generation_prompt=True；复用保存的 prompt IDs 并校验 hash。tokenization 在计时外，/generate 传 input_ids。
- greedy，max_new_tokens=512，保留自然 EOS；输出长度变化必须标注，不能强行忽略 EOS 来追求速度。仍是固定上限生成测试，不宣称完整解题耗时。
- 同时发送：bs 个独立请求，共同计划时刻；交错发送：第 i 个请求延迟 100*i ms。保留实际发送偏差。
- context2048、max-running16、max-total-tokens12000、mem-fraction.78、decode graph max16、page1、radix cache关闭、stream interval1、enable metrics；这些公共参数对照保持相同。
- 性能：每组预热2次、测量5次；每轮对 AR、旧 LoopSpec、新 LoopSpec 轮换顺序配对，同一 GPU 串行，启动/graph capture不计时。不能只把本次候选与昨天 AR 数字比较。
- Profile：每组另预热2次，采集1次完整 cohort；AR、新 LoopSpec 各10组，旧 LoopSpec 同期10组作为归因对照。现有历史 profile 保留，不被覆盖。Profile 和性能完全分开。
- `SGLANG_PROFILE_V2=0` 是前次使用的 profiler 实现选择，不代表禁用 LoopSpec V2 worker；分别记录 profiler 配置和 worker/overlap 路由，不混称。

### 精细 Profile 与最终报告

沿用前次每个请求、每个 output_index、每种 entry depth 的路径关联规则。d1=1 预期为 depth0/3，任何新增路径不得丢弃；prefill 首 token 单列。保存共享 step IDs，禁止将同一批工作重复计入全局总成本。

保持以下阶段：prepare_rows、core、readout、decision、commit、retain、plan/collect、scatter/readback、native result handler、客户端 ITL。新增 batch decision 的 pack/compute/readback 子阶段；逻辑 endpoint 数与实际批量调用次数分别报告。GPU kernel busy union、CPU 墙钟、CUDA stream 区间、回读等待分开，不能相加为 E2E。旧版不能可靠细分的部分标未归因，不逆推独占比例。

补充小规模低开销聚合计时，评估旧逐 endpoint span 与新 batch span 的插桩不对称；性能结论始终来自无 profiler 测量，不以 profile 节约替代。

最终交付：

- 两种发送方式各一张 bs1/2/4/8/16 表：AR、旧 LoopSpec、新 LoopSpec 的整批 E2E，单请求中位/P95，TTFT、decode、吞吐、输出长度与停止原因；新/旧改善、相对 AR 加速比、重复间波动。
- 新 LoopSpec 每个 bs 的 depth0/3 CPU/GPU 阶段表、路径数量、ITL，中间阶段与旧版本可比；prefill 单列。
- 决策微基准及结构证据：endpoint 数、批量调用／kernel launch 数、回读等待、graph命中与峰值显存；是否真正缩短关键路径。
- 完整序列差异、同前缀数值结果、sampling 统计与 KV 生命周期结果；未达标项不隐藏。
- 最终 SHA、环境、命令、退出码和原始数据索引，明确性能是否达到标准，不用之前情景推算代替实测。

X5 原始结果根目录：`/home/ruoxi/repo/test/results/loopspec/2026-10-30-batched-decisions/round-N/`。轻量报告与 CSV/JSON 保存本任务目录并通过 loopspec-test 同步，大型 trace/logits 留在 X5。

## 当前执行记录

2026-10-30：已只读核对实现、现有实验工具与任务规则，制定以上计划。已用 0/10；所有实现与验证均未执行。本轮只创建此计划记录，未 commit/push 或变更运行环境。

### 第 1 轮 / 10（开始）

用户批准计划后开始。候选范围为 sampling 批量 logits/greedy/残差运算、worker endpoint 合批，以及本任务决策测试／实验驱动。先采用批量 eager 操作，保持 core/readout graph；根据独立微基准决定是否需要额外决策图。对请求所有权、有效 proposal 生命周期和已有单／双候选语义做严格检查。

已复用 SSH session60877；X5 HEAD=7df281a0e，工作树干净，GPU0/1 无计算进程；现有隔离容器 image sha256:81726dc9e63a9c6778396c8176a7203716a527f28b2bbb6b0eac661a0b050afb。磁盘剩余约104GB，完整 trace 前再次核查。Mac 默认 Python 无 torch：本地做语法／diff检查，CPU和CUDA测试均在同镜像 X5 隔离环境执行，不能把未执行测试记为通过。

### 用户补充的归因交付要求

最终报告需要量化 LoopSpec/AR 加速比从 bs1 随 batch 增大下降的各阶段百分比。以明确的归一化损失定义进行分析，GPU kernel/CPU区间去重、重叠只计一次，保留未归因项和 profiler 扰动。决策修复的因果证据来自旧/新配对实测；attention 等未独立干预的阶段仅作时间分解/反事实估算，不冒称精确因果比例。报告每个 bs、两种到达方式，并区分 batch 增长与题目长度分布变化。

第一轮当前证据：本地 compileall/diff-check exit0；local/origin/X5 c1c895cc5430cc460bdd958fdbcc852d7a82121a。28项 CPU/CUDA 测试通过（10.52s，exit0）。同前缀1024位置实际graph分支计算通过固定native-reference-r4 envelope（exit0）。首次数值命令误用了自由生成AR B16输出作强制prefix，和参考在q2/7/8/10不同，导致exit1；保留numerical失败日志。改用参考自带cohort-0 tokens，源码不变，numerical-matched-prefix通过。这是纠正实验输入，不是放宽阈值。严格HTTP smoke（bs1/16、两种发送方式、64 token）exit0。

### 第 2 轮 / 10（实验工具候选）

新增 benchmark_decisions.py 和 paired_matrix.py，生产代码保持第一轮不变。旧版只读 detached baseline为7df281a0e，候选和baseline挂载路径分别核实。为保留每组2次预热+5次连续测量，按bs轮换AR/旧/新算法块顺序，在同一server内测两种到达方式；这是配对算法块，不声称逐重复ABBA。每项数据保存实际SHA。

独立微基准使用真实checkpoint target logits并构造可控endpoint role/proposal元数据，明确不是现场draft logits回放。测试单位是整个决策packet，包括同步回读；分别报告CPU提交、回读等待、GPU stream和kernel数。性能和profile矩阵分别运行。不会在第一轮正在运行的服务期间pull改变其源码。

第二轮：59641366ae5cb9c971d01969340d60f266852a25 已同步 X5。微基准 exit0；38项混合参数、EOS、取消、零token及KV复用HTTP检查 exit0。完整bs1/2/4/8同前缀数值检查继续运行，bs16已在第一轮通过。

### 第 3 轮 / 5（诊断容量修正，分析方案）

在既有计划内新增 profile 分区和报告工具，不改生产候选。使用与前次相同的共享step路径分析，另对Chrome GPU/CPU区间去重，GPU执行优先，主机只记未被GPU覆盖的区间。此为占用时间分区而非关键路径因果模型；把Profile份额映射到无Profiler E2E的百分比明确标为估算，并报告扰动倍数。

加速比损失使用 E_B=L_B-(L_1/A_1)A_B（保持bs1加速比应减少的LoopSpec时间），各阶段的有符号贡献之和为E_B；允许负贡献，保留未归因项。另报告旧/新实测的E_B下降，避免把旧版独有Profiler开销归因成真实运行收益。同步检查prompt IDs、全序列及停止原因差异、全部Profile覆盖；不改变既定数值/性能标准。

第二轮扩展numeric-b1在构图阶段exit1：旧诊断脚本固定捕获到32行，超过bs1×width4容量；未进入数值比较。第三轮仅在verify_loopspec.py添加与生产init_graphs相同的capacity筛选，不改生产代码或误差标准。失败日志保留round-2/numeric-b1.log。分析工具的方案已记录，尚未修改工具源码。

第三轮：1b941f6a92096001fa7da9478930f1706f33fdb3，本地语法/diff检查exit0，X5构图容量错误已消除。bs1/2/4的hidden/logits幅度均在固定范围内，但argmax翻转率分别2/64、2/128、3/256，超过固定1%；bs8通过。冻结旧版bs1的所有数值记录和新版本完全相同，旧版bs2同样2/128；bs4旧版仍在检查。保留失败，不修改阈值，不声称所有bs数值标准通过。此任务的合批决策严格离散等价测试已通过；性能比较只能在注明已有数值问题且核实未引入回归的前提下解释。

### 第 4 轮 / 5（配对实验分析工具）

新增partition_profile.py、report.py，执行前述互斥时间分区及带符号百分比方案，复用旧token-path分析器。不改生产或负载。先检查区间重叠去重和真实trace时钟对齐，再生成全部报告。完成剩余bs2/4旧版固定前缀对照后运行原计划完整性能及Profile矩阵；各组数据记录实际提交。

第四轮本地compile/diff及互斥区间验证exit0。真实历史trace检验exit1：Chrome同时含CPU user_annotation和同名GPU user_annotation，按名字选forward导致1153对641，时钟校验正确拒绝了该输入。未产生错误比例。

### 第 5 轮 / 5（分析标签过滤修正，最终候选）

partition_profile.py的forward时钟锚点仅选CPU user_annotation，与原生host span对应；GPU annotation不作重复锚点。其余代码不变。bs1/2/4旧版与新版全部record完全一致（64/128/256位置，含hidden/logits误差和argmax）；确认1%翻转率失败为本次修复前既有。bs8/16通过固定envelope。第5轮不改变数值标准，全部最终性能/Profile验证继续执行；如后续工具或实现需再次修改，按用户五轮上限停止并报告。

第五轮GPU矩阵继续运行，测试SHA固定16ed789a493981b61f63c5d9720564b66b4e8369，不在服务运行时pull。已完成bs1/2/4配对E2E，均改善。实测bs2同时发送的旧版加速比比bs1更高（AR17.584s，旧7.678s），E_B为负。只读审查发现report.py把None恢复比例按浮点格式输出将报错；保留“无下降”的真实结果，不能假定每档都单调恶化。

### 第 6 轮 / 6（用户批准追加1轮）

用户回复“追加”，授权仅修报告边界并验证。report.py对未定义恢复比例输出“无下降”，说明E_B负值的含义；不改生产代码、原始数据、负载或标准。GPU实验继续使用已冻结SHA；本地提交的分析修正待GPU采集结束后pull。最终完整分析将验证该分支与正常下降分支都可生成报告。

用户随后授权最高10轮。第6轮本地compile/diff通过，914df15bb已push，X5尚未pull以保持正在采集的16ed789a快照。

### 第 7 轮 / 10（归因估算的扰动敏感性）

只读审查发现：旧Profile的主机插桩扰动较大，按整段墙钟比例缩放会同时缩小实测GPU kernel时间。因此单一投影不足以给出稳健阶段百分比。目的：显式报告这项不确定性，并区分加速比本身的跌幅和“保持bs1加速比所需省下的时间”。步骤：时间轴分区保留GPU/CPU/未覆盖资源；除整体比例投影外增加GPU忙时固定、仅缩放CPU/空隙的敏感性口径；加入实际branch行工作量和按endpoint数核对的readout graph覆盖。范围仅partition_profile.py/report.py及记录。成功标准：资源时间总和等于Profile E2E，两种投影各阶段有符号贡献之和等于归一化损失；无下降/不可用情形明确标识。风险：第二口径仍假设Profile kernel时长和工作形状能代表无Profiler运行，二者差异不是置信区间，更不是精确因果比例。验证：本地重叠区间检查，采集后真实trace及全部报告检查；生产源码、GPU负载和验收阈值不变。
