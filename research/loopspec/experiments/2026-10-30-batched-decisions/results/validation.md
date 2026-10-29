# 决策合批：验证证据与边界

生产改动提交：`c1c895cc5430cc460bdd958fdbcc852d7a82121a`。旧版冻结于 `7df281a0e656e368c90f529d8d84305db99001f5`；完整性能和 Profile 的候选采集提交为 `16ed789a493981b61f63c5d9720564b66b4e8369`。后续提交只调整实验分析与报告，不改变生产候选。

## 逻辑与生命周期

- CPU/CUDA 28 项测试通过，包含固定 logits 下的 greedy 严格等价、混合请求参数、单/双候选、第二候选门控、每请求 RNG、保存的 proposal 分布所有权，以及 sampling 接受/残差统计。X5 `round-1/tests.log`，exit0。
- bs=1/16、同时/交错请求的严格内存 smoke 通过，64 token/request。X5 `round-1/smoke/`，exit0。
- 38 项 HTTP 生命周期检查通过：连续两轮16个混合请求、penalties/min-new-tokens/logprobs、取消及存活请求、EOS、零输出长度和 KV 回收复用。X5 `round-2/lifecycle/`，exit0。
- 旧的逐 endpoint 大词表运算已替换为批量操作；greedy 不再构造 one-hot 分布或调用随机采样。每个 packet 仍只有一次依赖性的主机回读。原 CDB、分支所有权、提交/回滚、core/readout CUDA graph 和公开接口保持原语义。

## 同前缀模型数值

保持此前独立 native full-R 计算校准的误差范围，没有因本次结果放宽阈值。使用参考自带的相同前缀，实际运行模型、分支和 CUDA graph；不是用自由生成的不同前缀比较 logits。

|bs|检查位置|argmax 翻转|原固定门槛|旧/新对照|
|---:|---:|---:|---|---|
|1|64|2|不通过：超过1%|全部逐位置数值记录完全相同|
|2|128|2|不通过：超过1%|全部逐位置数值记录完全相同|
|4|256|3|不通过：超过1%|全部逐位置数值记录完全相同|
|8|512|4|通过|本轮新候选检查|
|16|1024|5|通过|本轮新候选检查|

所有 bs 的 hidden/logits 幅度误差均在原校准范围内；小 bs 失败项仅为固定的 argmax 翻转比例。原失败保留，不能把“未引入数值回归”表述成“所有数值标准通过”或“自由生成严格无损”。详细误差、阈值及原始报告位置见 `numerical-summary.json`。自由生成完整序列和停止位置的对照另见最终 `token-comparison.csv`。

第一次 bs16 数值调用误用了自由生成 AR 的输出作为前缀，第二轮小 bs 诊断又遇到旧脚本捕获行数超过实际容量；这些失败日志均保留。前者通过改用正确参考前缀解决，后者修正诊断脚本的容量筛选，均未修改模型权重或误差阈值。

## 独立决策微基准

固定真实 checkpoint target logits，构造可控角色/候选元数据，每个 packet 为 `2×bs` 个 endpoint。这不是现场 draft logits 回放，也不代表 serving E2E 加速比。10 次预热、50 次计时，wall 包括主机回读；kernel 数另在单独的 profiler 调用中采集。

|bs|endpoint 数|旧 greedy wall ms|新 greedy wall ms|旧/新 kernel 数|
|---:|---:|---:|---:|---:|
|1|2|0.623|0.171|97 / 18|
|2|4|1.559|0.173|193 / 18|
|4|8|3.072|0.178|385 / 18|
|8|16|6.087|0.185|769 / 18|
|16|32|12.057|0.198|1537 / 18|

bs16 sampling packet 的 wall 从15.027降至1.477 ms，kernel 数1953→180。真实服务的 endpoint 数、角色、分支形状和等待依赖会变化，不能用该微基准倍数乘以整段 E2E。完整计时见 `decision-benchmark.json`。

## 性能口径

同一 X5 RTX 5090 GPU0、同一容器环境，Ouro-2.6B-Thinking，BF16，R=4，draft depth=1，普通 CDB/LoopSpec V2 与 overlap 开启。使用 checkpoint Thinking template、AIME 前bs题，最多512新token并保留自然EOS。两种方式为同时发送和每请求相隔100ms；每格2次预热、5次无Profiler测量。

公共CLI的graph max bs均为16，但SGLang原有cuda_graph_hook会为AR选择[1,2,4,8,12,16]，为spec模式选择[1,2,3,4,5,6,7,8,10,12,14,16]。旧新LoopSpec的完整图配置相同，本次没有修改这一默认策略。这里比较的是相同公开启动参数下的各算法，不能称为内部padding形状完全相同的消融实验。LoopSpec还将core捕获尺寸扩展到分支宽度4倍，readout仅在endpoint行数精确匹配时复用图。

配置中的`disable_overlap_schedule=False`指普通调度overlap；`SGLANG_PROFILE_V2=0`只选择Profiler实现，不关闭LoopSpec V2。原生V2接口与overlap开启也不意味着所有主机工作均可隐藏：分支提交仍依赖GPU决策的同步回读。服务启动时的权重/KV/图内存清单已记录在capture-validation.json；本次未连续采样服务过程的峰值显存，不能把启动清单作为峰值。

30格全部完成，exit0；10组新旧配对均满足 E2E≤旧版×1.02，且新版观测最大值均小于旧版观测最小值。完整测量的所有请求都达到512 token上限，因此结果不是完整AIME解题耗时或答题准确率评估。各档题目不同，尤其bs16加入648/861 token长输入，不能当作固定上下文的纯batch扩展。

性能单元前后的 Prometheus 差值另存 `event-counts.csv/json`：包含2次预热+5次正式运行，以前后差值扣除服务自己的启动预热。20个 LoopSpec 单元全部满足请求数=`7×bs`、完整验证数=`7×bs×511`、接受数+拒绝数=验证数，以及循环推进数=`4×验证数−3×已复用状态数`。这是depth0/3路径的计数一致性检查，不能替代数值测试。

bs16同时发送的旧/新接受率为88.725%/88.755%，每cohort共享core步数813.57/813.43；交错为88.969%/88.989%，步数861.00/865.86。接受率差异很小，主要性能改善不是由接受率提高造成。bs1新版本接受率91.781%，至bs16略有下降；不同题目和数值计算形状仍可能影响自由生成与分支路径。上述计数包含预热，不应表述为仅五次计时样本的统计。

Profile 在另一轮独立运行。CPU/GPU区间可以重叠，阶段可以嵌套，批工作可以被多个输出共享；不能把这些时间直接相加。阶段归因百分比是明确假设下的时间分解，保留负贡献和未归因部分，不声称 attention 等未单独干预步骤的精确因果比例。

原始数据根目录：`/home/ruoxi/repo/test/results/loopspec/2026-10-30-batched-decisions/`。
