# R14–R20：面向 batch、长度与设备资源的 VQ+ 复审

状态：20 轮优化额度已用完。R19、R20 均被完整模型 greedy 检查拒绝；默认分派已回退至 R13，最终回退及容量/输出预算验收全部完成。下列数值均来自 A800/SM80；没有跨 GPU 吞吐实测。本文是 design-review.md 的续篇，历史 R13 结果不被覆盖。

## 已确定的结论

R19 的资源驱动调度证明，固定线程几何和旧的分区选择在大 batch 下有明显成本。但单层输出/LSE误差检查通过，并不保证完整模型 greedy token 不变。R19 在 B32/N1024 的同一基准prompt的32个请求均从第20个输出 token 分歧；同一候选重复结果完全一致，R13 重启后 B32/B64/不齐长度的 A/B/A 对照全部一致。不能把这种候选当作已验收加速。

R20 B32/N1024的32个请求与R19完全相同，也从第20个token分歧；原lifecycle通过，wide gate失败，未执行失败候选的吞吐晋升测试。

R20 的 INT4→FP32 位构造对16种 nibble 都精确，却使 B16/N2048 延迟比 R19 增加4.63–4.78%。编译后的 I2F 减少并不等于总代价减少：两种索引实例合计 I2F52→4，FADD增加48，LOP3增加88；40寄存器、零stack/local不变。这是静态反汇编事实，不是硬件执行计数。

## 七轮消融

配对 CUDA13/int64、真实码本与快照、独立输出/LSE oracle；时间单位 μs。不同轮次的绝对数值只与其当轮控制比较。完整表和样本在 evidence/r14…r20。

| 轮次 | 改动 | 代表结果 | 决策/限制 |
|---|---|---|---|
| R14 | 压缩 grid，CTA 循环处理 split | B16/2K：R13 128.157；grid8 144.150 | 增至32B stack，拒绝 |
| R15 | 成对 BF16 舍入 | QK127.647 vs128.106；PV133.957 | QK收益<1%；PV产生16B stack，拒绝 |
| R16 | PV每线程16→8累加器 | 2K127.546 vs128.166；8K421.874 vs438.081 | 2K收益<1%；int32路径spill，未晋升 |
| R17 | QK query改为 shared 读，缩短寄存器存活 | 最好2K129.988 vs128.082 | shared读增加代价，拒绝 |
| R18 | 128/256线程与分区联合消融 | 2K118.869 vs128.184；1K/B32 121.495 vs238.348 | 仅证明映射/分区价值，固定split不是最终策略 |
| R19 | 资源驱动几何、GPU长度表 | 2K118.945 vs128.103；1K/B64 237.153 vs388.259 | 68实测形状、100边界/Graph、memcheck通过；服务greedy失败 |
| R20 | 精确INT4 FP32位构造 | 2K四loop124.747–124.893 vsR19 119.056–119.366 | 68形状、100边界/Graph、memcheck通过；性能回退；服务greedy同样失败 |

R19 查询实际 occupancy，并按 batch×heads、当前GPU长度、驻留容量选择 split；不在新策略中写死 B16 或108SM。2,949,210项合成容量/形状/长度不变量通过。它证明的是策略边界检查通过，**不是其他GPU速度或正确性已验证**。现有整个backend的encoder与旧reader仍有SM80限制；单独新kernel使用普通SM80+ CUDA不意味着整个服务已可移植。

## 服务层面的泛化差距

同机同GPU2/3、模型/码本/seed、TP2/R4，KV65536/context32768/maxrunning64、匹配Graph列表；128输出。各行3次无profiler测量中位数，完整batch/Graph由原evaluator检查。VQ+指已验收R13；B32/B64走其保留fallback。

| 输入 / batch | R13 token/s | BF16 token/s | R13相对BF16 |
|---|---:|---:|---:|
| 1K / B1 | 51.382 | 51.745 | -0.70% |
| 1K / B4 | 188.232 | 194.271 | -3.11% |
| 1K / B16 | 472.118 | 526.130 | -10.27% |
| 1K / B32 | 542.787 | 776.322 | -30.08% |
| 512 / B64 | 926.373 | 1445.852 | -35.93% |
| 8K / B1 | 38.970 | 31.622 | +23.24% |
| 8K / B4 | 103.787 | 106.162 | -2.24% |

2K/B16的maxrunning64最终匹配对照为回退版本372.003 vs BF16 419.880 token/s，仍低11.40%；冻结R13同flags为371.921。回退版本100边界/Graph、原lifecycle和wide greedy全部通过。额外较小容量配置和输出32/128/256测试均已完成。B64/8K只在kernel回放中测过，不声称该全模型服务负载能放入当前KV容量。

## 较小容量与输出预算的最终配对结果

KV32768/context16384/maxrunning16，1K输入，同GPU2/3、同模型/seed，三次无profiler中位数。两种方法的生命周期均通过；VQ还与原量化基线逐token相同。所有吞吐样本通过完整目标batch和Graph检查。该口径是完整服务中的decode吞吐，排除prefill和首decode，不是kernel-only速度，也不是含prefill的端到端吞吐。

| Batch | 输出token | 保留VQ+ token/s | BF16 token/s | 相对差距 |
|---:|---:|---:|---:|---:|
| 16 | 32 | 476.717 | 545.154 | -12.55% |
| 1 | 128 | 51.310 | 51.789 | -0.93% |
| 4 | 128 | 188.667 | 193.707 | -2.60% |
| 16 | 128 | 472.212 | 525.114 | -10.07% |
| 16 | 256 | 463.168 | 515.759 | -10.20% |

这说明固定输出预算的结果不能推广成所有长度的固定加速比。服务覆盖输入512/1K/2K/8K、batch1…64的指定组合；kernel覆盖更多尾部与不齐长度直到32768。没有声称遍历所有组合，也没有声称验证了其他GPU。

最终runtime提交`dedf55fff`仅关闭adaptive默认开关，payload和encoder与R13字节一致。新服务通过100边界/Graph、原lifecycle、B32/B64/不齐长度greedy；CUPTI每rank1920次payload、0次adaptive调用。所有测试服务停止，GPU compute process列表为空，30664端口释放，持久SSH保留。未晋升正式loopspec分支。

机器可读结果：[continuation-results.json](../continuation-results.json)；证据包哈希：[continuation-manifest.json](../continuation-manifest.json)。原始证据保留于本任务evidence/和BUS2的`/dev/shm/wangruoxi-vqrq-20261127`；大文件不进Git。失败候选没有通过全服务晋升，20轮已用尽；B16差距和跨设备泛化目标均未完成。

## 同步骤瓶颈与设计取舍

- **压缩表示读取**：当前K+V顺序有效载荷约208 B/token/head/loop，BF16为512 B；该推导未计码本gather、sector、HP和复用，不是实测HBM流量。减少容量不能直接换算吞吐。
- **重建与QK**：pre-RoPE K需要逐历史位置码本gather、INT4展开、scale、FP32重建、RoPE和BF16舍入；BF16历史KV不承担这些解码工作。RQ官方保存post-RoPE key，不能忽略该合同区别。其消费单元共享元数据和驻留率驱动调度值得借鉴；直接照搬其成对低精度算术不符合本任务舍入点。
- **PV**：R13的pv16已经摊薄元数据并保留FP32累加。更少累加器或paired转换不必更快；本轮R15/R16给出实际spill或重复工作代价。
- **并行映射**：R18/R19是最强正面证据，尤其B32/B64。但同时改变split和归约分组，完整模型结果变化；不能用单层容差替代greedy验收。缺少logit级溯源，目前“求和重关联导致分歧”仍是解释假设。
- **写缓存与其他步骤**：历史R13/BF16同阶段CUPTI reader均值约127.6/103.9μs，原VQ+约184.2μs；编码在该服务trace占比很小。R19诊断trace均值117.854μs，确实降低reader活动时间，但它已被正确性拒绝，profile吞吐也不是验收吞吐。

## 下一项优先实验（仅计划，未实施）

先做**逻辑归约顺序与物理线程调度解耦**，而非继续改INT4转换或移码本到shared。保留R13及其fallback的逻辑分区、每个累加器的token顺序、warp归约树和最终split合并次序；用不同物理warp数量处理相同逻辑工作。例如256线程依次承担512线程的逻辑warp组，保留分组结果后按既定顺序合并。是否能减小寄存器/调度成本未知，不能承诺更快。

拟修改 residual_attention_adaptive_cuda.py::source_for/prepare 与实验映射，保留 residual_attention_payload_cuda.py 作为冻结控制；ResidualAttention.forward 仅在全部验收后启用。几何选择依实际寄存器/共享内存/SM资源与batch、head、length，而非GPU名称或单个batch查表；必须把改变**物理执行**与改变**浮点运算顺序**区分开。若旧数学顺序无法映射为更有效执行，应报告该合同的性能代价，而非暗改精度。

第一个最小实验是B16/N2048/2080和B32/N1024的逻辑顺序保真映射；先逐层检查output/LSE位模式相对于冻结控制（比既有误差门槛更强的诊断），再跑既有独立oracle、100边界/动态Graph/memcheck、全模型lifecycle与wide_greedy。内核过关后才做同GPU、相同服务flags、无profiler全服务计时；B1…64、短/长/不齐长度、两档KV容量和输出预算必须匹配。仍要求≥5%B16服务提升且任何已测场景回退≤5%，失败即拒绝。新GPU需独立编译资源和相同正确性/服务验证；不可用A800收益推断Hopper/Blackwell收益。

不需要等NCU才能开展上述针对性消融。有效计数可进一步区分gather延迟、转换吞吐与访存sector，但三台BUS容器目前均ERR_NVGPUCTRPERM，不能虚构带宽或stall结论。20轮已耗尽，此后代码实验需要新增授权；当前只完成已有候选的验收、证据与安全回退。
