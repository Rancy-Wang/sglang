# VQ+ attention 设计复审

状态：第13/20轮微基准与普通正确性验证完成，memcheck零错误通过，完整VQ+服务验收与同机BF16对照均完成。这里区分已测结果、源码事实和待验证解释；最终接受还需要同机完整 serving。

## 核心判断

现有表示确实减少了顺序 KV 有效载荷，但原 reader 付出了较多重复寻址、解码、寄存器占用和调度成本。优化应先让一个线程持有并复用完整的消费单元，再决定并行分区。把码本搬到 shared memory 本身没有保证：本次专门改过 bank 排列、加入异步复制后仍明显变慢。

本轮不改训练出的码本、pre-RoPE K、INT4/group32残差、四个loop、FP32累加或BF16舍入位置。因此以下结果是同一表示的执行优化，不依赖重新训练或降低精度。

## 与 ResidualQuant 对应步骤比较

| 步骤 | 当前 VQ+ 的额外工作 | ResidualQuant 源码中可借鉴的部分 | 本轮处理 |
|---|---|---|---|
| 写缓存 | 联合索引和逐loop残差编码；HP窗口迁移 | anchor/residual分开组织；压缩写入 | 不改编码，完整服务单独计入 |
| 准备查询 | staging；历史K路径还要做位置RoPE | 查询旋转独立prepare；occupancy驱动工作量 | 保留数值路径；读取实际occupancy |
| QK | 两组码本gather、INT4展开、scale、RoPE、BF16舍入、FP32点积 | 连续通道共享元数据；解码与消费紧邻 | pair8共享页地址与分支；精确重建FMA |
| softmax | split内共享score和块同步 | 每组流式统计后合并 | 原采样只约1%CTA阶段周期，暂不优先重写 |
| PV | 同一token多次加载索引/残差/scale | 每个消费块只读一次元数据；成对算术 | pv16一次读4个索引、16个残差和1个scale，FP32累加保留 |
| merge | 分区越多，partial/LSE与合并成本越高 | 根据实际内核驻留率调度 | 按长度选择分区；固定grid支持Graph |

RQ参考源码：[compute.cuh](https://github.com/heejunkim00/ResidualQuant/blob/f4c3cccb76cb6df37eb6233fa92c7aaa054d6695/plugins/residual_quantization/csrc/attention/compute.cuh)、[launch.cuh](https://github.com/heejunkim00/ResidualQuant/blob/f4c3cccb76cb6df37eb6233fa92c7aaa054d6695/plugins/residual_quantization/csrc/attention/launch.cuh)。这些是源码观察，未在本机移植并实测RQ，因此没有声称新reader优于RQ。

RQ官方Ouro在进入attention_slot前已经对query/key调用rotary_emb（plugins/looped_transformers/src/looped_transformers/ouro.py:109）；其packed_backend随后保存这些key。当前VQ+则保存pre-RoPE表示。这是QK步骤成本的实质区别，不能把两个reader的全部差额都归于压缩解码效率。

## 压缩率不能直接当作访存加速比

按D128、group32、RQ位宽[2,2,2,4]计算：RQ非anchor的K+V读取需要anchor和当前残差，共228B；anchorloop144B，四loop平均207B/token/head/loop。当前VQ+每loop顺序载荷为2×(32B索引+64B残差+8Bscale)=208B。该比较尚未包括码本gather、RoPE、HP、cache sector浪费和复用；它是源码推导的有效载荷，**不是NCU实测流量**。持久容量则分别约100B与160B/loop，表示容量与每次读取量必须分开讨论。

## 码本代数改写为何受限

令K_t=BF16(R_t(C[index_t]+s_t*r_t))。在当前数值合同下，Q·C的表不能跨位置直接共享，因为R_t随位置变化；把R_t移到Q后，也不能忽略BF16舍入误差。V侧先按index聚合概率再乘码本，在实数中可以分解，但当前BF16(C+s*r)不等于C+BF16(s*r)，要维持同一结果还需处理逐token舍入修正。

因此本轮选择直接在寄存器中解码并消费，重点消除重复元数据操作，而没有凭借无效的线性分解声称能消除全部逐token计算。跨请求共享码本也不意味着可以把不同请求的历史K合成同一个标准GEMM；每个请求的索引和位置不同。

## 已测消融与保留决策

A800-SXM4-80GB，B16/N2048/localH8/D128，真实训练码本与快照，CUDA Graph配对测量；各行来自各自配对运行，详细样本见record.md与evidence。

| 候选 | 延迟μs | 决策 |
|---|---:|---|
| 原page-head reader | 222.09 | 基线 |
| 仅分区调度 | 178.37 | 证明调度有影响 |
| PV元数据合并 | 144.00 | 有效 |
| QK/PV合并＋限制寄存器 | 124.12 | 有8B真实spill，不晋升 |
| 再加精确重建FMA | 123.82 | 40寄存器，零spill；保留 |
| QK点积重排 | 123.25 | 同次对照只改善约0.4%，不采用 |
| bank-aware shared码本 | 181.94 | 搬运/驻留率代价过大，不采用 |
| 第12轮动态调度集成 | 133.56 | 正确性通过；重复调度计算有额外成本 |
| 第13轮常量表调度集成 | 125.12 | 同次原实现221.40；真实矩阵与边界通过 |

原reader的匹配CUDA dense控制为101.33μs，实际native BF16 reader约97.78μs。CTA时钟采样中原reader约63%在QK、36%在PV；周期是CTA驻留周期，不能加总为设备延迟，也不能替代硬件stall归因。三个BUS主机root NCU均返回ERR_NVGPUCTRPERM，容器缺CAP_SYS_ADMIN；没有有效NCU报告。

## 当前集成方案与验收

新reader保持512线程，pair8/pv16载荷所有权、精确FP32重建FMA、原RoPE和点积求和；编译40寄存器、零stack/local，实际容量324CTA。第13轮将确定的分区规则写入429B常量表，在GPU上按长度查表，避免重复整数除法。自动dispatch暂限已测H8/108SM/B4..16，其余继续原路径，包括B2的R2。

第12、13轮已通过100项边界/LSE/动态Graph/填充检查，以及N1K/2K/4K/8K和B1/2/3/4/5/8/15/16真实输入测试（2K覆盖全部4loop），memcheck均零错误。同机原实现/候选生命周期与greedy-ID对照通过，完整吞吐提升超过预设5%门槛，第13轮已验收。


第13轮相同target=5对照进一步显示：full178.17μs、payload123.88μs；平均CTA的QK周期反而从44084增至46932，PV从26215降至21338。这不支持简单宣称“QK周期减少了”。63寄存器的原512线程内核最多容纳2CTA/SM，新40寄存器内核实际查询得到3CTA/SM容量；同样640个有效CTA可以减少资源容量要求的wave数。实际运行占用率、stall和cache流量仍需NCU验证，CTA时钟不能代替这些指标。


## 完整服务基线 trace

BUS2同一GPU2/3，Torch2.14.1cu130/Triton3.8/sglang-kernel0.4.9，完整TP2/R4，B16/N约2K。原实现生命周期检查通过。CUPTI活动trace含每rank1920次历史attention调用，对应10个decode step×48层×4loop；实际LOC为int64。

TP0累计GPU kernel时间542.16ms：历史reader65.23%，attention merge1.16%，GEMM及split-K21.31%，RMSNorm4.98%，collective2.91%，当前tokenRoPE0.80%，HP写入0.74%，索引编码0.24%，残差编码0.22%。TP1结论一致（reader64.85%）。这些是各rank内kernel时长的求和，不能替代CPU/通信重叠后的端到端时间。完整模型采用CUDA13编译，与隔离CUDA12微基准的绝对延迟不可混拼。

这份trace支持继续优化历史reader，而没有证据把本次主要瓶颈归于周期性残差编码。原始trace和分析保存在evidence/serving-baseline-profile/。


## 完整服务配对验收结果

原生产实现56fbb58de与候选4743e9430，在同一A800 GPU2/3、同一Torch2.14.1cu130服务环境、同一模型/码本/输入/seed605366612、TP2/R4/CUDA Graph下测量。2048输入、128输出，逐请求共同token屏障2→128，排除prefill与首个decode；每batch三次，先warmup。原始SSE、scheduler batch/Graph证据、配置与token IDs完整保留。

| Batch | 原VQ+ token/s | 新VQ+ token/s | 提升 |
|---:|---:|---:|---:|
| 1 | 49.055 | 49.118 | +0.13% |
| 2 | 85.577 | 85.558 | -0.02% |
| 4 | 129.056 | 168.965 | +30.92% |
| 8 | 189.413 | 264.672 | +39.73% |
| 16 | 293.573 | 371.748 | +26.63% |

候选与原实现的before/after、B3/5/15/16全部greedy output IDs一致，取消请求后缓存复用通过。候选CUDA13编译两个LOC宽度均40寄存器、零stack/local；原reader为64寄存器（CUDA12微基准是63）。新内核实际被调用，每rank1920次payload_stage1<long>。

相同完整服务profile中，TP0历史reader平均184.195→127.557μs，TP1为184.427→127.834μs；索引编码约658μs/次、残差编码约586μs/次基本不变，各只有2次。TP0累计kernel时间542.159→432.564ms，reader占比65.23%→56.62%。因此完整服务的主要收益可定位到历史reader，不能声称编码器得到加速。profile扰动后的吞吐不用于验收。

微基准恰好N2048与完整生成N>2048有分区边界差异：旧reader的active=max(ceil(N/1024),target)，B16 target=2；生成阶段越过2048后至少3个分区，而新策略在该区域保持5个。因此不能把微基准221→125μs与完整trace184→128μs混拼为跨层cache损失，工具链、索引宽度、长度与调度也不同。以上都是源码/trace支持的区别，精确硬件归因仍需要有效NCU计数。


## 同机BF16参考与剩余差距

BF16同模型、同输入、同TP2/R4、同GPU和服务环境，去掉VQ参数，采用原生Triton reader的8个KV splits。其生命周期检查通过；BF16不要求与量化模型输出token一致。每batch同样三次无profiler计时。

| Batch | 原VQ+ | 新VQ+ | BF16 | 新VQ+相对BF16 |
|---:|---:|---:|---:|---:|
| 1 | 49.06 | 49.12 | 47.37 | +3.69% |
| 2 | 85.58 | 85.56 | 90.44 | -5.39% |
| 4 | 129.06 | 168.97 | 170.09 | -0.66% |
| 8 | 189.41 | 264.67 | 262.93 | +0.66% |
| 16 | 293.57 | 371.75 | 419.64 | -11.41% |

单位token/s。新VQ+在B4/B8已接近BF16，B16仍低11.41%；不能宣称全面胜过BF16，更不能宣称全局最优。微基准B16/N2K的native97.86μs versus候选125.12μs也保留报告。BF16完整trace中历史stage1每rank1920次，TP0平均103.910μs、TP1为103.568μs；其merge约4.93μs/次。新reader约127.6–127.8μs，剩余历史attention开销依然真实存在。

本轮完成新增额度中的13轮；保留最优已验收R13，未为用满额度而晋升没有证据支持的变体。当前数值合同下，后续第一优先是取得有效SM80硬件计数，在相同长度、索引宽度、分区与服务工具链下分清码本gather、RoPE/转换指令、HBM/L2 sector和eligible-warps瓶颈，再选择下一项改动。预先改成post-RoPE训练表示、放宽BF16舍入或替换FP32累加属于新的算法/质量合同，不计作这次的无损执行优化。

完整原始证据位于`evidence/serving-{baseline,candidate,bf16}/`，汇总`evidence/serving-analysis.json`由`analyze_serving.py`生成。所有测试服务均已停止，GPU恢复空闲；没有改变系统驱动、没有正式分支晋升。
