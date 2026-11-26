# VQ+ attention：面向大 batch 的重新设计

日期：2026-11-26。状态：**有源码与历史实测支持的设计方案；新方案尚未做 GPU 验证，不能称为已证明最优。**

## 1. 目标、范围与结论

目标是在 Ouro-2.6B-Thinking、A100-SXM4-80GB、TP2、完整4 loop 的普通 decode 中，兼顾量化质量、KV容量与大batch吞吐。优先 B8/B16、1K–8K，B1/B2/B4作为回归；B32/B64仅在三方都有相同可运行容量时扩展。

推荐分两层推进：

1. **先检验现有格式的执行效率上限**：重写为页级向量化、group32协作加载、紧凑有效任务调度的 decode kernel。保持当前量化与舍入，以独立实验隔离实现问题。
2. **下一代算法目标是直接计算压缩表示**：post-RoPE K 的中心logit LUT + 流式低比特残差；V端独立选择直接gather、按code聚合或纯标量量化。让codebook的作用成为“支持更低的残差位宽/更小的质量损失”，并以实际成本证明它值得保留。

不建议继续把“R1/R2、512线程、shared/global”作为全部设计空间；也不建议直接把完整路径换成Tensor Core。A100上的单token MHA有特定形状约束，低比特存储不等于低比特矩阵乘。

本次只交付计划；未修改kernel、未重训、未部署、未提交/推送。既有VQ+实现预算已到10/10，本计划不重置它；未来实施需要用户明确启动后续工作及预算。已取消的大规模准确率评测不恢复。

## 2. 任务合同与证据基线

工作区：`/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang/research/loopspec/experiments/2026-11-26-vqplus-residual/`。

源码根：`/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang`；分支 `loopspec-test`；HEAD `2eb4f0d380effc82cd4008fcf8096276098cdcbe`，测试代码 `4f9f8398d0955824e7c64dfcf63a826bbb70dc05`。开始分析时只有另一任务的未跟踪 `2026-11-23-loopvq-reproduction/evidence/tp2/snapshots/`，未修改。

训练参考：`/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/LoopVQ`，HEAD `dc3af341ba4593f580272470da803a5db839a423`，用户已有 `experiments/residual_ablation/record.md` 修改，保留。

| 项目 | 当前合同 | 证据/限制 |
| --- | --- | --- |
| 模型 | Ouro-2.6B-Thinking，48 base layers × 4 loop，D128，TP2本地8heads | 原结果与代码；Q/K头数相同的当前MHA路径，不泛化到GQA |
| 输入 | BF16 Q `[B,H,128]`，索引/残差页表，FP32 RoPE表 | `ResidualAttention.forward`；Q允许对应stride |
| codebook | C256；每code联合4loop×4channel；每loop解码4channel | `loopvq_codebook.py`；bundle G16不是每loop16channel |
| 存储 | index uint8跨loop共享；每loop独立INT4残差、group32 FP16 scale | `loopvq_pool.py`、`residual_encode.py` |
| 数值 | 中心先BF16；K先合成再RoPE再BF16；V合成后BF16；FP32 logits/softmax/accumulation，最终BF16输出 | 当前残差CUDA和独立oracle；不把HF原生所有舍入规则视作相同 |
| HP | prefix8、recent64，page8 flush；HP中保留pre-RoPE K | `OuroAttention.forward`、pool、allocator |
| mask | 完整因果历史、变长、空行、dummy页、混合HP/quant | 当前无prefix reuse、无chunked prefill、无speculative支持 |
| GPU环境 | 历史测试A100/SM80，服务Torch2.14.1+cu130/Triton3.8.0 | 本次未远程核实版本/可用GPU；A10是主机名，不是GPU型号 |
| correctness | codec逐bit；output atol=.002/rtol=.01；LSE atol=rtol=2e-5 | `check_residual.py`；新格式不得伪称旧codec逐bit兼容 |
| 目标 | 相同batch/长度的decode吞吐、完整forward、含flush摊销、容量 | 不能拿最大容量吞吐替代相同batch的速度比较 |
| 未知 | 真实Q/K/V快照是否仍完整可用；NCU权限；可接受新格式的质量预算 | 新kernel实现前补齐；不阻止当前方案分析 |

### 2.1 已核对的调用链

`srt/models/ouro.py::OuroAttention.forward`
→ `RadixAttention`
→ `srt/layers/attention/triton_backend.py::forward_decode`
→ `srt/layers/attention/loopvq_backend.py::forward_decode`
→ `kernels/ops/loopvq/residual_attention.py::ResidualAttention.forward`
→ R1 `residual_attention_cuda.py::stage1` / R2 `residual_attention_grouped_cuda.py::residual_grouped_stage1`
→ `official_reduce.py::official_reduce`。

编码：`LoopVQKVPool.set_kv_buffer`收齐四loop；decode由`loopvq_allocator.py::alloc_for_decode`触发`vq_flush_k/v`，依次写joint index、residual、scale，完成后发布映射。保留上一forward读完→flush→publish→下一forward的stream依赖。

强BF16基线：同一`optimized`服务配置的native paged KV和`triton_backend.py`的`decode_attention_fwd`分派；不是缓慢的HF独立评测实现。原VQ走`OfficialAttention`。

### 2.2 已有测量，不是本次重测

完整数据见上级 [results.md](../results.md)、[results.json](../results.json)。B16/2K完整4loop，TP0 profiler均值：

| 项目 | BF16 | 原VQ | VQ+ global |
| --- | ---: | ---: | ---: |
| 主attention，ms | 19.940 | 17.754，另HP 2.024 | 38.580，含HP |
| split reduce，ms | .954 | .790 | .831 |
| 完整forward graph span，ms | 38.482 | 38.893 | 57.620 |
| 无profiler真实decode，tokens/s | 418.62 | 413.46 | 275.72 |

VQ+普通图间隔约.081ms，flush约2.755ms，每8step一次，额外摊销约.334ms/step。它不能解释约19ms的forward差距。各方法旧服务测试顺序执行，非随机交错；小百分比差异不宜外推。

### 2.3 诊断分级

| 发现 | 证据类型 | 含义 |
| --- | --- | --- |
| 大B差距集中在主attention | 历史实测 | 首先改数据路径，MLP/CPU不是主要抓手 |
| row4每4channel读一次group32 scale | 源码事实 | 可减少load指令和地址计算；实际HBM降幅未知 |
| scalar unpack/convert/RoPE/FMA，未使用MMA | 源码事实 | 融合不等于消除解码成本；不是“Tensor Core利用率低”的计数器结论 |
| B16走R2，PV逐请求顺序做；固定64split网格 | 源码事实 | 应比较新的调度，不能假定双请求最优 |
| 去掉shared整表仅有限改善 | 历史实测 | 单独压缩码本/改global难以解释全部性能差距 |
| 寄存器spill是主因 | 无支持；旧报告R2 STACK/LOCAL=0 | 不采用此结论 |
| LSU/指令依赖/带宽/缓存/bank conflict谁主导 | 未验证假设 | NCU可用则读计数器，不可用则做分组件消融 |

## 3. 重新计算性能账本

令D=128、每loop的code group维度d=4、G=D/d=32、C=256、loop数L=4。

### 3.1 容量与读取不能用同一数字

每token/head/有效层，单种K或V：

| 项目 | 持久容量摊销 | 当前一次attention逻辑读取 |
| --- | ---: | ---: |
| index | 32/4=8B | 32B |
| residual | 64B | 64B |
| scale | 8B | 8B |
| 总和 | 80B=5bit/元素 | 104B=6.5bit/元素 |

K+V为160B容量、208B逻辑读取；BF16为512B。仅这些数组的理想读取比是512/208≈2.46，而不是3.2。四loop的index cache hit可能减少HBM，但当前读取指令仍存在；它们不会自动共享query或已重建KV。

每个token还逻辑查出K/V中心各256B，读取128个FP32 RoPE系数共512B；cache/shared可以复用这些数据，但load、地址计算、shuffle和依赖并非免费。不能把这些数字直接加成实测DRAM流量。

B16/H8/N=2048时，K+V压缩数组逻辑读取为52MiB/有效层，BF16为128MiB；全192层分别约9.75GiB和24GiB，忽略HP和元数据。在仅作说明的R2有效split=5下，320个CTA若各装入K/V两张64KiB表，shared staging逻辑搬运又约40MiB/层。实际2080上下文、缓存命中和HP会改变这些数值。

**codebook只是容量小，并不等于每步只读取一次；低精度残差只是字节少，并不等于其解包与浮点计算便宜。**

### 3.2 不能免费得到的并行与查表

- 不同请求的KV独立。把B个Q与拼接的所有KV相乘会产生无用交叉请求计算；block-diagonal packing也有padding/无效乘法成本。
- 当前每head每请求只有一个query。B增长增加CTA，不增加同一KV tile的query复用。四loop依赖不能为MMA而重排到一起。
- 对pre-RoPE中心，`qᵀR_p C_z = (R_pᵀq)ᵀC_z`。右侧query依赖历史p，不能建一张位置无关`T[g,z]`覆盖全历史。
- RoPE的每个channel pair频率不同；一个4channel group也没有统一旋转系数。当前相距64的pair甚至来自两个独立code。改成配对group可改善排布，但不会消除p依赖。
- 按page建RoPE LUT不是免费解决方案：query还依赖page base；若为每个位置建完整LUT，构建量为O(NCD)，不值得。
- 不能合并“同一K code”的tokens后只做一次softmax。不同位置、残差、其他product groups、mask都会改变完整logit。

## 4. 路线A：保留现有数值的页级解码

这是最先应实现并证伪的工程候选，作用是量出**当前格式的执行上限**；它不承诺从算法上消除O(ND)重建。

### 4.1 按消费粒度重新组织数据

页仍为8token，保留生命周期。建议试验布局：

- index：`[base_layer,page,head,group32,token8,code_in_group8]`，uint8。
- residual：`[slot,page,head,group32,token8,packed16B]`。
- scale：`[slot,page,head,group32,token8]`，FP16。

page内同head的数据紧密，避免当前token-major布局中相邻token隔着全部heads。group32为共同读取单位，K/V允许不同的warp分工；不要求强行共用row4。

每group32/token：8B indices、16B packed residual、2B scale。一个8lane子组各处理4channels：每lane读一个code和对应2B residual，一lane取scale后shuffle给其余7lane。对于RoPE，低/高两半协作，scale的两个group各保留自己的值。若更宽load需要跨lane重排，比较节省的load指令与新增shuffle，不能以“更宽”推断更快。

用32/64token tile（4/8个物理page）双缓冲；128/256线程起步。stage预取元数据、index和packed residual，再解码到寄存器；只在片上暂存当前tile，**不生成全历史浮点KV**。K/V采用独立loader，HP采用block-uniform边界处理，避免每个row4重复判断tier。

候选codebook访问：先global只读/L2，后测小group shard shared；不默认每CTA搬完整64KiB。只有复用足够覆盖搬运与smem代价时才stage。cp.async只用于实际合并的连续payload，不能解决随机中心lookup本身。

数值保持：所有center/residual合成、FP32 RoPE操作顺序、BF16舍入和softmax一致；先保留显式FMA边界。向量bf16x2/half2算术不自动等价，必须单独验证。

### 4.2 调度随任务量改变

图外或已有metadata阶段生成设备端有效任务描述 `(request,head,start_page,end_page,partial_slot)`；普通decode已知CPU序列长度，不需要额外GPU长度回读。

比较两种有界方案：

1. B/长度bucket各捕获不同紧凑split网格，设备端mask短行。减少固定32/64split远大于有效split的情况。
2. 固定小grid的persistent CTA消费设备端任务队列。head-major安排便于复用codebook；负载不均时再用work stealing，避免全局队列每token争用。

固定地址的`partial/LSE/valid_count`和任务容量在capture前准备；reducer只读本次有效槽，不能依赖陈旧LSE。队列溢出必须报错/fallback，不能漏算。在线softmax或每split本地softmax均可，选择由实测决定；没必要为了“Flash”命名强行每tile重归一化。

R1为可解释基线，R2/R4仅在共享码本/RoPE收益超过额外状态与串行化时启用。不能跨head复用不同的codebook。长上下文保留split；高B也不能机械设split=1。

## 5. 路线B：真正的压缩域attention

此路线改变数值/量化格式，必须使用新schema、独立quality对照和独立bundle。它是更有研究价值的方向，但风险高于A。

### 5.1 K采用post-RoPE编码，中心只投影一次

定义新缓存近似：

\[
\bar k_{p,g}=C^K_{g,z^K_{p,g}}+s^K_{p,a(g)}u^K_{p,g},\qquad
\bar v_{p,g}=C^V_{g,z^V_{p,g}}+s^V_{p,a(g)}u^V_{p,g}.
\]

其中K在已做RoPE的坐标中；`a(g)`对应32channel scale group；index仍可四loop联合，残差仍每loop独立。先保持d4/C256/INT4以隔离坐标变化，之后单独试d8/16和INT2。

对当前query（已经post-RoPE）建中心logit表：

\[
T^K_{b,h,g,c}=q_{b,h,g}^{T}C^K_{h,g,c},\quad
\ell_{b,h,p}=\frac{\sum_g T^K_{b,h,g,z^K_{p,g}}+
\sum_a s^K_{p,a}\sum_{j\in a}q_{b,h,j}u^K_{p,j}}{\sqrt D}.
\]

K residual scale可提到group32局部dot外；不能提到token求和外。中心不逐token重建，不在read端做历史RoPE。lookup和packed residual计算可并行发起，然后在softmax前合并logit。

代价账本：每query构建CD=32768次中心乘加；读取每token G=32个标量LUT，替代D=128个中心坐标和中心dot。N=2048时构建量是ND的1/8；这是运算量关系，不是速度比。G4点积很窄，按head/group批处理；不要为调用大GEMM构造32倍零padding。先自定义SIMT投影，与padding后的MMA同时计时再决定。

FP32 LUT为32KiB/query/head；B16/H8共4MiB/层，FP16为2MiB但需额外误差审计。不同query的LUT不能共享。每split搬完整LUT会抵消收益；比较全局L2 LUT、group shard、以及每request持久CTA构建后消费。LUT的构建、写入、读取、kernel launch全部进入计时，不能预计算Q相关表后只计lookup。

这项改变的质量风险很实际：post-RoPE分布更分散，原LoopVQ特意选pre-RoPE来改善量化。不能把旧码本简单旋转一次供全部位置使用。必须重新标定K，并检验残差熵、clipping、KV误差、logit/LSE/output误差、最终KL和长context位置外推。失败则回A，保留V侧探索。

### 5.2 新schema必须明确舍入，不能偷换等价性

旧合同：`BF16(R_p(C+s*u))`与`BF16(C+s*u)`。

压缩域线性展开采用的是未额外BF16截断的`C+s*u`，中心/scale虽为低精度存储，中间运算与accumulation是FP32。这与旧合同通常不同。例如：

\[
\mathrm{BF16}(C+su)\ne C+su,
\quad \sum_p a_p\mathrm{BF16}(C_{z_p}+s_pu_p)
\ne \sum_p a_pC_{z_p}+\sum_p a_ps_pu_p.
\]

用新格式的独立oracle定义参考，不拿“看起来更高精度”证明模型质量。若必须严格保留旧合成舍入，路线B的简单线性分解不适用，应回A或显式加入rounding correction并重新核算成本。

保持同一个稳定softmax；不能算`attention(center)+attention(residual)`。HP的分数也进入同一归一化，最后按LSE合并。

### 5.3 V端不对称：先直接gather，长context再试聚合

令完整softmax权重为`a_p`。V可分解成：

\[
O_g=\sum_c W_{g,c}C^V_{g,c}+\sum_p a_p s^V_{p,a(g)}u^V_{p,g},\quad
W_{g,c}=\sum_{p:z^V_{p,g}=c}a_p.
\]

这是真正codebook复用：重复码字只需要聚合权重后投影一次。然而当前G32/C256/d4下，FP32 W为32KiB/request/head；每token还需32次按code scatter-add，清零8192 bins，再做CD=32768次乘加。B16/H8、S个split若落全局，W单份就是4MiB×S，需额外归并与流量。原V直接PV只有ND次乘加，因此**V直方图不能无条件成为默认**。

首个B候选只改K的LUT，V仍用高效group32直接gather/解包，形成可归因结果。V直方图单独比较：

- split本地softmax求出`m,l`后，以`exp(logit-m)`累计W与residual accumulator；split末将`W×C + residual`化成128维向量，再走现有LSE reduce，避免全局W。
- 若跨tile在线更新m，W和residual都必须乘`exp(m_old-m_new)`。每tile缩放8192bins可能很贵，故不默认这种设计。
- 使用warp-private/分group直方图，局部合并；不使用每token对全局W做atomicAdd。记录冲突率、零初始化成本和实际code分布。
- 均匀code、单热点code、真实code分布均测。短split内重复少且固定bins多，优先直接gather；必要时只对较长split和更大d启用聚合。

不建议按code对全部KV排序：K/V有多个独立product groups，不存在一个同时有利于所有group的全局顺序；倒排索引与flush维护开销也必须计入。

### 5.4 残差：先省字节，再谈低精度计算

保持INT4作为格式切换控制组，之后逐项试K4/V2、K2/V4、K2/V2，group32不变。持久位宽分别为平均4/4/3bit，单次读取逻辑位宽平均5.5/5.5/4.5bit（d4/C256、每loop重复index）。scale和HP都计入。

仍用packed读→向量unpack→浮点dot作为A100初版。A100的INT4 MMA要求integer operands并累加INT32；不存在可直接把任意BF16 Q和INT4 residual相乘的免费混合指令。Q与P额外量化、token/group scale处理、MMA padding和数值误差必须单独测。逐token scale尤其令PV不再是简单的同scale整数GEMM。

新格式若可让残差降到INT2，同时保持质量，这才是codebook相对直接INT4最有价值的优势之一。若只能达到INT4残差且质量不优于直接INT4，应允许去掉对应K/V码本，而不是维护低效结构来满足名称。

### 5.5 保持训练框架最小变化

遵循原聊天偏好：保留Fisher加权k-means初始化与最终输出KL微调，不新增一组局部loss、旋转loss或多项正则。

独立消融顺序：

1. 固定现有bundle，验证A（不重训）。
2. 固定旧pre-RoPE格式，只比较合成舍入变更，量化B所需线性展开的数值差异。
3. 仅将K数据采集、Fisher坐标、编码/重建统一改为post-RoPE；V、位宽、loss不变。不能复用旧pre-RoPE Fisher权重冒充相同目标。
4. 新前向稳定后，按现有detached-residual方式进行相同KL目标的少量微调；残差冻结只是代理梯度，导出后重新assign和encode并评估真实离散前向。
5. 每次只改变一种残差位宽或group维度。d4→d8/16改变联合code维数与误差，需新bundle，不是布局优化。

质量门槛先用已有held-out校准文本、teacher logits和短rollout定义。原benchmark已取消，不能把取消的完整成绩填作证据；若未来需要正式准确率结论，应作为用户另行启动的任务。

## 6. 必须加入的竞争对照与被降级的方案

| 路线 | 价值 | 风险/拒绝条件 |
| --- | --- | --- |
| A 页级向量化+有效调度 | 保持现有质量，最快分离实现与格式问题 | 若真实KV微基准无法显著改善，停止围绕row4继续调参 |
| B K-LUT+V直接gather，后续2bit residual | 把重复中心计算移到query级，移除历史RoPE | post-RoPE质量、LUT流量/构建可能抵消收益 |
| B-V code概率聚合 | 以code复用减少V中心访问 | scatter/初始化/split直方图成本可能大于收益，独立开关 |
| 非对称K VQ+、V普通INT4/2 | 去掉V code lookup，保留K质量收益 | 必须重测V误差/输出KL，不能只看KV RMSE |
| 无codebook的直接INT4 group32 | 4.5bit容量/读取，是必要强对照 | 同质量或同容量比较；其量化误差可能更大 |
| 当前codebook INT8/4 | 可能降低中心cache/shared成本 | 不是主线；新增每lookup解码，scale定义与残差重编码使复杂度上升 |
| 每次展开全历史BF16后跑FA | 仅作诊断对照 | 每step全局写读展开通常破坏压缩收益，不默认部署 |
| Tensor Core fused dequant attention | 多query共享KV时更合理 | 当前MHA单query要展示有效MMA占比、padding和跨请求无用计算；不以峰值TOPS论证 |
| 只读top-k residual/稀疏残差 | 可能少读大段residual | 改变算法/softmax质量，单独近似研究；不能称等价优化 |
| 把四loop一起重建缓存 | 可能重用index | loop之间隔有模型计算，片上驻留不成立；全局缓存成本必须计入 |

## 7. 文件与函数级修改图（均为计划）

| 文件，相对SGLang根 | 计划修改 |
| --- | --- |
| `python/sglang/kernels/ops/loopvq/residual_attention.py::ResidualAttention` | 新variant/调度元数据，按B/N/schema分派，保持旧路径对照 |
| `.../residual_attention_cuda.py::row4/stage1`、`residual_attention_grouped_cuda.py` | 保留冻结基线；新A不再按row4扩充 |
| 新`.../residual_attention_paged_cuda.py` | page/group32加载器、流水tile、稳定softmax、compact partial |
| 新`.../residual_attention_lut_cuda.py` | B的K LUT构建/消费，V聚合作为独立variant |
| `.../residual_encode.py::_encode_residual` | 先仅输出page-major布局；B阶段才新增坐标/位宽schema |
| `.../encode.py::encode_steps/encode_from_hp` | 保留joint assignment；B接收post-RoPE真实K，禁止混用旧数据 |
| `python/sglang/srt/mem_cache/loopvq_pool.py` | 新arena view与字节统计、layout/schema，HP K坐标明确 |
| `.../loopvq_codebook.py::load_loopvq_codebook`、`loopvq_pack.py` | schema校验、旧bundle拒绝新格式误读；d变化才改pack合同 |
| `.../loopvq_allocator.py::alloc_for_decode` | 沿用写完再publish；新增任务描述不改变原page所有权 |
| `python/sglang/srt/models/ouro.py::OuroAttention.forward` | 仅B改K存储为已RoPE，prefill/decode/HP/flush统一；A无需改 |
| `python/sglang/srt/layers/attention/{triton_backend,loopvq_backend}.py` | 新kernel初始化与dispatch；不把index指针传入BF16后端 |
| `python/sglang/srt/model_executor/pool_configurator.py` | 实际capacity、LUT/partial/task scratch全部计入 |
| `python/sglang/srt/arg_groups/kv_cache_hook.py` | 明确旧schema与新schema兼容性，不暗改默认 |
| graph runner，先用`rg --files python/sglang/srt | rg 'decode_cuda_graph_runner'`定位 | 仅任务buffer/bucket确需新增capture状态时修改，避免凭旧文件名猜路径 |

训练参考根LoopVQ：`calibration/build_ouro_bundle.py::make_fisher_cache_cls`与K数据采集，`calibration/finetune_codebook.py::encode/rebuild/served_cache`，`calibration/residual_codec.py::encode_residual/decode_residual`，`experiments/residual_ablation/hf_reference.py::CoupledCodec`。全部是未来B阶段范围，不在A一起改。

## 8. 执行步骤与可复现命令

### E0：冻结证据和建立诊断基线

依赖：后续实施授权与预算；远程按LoopSpec AGENTS中的A10>A11>A12>A13>X5，A系列经X5，先读持久会话/实验技能并核对空闲GPU。不要照抄旧GPU4/5占用当前资源。

保存源码SHA、工作树diff、model/codebook SHA256、真实`sglang.__file__`、Torch/CUDA/Triton/NVCC/driver、GPU型号/SM、可用显存、服务完整命令和输入IDs。旧命令在工作区根`reports/vqplus-20261126/r10-global-command.txt`，本次已读，重跑时另选输出目录和端口。

现有GPU正确性命令（在核实后的目标SGLang checkout，使用服务环境Python；本次未运行）：

```bash
python research/loopspec/experiments/2026-11-26-vqplus-residual/check_residual.py --global-table --correctness-only --output "$VQ_RUN/baseline-correctness.json"
python research/loopspec/experiments/2026-11-26-vqplus-residual/check_pair.py --compare global --output "$VQ_RUN/table-ablation.json"
```

`VQ_RUN`是新建绝对输出目录。第二个是已存在的shared/global随机KV消融，不能替代真实快照。旧benchmark没有“保存真实KV并重放四对照”的接口，不能假装现成支持。

**拟新增harness** `experiments/.../diagnose_attention.py`（尚不存在）须先实现以下接口和记录项：

- `--snapshot <manifest>`：Q、四loop索引/残差/scale、HP、logical positions、页表、codebook hash，层0/23/47、四loop，不能混rank。
- `--variant current|materialized-oracle|scalar-int4|paged|lut-k|hist-v`；unsupported variant明确拒绝。
- `--batches 1 2 4 8 16 --lengths 1024 2048 4096 8192 --repeats 30 --output <json>`。
- `--profile-one`：warmup/JIT后只提交指定一次kernel，并记录实际kernel名/shape；供NCU使用。
- materialized-oracle使用**同一VQ+解码后的**K/V，离线重建不计热kernel，但完整重建另列计时，不作为可部署加速比。scalar-int4必须从对应真实BF16 KV重编码，不能从VQ+近似KV二次量化来比较质量。
- 每variant至少warmup到稳定；同一输入、同一图模式、交错AB/BA，30组配对样本，报median/p10/p90和置信区间；轮转层/codebook避免单层hot-cache微基准偏乐观。

产物：`baseline.json`、`snapshot-manifest.json`、`environment.json`、`breakdown.json`。退出条件：可重放当前正确输出，确认kernel差距；有当前真实输入分布，而非只用随机code。

### E1：取得瓶颈证据

NCU能力发现命令（目标机执行，本次未运行）：

```bash
ncu --version
ncu --list-sets
ncu --list-sections
ncu --query-metrics
```

在独立`profile/<unique-run>/{harness,reports,analysis}`记录每次采集。已存在源码带`-lineinfo`；按实际kernel名筛选R1/R2，B16/2K为第一形状，再B1/8K和B16/8K。使用新harness的`--profile-one`防止profiling到codec warmup。

验证SM80支持后采`--set full`和`--set source --section SourceCounters`，PM sampling只在本机工具/GPU支持时加，不照抄B200指标。用安装的`ncu_report`枚举metric names，记录DRAM实际bytes、L1/L2流量命中、global/shared transaction、指令构成、eligible/active warps、寄存器/local memory、stall与源行。profiler时长不作为性能接受样本。

若仍`ERR_NVGPUCTRPERM`：保存错误，不改系统权限/驱动；继续SASS/ptxas资源报告和诊断variant，不把缺少计数器当作停止所有分析的理由。拆出K重建、V重建的harness只能定位成本，不能将删除计算后的时长当成正确候选。

### E2：A的数据路径与调度分别消融

第一候选只改page/group32 loader和布局，保留旧split/softmax；第二候选才改compact任务调度。这样可以区分访存与调度贡献。每个候选都保留父SHA、diff、编译flags与结果，不能用多次修改共用一轮记录。

正确性：旧codec逐bit；逻辑slot到page重排可逆；0/8/64/72/79/80/127/128及32K；非连续page；奇数B3/5/15；混合长度；四loop/TP2；BF16 ties、tiny scale、NaN/Inf拒绝；graph padding、连续flush、abort和复用；memcheck/racecheck。

验收：output/LSE保持现有容差，固定服务验证prompt维持旧VQ+ token一致。新归约引起argmax分歧时记录首次logit差异，不能静默放宽。先micro，后full forward，再真实serve。若没有超出噪声的改善，拒绝该变体，不靠改精度制造提升。

### E3：B先验证数值，再验证LUT，再考虑V聚合

依赖：E0有真实KV，质量门槛已在看结果前定义。先做CPU/torch代数oracle验证，再少量层/四loop的重编码；post-RoPE、舍入、INT2、d变化逐个加入。

B的oracle采用新schema显式重建FP32 `C+s*u`，计算完整FP32 attention；与压缩域logits、LSE、output比较。FP32 LUT初版优先，FP16 LUT单独消融。任何质量收益必须基于重新encode的exported bundle，不能只用训练态loss。

GPU第一候选K LUT + V直接gather；只有它通过后才考虑hist-V。对V同时测直接gather与hist，统计bins初始化、scatter、W×C、额外scratch与split merge。hist若只在长N赢，用长度dispatch；全域失败就删去该候选，不把理论乘法减少当成功。

### E4：端到端与质量Pareto验收

已存在服务测试命令（服务先按旧完整命令启动并核对配置，以下变量由E0确定）：

```bash
python research/loopspec/experiments/2026-11-26-vqplus-residual/check_serving.py --url "$VQ_URL" --model "$VQ_MODEL" --out "$VQ_RUN/serve-timing" --server-log "$VQ_SERVER_LOG" --method vqplus --suite timing --input-length 2048 --output-length 128 --batches 1 2 4 8 16 --repeats 3
```

沿用输入IDs、ignore-EOS、greedy和TP2/fullgraph协议；先用旧三重复复现，再通过相同命令的`--repeats`扩充配对样本。BF16、原VQ、当前VQ+、候选依次对同一GPU卡对运行并交错轮次。新scalar方案若需扩充`--method`，应先实现测试接口，不伪造已支持。

报每shape吞吐/延迟、正常step、flush平均与p99、prefill、整个请求端到端、最大显存、相同内存预算可承载batch。容量收益和相同batch速度分列。新B32/B64超出旧harness/graph容量需明确扩展，不假称已支持。

## 9. 晋升、拒绝与停止规则

以下是**拟议工程门槛**，不是已有实测或用户已批准的质量阈值：

- A：保持旧正确性标准；目标B8/B16主attention中位延迟比旧VQ+至少下降15%，且配对95%区间排除零收益，才能进入昂贵serve验证。未达门槛仍保存结果，不称最优。
- 完整目标：在可共同运行的B8/B16、1K/2K/4K/8K集合，对强BF16的吞吐几何平均至少1.10×，每shape不低于0.95×；小B不低于其已选VQ+路径0.95×。失败则限定适用域或拒绝，不把局部收益说成全局胜出。
- 新B格式：没有批准的quality budget时，只能产出质量/速度Pareto，不自动晋升。建议先要求held-out KL不劣于旧VQ+误差预算、KV重建/attention输出/长上下文rollout均报告，不能仅用更低RMSE代表准确率。
- 与普通INT4比较：分别做相同有效位宽、相近quality两类比较；若VQ+在两类都被支配，保留非对称或去codebook方案。
- 所有LUT构建、格式转换、flush、调度、scratch必须计费；不得更改模型loop数、缩短真实历史、改变evaluator计时窗口。
- 达到标准即停止不必要调参。未达到时根据E1/E2证据选择B或宣布当前格式上限，不无限枚举线程参数。实现轮数需遵守后续获准预算，原10/10保持原样。

## 10. 第一实验与关键未知

**第一实验：同一真实Q/K/V快照上的B16/H8/N≈2K四路诊断：当前VQ+、离线重建同一VQ+后用强BF16 kernel、直接INT4、page/group32候选A。** 先冻结质量与输入，再比较完整计费的kernel，区分“格式有价值但decoder慢”和“codebook在4bit残差下已不划算”。准备阶段先完成前两个无新kernel对照与真实快照，A在后续实现授权后加入。

关键未知：实际NCU瓶颈；直接INT4对比VQ+的质量差；post-RoPE重编码的误差增幅；INT2残差能否保持质量；真实码字重复/热点分布；LUT构建与scatter在SM80的成本；可接受的quality budget。

本次没有把任何候选标成更快，也没有将历史报告当成新测量。设计推荐可以明确，最快variant仍需上述对照确定。

### 本次已做的代数校验

用NumPy2.2.1、seed20261126、合成N97/D32/d4/C16，在CPU float64下检查新加性schema：K-LUT与显式重建logit最大差`6.66e-16`，V直方图与显式PV最大差`2.78e-16`。反例中，忽略旧BF16合成舍入产生约`6.96e-4`的输出差；误把pre-RoPE K当位置无关LUT产生约`2.56`的logit差；两次独立softmax再相加产生约`.178`输出差。它们只验证代数成立条件和反例，**不是实际模型误差、容差失败结论或性能测量**。结果见 [algebra-check.json](analysis/algebra-check.json)。

## 11. 参考与证据

- 原聊天：[克隆 LoopVQ 并梳理代码](codex://threads/01a1f8df-2e9d-74f3-b471-0ec6577f4ee2?hostId=remote-control%3Aenv_e_6a68ec597c9c8325a8035cb714e0bc83)。已读取残差设计目的、训练收窄、实现及最终退化回报的相关turn；没有声称重读全部无关评测过程。
- 源码与历史测量：[原记录](../record.md)、[原结果](../results.md)、[机器可读结果](../results.json)、[正确性harness](../check_residual.py)、[配对harness](../check_pair.py)、[服务harness](../check_serving.py)。本次source hash清单见`analysis/source-manifest.json`。
- [LoopVQ论文§4.3/Appendix C](https://arxiv.org/html/2611.05304v1)：支持pre-RoPE量化、每loop读索引、跨请求复用RoPE/码本的原设计背景；论文的GPU/模型/无残差收益不能代替本任务测量。
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)：用于核对SM80共享内存、异步拷贝和BF16/INT4 MMA类型；未将Hopper/Blackwell TMA/WGMMA/TMEM用于A100。
- [NoMAD-Attention](https://arxiv.org/html/2403.01273v1)：乘加转LUT的相关原始研究，目标CPU SIMD；这里的post-RoPE K + residual方案是针对当前系统的推导，不借用其CPU速度作为GPU证据。
- 使用KDA的kernel planning流程、Loop Transformer研究技能和NCU报告技能。NCU技能的B200默认配置已按SM80和现有权限限制收窄；本次未生成NCU报告。
