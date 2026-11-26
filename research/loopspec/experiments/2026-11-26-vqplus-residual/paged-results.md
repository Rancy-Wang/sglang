# VQ+ 页级路径：10轮实施与最终结果

日期：2026-11-26。**实施及验证完成，性能目标未达成。** 在保留现有训练、codebook和pre-RoPE量化格式的约束下，本轮选出的实用候选是 head-local 页布局配合 CUDA R1/R2 分派。B16/2K完整服务吞吐从276.48提高到294.21tokens/s，提升6.4%；BF16为420.82tokens/s。这不是“已经找到全局最优”的证明，也没有实现大batch超过BF16的目标。

新授权的10/10轮已用完；未开始第11轮。代码已同步到实验分支`loopspec-test`，测试代码提交为`7d1c54b5ab30a98ffda5b4cd5bbd0b797479b064`。默认`token`布局保留，未晋升正式`loopspec`分支。全部退化候选、失败和决策保存在[paged-record.md](paged-record.md)。机器可读结果见[paged-results.json](paged-results.json)，完整20形状微基准见[paged-micro.csv](paged-micro.csv)。

## 最终实现

- 布局：index、residual、scale均按`[layer/slot,page,head,token8,payload]`存储，保留每个token通道连续性。`encode.py`和`residual_encode.py`的prefill及flush直接写目标布局；`LoopVQKVPool`沿用写完再发布映射的生命周期。服务中不做全历史重排或BF16展开。
- reader：全局只读码本，512线程CUDA融合重建、FP32 RoPE、BF16舍入、QK/softmax/PV。量化数据与HP prefix8/recent64共同处理，FP32 partial/LSE。没有借助改精度或模型深度提速。
- 分派：`ResidualAttention.use_grouped()`在head-local布局下仅B2使用双请求R2，其余使用R1；仍按已有设备端长度规则确定有效split。分派在capture时固定，无GPU长度回读。显式forcedR2诊断和原token布局策略保留。
- 接线：`triton_backend.py`初始化正确reader，CLI/pool检查要求INT4/global/SM80。原group32页布局、Triton、staging等失败候选作为显式实验保留，不作为默认推荐。

启用最终候选时，在已验证的VQ+服务配置上增加：

```bash
--loopvq-residual-layout page-head --loopvq-residual-page-kernel cuda
```

仍需要原有`--loopvq-codebook-dir`、`--loopvq-residual-bits 4`及`--loopvq-residual-codebook global`。关键源码：[pool](../../../../python/sglang/srt/mem_cache/loopvq_pool.py)、[CUDA页reader](../../../../python/sglang/kernels/ops/loopvq/residual_attention_paged_cuda.py)、[最终分派](../../../../python/sglang/kernels/ops/loopvq/residual_attention.py)、[R2 reader](../../../../python/sglang/kernels/ops/loopvq/residual_attention_grouped_cuda.py)。

## 完整服务实测

A12，同一对A100-SXM4-80GB，TP2，Ouro-2.6B-Thinking完整4loop×48层，原生Thinking模板，2048输入token、128输出token，greedy/ignoreEOS。优化服务配置、page8、max-running16、max-total65536、max-prefill16384、context32768；关闭radix复用、chunked prefill和speculation；decode full CUDA graph捕获B1/2/4/8/16。每形状warmup后3次测量，中位数如下，单位aggregate tokens/s。

|Batch|BF16|现有token VQ+|最终page-head VQ+|相对现有VQ+|
|---:|---:|---:|---:|---:|
|1|47.53|49.14|49.23|0.17%|
|2|90.63|85.40|85.72|0.37%|
|4|170.43|127.89|129.25|1.07%|
|8|263.58|186.32|189.83|1.88%|
|16|420.82|276.48|294.21|6.42%|

计时使用全batch共同完成第2至第128个token的壁钟区间，排除prefill和首次decode；所有样本通过真实active batch、full graph与无retraction证据检查。完整输入IDs和codebook哈希跨方法一致。原始事件、配置、逐步日志、3次样本和客户端最后token耗时均保留。BF16代码为9d6e92a33，token基线为32a69ef16，最终候选为7d1c54b5a；其间BF16和token算法行为未改变。各服务顺序运行，没有跨机器拼接性能。三次重复并非独立随机化的多轮服务交错，不应将约0.2%–1%的小差异作强结论。

第8轮始终R1页版B16达到293.84tokens/s，但B2只有79.12；最终B2恢复到85.72，B16仍为294.21。最终B16三次为294.252/294.160/294.213，现有VQ+为276.486/276.438/276.476，BF16为419.960/421.511/420.824。

本轮完整服务只重测2K；1K–8K覆盖来自下述微基准。没有宣称其他上下文已有同等服务收益，也没有扩展B32/B64。8K×B16不适合当前65536token的共同服务容量，未通过扩大候选容量制造比较优势。

## 四路对照

真实HF teacher-forced Q/K/V来自16个独立请求，使用训练好的codebook；三个层位置0/23/47、全部四loop。Q按服务FP32 RoPE约定处理。这不是live SGLang生成轨迹，也不是正式准确率评测。

微基准为单层静态数据重复CUDA graph，30个交错/轮换顺序样本，每样本160次调用；setup/编码/物化单独记录。代表性layer23/loop0、B16耗时如下，单位μs：

|长度|现有VQ+|最终页版|Scalar INT4对照|等价BF16 attention|页版延迟下降|
|---:|---:|---:|---:|---:|---:|
|1024|130.04|114.80|95.82|50.87|11.72%|
|2048|226.30|222.52|218.57|102.62|1.67%|
|4096|356.92|343.90|338.52|179.69|3.65%|
|8192|649.15|627.85|611.79|344.84|3.28%|

“等价BF16”是把相同VQ+重建后的K/V离线物化，再使用原生decode kernel，采用服务max8及实际动态split（B16/2K为5）；不把物化成本隐藏为可部署收益。此前轮次的fixed48诊断保留原标签，不能冒充这个服务调度基线。

Scalar INT4使用不同近似，只是成本/质量对照；其reader不是对所有成熟INT4实现的最优性基准，也未接入生产替换VQ+。144组抽样attention输出RMSE中位数：VQ+0.000297，scalar0.000948；各自最大0.003117/0.012045。这个样本支持码本仍有质量价值，不能推断完整模型准确率。

未锁定GPU频率，观察到idle210MHz、启动阶段1140MHz和持续服务1410MHz。部分小B微基准存在热身跃迁与跨次波动，例如B8/1K本轮显示约16%而此前约8%；保留全部样本，不据此宣称达到广泛15%门槛。最终B16四个长度的改善均小于15%。

## Profiler定位

以下为相同卡对、2K、B16、TP0，输出32token之后采样10个decode graph；按cudaGraphLaunch correlation归属kernel，避免将图外flush混入。单位ms/完整192层forward：

|指标|BF16|现有VQ+|最终页版|
|---|---:|---:|---:|
|主attention kernel时间之和|19.937|38.602|35.340|
|GPU graph墙钟包络|38.316|57.490|53.865|

TP1最终graph包络53.864ms、主attention35.337ms，与TP0一致。组件kernel时间之和可能与通信重叠，不能简单相加作为墙钟。最终页版普通图间隔与flush记录见JSON；一次flush约2.6ms、每8步一次，额外摊销约0.3ms/step，远小于主attention相对BF16的约15.4ms差距。单独优化flush不是当前主要收益来源。

NCU2026.3.1权限探测返回`ERR_NVGPUCTRPERM`，未取得硬件计数器。不能把上述差距具体归因为DRAM饱和、L1/L2命中、bank conflict或实际occupancy；源码、静态资源和配对消融支持进一步检查，不替代计数器。

## 十轮取舍

|轮次|候选|实测判断|
|---|---|---|
|1|group32页布局+协作scale load|B16/2K258.71 vs226.51μs，退化|
|2|紧凑grid、1024token局部tile|compact+coop257.67、compact token/plain229.51 vs226.53，拒绝|
|3|cp.async双缓冲payload|323.11 vs227.54，拒绝；racecheck通过|
|4|PV每warp一个token、4个acc/lane|303.12 vs228.11，大batch拒绝|
|5|RoPE或全部payload绕L1的load|241.49/264.51 vs226.43，拒绝|
|6|直接页编码/池/服务接线及Triton reader|codec通过；Triton2D contiguity hint编译失败|
|7|修正Triton hint|正确但300.04 vs226.37，拒绝性能晋升|
|8|head-local、保持token通道连续|B16/2K222.20 vs226.41；服务+6.3%，但B2长上下文退化|
|9|页布局恢复原R1/R2策略|修复B2；B16部分长度R2不及R1，不能照搬原分派|
|10|页版B2用R2，其余R1|最终验收通过；服务B16+6.4%，仍未达BF16目标|

没有把失败轮次改名重算，也未重训、改pre-RoPE格式、减少loop数、改变容差或缩短历史。历史第一批10/10仍是独立记录，不与新授权预算混淆。

## 正确性、容量与限制

- 编码：两种页布局的index/packed residual/FP16 scale逆变换逐bit一致；覆盖masked flush、invalid destination、3base×4loop、strided prefill，以及真实>2GiB sidecar地址配合int32 location。
- 最终100组边界、空行、变长、奇数batch、动态graph与陈旧scratch检查通过；output atol.002/rtol.01，LSE2e-5。144组真实跨层/loop/长度/B1/B2/B16对照通过，最大实现误差0.0009765625。
- 20项CPU兼容/容量合同通过。最终R2页reader在第9轮forcedR2模式下memcheck0errors；第2轮memcheck、第3轮staged racecheck分别0errors/0hazards，后两者只证明各自候选，未冒充最终全覆盖sanitizer。
- 最终TP2 fullgraph服务before/pad3/5/15/16/after输出IDs与token基线逐项一致；连续flush、abort及页回收复用通过。这是实现一致性，不是BF16与有损VQ+生成一致的保证。
- 页排布不改变KV容量：每元素摊销5bit，当前单loop实际读取index/residual/scale合计6.5bit；K+V payload208B/token/head，而BF16为512B。中心和RoPE访问及解包指令另计，不能将这个逻辑字节比直接当作HBM流量或加速比。
- 所有实验服务已正常停止，退出码0，GPU进程清空；X5/A系列持久SSH会话保留。模型、codebook和大原始快照未进入Git。

## 结论与下一步

在本次固定格式约束和已测范围内，推荐的实验实现是**head-local直接页编码+global CUDA融合reader+B2/R2、其余R1**。它改善了已有VQ+，但B8/B16主attention广泛下降15%及对BF16达到1.10×吞吐的门槛均未满足，因此保持opt-in，不晋升默认路径，不称“最效率已解决”。

源码事实是：pre-RoPE中心对query的贡献依赖历史位置，不能直接复用一张位置无关的中心logit LUT；当前BF16(C+residual)舍入也阻止无条件把中心与残差分开累加。页布局只能降低部分访问/地址成本，没有消除每token的重建、RoPE和浮点计算。实测还表明，协作shuffle、异步staging和Triton在线tile并非自动更快。这些结果否定了本轮候选的普遍收益，不证明固定格式已经达到理论上限。

**下一项最有价值的实验**是在具备GPU性能计数器权限的同类A100上，用现有`diagnose_paged.py --profile-one page_head_auto`及`--profile-one materialized_serving`分别采集B16/N2048、layer23/loop0的单次kernel，先用NCU列出可用sections，再收集调度/warp stall、memory workload和source counters。其余输入、codebook、编译参数和GPU保持一致，比较指令/依赖与访存的实际限制；不要先继续枚举线程数。当前A12权限不足，该计数器实验未执行。

进一步代码修改需要新预算；若继续研究post-RoPE/LUT或更低残差位宽，还需要单独扩展格式授权及质量门槛。本轮没有执行这些改动，也没有恢复已取消的大规模准确率评测。
