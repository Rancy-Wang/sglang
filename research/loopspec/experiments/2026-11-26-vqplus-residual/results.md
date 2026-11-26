# VQ+ 残差 attention：实现及性能验收（2026-11-26）

已完成批准范围内的 10/10 轮修改与验证。测试源码为 `4f9f8398d0955824e7c64dfcf63a826bbb70dc05`，已推送用户 fork 的 `loopspec-test`；本报告后续提交仅补充结果，不改变测试源码。正式分支未晋升。

**结论：VQ+ 已能直接使用压缩残差执行 TP2 full-CUDA-graph 推理，显著降低抽样 KV 重建误差，但目前没有实现普遍的 BF16 吞吐加速，不能称为“最优效率”。** 真实服务中 global 查表通常略优于 shared；batch1 长上下文有收益，大 batch 的主 attention 仍明显慢。以下保留全部退化结果。

## 1. 取消与保留的范围

上一轮七组 TP2 准确率评测已停止，相关本地/远端输出、准确率表及 24 个 W&B 评测 run 已删除，自动跟进已关闭。训练 codebook、训练日志与两个训练 run 保留。这里的 oracle、合成提示 serve 和 KV 重建审计是算子验证，没有重新启动 AIME/GPQA/HumanEval/MBPP 准确率评测，也没有推断其准确率提升。

所有本次 GPU 测试结束：A10 GPU4/5 与 A12 GPU0–7 均已释放；只停止本任务进程，未触碰他人作业。SSH/X5 持久连接保留。

## 2. 最终实现怎样工作

1. **存储**：原 0.5bit 联合四 loop 的索引不变；每个 loop 独立增加 signed INT4 残差和 group32 FP16 scale。sidecar 使用相同 quant 物理槽位、相同页生命周期，各 loop 不共享残差。
2. **编码**：中心先按已有训练解码约定转为 BF16。`e=x-center`，`s=max(FP16(maxabs(e)/7),FP16.tiny)`，再以已舍入的 s 做 round-to-even，限幅到 [-7,7] 并两码打包。K 取 pre-RoPE，V 取原始空间。非有限值/溢出在 GPU 编码阶段拒绝。
3. **写入**：prefill 收齐四 loop 后写索引与残差；decode 每 8 token flush 时先联合分配索引，再编码 K/V 残差，完成后发布映射。没有每层 decode 的独立解量化 kernel，没有完整 BF16 历史展开。
4. **attention**：同一个 stage1 内 lookup → INT4 unpack/scale → K RoPE → BF16 舍入 → QK → softmax → V 重建/BF16 舍入 → PV；HP 与量化区共享 softmax，随后一个 LSE reducer 合并 split。所有空 split 覆写输出/LSE，去掉单独 reset。
5. **调度**：512线程；R1 单请求 CTA、最多32split，R2 双请求共用码本/RoPE、最多64split；不同请求的真实 KV/softmax 独立。默认 B2、B>=16 用 R2，其余 R1。实际 split 依据设备端长度，batch 分派在 graph capture 时确定，不读回 GPU 长度。
6. **码本访问**：shared 路径搬入64KiB表，R1 用 cp.async，R2 同步；global 路径省掉整表搬运，直接用只读缓存 lookup。保留 shared 默认，global 为显式选项；实际模型收益与随机索引微基准不完全一致。

代码入口：

- `python/sglang/kernels/ops/loopvq/residual_encode.py`：残差量化、scale、打包。
- `python/sglang/kernels/ops/loopvq/residual_attention.py`：R1/R2、graph、安全分派。
- `residual_attention_cuda.py` / `residual_attention_grouped_cuda.py`（同目录）：融合 CUDA stage1。
- `python/sglang/srt/mem_cache/loopvq_pool.py`：残差池、prefill/flush、容量。
- `python/sglang/srt/layers/attention/triton_backend.py`：SGLang 接线。
- `python/sglang/srt/arg_groups/kv_cache_hook.py`：数值 schema/功能兼容性校验。

在已验证的 standalone LoopVQ 启动命令上增加：

```text
--loopvq-codebook-dir /homes/ruoxi07/experiments/loopvq/2026-11-26-residual/artifacts/residual-ft
--loopvq-residual-bits 4
--loopvq-residual-codebook global
--loopvq-prefix-tokens 8 --loopvq-recent-tokens 64
```

完整命令与依赖见原始证据里的 `r10-global-command.txt`、各 context 的 `command.txt`；不能只凭以上片段在不同环境宣称复现。CLI 原有 codebook help 的 TP1 描述陈旧，实际 hook 与此次 TP2 验证为准。

## 3. 真实 serve 的纯 decode 吞吐

Ouro-2.6B-Thinking，完整4loop，A100-SXM4-80GB，TP2/DP1，原生 Thinking 模板，page8、HP8+64、greedy、ignore EOS、full CUDA graph。服务 Torch2.14.1+cu130/Triton3.8.0。禁用 prefix reuse/chunked prefill/LoopSpec。

**本表 1K/2K/4K/8K 指精确输入 token 数；每次新增128 token。不是此前的长输出上限矩阵。** 每形状先32token warmup，再3次测量取中位数。所有请求共同完成第2个 token 到共同完成第128个 token 计时，排除 prefill 和首 token；报 aggregate tokens/s，非单请求速度。原始 arrival events、完整 input IDs、服务器真实 batch/fullgraph/retraction 证据保留。19形状×4方法×3重复=228次计时批次，全部通过测量证据检查。

同一行四种方法使用同机同卡对：1K=A12 GPU2/3，2K=A10 GPU4/5，4K=A12 GPU4/5，8K=A12 GPU6/7。各方法顺序运行，真实服务未做随机交错运行；微基准另有同进程交错消融。不要把小百分比差异外推到所有提示或机器。BF16/原VQ为冻结的 R3(A10)/R7(A12)；shared为R9(A10)/R10(A12)，global为R10；原路径计算未随此次残差候选改变。

| 输入 | Batch | BF16 | 原 LoopVQ | VQ+ shared | VQ+ global | global/BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 52.04 | 47.30 | 50.31 | 51.31 | 0.986× |
| 1K | 2 | 100.25 | 85.72 | 88.05 | 91.24 | 0.910× |
| 1K | 4 | 194.30 | 171.54 | 154.42 | 156.67 | 0.806× |
| 1K | 8 | 328.16 | 278.88 | 242.36 | 244.67 | 0.746× |
| 1K | 16 | 525.62 | 500.81 | 354.64 | 354.60 | 0.675× |
| 2K | 1 | 47.47 | 46.83 | 47.56 | 49.06 | 1.034× |
| 2K | 2 | 90.39 | 84.71 | 83.21 | 85.31 | 0.944× |
| 2K | 4 | 170.31 | 143.33 | 123.74 | 127.58 | 0.749× |
| 2K | 8 | 262.88 | 257.63 | 178.71 | 185.82 | 0.707× |
| 2K | 16 | 418.62 | 413.46 | 272.39 | 275.72 | 0.659× |
| 4K | 1 | 41.07 | 41.85 | 43.80 | 45.56 | 1.109× |
| 4K | 2 | 77.56 | 71.50 | 72.85 | 75.83 | 0.978× |
| 4K | 4 | 142.34 | 136.53 | 87.81 | 91.30 | 0.641× |
| 4K | 8 | 200.87 | 211.11 | 136.21 | 140.62 | 0.700× |
| 4K | 16 | 309.26 | 297.51 | 170.60 | 177.02 | 0.572× |
| 8K | 1 | 31.97 | 40.09 | 37.09 | 38.47 | 1.203× |
| 8K | 2 | 59.67 | 67.69 | 56.82 | 59.91 | 1.004× |
| 8K | 4 | 106.58 | 111.86 | 65.50 | 69.06 | 0.648× |
| 8K | 8 | 134.48 | 148.84 | 83.82 | 89.93 | 0.669× |

8K×batch16 的 BF16 KV 容量超过80GiB/rank，未运行，不用 VQ+ 单边结果推导加速比。2K容量65536，其余81920，各自方法容量一致。

当前建议：本机已测真实提示可显式选择 global；1K batch16 与 shared 基本持平。不要据此删掉 shared：同进程随机索引微基准中，global 的1K B1约1.167×、8K B1约0.903×、2K B16约0.932×，说明实际索引分布与缓存命中会改变排序。未实现未经验证的按长度自动选路。

## 4. 一个完整 forward 的时间

以下来自 A10 同卡对、2K输入，输出约32token后采样10step，列 TP0 均值，单位ms。forward包括全部192个有效层（4loop×48层）；TP1单独保存在 results.json。这是 profiler 数据，不能替代上表无 profiler 吞吐。按 cudaGraphLaunch correlation 归属 kernel，避免把 graph-id=0 的图外工作混入。

| 步骤 | BF16 B1 | VQ+ global B1 | BF16 B16 | 原VQ B16 | shared B16 | global B16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attention主kernel（VQ+含HP） | 4.765 | 3.985 | 19.940 | 17.754 | 39.704 | 38.580 |
| 独立HP kernel | 0.000 | 0.000 | 0.000 | 2.024 | 0.000 | 0.000 |
| attention split/LSE reduce | 0.729 | 0.484 | 0.954 | 0.790 | 0.824 | 0.831 |
| 线性层 GEMM/GEMV及split-K reduce | 9.924 | 9.980 | 11.136 | 11.555 | 11.641 | 11.607 |
| RMSNorm | 2.867 | 2.818 | 2.701 | 2.734 | 2.726 | 2.731 |
| TP通信 | 1.212 | 1.306 | 1.629 | 1.536 | 1.670 | 1.704 |
| fused RoPE | 0.464 | 0.441 | 0.442 | 0.442 | 0.450 | 0.450 |
| KV/HP写入 | 0.354 | 0.376 | 0.374 | 0.409 | 0.402 | 0.401 |
| MLP fused activation | 0.379 | 0.370 | 0.389 | 0.387 | 0.384 | 0.383 |
| 模型 residual add | 0.661 | 0.656 | 0.776 | 0.789 | 0.793 | 0.794 |
| LSE fill / tier metadata | 0.000 | 0.000 | 0.000 | 0.284 | 0.000 | 0.000 |
| 其他图内kernel | 0.024 | 0.007 | 0.013 | 0.013 | 0.013 | 0.013 |
| **完整 GPU graph span** | **21.559** | **20.597** | **38.482** | **38.893** | **58.729** | **57.620** |

组件是各 kernel duration 之和，通信可能重叠；graph span 是墙钟包络，不能把组件简单相加再叠加 CPU synchronize 等待。MLP列仅激活，线性计算在 GEMM/GEMV 列中。这里的线性列包含 cuBLAS split-K reduce，早期摘要将它放在other；总span与attention结论不变。

B16 的 attention 主kernel按四loop分别为：

| Loop | BF16 | VQ+ shared | VQ+ global |
| --- | ---: | ---: | ---: |
| 1 | 4.991 | 9.933 | 9.658 |
| 2 | 4.994 | 9.922 | 9.642 |
| 3 | 4.981 | 9.923 | 9.640 |
| 4 | 4.975 | 9.926 | 9.640 |

四个loop开销接近，未见某个loop单独失控。global相对BF16的forward约多19.14ms，其中主attention约多18.64ms；优先级应落在残差attention数据路径，而非重新改MLP或CPU提交。

**图间隔与flush：** global B1普通间隔中位数0.079ms，flush尖峰0.350ms；B16普通0.081ms，flush尖峰2.755ms。10step trace内一轮flush的两个联合索引编码kernel总计1.335ms，两个残差编码kernel总计1.239ms（B16）。相对普通间隔，额外约2.674ms；按每8step一次约0.334ms/step，为摊销估计，不是单独测得的端到端提速上限。B1对应索引0.174ms、残差0.082ms。

图内所有kernel区间取并集后，global约0.177ms/step未被kernel覆盖；这不是CPU阻塞的直接证据。正常图间隔中也包含GPU metadata、sampling等工作，不能全部称为CPU空等。CPU cudaEventSynchronize与GPU执行重叠，不应重复累加。当前没有持续毫秒级CPU图提交空洞；flush是真实额外GPU计算。

VQ+把QK、RoPE、softmax、HP和PV融合在stage1中，现有trace无法分别计时这些内部阶段。NCU硬件计数器因权限受限，尚不能将慢因确定为DRAM带宽、L1/L2命中率或shared bank conflict；任何此类解释都只能算待验证假设。

## 5. 显存和数值验证

残差实际成本为 **0.5bit索引 + 4bit残差 + 16/32bit scale = 5bits/元素**。不能称为0.5bit缓存，也不能忽略scale说总共4bit。TP2每rank全模型quant token：原VQ 24KiB，VQ+ 240KiB，BF16 768KiB。VQ+历史部分相对BF16约3.2倍压缩，相对原VQ扩大10倍。

在capacity65536的实际pool日志（每rank）中：

| 方法 | K+V pool（GiB） |
| --- | ---: |
| BF16 | 48.00 |
| 原LoopVQ | 约2.98 |
| VQ+ | 约16.48 |

VQ+含固定HP/码本约1.478GiB；不是进程总显存峰值。权重、prefill暂态、activations、CUDA graph scratch等另计，不能由本表宣称端到端峰值节省相同比例。

最终验证：20项CPU合同通过；codec bytes/scale独立oracle逐bit通过；attention含0/8/64/72/79/80/127/128及1K–32K长度、混合长度、奇数batch、graph padding通过；final共享/global套件最大绝对output误差1.5259e-5，在预先固定atol0.002/rtol0.01内。异步R1/R2 memcheck/racecheck为0errors/0hazards；finalglobal R1/R2 memcheck为0errors。未放宽容差。

真实TP2中，连续flush、B3/5/15/16进出、abort后页回收复用通过；同一验证提示的R7 eager、R7 graph、R9 shared、R10 global输出token IDs全部一致（before/after各128token，其余batch各48token）。这是生命周期/实现一致性检查，不是BF16与有损量化输出相同的保证。

额外抽样真实HF KV：层0/23/47、head0/7/8/15、256token、完整4loop。同一输入下，GPU assignment与训练FP32 cdist的K/V各98304联合索引均无差异。以下RMSE以**同一residual-ft码本**比较不加/加残差；不是原始0.5bit码本与新训练码本的准确率比较。

| 类型/层 | 仅中心RMSE | 中心+INT4残差RMSE |
| --- | ---: | ---: |
| K / 0 | 0.270355 | 0.030383 |
| K / 23 | 0.583655 | 0.067879 |
| K / 47 | 0.554720 | 0.062214 |
| V / 0 | 0.056252 | 0.005547 |
| V / 23 | 0.245365 | 0.023584 |
| V / 47 | 0.136508 | 0.013450 |

误差约降低9–10倍，但不能推出benchmark准确率提升。训练K来自BF16 post-RoPE逆变换，服务K来自原始pre-RoPE；抽样引起549/98304（0.558%）联合索引差异，V为0。为兼容HF模型，该审计用既有训练env Torch2.8/Triton3.4/Transformers4.56.2；服务env为Torch2.14/Triton3.8。该结果不能替代跨环境端到端逐bit证明；服务独立oracle另外通过。没有修改checkpoint来掩盖输入差异。

## 6. 10轮优化的复查结论与后续优先级

已尝试：Triton→专用CUDA、128/256/512线程与动态split、token unroll、双请求共用码本、顺序PV减少同时存活累加器、按batch分派、cp.async、global查表。每轮目的、风险、命令、退出码和退化见record.md。R2大B最终64registers、STACK/LOCAL=0；不能宣称发生寄存器spill。异步表搬运同进程消融通常只有0–5%收益，不能用早期独立运行的时钟差异宣传8–13%。global真实serve较shared通常改善约1–7%，远不足填平大B对BF16的差距。

剩余优化计划（**尚未实施，不计为已取得收益**）：

1. **优先优化重建/访存指令密度。** 当前`row4`每4channel lookup一次center、读一份scale、unpack；group32 scale在多个lane/调用中重复读取。设计group32协作加载/广播与更宽packed读取，保持BF16舍入，减少同token/head重复地址计算与scale load。必须看反汇编和同卡oracle/延迟，不能仅按理论字节数认定更快。
2. **重新匹配R2、split与global资源。** 当前R2和512线程策略是在shared占用下选出，global去掉64KiB shared后不一定仍最优；分别扫描R1/R2、线程数、每split长度，依据实际模型KV分布做同进程交错测量，报告短/长和各batch退化。不能直接把不同请求当成共享KV的矩阵乘。
3. **优化flush但不误判其主次。** 可以融合索引确定后的残差编码、减少中间索引读回和kernel launch；难点是四loop联合索引与group32 scale归约。当前B16摊销约0.33ms/step，远小于主attention的约19ms差距，优先级低于前两项。
4. **数值合同单独消融。** 若要用更激进的query侧RoPE、拆分PV的center/residual累加或Tensor Core tile，需要独立证明舍入变化和decode形状利用率；不能把它们当作当前训练语义下无损替换。先不改训练/量化目标掩盖kernel瓶颈。

已用完批准的10轮，本次不继续第11轮。当前交付是可运行且有完整性能边界的实现，并非最优性证明。支持范围仍是已验证的sm80/A100、D128、R4、TP2、HP8+64、最长32K；未验证LoopSpec、prefix reuse、chunked prefill或offload。

## 7. 证据与复现位置

- 本地完整原始证据：`reports/vqplus-20261126/`（工作区根目录下）；`final-throughput.json`、`final-step-summary.json`、`final-lifecycle-comparison.json`为完整摘要。
- 本目录`results.json`为可随Git同步的数字及证据映射；所有吞吐summary保存SHA256。
- 远端原始根目录：A10/A12共享`/homes/ruoxi07/experiments/loopvq/2026-11-26-vqplus/`。
- final source checkout：`/homes/ruoxi07/repo/sglang-vqplus-r10`，冻结于测试SHA；没有改变运行中的旧checkout。
- A10 `r10-global-{lifecycle,timing,profile}.exit`和server.exit为0；A12六组final shared/global context client/server.exit均0，GPU已释放。
- shutdown后存在Python临时目录清理`OSError: Directory not empty`（NFS目录），发生在计时完成和服务退出清理阶段；原始stderr保留，未把它隐藏为“日志全无错误”。没有推理异常/测量证据失败，server退出码仍0。
- NCU计数器不可用、训练env首次HF接口不匹配、早期失败候选等证据均保留；失败没有被删除或替换成成功记录。
