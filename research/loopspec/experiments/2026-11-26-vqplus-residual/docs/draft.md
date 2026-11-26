# VQ+ attention 重新设计：初始分析

2026-11-26；本次是设计审查，不修改推理代码、不启动训练或 GPU 实验。

## 恢复的设计目的

已读取聊天“克隆 LoopVQ 并梳理代码”中残差方案的原始提议、随后收窄训练范围的要求、VQ+ 实现计划、10/10 轮结果以及大 batch 退化分析，并核对本地实现。原目标是降低 LoopVQ 相对 BF16 的误差，同时仍利用压缩 KV 提升推理效率。后续用户取消完整 BF16 残差，要求保留 LoopVQ 的 Fisher 加权初始化和最终输出 KL 微调框架，逐个增加变量。大规模准确率评测已取消，不应在设计审查中恢复。

现有源码 SGLang HEAD 为 `2eb4f0d380effc82cd4008fcf8096276098cdcbe`，最终测试代码为 `4f9f8398d0955824e7c64dfcf63a826bbb70dc05`。现有实现是 A100/SM80、Ouro-2.6B-Thinking、TP2、D128、4 loop、48 base layers、page8、HP8+64 的普通单 token MHA decode。

## 初步判断

1. 已有融合避免了全历史 BF16 展开，但仍为每个历史 token 执行 codebook gather、INT4 解包、scale、重建、K RoPE、BF16 舍入、QK/PV。它主要实现了压缩存储，没有充分实现压缩域计算。
2. 大 batch 增加独立请求的并行度，不让这些请求共享 K/V。不能把 batch16 直接当成普通 attention 的 16 个 query rows 来获得免费 Tensor Core 复用。跨 loop 的 query 还有模型依赖，不能一起计算。
3. 5bit 是持久容量。每有效层每 head 每 token 单种 K/V 实际读取 32B index + 64B residual + 8B scale = 104B，即 6.5bit/元素；四 loop 各读同一索引。缓存可能使部分访问避免 HBM，6.5bit 同样不是实测 DRAM 流量。
4. 当前 pre-RoPE K 阻止普通位置无关 `Q × codebook` LUT。`qᵀR_pC=(R_pᵀq)ᵀC` 仍依赖 p；query 已做当前 token 的 RoPE 也不能消去相对位置。
5. V 端可在实数代数上先按 code 聚合概率，再乘中心。但是当前 `BF16(center + residual)` 不满足线性分解；group4/C256 的直方图成本也可能高于直接 gather。
6. 大 batch 仍使用固定32/64split网格，R2每CTA顺序执行两请求PV，设计需要重新组织。不过仅改split和scale广播不应被当成完整的下一代设计。

## 待比较的路线

- **兼容路线**：page 内按 head/group32 排列 packed 数据；协作向量加载；K/V 分别排布；按 page tile 流水化；有效任务队列取代大量空 CTA；保留精确重建和舍入规则。优先验证，以便判断现有格式还能挽回多少性能。
- **压缩域路线**：K 在 post-RoPE 空间训练/编码，计算中心 logit LUT 加低比特残差点积；V 先比较直接 gather 和概率按码字聚合。明确定义新的舍入 schema，保留同一个 softmax。最终目标是在质量允许时将残差降为2bit；4bit仅作为过渡控制。
- **非对称路线**：K 保留 VQ+，V 用普通 INT4/INT2，或保留 VQ+ 但直接 gather。codebook 是否值得存在，必须对比同预算普通标量量化，不能只比无残差 VQ 和 BF16。
- **Tensor Core 路线**：只在 tile 利用率账本和实际微基准支持时采用。当前 MHA 一 token decode 不保证有收益；prefill、多 query 验证、真实 GQA 是不同适用域。

## 当前证据与边界

原报告 B16/2K 的主 attention：BF16 19.940ms，VQ+ global 38.580ms，完整 forward 38.482ms 对57.620ms；这是已有 profiler 记录，本次没有重测。NCU 计数器权限受限，不能确认 memory-bound、shared bank conflict 或某种 stall。

下一步必须以同一真实 KV 快照比较：现有 VQ+、离线重建后运行强 BF16 attention 的诊断对照、同预算直接 INT4、兼容页级解码；然后才决定是否值得进入 post-RoPE 新格式。离线重建对照只隔离热路径，不把遗漏重建成本的速度当成部署收益。

完整代数、成本模型、文件改动与验收步骤见 [plan.md](plan.md)。原实现已完成10/10轮，本次不重置计数，也不执行第11轮。
