# DAS-Wave V2 与 Ouro DFlash 实施结果

2026-11-05。运行时验证提交：`fa023257486d319c9be18c0673e5ff56664f6447`，分支 `loopspec-test`，已同步 X5。已用 **9/10** 轮（第9轮完整负载验证进行中）。原先授权的正式多 batch 晋升提交为 `82a851f6fe12c6e17354054fd7b160594f203b66`，标题 `[update] Support LoopSpec for bs>1`；本次 DAS-Wave 仍只在实验分支。

## 实现

- 新增独立 `--speculative-algorithm DAS_WAVE`，接入原生 V2/CDB。多请求的活跃位置合批执行，共享投影/attention launch。
- draft 读取已物化前缀的最终循环 KV，活跃后缀使用私有 KV；跨轮保留 hidden。Ouro 没有独立 input injection，这一 carry 适配的提议质量仍需完整实验评估。
- target verifier 从 embedding 独立执行完整 R4，不接收 carried draft hidden。接受前缀后转移 target KV，拒绝后缀释放，全接受不增加额外 bonus。
- CUDA Graph 分别覆盖 draft 更新加 readout，以及完整 R4 verify 加 readout；图输出在共享池复用前拷贝保存。高性能执行路径已经接入，但尚无正式加速比结论。
- 当前限定 greedy；sampling、logprob 和部分 logit 调整明确拒绝，不能宣称支持。

## 已执行验证

| 项目 | 结果 |
| --- | --- |
| CPU 算法/HTTP 契约 | 6项通过 |
| Wave bs1 eager，2×64 tokens | 与 AR IDs/text/stop 全部相同 |
| Wave eager overlap，bs1/2/4/8/16 | 62请求完成，严格匹配24；分别2/2、4/4、6/8、4/16、8/32 |
| Wave bs1，128 tokens | eager与graph各两次均未严格匹配AR；graph首次分歧为第105个token |
| 固定prefix探针 | 24组、440位置；同shape graph/eager logits全部逐位相同；跨shape有1处argmax翻转，logits最大绝对差0.5 |
| Wave 生命周期 | eager/graph、overlap均通过：零/短输出、不同完成时刻、stop token、取消、survivor、flush、复用 |
| DFlash bs1 | eager和graph各两次64 tokens均严格匹配AR |
| DFlash graph 生命周期 | bs8、零/短输出、stop、取消与复用通过 |

固定prefix探针使用重新prefill的同一前缀，并检查了选定物理层的KV。它支持“不同计算形状引入BF16数值差异”的解释，但不替代失败的自由生成验收。AR自身在graph/eager下也出现长序列分歧。没有通过放宽阈值将这些失败改标为通过。长前缀完整矩阵、强制显存压力/retraction及完整AIME负载仍未验收。

详见 [validation-summary.json](results/validation-summary.json)、[固定前缀探针](results/wave-probe.json)。

## DFlash checkpoint

X5 规范推理目录：`/home/ruoxi/models/Ouro-2.6B-Thinking-DFlash`。2026-11-05按用户明确要求替换，原训练态目录备份为`Ouro-2.6B-Thinking-DFlash-training-backup-20261105T115931Z`，替换前后hash均核对通过。新路径已实际完成原生DFLASH graph生成，见[替换记录](results/canonical-dflash-replacement.json)。

- 使用用户给定训练脚本对应的 SpecForge 配置和 step10000 checkpoint；5层、block size16、mask ID16、target layers `[1,12,23,34,45]`。
- 严格 materialize/load，导出后 **58/58 BF16 tensors逐元素一致**；新目录所有artifact再次按hash核对。原训练文件不变。
- 经X5访问A10，核对到训练wrapper每次循环覆盖同一个物理层hook，因此最终特征是**第四次循环的物理层输出，循环末norm之前**。原生Ouro按此语义捕获，并显式返回aux以支持prefill body图。
- A10与X5完整target权重SHA256相同：`c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080`。config/tokenizer配置也匹配。
- native capture与对应最终层hook逐位一致；capture开关不改变target logits。4个HF对照检查点argmax相同，feature/logit数值误差保留；未声称HF/native逐位一致。
- A10与X5的HF模型代码存在causal-mask接口适配差异，诊断显式传入完整causal mask，并补回历史`pad_token_id=None`默认值，未改原模型文件。

证据：[导出清单](results/dflash-export-manifest.json)、[特征对照](results/dflash-features.json)。

## 尚未执行

**AIME25全30题、加速比、平均acceptance rate和详细profile均尚未执行，不能把smoke测试代替这些结果。**

用户已明确要求继续全30题与Profile。当前保留原R4/BF16、完整披露数值分歧；第9轮运行器已完成五种配置各2×64 tokens短负载，正式全题和独立Profile进行准备/验证，未完成前不报告其结果。

公开AIME25题面已冻结在X5 `aime25-matharena.jsonl`，共30题；旧文件存在LaTeX控制字符错误，因此原文件保留、正式输入另存。数据来源为 [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)，版本和hash见 [数据清单](results/aime25-frozen-manifest.json)。数据不进入Git。

完整原始证据位于X5 `/home/ruoxi/experiments/loopspec/2026-11-05-das-wave/`。当前GPU0用于第9轮实验；X5原会话已被工具回收，确认旧ControlMaster不存在后重建持久连接，session 84278，shell PID880167。
