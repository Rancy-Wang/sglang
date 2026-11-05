> 历史归档（2026-11-06）：DAS-Wave 已按用户要求撤除；本文描述撤除前的实现和结果，不代表当前可用算法。DFlash/LoopSpec 公共工具与修复保留。见 [撤除记录](../2026-11-06-remove-das-wave/record.md)。

# DAS-Wave V2 与 Ouro DFlash 实施结果

2026-11-05。当前实验提交：`da86ebecb1bccfa6aea1e73812e38cefa5c2a05a`，分支 `loopspec-test`，已同步 X5。已用 **10/10** 轮（完整主矩阵与45个Profile窗口已运行完毕）。原先授权的正式多 batch 晋升提交为 `82a851f6fe12c6e17354054fd7b160594f203b66`，标题 `[update] Support LoopSpec for bs>1`；本次 DAS-Wave 仍只在实验分支。

## 实现

- 新增独立 `--speculative-algorithm DAS_WAVE`，接入原生 V2/CDB。多请求的活跃位置合批执行，共享投影/attention launch。
- draft 读取已物化前缀的最终循环 KV，活跃后缀使用私有 KV；跨轮保留 hidden。Ouro 没有独立 input injection，这一 carry 适配的提议质量仍需完整实验评估。
- target verifier 从 embedding 独立执行完整 R4，不接收 carried draft hidden。接受前缀后转移 target KV，拒绝后缀释放，全接受不增加额外 bonus。
- CUDA Graph 分别覆盖 draft 更新加 readout，以及完整 R4 verify 加 readout；图输出在共享池复用前拷贝保存。图执行与合批路径已接入；实测DAS-Wave仅为AR的0.81–0.86×，未达到加速目标，不能将“接入高性能路径”等同于实际加速。
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

固定prefix探针使用重新prefill的同一前缀，并检查了选定物理层的KV。它支持“不同计算形状引入BF16数值差异”的解释，但不替代失败的自由生成验收。AR自身在graph/eager下也出现长序列分歧。没有通过放宽阈值将这些失败改标为通过。每题最多1024输出的AIME全30题已在bs1/2/4/8及受容量限制的并发上限16完成；默认bs16显存压力测试出现流式IDs失败及graph OOM，失败保留，未改标通过。更长自然解题、sampling和carry/deep-KV消融未执行。

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

## 全30题与 Profile

完整结果见 [benchmark-report.md](benchmark-report.md)，25组CSV与图表在 [results/](results/)。每配置30题，共750请求/768000 tokens；每题最多生成1024 tokens，不额外缩短总上下文，所有方法开启CUDA Graph。最终有效组无retraction，所有题都触及1024上限，不能称自然解题完成。bs1全部输出未解析到最终boxed答案。

| 并发上限 | DFlash / AR | LoopSpec D1 / AR | LoopSpec D1/3 / AR | DAS-Wave / AR |
| --- | --- | --- | --- | --- |
| 1 | 1.72× | 2.33× | 2.49× | 0.81× |
| 2 | 2.03× | 2.29× | 2.47× | 0.85× |
| 4 | 2.06× | 2.04× | 2.20× | 0.86× |
| 8 | 1.54× | 1.67× | 1.70× | 0.86× |
| 16（实际峰值9–10） | 1.79× | 1.70× | 1.72× | 0.84× |

这些是单pass吞吐比；严格AR输出等价未通过，不称已证明的无损加速。B16因32GB容量约束，五方法统一改为12000-token KV池与完整预算预留；其他B为13312。B1/16在GPU0、B2/4/8在GPU1，同B同卡配对，但跨B包含资源和调度差异。原始失败与资源限制详见报告。

bs1无Profile：AR每token32.755ms；DFlash平均完整draft+verify周期42.797ms；DAS-Wave78.007ms。LoopSpec D1/D1+3平均每输出13.999/13.118ms；深度条件周期与prepare/commit/forward/attention拆分见 [独立Profile报告](profile-report.md)。Profile使用q0/2/20、各自轨迹0/50/90%位置、32输出，共45窗口，采集完成；采集耗时不能替代无Profile主测试数据。

公开AIME25题面来自 [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025)，冻结版本和hash见 [数据清单](results/aime25-frozen-manifest.json)。旧LaTeX控制字符损坏文件保留而未使用。大型trace、完整tokens与日志留在X5 `/home/ruoxi/experiments/loopspec/2026-11-05-das-wave/`，不进入Git。

已达到：原生V2/多请求/graph执行、DFlash正确推理导出及规范替换、限定输出的完整主矩阵、独立Profile采集。未达到：严格greedy AR等价、DAS-Wave速度提升；未执行：sampling、自然完成AIME、carry/deep-KV消融与完整参数搜索。10/10代码轮次已用尽，未进行第11轮，也未自动将新DAS-Wave晋升到正式分支。
