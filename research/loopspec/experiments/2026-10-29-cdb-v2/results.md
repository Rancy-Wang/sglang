# Ouro CDB 与 LoopSpec V2 验收记录（2026-10-29）

生产代码测试提交：`9bf97cdca4f4dc8e9c9508f059ee6bf06ee0f440`，分支 `loopspec-test`。本任务使用到第18/20轮。正式 `loopspec` 仍为用户此前批准的 bs=1 提交 `237f4d44aee9f9751234758c86d3154421c852e0`，未晋升本次实验代码。逐轮计划、命令和失败见 [record.md](record.md)。

## 实现归属与范围

这是用户 SGLang v0.5.22 fork 中新增的**普通 recurrent Scheduler CDB**，LoopSpec V2 作为其策略使用者。没有运行官方 LoopSpec 的独立 serving scheduler，也没有继承/复用 Eagle、Dflash 算法代码。固定结构参考为 vllm-rlt `6b164d3844d35ab8e753df1864f0c87dc180401a`；它不是上游 SGLang v0.5.22 已有的完整 CDB 功能。

- `scheduler_components/recurrent.py` 对应 phase 队列、take_phases、结果回填；每次只推进一个 core loop，返回原生事件循环重新准入。prefill 支持 token chunk 与 depth 两种边界；多个 decode 可处于不同 depth。
- `model_executor/recurrent_runner.py` 与 `mem_cache/timeline_pool.py` 提供通用 carried state、混合深度执行、depth 独立 KV 和 CUDA graph；普通 AR 与 LoopSpec 共用。
- `speculative/loopspec/{worker,utils,info,pipeline}.py` 负责 proposal、残差验证、请求私有分支、提交和回收。4/7 个执行行不是4/7个新用户请求；owner 始终是原生 Req，行号可以重排。只有完成 R4 的 slot 才转交原生请求，拒绝分支只释放仍属该 context 的私有 slot。
- 原生 V2 FutureMap、stream/event、GenerationBatchResult、HTTP/OpenAI API、停止与取消路径保留。`--speculative-algorithm loopspec --speculative-draft-depth 1`；没有新增 CDB 用户开关。

本次验证范围：Ouro-2.6B 非 Thinking、BF16、完整 `total_ut_steps=4`、CUDA/Triton、page_size=1、radix cache 关闭、TP=PP=DP=CP=1。Raven CDB、多 GPU 通信/跨 GPU bubble、其他 attention backend、共享 radix cache 等未获认证；配置门控明确拒绝不支持的 LoopSpec 组合。本报告不宣称所有 loop transformer 或原生 SGLang 的所有功能均已支持。

实测trace验证了loop边界准入：普通AR中Q2/Q3开始执行时Q1仅完成depth1；2096次core step中2048次合并了不同depth。LoopSpec profile中753次step有749次混合depth，峰值8个原生请求、32个分支行，说明不是把bs=1请求逐个串行执行。详见 `round-18/cdb-trace-proof.json`。

## 环境与数值标准

X5 RTX 5090，容器 `lmsysorg/sglang:v0.5.22-cu130`；Torch 2.14.1+cu130、Transformers 5.19.0、Triton 3.8.0。只读源码挂载 `/home/ruoxi/repo/sglang` → `/workspace/sglang`，导入 `/workspace/sglang/python/sglang/__init__.py`。模型与数据集没有修改。前8道AIME原始 `problem`，数据SHA256 `b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331`。

数值包络冻结为 [numerical_envelope.json](numerical_envelope.json) v3。独立普通 KV、整 R forward、关闭 CDB/LoopSpec 的32行控制复现了 depth0 最大RMSE `0.1017152667`，来自现有 fused RMSNorm 路径；原始 RMSNorm 控制为 `0.0674779192`。沿用独立控制最大值加25%余量并向上取整的规则，只将 depth0 hard RMSE上限从0.085固定为0.13。其他P99、hard max、最终logits、argmax翻转比例≤1.25%、参考margin>0.5禁止翻转等标准未变。原始失败报告保留，不追随候选结果继续放宽。

固定前缀诊断只强制最终已验证token选择；草拟、分支计算、混合深度、KV物化和剪枝都使用生产代码。每种8题×497位置，所有4个depth及最终logits均检查；每个请求均走过复用与重新开始。

| 模式 | 位置数 | argmax分歧 | 分歧最大参考margin | 结果 |
| --- | ---: | ---: | ---: | --- |
| d1=1，graph（round17，计算代码与最终相同） | 3976 | 21 | 0.25 | 通过 |
| d1=1，eager | 3976 | 25 | 0.25 | 通过 |
| d1=2，graph | 3976 | 18 | 0.25 | 通过 |
| d1/d2=1/2，graph | 3976 | 20 | 0.25 | 通过 |
| d1/d2=1/3，graph | 3976 | 29 | 0.25 | 通过 |

这不等于不同batch形状下逐token完全相同。自由生成的8题首次分歧均在原生参考margin=0.125的位置；旧/新bs=1也在index228出现分歧。数值检查使用相同前缀，避免用已分叉文本继续逐位置比较。每个最终10次性能组内部的token IDs和停止位置均完全一致。

## 服务与资源生命周期

CPU 9项契约通过，包括地址/depth隔离、重排、one-loop准入、分支接受/拒绝、relay过滤合并、真实penalizer在原batch释放后继续merge/filter；3组各100000次残差采样检验在5σ范围内。

GPU已通过：8请求交错长输出；d1=2 HTTP；双depth=(1,2)实际第二proposal接受；chunk32下两轮不等长输出与logprobs；prefill/decode两时刻取消与survivor；flush；零token；带top-k/top-p、frequency/presence/repetition penalty、min_new_tokens的8请求sampling；OpenAI completion。普通AR同一扩展生命周期也通过。原生 `--disable-overlap-schedule` 与 eager 组合下的完整扩展生命周期同样通过。

1200逻辑KV容量压力测试触发两次真实retraction，回收162/202 tokens，重新入队后8请求各128tokens完成；严格busy KV守恒检查开启，无泄漏/重复释放报告。采样服务测试验证接线与有限输出，分布正确性另由解析算法和Monte Carlo契约验证，不以相同seed序列要求替代分布检验。

## 无 profiler 配对性能

每组2次预热+10次测量，greedy、max_new_tokens=512、正常EOS；到达间隔为 `[0,10,25,45,75,120,200,320]` ms。性能测量不与其他GPU工作或重型trace分析重叠。下表为中位数，单位秒。

| 工作负载 | 改动前控制 | 完成普通CDB（round12） | 最终AR（round18） | 最终LoopSpec（round18） |
| --- | ---: | ---: | ---: | ---: |
| 第1题LoopSpec bs=1 | 9.842 | — | — | **6.529** |
| 8请求整组makespan | 22.003（AR） | 18.532 | **18.541** | **13.532** |

bs=1 E2E下降33.66%。最终普通AR相比round12仅增加0.046%，每请求增加0.0459%–0.0476%，满足≤2%标准；相对旧普通AR，7个等长请求E2E下降15.84%–16.25%。整体收益包含fused RMSNorm、图与metadata改进，不能全部归因于CDB调度。

| 题目 | 旧AR E2E | 最终AR E2E | LoopSpec E2E | AR TTFT | LoopSpec TTFT | AR/LoopSpec输出token |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 21.932 | 18.367 | 12.753 | 0.033 | 0.041 | 512/512 |
| 2 | 21.922 | 18.396 | 13.279 | 0.105 | 0.175 | 512/512 |
| 3 | 21.907 | 18.381 | 13.507 | 0.091 | 0.160 | 512/512 |
| 4 | 21.888 | 18.397 | 12.124 | 0.143 | 0.281 | 512/512 |
| 5 | 21.857 | 18.367 | 11.682 | 0.113 | 0.318 | 512/512 |
| 6 | 21.813 | 18.358 | 12.778 | 0.134 | 0.318 | 512/512 |
| 7 | 19.134 | 18.168 | 12.292 | 0.128 | 0.310 | **508/453** |
| 8 | 21.683 | 18.221 | 12.888 | 0.080 | 0.240 | 512/512 |

第7题旧AR也是453tokens，而普通CDB AR为508；EOS差异来自低margin分叉。该行不是等计算量加速比。其他7题LoopSpec相对最终CDB AR E2E下降26.52%–36.40%。**TTFT并非全部改善**：LoopSpec多请求TTFT高于普通CDB AR，当前验收以E2E≤2%为门槛，不能声称所有阶段都更快。

## bs=1 精细 profile

同一第1题、draft depth1、R4、512tokens。首token来自prefill，其余511个输出按开始depth分组。新实现depth0有54个输出、各新跑4个loop；depth3有457个输出、各新跑1个loop。没有depth1/2样本，不填造估计值。对应673次core推进（旧版676次）。

下表为**每个输出内同名阶段GPU stream interval之和的中位数**，单位ms。包含profiler开销；嵌套区间不能再相加，CPU提交时间不等于GPU忙碌时间。旧/新scope划分也有差异。

| 阶段 | 旧depth0 | 新depth0 | 旧depth3 | 新depth3 |
| --- | ---: | ---: | ---: | ---: |
| core | 55.737 | **37.579** | 15.461 | **9.364** |
| 准备行（新包含prepare） | 0.995 | 2.149 | 0.264 | 0.647 |
| 旧pre单列 | 0.291 | 包含于准备 | 0.074 | 包含于准备 |
| post/readout | 0.726 | 1.037 | 0.182 | 0.257 |
| proposal/verify（旧分列） | 1.372/0.443 | decision合计2.155 | 0.374/0.437 | decision合计1.161 |
| commit | 0.047 | 0.047 | 0.046 | 0.047 |
| retain | 0.032 | 0.038 | 0.030 | 0.033 |

新每输出CPU区间中位数为44.224ms（4loop）/11.989ms（1loop），旧为62.604/17.992ms。profile E2E从12.131s降到8.279s；GPU kernel忙碌并集从8051.910ms降至5173.177ms。准备、读出和decision仍有局部开销增加，是后续优化点；当前整体E2E满足本次标准。

8请求profile E2E整组19.397s、GPU kernel忙碌并集7782.143ms；4029个decode输出中depth0为454，depth3为3575。多请求共享step和kernel必须按共享批次看，不能将各请求关联到的共享GPU时间累加成总耗时。

## 结果与复现位置

X5根目录：`/home/ruoxi/repo/test/results/loopspec/2026-10-29-cdb-v2/`。

| 项目 | 相对目录/文件 |
| --- | --- |
| 旧bs=1精细profile | `round-2/loopspec-b1-profile/` |
| 旧普通AR交错profile | `round-2/ar-b8-profile/` |
| 普通CDB精细profile | `round-12/cdb-b8-profile/` |
| 旧bs=1十次控制 | `round-13/ls-b1-control/` |
| 独立原生数值校准 | `round-16/native32/`、`round-16/native32-fused/` |
| 新bs=1/8请求profile | `round-18/ls-b1-profile/`、`round-18/ls-b8-profile/` |
| 最终十次性能 | `round-18/{ls-b1,ls-b8,ar-b8}-benchmark/` |
| 配对汇总 | `round-18/paired-performance.json`、`ls-performance.json` |
| 数值/生命周期/低容量 | `round-17/ls-graph-full/`、`round-18/ls-*-full/`、`*-lifecycle/`、`ls-pressure/` |

每个profile目录含原生Chrome trace、逐step recurrent JSONL、`summary.json`、逐输出 `output-paths.json` 与 `shared-batches.json`；性能目录含完整manifest、命令、GPU快照、每次请求token/停止/时间及退出码。大日志均留X5，没有进入Git。

主要命令（容器中，工作目录 `/workspace/sglang`；输出目录必须新建）：

```bash
python research/loopspec/experiments/2026-10-29-cdb-v2/profile_workload.py \
  --algorithm loopspec --depths 1 --first 8 --output /results/<new-directory>
# bs=1加 --first 1；profile加 --profile --repeat 1；AR去掉algorithm参数。
python research/loopspec/experiments/2026-10-29-cdb-v2/verify_loopspec_cdb.py \
  --reference /results/round-5/native-joint-long \
  --tokens /results/round-2/ar-reference --graph --depths 1 2 \
  --output /results/<new-numerical-directory>
python research/loopspec/experiments/2026-10-29-cdb-v2/analyze_profile.py \
  /results/<profile-directory>
```
