# 统一 staged Drop / Reposition

`PLAN-CS-20261107-R4` 新增 staged 的 PD 支持，见文末。下文 R2 的版本、限制和
数值结果保留为历史记录；mini Drop 已由后续 R3 修复，R4 使用修复后的参考。

批准计划：`PLAN-CS-20261107-R2`。基线为 `system-test` 的
`2997bc2ba466d047a4aaa8f6af4ae07fa60492f7`。调度与所有权验证已通过；
**BCP 数值门禁尚未通过，不能据此宣布迁移验收通过。**

## 执行与所有权

启动参数为 `--context-execution-mode occurrence|staged`，默认 `occurrence`。
scheduler 显式向有 `ContextProgram` 的请求传递模式；普通请求没有 staged 状态。
定义见 `python/sglang/srt/arg_groups/fields/schedule.py:26-37` 的 `Schedule`；
传递点见 `python/sglang/srt/managers/scheduler.py:2794-2795`；
初始化见 `python/sglang/srt/managers/schedule_batch.py:1025-1026` 的 `Req`。

完整历史只生成一份模板 token/provenance 和不可变 `ContextProgram`。
extended Radix 返回实际 raw-token 匹配边界、源位置及 Full/SWA residency。
既有 `Req.prepare_context_recovery` 闭合必要缺页依赖，保留可复用区间；
`StagePlan` 将待计算区间按 Drop、有效 Reposition 和 native chunk 预算切段。
没有 0.95 命中率阈值、前端探测请求或重复消息模板 warmup。
命中消息中间时从实际 token 边界继续，完全命中仍保留原生生成所需的最后一个 query。

源码：`python/sglang/srt/managers/schedule_batch.py:1581-1669`
（`prepare_context_recovery`）、`python/sglang/srt/context_system/staged.py:16-38`
（`StagePlan`）、`python/sglang/srt/managers/schedule_policy.py:948-983`
（`PrefillAdder._fit_context_admission`）。本轮复用原恢复规划器。

每个非最终阶段只物化该阶段的位置版本。Drop 删除不可再见的 KV，不重编号幸存绝对位置；
显式 Reposition 才变换位置。下一阶段根据实际保留位置决定是否复制、旋转，
不能把前一阶段的 terminal 槽误当作下一阶段目标版本。末尾没有 query 的事件
在最后一次 forward 内完成，不插入虚构 token。中间阶段沿用 native chunk 的结果处理，
不提交采样输出。

源码：`python/sglang/srt/context_system/occurrence.py:250-310`
（`compile_occurrence_window`）、`:457-469`（`plan_occurrence_materialization`）、
`:682-698`（`OccurrenceState.plan`）；
`python/sglang/srt/managers/schedule_batch.py:1759-1776`（`plan_context_prefill`）。

中间版本由请求私有持有，保留原匹配分支的锁，不用完整程序的最终位置键发布。
已有 occurrence COW、Full/SWA 所有权及 completion barrier 负责复制和释放。
中间 Drop 后不再需要的私有页可以释放；最终发布仍须满足既有 sparse Drop proof。
中途终止禁用插入并走原生清理；结束后清除 stage plan。
因此其他请求不能在该请求完成前复用其私有阶段结果。

源码：`python/sglang/srt/context_system/staged.py:41-46`
（`has_unpublished_stage`）；`python/sglang/srt/mem_cache/unified_radix_cache.py:1055-1089`
（`cache_finished_req`）及 `:1249-1262`（`cache_unfinished_req`）；
`python/sglang/srt/context_system/occurrence.py:701-832`（`OccurrenceState.advance`）。

本轮限定文本 Full Attention、GPT-OSS Full/SWA、page size 1、普通自回归调度。
staged Insert、hybrid Linear attention 和 PD 显式拒绝；其他限制沿用既有 capability 检查。
源码为 `python/sglang/srt/context_system/capabilities.py:10-29` 的
`validate_context_config`。CUDA Graph 和 overlap 未关闭。模型实验为 TP1，
不能声称多卡模型验证已经通过。

## 验证方法

`test/registered/context_system/test_staged_plan.py:28-174` 使用独立可见性模拟器，
逐 query 比较有序 `(raw token, position)`，立即回收并复用物理槽，覆盖冷缓存、
消息中间命中、末尾命中、连续 Reposition、缺页区间和末尾无 query 事件。

`test/registered/context_system/test_staged_runtime.py:22-244` 检查真实 CLI、native admission、
中间发布保护和 capability 拒绝；使用原生 Full/SWA 分配器，在阶段间施加 eviction，
分别验证冷启动、借用缓存、阶段中止、最终发布后的页及锁归还。

`test/registered/context_system/mini_staged_oracle.py` 的 `main` 调用 mini 的真实
Drop 分消息 warmup、Reposition open/step/ack、模型和 attention 路径。
采集器过滤 `is_warmup` 和非 committed sample，验证固定 token 输出及 logits 行数，
将内部 warmup 与正式记录分开。没有替换模型 attention，也没有修改 mini 源码。
mini 旧 Drop 参考本身仍包含探测及 0.95 分支；这不是新 sglang executor 的行为。

BCP 使用真实 case 778 的冻结历史和工具定义，每次正式请求生成 8 个 token。
验证无操作、Drop、Drop+Reposition、热命中、Retry 和连续 Reposition。
固定 token 对照先断言两端完整输入 token 一致；自由生成仅记录观察，
与固定 token logits 门禁分开。此实验不是 BCP 全量准确率或吞吐测试。

入口为 `test_staged_runtime.py::test_bcp_staged_reference`，沿用既有 BCP 的
max/mean/p99 阈值并补充逐 token max/mean 检查。每条复用路径与具有相同缓存历史的
mini 记录配对，hot 不再误用 cold 参考；收集全部数值失败后才判定失败。
各上限为同次无操作基线的 2 倍与固定下限的最大值：
max 0.125、mean 0.02、p99 0.0625。阈值没有为 staged 放宽。

## 已完成结果

| 版本 / 检查 | 结果 |
| --- | --- |
| `3434db4b7f`，本地阶段/恢复/所有权 CPU | 20 passed，6 GPU-only skipped |
| `eda7f7bd22`，BUS-2 阶段、原生 admission、既有 ownership | 62 passed，7 skipped |
| `eda7f7bd22`，Qwen HTTP chunk=64、混合请求、Retry、空操作流式/非流式 | 7 passed，1 skipped，1 deselected |
| `35d9f87f7a`，本地阶段/能力检查 | 8 passed，17 Linux/GPU-only skipped |
| `421781bbfa`，BUS-2 原生阶段/Full/SWA 清理 | 19 passed，1 独立 BCP 入口 skipped |
| `421781bbfa`，BUS-2 HTTP 容量拒绝后的恢复 | 1 passed；503 后可完成新的 90-token 请求 |
| `fed78ce3be`，Qwen / GPT-OSS 完整配对 BCP | 两者数值 FAIL；所有输入、固定输出、形状检查通过 |
| `46d20bd0d2`，本地诊断开关验证 | 3 passed，17 Linux/GPU-only skipped |

本地使用已有 `myenv` Python、工作区既有 TVM FFI/ninja，没有修改仓库依赖。
BUS-2 使用 `/share/wangruoxi/local/akv/activate-bus2.sh`，PyTorch 2.13.0+cu130；
pytest 仅安装到任务目录 `deps`。mini 使用 `minisgl-gpt-oss-r4` 环境、
PyTorch 2.9.1+cu128；实际 HEAD 为 `fb248835b908da925d14f607748536483e5fac54`。

Qwen3-0.6B、BF16、TP1、输入 10482 tokens 的 BCP：

| attention | mode | 无操作 max | Drop max / mean / p99 | 结论 |
| --- | --- | --- | --- | --- |
| Triton | staged | 0.625 | 2.4375 / 0.174954 / 0.714417 | FAIL |
| Triton | occurrence | 0.625 | 1.4375 / 0.164091 / 0.65625 | FAIL |
| FlashInfer | staged | 0.625 | 2.75 / 0.167223 / 0.71875 | FAIL |
| FlashInfer | occurrence | 0.625 | 2.8125 / 0.177388 / 0.75 | FAIL |

Triton staged Drop+Reposition cold/hot/Retry/连续的 max 分别为
0.8125 / 0.6875 / 0.5625 / 0.8125（hot 使用配对的 mini hot 参考）。
除 Drop 外，cold 和连续路径的第 6 个 token 均值为 0.116310，超过原门禁 0.083301。
因此不能仅依据 aggregate 通过宣布这些路径数值通过。
staged/occurrence 的无操作 logits 完全一致；固定 token 场景
两种 sglang 模式的 raw argmax 均为 8/8 相同，但 argmax 相同不能替代数值门禁。

staged 热命中实际只 prefill 1 token；Retry prefill 2002 tokens；连续 Reposition
prefill 3 tokens。其余分别由 cached、repos、Drop-skipped 计数解释，符合按实际
匹配边界续算。自由生成与 mini 不完全一致，保留原始观察。

GPT-OSS-20B、MXFP4、BF16 activation、TP1、输入 9671 tokens：无操作 max 1.3125；
Drop max/mean/p99 为 2.8125 / 0.185938 / 0.754883。
Drop+Reposition cold/hot/Retry/连续的 max 为 1.140625 / 1.125 / 0.96875 / 1.140625。
hot 的第 7 个 token max 0.75、mean 0.104148，分别超过 0.5 和 0.056746；
cold/Retry/连续通过原数值门禁。热命中 prefill 1 token，连续 prefill 2 tokens。
Retry 因 SWA 必要缺页恢复计算 9671 tokens，不能把 Radix 匹配当成全部 KV 驻留。

## 偏差定位：语义与数值须分开

**已确认 mini 旧 Drop staged 参考存在历史读集错误，不能把整套差值归为浮点误差。**
mini `python/minisgl/server/api_server.py:572-665` 的 `run_contextual_warmup`
先实际执行完整历史探测，再 warmup `messages[:1]` 到 `messages[:8]`，未在 fallback
末尾预填最后一条历史消息。正式请求按最终 Drop 状态压缩输入，导致尚未完成的历史
query 提前丢失未来才应删除的键。tokenizer 压缩点为
`python/minisgl/tokenizer/tokenize.py:1676-1690`。

case 778 的 Qwen 实测：Drop message 4 应在 raw 10479 生效；正式请求从 raw 8690
开始补算时，已经去掉 raw `[4488,7242)` 的 2754 个 token。该 query 应有 5841 个键，
实际只有 3087 个。raw token 对应的输入 id 和绝对位置均逐项一致，排除了模板和
位置编号不一致；这是历史可见性错误。最初完整探测也有错误读集，但物理槽追踪没有
证明其错误槽直接留到正式请求，因此不将“探测污染缓存”当作已确认的直接原因。

诊断开关 `CONTEXT_MINI_DIAGNOSTIC_FINAL_WARMUP=1` 仅在采集器中追加一次最后历史消息
warmup，原 probe、前缀 warmup、tokenizer、模型和 attention 内核均保留。
结果被标记为 `diagnostic-final-history-warmup`，正式 BCP 测试拒绝将其作为生产参考。
mini 源文件未修改。正式请求随后从 raw 10479 开始，缓存 4875 tokens，预期/实际键数
同为 4876，缺键数为零。

| Qwen Drop 配对 | max abs | mean abs | p99 abs |
| --- | ---: | ---: | ---: |
| 原 mini staged vs sglang staged | 2.4375 | 0.174954 | 0.714417 |
| 补末消息 warmup vs sglang staged | 0.6875 | 0.070619 | 0.25 |
| 补末消息 warmup vs mini mask | 0.625 | 0.065191 | 0.25 |
| mini mask vs sglang staged | 0.53125 | 0.064016 | 0.25 |

该干预确认 Qwen Drop 大偏差主要由旧参考的语义错误造成；剩余数值接近无操作误差
量级，但这不是对所有输入的理论误差界。GPT-OSS 相同干预后仍有残差，且自由生成在
第 5 个 token 分歧；只能比较前 5 行同前缀 logits，max 2.552246、mean 0.104686。
不能以 Qwen 的结果宣布 GPT-OSS 的剩余偏差已经解释。

Qwen mini Drop+Reposition 四种缓存路径的各阶段首尾 query，实际 raw 读集和位置均
与独立可见性规则一致；这项检查不是每层 GPU KV 内容证明。mini 自身 cold/hot 第 6
个 token 的均值变化为 0.098567，sglang 自身为 0.042387。GPT-OSS 自身 cold/hot
第 7 个 token max 0.78125、mean 0.098024。它们说明计算形状和复用路径需要单独
对照，不能只用无操作逐 token 阈值认定根因。正式失败记录和阈值均保留。

**GPT-OSS 无操作基线还包含既有 SWA 窗口语义差异，不是纯浮点校准。**
两端模型都将配置 128 转为左侧距离 127：mini
`python/minisgl/models/gpt_oss.py:42-50`，sglang
`python/sglang/srt/models/gpt_oss.py:129-130`。但 sglang Triton 原生 decode 将该值作为
总容量（`python/sglang/srt/layers/attention/triton_backend.py:554-560`、`:870-884`），
Context 路径按包含当前位置的距离规则读取 128 个键。BUS-2 的只读 forward hook
实测 none 为 127，Drop 为 128。原生控制沿用既有行为，并非本次 staged 变更新增。
“误差小于两倍无操作基线”仍记录为原门禁结果，不能据此宣布数学语义等价。

只读内核诊断使用 `serving_logits_probe.create_decode_read_probe` 和
`create_bounded_attention_probe`，GPT-OSS 全 24 层，TP1，page 1；为执行 Python hook
单独关闭 CUDA Graph，原 BCP 运行保留 graph/overlap。对照包括真实物理 Full/SWA
读索引、当前 K/V 写回一致性，以及相同 Q/K/V/sinks 的 FP32 dot/softmax/reduction。
容差为逐元素 `0.0078125 + 0.0078125 * abs(expected)`，与最终 logits 门禁独立。
这只能验证被采集的 decode 算子，不证明全部历史 KV 在生成时语义正确，也不代表
整个模型输出满足某个理论误差上限。overlap 产生的最后未提交 decode 也被 hook 记录。

`gpt20-read-v2` 共 1368 个 layer/decode 观察，物理读索引和当前 K/V 写回全部一致，
FP32 容差通过 1365 个。未通过的 3 个是 cold、Retry、连续路径重现的同一点：
layer 21、position 4461、Full Attention，max 0.030614、mean 0.001516。
不能写成“全部 attention 数值检查通过”。关闭 graph 的 8 组最终 logits 与原 graph
运行逐元素完全一致，说明该诊断复现了原异常，并未因关闭 graph 消除问题。

冷、热路径最终 prefill query 的 token/position 相同，位置 4460，批内 query 数分别
为 2 和 1。第 0 层进入 attention 前 Q 的 max 差值已经是 0.001953125、V 为
0.000001907349，K 相同，attention 输出 max 差值 0.015625。
第 7 个输出 token 对应 position 4466：前两层 Q/K/V/output 相同，第 2 层 Q/K/V
相同而 attention output 相差 0.03125，随后逐层放大。
因此热命中残差有计算形状导致数值传播的直接证据；它不能消除上文已证明的语义问题，
也不能替代整个模型误差上界。Qwen 剩余逐 token 超限与 GPT-OSS Drop 剩余残差
仍未完全归因。

分类：旧 mini Drop 及原生 SWA 差异为 `PRE_EXISTING_FAILURE`；把所有剩余偏差宣布为
可接受浮点误差属于 `UNSUPPORTED_CLAIM`。没有据此确认 staged 新增回归，但也没有
足够证据解除数值验收阻塞。此次只增加诊断采集、记录结果，不修改 mini 或原生 SWA
生产代码，不放宽验收阈值。

## 复现与产物

主机 `InfiniAI-BUS-2`，仓库 `/share/wangruoxi/repo/sglang`，通过 Git fast-forward 同步。
模型实验用物理 GPU 0（mini）、1（staged）、2（occurrence）；端口 28771/28772，
各模式日志和 Triton cache 分开。产物根目录 `/share/wangruoxi/runtime/staged-20261107-r2`。
Triton 初次 BCP 版本为 `35d9f87f7a`，FlashInfer 为 `421781bbfa`，完整配对为
`fed78ce3be`，诊断为 `46d20bd0d2`，这些版本生产代码相同。最后一个版本已 push；
BUS-2 直连 GitHub fetch 停滞后，通过该已推送提交的 Git bundle 执行 `git pull --ff-only`
同步到 `46d20bd0d2b86d3fc69628c6894f21f9cc6d2b1d`，没有复制覆盖远端源码。

```bash
source /share/wangruoxi/local/akv/activate-bus2.sh
cd /share/wangruoxi/repo/sglang
export PYTHONPATH=$PWD/python:/share/wangruoxi/runtime/staged-20261107-r2/deps
CUDA_VISIBLE_DEVICES=99 python -B -m pytest --tb=short -o addopts= \
  -p no:cacheprovider -q test/registered/context_system/test_staged_plan.py \
  test/registered/context_system/test_staged_runtime.py \
  test/registered/context_system/test_context_admission.py \
  test/registered/context_system/test_occurrence_ownership.py

export CONTEXT_SERVER_MODEL=/share/public/wangruoxi/models/Qwen3-0.6B
export CONTEXT_START_TIMEOUT=600 CONTEXT_CHUNK_SIZE=8192
export CONTEXT_MAX_LENGTH=24576 CONTEXT_KV_CAPACITY=24576
export CONTEXT_BCP_ORACLE=/share/wangruoxi/runtime/staged-20261107-r2/mini-qwen-formal.json
export CONTEXT_TRACE_DIR=/isolated/new-traces
export CONTEXT_SERVER_LOG=/isolated/new-server.log
export CONTEXT_SERVER_PORT=28771 CONTEXT_TEST_ATTENTION_BACKEND=triton
export CONTEXT_SERVER_EXTRA_ARGS='["--context-execution-mode","staged"]'
CUDA_VISIBLE_DEVICES=1 python -B -m pytest -s --tb=short -o addopts= \
  -p no:cacheprovider -q \
  test/registered/context_system/test_staged_runtime.py::test_bcp_staged_reference
```

参考采集需要 `CONTEXT_MINI_ROOT=/share/wangruoxi/repo/mini-sglang`、
`CONTEXT_BCP_FIXTURE=/share/public/wangruoxi/local/sglang-system-20260917/bcp778-native-tools.json`、
`CONTEXT_ORACLE_OUTPUT=/isolated/new-mini.json`，由 mini 环境运行
`test/registered/context_system/mini_staged_oracle.py`。
因果诊断在相同命令上设置 `CONTEXT_MINI_DIAGNOSTIC_FINAL_WARMUP=1` 并指定新输出路径。
GPT-OSS 内核诊断在上面的 BCP 命令中增加：

```bash
export CONTEXT_SERVER_MODEL=/share/public/wangruoxi/models/gpt-oss-20b
export CONTEXT_BCP_ORACLE=/share/wangruoxi/runtime/staged-20261107-r2/mini-gpt20-formal.json
export PYTHONPATH=$PWD/benchmark/context_system:$PYTHONPATH
export CONTEXT_BCP_MODE=fixed CONTEXT_BCP_LAYER_PROBE=1
export CONTEXT_TRACE_DIR=/isolated/new-read-traces
export CONTEXT_SERVER_LOG=/isolated/new-read-server.log
export CONTEXT_SERVER_EXTRA_ARGS='["--context-execution-mode","staged","--moe-runner-backend","triton_kernel","--disable-cuda-graph","--forward-hooks","[{\"name\":\"reads\",\"target_modules\":[\"*.self_attn.attn\"],\"hook_factory\":\"serving_logits_probe:create_decode_read_probe\",\"config\":{\"model\":\"/share/public/wangruoxi/models/gpt-oss-20b\",\"layers\":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]}},{\"name\":\"layers\",\"target_modules\":[\"*.self_attn.attn\"],\"hook_factory\":\"serving_logits_probe:create_bounded_attention_probe\",\"config\":{\"rows\":1}}]"]'
CUDA_VISIBLE_DEVICES=1 python -B -m pytest -s --tb=short -o addopts= \
  -p no:cacheprovider -q \
  test/registered/context_system/test_staged_runtime.py::test_bcp_staged_reference
```

现有 `mini-qwen-formal.json(.pt)`、`qwen-{staged,occurrence}/comparison.json`、
`qwen-fi-{staged,occurrence}/comparison.json`、`qwen-diagnostic.json` 和各日志应保留。
最早的 `mini-qwen.json` 混入阶段内部采样，已弃用，不得用于验收。

诊断产物包括 `semantic-readset-diagnostic.json`、`final-warmup-causal-diagnostic.json`、
`gpt20-final-warmup-causal.json`、`mini-qwen-final-warmup-v2.json(.pt)`、
`gpt20-native-read-audit.json`、`gpt20-hot-layer-diagnostic.json`、`gpt20-read-v2/`、
`{qwen,gpt20}-paired-staged/gate-failures.json`；与生产参考分开保存。
`mini-qwen-final-warmup-diagnostic.json` 在远端同步完成前运行，未包含诊断干预，
不得作为干预证据；有效文件为 `mini-qwen-final-warmup-v2.json`。

仍需完成：解释并解决剩余数值门禁失败；多卡、真实
HTTP 中途 abort/retraction 和性能压力未由本轮结果覆盖。原生 allocator 中止测试
不能替代这些模型级验证。需要回退运行行为时显式选择默认 `occurrence`。


## R4：staged 的 PD 最终状态移交

批准计划 `PLAN-CS-20261107-R4`，起点 `9c156eb1d4`。P/D 都使用
`--context-execution-mode staged`，不增加另一种阶段协议。P 仍按 extended Radix 的
实际命中边界及必要恢复缺页执行 StagePlan；Drop 保留幸存绝对位置，Reposition 才
变换位置。普通请求与 occurrence 的发送路径保持原行为。

`python/sglang/srt/disaggregation/prefill.py` 的
`SchedulerDisaggregationPrefillMixin.send_kv_chunk` 对 staged 非最终 chunk 直接返回，
不推进发送游标。最终结果处理先完成既有 `copy_done` 和 Context completion，
再检查终态并发送。等待 bootstrap 的完成请求也复用同一个最终发送入口。
“一次移交”是提交一个最终状态，Mooncake 仍可拆分传输块。

`disaggregation/context_transfer.py::validate_staged_handoff` 只检查 CPU 元数据：
已完成的 raw 长度、最终 active token 对应的 terminal 行、实际保存的位置版本、
首次 decode 所需的 SWA 尾部驻留。最终 `ContextTransferPlan` 的事件签名、active
坐标、next position 和 usage 使用原有 v2 元数据；不传递阶段执行进度。
D 的 `allocate_context_destination` 清除 stage plan，建立最终布局。

P 在最终 forward 后保留请求私有的最终 KV，直到原生 inflight 传输退休时才发布或
释放。不能在发送前用 `cache_unfinished_req` 的 insert/rematch 将本次新算 KV
去重成旧缓存页：虽然两者可属于相同语义键，计算形状可能使数值不同，继而改变 D
实际接收的内容。原匹配分支的锁与请求私有页继续受既有退休机制保护。

失败处理沿用原生路径：接收失败禁止插入 D 缓存；D 发起中止时，尚可能被 P 写入的
目标页等待 drain 后才释放。P 的中间阶段由已有 unpublished-stage guard 禁止发布。
真实 rebootstrap 使用 `Req.build_rebootstrap_payload` 携带完整历史及已生成前缀，
由 P 按 staged 重算；D 重放已输出的边界 token，不重复提交。

范围为文本 Full Attention、GPT-OSS Full/SWA、page size 1、FP16/BF16 KV、Mooncake、
常规自回归。Insert、Linear attention、PP/CP/DCP、推测解码、KV offload/checksum 和
Mooncake staging buffer 仍保留原限制。`staged` 执行与 Mooncake staging buffer
是两种不同机制。D Radix 为已有可选功能，默认关闭；SWA + D Radix 的原生启动检查
仍要求 `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1`，即使当前环境提示该变量已弃用。

### 验证口径

`test_staged_pd.py` 以真实阶段规划及 occurrence 状态推进验证最终行、位置和 SWA，
并验证原生发送入口在非最终 chunk 不访问发送器或推进游标。
`test_staged_runtime.py::test_staged_native_ownership_cleanup` 额外覆盖最终发布延后、
借用缓存、阶段中止及 Full/SWA 页释放。

PD BCP 入口是 `test_bcp_pd_numeric.py`。参考为 mini R3 的真实生产 staged 输出；
hot、Retry、连续 Reposition 使用对应缓存历史的参考。可设置
`CONTEXT_PD_INTEGRATED_TRACES`，同时对比同一输入及固定输出前缀的一体部署 sglang
staged logits。两条对照轴独立保留 aggregate 和逐 token 门禁，不放宽原阈值，
所有超限写入 `gate-failures.json` 后统一失败。

D Radix 开启时，D 的缓存历史不能视为与一体部署相同：D 可保留上一请求的 KV，
P 则可能重算末尾 query。此路径与默认全量接收分别测试，不用 token 相同替代 logits
门禁，也不把所有差异自动判作可接受浮点误差。

远端使用 BUS-2 `/share/wangruoxi/repo/sglang`，GPU 0–3，独立端口和输出目录
`/share/wangruoxi/runtime/staged-pd-20261107-r4`。现有 akv 环境的 Mooncake 通过任务
`deps` 链接已有 `sglang-system-20260917` 的 0.3.13 安装，并仅为测试进程补充
`pd-drop-c8-20260919/rdma/lib`；未修改公共环境或远端源码。已 push 提交通过 Git bundle
执行 `git pull --ff-only` 同步。传输使用 `MC_FORCE_TCP=1`，不代表跨主机 RDMA 验证。

截至生产修正 `c4489721c1`，原生 CPU/所有权检查为 47 passed、2 个显式模型入口
skipped；本地 CPU 可运行子集为 18 passed。两项依赖原生 SRT 的本地测试因缺
`orjson` 及连带导入错误失败，已由远端原生测试覆盖并通过。

实测使用 Triton、BF16 activation，GPT-OSS 保留 MXFP4 权重。
TP1、chunk=8192、D Radix 关闭时，与一体部署 staged 配对的 none、Drop、
Drop+Reposition cold、Retry、连续 Reposition 及 none Retry source，两个模型均
逐元素相同。hot 仍有差异：

| 模型 | hot max / mean / p99 | 前两行（P 首 token 与 D 首次 decode） |
| --- | --- | --- |
| Qwen3-0.6B | 0.375 / 0.0347066 / 0.15625 | 逐元素一致 |
| GPT-OSS-20B | 0.25 / 0.0156667 / 0.078125 | 逐元素一致 |

修改前，即使 D Radix 关闭，hot 也分别有 max 0.44140625 / 0.78125，首次 decode
已不同。延后 P 最终缓存发布消除了首次 decode 的差异，但不能据此声称所有后续
残差已证明为可接受误差。D Radix 开启的早期 Qwen 实验还观察到全部 4878 个 active
页复用、hot 的 D logits 与 cold 逐元素相同；这与 P 本次重算的计算历史不同。
此观察保留在 `qwen-tp1`，不冒充最终修正后的结果。

相对 mini 的既有数值失败继续保留，两模型的 BCP 测试均报告数值 FAIL；同次
流式/非流式 usage roundtrip 均 PASS。测试没有通过放宽门禁得到验收结论。
Qwen 真实 HTTP 生命周期测试 PASS，覆盖已进入 stage 后中止、decode 中止、
一次真实 retract/rebootstrap、后续恢复以及独立 P/D 热缓存。512-token 重算场景
返回 512 个输出，usage 为 actual_prefill_tokens=530、actual_decode_tokens=511。
这些检查证明已覆盖请求的生命周期，不是全部输入的数值误差界。

GPT-OSS Full/SWA 的同一 HTTP 生命周期用例也 PASS，开启 D Radix。容量用例最终
使用 raw 长度 100、Drop@80、Reposition@99，P 池 110、D 池 640；旧页与旋转后新页
的共存需求触发 P 503，D 在同次请求返回 500，随后 90-token 请求成功生成 8 tokens。
初次 140-token 用例在 HTTP 层先返回 400，D 等待超时；该无效用例的日志保留在
`qwen-capacity`，有效结果在 `qwen-capacity-final`，不把前者算作通过。

TP2 使用 P GPU 0,1、D GPU 2,3，并分别采集同 TP2 的一体部署参考。两模型的
none、Drop、Drop+Reposition cold、Retry、连续 Reposition 和 none Retry source
与对应参考逐元素相同；hot 结果如下。两行的 D Radix 配置不同，不能用它们比较模型
本身的误差大小，也不构成 D Radix 开关与模型、TP 的完整交叉矩阵。

| 模型 / D Radix | hot max / mean / p99 | 已保存的门禁超限数（mini + integrated） |
| --- | --- | --- |
| Qwen3-0.6B / 关闭 | 0.875 / 0.0716147 / 0.40625 | 17 |
| GPT-OSS-20B / 开启 | 0.203125 / 0.0172606 / 0.0703125 | 20 |

Qwen TP2 的前两行仍逐元素相同。GPT-OSS TP2 仅 P 首 token 行相同，D hot 复用
4334 个 Full 页，末尾 127 个页重新接收；这符合 `ContextDecodeReuse.build` 对
SWA 尾部不借用共享目标页的约束（`context_transfer.py:190-216`）。其 Retry 和连续
Reposition 的 D 侧记录均为 reused=0、copied=0、missing=4461；完整接收后的
logits 与一体部署相同，但未覆盖预期的复制复用路径。现有测试在
`test_bcp_pd_numeric.py:392` 的 `copied > 0` 断言处失败，未删除该断言；数值失败
列表此前已写入 `gate-failures.json`。这不能报告为 SWA 复制复用验证通过，零命中的
具体原因仍需进一步定位。两组 TP2 的独立 usage roundtrip 均 PASS（各 8 条记录）。

最终生产代码为 `c4489721c1`；TP2、GPT 生命周期及有效容量测试 checkout 为
`9a6d534889`，其后续差异仅为测试。mini 参考来自 R3 修正 `75056003`。所有 BCP
结果仅覆盖 case 778 的冻结历史、工具及 8 个输出 token。当前结论为：终态 PD
移交及上述生命周期用例已验证，**完整数值验收仍为 FAIL**。mini 对照轴保留既有
失败；PD hot 残差尚未证明为有界浮点误差，不能排除剩余计算语义问题。

未覆盖跨主机 RDMA、P/D 不同 TP 度数、全量 BCP 或吞吐收益。GPT-OSS SWA 的
Retry/连续复制复用路径亦未由本次真实模型运行覆盖。运行数据摘要为 R4 目录下
`final-summary.json`，原始比较与日志分别保存在 `qwen-final-tp1`、`gpt20-final-tp1`、
`qwen-pd-tp2`、`gpt20-pd-tp2` 以及对应同级 `.log` 文件；保留失败证据，不以生成
token 相同代替数值验收。

### R4 复现命令

以下均在 BUS-2 的已同步 checkout 执行；启动器拥有并清理自己创建的 P/D 进程。
先核对 GPU 空闲、使用新的输出目录，避免覆盖已存证据。

```bash
source /share/wangruoxi/local/akv/activate-bus2.sh
cd /share/wangruoxi/repo/sglang
export LD_LIBRARY_PATH=/share/wangruoxi/runtime/pd-drop-c8-20260919/rdma/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PWD/python:/share/wangruoxi/runtime/staged-pd-20261107-r4/deps:/share/wangruoxi/runtime/staged-20261107-r2/deps
CUDA_VISIBLE_DEVICES=99 python -B -m pytest -o addopts= -p no:cacheprovider --tb=short -q \
  test/registered/context_system/test_staged_pd.py \
  test/registered/context_system/test_staged_plan.py \
  test/registered/context_system/test_context_transfer.py \
  test/registered/context_system/test_staged_runtime.py

export MC_FORCE_TCP=1 SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export CONTEXT_START_TIMEOUT=600 CONTEXT_CHUNK_SIZE=8192 CONTEXT_MAX_LENGTH=24576
export CONTEXT_P_KV=24576 CONTEXT_D_KV=24576 CONTEXT_TEST_ATTENTION_BACKEND=triton
export CONTEXT_SERVER_MODEL=/share/public/wangruoxi/models/Qwen3-0.6B
export CONTEXT_PD_BCP_ORACLE=/share/wangruoxi/runtime/staged-20261107-r3/mini-qwen.json
export CONTEXT_PD_INTEGRATED_TRACES=/share/wangruoxi/runtime/staged-20261107-r3/qwen-staged
export CONTEXT_TRACE_DIR=/isolated/new-pd-traces
export CONTEXT_P_GPU=0 CONTEXT_D_GPU=1 CONTEXT_PD_PORT=29241 CONTEXT_PD_RADIX=0
export CONTEXT_SERVER_EXTRA_ARGS='["--context-execution-mode","staged"]'
python -B -m pytest -o addopts= -p no:cacheprovider -s --tb=short -q \
  test/registered/context_system/test_bcp_pd_numeric.py
```

GPT-OSS 改用 `gpt-oss-20b`、`mini-gpt20.json`、`gpt20-staged`，并在 extra args 中
追加 `"--moe-runner-backend","triton_kernel"`。TP2 设置 `CONTEXT_TEST_TP=2`、
`CONTEXT_P_GPU=0,1`、`CONTEXT_D_GPU=2,3`，integrated traces 必须改用同 TP2 的参考。
该参考由 `test_staged_runtime.py::test_bcp_staged_reference` 采集，使用相同模型、
chunk、KV capacity、TP2 和 staged，`CONTEXT_BCP_MODE=fixed`；R4 目录中已有
`qwen-integrated-tp2` 与 `gpt20-integrated-tp2`。

生命周期测试使用同一 PD 启动器，设置 `CONTEXT_PD_LIFECYCLE=1`、
`CONTEXT_PD_SMOKE=1`、`CONTEXT_CHUNK_SIZE=32`、`CONTEXT_PD_RADIX=1`，extra args
增加 `"--log-level","debug"`。只选
`test_staged_pd.py::test_http_rebootstrap_abort_and_recovery`；它通过原生日志确认
进入中间阶段后才中止，随后验证真实 rebootstrap、decode 中止及冷热恢复。
容量测试设置 `CONTEXT_PD_CAPACITY_PRESSURE=1`、`CONTEXT_P_KV=110`、
`CONTEXT_D_KV=640`、chunk=32，选择 `test_serving_capacity_pd.py`。各次使用独立输出
目录与端口，均不设置关闭 graph/overlap 的参数。

R4 源码证据：`python/sglang/srt/disaggregation/prefill.py:850-856`
（最终发布延后）、`:1328-1349`（发送门禁），
`python/sglang/srt/disaggregation/context_transfer.py:98-119`（终态校验）、
`:277-280`（D 阶段状态），以及已有的
`python/sglang/srt/disaggregation/decode.py:2549-2568`（失败后的目标页释放）。
