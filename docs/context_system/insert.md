# Execution-order Insert

实现计划：`PLAN-CS-20261103-R1`、`PLAN-CS-20261103-R2`，以及 `PLAN-CS-20261103-R3` 的已批准非 Linear 部分。本实现保留 `drop_message`，允许重复 Drop；
SWA 先沿用原位置窗口条件，再选 logical 最后 W 项（W=原生距离窗口+1，包含 query）。Qwen3.8/Qwen3.5 的 Linear/Full hybrid 模型暂不支持 Insert，
请求会返回格式/能力错误；其原有 Drop/Reposition 路径保留。

## 接口与时机

`messages` 按最终 logical 顺序给出，但仍包含被 Drop 的历史消息，供无状态恢复。
普通消息的 index 要么全部省略，要么全部明确写为 `0,1,2,...`；负编号不占普通编号。
插入消息必须显式使用 `-1,-2,...`，不能重复或跳号。字段省略或 null 都表示自动普通编号。
`insert` 的 value 必须是负整数，每个负编号恰好有一个触发器。JSON 对象的 key 使用规范十进制字符串。

```json
{
  "messages": [
    {"role": "system", "content": "System instructions"},
    {"role": "user", "content": "message 1"},
    {"role": "assistant", "content": "message 2"},
    {"role": "user", "content": "message 3"},
    {"role": "assistant", "content": "message 4"},
    {"role": "user", "content": "message 5"},
    {"index": -1, "role": "user", "content": "summary of 1–5"},
    {"role": "assistant", "content": "message 6"},
    {"role": "user", "content": "message 7"},
    {"role": "assistant", "content": "message 8"},
    {"role": "user", "content": "message 9"}
  ],
  "insert": {"9": -1},
  "drop_message": {"3": [1, 2], "8": [1, 2, 3, 4, 5]},
  "reposition": [3, -1]
}
```

执行为：

```text
A0 A1 A2 A3 D(1,2) R A4 A5 A6 A7 A8 D(3,4,5) IB S(-1) IE R A9
```

第二次 Drop 中的 1/2 已经删除，不再产生 Delta 记录。Drop 在触发消息完成后执行；
普通消息上的 Insert 在该消息开始前执行；Reposition 在触发消息的后置操作阶段执行。
负编号链 `{"9":-1,"-1":-2}` 先完整执行 I(-1)，再执行 I(-2)。即使客户端首次
发送这两个 Insert，系统也按这个顺序赋予 birth depth。子 Insert 先于父消息的
Drop/Reposition 后置钩子；对象成员顺序不改变执行顺序。

Insert 锚点由 logical 顺序中第一个已出生且未 Drop 的后继 token 确定。
没有这个后继、引用未出生消息进行 Drop、缺失引用、重复插入、编号模式混用或循环
都报错。空文本可以拥有模板 token；模板处理后完全没有 token 的插入消息报错。
响应 `sglext.context_message_indices` 给出规范化消息编号，
`context_next_message_index` 给出下一个普通编号。

实现入口：`python/sglang/srt/context_system/rules.py:54-119`
（`normalize_message_indices`、`InsertRule`）；
`python/sglang/srt/entrypoints/openai/protocol.py:892-912`
（`validate_context_indices`、`context_index_metadata`）。

## 从逻辑文本到 Radix

沿用完整 native template 的 provenance/消息分割规则，不对历史前缀逐段重新 tokenize。
对完整 logical token 序列建立 execution permutation，generation prompt 最后进入。
消息编号只用于前端事件编译，不进入 token Radix 的插入锚点。

IB 为 `(3, birth_depth, 0)`，IE 为 `(4,0,0)`。birth depth 从 1 开始，计数所有执行过的
真实 token，包括已删除 token 和先前 Insert 的 token；虚拟节点不计数。
IB 不包含 final span。TOKEN 记录仍包含最终位置，D/R 仍使用原有编码。
普通 Append 的位置游标不被 Insert 改写；Drop 不改变幸存位置；有效 R 按 active logical
顺序压缩位置。原有无效 R/tail Drop 的游标行为保留。长 Insert 可以越过普通 Append 游标，
因此在显式 R 之前允许位置重叠；所有实际 birth、中间旋转和 terminal 位置仍受 RoPE 上限约束。

同样的 execution prefix、操作和锚点产生相同历史语义。Exact matching 比较完整记录；
retrigger 只忽略 TOKEN 最终位置，不忽略 D/R/IB/IE 或锚点。匹配后延用现有 COW 旋转，
不改写共享源 KV。树边切分不改变 birth depth；IE 作为下一真实 token 的前导事件，
末尾事件保留于请求程序，生成 token 追加时进入 key。

R3 张量 wire 的 Insert 版本为 7，JSON wire 为 8；无 Insert 保持原版本 3/4 和字段集合。
旧 Insert 5/6 明确拒绝，避免静默混用不同 SWA 语义；所有 worker 必须使用同一版本并隔离旧缓存。

实现入口：`python/sglang/srt/context_system/planner.py:317-544`（`ContextProgram`）、
`planner.py:547-664`（`_compile_insert_program`）；
`python/sglang/srt/context_system/ir.py:293-348`（`compile_execution_layout`）、
`ir.py:371-502`（`ContextKeyData`）。

## 复用、恢复、attention 与回填

缓存地址和生命周期继续按 execution token 管理；attention 的 prefix 索引按 logical 顺序
组织。Insert 的 query 只看其左侧已出生、未删除的 KV，后缀原 KV 保留。
同一层在所有 birth KV 写入和必要 COW 旋转后，使用 segment batch 执行原生 attention。
未命中需要计算的 query 仍按 execution 顺序排列，段之间的 KV 索引可指向保留后缀。

恢复规划计算真实读集合，并对缺页做反向依赖闭包。SWA 的窗口条件仍是原生位置距离，
只对因果可见的 logical KV 选择窗口。Insert 分支的 Radix 匹配不再要求 SWA 必须形成
execution 尾部连续区间；实际 Full/SWA residency 由恢复规划和 lease 再验证。
完成后的 terminal rows 按 execution 顺序发布回原生 Radix，保留所有已完成事件。
原有 Drop proof、source lease、共享页所有权和回收规则继续适用。

SWA 缺页不等于所有已旋转的 Full KV 都必须重算。恢复规划按 query 真正需要的 occurrence
位置判断旧版本需求：query 位于最后一次 R 之后、且缓存 Full KV 已在对应位置时，仍可直接
复用。只有被选中的历史 query 需要其他位置版本，才进入原有的 rewind 恢复规则。
`Req.prepare_context_recovery` 传入匹配得到的源位置；不能只用“曾经 R 过”作为全局失效条件
（`python/sglang/srt/managers/schedule_batch.py:1578-1668`）。
匹配点之后的新 query 可以通过后续 R/COW 获得目标位置版本；不能因为未来旋转，
反过来把正在修复的历史 query 所使用的正确源 KV 标为失效。

Decode 的 SWA 条件为 `q_pos-k_pos <= window`，先在因果可见的 logical KV 中筛选，
再从后往前保留最多 `W=window+1` 项，输出索引仍按正向 logical 排列。例：两段位置
均为 1…512、query 位置为 513、距离窗口 128，位置条件给出 257 个候选；R3 最终
只读最后 129 个（包含 query）。四条请求使用 516 个索引，适配原生 graph 缓冲。
这替代 R2 的“所有位置候选都读入、超过容量时 eager”规则。Full 容量不足等独立
限制仍可能导致原有 eager 回退，但位置重叠本身不再要求扩大 SWA graph 缓冲。

当前未入选的旧 KV 不能一律永久释放。例：logical prefix 位置为 `[20, 1, 1]`，
下一个 query 位置为 3、W=3，位置 20 被后面的候选挤出；到 query 位置 4 时，
两个位置 1 同时过期，位置 20 又可入选。`window.py` 的 run maximum 索引每次重新选取合法
候选；terminal 恢复需求覆盖前 W−1 个未来 query 的并集。之后新 query 已填满窗口，
不再需要 prompt KV。R 后严格单调的位置只需一次二分和尾部截取。

容量入口为 `python/sglang/srt/layers/attention/triton_backend.py:1107-1118`、
`flashinfer_backend.py:1038-1045`、`flashattention_backend.py:2230-2238` 的
`context_decode_fits_graph`；准入位于
`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:635-726`（`can_run_graph`）。
FlashInfer 将 window 和完整 logical 长度传入 gather，避免把非单调窗口当作连续尾部
（`python/sglang/srt/layers/attention/flashinfer_backend.py:1751-1871`，`call_begin_forward`）。

实现入口：`python/sglang/srt/context_system/occurrence.py:159-321`
（`compile_occurrence_window`）；`python/sglang/srt/context_system/recovery.py:305-400`
（`_plan_insert_recovery`）；
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1059-1178`
（`_match_context_retry`）；`python/sglang/srt/layers/attention/context_backend.py:713-905`
（`ContextDecodeRegistry`）。

旋转后私有 SWA 页的释放不能从 source match 的 `cache_protected_len` 开始。
该边界保护的是被借用的源分支，边界之前也可能存在本请求的 COW 目标页。
`OccurrenceState.first_private_terminal` 在不可变状态上缓存最早私有 raw 行，
回收起点同时服从已经回收的位置和存活依赖下界；`live_swa_ranges(..., owned_only=True)`
排除共享源页和已缺失的 SWA。decode 和完成后的 publication 都使用相同规则，
超过原生历史表容量时从扩展 raw row 读取地址。
实现为 `python/sglang/srt/context_system/occurrence.py:936-973`、
`python/sglang/srt/managers/schedule_batch.py:4211-4241` 和
`python/sglang/srt/mem_cache/unified_cache/components/swa.py:1076-1103`。

## 性能边界

- 无 Context 请求继续原生路径；没有 Insert 的 Context 请求继续原有编译器。
- logical/execution 逆映射线性构建。匹配使用预计算 `has_insert`，不复制扫描完整 key。
- decode 的位置单调性、排序位置和 raw 后缀最小值只计算一次。
  CPU 窗口计数/安全回收边界用二分查找，不在每轮 decode 扫描整个历史。
- 已绑定批次缓存 Full 总读数和最大行宽。所有 SWA 请求的 W 上界均可 O(1) 判断
  graph 容量，不进行 GPU→CPU 同步，不统一扩大 graph 缓冲。
- Reposition 后的位置单调路径继续原生快速窗口 gather。未 Reposition 的非单调路径
  用连续位置 run 的最大值树跳过不满足条件的区间，最多输出 W 个 logical 索引；
  不再每步扫描完整 prompt。Full/SWA 原生物理槽翻译融合在 gather 中。
- 非连续位置的 ragged SWA adapter 只拆分每段开始的最多 W−1 个 query；当新 query
  填满窗口，余下 token 合为一个原生 segment。Triton 对位置重叠也使用该有界读集。
  计划每个 pool/window/forward 只构建一次。不存在生产用 N×N attention mask。
- 缺页恢复的最坏开销取决于 segment/read-set 数量；大量 Insert/R 不保证与纯 Append
  相同速度。真实 summary 场景需看 TTFT、实际 prefill tokens 和 cache hit，而不能只看命中长度。

## 验证与 48K 实验

测试包括独立逐 token 状态机、60 组连续 turn 操作、负编号链、重叠 Drop 的 Delta 等价、
wire、partial prefix、原生树分叉、Full/SWA 缺页闭包、native segment attention 和 decode graph。
CPU/GPU 用例分离，skip 不计通过。GPT-OSS HTTP 测试比较热 retrigger、exact、冷恢复
以及后续在旧 Insert 前再次 Insert 的全词表 logits。

```bash
python -m pytest -q test/registered/context_system/test_insert_*.py
RUN_CONTEXT_GPU=1 python -m pytest -q test/registered/context_system/test_insert_attention.py
python -m pytest -q benchmark/context_system/test_insert_summary48k.py
```

48K 驱动：`benchmark/context_system/insert_summary48k.py`。固定题号：
238、241、775、786、787、822、844、864、870、875。
阈值是 `active logical input >= 49152`；保留约 20000 tokens 的完整近期协议组。
旧 summary 可被下一次 summary 总结并 Drop。每次 summary 使用同一被测模型新生成，
提示格式参考原 SWE trajectory 的 Goal/Constraints/Progress/Decisions/Next Steps/Critical Context。
summary 截断、缺少标题或没有缩短 active input 时，不提交历史变更。
48K 是触发阈值；保留约 20K 是目标，完整的超大历史消息组不能强制后续请求一直超限。
选择范围时预留 summary 预算，并保留已计算的 Insert 锚点。若单次 pending 输入本身已
超过 48K，则尽量压缩旧历史、记录 `oversized_pending`，交由原生准入检查实际模型上限；
下一 turn 再总结这段已计算的输入。普通 pending 输入压缩后仍超过 48K 继续判为失败。
这些驱动边界修正在 `26eabc33e6`，没有改变 `4c26d4713c` 的生产运行代码。

```bash
python benchmark/context_system/insert_summary48k.py \
  --url http://127.0.0.1:28765 \
  --model /share/public/public_models/gpt-oss-120b \
  --data /share/public/wangruoxi/local/throughput_bcp_full_context \
  --output /path/to/new-output-directory
```

这是“新生成 agent/summary + 录制 tool feedback”的系统实验，不是 live search 的 BCP 准确率。
默认沿用 source 每 turn 的输出预算；`--probe-tokens` 仅用于缩短诊断，结果明确标记，
不能代替完整预算验收。原始数据 SHA256、请求/响应、summary 文本、输入 hash、耗时和
服务端 usage 都留档。每道题必须实际触发 summary 才算覆盖；Qwen Insert 仅验拒绝行为。

默认在每个 summary 事件后冻结同一请求，比较 warm、exact、flush 后 cold 的首个解码步
全词表 logits（`context_trace_count=1`）；这不是逐个生成 token 的输出一致性验证。
每个事件先保存 Insert 的 warm/exact 结果，再用普通 active 文本请求测量 native 的
warm/cold 与 exact/cold 差异；按照预先固定的 `max(BF16 floor, 2 × native envelope)`
生成容差，之后才比较 Insert，不能根据 Insert 结果放宽容差。普通 active 文本只用于
数值重复性校准，不是 cache engineering 的语义参考。
数值失败会保留证据并将该题标记为 failed，同时继续后续 turn，覆盖重复 summary。
HTTP 200 携带 `finish_reason=abort/error` 或零生成 token 时直接判为失败，不追加空 assistant。
因此服务端需开启 `--enable-custom-logit-processor`，且驱动与服务端能访问相同的输出路径。
`--skip-numeric` 用于独立无 flush 长程匹配或性能实验；另开冻结请求 replay 完成数值验收，不能用跳过比较的运行宣称 logits 通过。包含 flush 的整轮耗时
不是纯吞吐测试结果，不能据此声称端到端无性能回退。

## 已完成验证与待验收项

以下为 R1/R2 历史证据，运行时检查点为 `4c26d4713c`；R2 容量修改为 `bcdf3ff33d`、`f72a1e9636`。
下列命令之间存在用例重叠，不应把通过数相加。

| 验证 | 结果 | 范围 |
| --- | --- | --- |
| 早期本地 CPU 混合回归 | 427 passed / 13 skipped | Insert、IR、rules、attention plan、decode |
| 早期 InfiniAI-BUS-2 原生回归 | 295 passed / 13 skipped | frontend、unified cache、ownership、admission、storage、GPU |
| Tengxuan-BUS-2，`a89fc3fe4f` | 60 passed | GPU attention、实际模型 FP32 oracle、校准 HTTP 请求链、长 Insert |
| 本地 R2 CPU 回归 | 108 passed / 3 skipped | 容量、重复位置、IR、frontend、decode、benchmark |
| R2 原生 GPU 首次验证 | 66 passed / 1 failed | 混批普通 lane 少读边界 1 个 KV；失败后修正 |
| `f72a1e9636` GPU 复验 | 2 passed | 四条重叠请求、普通/Context 混批；Triton/FA 容量与 FlashInfer 实际 gather |
| `f72a1e9636` GPT-OSS HTTP | 2 passed | 短/长 Insert，无 R，cold/warm decode；长例超过 graph SWA 缓冲容量 |
| `c53276cca3` 本地恢复回归 | 110 passed / 3 skipped | 位置版本依赖 oracle、IR、planner、raw storage、benchmark |
| `4c26d4713c` 原生 CPU 回归 | 169 passed / 3 skipped（两次运行） | 17 项先因原生测试工具不接受空 CUDA_VISIBLE_DEVICES 而初始化失败；改为不可见设备 99 后全部通过，无源码修改 |
| `4c26d4713c` GPT-OSS HTTP | 3 passed / 3 deselected | 热 retrigger、exact、在旧 Insert 前再次 Insert、短/长重叠位置 decode |
| `26eabc33e6` summary 驱动 | 本地和 Linux 各 13 passed；ruff 通过 | 超长 pending 保留已出生锚点、下一 turn 删除超大历史协议组、不截断或伪造 summary |

GPT-OSS 使用模型原始权重、BF16、MXFP4、TP2、Triton attention 和 triton_kernel MoE。
小型 probe 在两个 TP rank 的前四层检查实际 query、存储 KV、prefix KV、positions 和
attention 输出；独立 FP32 attention oracle 使用实际 sinks、GQA 和位置窗口。
这个检查通过不等价于已经证明所有长上下文的最终 logits 一致。

在相同配置下，基线 `05dc3b7d1b` 与 Insert 版本的普通/identity/drop 数值诊断结果相同。
例如普通 native warm/cold logits 最大绝对差约 2.05、均值约 0.293；因此固定的微小
BF16 logits 容差不适用于这个 MXFP4/MoE 配置。上述校准沿用 native 误差包络规则，
另外保留实际 KV/FP32 attention oracle，不能只靠最终 logits 的宽容差宣称语义正确。

本地 49152-token 合成程序的旧测量（嵌套 Insert、Drop、R；12 次取后 10 次中位数）为：
编译约 10.29 ms、key 构建约 0.74 ms、warm recovery 规划约 1.47 ms。
这是 CPU 元数据测量，不含 tokenizer、模型 forward 或网络，不代表端到端速度。

早期 10 题运行 `bcp48k-full-v2` 尚未通过：238 第二次 summary 的数值检查超出固定容差；
241 两次 summary 数值检查通过，但 turn 29 在准入队列停留 453 秒、GPU 空闲，随后中止。
该轮中止响应为 HTTP 200/finish_reason=abort，不能计为已完成 turn。
诊断确认 Radix 命中了 84768 token；仅缺少 9 个 SWA token，却被旧恢复规则扩展成
37027 个 query。`c53276cca3` 对保存下来的同一状态规划出 2929 个 query，
即 9 个修复 query 加 2920 个新 query，不改变 KV 池容量。原请求端到端复验在同一配置下
实际 prefill 2929 token、11.046 秒返回，未再出现该准入阻塞。其冷 source 重新生成了输出，
匹配点与保存的旧快照略有差别，不能把这次耗时当作所有场景的加速比。
`4c26d4713c` 进一步把新 query 的未来 R/COW 旋转与历史修复分开，覆盖热 summary 触发时
旧尾部缺页的情形。最新真模型 HTTP 请求链为 3 passed；最终 V4 中 241 的 turn 29
实际 prefill 2934 token、10.63 秒完成。中途的 v3 运行用于诊断，未完成，不计为验收通过。

### 最终 10 题结果：数值验收未通过

`bcp48k-v4-a` 至 `f` 共完成所选 10 题的 393 个 turn、28 次真实 summary。
最终选定轨迹共 617 次模型请求（含 summary 和数值对照），均正常返回；56 项首步全词表
logits 比较中 46 项通过、10 项超过预先固定的 native-control 门槛，涉及 6 题。
整体结果为 **FAIL**，不能据此宣布完整数值等价或“完成无误”。

| BCP | 完成 turn | Summary | 数值结果 | 最终记录 |
| --- | ---: | ---: | --- | --- |
| 238 | 45 | 3 | 通过 | V4-A |
| 241 | 40 | 3 | 第 1 次 warm/exact 超限 | V4-A |
| 775 | 44 | 3 | 第 3 次 warm 超限 | V4-A |
| 786 | 40 | 3 | 第 2 次 warm/exact 超限 | V4-C |
| 787 | 44 | 3 | 通过 | V4-C |
| 822 | 39 | 3 | 第 2 次 warm 超限 | V4-B |
| 844 | 41 | 3 | 通过；独立补跑 | V4-D |
| 864 | 31 | 3 | 第 1 次 warm/exact 超限；策略补跑 | V4-F |
| 870 | 35 | 2 | 通过 | V4-C |
| 875 | 34 | 2 | 第 1 次 warm/exact 超限 | V4-E |

例如 241 第 1 次 exact 的最大绝对差为 1.6875，门槛为 1.0；平均绝对差约 0.2643，
门槛约 0.1820。不能因为采样 token 相同、普通模型也存在浮点波动、或 key 检查通过，
就将这些失败改判为通过。尚未将长上下文差异归因于数值路径或缓存复用错误。
后续需要在相同逐 turn 执行划分下补充 native 对照，并定位更深层的 KV/attention 误差。

844 首次运行在 4096-token summary 预算处截断，未提交 D/I/R；V4-D 以 8192 预算重新
执行完整轨迹。864 首次运行遇到单条 pending tool 输入 51762 token，压缩后仍超过
48K，旧驱动停止；V4-F 使用 `26eabc33e6` 的策略修正及 8192 summary 预算完成补跑。
两次补跑都会重新生成 agent/summary 文本，不能将结果变化仅归因于预算差异；旧失败
请求和报告均保留。其余运行使用 4096 summary 预算，原 turn 的生成预算始终未缩减。
所有服务端运行时代码均为 `4c26d4713c`；`26eabc33e6` 只修改 benchmark 和对应测试。

864 的 turn 24 在压缩旧历史后保留 53578-token active input，由原生模型上下文上限
进行准入。turn 25 再总结该超大历史组，active input 从 55525 降到 3379；热请求实际
prefill 2915 token、7.35 秒完成。它验证了“超长 pending 先计算、下一 turn 再总结”的
边界，未截断录制的 tool 内容。策略实现见
`benchmark/context_system/insert_summary48k.py:126-174`（`select_summary`、`validate_summary_size`）。

### 匹配与性能证据的范围

对全部实际 turn 重建 execution tokens，均与原请求记录一致；383 个相邻 turn 的
exact/retrigger key 比较均与独立逐记录 LCP 扫描一致。所有 retrigger 长度都达到实际
execution token 序列的 LCP，没有发现新增 D/R/IB/IE 编码造成额外前缀损失。
这验证 key 的比较规则，不是对所有实际缓存 residency 或深层 KV 数值的等价证明。

其中 13 个相邻 turn 的历史输入前缀损失超过 32 token。它们的历史 messages 未改变，
但原生 Harmony 模板在出现新 final 后移除了此前的 analysis；保留 analysis 的对照
在全部 13 处都恢复了完整历史输入匹配。864 的 turn 29 已经历三次 Insert，仍因同一
模板行为损失 6556 个历史输入 token，turn 30 的历史输入前缀损失恢复为 0。
因此实际运行确有较大失配，但这些样本没有显示由 Insert 操作编码造成的持续失配。
模板行为没有在本实现中修改，见 `python/sglang/srt/parser/gpt_oss_encoding.py:264-295`
（`_drop_harmony_analysis_before_last_final`）；对照见 `bcp48k-v4-template-diagnostic.json`。

28 次热 summary 请求的实际 prefill 中位数为 3590 token，HTTP 耗时中位数约 9.08 秒；
28 次 exact 请求均只计算 1 个 prefill token。最大的热请求计算 53115 token，耗时
89.70 秒，对应 864 的 51762-token 全新 pending 输入。真实 summary 生成本身的
耗时中位数为 59.84 秒，应与 Insert 开销分开统计。
这些时间包含验证用 logits 保存和完整 token 返回，没有进行生产吞吐或无功能路径的
统计显著性比较，不能据此宣称全部配置无性能回退。

最终报告及逐请求计数保存在实验根目录 `/home/wangruoxi/insert-20261103/`：
`bcp48k-v4-final-report.json`、`bcp48k-v4-prefill-accounting.json`、
`bcp48k-v4-key-audit.json`、`bcp48k-v4-template-diagnostic.json`；
原始请求/响应、logits、配置和日志保留在各分组目录。
每个分组配置记录源码 hash、GPU、完整客户端/服务端参数。A/B/C 在其分配题目完成后
停止，日志中下一题的 case_start 不计为完成结果。

GPU 主机为用户指定的 Tengxuan-BUS-2；测试使用 0/1、3/4、5/6 三个隔离 TP2 实例，
GPU 2 的既有其他任务未操作。0/1 用于主验证、2 被占用，后续实例使用剩余空闲的 3/4、5/6。
每个并行实例分离端口、checkout、cache、日志和输出；运行时代码、模型及容量一致。
补跑仅在前一实例退出后复用其 GPU/端口/cache，单独记录日志及输出。实验结束后已释放本次 GPU。
Qwen3.8-27B 不进行 Insert 成功路径验收，仍按用户要求保持不支持。


## R3 实施与验证（数值验收尚未通过）

Qwen/Linear Insert 的全部支持修改和模型实验继续 Pending；当前仍拒绝 Insert。
原生 Harmony 模板没有修改。下列诊断不进入生产默认路径：

- `benchmark/context_system/insert_state_audit.py` 的 `create_cache_audit` 通过原生
  `--forward-hooks` 显式启用。独立逐 record 比较并遍历真实树，核对 maximal retrigger
  前沿、所选分支 exact 边界、返回物理槽、source positions 和 Full/SWA residency。
  `retry_only = matched - selected_exact`，不能把两种命中长度相加。
- `test_insert_long_match.py` 在原生 Full 和 Full+SWA 树中连续执行 40 turn，覆盖重复
  Drop、R、在历史 Insert 前再插入、同请求负编号链和边分裂后的完整回填。
- `test_insert_attention.py` 加入整段 Q 的置换对照：同步置换段的 Q/K/V、位置、读集和
  causal 边界后，逆置换输出应相等。只改变 Q 的顺序而未保留这些关系不具备等价保证。

本地 CPU 已运行 frontend/IR/recovery/decode/audit 组合：147 passed、77 skipped（组合中包括 GPU/服务测试跳过）。
Tengxuan-BUS-2 新环境的原生组合回归在 `96bff08496` 为 224 passed、52 skipped；
包括 Full/Full+SWA 各 40 turn 的实际树验证。跳过项为 48 个旧模型 KV trace fixture
和 4 个需要单独 HTTP/BCP 服务的用例，不计为通过。

实际模型随后发现 eager 的 int64 长度会使新的 SWA Triton 计数循环发生类型提升、
编译失败；graph 使用 int32，原 GPU 测试只覆盖了后者。`a93584b3ad` 将有界 token
长度统一为内部 int32，并补充两个输入 dtype 的 eager/graph 随机测试。修复后
`test_insert_attention.py test_context_decode.py` 为 24 passed、48 skipped，
其中随机窗口共 600 组长度/位置/物理槽对照通过。这个失败发生在 logits 比较前，
不是对旧 logits 差异的解释。

随后真实 8 步 decode 在 238/241/786/844 以及 1641-token 短例中发现 127 个 SWA
槽位未释放。Full 池仍平衡，Radix key/位置审计通过；问题在于私有旋转目标落在源
`cache_protected_len` 之前，旧回收游标跳过它们，publication 又将其标记为已无 SWA。
`5c9154c10b` 修正所有权边界，`71110c9283` 补齐扩展 raw row 的读取。
两种新输入长度的 warm→8-step decode→exact 请求链，以及两种无 R 的长重叠 Insert
请求均通过，共 4 个真实模型 HTTP 用例。原生 CPU 组合复验为 173 passed、8 skipped；
后者需要独立 GPU 或 HTTP 服务，不计为通过。

新的冻结 241 首次 8 步回放仍 FAIL：零基 step 2 的 exact，以及 step 4 的 warm/exact
超过逐步 native-control 阈值。所有步骤强制使用 cold 生成的同一组 token，输入 hash
一致，不能将不同续写造成的差异混入比较。聚合阈值通过也不能掩盖单个步骤失败。
增加的 `create_decode_read_probe` 比较实际物理 gather 与独立扫描 logical 状态的参考，
并用模型 sinks/GQA 做 FP32 attention。此探针只用于诊断，不进入性能测量。
对该冻结事件两个 TP rank 的 36 层，采样的最终 prefill 行及 7 个后续 decode 行
均使用相同 token/position。后续 decode 的 layer 0 采样 Q/K/V/output 相同，首次
attention output 差异在 layer 1；prefill 的 layer 0 已有小幅差异（rank 1 exact 的 Q
最大差约 3.05e-5）。这是末尾 query 的定位证据，不是完整历史 KV 值的等价证明。
实际 decode gather 另按全部被读 KV 的槽号检查，当前 query 的存储 K/V 逐元素核对；
深层 FP32 attention 参考在 native 与 Context 两边均出现超限，仍逐项保留为 FAIL。
原生 kernel 的 softmax 权重转 BF16 后做 P·V 是一个可能的数值来源，不能仅凭源码
把最终 logits 失败归因于它（`python/sglang/kernels/ops/attention/decode_attention.py:729-739`）。

诊断还确认一个既有边界差异：该 GPT-OSS 的层距离参数为 127；Context 路径按已批准
规则读最多 128 项，而纯 native Triton decode 的 `update_sliding_window_buffer`
直接将参数作为总容量，只读 127 项。该 helper 在 R3 基线已有相同行为
（`python/sglang/srt/layers/attention/triton_backend.py:2665-2715`）。
普通请求继续原生路径，R3 没有修改这个 helper。`5e7cb2915b` 修正普通请求与 Context
请求混批时的容量选择：普通 lane 读 127 项，Context lane 读 128 项，CPU 准入和 GPU
计数/gather 使用同一分流规则。FlashInfer/FlashAttention 沿用各自原有边界。
最终相关 GPU 回归为 25 passed、48 skipped，覆盖真实 Triton adapter 的 eager/graph
索引与原生 helper 对照；CPU 组合为 174 passed、8 skipped。
其中 GPU 48 项需要 `CONTEXT_INSERT_KV_TRACES` 指定的旧 fixture；CPU 8 项需要独立
CUDA/HTTP/BCP 服务。它们均不计入通过数。native control 的读数审计
按其既有容量规则单独标记；不能据此宣称普通 native 和 identity Context 的 decode 边界完全相同，
也不能将这项差异用作放宽 Insert logits 门槛的理由。

独立环境位于 `/home/wangruoxi/insert-r3-20261103/env/venv`，Python 3.12.16、
PyTorch 2.13.0+cu130、Triton 3.7.1；用户态 CUDA compatibility 库位于同目录下，
主机驱动未修改。依赖 freeze 和源码 hash 已归档。可使用
`source /home/wangruoxi/insert-r3-20261103/env/activate-runtime.sh` 激活完整运行环境。
主机驱动保持 535.161.08，CUDA 13 兼容库仅通过此环境加载。

10 题无 flush 轨迹已完成 393 turn、27 次真实 48K summary；每个 turn 沿用原始输出预算。
822 本次触发 2 次 summary，其余题目与 V4 次数一致。agent 与 summary 都重新生成，
因此这不是固定文本的两版本数值对照。轨迹使用 `--skip-numeric`，冻结数值验收单独运行。
按响应 request ID 排除初始 HTTP 测试后，213 个含 Insert 请求在两个 TP rank 上产生
426 次真实树匹配和 426 次发布审计，均通过；最大匹配深度 130043，最大发布深度
130399 个 execution token。观测范围内没有发现驻留兼容分支被漏选或固定截止在 IB/IE。
383 次相邻 turn 的离线重建都与实际服务 input IDs 一致；按记录中的 Harmony 日期
重建，避免分析日期变化混入 token 差异。27 次 retrigger 比 exact 更长，operation
额外前缀损失为 0。另有 15 次实际 token 前缀变化，未逐项重新归类为 analysis 清理；
它们与 operation 编码造成的损失分别计数。
服务端 3418 条 decode 日志样本全部使用 CUDA graph；这是采样日志，并非所有 token
的逐个统计，且 A 组包括最初四个 HTTP 回归用例。

无诊断探针的性能对照已完成：同机 GPU 0/1、同一模型/配置/环境，每版本三轮交错测量。
基线为 `3a889579b8`，当前生产代码为 `5e7cb2915b`（后续 `d6f4cd18c6` 只改离线审计）。
JIT、预热和 source cache 填充不进入统计；普通/Drop-R 热请求各 36 条、冷 TTFT 各 9 条，
summary warm prefill 每版 3 条。跨午夜和修复前的两个试跑轮次排除，原记录保留。

| 指标 | 基线 | R3 | 变化 |
| --- | ---: | ---: | ---: |
| 普通 decode p50，ms/token | 8.9215 | 8.9247 | +0.04% |
| 普通 decode p95，ms/token | 8.9294 | 8.9481 | +0.21% |
| 普通热 TTFT p50，ms | 124.37 | 126.19 | +1.46% |
| 普通冷 TTFT p50，ms | 10280.23 | 10280.59 | +0.004% |
| Summary warm prefill p50，ms | 7857.45 | 7843.85 | −0.17% |
| Drop/R decode p50，ms/token | 8.7695 | 8.7707 | +0.01% |

以上满足计划门槛。decode p50/p95 是每条 128-token SSE 请求的平均单 token 延时分布，
不是逐 token 到达间隔的分位数；TTFT 取首个可见内容/reasoning/tool delta。其他诊断
实例使用独立 GPU，但共享主机 CPU/内存。样本不足以外推所有并发负载或宣称统计显著加速。
KV 池预分配为每 rank Full 131072 槽、SWA 104857 槽，约 4.05 GiB/rank；
日志中的使用率是采样值，不是精确驻留高水位。

### 冻结 28 个旧事件的最终数值结果

28 个旧 summary 事件全部完成 8 步 teacher-forcing replay；它们与本次重新生成的
27 次 summary 长轨迹分别统计。每个事件的 cold/warm/exact 使用相同 execution input
IDs，并在后续步骤强制使用 cold 生成的 token。cold 从空缓存重放同一 A/D/I/R 程序，
warm 先填充原先请求的缓存，exact 再次执行同一目标请求；比较的是后两者分别与 cold
的全词表 logits。普通 active-text 的 native 对照仅校准浮点重复性，不是 Insert 语义参考。
实现见 `benchmark/context_system/insert_state_audit.py:315-490`（`replay`）。

| 验收口径 | 通过 | 失败 |
| --- | ---: | ---: |
| 首步 warm/cold 与 exact/cold，28×2 | 45 | 11 |
| 每步独立门槛，28×2×8 | 343 | 105 |
| 跨 8 步聚合门槛，28×2 | 50 | 6 |
| 事件：两条路径全部步骤和聚合均通过 | 3 | 25 |

仍使用预先固定的 `max(BF16 floor, 2 × native envelope)`，没有放宽阈值；整体验收为
**FAIL**。事件级通过要求更严格，不能拿 3/28 与旧的首步 46/56 直接比较。旧 R1/R2
的 10 项失败原样保留；新环境、实际模板日期和逐步门槛不同，也不能把新增失败数直接
解释为 R3 引入的回归。三次跨午夜的未完成对照只有输入第 35 号 token 因 Harmony
日期改变，已排除并完整重跑；28 组有效对照各自内部日期与 input hash 一致。

decode 独立读集探针在 Context 和 native 两组各检查 42240 个 layer/rank/forward
样本，物理槽号顺序均符合各自规则；各 41472 个当前 query 的存储 K/V 检查均通过。
初始四个事件只采样前四层，后续事件采样全部 36 层；异步调度可能多产生一个未提交
token 的 forward，故样本数不是 28×8×36×2。FP32 attention 对照仍分别有
209 / 202 个超限样本，最大绝对差约 0.1661 / 0.1594，保留为失败。
这里验证的是读槽和当前写入，尚未逐元素证明全部历史 KV 的值与来源正确。
探针见 `test/registered/context_system/serving_logits_probe.py:4-103`
（`create_decode_read_probe`）。

额外在同一新环境、同一日期和相同 input IDs 下回放旧基线 `3a889579b8` 的 875
第 1 个事件。完成的 cold 和 warm 各 8 步分别与 R3 **逐位相同**；两版本 warm/cold
最大绝对差都是 2.09375、平均差约 0.19267。此样本的差异在 R3 前已经存在。
但旧基线在 warm 完成后的原生空闲检查中复现 127 个 SWA 槽位泄漏，服务退出，
exact/native 对照未完成，因此它不是完整的基线数值验收，也不能解释其他事件。
没有修改基线源码或关闭该检查来把失败变为通过。

实际 layer-0 extend query 计数：cold 中位数 80963.5、warm 中位数 3683.5；
28 次 exact 均只计算 1 个 prefill token。最大 warm 为 53114，对应包含超长全新
pending 输入的 864 事件；这些计数包含恢复 query，不包含 decode、复制或元数据工作。
诊断 replay 关闭 graph 以采集完整读集，不能用其 HTTP 时间衡量生产性能。
上面的无探针性能实验和无 flush 长轨迹使用开启 graph 的独立服务。

### 复现、归档与未解决项

Tengxuan-BUS-2 的源码目录为 `/home/wangruoxi/insert-20261103/sglang`。
CPU 组合在 `d6f4cd18c6` 运行，GPU 组合在 `5e7cb2915b` 运行；两者生产代码一致。
下列为实际测试命令（先激活前述环境并进入该目录）：

```bash
CUDA_VISIBLE_DEVICES=99 python -m pytest -q \
  test/registered/context_system/test_insert_frontend.py \
  test/registered/context_system/test_insert_ir.py \
  test/registered/context_system/test_insert_recovery.py \
  test/registered/context_system/test_insert_cache.py \
  test/registered/context_system/test_insert_long_match.py \
  test/registered/context_system/test_insert_state_audit.py \
  test/registered/context_system/test_context_admission.py \
  test/registered/context_system/test_context_decode.py \
  benchmark/context_system/test_insert_summary48k.py
# 174 passed, 8 skipped

CUDA_VISIBLE_DEVICES=2 RUN_CONTEXT_GPU=1 python -m pytest -q \
  test/registered/context_system/test_insert_attention.py \
  test/registered/context_system/test_context_decode.py --tb=short --maxfail=1
# 25 passed, 48 skipped
```

无 flush 长轨迹的运行时代码为 `71110c9283`，后续 `5e7cb2915b` 只修正普通请求
与 Context 混批的原生容量，相关 mixed-lane GPU 用例另行通过。冻结 replay 的运行
版本和配置分别留在各目录的 server/config JSON；不能将所有实验冒称为同一 HEAD。
28 个冻结请求的 target/source/steps 保存在每个 `report.json`，用
`insert_state_audit.py --url ... --model ... --target ... --source ... --steps 8
--layer-probe --output ...` 可在同配置独立服务重放；退出码为 0 不代表数值通过，
必须检查报告的 `pass`。性能运行使用 `--timing` 且不启用 forward hooks。

实验根目录 `/home/wangruoxi/insert-r3-20261103/` 保留：
`frozen-final-report.json`、`frozen-prefill-accounting.json`、`frozen-rendered-dates.json`、
`midnight-exclusions.json`、`baseline-numeric-comparison.json`、`241-layer-localization.json`、
`trajectory-final-report.json`、`trajectory-key-audit.json`、`performance-final-report.json`，
以及原始 logits、请求/响应、日志、依赖 freeze、各轮配置。实验服务已停止，GPU 已释放；
没有删除旧失败或跨午夜记录。

剩余的主要问题是长上下文 logits 差异的根因：需要继续区分历史 KV 数值/旋转误差、
prefill 分段及量化计算路径的影响。当前证据不足以宣称所有复用数值正确，也不足以
将全部失败归因于浮点误差；SWA 回收修复没有消除这些差异。Qwen/Linear Insert
按用户要求全部 Pending。FlashAttention/FlashInfer 的全模型覆盖、其他硬件和更大
并发下的吞吐未在本轮验证，不能从 A800/Triton 的结果外推。
