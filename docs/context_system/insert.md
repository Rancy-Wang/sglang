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
（`python/sglang/srt/managers/schedule_batch.py:1577-1665`）。
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

容量入口为 `python/sglang/srt/layers/attention/triton_backend.py:1104-1112`、
`flashinfer_backend.py:1038-1045`、`flashattention_backend.py:2230-2238` 的
`context_decode_fits_graph`；准入位于
`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py:635-726`（`can_run_graph`）。
FlashInfer 将 window 和完整 logical 长度传入 gather，避免把非单调窗口当作连续尾部
（`python/sglang/srt/layers/attention/flashinfer_backend.py:1751-1871`，`call_begin_forward`）。

实现入口：`python/sglang/srt/context_system/occurrence.py:146-308`
（`compile_occurrence_window`）；`python/sglang/srt/context_system/recovery.py:301-385`
（`_plan_insert_recovery`）；
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1059-1178`
（`_match_context_retry`）；`python/sglang/srt/layers/attention/context_backend.py:678-865`
（`ContextDecodeRegistry`）。

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


## R3 验证进度（尚未验收）

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

本地 CPU 已运行 frontend/IR/recovery/decode/audit 组合：109 passed、3 skipped。
GPU、实际树 Linux 用例、10 题无 flush 轨迹、冻结 logits 和性能对照仍待独立新环境验证。
旧的 56 项比较中 10 项失败保持 FAIL；本轮尚未确定其原因，不能归为 BF16 误差。
