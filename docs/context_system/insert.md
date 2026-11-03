# Execution-order Insert

实现计划：`PLAN-CS-20261103-R1`。本实现保留 `drop_message`，允许重复 Drop，
SWA 沿用原来的位置窗口条件。Qwen3.8/Qwen3.5 的 Linear/Full hybrid 模型暂不支持 Insert，
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

实现入口：`python/sglang/srt/context_system/rules.py:54-124`
（`normalize_message_indices`、`InsertRule`）；
`python/sglang/srt/entrypoints/openai/protocol.py:884-915`
（`validate_context_indices`、`context_index_metadata`）。

## 从逻辑文本到 Radix

沿用完整 native template 的 provenance/消息分割规则，不对历史前缀逐段重新 tokenize。
对完整 logical token 序列建立 execution permutation，generation prompt 最后进入。
消息编号只用于前端事件编译，不进入 token Radix 的插入锚点。

IB 为 `(3, birth_depth, 0)`，IE 为 `(4,0,0)`。birth depth 从 1 开始，计数所有执行过的
真实 token，包括已删除 token 和先前 Insert 的 token；虚拟节点不计数。
IB 不包含 final span。TOKEN 记录仍包含最终位置，D/R 仍使用原有编码。
普通 Append 的位置游标不被 Insert 改写；Drop 不改变幸存位置；有效 R 按 active logical
顺序压缩位置。原有无效 R/tail Drop 的游标行为保留。

同样的 execution prefix、操作和锚点产生相同历史语义。Exact matching 比较完整记录；
retrigger 只忽略 TOKEN 最终位置，不忽略 D/R/IB/IE 或锚点。匹配后延用现有 COW 旋转，
不改写共享源 KV。树边切分不改变 birth depth；IE 作为下一真实 token 的前导事件，
末尾事件保留于请求程序，生成 token 追加时进入 key。

张量 wire 的 Insert 版本为 5，JSON wire 为 6；无 Insert 保持原版本 3/4 和字段集合。
需要所有 worker 使用相同实现，不可将 Insert 请求发给旧 worker。

实现入口：`python/sglang/srt/context_system/planner.py:317-545`（`ContextProgram`）、
`planner.py:547-664`（`_compile_insert_program`）；
`python/sglang/srt/context_system/ir.py:293-350`（`compile_execution_layout`）、
`ir.py:371-505`（`ContextKeyData`）。

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

实现入口：`python/sglang/srt/context_system/occurrence.py:142-305`
（`compile_occurrence_window`）；`python/sglang/srt/context_system/recovery.py:301-379`
（`_plan_insert_recovery`）；
`python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1059-1182`
（`_match_context_retry`）；`python/sglang/srt/layers/attention/context_backend.py:678-833`
（`ContextDecodeRegistry`）。

## 性能边界

- 无 Context 请求继续原生路径；没有 Insert 的 Context 请求继续原有编译器。
- logical/execution 逆映射线性构建。匹配使用预计算 `has_insert`，不复制扫描完整 key。
- decode 的位置单调性、排序位置和 raw 后缀最小值只计算一次。
  CPU 窗口计数/安全回收边界用二分查找，不在每轮 decode 扫描整个历史。
- Reposition 后的位置单调路径继续原生快速窗口 gather。未 Reposition 的非单调路径
  按 logical 顺序压紧选中的 KV；该 GPU 分支仍需扫描非单调列表。
- 非连续位置的原生 ragged SWA adapter 保留单调二分路径；非单调段先筛出首窗口候选，
  后续 query 只检查候选，避免每个 query 全前缀扫描。不存在生产用 N×N attention mask。
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
summary 截断、缺少标题或压缩后仍达到阈值时，不提交历史变更。

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

实际运行结果随验证完成后补充，未完成的端到端实验不能据上述用例描述视为通过。
