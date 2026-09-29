# SGLang system-test 相对 main 的代码、参数与无功能路径审计

审计日期：2026-09-29。仓库：Rancy-Wang/sglang，工作分支：`system-test`。

## 1. 同步与比较口径

- 固定基线 `main = origin/main = 9d0a8d75364ea4571e05ba2e37227ec2579324f2`。
- 本轮开始的 `system-test = 1dbda383ea`，工作树干净。
- `git fetch origin` 后的 `origin/system = f6483fd509`；本地 `system = 320ccb33d6` 不是最新。
- 以以下四个提交为来源合入运行时和依赖；按用户要求，保留测试分支原有测试、benchmark 与 README，本次不更新这些文件：

| 提交 | 功能 |
| --- | --- |
| `9bb4d7b2c0` | 原生 GPT-OSS Harmony prompt 编码与 token 流工具解析；六份依赖清单升级 openai-harmony |
| `eef468c4b1` | `/server_info` 的 scheduler KV、idle 与负载遥测 |
| `732c0801d6` | 移除最后一次 assistant final 之前的 analysis，保留未完成工具调用轮次的 analysis |
| `f6483fd509` | Mooncake intra-node NVLink worker 与线程池线程绑定本 rank 的 GPU |

合并无冲突。`python/` 与 `3rdparty/` 的最终内容与 `origin/system` 相同。
本轮同步实际涉及 **11 个实现/依赖文件：新增 2、修改 9**；审计文档另计。
源提交的测试变更不合入；本轮曾尝试的 benchmark adapter 修复及配套测试也已撤回。
`git diff HEAD -- test benchmark README.md`（提交前 HEAD 为原 system-test）为空，确保原有内容不变。

比较使用 **`git diff main -- python/sglang python/pyproject*.toml 3rdparty/amd/wheel/sglang/pyproject.toml` 的最终文件内容**，
不是仅列本轮同步，也不是把提交数量当作文件差异。
按“支持系统功能的实现代码”口径：**57 个文件，新增 20、修改 37、删除 0**。
另有 **6 个依赖配置文件修改**；合计 **63 个文件，新增 20、修改 43、删除 0**。
测试、fixtures、benchmark/实验驱动、历史文档和本报告均不计入系统代码清单，也不作为新增系统功能统计。
历史 `docs/context_system/source_inventory.json`、`migration.md`、`r2_status_20260918.md` 是旧时点记录，不能替代当前源码。

## 2. 所有改动支持的功能

### 2.1 请求协议与完整模板 provenance

`/v1/chat/completions` 增加 `drop_message`、`drop_rule`、`reposition`。
完整历史先经过模型模板得到唯一 token 流与 message owners，再把公开 message ID 转为精确 token 边界；
不是分别 tokenize 每条消息后拼接。Jinja 路径追踪字符归属及 tokenizer offset；GPT-OSS 路径从 Harmony token/字节边界恢复归属。
MiniMax M2.7 的模板宏与部分 RoPE 也有专门处理。

- `message_drop`：在指定消息的 prefill 完成后，使指定历史消息对之后的 query 不可见。
- `text_drop`：按消息对齐的字符串/子串列表选择内容，`occurrence` 从 1 开始；默认第 1 次匹配。最后一条 user 消息完成后触发。
- `keep_text_drop`：`messages` 给出保留后的有序内容，`full_messages` 给出完整历史；按协议字段与内容匹配，从完整 token 流删除未保留部分。
  部分保留会保留模板结构与跨边界 token。默认匹配失败即报错；`force: true` 在投影失败时退回可见历史重新计算，并不保证复用原 KV。
- `thinking_drop`：保留可定位的 assistant `reasoning_content` 或起始 `<think>...</think>`，然后在所属消息完成后 drop 对应 KV。
  同一条消息同时提供两种 thinking 来源会拒绝。
- `reposition`：递增且无重复的公开 message ID；在该消息边界重排幸存位置，后续 token 从新的位置继续。
  同边界先 Drop 再 Reposition。Drop 单独不改变幸存 token 的绝对位置。

消息 ID 从 0 开始。`drop_message` 与 `drop_rule` 不能同时传。
`drop_message: {}` 仍会进入 Context；`reposition: []` 单独传入不会触发 Context。

### 2.2 IR、Radix 身份与可复用 KV

新增 CPU C++/TVM-FFI 编译器，生成 TOKEN/DELTA/REPOSITION 结构记录、birth/final positions、可见期限和转换区间。
公开 metadata 经过整数类型、边界和 narrowing 检查；核心位置/区间为 int32，不能把超范围整数绕回有效槽位。
通过原生 tensor IPC 传递编译结果；PD rebootstrap 可使用 JSON wire。

Radix key 同时记录 token、Drop/Reposition 历史与最终位置。无事件的兼容前缀可共享原生 namespace；
历史不同不能仅凭 token 相同误复用。
Retry 查找最长兼容来源，允许处理位置不同的 KV，返回精确前缀、源位置、Full/SWA 驻留信息。
这一实现依据真实缺页与依赖闭包决定重算区间，**没有一个供用户设置的“命中率低于 X 就全量重算”参数**。

### 2.3 执行、所有权与长历史

Prefill 将 query 按事件阶段组织为 native Triton 可消费的分段 attention 元数据；
同一个 raw token 可有 birth、转移、terminal 等不同位置版本，按需独立分配、复制和旋转。
Reposition 对 K 做位置变换、V 做复制，支持 Full/SWA 池和 MiniMax partial RoPE，保护原始共享源页。
临时页在完成屏障后释放；未物化的 terminal 版本不能提前发布给 Radix 或 D 端。

调度器把额外 COW 页、后续执行进度与 decode 保留量纳入 admission；容量不足时可缩小 chunk、释放自锁来源后冷重试、等待其他请求释放，
或返回容量错误。保留完整 raw 历史与 active KV 的区别：raw 历史超过默认请求表宽度时，Context 请求可使用额外 raw row，
但所有中间 RoPE 位置、active KV 与模型上下文/物理容量仍须合法，不能把 Reposition 当作任意越界许可。

Decode 使用 active KV 索引与逻辑位置；SWA 按位置距离计算，不能按压缩后的数组下标代替。
支持普通请求与 Context 请求混合、chunk/overlap、原生 CUDA graph 路径及 retraction 后重建。
这些是源码实现范围；本轮没有重新进行模型/GPU 验收。

### 2.4 Drop-aware eviction 与缺页恢复

`--context-drop-aware-eviction` 默认 `False`。
启用后，对已证明不会再读且真实匹配路径已有 Drop 证据的 KV，把物理页引用换成路径引用，保留树结构而释放可回收页。
共享读者、Full/SWA lease、路径引用与缺页标记共同防止提前释放或重复释放；后续按实际 query 依赖补回所需缺页。

**实际回收顺序是 proven Drop 候选优先、普通 leaf 随后**：`DropEvictionCandidates.pop()` 遍历 `(1, 0)`。
`memory.py` 当前 CLI 帮助里 “Reclaim leaves before ... internal pages” 与实现不一致，应按实现及测试理解。
本轮未改动来自 system 的参数帮助或运行时策略。
不传这个开关只关闭该主动 Drop lease 优化，不能关闭整个 Context 系统，也不能恢复 main 的所有代码。

### 2.5 P/D 分离与 Mooncake

Context P/D 使用 Mooncake，传递最终 active、已稳定的 KV，而不是所有 raw 历史/临时 occurrence。
D 端可以启用 Radix，并通过复用 bitmap 只传缺页；位置变化的复用仍要求独立 COW。
metadata 携带 Context 身份及 usage，接收端核验，支持 rebootstrap/retraction 计数。
容量失败会等待 P 端各 rank 的传输排空后通知 D，避免页仍在写入时被重用。

通用 Mooncake 改动还包括：每次底层 batch 最多 1024 个传输描述符，失败即停止；
abort 时按 chunk 减 outstanding，不提前清掉同 room 的其他 worker；
`INTRA_NODE_NVLINK` 的 worker 与 executor initializer 执行 `torch.cuda.set_device(kv_args.gpu_id)`。
这些通用修复也作用于没有 Drop 的请求。

### 2.6 GPT-OSS Harmony 与其他通用修复

- GPT-OSS chat 输入使用 `HarmonyEncoder` 和 `openai-harmony==0.0.8`，无 Drop 请求也启用。
  system/developer 指令、工具 schema、assistant analysis/final/tool call、tool result 都按 Harmony 编码；畸形工具 arguments 字符串原样保留。
- 普通请求和 message Drop/Reposition 使用相同 reasoning 清理策略：删除最后一个 assistant final 之前的 analysis；
  保留尚未完成工具轮次的 analysis。只有显式 `thinking_drop` 要保留这些 analysis 以编译 Drop。
- 开启 GPT-OSS reasoning/tool parser 后，从 `output_ids` 解析增量/累计 token；支持 recipient/channel 两种顺序、Unicode、多个 choice 状态、
  多工具调用、重复累计片段、终止原因与有限 legacy fallback。未完成的 length/abort 不凭空完成工具调用。
  工具调用以完整调用的形式发出；不能假定仍按 JSON 参数逐字符输出。
- 旧 `harmony_parser.py` 仍保留空白正则回溯修复；它与新 `harmony_chat_parser.py` 是两个文件。
- SM80/A800 的 native Triton MXFP4 小 expert slice：满足 dtype/大小且没有显式 num_warps 约束时临时设 4 warps，之后恢复约束。
  它同样不依赖 Drop。
- Humming 增加 `SGLANG_HUMMING_USE_BATCH_INVARIANT`，默认 false，传入 dense/MoE compute 和 tuning。
  它不是 Context 对 `--enable-deterministic-inference` 限制的豁免，也不等价于整个模型完全确定性。
- 两个 JIT C++ header 修复限定 `tvm::ffi::get` 与补 `<cassert>`。

### 2.7 服务端可观测性

Context 响应 `sglext.context_usage[choice_index]` 提供 `cached_tokens`、`repos_tokens`、`drop_skipped_tokens`、
`actual_prefill_tokens`、`actual_decode_tokens`；按完成的 forward 计数，重算会计入，缓存读取使用集合语义。
无 Context 请求不提供这组计数，不能把缺失当作零。
流式请求使用 `stream_options.include_usage: true`，从 `sglext.context_usage` 读取；
非流式使用 `return_meta_info: true`，从 `choices[i].meta_info.context_usage` 读取（不是同一个响应字段）。
`/server_info` 新增每 scheduler/DP 记录的 `pd_telemetry`；TP 镜像不能重复相加，request row 数也不是 token slot 数。


## 3. 如何启动及发请求

以下为 Linux + NVIDIA CUDA 的源码适用配置示例；模型文件、依赖和显存需已就绪，TP 按实际卡数与模型设置。
命令是本次按代码核对的用法，不表示本轮已启动 GPU 服务。

### 3.1 单实例服务器

在运行服务器的 Python 环境安装新依赖并重启旧进程：

```bash
python -m pip install 'openai-harmony==0.0.8'
python -c 'import importlib.metadata; print(importlib.metadata.version("openai-harmony"))'
```

Qwen3 / AgenticQwen（对应受支持 Qwen3 架构）：

```bash
CUDA_VISIBLE_DEVICES=0 SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3-model --tp-size 1 --dtype bfloat16 \
  --attention-backend triton --page-size 1 \
  --reasoning-parser qwen3 --tool-call-parser qwen25 \
  --host 127.0.0.1 --port 30000
```

GPT-OSS：

```bash
CUDA_VISIBLE_DEVICES=0,1 SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
python -m sglang.launch_server \
  --model-path /path/to/gpt-oss-model --tp-size 2 --dtype bfloat16 \
  --attention-backend triton --page-size 1 \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --disable-hybrid-swa-memory \
  --host 127.0.0.1 --port 30000
```

这里的共享 Full/SWA 设置便于与 D-cache 配置一致；不是说普通实例全部都必须关闭 hybrid SWA。
执行 dtype 与 KV 要 FP16/BF16；这不等于所有权重必须 BF16，GPT-OSS 的 MXFP4 权重路径有单独实现。
如需 proactive Drop 回收，在上述命令加 `--context-drop-aware-eviction`。
只需 Harmony 工具修复、不使用 Context 时，不必为了此修复强制 page1/Triton/shared-SWA。

MiniMax M2.7：使用同样的 `--attention-backend triton --page-size 1 --dtype bfloat16`，
模型换成 M2.7，parser 改为 `--reasoning-parser minimax --tool-call-parser minimax-m2`，TP 按模型与卡数配置。
参数 parser 的名称来自当前 registry；模型 admission 还检查 `MiniMaxM2ForCausalLM`、全 attention、合法偶数 partial rotary_dim。

### 3.2 功能开关在请求体，不是 `--drop` / `--repos`

服务器没有新增通用 `--drop`、`--repos` 或 `--enable-context-system`。
`--context-drop-aware-eviction` 是唯一新增的 Context 服务器 CLI 字段；以下请求体才决定是否编译 Context。

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [
      {"role":"user","content":"旧资料 ABC"},
      {"role":"assistant","content":"已阅读"},
      {"role":"user","content":"现在回答新问题"}
    ],
    "drop_message": {"2":[0]},
    "reposition": [2],
    "max_tokens": 64,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

`model` 可换成服务实际名称。这个例子在消息 2 完成后 drop 消息 0，再执行 Reposition。
去掉 `reposition` 就是 Drop-only；去掉 `drop_message` 就是 Reposition-only（未曾丢弃时未必产生位置变化）。
使用 OpenAI Python 客户端时，这些扩展放在 `extra_body` 中。

等价 message rule：

```json
{"drop_rule":{"type":"message_drop","drop_messages":{"2":[0]}},"reposition":[2]}
```

上例的 text Drop（只选择 ABC）：

```json
{"drop_rule":{"type":"text_drop","drop_messages":[
  {"role":"user","content":"ABC","occurrence":1},
  {"role":"assistant","content":null},
  {"role":"user","content":null}
]}}
```

Keep projection 示例，作为同一个 chat 请求的 `messages` 与 `drop_rule` 字段：

```json
{
  "messages":[{"role":"user","content":"保留片段"},{"role":"user","content":"新问题"}],
  "drop_rule":{"type":"keep_text_drop","full_messages":[
    {"role":"user","content":"旧资料；保留片段"},
    {"role":"assistant","content":"旧答复"},
    {"role":"user","content":"新问题"}
  ],"force":false}
}
```

Thinking Drop：消息中必须有可映射的 thinking 来源，然后传 `"drop_rule":{"type":"thinking_drop"}`。
GPT-OSS 的 `reasoning_effort` 在 Harmony 路径只接受 `low`、`medium`、`high`；默认 medium。
`enable_thinking:false` 在未指定 effort 时映射为 low，不是完全关闭 reasoning。
新 Harmony 路径拒绝 `continue_final_message` 和请求 `chat_template_kwargs.chat_template`。
服务器上旧的 retained Jinja 文件不会让新 Harmony 路径恢复“保留全部 analysis”的策略。

### 3.3 P/D 与可选开关

P、D 两端同一模型、dtype、TP 与 Context 支持配置，分别启动在不同 GPU/端口。例如共享参数为上面的 GPT-OSS 参数：

```bash
# P: 在一个终端，GPU 0,1
CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
  --model-path /path/to/gpt-oss-model --tp-size 2 --dtype bfloat16 \
  --attention-backend triton --page-size 1 --disable-hybrid-swa-memory \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 --port 30001 --host 127.0.0.1

# D: 在另一个终端，GPU 2,3
CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
  --model-path /path/to/gpt-oss-model --tp-size 2 --dtype bfloat16 \
  --attention-backend triton --page-size 1 --disable-hybrid-swa-memory \
  --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --disaggregation-decode-enable-radix-cache --port 30002 --host 127.0.0.1
```

仍需 PD router 或仓库 `Transport` 这种配对客户端，为 P/D 使用同一 bootstrap room 与匹配的请求体；
不能仅把普通 curl 打到 D 端就视为完成 P/D 路由。`test_serving.py` 用 `--url` 指 D、`--prefill-url` 指 P、
`--bootstrap-port 8998` 配对。D-cache 可通过去掉 `--disaggregation-decode-enable-radix-cache` 关闭。
GPT-OSS Context + D-cache 明确要求 `--disable-hybrid-swa-memory`。

同机 NVLink 可在两端进程启动前设置：

```bash
unset MC_FORCE_TCP
export MC_INTRANODE_NVLINK=1
export MOONCAKE_PROTOCOL=nvlink_intra
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK
```

前提是该主机硬件与 Mooncake 安装支持此传输；此时新线程绑定修复自动生效。
TCP 对照则清除这组 NVLink 环境变量并按实验驱动使用 `MC_FORCE_TCP=1 MOONCAKE_PROTOCOL=tcp`。
不要把同机 NVLink 配置外推为跨主机传输。

| 功能 | 需要的参数 / 条件 |
| --- | --- |
| Drop / Reposition / 四类 rule | 请求字段；server `--page-size 1 --attention-backend triton` |
| Drop-aware eviction | 另加 `--context-drop-aware-eviction`，默认关闭 |
| D 端 Radix 复用 | `--disaggregation-decode-enable-radix-cache`；GPT-OSS Context 需 shared Full/SWA |
| Humming batch invariant | 实际选用 Humming 时设 `SGLANG_HUMMING_USE_BATCH_INVARIANT=1`；影响数值/性能，不是默认 |
| SM80 MXFP4 调优 | native `triton_kernel` MoE 路径及源码条件满足时自动生效，无新增专用 CLI |
| telemetry | 查询 `/server_info`，无需新开关；频繁轮询本身有开销 |
| Context usage | streaming include_usage / nonstream return_meta_info；无 Context 时缺失 |

Context admission 当前只支持指定 Qwen3、GPT-OSS、MiniMax M2 架构、文本 FP16/BF16 执行、未量化 KV、TP；
要求 PP/attention CP/DCP 为 1，非 speculative/DLLM，不启用全局 deterministic inference。
不支持 Context + multimodal、LoRA、session KV、beam、HiCache/external linker/HiSparse/LMCache/FlexKV、自定义 radix backend；
Context Radix 要 Python tree core，不能导出 KV cache events。
Context PD 还拒绝 staging、KV checksum、decode KV offload 等组合。完整门禁见 `capabilities.py` 与 tree core 验证函数。
这些限制仅针对显式 Context 请求，不是整个 SGLang 所有普通请求的限制。

## 4. 不传 drop/reposition，是否与 main 完全一样、没有性能影响？

**不能这样保证；严格来说执行路径已经不相同。** 需要区分四种情况：

| 场景 | 可以确认的事实 | 不能声称的事 |
| --- | --- | --- |
| 非 GPT-OSS，干净服务，全程无 Context | 前端不构建 provenance/IR；不执行 Context COW/Reposition；未激活 registry 时保留原生 attention 主分支 | 全栈逐条指令完全相同或耗时绝对零变化 |
| GPT-OSS，无 Context | 同样不编译 Context IR，但输入变成新 HarmonyEncoder；parser 配置满足时也改用 token-native 输出解析 | prompt tokens、reasoning 历史、工具输出时机与 main 必然相同 |
| 本请求无 Context，但同一 worker 处理过 Context | registry 一旦存在就用于后续 decode，包括普通请求；混合 prefill batch 也会构建普通 lane 的 ContextSequence | “只要这个 HTTP 请求没传参数就完全走 main” |
| PD、SWA 或 SM80 MXFP4 等特殊配置 | 多处通用修复/策略不以 Context 请求为条件 | 用“不传 drop”隔离全部改动 |

具体证据与成本来源：

1. `Req.__init__` 为每个请求初始化多个 Context 字段；prefix key 构造、admission、batch `any(...)` 扫描、row 分配/写入/释放、
   cache 发布和结果处理均增加函数/条件检查。大模型 GPU 时间可能掩盖它们，但源码不能证明没有开销。
2. `UnifiedTreeCore._context_note_child_link` 在普通 add/split 时也维护 `context_descendant_bound`，可向祖先传播，
   不只是“一个 bool 判断”。节点也多了 Context 元数据，CPU 内存布局已变化。
3. `ForwardBatch.init_new` 的条件为“registry 已存在 **或** 本 batch 有 Context”；`ContextDecodeRegistry.bind_batch` 对普通 lane
   仍组装 CPU metadata、传到设备并更新 rows，随后走 Context gather。当前没有自动恢复到未激活状态的代码。
4. mixed prefill 通过 `ContextSequence.ordinary` 纳入统一计划；存在 overflow raw row 时，混合写入可走 row-pointer kernel。
5. `registry.default_radix_cache_factory` 将 `disable_radix_cache && disaggregation_mode == decode` 也导向 unified cache；
   page1 SWA 的 D 端 tail 预分配与 budgets、已结束请求容量计数也改变了原生 PD 行为。
6. P/D 即使普通请求也写零 Context metadata 并读取/校验对应 cells；Mooncake 描述符上限、outstanding 修复和 NVLink GPU 绑定普遍适用。
7. `triton_kernels_moe.py` 的 SM80 MXFP4 调优没有 Drop 条件；SWA graph buffer 从 window 扩大为 window+1；
   Humming 配置字段、Harmony 正则/新解析也不等同于 main。
8. 仅设置 page1/Triton/shared-SWA 来对比默认 main，也已改变实验配置，不能将性能差全部归因于 Drop 系统开销。

因此当前合理表述是：**无 Context 请求通常绕过 Context 的重计算，但并非全栈 main 等价；性能变化方向与幅度需测量。**
这不等于已经测出回退，也不意味着所有新增开销都显著。

若需要“main 基线”，应使用固定 `main` checkout 和匹配依赖；修改版 no-drop 是另一个对照组。
若需要“保留 Harmony 修复的原生基线”，应单独定义 main + 明确通用修复集合，不能仍称 untouched main。
本轮未新增全局禁用 Context 的开关，也未擅自优化这些无功能路径。

建议验收矩阵（本轮未执行）：同模型、dtype/KV/quantization、GPU/TP、attention、page-size、CUDA graphs、chunk/并发、cache 热度，
分别比较 main、修改版全程无 Context、修改版先 Context 后普通、混合请求；GPT-OSS 额外记录 prompt token 哈希与 reasoning 渲染策略。
预热 JIT 后报告吞吐、TTFT/TPOT、CPU 调度时间、显存和多轮方差；不能用旧版本/不同 token 工作量的结果证明零开销。

## 5. 本轮验证与边界

本机 macOS，没有 Linux SRT/CUDA/Mooncake 运行环境。使用 `/opt/anaconda3/envs/myenv/bin/python`，
其 `openai-harmony` 为 0.0.8；仅在 `/tmp/sglang-audit-20260929-deps` 安装 CPU TVM-FFI/ninja，未改项目依赖。
**使用测试验证实现，不代表将测试文件纳入同步或系统代码清单。**

- 撤回所有测试改动后重跑现有 Context 可运行集：`386 passed, 184 skipped, 1 deselected`。
  排除需要原生 Humming runtime、MiniMax tokenizer fixture 的文件与一个原生绑定测试。
- 原生 Harmony parser 7 项 + encoder 2 项：9 项通过。源分支的测试仅复制到临时目录执行，使用真实源码和 openai-harmony，
  绕过包顶层 Linux runtime 初始化；不等于 HTTP 集成通过。
- Mooncake 线程设备回归：3 项通过。测试仅保留在临时目录，从实现源码 AST 加载两个方法，模拟 CUDA set_device；
  验证 worker/线程池顺序，不覆盖真实 NVLink 传输。
- 实现范围内全部 Python 文件通过 AST 语法检查；`git diff --check` 通过。
- 首次全 Context 尝试的环境失败包括 PYTHONPATH、原生 Humming/绑定模块及缺失 MiniMax tokenizer；
  纠正可修复的路径问题后，采用上述明确排除项的可运行集，不宣称全套通过。
- 原有 benchmark 有已在合并前 HEAD 复现的 CASES 数量断言失败（预期 10、实际 12）。它不属于本次实现同步范围。
- 原有 benchmark 的 `NativeTemplateAdapter` 跳过服务器构造函数，未初始化新 `is_gpt_oss` 属性，也未接入原生 Harmony；
  若使用该客户端的本地原生模板路径，需要后续单独适配。按“不含测试文件”的要求，本轮不改它；模型 fixture 相关测试跳过也不能证明此路径可用。
- Linux 完整 serving_chat 单测、真实模型前端、GPU kernels、服务端混合调度、P/D 与吞吐均未在本轮执行；不能将 skip 记为 PASS。

主要命令（本地绝对路径仅记录复现环境）：

```bash
PATH=/tmp/sglang-audit-20260929-deps/bin:$PATH \
PYTHONPATH=/tmp/sglang-audit-20260929-deps:python:benchmark/context_system \
/opt/anaconda3/envs/myenv/bin/python -m pytest -q test/registered/context_system \
  --ignore=test/registered/context_system/test_humming_batch_invariant.py \
  --ignore=test/registered/context_system/test_minimax_context_frontend.py \
  -k 'not test_decode_reuse_binding_supports_partial_native_rope'

/opt/anaconda3/envs/myenv/bin/python /tmp/sglang-audit-20260929/check_harmony.py
/opt/anaconda3/envs/myenv/bin/python /tmp/sglang-audit-20260929/check_mooncake.py
git diff --check
git diff --exit-code origin/system -- python 3rdparty
```

临时 harness/log 存在 `/tmp/sglang-audit-20260929/`；不会提交机器临时资产。
下面的符号定位及文件表仅包含系统实现和必需依赖。

## 6. 关键源码定位

以下行号对应本轮最终工作树，修改后应重新核对。

| 文件与范围 | 符号 |
| --- | --- |
| [python/sglang/srt/entrypoints/openai/serving_chat.py:833-841](python/sglang/srt/entrypoints/openai/serving_chat.py#L833) | `_uses_harmony_chat_parser` |
| [python/sglang/srt/entrypoints/openai/serving_chat.py:1324-1519](python/sglang/srt/entrypoints/openai/serving_chat.py#L1324) | `_process_messages` |
| [python/sglang/srt/entrypoints/openai/serving_chat.py:1521-1877](python/sglang/srt/entrypoints/openai/serving_chat.py#L1521) | `_apply_jinja_template` |
| [python/sglang/srt/context_system/capabilities.py:10-98](python/sglang/srt/context_system/capabilities.py#L10) | `validate_context_request` |
| [python/sglang/srt/context_system/planner.py:48-82](python/sglang/srt/context_system/planner.py#L48) | `resolve_reposition_token_boundaries` |
| [python/sglang/srt/context_system/planner.py:604-661](python/sglang/srt/context_system/planner.py#L604) | `compile_chat_program` |
| [python/sglang/srt/context_system/rules.py:164-248](python/sglang/srt/context_system/rules.py#L164) | `MessageDropRule` |
| [python/sglang/srt/context_system/rules.py:252-464](python/sglang/srt/context_system/rules.py#L252) | `TextDropRule` |
| [python/sglang/srt/context_system/rules.py:468-719](python/sglang/srt/context_system/rules.py#L468) | `KeepTextDropRule` |
| [python/sglang/srt/context_system/rules.py:723-810](python/sglang/srt/context_system/rules.py#L723) | `ThinkingDropRule` |
| [python/sglang/srt/context_system/rules.py:816-849](python/sglang/srt/context_system/rules.py#L816) | `parse_drop_rule` |
| [python/sglang/srt/context_system/ir.py:117-282](python/sglang/srt/context_system/ir.py#L117) | `compile_context_layout` |
| [python/sglang/srt/context_system/ir.py:300-436](python/sglang/srt/context_system/ir.py#L300) | `ContextKeyData` |
| [python/sglang/srt/context_system/recovery.py:33-70](python/sglang/srt/context_system/recovery.py#L33) | `DropEvictionCandidates` |
| [python/sglang/srt/context_system/recovery.py:253-391](python/sglang/srt/context_system/recovery.py#L253) | `plan_recovery` |
| [python/sglang/srt/context_system/occurrence.py:470-918](python/sglang/srt/context_system/occurrence.py#L470) | `OccurrenceState` |
| [python/sglang/srt/context_system/occurrence.py:968-995](python/sglang/srt/context_system/occurrence.py#L968) | `ContextPrefillCompletion` |
| [python/sglang/srt/context_system/request_storage.py:247-272](python/sglang/srt/context_system/request_storage.py#L247) | `prepare_request_row` |
| [python/sglang/srt/context_system/request_storage.py:286-309](python/sglang/srt/context_system/request_storage.py#L286) | `write_request_slots` |
| [python/sglang/srt/context_system/request_storage.py:312-334](python/sglang/srt/context_system/request_storage.py#L312) | `validate_positions` |
| [python/sglang/srt/managers/schedule_batch.py:1593-1622](python/sglang/srt/managers/schedule_batch.py#L1593) | `make_prefix_key` |
| [python/sglang/srt/managers/schedule_batch.py:3249-3321](python/sglang/srt/managers/schedule_batch.py#L3249) | `_prepare_context_occurrences` |
| [python/sglang/srt/managers/schedule_batch.py:3869-3970](python/sglang/srt/managers/schedule_batch.py#L3869) | `prepare_for_decode` |
| [python/sglang/srt/managers/schedule_policy.py:755-841](python/sglang/srt/managers/schedule_policy.py#L755) | `_fit_context_admission` |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1869-1878](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1869) | `_context_note_child_link` |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:1042-1149](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L1042) | `_match_context_retry` |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:659-771](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L659) | `configure_context_drop_lock` |
| [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:990-1009](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py#L990) | `_validate_context_key` |
| [python/sglang/srt/mem_cache/registry.py:80-149](python/sglang/srt/mem_cache/registry.py#L80) | `default_radix_cache_factory` |
| [python/sglang/srt/model_executor/forward_batch_info.py:840-1129](python/sglang/srt/model_executor/forward_batch_info.py#L840) | `init_new` |
| [python/sglang/srt/layers/attention/context_backend.py:568-684](python/sglang/srt/layers/attention/context_backend.py#L568) | `ContextDecodeRegistry` |
| [python/sglang/srt/layers/attention/context_backend.py:588-639](python/sglang/srt/layers/attention/context_backend.py#L588) | `bind_batch` |
| [python/sglang/srt/disaggregation/decode.py:462-475](python/sglang/srt/disaggregation/decode.py#L462) | `_uses_swa_tail_prealloc` |
| [python/sglang/srt/disaggregation/context_transfer.py:270-286](python/sglang/srt/disaggregation/context_transfer.py#L270) | `write_context_metadata` |
| [python/sglang/srt/disaggregation/context_transfer.py:289-310](python/sglang/srt/disaggregation/context_transfer.py#L289) | `commit_context_metadata` |
| [python/sglang/srt/disaggregation/mooncake/conn.py:651-677](python/sglang/srt/disaggregation/mooncake/conn.py#L651) | `_transfer_data` |
| [python/sglang/srt/disaggregation/mooncake/conn.py:1876-1884](python/sglang/srt/disaggregation/mooncake/conn.py#L1876) | `init_transfer_thread_device` |
| [python/sglang/srt/parser/gpt_oss_encoding.py:66-534](python/sglang/srt/parser/gpt_oss_encoding.py#L66) | `HarmonyEncoder` |
| [python/sglang/srt/parser/gpt_oss_encoding.py:281-319](python/sglang/srt/parser/gpt_oss_encoding.py#L281) | `_drop_harmony_analysis_before_last_final` |
| [python/sglang/srt/parser/harmony_chat_parser.py:275-452](python/sglang/srt/parser/harmony_chat_parser.py#L275) | `HarmonyChatParser` |
| [python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py:33-63](python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py#L33) | `_small_ampere_mxfp4_warps` |

## 7. 相对 main 的系统实现与依赖完整清单

A = 新增，M = 修改；行数为 git numstat 的新增/删除行。测试、benchmark、文档不在此清单内。

### 运行时与 kernels（57 个；A=20，M=37）

| 状态 | 文件 | + / − | 用途 |
| --- | --- | --- | --- |
| M | [python/sglang/kernels/jit/csrc/distributed/ipc.cuh](python/sglang/kernels/jit/csrc/distributed/ipc.cuh) | 2 / 2 | 限定 tvm::ffi::get，修正 IPC header 构建 |
| M | [python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh](python/sglang/kernels/jit/include/sgl_kernel/distributed/ptx.cuh) | 1 / 0 | 补充 assert 所需的 cassert include |
| A | [python/sglang/kernels/ops/attention/context_page_table.py](python/sglang/kernels/ops/attention/context_page_table.py) | 76 / 0 | Decode active/raw 索引映射、SWA 位置窗口 Triton kernels |
| A | [python/sglang/kernels/ops/attention/context_plan.py](python/sglang/kernels/ops/attention/context_plan.py) | 832 / 0 | CPU C++/TVM-FFI 事件、位置、Radix 记录及文本匹配编译器 |
| A | [python/sglang/kernels/ops/attention/context_reposition.py](python/sglang/kernels/ops/attention/context_reposition.py) | 236 / 0 | 全层 K RoPE 变换与 V 复制，支持 partial RoPE/SWA 跳过缺页 |
| M | [python/sglang/kernels/ops/attention/extend_attention.py](python/sglang/kernels/ops/attention/extend_attention.py) | 69 / 9 | 原生 extend attention 增加可选 query/KV 绝对位置窗口 |
| M | [python/sglang/kernels/ops/memory/common.py](python/sglang/kernels/ops/memory/common.py) | 6 / 5 | request KV 写入 kernel 支持可选独立 row pointer |
| A | [python/sglang/kernels/ops/memory/context_rows.py](python/sglang/kernels/ops/memory/context_rows.py) | 15 / 0 | overflow raw row 的解码 slot scatter kernel |
| M | [python/sglang/srt/arg_groups/fields/memory.py](python/sglang/srt/arg_groups/fields/memory.py) | 6 / 0 | 新增默认关闭的 context_drop_aware_eviction CLI 字段 |
| A | [python/sglang/srt/context_system/__init__.py](python/sglang/srt/context_system/__init__.py) | 1 / 0 | Context System 模块入口说明 |
| A | [python/sglang/srt/context_system/capabilities.py](python/sglang/srt/context_system/capabilities.py) | 98 / 0 | 显式 Context 的模型/配置/请求 admission 限制 |
| A | [python/sglang/srt/context_system/ir.py](python/sglang/srt/context_system/ir.py) | 436 / 0 | ContextLayout、CPU 编译封装、结构化 ContextKeyData |
| A | [python/sglang/srt/context_system/occurrence.py](python/sglang/srt/context_system/occurrence.py) | 1006 / 0 | 阶段位置版本、COW 所有权、terminal 发布与完成后释放 |
| A | [python/sglang/srt/context_system/planner.py](python/sglang/srt/context_system/planner.py) | 661 / 0 | message owner 到 Drop/Repos token 边界、ContextProgram 与 IPC/JSON wire |
| A | [python/sglang/srt/context_system/provenance.py](python/sglang/srt/context_system/provenance.py) | 407 / 0 | 完整 Jinja 一次渲染与 token/字符 owner；MiniMax 宏识别 |
| A | [python/sglang/srt/context_system/recovery.py](python/sglang/srt/context_system/recovery.py) | 391 / 0 | 缺页依赖闭包、SWA read demand、Drop 候选 heap |
| A | [python/sglang/srt/context_system/request_storage.py](python/sglang/srt/context_system/request_storage.py) | 334 / 0 | raw row 溢出、位置/容量校验、进度预留及冷重试 |
| A | [python/sglang/srt/context_system/retry.py](python/sglang/srt/context_system/retry.py) | 93 / 0 | 按事件身份寻找最长兼容 Radix 来源 |
| A | [python/sglang/srt/context_system/rules.py](python/sglang/srt/context_system/rules.py) | 1246 / 0 | message/text/keep_text/thinking 四类 Drop rule 与精确文本匹配 |
| A | [python/sglang/srt/context_system/thinking_template.py](python/sglang/srt/context_system/thinking_template.py) | 89 / 0 | 按请求保留 thinking 历史的模板变换，含 MiniMax guard |
| A | [python/sglang/srt/context_system/usage.py](python/sglang/srt/context_system/usage.py) | 117 / 0 | 缓存实际读取/旋转/跳过及 PF/D 完成计数 |
| A | [python/sglang/srt/disaggregation/context_transfer.py](python/sglang/srt/disaggregation/context_transfer.py) | 310 / 0 | PD active 终态传输、稀疏复用 bitmap、COW 与 metadata |
| M | [python/sglang/srt/disaggregation/decode.py](python/sglang/srt/disaggregation/decode.py) | 179 / 20 | Context D admission/预分配/恢复/接收；page1 SWA tail 预算 |
| M | [python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py](python/sglang/srt/disaggregation/decode_schedule_batch_mixin.py) | 3 / 1 | PD batch 拷贝使用可能溢出的 raw request row |
| M | [python/sglang/srt/disaggregation/mooncake/conn.py](python/sglang/srt/disaggregation/mooncake/conn.py) | 73 / 12 | Mooncake 稀疏传输、1024 描述符上限、abort outstanding、线程 GPU 绑定 |
| M | [python/sglang/srt/disaggregation/prefill.py](python/sglang/srt/disaggregation/prefill.py) | 84 / 4 | Context 稳定终态 chunk 发送、容量失败 drain 和完成释放 |
| M | [python/sglang/srt/disaggregation/utils.py](python/sglang/srt/disaggregation/utils.py) | 3 / 0 | PD metadata 写入 Context header/usage 或普通请求零标记 |
| M | [python/sglang/srt/entrypoints/openai/protocol.py](python/sglang/srt/entrypoints/openai/protocol.py) | 8 / 0 | chat Drop/Repos 请求字段及 Context usage 响应结构 |
| M | [python/sglang/srt/entrypoints/openai/serving_chat.py](python/sglang/srt/entrypoints/openai/serving_chat.py) | 303 / 39 | Context 编译入口、Harmony 编码/流式与非流式解析及 usage |
| M | [python/sglang/srt/environ.py](python/sglang/srt/environ.py) | 1 / 0 | Humming batch-invariant 环境开关 |
| A | [python/sglang/srt/layers/attention/context_backend.py](python/sglang/srt/layers/attention/context_backend.py) | 684 / 0 | 分段 attention 计划、模型 RoPE/pool binding、COW、decode registry |
| M | [python/sglang/srt/layers/attention/triton_backend.py](python/sglang/srt/layers/attention/triton_backend.py) | 135 / 6 | Context prefill 分支、decode gather、SWA 和 graph metadata 集成 |
| M | [python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py](python/sglang/srt/layers/moe/fused_moe_triton/triton_kernels_moe.py) | 58 / 20 | SM80 小 MXFP4 expert GEMM 的 4-warp 调优及恢复 |
| M | [python/sglang/srt/layers/moe/moe_runner/humming.py](python/sglang/srt/layers/moe/moe_runner/humming.py) | 3 / 0 | Humming MoE compute/tuning 传递 batch-invariant 配置 |
| M | [python/sglang/srt/layers/quantization/humming.py](python/sglang/srt/layers/quantization/humming.py) | 1 / 0 | Humming dense compute 传递 batch-invariant 配置 |
| M | [python/sglang/srt/managers/io_struct.py](python/sglang/srt/managers/io_struct.py) | 7 / 0 | ContextProgram 经原生请求/并行采样/IPC 传递 |
| M | [python/sglang/srt/managers/schedule_batch.py](python/sglang/srt/managers/schedule_batch.py) | 440 / 12 | Req Context 生命周期、key/recovery/COW、混合批次、retraction/SWA |
| M | [python/sglang/srt/managers/schedule_policy.py](python/sglang/srt/managers/schedule_policy.py) | 204 / 26 | prefix key/Retry 匹配、COW 容量 admission、chunk/进度预算 |
| M | [python/sglang/srt/managers/scheduler.py](python/sglang/srt/managers/scheduler.py) | 103 / 6 | Context 请求与容量失败处理；server_info 的 PD telemetry |
| M | [python/sglang/srt/managers/scheduler_components/batch_result_processor.py](python/sglang/srt/managers/scheduler_components/batch_result_processor.py) | 4 / 0 | PF/D 完成屏障后的 occurrence 释放和计数 |
| M | [python/sglang/srt/managers/scheduler_components/output_streamer.py](python/sglang/srt/managers/scheduler_components/output_streamer.py) | 6 / 0 | 输出每请求 Context usage 标量快照 |
| M | [python/sglang/srt/managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py) | 32 / 3 | Context capability/位置校验、禁止隐式截断、IPC、usage 汇聚 |
| M | [python/sglang/srt/managers/utils.py](python/sglang/srt/managers/utils.py) | 16 / 1 | Context active/中间位置校验，避免 raw-history 自动截断 |
| M | [python/sglang/srt/mem_cache/allocation.py](python/sglang/srt/mem_cache/allocation.py) | 3 / 0 | KV index 写入传入可选 Context row pointers |
| M | [python/sglang/srt/mem_cache/allocator/swa.py](python/sglang/srt/mem_cache/allocator/swa.py) | 13 / 0 | Full 分配与 active SWA tail 分配解耦 |
| M | [python/sglang/srt/mem_cache/base_prefix_cache.py](python/sglang/srt/mem_cache/base_prefix_cache.py) | 35 / 1 | match/insert/lease/result 扩展及 raw row 释放入口 |
| M | [python/sglang/srt/mem_cache/memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py) | 12 / 1 | 分配/写入/释放请求槽时处理 overflow row |
| M | [python/sglang/srt/mem_cache/radix_cache.py](python/sglang/srt/mem_cache/radix_cache.py) | 108 / 2 | RadixKey 编码 Context 事件/位置，保留兼容普通前缀 |
| M | [python/sglang/srt/mem_cache/registry.py](python/sglang/srt/mem_cache/registry.py) | 4 / 1 | 禁用 D Radix 时也选择可处理 Context 所有权的 unified cache |
| M | [python/sglang/srt/mem_cache/unified_cache/components/full.py](python/sglang/srt/mem_cache/unified_cache/components/full.py) | 33 / 4 | Full drop 候选/路径 lease/缺页计数及释放 |
| M | [python/sglang/srt/mem_cache/unified_cache/components/swa.py](python/sglang/srt/mem_cache/unified_cache/components/swa.py) | 146 / 1 | SWA 独立 residency、精确 read-range lease、稀疏发布/释放 |
| M | [python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py](python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py) | 574 / 14 | Context Retry、缺页树节点、稀疏插入/拆分、lease 与回收 |
| M | [python/sglang/srt/mem_cache/unified_radix_cache.py](python/sglang/srt/mem_cache/unified_radix_cache.py) | 380 / 31 | Context 匹配/发布/驻留/私有与借用所有权、source lease、回收 |
| M | [python/sglang/srt/model_executor/forward_batch_info.py](python/sglang/srt/model_executor/forward_batch_info.py) | 44 / 0 | 传递 Context prefill metadata，延迟创建并复用 decode registry |
| A | [python/sglang/srt/parser/gpt_oss_encoding.py](python/sglang/srt/parser/gpt_oss_encoding.py) | 534 / 0 | 原生 Harmony prompt 编码、analysis 清理、token/字符 ownership |
| A | [python/sglang/srt/parser/harmony_chat_parser.py](python/sglang/srt/parser/harmony_chat_parser.py) | 452 / 0 | 原生 Harmony output_ids 增量/累计解析、工具/analysis/final 与 fallback |
| M | [python/sglang/srt/parser/harmony_parser.py](python/sglang/srt/parser/harmony_parser.py) | 13 / 5 | 旧 Harmony 文本解析器避免超长空白正则回溯 |

### 必需依赖配置（6 个；A=0，M=6）

| 状态 | 文件 | + / − | 用途 |
| --- | --- | --- | --- |
| M | [3rdparty/amd/wheel/sglang/pyproject.toml](3rdparty/amd/wheel/sglang/pyproject.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
| M | [python/pyproject.toml](python/pyproject.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
| M | [python/pyproject_cpu.toml](python/pyproject_cpu.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
| M | [python/pyproject_npu.toml](python/pyproject_npu.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
| M | [python/pyproject_other.toml](python/pyproject_other.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
| M | [python/pyproject_xpu.toml](python/pyproject_xpu.toml) | 1 / 1 | openai-harmony 固定依赖从 0.0.4 升到 0.0.8 |
