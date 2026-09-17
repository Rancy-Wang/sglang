# Drop / Reposition 迁移执行账本

批准计划：`PLAN-CS-20260917-R2`，实施轮次 1。本文记录已批准的目标、证据和剩余工作，
不是功能已经实现或性能已经达标的声明。

## 基线与完成状态

| 项目 | 固定版本 / 状态 |
| --- | --- |
| SGLang 官方基线 | `9d0a8d75364ea4571e05ba2e37227ec2579324f2` |
| Rancy-Wang/sglang main | 已从 `b3570a453` fast-forward 到上述基线并推送；未改写历史 |
| 实施分支 | `system`，从上述基线创建并推送 |
| mini-sglang System | `89d8a9fd22a2784a8989bdd7b80e7a232d5e877e` |
| mini-sglang main | `9a91cfafe754aa85daee49998176275667eb58f2` |
| mini-sglang System-test | `2966eb49a522041f9c42bce7dca07119ef6929de`；只读参考 |
| 历史清单 | 找到 52 份项目相关会话；已建立消息索引，设计决策核对仍在进行 |
| System 相对 main | 75 个文件；逐文件范围见 `source_inventory.json` |
| 生产功能 / GPU 验证 | 尚未完成，不能据此部署或宣称等价 |

基线冻结以后不持续追逐上游移动的 main。后续实验记录实际 `system` commit、模型配置、
权重版本、依赖版本、GPU 映射和后端选择。审计时使用过的 `923e4a56d` 与最终冻结点相比，
本次迁移涉及的 SRT 源码没有变化；kernel 变化仅为 CPU 构建依赖。

## 当前实测状态（2026-09-17）

普通调度的终态发布修复为 `feff22efd`，首轮真实 PD 接线为 `feb289a24`。
`760592534` 进一步修复了 Context 的原生 host-pool 备份读集、SWA 回收和 PD 双池容量
估算。`ec321ae8f` 的真实并发 retract 验证已通过；`83cfdf638` 恢复 P 分块传输后，
同一 BCP 的六条冷/热/Retry 固定路径也通过。以下结果只覆盖 Qwen3-0.6B，不能推断
AgenticQwen、GPT-OSS 或吞吐验收已经通过。

BCP case 778 使用同一规范化原生工具描述、原始 10482 tokens、终态 active 4878 tokens。
mini 固定版本默认 mask/page-occurrence、BF16、FA3；SGLang 使用 BF16/Triton，保留
原生 overlap、CUDA Graph 和 512-token chunk。对齐的是同一模型、权重、输入和固定
64-token 路径；后端不同引入的误差由无功能运行校准。

| 路径 | 普通 SGLang max / mean / p99 | P/D max / mean / p99 |
| --- | --- | --- |
| 无功能 | 1.125 / 0.054271 / 0.21875 | 1.125 / 0.054271 / 0.21875 |
| Drop | 0.5625 / 0.049975 / 0.1875 | 0.5625 / 0.049975 / 0.1875 |
| Drop+R 冷 | 1.125 / 0.065057 / 0.3125 | 1.125 / 0.065057 / 0.3125 |
| Drop+R 热 | 1.046875 / 0.064866 / 0.296875 | 1.125 / 0.065100 / 0.3125 |
| Drop+R Retry | 0.625 / 0.050695 / 0.1875 | 0.625 / 0.050695 / 0.1875 |

每项门限是 `max(绝对下限, 2 × 同模型无功能误差)`；max/mean/p99 下限分别为
0.125/0.02/0.0625。这是本次跨后端 BF16 的诊断标准，不是所有模型的通用误差标准。
R 热命中 cached=4877、repos=0、drop_skipped=5604、实际 prefill=1；Retry 分别为
1508、4122、2850、2002。两种调度均为 63 次实际 decode 加 P/普通 prefill 的首 token。

普通调度先前存在首 decode 使用旧位置 KV 的真实错误（R cold max=16.6875、Retry
max=17.9375）。`ScheduleBatch._prepare_context_occurrences`
（`python/sglang/srt/managers/schedule_batch.py:3232`）在原生 overlap 消费前发布最终
raw→terminal 映射，保留独立 birth 来源，修复后的结果如上。

P/D 的 P、D 分别位于 BUS 物理 GPU 0、1，独立进程，D Radix 关闭，无 D→P 输出传输。
`ContextTransferPlan` 对最终 active 行、位置和事件身份做一致性校验；D 只分配实际
active Full 和必要 SWA 页（`disaggregation/context_transfer.py:124`
`allocate_context_destination`、`:181` `commit_context_metadata`，路径前缀为
`python/sglang/srt/`）。`83cfdf638` 的 P 按完成的 raw chunk 发送对应的终态 active
Full KV（`disaggregation/context_transfer.py:53` `full_chunk`、
`disaggregation/prefill.py:1325` `_send_kv_chunk`），跨 Drop holes 后保持连续目标次序。
请求的原生锁保护已发布终态，未来 birth 页独立写入；保留最后一页随最终元数据/SWA
一同发送，避免传输层凭累计页数提前宣布完成。该版本 BCP 数值为表中 PD 结果，
`1 passed in 144.06s`，含启动时间，不能用作吞吐或传输重叠收益测量。

原生 host-pool retract 验证为 `1 passed in 158.31s`：两个并发 BCP 中一条实际发生
4 次回收恢复，64 个固定输出 token 与 mini 一致；该条 max/mean/p99 为
1.0625/0.068561/0.3125，落在同一无功能基线门限内。实际 decode=67，包含原生 overlap
被丢弃后重放的 4 次 forward，不能误记成 63。初次实验因网络到达错开未触发 retract，
后续仅在测试 probe 中等两条传输就绪；第二次暴露 probe 未回退被丢弃采样的问题，
`ec321ae8f` 令 probe 随原生回收回退到已提交 output 长度，未替换生产 sampler。

BUS 安装的是当前 SGLang Docker 配置对应的 Mooncake CUDA13 0.3.13，实际日志显示
未发现 HCA，使用 TCP 传输。本次证明真实跨 GPU KV 传输的功能，不能视为 RDMA 性能。
首轮 PD 测试为 `1 passed in 190.49s`（含启动，不是吞吐）；普通数值测试为
`1 passed in 85.80s`。结果目录在 BUS 的
`/mnt/public/wangruoxi/local/sglang-system-20260917/`：

- `bcp778-mini-aligned-qwen.json` 和同名 `.json.pt`：复用的 mini 数值参考。
- `bcp778-sg-aligned-qwen/`：普通调度固定路径结果；旧实际生成记录注明为复用。
- `bcp778-sg-pd-qwen/`：P/D 原始响应、分端 logits、comparison.json 和启动日志。
- `bcp778-sg-pd-retract-qwen/`：首次未触发回收的失败证据，保留。
- `bcp778-sg-pd-retract-barrier-qwen/`：probe 未回退导致固定路径跳 token 的失败证据。
- `bcp778-sg-pd-retract-rewind-qwen/`：`ec321ae8f`，4 次真实回收恢复通过。
- `bcp778-sg-pd-chunked-qwen/`：`83cfdf638`，分块传输后的完整 PD BCP 数值矩阵。

可复现的 PD 测试入口：

```bash
CONTEXT_PD_BCP_ORACLE="$EXP/bcp778-mini-aligned-qwen.json" \
CONTEXT_TRACE_DIR="$EXP/bcp778-sg-pd-qwen" \
CONTEXT_SERVER_MODEL="$MODEL_PATH" CONTEXT_TEST_ATTENTION_BACKEND=triton \
CONTEXT_P_GPU=0 CONTEXT_D_GPU=1 \
python -m pytest -s -q test/registered/context_system/test_bcp_pd_numeric.py
```

这里的 EXP 是上述 BUS 实验目录，MODEL_PATH 是 Qwen3-0.6B 的本地权重路径。
需使用已记录的隔离 CUDA13 环境及 compat 库路径；不可用同名输出目录覆盖既有结果。
retract 实验额外设置 `CONTEXT_PD_RETRACT=1` 和指向首轮 comparison.json 的
`CONTEXT_PD_CALIBRATION`，复用无功能校准，只新增两个并发 BCP 请求的抢占恢复检查。
P/D 的 cache、临时目录、端口、日志和 GPU 分开。

实际生成尚未验收：此 BCP 下 mini Qwen3-0.6B 默认工具语法首先输出终止符，但
`ignore_eos` 继续排满 64 tokens；SGLang 的同描述原生结构化语法生成两个 tokens 后
正常终止。不能为凑齐长度替换 SGLang sampler，也不能把固定路径误差通过当作实际
Agentic 回答通过。`9a2bfc811` 将实际生成与固定路径验证拆开，支持
`CONTEXT_BCP_MODE=actual|fixed|all`，自由生成不注入 logit processor，也不读取旧输出
充作本次结果。该版本对三种功能各做一次新实际生成，均得到相同的两个 tokens；
`bcp778-sg-native-actual-qwen/` 保存原始响应与 mini 文本/结束原因对照，执行测试为
`1 passed in 70.77s`，其含义是观察记录完整，不是回答质量或 R2 验收通过。
源码确认 SGLang `managers/schedule_batch.py:2186` 对已终止 grammar 单独结束请求，
不受 `ignore_eos` 控制；mini `engine/tool_grammar.py:180` `_accept` 则在终止后保留
上一份 mask。迁移没有为了凑长度修改这两个原生语法行为。需在必验大模型上完成
实际生成对照，并据其原生协议记录真实终止原因。

GPT-OSS 窗口边界经端到端源码核对，并不存在“mini 128 对 SGLang 129”的差异：
两者模型均把配置 window 转为 `window - 1` 的左窗口。
SGLang `models/gpt_oss.py:129` `get_attention_sliding_window_size` 的结果由
`model_executor/model_runner_components/load_model_utils.py:139`
`resolve_sliding_window_size` 优先采用；mini 的 `models/gpt_oss.py:42` 同样减一。
SGLang 引用前缀为 `python/sglang/srt/`，mini 为 `python/minisgl/`。
该核对不替代 GPT-OSS 全模型数值测试。

GPU 资源阻塞已解除：按用户随后授予的 GPU 0–3 全部任务释放权限，核实宿主进程
归属后停止了前四卡的占用任务。GPU 4–7 的其他训练保留；特权会话已关闭。当前必验
模型实验显式分配 0–3，记录独立进程、输出、端口及实际 HEAD。后续只对新修改和未覆盖
功能运行必要测试，不重复完整旧测试集。

本次分块区间 CPU 测试 `python -m pytest -q
test/registered/context_system/test_context_transfer.py` 为 1 passed。新增 helper 和
BCP 测试的 Ruff 通过；完整 `prefill.py` 的 Ruff 在修改前后均有 29 项相同既有
诊断，无新增项，未为此格式化无关原生代码。全部 52 条会话已建立索引，但逐条完整
审阅及全部源码差异审计尚未完成；本页风险清单不冒充全量审计通过。

## 追加实测与 SWA 运行设置（2026-09-17）

AgenticQwen-8B 的普通与 PD 固定路径均已执行。Retry 必须与 mini 的同源 Retry 对照：
mini 自身 Retry 相对冷算已有差异，不能将两种不同计算路径混为误差基线。使用已保存的
mini Retry logits 复核后，普通及 PD 的 max/mean/p99 为
1.33984375/0.068649568/0.359375，低于同模型无功能校准门限；没有重跑 GPU 来生成
重复证据。原始与匹配对照保存在 `bcp778-mini-agentic-retry.json`、
`bcp778-agentic-matched-retry-comparison.json`。实际生成的原生语法结束行为仍单独记录。

GPT-OSS-20B 普通调度在 `4a726e827` 的 BCP 验证通过，测试耗时 361.42 秒（含启动和
logits 诊断，不是吞吐）。none/Drop/Drop+R 冷算的实际输出各 64 tokens 均与 mini 相同。
输入先对齐到 SGLang 原生 Harmony 模板的 8927 tokens，mini 仅在测试 renderer 中
使用同一输入；这证明计算对齐，不证明两个服务端的默认模板相同。

| GPT-OSS-20B 路径 | max / mean / p99 |
| --- | --- |
| none | 2.7109375 / 0.069612078 / 0.4375 |
| Drop | 4.875 / 0.101740994 / 0.875 |
| Drop+R 冷 | 2.279296875 / 0.064363368 / 0.412109375 |
| Drop+R 热 | 1.734375 / 0.062313691 / 0.390625 |
| 同源 Retry | 1.8125 / 0.060070530 / 0.359375 |

各路径 raw argmax 均 64/64 一致。热路径 cached/repos/skipped/prefill 为
3384/0/5542/1，同源 Retry 为 3252/3916/0/1759。结果位于
`bcp778-sg-gpt20-aligned/`，参考为 `bcp778-mini-gpt20-native-input.json`。
以上是独立 SWA 池旧配置的证据；下述新配置正在补齐验证，不能直接沿用为其验收结果。

用户要求 SWA+Drop+R 使用 mini 机制。mini 的 `python/minisgl/kvcache/mha_pool.py`
`MHAKVCache` 将所有层绑定在同一 token page；SGLang 的独立 SWA pool 会回收窗口外
SWA 同伴页，未来 Retry 可能多做修复。R2 运行配置从 `342daa299` 开始对 GPT-OSS
普通、P、D 显式传入原生 `--disable-hybrid-swa-memory`。它通过
`configs/model_config.py` 的 `ModelConfig._derive_hybrid_model` 关闭独立 SWA pool，
**不关闭 SWA attention**：原生 GPT-OSS 层的窗口、sinks、MoE 和默认 Triton backend
全部保留。mask/page-occurrence 按 query 所在阶段的 true position 选择可见版本，
Retry 与正常 Drop 共用 Full/SWA 页生命周期，来源页只读、目标 RoPE 旋转使用独立页。
生产无功能请求仍走原生计算分支。只开 `--context-drop-aware-eviction` 不会隐式改变
SGLang 的物理池；部署时必须同时采用这里的 SWA 配置。

这一配置使 P/D 均使用同构全层 KV 页，传输最终 active 页；D Radix 仍关闭，无 D→P
输出 KV。它增加了相对独立 SWA 尾窗口传输的字节数，不能宣称已保留该优化的带宽收益。
最小吞吐的原版 SGLang 对照也采用相同物理池设置，避免把不同 KV 容量当成补丁开销。
原生独立 SWA 池仍保留，但不作为本次 mini 等价路径的完成依据。

PD 旧双池测试暴露两个问题：page_size=1 的普通长 prompt 错受 D 的 SWA 小池限制
（8927 > 2052），`ebbd491d9` 增加页大小为一的尾窗口分配；随后已完成请求在 overlap
队列中尚未过滤、Context layout 已释放，admission 访问空 layout 导致 D 退出。
`2c24bf6b9` 将无 KV 所有权的请求排除出回收预算，测试增加不注入 probe 的 PD 实际
生成对照。失败日志 `bcp778-sg-pd-gpt20-tail/` 保留，不能算作 PD 完整通过。

GPT-OSS-120B 的 mini TP2 固定参考已生成，路径为
`bcp778-mini-gpt120-tp2-store.json`；测试 helper 让 mini 自己持有 rendezvous store，
避免 torchrun 的 agent-store 环境令所有 rank 同时等待一个不存在的服务端。
SG TP2 普通新配置与 GPT-OSS-20B PD 新配置正在验证，最终结果需另行填写。

BCP 吞吐 helper `benchmark/context_system/run_minimal.py` 保留完整任务、原输出长度、
seed 42、C1/N2 与 C2/N4、K12/96Ki，不测 SLO。`a9e7970d9` 将原生 SG client 的
journal 改为与 mini 相同的异步 FIFO 写入；计时前用无关合成输入单独预热，记录在
`warmup.json`，不预热被测任务。mini 没有 HTTP cache flush，因此各引擎均保留这段
无关前缀，并如实记录，不能声称测试从物理空缓存开始。仍需检查测量区间有无新 JIT。

## 最终设置

1. 第一阶段必验 Qwen3、AgenticQwen、GPT-OSS。保留 SGLang 原生模型、MoE、
   tool/reasoning parser 和调度器，不移植 mini 的替代实现。
2. 生产路径只支持 mask prefill 与 paged-occurrence。mini staged 用作计算语义对照，
   不作为 SGLang 生产路径。一次正式请求直接保留 prefill 产生的首 token；不增加请求级 warmup。
3. 消息 ID 使用用户 `messages` 数组的下标。`drop_message[n]` 在消息 n 算完后生效，
   消息 n 仍可读被选中的历史 KV。同时保留结构化 DropRule 的语义。
4. Drop 不改变幸存 token 的位置；Reposition 才压缩位置。同一边界先 Drop 后 Reposition。
   原始 token 下标、当前 RoPE position、active 下标、物理 KV slot 分别维护。
5. 最终 Radix key 包含 token / Drop / Reposition 历史以及最终位置。保留最终位置 KV，
   兼容 Retry 生成目标位置的新 KV；不原地旋转其他请求可共享的来源 KV。
6. 普通调度与 P 端支持 drop-aware eviction；先淘汰无引用叶节点，再淘汰符合条件的内部
   Drop 区间。拓扑引用与实际 KV 读引用分开，不能释放仍有 GPU 读者的页面。
7. PD 使用不同 GPU 上独立 P/D 进程。D 端默认关闭 Radix；不将 D 生成的 KV 回传 P。
   下一轮 prompt 中包含上轮输出时，P 缺少的部分正常重算，并计入实际计算量。
8. GPT-OSS 使用 SGLang 在实验硬件上自动选择的默认 attention backend，不强制切换 FI/FA。
9. 无功能请求走原生快速路径；原生 chunking、overlap、mixed batch、retract 保持有效。
   按用户后续修订，首阶段仅支持 `page_size=1`；撤回本次新增的大页 Reposition 分支与
   多页功能验证矩阵。Context cache/Retry/attention 入口显式拒绝其他页大小；
   SGLang 原生无功能请求的大页能力不变。

## 已确认的实现差异与接入要求

以下 SGLang 引用均以冻结基线为准；mini 引用以 System-test 固定版本为准。

| 差异 | 源码证据 | 迁移要求 |
| --- | --- | --- |
| 默认缓存是 UnifiedRadixCache | `python/sglang/srt/mem_cache/registry.py` 的 `default_radix_cache_factory`；`kv_cache_builder.py` | 接入 unified tree/components 的插入、分裂、锁和回收，不能只补旧 RadixCache |
| 两者都缓存已计算的输出 KV | SGLang `unified_radix_cache.py:959` `cache_finished_req` 使用 prompt+output；mini `core.py:764` `append_host` | 普通 scheduler 保留输出缓存；未做 forward 的最终采样 token 不能冒充 KV |
| SGLang 会缓存未完成 chunk | `unified_radix_cache.py:1105` `cache_unfinished_req`；mini `scheduler/cache.py:422` 对 contextual/delta unfinished 返回 | chunk cache 必须带阶段/版本语义；不能把临时 occurrence 作为最终 cache 提交 |
| Unified insert 自行释放重复输入 KV | `unified_tree_core.py` 的 `_insert_walk_step` 生成 `FreeDeviceKV`；`unified_radix_cache.py` 的 `insert` 执行动作 | 调用方不得再次释放 `prefix_len` 对应输入；本次 Retry 测试的重复释放已据此修正 |
| unified SWA 有独立有效范围与锁 | `unified_cache/components/{full,swa}.py`、`unified_tree_core.py` | full 与 SWA 有效页不同，Drop-skipped 不能因 SWA 窗口裁剪而增加 |
| 原生支持真实 retract | `managers/schedule_batch.py` 的 `reset_for_retract`、`retract_decode` | 释放 KV 后保留可重建的 Context 状态，恢复不能退化为裸 token 普通 prefill |
| mini occurrence 某些混合 batch 限制 | mini `attention/{fi,fa}.py` | 不迁移其 mixed-batch 限制到 SGLang；请求间 metadata 必须独立 |
| 模型默认后端依硬件而定 | `arg_groups/model_overrides/gpt_oss.py` | 记录解析后后端；A800 的当前默认选择为 Triton，保留 sinks、SWA、原生 MoE |
| 原始长度可以超过 RoPE 上限 | mini 历史 128K 越界修复 | 按 raw 长度给 raw 表定界；模型边界按实际计算 position 与合法 active 容量检查 |

## PD 原生流程与迁移边界

P 的 bootstrap 队列完成连接与 D 容量协商，再进入正常 prefill 等待与计算；已可传输的
KV 进入传输队列，GPU stream event 保证写完成后发送。P 的最终 prefill 还产生首个输出
token。D 先分配自己的目标页，接收 KV 和首 token 等元数据，构造接续 decode 的 batch，
通常不再执行一次完整 prompt forward。

关键入口：`disaggregation/prefill.py` 的 `finalize_bootstrap`、`process_batch_result`、
`_send_kv_chunk`、`optimistic_release_and_requeue`；`disaggregation/decode.py` 的
`_match_prefix_and_lock` 与预分配路径。连接接口位于 `base/conn.py`、`common/conn.py`、
`mooncake/conn.py`。配置 `arg_groups/fields/disagg.py` 的 D Radix 默认值为关闭。

迁移传输的是最终 active 的 K/V、相应位置及可验证的 Context 元数据。P 上用于早期 query
的 birth/临时 occurrence 不是 D 所需的终态 KV；P 必须保留这些页直到最后一个 GPU 读者
完成。D 按 active 长度而非 raw 历史长度预分配，并沿最终位置继续 decode。P/D 页号各自
独立，传输协议不能把 P 页号当成 D 页号。

在存在最终位置变化的区间，不能直接复用原生的“按 raw prompt 长度发送全部页”假设。
`83cfdf638` 已恢复完成 chunk 中稳定终态 Full 页的传输，并保留末页至最终元数据
就绪；仍需在隔离 GPU 上测量终态整理与传输等待的代价。
连接失败、取消、重复完成通知、重试时复用原生连接状态机制，同时隔离请求尝试的 KV 所有权。

本轮不增加 D Radix 重试矩阵，也不增加 D→P 输出 KV 传输。P Radix 命中和 Context Retry
仍须验证；这与网络传输 retry 是不同的机制。

## 必须重点防止的效率回退

| 风险 | 最终约束 / 需要观察的证据 |
| --- | --- |
| 每 token 整树 DFS 或完整 integrity 检查 | 生产热路径只做增量维护；完整核验在测试/诊断开启 |
| 每条消息重新渲染整个前缀 | 一次规范模板渲染的 provenance；不按反复 tokenize/LCS 猜 owner |
| 全历史 stage tensor clone/IPC/各 TP 重复编译 | 传输紧凑不可变 IR；当前 chunk 按需编译；避免累计 tensor 的重复序列化 |
| SWA 按 query 在 Python 展开大量 segments | 保留精确位置语义，使用批量/native 编译；测 compile 与 kernel launch 时间 |
| 容量探测反复物化整个 occurrence 计划 | 先最大终点快路径，复用预处理；只物化获选 chunk |
| `tolist/item/isin` 等引入 GPU↔CPU 同步 | 优先 CPU 元数据和批量 H2D；逐处记录 stream 与生命周期 |
| 运行请求时首次 JIT/AOT 编译 | 启动时准备所需 kernel；startup 时间单列，不藏入首请求 |
| 过早压缩或释放造成 overlap 阻断 | GPU event 驱动回收；CPU 不全局同步等待无关 batch |
| 所有历史 occurrence 永久预留 | 只保留当前读集、必要 birth 来源和待传输终态；及时归还临时页 |
| 内存不足直接清空前缀重算 | 先 evict、等待/调度其他请求或原生 retract；最小可推进 chunk 检查 |
| Retry 扫全树、选错分支、原地 RoPE | 长度界限剪枝、最长兼容匹配、只物化胜者；新目标页与源页隔离 |
| 重复逆旋转累积误差 | 保留正确 birth 来源；同位置直接复制；RoPE 参数来自原生模型 |
| Drop-skipped 按终态 inactive 计算 | 必须覆盖本次所有实际 query 与所有 chunk 的真实读集 |
| 用逻辑 prompt tokens 冒充计算吞吐 | cache hit、Drop skip、RoPE copy、graph padding 不计新 forward tokens |
| 页配置或后端不同导致性能结论失真 | 按批准范围，功能开关均使用 page_size=1 和默认后端；结论限于该配置，不外推到原生默认页配置 |

历史决策链重点：`01a05dc4…`（key、CPU 热点、重复模板），`01a077b2…`（一次正式
prefill、usage、staged oracle），`01a07c59…`（occurrence/临时 KV/chunk），`01a086ac…`
（SWA/AOT/raw 越界），`01a08e0b…`（最终位置、Retry、overlap），`01a099fc…`
（drop-aware、holes、高压调度），`01a0ab12…`（最新测速方法）。早期“禁用 Retry”、
“遇任意 R 就停止”、旧 staged 默认、旧 drop-aware 优先级均不能作为最终要求。

缓存 usage 以真实 raw token 集合计数：R 为本次匹配驻留集合，U 为本次所有计算 query
在 full attention 上读取的缓存集合，T 为其中经过位置变换的集合。
`cached=|U\T|`，`repos=|U∩T|`，`drop_skipped=|R\U|`；应满足三者之和等于 |R|。
虚拟标记、空洞、重复 occurrence 不增加 R；仅为 cache-back 而复制、未被 query 使用的
KV 不增加 repos。SWA 自身窗口淘汰不当作 Drop-skipped。

## 验证矩阵与通过依据

| 层级 | 场景 | 通过依据 |
| --- | --- | --- |
| 规则与边界 | legacy/structured Drop、重复文本、思考历史、tool、多 Drop/R 同边界 | 与固定 mini 版本的 token IDs、owner、raw 区间、位置、key 对照 |
| 计划与缓存 | 冷/热/部分命中、相同/不同 R、最长 Retry、跨分支、cache-back | 逐 query 可见集相同；目标 K/V 与来源所有权独立 |
| 生命周期 | chunk/mixed/overlap/abort/retract、叶/内部 evict、hole recovery | 无悬空/重复释放/泄漏；恢复后输出与无洞参考在允许误差内 |
| 数值 | 三个必验模型，固定 token teacher forcing 与实际生成 | 按模型/dtype/backend 校准 logits 误差；定位首次分歧，报告原始数据 |
| GPU 内核 | 同位置 copy、多次 R、SWA/sinks、page_size=1 的非连续物理 slot | 对照实际 mini staged/occurrence 路径；不使用简化 dense 模型冒充 oracle |
| PD | 普通 SGLang↔PD↔mini，首 token/终态/取消/传输重试 | D Radix 关闭，无输出回传；元数据和 active KV 对齐，无多算首 token |
| 压力 | raw>128K 但 position 合法、低 KV、多请求长期运行 | 原生调度可推进，drop-aware 无 -1 进入 attention，池容量可回收 |
| 性能 | 原生无功能↔带功能；mini↔普通 SGLang；PD 小矩阵 | 成对重复、实际计算量、TTFT/TPOT/E2E；约 3% 为目标，统计不足不得宣称达标 |

最小吞吐矩阵：GPT-OSS-120B；普通 mini/SGLang TP2；PD P=TP2 GPU0/1、D=TP2
GPU2/3。所有对照固定 page_size=1；每种系统做 C1/N2、C2/N4，各含 no_drop+普通 eviction、Drop+drop-aware
（PD 仅 P 开启）。同一固定 seed 的任务子集，完整轨迹与源输出长度，不截断 turn 或 output。

测速沿 `01a0ab12-b9ee-7751-bee0-6a78dcac9c1f` 的 `test_serving.py`：同一 task 串行
推进，把上轮实际生成重建的消息加入下轮；持续维持 C 个不同 task；长尾填充成功请求计入
总体并单列，所有首次任务终结时截止。消息重建有损，这是已接受的客户端口径，不能声称
精确还原 sampled token IDs。固定 token 数值测试另做。Rolling Drop 保留 12 条 tool
response，TR13 后丢 TR1；按当前位置到 96 Ki 后在合法边界 Reposition，保留历史事件。

同时报告逻辑吞吐与实际 prefill/decode 吞吐；缺少实际计算计数时报告缺失，不能估造精确值。
标准 TPOT=(E2E−TTFT)/(输出数−1)，首 decode gap 单列。SSE chunk 间隔不能当作 GPU
token 间隔；server token timing 也不是 GPU 硬件计时。本轮不测 SLO。

## 历史检查点（以下描述其当时状态）

### 初始基础编译器检查点

新增 `context_system/ir.py` 与 CPU `kernels/ops/attention/context_plan.py`：保持
mini 的两遍 native 编译、最终位置 key、Drop→R 顺序、尾部 Drop 后无效 R 的语义。
整型元数据在转 int32 前检查范围，避免截断后错误寻址。`occurrence.py` 只展开当前
query window；`recovery.py` 用可见性依赖闭包恢复洞，并保留可复用后缀。

本地 macOS / Python 3.13 CPU 检查：

```bash
PATH=/tmp/sglang-context-cpu-20260917/bin:$PATH \
  /tmp/sglang-context-cpu-20260917/bin/python -m pytest -q \
  test/registered/context_system -k 'not fixed_mini'
```

结果：321 passed，1 deselected。包含 304 个事件程序、逐 query/window 可见集对照、
1000 个随机缓存洞依赖图和 raw 140000 的位置压缩场景。测试独立加载 CPU 模块，
没有初始化 SRT；这不覆盖 scheduler、KV 分配器、attention、PD 或模型输出。
mini 原始 native 编译器的可选直接对照因其 Ninja 输入路径未转义空格在本机失败，
归类为 `ENVIRONMENT_BLOCKER`，待远端无空格路径复核，不能将 deselected 算通过。

Zhangyudong-BUS：隔离 checkout 已通过 HTTPS Git clone 到
`/mnt/public/wangruoxi/repo/sglang-system`，初始 HEAD 为冻结官方基线；原有 detached
SGLang checkout 保持不变。现有 Torch 2.11 cu130 原先无法初始化 GPU；将 NVIDIA
`cuda-compat-13-0 580.178.04` 解包到独立实验目录并仅设置该进程 `LD_LIBRARY_PATH`
后，GPU0 上设备识别和 `torch.arange(100, device='cuda').sum()` 通过（4950）。
没有更新系统驱动。最新上游所需的隔离 Python 环境依赖安装仍在进行，尚无模型实验结果。

兼容包 SHA256：`14a3d14373f882297f368d6282fc7fba85e46682f34166291d61df1913a59c8f`。
配置依据：[NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)。

- 完成 52 会话的决策/失败证据核对与 75 文件、System-test 增量的迁移分类。
- 实现紧凑 Context IR、native template 接入及规则对齐。
- 实现普通 scheduler/cache/attention，再接 PD，按上述门禁验证。
- 所有实验在 Zhangyudong-BUS 的隔离 checkout/environment 上运行；通过 Git 同步代码。
- 每个实现检查点精确提交到 `system`；测试失败按证据修复，不回写 mini，不覆盖远端已有修改。

## 2026-09-17 页大小范围收缩

用户明确要求先只支持 `page_size=1`。保留 paged-occurrence 的 birth/终态版本、
Drop-skipped、最长兼容 Retry 和 COW RoPE 设计；这里的 occurrence 机制不要求一页
包含多个 token。撤回新增 Reposition HND 四维大页寻址；Context Radix 的结构化
key 仍是一项真实 token 对应一项 KV，保留 Drop/R 事件和最终位置的身份区分。

已推送历史不做 reset 或重写，通过后续提交撤回相关实现。旧 HEAD `5e2f78326` 的
多页 GPU 回归已按此范围修订主动中断（31 passed 后 KeyboardInterrupt），不作为
最终验收。后续只运行 page_size=1 的正向功能测试；大页仅保留拒绝测试及原生能力
未被改变的回归测试。模型和 PD 验证仍待完成。

## 历史：Native chat / IPC 初始接线

OpenAI chat 请求现在显式解析 DropRule、legacy Drop 和严格整数 Reposition。
Context 请求在 native Jinja 规范化之后用一次完整渲染取得 token provenance；
`continue_final_message` 保留原生独立编码 assistant prefix、去除开头 BOS 的行为，
不把拼接后的文本重新 tokenize。KeepText 使用完整历史进行同一原生模板处理。
未带功能的请求继续使用原有 render/encode 路径。

编译结果使用现有 tensor-buffer MessagePack IPC，避免转换为逐 token Python 列表；
接收方检查 schema、dtype、原始 token IDs、最终 key positions 和 visibility。
此检查点仍保留 tokenizer admission guard：scheduler 的 Context KV 生命周期尚未
接通时明确拒绝正式生成，防止把 Drop/Reposition 请求静默算成普通 attention。
这不是生产功能通过的声明。native chat、实际模型 tokenizer 与 IPC 的 Linux 测试
在 Zhangyudong-BUS 执行；完整模型、PD 与吞吐仍待后续验证。
