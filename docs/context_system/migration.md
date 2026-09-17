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
   多 page 支持和默认 page size 的代价需要实测，不能仅用 page size=1 的结果宣称效率等价。

## 已确认的实现差异与接入要求

以下 SGLang 引用均以冻结基线为准；mini 引用以 System-test 固定版本为准。

| 差异 | 源码证据 | 迁移要求 |
| --- | --- | --- |
| 默认缓存是 UnifiedRadixCache | `python/sglang/srt/mem_cache/registry.py` 的 `default_radix_cache_factory`；`kv_cache_builder.py` | 接入 unified tree/components 的插入、分裂、锁和回收，不能只补旧 RadixCache |
| 两者都缓存已计算的输出 KV | SGLang `unified_radix_cache.py:959` `cache_finished_req` 使用 prompt+output；mini `core.py:764` `append_host` | 普通 scheduler 保留输出缓存；未做 forward 的最终采样 token 不能冒充 KV |
| SGLang 会缓存未完成 chunk | `unified_radix_cache.py:1105` `cache_unfinished_req`；mini `scheduler/cache.py:422` 对 contextual/delta unfinished 返回 | chunk cache 必须带阶段/版本语义；不能把临时 occurrence 作为最终 cache 提交 |
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
先保证最终版本正确，再恢复可证明稳定区间的传输重叠；需测量终态整理与传输等待的代价。
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
| 强制 page size=1 或更换默认后端掩盖代价 | 同后端/页配置比较特性开关，同时报告原生默认配置结果 |

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
| GPU 内核 | 同位置 copy、多次 R、SWA/sinks、page 跨界 | 对照实际 mini staged/occurrence 路径；不使用简化 dense 模型冒充 oracle |
| PD | 普通 SGLang↔PD↔mini，首 token/终态/取消/传输重试 | D Radix 关闭，无输出回传；元数据和 active KV 对齐，无多算首 token |
| 压力 | raw>128K 但 position 合法、低 KV、多请求长期运行 | 原生调度可推进，drop-aware 无 -1 进入 attention，池容量可回收 |
| 性能 | 原生无功能↔带功能；mini↔普通 SGLang；PD 小矩阵 | 成对重复、实际计算量、TTFT/TPOT/E2E；约 3% 为目标，统计不足不得宣称达标 |

最小吞吐矩阵：GPT-OSS-120B；普通 mini/SGLang TP2；PD P=TP2 GPU0/1、D=TP2
GPU2/3。每种系统做 C1/N2、C2/N4，各含 no_drop+普通 eviction、Drop+drop-aware
（PD 仅 P 开启）。同一固定 seed 的任务子集，完整轨迹与源输出长度，不截断 turn 或 output。

测速沿 `01a0ab12-b9ee-7751-bee0-6a78dcac9c1f` 的 `test_serving.py`：同一 task 串行
推进，把上轮实际生成重建的消息加入下轮；持续维持 C 个不同 task；长尾填充成功请求计入
总体并单列，所有首次任务终结时截止。消息重建有损，这是已接受的客户端口径，不能声称
精确还原 sampled token IDs。固定 token 数值测试另做。Rolling Drop 保留 12 条 tool
response，TR13 后丢 TR1；按当前位置到 96 Ki 后在合法边界 Reposition，保留历史事件。

同时报告逻辑吞吐与实际 prefill/decode 吞吐；缺少实际计算计数时报告缺失，不能估造精确值。
标准 TPOT=(E2E−TTFT)/(输出数−1)，首 decode gap 单列。SSE chunk 间隔不能当作 GPU
token 间隔；server token timing 也不是 GPU 硬件计时。本轮不测 SLO。

## 下一检查点

### 基础编译器检查点（仍未接入运行时）

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
