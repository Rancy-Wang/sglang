# LoopVQ 独立移植与 Ouro-2.6B-Thinking 复现计划

日期：2026-11-23，Asia/Shanghai。状态：**用户已授权实施，正在执行**。首期仅0.5 bit；W&B监控；实测准确率及bs=1/2/4/8/16吞吐加速比；所有长任务在目标机tmux运行。以下原计划快照后的实施状态以末尾日志为准。

## 身份、范围和预算

- 任务 ID：2026-11-23-loopvq-reproduction。
- 官方参考及用户 LoopVQ fork main：d5d58c83fe5bc9a4ce470ae96c69b55f085e2977。
- 本地 SGLang：loopspec-test，915e79f4ce7dc08e7e980987e0c8729f523eb74b；包含 v0.5.22 基线 baa9befa381511cd4d56f83980951e5a45fffc6d。
- 用户新约定：LoopVQ 的 origin 使用 git@github.com:Rancy-Wang/LoopVQ.git，test 用于开发/提交/push/pull；main 只作正式版本，晋升需明确批准。保留 mscheong01/LoopVQ 为 upstream。SGLang 仍仅通过 Rancy-Wang/sglang 的 loopspec-test 开发和远端同步，正式 loopspec 分支不变。
- 统一候选轮次 **2/10（评测入口语法修复，其他组件继续验证）**。本任务的训练/评测脚本及推理代码候选修改加验证累计计数；跨阶段不重置，固定候选多项测试不重复计数。只读调查、写计划、纯环境准备不伪计代码轮次。
- 主范围：Ouro-2.6B-Thinking、完整 R4，BF16 对照和纯 LoopVQ **0.5 bit（G16/c4）**。原计划0.25 bit移至后续，不在首期训练或评测。不运行 TaSQ 统计、TaSQ/TaSQ+LoopVQ。CQ可作后续附加对照，未列为必需工作。保留微调前码本用于诊断。
- 用户验收口径：不要求跨机器答案或论文分数完全一致；必须记录真实准确率、差异和运行条件。吞吐是必需交付：bs=1/2/4/8/16分别与同机无LoopVQ的BF16普通AR比较。工程正确性测试不因容许采样差异而放宽。
- 首期运行：TP/PP/DP=1，普通多请求 AR server，验证 eager 和 decode CUDA graph；按参考评测关闭 radix cache、chunked prefill 和 piecewise graph。显式禁止与 LoopSpec、DFLASH、recurrent-CDB 混用；关闭 LoopVQ 时现有功能必须回归通过。
- 本轮仅更新此计划文档，不切分支、不修改 remote、不安装环境、不运行 GPU、不 commit/push。

## 只读核查（不是未来资源预留或实验验收）

- 本地 LoopVQ、A10 /homes/ruoxi07/repo/LoopVQ 均为上述 main，clean，origin 仍为 mscheong01/LoopVQ。git ls-remote 用户 fork 确认 main 同 SHA、尚无 test。
- 本地 SGLang clean。A10 /homes/ruoxi07/repo/sglang 是用户 ddtree，HEAD 398fc2c7261af88b93b772ab1c9d6c4dba612745，不能切换它；/homes/ruoxi07/repo/sglang-loopspec-tp2 为 clean loopspec-test，同本地 915e79f4。
- A10：ruoxi07@a10.mit.edu，经 X5。8×A100-SXM4-80GB，本次各卡0 MiB、0%利用率，无 compute-apps；主机 available 1.9 TiB。驱动570.195.03，nvcc12.8.93，g++存在。
- 存储快照：/homes 可用675G，/dataset可用190G，根分区/tmp可用5.2G。环境、pip/HF/torch extension缓存、TMPDIR和日志须放已核实的用户大盘目录；安装前重新查配额、可写性和空间。
- Conda有 base/dfl/spec，无专门 LoopVQ 三环境。已有旧 SGLang venv 不表示当前任务兼容性已通过。
- 找到模型：/homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking。config 为48层、R4、16 Q/KV heads、head_dim128、BF16，model.safetensors 5336011242 bytes。config SHA256 ad4b51dffae60bebeeac5e1d5055e78d01fc01b1a29cd89cd2e50edb848f645a；tokenizer_config SHA256 0d304de54d3f8e0c0573d678c0f1d80cc26e9d23dcb328405f0ec99c4293d04f；modeling_ouro.py SHA256 263c097a2ed6870ce549a65ef6d6bb9e8e167423f87d2c494e0988b9a4f63f4b。不同于旧X5 manifest的config hash，实施前必须核对差异、原始Hub revision及权重hash，不将目录名当来源证明，不修改原模型文件。
- 11月20日旧SSH工具句柄和socket已失效。本轮重新建立 X5 session98734/socket /tmp/loopvq-20261123-x5.sock（shell2094336），复用其转发建立A10 session87703/socket /tmp/loopvq-20261123-a10.sock（shell1281273）。会话保留，后续先检查存活。

## 修改前六项计划

1. **目的**：建立用户fork中可通过W&B查看的0.5-bit训练链路，把纯LoopVQ独立集成进现有SGLang，使用真实HTTP服务记录Thinking准确率及五档batch吞吐加速比，并补充其他benchmark的Thinking扩展。
2. **步骤**：仓库/环境/模型/数据锁定 → W&B连通及tmux落盘验证 → 三方BF16数值基线 → 官方配方校准/G16拟合及微调 → 在SGLang按布局/内存、eager、图与并发的里程碑集成 → 固定提交真实serve → 全量评分、配对吞吐及论文差异报告。训练与只读适配分析可并行，正式量化验收依赖码本。
3. **范围**：LoopVQ/test的calibration/{build_ouro_bundle.py,finetune_codebook.py}增加必要的W&B、计时与checkpoint支持，新增纯LoopVQ runner、manifest、必要的HTTP payload审计及评测外部server入口；SGLang/loopspec-test新增码本/布局/pool/allocator/kernels，按当前arg_groups、KVCacheConfigurator、scheduler、attention和CUDA graph接口接线；本任务research目录增加tmux runner及准确率/吞吐记录汇总。具体模块见下表。保留现有Ouro/LoopSpec数学和分支，不整仓覆盖。
4. **成功标准**：两仓同步SHA及导入路径正确；W&B曲线可见、日志与断点可核对，tmux脱离连接后任务仍运行；G16训练按锁定配方完成、产物可读且全量shape/finite/hash检查通过、最优held-out KL不高于初始化（严格改善另记）；布局和索引语义正确；kernel与显式重建对照、生命周期、并发、graph及回归通过；最终30个主准确率评测单元无缺项、每条请求有可追溯输入输出和评分；五档batch共同长度吞吐矩阵完成，论文长输出容量/吞吐另表。未达标单元明确记为失败或阻塞，不要求加速比必定大于1，也不要求论文分数逐位一致。
5. **风险**：依赖/驱动/JIT兼容；共享文件系统和缓存空间；GPQA/W&B访问；原版与目标BF16舍入不同；跨轮索引alias、pre-RoPE、flush与forward竞争、graph padding、回收；发布脚本的seed可能未传到HTTP；真实batch小于客户端并发；长序列BF16容量不足；官方FT无完整训练恢复。禁止通过改深度、缩短准确率输出预算、换模型或改评分器掩盖。性能长度和窗口单独冻结并标注。
6. **验证计划**：本地静态及CPU布局测试 → 本地commit/push对应测试分支 → A10 ff-only pull及SHA/import核对 → W&B小run、tmux脱离/重连、FT断点恢复验证 → 同机BF16/量化分层验收（含bs16） → 原生server HTTP小样本 → 固定提交全量评测及配对吞吐。记录完整命令、退出码、模型/数据/码本hash、prompt IDs、停止条件、主机GPU、环境、原始日志和结果。每次候选改动启动前记录n/10；第10轮未通过停止改动并给出证据。

## 阶段0：仓库、环境与来源锁定

### Git

实施时先重新检查状态。LoopVQ本地保留现origin为upstream，origin改用户fork，从已核对main创建test，首次push -u origin test；A10同样设置两个remote并跟踪origin/test。后续本地修改 → git push origin test → A10 git pull --ff-only origin test。若已有test或其他改动，先比较，不覆盖。

SGLang复用clean且同SHA的 /homes/ruoxi07/repo/sglang-loopspec-tp2（实施时检查是否被其他任务使用，存在冲突则另建隔离clone），不动ddtree工作树。同步只走loopspec-test；不合并LoopVQ所带新SGLang历史、不用scp/rsync同步源码。

### 四套隔离环境

| 环境角色 | 配方/路径要求 |
|---|---|
| loopvq-build | 官方Python3.11、Torch2.8.0 cu128、Transformers4.56.2，校准与FT |
| loopvq-reference-serve | 官方Python3.10、Torch2.9.1、Transformers5.3.0及其python/依赖，官方服务数值对照 |
| loopvq-eval | Python3.10、CPU Torch2.5.1、Transformers4.53.2、lm_eval0.4.9、evalplus0.3.1、math-verify0.9.0 |
| sglang-loopvq-test | 当前loopspec-test依赖的独立环境，import sglang必须来自目标仓库；不将参考依赖强行覆盖至此 |

以用户路径安装，导出lock/freeze及CUDA/GCC/nvcc版本；参考serve通过ENV_SERVE覆盖环境名。由于两项目都import sglang，必须逐环境检查sys.executable、sglang.__file__、torch/transformers/triton/flashinfer版本。先检查GPU初始化和小kernel JIT；若需要兼容库，仅在隔离环境处理，不升级系统驱动。

模型核对Hub来源、权重及config/tokenizer/remote-code hash。数据锁定revision、下载文件hash、题目IDs、许可/GPQA访问；凭据缺失作为准备阻塞，不更换校准集。规划用户目录 /homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction/{envs,cache,artifacts,checkpoints,logs,results,wandb}，实际创建前核实空间/配额。码本、权重、大日志不进Git。

验收：仓库/分支/SHA/clean一致；四环境隔离可导入；模型来源对齐；HF数据可读；工具链可编译；资源重新检查。这里只检查本任务占用，不终止他人进程。

### W&B：配置、指标与可恢复记录

当前官方训练链路没有W&B接入，需要在用户test分支增加记录支持，不改变损失函数、数据切分、更新顺序或训练步数。在build/eval所需环境安装并锁定兼容SDK；SGLang服务进程无需承担上传任务，由测量/汇总进程上传结果。

- 所需配置：entity（个人或团队slug）、project；一个现有项目URL即可提供两者。未建项目时建议project名loopvq-ouro-2p6b-thinking。默认公共云wandb.ai；私有部署另给base URL。账号必须有该项目写入权限。
- 认证：环境就绪后由用户在A10的ruoxi07账户交互执行wandb login；若已登录则只验证小run能写入指定项目。不要求用户在聊天中提供API key；key不进入Git、脚本参数、日志或manifest。WANDB_API_KEY如由用户已有私密配置注入，仅使用，不打印。
- 分组：group=2026-11-23-loopvq-reproduction；job_type区分calibration、fit、finetune、accuracy、throughput。run标识包含方法/seed/协议/候选提交；持久化run ID、URL和本地日志路径。不同候选不得混为同一训练轨迹。
- 校准/fit：完成序列/层/head、k-means iteration、已有的目标/误差统计、阶段耗时和ETA；不为记录额外做大规模GPU评估。训练每次optimizer update记录train KL、lr、global_step、用时；初始和每epoch记录held-out KL、best KL及对应epoch。同步GPU显存/利用率、主机、模型/数据/代码/配方hash。
- 评测：每benchmark/seed的accuracy或pass@1、样本数、失败/截断、平均输出长度；每bs的BF16/LoopVQ throughput、加速比、延迟、显存及实际batch分布。完成后提供总表和W&B链接。
- 数据范围：W&B展示训练指标、进度、配置和结果表，校准样本以Record ID、split及token数追溯。原始GPQA全文、完整prompt/生成、权重和大缓存默认保存在A10，本地manifest记录绝对路径与hash；不自动上传大文件或整份训练语料。
- 稳健性：所有关键指标同时追加到本地JSONL/CSV。全训练前先确认在线run可见。网络暂时中断时保留本地记录、明确报告监控延迟；离线运行通过锁定SDK支持的wandb sync <run-directory>补传，不假定切offline会自动续接同一run。训练断点与W&B run恢复分开验证；必要时使用关联的resume segment记录重跑步骤，避免将重复计算当新增有效训练步。

### tmux：运行位置与进程管理

所有安装长任务、校准、fit、FT、serve、准确率和吞吐评测在**A10本机**tmux session loopvq-20261123中运行，使用setup、calib-fit、finetune、serve、accuracy、throughput、monitor等窗口。通过既有X5跳板与A10持久连接控制；断开客户端不会主动结束tmux或服务。

每个任务记录session/window、host、GPU UUID、cwd、Python绝对路径、两仓SHA、完整命令（无凭据）、PID/PGID、端口、日志路径、开始/结束时间和退出码。stdout/stderr落盘并确保tee不掩盖原命令退出码；完成结果原子写入，成功标记依赖退出码和有效manifest。重连先attach并检查原进程，避免重复训练、重复服务或抢占端口；只管理本任务拥有的进程。

配对性能测量时同一GPU上BF16/LoopVQ顺序运行，无训练或其他任务竞争；启动前重新检查GPU、主机内存/swap、进程和端口。tmux保障SSH断线后的进程存活，主机重启、OOM或进程崩溃依靠训练checkpoint和评测按单元恢复，不承诺tmux本身防止这些故障。

## 阶段1：纯LoopVQ码本训练及微调

### 锁定配方

| 参数 | 值 |
|---|---|
| checkpoint | Ouro-2.6B-Thinking，48基础层×4轮，16 heads，D128 |
| 校准语料 | GPQA main按Record ID去除Diamond；官方raw文本构建，不应用chat template |
| 校准规模 | 16×2048 tokens，positions2048，seed0 |
| KV | K pre-RoPE；K和V均跨4轮联合编码 |
| 码本 | 256 entries，8-bit index；仅G16/c4，即8/16=0.5 bit/标量的索引率；FP16窗口、码本等额外内存单列 |
| 初始化 | CE经验Fisher加权k-means++/Lloyd，15 iterations |
| FT | 冻结模型，只更新K/V codebook；12 train+4 held-out；Adam lr1e-3；4 epochs=48 updates |
| 目标 | joint final-output KL(clean||quant)，compound1，sink8，rep-chunk512，grad checkpointing |
| 选择 | 最低held-out KL的码本（包括初始码本作为候选），记录初始/每epoch/最终KL |
| 不启用 | k-qweight、k-cov、pooled norm、TaSQ或额外loop sensitivity变换 |

保留FT sink8与线上reasoning64/256的区别，不自行把训练改成线上逐token仿真。官方loopvq_reasoning/build.sh会调用general/build.sh，强制构建TaSQ W且引用4个GPU槽，因此新增只调度纯LoopVQ的runner；算法配方不改。第一轮候选纳入runner、W&B及checkpoint记录支持，计入同一10轮预算。

运行顺序：

1. build_ouro_bundle.py --calib-only --grad-ckpt，生成一次k/v/gk/gv共享缓存（2.6B几何约48GiB）。
2. 仅G16：--method loopvq --loops-coupled 4 --v-method loopvq --abits 8，使用上述calib-cache、15次k-means。
3. finetune_codebook.py对G16执行锁定FT，记录trace/W&B和checkpoint；单任务使用一张独占GPU，不使用TP。
4. 保留初始和e2e两个版本。训练失败不得仅凭目录或文件存在跳过；核对完成标记、shape、NaN/Inf、metadata及内容hash。

### 最终保存形式

目录（默认官方相对布局，实际大盘路径写manifest）：

```text
artifacts/ouro_2p6b_think_loopvq8b_g16_lvgqe2e/
  codebook.pt
  vq_v_codebook.pt
  k_rotation_qqt_r_h_pbr.pt
  v_rotation_sst_r_h_pbr.pt
  build_info.json
```

`.pt`是torch.save的Python字典，不是新的模型权重checkpoint，也不是某批请求的KV索引。codebooks[(slot,head)][group]是FP16质心tensor，形状[256,16]；同时保留forward/inverse/mean/bounds、loop_coupled={loops:4,channels:4,base_layers:48}等元数据。纯LoopVQ变换为identity。FT内质心为FP32可训练参数，保存转FP16；两个rotation文件保留官方兼容格式；build_info包含配方及e2e_finetune信息。初始码本目录去掉e2e后缀。

在线加载形成encode [48,16,32,256,16]与decode [192,16,32,256,4]视图；服务新请求时动态生成uint8 indices。服务仍须原始Thinking checkpoint和完整bundle。

原FT脚本末尾保存最优结果，不提供完整optimizer级resume。本次补充**epoch边界恢复**：初始化验证后及每个完整epoch验证后，原子保存FP32可训练码本、Adam状态、已完成epoch/global_step、最佳码本与held-out KL、Python/NumPy/Torch/CUDA RNG、配方/数据manifest hash及W&B run ID。恢复时严格核对配置；中断epoch从最近完成边界重跑，明确标记重复计算，最终仍为48次有效更新。保留至少最近有效checkpoint与最佳结果，训练checkpoint放checkpoints/，不作为推理bundle加载。

小规模验证同一候选的连续训练与“完成epoch→保存→退出→恢复”对照，确认优化器、计数、数据顺序、选择的best和最终张量在预先固定容差内一致；未通过不得声称可恢复。校准/fit首期按完整有效缓存或bundle恢复，未完成单位重做，不承诺任意head中断续训。W&B恢复历史不能代替恢复Adam和码本。

### 时间估计及启动后ETA

论文§5.1报告单RTX6000 Ada：Ouro-2.6B k-means calibration14.3min、FT5.9min。未明确该数字的Thinking/位宽或首次Fisher/下载计时边界；不是A10实测保证。

预启动工程排期：环境/数据就绪且GPU独占时，为首次校准及单个0.5-bit fit+FT保守预留1–3小时，置信度低，不包含安装/下载/排队/兼容性修复；不是A10实测或完成承诺。首次Fisher采集仍需完成，不能把原两位宽排期简单折半。W&B/checkpoint I/O耗时单列。

启动后：首2个calib序列计时→外推16；K和V各首6基础层计时→各外推48；首epoch计时→外推48训练步及24次held-out序列评估（初始4+4epoch×4+最后4）。ETA = 未完成calib + 未完成fit + 未完成FT + I/O。首次有效样本后报告区间和预计完成时间，每阶段更新并写W&B。轮次不因计时或查看日志增加。

## 阶段2：移植到当前loopspec-test并验证

### 先建立数值基线

当前Ouro已有HF BF16语义：RMSNorm舍入、native RoPE、SiLU中间舍入、ouro_bf16_rounding专用attention。官方LoopVQ的融合算术不同。先用相同token IDs、R4及停止条件比较HF、现有目标BF16、官方LoopVQ BF16，保存teacher-forced logits和首个greedy分歧。

不覆盖现有Ouro。需要论文算术模式时增加明确标记的独立profile，同profile配对BF16/VQ，默认路径保持不变；增加profile的改动仍计入10轮。在第1轮前通过参考运行锁定模型级数值阈值，并写入记录；不能在候选失败后放宽。

### 文件/模块范围

以下来源位于LoopVQ/python/sglang；目标位于sglang/python/sglang。

| 功能 | 官方来源 | 目标接入 |
|---|---|---|
| 布局/码本 | srt/mem_cache/loopvq_pack.py、vq_codebook.py | 新增专用loader/layout，首期仅identity/coupled K/V/FP16码本/uint8索引 |
| KV pool/allocator | unified_kv_pool.py、unified_kv_allocator.py、common.py | 新增LoopVQ pool/allocator；接mem_cache/kv_cache_configurator.py及model_executor/pool_configurator.py、common.py |
| 模型输入输出 | models/ouro.py | 在现有OuroAttention增加受控pre-RoPE K写入路径，保留旧默认与recurrent/DFLASH接口 |
| Attention | layers/attention/triton_backend.py、triton_ops/decode_attention.py | 当前backend增加HP/VQ分支、metadata与LSE合并 |
| Kernels | vqwide_ng/v2/g、vq_codebook encoder、QuantKernel/gpu_flush_int2.py | 按目标python/sglang/kernels/ops结构组织必要Triton/CUDA kernel，不搬无关TaSQ/scalar-int2依赖 |
| 调度/graph | schedule_batch、scheduler相关事件、cuda_graph_runner | flush页迁移、forward完成事件、静态buffer和padding、EOS/abort/retract/回收 |
| 参数/启服 | serve_method.sh与环境开关 | 目标arg_groups/fields/model.py、arg_groups/kv_cache_hook.py及server校验；明确LoopVQ模式及限制 |
| 验证/评测 | 原参考脚本 | research/loopspec/experiments/本任务：CPU布局、CUDA对照、HTTP smoke、protocol manifest与结果汇总 |

目标没有旧triton_ops目录，不能按原路径整文件覆盖。LoopVQ关闭时不改变其他模型或现有LoopSpec行为。

### 分层验收

1. **bundle/layout**：G16完整覆盖192 slots；channel-major/loop-minor；slot slice、code_layer、索引alias及encode/decode roundtrip精确一致。最近邻与FP32穷举一致，等距tie单独固定规则。
2. **kernel**：显式解码KV+PyTorch reference，对照fused attention的output与LSE；覆盖pre-RoPE位置、无quant/混合区、长序列split。首轮前固定BF16/FP32容差，报告max/mean误差、NaN/Inf。拒绝不支持geometry而不是静默走不同数学路径。
3. **生命周期**：prefix/recent边界、flush前后7/8/9、ring多次wrap；batch1/2/4/8/16、长短混合、连续请求、EOS、abort、池容量边界及retract/显式防护；请求全释放后pool回基线，无跨请求污染、重复释放或悬挂alias。
4. **graph/stream**：eager与decode graph对照；batch变化/padding写trash slot；overlap下flush不可覆盖前一次forward仍读取的KV；未通过则保持门槛未达，不能以关掉图冒充完整验收。
5. **模型**：Thinking完整R4和自带template，保存prompt IDs；同码本对照官方路径及目标算术下显式重建。VQ是有损，不要求与BF16逐token相同；sampling按实际分布和协议检查，不以同seed同序列证明正确。
6. **真实服务**：目标环境python -m sglang.launch_server，HTTP /health、/v1/chat/completions及需要的/generate；证据包括server PID命令、SHA、sys.executable、sglang.__file__、码本hash、模型manifest。不能继续调用官方serve_method.sh导致PYTHONPATH指回LoopVQ/python。
7. **回归**：LoopVQ关闭后的现有Ouro BF16与原LoopSpec smoke通过；本任务不验证联合启用。

建议里程碑为布局/内存、eager服务、graph/并发三段，不承诺固定轮数。任何全量benchmark暴露的源码/测试/运行行为修复继续消耗同一10轮。若超限则停止并保留失败证据。

## 阶段3：固定目标提交的真实serve、准确率与吞吐复现

### 3A 论文直接对应的Thinking主表

2方法（BF16、LoopVQ0.5）×5任务×3 seeds = **30单元**；每方法每seed800道，共**4800条generation**（不含诊断/重试）。

| Benchmark | 每seed数量 | 指标 |
|---|---:|---|
| AIME2024 | 30 | zero-shot generative exact match |
| AIME2025 | 30 | 同上 |
| GPQA-Diamond | 198 | 生成答案字母exact match，Record ID固定选项排列 |
| HumanEval+ | 164 | EvalPlus plus-suite pass@1 |
| MBPP+ | 378 | EvalPlus plus-suite pass@1 |

数据量在正式下载后再次核验。模板enable_thinking=True/add_generation_prompt=True；temperature1.0、top_p0.7、max_new_tokens32768、context40960；quant prefix64/recent256、page8。BF16与VQ相同输入/模板/停止/采样。AIME stops为Question:、</s>、<|im_end|>、<|eot_id|>；GPQA去掉Question:；EvalPlus依EOS与长度上限。保留原始输出、finish_reason、输出token数、提取后的答案/代码和逐题得分，超时/截断/HTTP异常单列，不静默删除样本或计零掩盖基础设施错误。评分沿用官方AIME/GPQA提取器及EvalPlus代码提取/sanitize流程，锁依赖版本。

论文Table3参考均值（%），完整mean±std从原文对照：

| 方法 | AIME24 | AIME25 | GPQA-D | HumanEval+ | MBPP+ |
|---|---:|---:|---:|---:|---:|
| BF16 | 57.78 | 51.11 | 54.04 | 86.18 | 75.49 |
| LoopVQ0.5 | 47.78 | 35.56 | 47.81 | 84.76 | 74.34 |

先小样本确认全链路，再冻结提交和协议，跑全量BF16及0.5 bit。输出每seed、mean±sample-std（ddof1）、注明方法的95%区间及相对论文/本机BF16的百分点差；AIME小样本波动大，不要求生成文本相同或分数逐位一致。记录机器、数值实现、有效seed、输出长度和协议偏差；论文均值±std不足以独立证明统计等价，不能把区间重叠当成证明。实现通过、实际准确率、与论文差异分别下结论，明显差异仍定位，不以调参/换种子挑最好结果冒充复现。

### Seed协议审计

发布run_reasoning_api.py仅将args.seed传simple_evaluate的本地RNG，没有在API model_args传seed。lm_eval0.4.9 TemplateAPI默认HTTP seed1234；EvalPlus则直接发送a.seed。这是源码证据，尚未在安装后的HTTP请求中实测。

环境完成后先抓取真实payload。主结果按论文声明发送独立0/1/2 seeds，若与发布脚本行为不同则明确记录修正；若差异影响与论文对照，保留/补跑发布脚本兼容模式，分开命名结果，不能混合平均。模板、stop、预算及HTTP seed在运行前冻结。客户端不能以文件名s0/s1/s2代替实际seed证据。

### 3B 其余benchmark的Thinking扩展

固定Thinking时，下列不是论文原表的同模型复现，不拿base数值当Thinking目标。按用户“各个Benchmark”纳入后续扩展计划，分别报告协议变化：

- General：GSM8K（3-shot）、MATH500（zero-shot）、HumanEval/MBPP base-suite；沿各任务数据和评分，明确Thinking模板/采样策略与论文base greedy协议的不同，保留base协议作为单独可选对照，不混表。
- NIAH：8项RULER，4K/8K/16K/32K/64K，每长度每data-placement seed共200条，3 seeds；量化窗口64/256。校验加入模板后的最终token长度，不将64K prompt加32K输出未经说明当模型64K内结果。遵循原任务输出预算、记录精确截断与位置。
- LongBench：16英文任务，3750条，任务自己的F1/ROUGE/accuracy等指标和预算；记录Thinking模板/停止处理，不能统一改32K输出或称所有指标为accuracy。64K截断/CPU暂存峰值单独验收。
- 首期交付顺序为3A主表与3C五档吞吐优先，随后做本节Thinking扩展；1.4B/Huginn/组件消融不冒充本模型结果。

全量时间启动后按每任务首批实测平均输出tokens及有效吞吐外推：sum(N×seeds×methods×avg_output_tokens)/aggregate_output_tokens_per_second，加prefill/评分/队列时间。不能由码本训练耗时推断4800条长推理请求也很快完成。用独立输出目录区分smoke/full、target/reference、paper-seeds/compat-seeds，缺失单元不得显示通过。

### 3C 必需吞吐矩阵：bs=1/2/4/8/16

主对照是**当前loopspec-test同一提交中的BF16普通AR**，关闭LoopVQ、LoopSpec及其他speculative路径。两方法使用同一A10 GPU UUID、模型、R4、TP=1、环境、算术profile、attention/graph配置和输入token IDs，顺序运行。关闭radix以排除跨请求前缀命中；同一pair不并行跑训练。不以官方参考服务或其他机器的BF16作为分母。

为区分引擎速度与长序列容量，冻结两个协议，分别命名、分别汇报：

| 协议 | 输入/输出token数 | LoopVQ FP窗口 | 用途 |
|---|---|---|---|
| common-shape | 默认2048 / 512，所有bs使用同一长度 | prefix64/recent256，与Thinking准确率设置一致 | 完整五档BF16/0.5-bit配对加速比，属于本机工程测量，不称论文长输出复现 |
| paper-long | 2048 / 32768 | prefix8/recent64，按官方效率脚本 | 论文效率长度/窗口口径的吞吐和容量复现，明确与准确率配置不同 |

common-shape在正式测量前对两方法、五档batch完成容量预检后冻结；若默认长度无法共同容纳，在预检阶段调整共同长度并记录原因，所有bs/两方法一致重测。不得在看过计时结果后挑有利长度。paper-long不缩短；2K→4K仅可用作诊断，官方脚本注释说明最终figure为2K→32K。无论哪个协议，BF16 OOM都记OOM/capacity-limited，对应speedup=N/A，不记0吞吐或无限加速。共同长度矩阵若仍缺条件则该交付未完成，须说明阻塞。

每协议2方法×5档=10条件。先至少2个预热batch、再每条件3次完整计时，保留原始测量并报告中位数和范围；同GPU按配对轮交替方法测量，加载/JIT/预热时间不计入吞吐。长输出总时间另估，不截断为短跑后外推为实测值。GPU分配策略及memory fraction/cap记录，不通过CPU offload或改变R4帮BF16凑容量。

- **负载**：真实目标HTTP server，固定完整输入token IDs；同一组请求齐发，强制固定输出数（性能专用ignore_eos/不设内容stop，确认server实际支持并核对返回token数）。准确率评测仍按原EOS/stop自然结束，不能混用这一性能设置。benchmark自然生成速度可另报，量化导致输出变短不能单独作为引擎加速证据。
- **batch语义**：client concurrency和max-running-requests只是控制量。记录server逐步decode batch size、活跃请求数、prefill分批、排队、retraction及重计算。稳定decode阶段确实达到目标bs才计入该batch矩阵；被容量限制为小批串行的结果另记“并发请求服务吞吐”，不冒充bs16算子吞吐。
- **主指标**：E2E output throughput = 成功请求实际输出token总数 / 从该batch首次发出请求至最后响应完成的wall time。speedup(bs) = LoopVQ吞吐中位数 / BF16吞吐中位数，同协议同bs分别计算。
- **辅指标**：server计时的decode throughput、prefill时间、客户端TTFT/TPOT、E2E、输入/输出token数、峰值显存、OOM/重计算。decode计时用明确server事件边界和对应区间token数，不用E2E减去单条TTFT近似整批decode。另报官方overall=(input+output tokens)/E2E时独立命名，不能与output-only指标相除。
- **公平性**：记录实际batch和可用GPU资源；出现其他任务竞争则标记测量无效并重测。若换机器，整组BF16/VQ配对重测，绝不跨机器拼加速比。数值≤1也如实报告。
- **结果**：CSV/JSON保留每次原始测量，W&B中绘制bs—吞吐及bs—加速比，表列协议、bs、实际batch、BF16 tokens/s、LoopVQ tokens/s、speedup、TTFT/TPOT、显存及状态。accuracy与throughput链接到同一模型/提交/码本manifest。

## 交付与实施前状态

- 阶段0：Git/环境/模型/数据manifest、W&B可见的小run、tmux会话及可用性记录。
- 阶段1：0.5-bit初始+微调bundle、训练恢复checkpoint、KL trace/W&B链接、完整参数、hash、分阶段耗时和ETA实际偏差。
- 阶段2：固定loopspec-test SHA，测试矩阵结果、完整命令/日志、实际import来源及已知限制。
- 阶段3：30单元主表、逐题输出/评分、与论文及本机BF16对比；bs=1/2/4/8/16共同长度的完整配对吞吐表/加速比及论文长输出表；Thinking扩展各任务结果；任何未完成/不相容/OOM项明确标注。
- 正式分支晋升：无授权，保持LoopVQ/main和SGLang/loopspec不动。
- 本轮验证：仅计划更新及源码/文档核对；上述机器资源为先前快照，未为本次计划更新重新占用GPU。尚未安装W&B、启动训练、完成CUDA数值验收或准确率/吞吐复现。累计候选修改+验证仍0/10。

## 需要用户提供的信息

1. **W&B项目URL**，或entity（个人/团队slug）和project。没有现成项目可仅提供entity，实施时使用默认名loopvq-ouro-2p6b-thinking。
2. **认证状态**：A10 ruoxi07账户是否已完成wandb login；未登录时待专用环境准备好，由用户在该机终端交互登录，再告知已完成。不要在聊天中发送API key。
3. **仅私有部署需要**：W&B服务base URL；否则默认wandb.ai。
4. **仅在数据访问核查失败时需要**：有GPQA访问权的Hugging Face账号在A10完成认证/授权。先复用并验证已有访问，不要求预先重复提供token。

默认只上传训练指标、配置和结果汇总；如用户“训练数据”还指原始样本文本的在线浏览，另确认希望展示的样本范围。当前任务只更新计划，收到项目配置不等于本轮已开始训练。

## 证据入口

- 论文：https://arxiv.org/html/2611.05304v1 （§4、§5.1、Table3、Appendix D.5）。
- 用户fork：https://github.com/Rancy-Wang/LoopVQ 。
- 官方配方：LoopVQ/experiments/loopvq_general/build.sh；loopvq_reasoning/{build.sh,scripts/run_cell.sh,make_table.py}。
- 码本格式：LoopVQ/calibration/{build_ouro_bundle.py,finetune_codebook.py}；python/sglang/srt/mem_cache/loopvq_pack.py。
- API seed核对：https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/v0.4.9/lm_eval/models/api_models.py 和 openai_completions.py；实际payload待实施时抓取。
- 效率协议：LoopVQ/experiments/loopvq_efficiency/scripts/run.sh（THRU_IN/THRU_OUT、BSS、PREFIX_TOKENS/RECENT_TOKENS及最终figure注释）。
- W&B官方认证/同步CLI实现：https://github.com/wandb/wandb/blob/main/wandb/cli/cli.py 。实施时锁SDK版本并按该版本help核验，不依赖变化过的--sync-all选项。
- 项目约束：根AGENTS.md及loopspec-code-change、x5-loopspec-experiment、x5-persistent-session技能。

## 实施日志

### 2026-11-23：候选1/10开始

- 用户明确授权执行完整计划，W&B项目为529513043-tsinghua-university/loopvq-ouro-2p6b-thinking，允许在终端执行login。邮箱不是API凭据，不记录或传输密钥。
- 目的/步骤/模块/成功标准/风险/验证沿用上述六项计划。第一候选覆盖纯G16 runner、W&B及epoch恢复；并行开发SGLang独立LoopVQ的布局/内存/attention接线，先审核候选再做GPU验证。源码变更仅本地，Git同步测试分支。
- 开始状态：LoopVQ main=d5d58c83fe5bc9a4ce470ae96c69b55f085e2977 clean，用户fork尚无test；SGLang loopspec-test=915e79f4ce7dc08e7e980987e0c8729f523eb74b，仅此任务记录未跟踪。
- 历史SSH工具session98734/87703不存在，原X5 ControlPath不存在；确认失效后重建本对话跳板连接，禁止绕过X5直连。
- 验证/提交/退出码：待执行，不能把计划或编译检查当训练/服务通过。

### 候选1实施进展

- 已建立并复用X5 session44128（shell2129780，ControlPath /tmp/loopvq-20261123-impl-x5.sock）及经此转发的A10 session63456（shell1350260，ControlPath /tmp/loopvq-20261123-impl-a10.sock）。A10检查8张A100均空闲、available RAM1.9TiB；数据和缓存放用户目录，未写系统CUDA/驱动。
- 本地/A10 LoopVQ均已将官方origin改名upstream，新增用户fork origin、创建/跟踪test并push/pull。训练候选提交bf1cf18b8e58ab1cc772d7467867994add91f787已两端一致；main未改。SGLang源码仍在本地候选开发中。
- A10已创建tmux session loopvq-20261123，安装窗口setup-build/setup-eval/setup-reference/setup-target，另有wandb-login/checks/data-check/monitor。日志根 /homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction/logs。
- 环境准备遇到Anaconda默认channel未接受ToS，未代用户接受，改本任务命令使用conda-forge；并行conda下载共享cache发生partial文件竞争，reference改独立cache重试。这些是基础设施修复，未改变候选代码。build环境安装已exit0，Torch2.8.0 cu128、Transformers4.56.2、W&B0.31.2。
- 用户授权使用其提供凭据验证。已通过隐藏终端输入完成A10 W&B认证并写入指定项目，authentication_verified=true、project_write_verified=true；仅本机凭据文件保存，权限600，Git和任务记录不含key。在线验证run：https://forge.coreweave.com/wandb/529513043-tsinghua-university/loopvq-ouro-2p6b-thinking/runs/xes9bpio 。使用显式WANDB_BASE_URL=https://api.wandb.ai；服务返回当前Web入口为forge.coreweave.com/wandb。
- 训练候选本地py_compile和diff检查exit0；本地2个测试通过、Torch恢复测试因无依赖skip。A10已在checks窗口启动完整CPU恢复测试，结果待核实。模型来源/GPQA数据访问在data-check窗口核查，未启动训练。

- A10训练CPU恢复3/3通过，train-cpu-check.log，退出0。模型config/tokenizer_config/modeling_ouro/configuration_ouro/5.336GB权重全部与Hub f1edd81e7ac41355db670500ceaf204e0f73af68逐项hash匹配。GPQA revision83022cefff930aea54f654c0b282e74b9eeda5c6访问失败：gated dataset未认证；已准备hf-login tmux窗口并请用户接受条件后交互登录，此为数据访问阻塞，不改用别的校准集。
- 候选1评测HTTP集成测试4项通过、diff检查通过；run_evaluation.py首次AST检查失败（149行coverage列表推导缺右方括号），尚未提交此评测文件。训练提交不受影响；推理GPU验证尚未执行。

### 候选2/10：修正评测入口语法

- 目的：使统一评测driver可解析执行；步骤：补齐coverage列表推导括号并重跑同一组静态/HTTP检查。
- 范围：LoopVQ/experiments/loopvq_reproduction/run_evaluation.py；不改变采样、评分、训练或推理数学。
- 成功标准：AST/py_compile、HTTP恢复/失败记录/seed检查全部通过；风险：仅静态通过仍不表示真实server评测通过。
- 验证：本地脚本解析和已有4项HTTP集成测试；之后与训练及推理候选一起由root审核同步、A10运行。其他组件沿用候选1的未变更实现，任何随后失败再修改继续累计。
