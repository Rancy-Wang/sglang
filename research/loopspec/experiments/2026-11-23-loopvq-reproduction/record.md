# LoopVQ 独立移植与 Ouro-2.6B-Thinking 复现计划

日期：2026-11-23，Asia/Shanghai。状态：**用户已授权实施，正在执行**。首期仅0.5 bit；W&B监控；实测准确率及bs=1/2/4/8/16吞吐加速比；所有长任务在目标机tmux运行。以下原计划快照后的实施状态以末尾日志为准。

## 身份、范围和预算

- 任务 ID：2026-11-23-loopvq-reproduction。
- 官方参考及用户 LoopVQ fork main：d5d58c83fe5bc9a4ce470ae96c69b55f085e2977。
- 本地 SGLang：loopspec-test，915e79f4ce7dc08e7e980987e0c8729f523eb74b；包含 v0.5.22 基线 baa9befa381511cd4d56f83980951e5a45fffc6d。
- 用户新约定：LoopVQ 的 origin 使用 git@github.com:Rancy-Wang/LoopVQ.git，test 用于开发/提交/push/pull；main 只作正式版本，晋升需明确批准。保留 mscheong01/LoopVQ 为 upstream。SGLang 仍仅通过 Rancy-Wang/sglang 的 loopspec-test 开发和远端同步，正式 loopspec 分支不变。
- 统一候选轮次 **9/10（长实验编排与数据冻结，推理数学保持候选6）**。本任务的训练/评测脚本及推理代码候选修改加验证累计计数；跨阶段不重置，固定候选多项测试不重复计数。只读调查、写计划、纯环境准备不伪计代码轮次。
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

1. build_ouro_bundle.py --calib-only --grad-ckpt，生成一次k/v/gk/gv共享缓存（A10实际完整2.6B几何为96GiB，原计划48GiB估算已纠正）。
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

- 候选2评测入口复验：5文件py_compile、原HTTP4项、diff检查全部exit0。baseline_probe.py本地语法/CLI及9次成功HTTP+1次503失败保存测试exit0。吞吐measure/aggregate脚本及5项协议测试exit0。均不代表真实模型GPU验收。
- SGLang独立LoopVQ提交48b5611b2b18cfa221014181d116e0f96e3483a3已push/pull；CPU核心测试3项通过、2项CUDA测试skip（core-cpu.log，exit0）。推理实现采用现有Ouro BF16舍入的Triton混合查表attention，并未整段搬入官方速度优化CUDA kernel；性能待实测。
- W&B和HF凭据已按用户授权通过隐藏输入验证；GPQA main448、Diamond198可读，差集250；revision固定83022cefff930aea54f654c0b282e74b9eeda5c6，凭据不进记录。build/eval/reference环境安装exit0，目标venv隔离复制exit0；目标CUDA/import仍待检查。
- 首次训练启动被clean检查正确阻止：reference editable安装自动改写已跟踪python/sglang/_version.py（仅版本字符串/commit_id）。确认安装生成差异后备份到results/reference-install-generated-version.py，用Git恢复该文件到HEAD，未覆盖用户改动或修改运行数学；重试仍为训练bf1cf18，不新增代码轮次。
- 2026-11-23 13:01 UTC左右在A10 GPU0、tmux calib-fit启动校准，W&B run l7god8t9：https://forge.coreweave.com/wandb/529513043-tsinghua-university/loopvq-ouro-2p6b-thinking/runs/l7god8t9 。8/16序列完成46.3s，按此速度剩余计算约46s；完整缓存96GiB，落盘/校验时间另计。尚未完成fit或FT。
- 新增可复用A10目标环境入口和三方BF16 probe提交18c649e6f（Git pull进行中）；沿用用户目录CUDA13 compat/toolchain，系统驱动未修改。所有compiler cache隔离至本任务目录。

- A10目标环境import来源和CUDA张量检查通过，提交18c649e6f已两端一致；GPU wiring 6/6、core 5/5通过，其中mixed attention bs1/2/4/8/16最坏绝对误差0、CUDA graph replay与eager一致，logs/target-core-cuda.log exit0。这是合成算子测试，尚不代替真实模型验证。
- 校准16条序列计算95s，含96GiB缓存写入/全量校验约10分钟，driver-calibration-v2退出0。后续fit/FT在GPU3运行，fit run ffmnphq6，前18/48层64s，预计K/V共约6分钟；FT ETA尚待首步。GPU0曾出现其他任务，已避开，未终止其进程。
- HF full-R4无缓存参考3条32token probe已退出0，results/baseline-hf；目标BF16实际服务30540/GPU1已健康，8192token池仅用于smoke。官方reference第一次启动受未锁定kernels0.19.0与Transformers5.3.0不兼容阻止；按已安装Transformers声明固定kernels>=0.10.2,<0.11，安装0.10.5退出0。这是隔离环境修复，未修改源码。

### 候选3/10：吞吐记录适配当前服务配置

- 目的：修正吞吐脚本使用旧版CUDA graph字段导致真实/server_info被拒绝的问题；真实返回已经保存results/target-bf16-server-info.json。
- 步骤：以cuda_graph_config.prefill/decode.backend及decode.bs为权威，结合启动日志的实际capture buckets；更新fixture和校验，并执行真实服务短输出smoke。
- 范围：benchmark_throughput.py、test_benchmark_throughput.py及本记录；推理数学、训练配方、采样和评分保持既有候选。
- 成功标准：现行schema的full/eager/prefill拒绝/buckets缺省与差异测试通过；真实请求的token计数、稳定batch、日志覆盖、graph状态能审计，不能确定的padding记unknown。
- 风险：配置只表示意图，实际runner可能过滤buckets；日志不足时不能报告已达到目标batch或零padding。
- 验证：本地针对性测试+语法+diff检查，经loopspec-test同步后A10真实BF16服务smoke；记录命令、退出码和结果。此前5项吞吐测试仅覆盖旧fixture，不能作为真实服务兼容证据。

- 候选3本地冻结验证：吞吐schema相关9/9测试、py_compile、diff检查exit0；待真实服务smoke。target BF16三条32token（共96）greedy ID/停止位置与HF全部一致，96个teacher-forced位置argmax也一致；top20 logprob最大绝对差0.34365/0.43015/0.50000，保留原始概率，未声称概率逐位相同。reference第二次启动因未激活环境PATH找不到已安装ninja失败；加入该环境bin路径重启v3，已加载权重并完成graph capture，尚在warmup。

- 候选3已提交22aa032cf029993b84b705f0c9ddf281581d102d并两端同步。真实BF16 HTTP bs2、64输入/32输出、2预热+3测量全部status=ok、exit0；实际decode batch/graph证据通过，logs/throughput-smoke-bf16.log，仅作为流程smoke不计正式性能矩阵。reference BF16也完成3×32 greedy和96 teacher-forced探针exit0；待统一比较其logprob。
- 初始fit完成351.8s；FT run1r6f3gzx，初始heldout KL0.0678，首epoch54.2s/heldout0.05621675，估余三轮约179s。复制原子epoch checkpoint并在空闲GPU4独立恢复到单独诊断目录，用于真实GPU恢复等价性检查，正式训练GPU3不中断。

### 候选4/10：真实量化服务生命周期验收

- 目的：补足合成测试未覆盖的真实pool写入、长prefill、padding和异步释放路径；不改变量化数学。
- 步骤：新增外部HTTP诊断driver，保存checkpoint原生Thinking模板/input IDs、raw请求响应、完整output IDs/logprob和错误；先eager再full graph执行相同案例，并比较干扰前后重复请求。
- 范围：research/loopspec/experiments/2026-11-23-loopvq-reproduction/serving_probe.py及必要的协议测试/本记录；服务器代码修复若另有失败再累计下一轮。
- 成功标准：短HP路径对BF16、VQ eager/full相同上下文的token/停止对照；真实量化decode有限且计数正确；bs1/2/4/8/16及padding3/5/15，边界319/320/321/333等，4096长prefill、264-ring多圈、abort/free/reuse/最终flush均健康且空闲资源恢复。非预期分歧保留首位置，不用flush掩盖释放泄漏。
- 风险：LoopVQ量化本身允许不同于BF16；当前target使用BF16舍入的两遍attention，官方参考使用FP32 decode/split数学，不要求二者逐位相同。异常中止可能污染缓存，必须单独检验后续请求。
- 验证：本地解析/HTTP协议检查，经Git同步后A10真实最终bundle服务，先小后大，保存raw日志、metrics和退出码；正式accuracy/throughput在该门槛后运行。

- 候选4实际VQ启动失败（target-vq-eager-server.log）：handle_loopvq_compatibility读取resolved_view.attn_dcp_size，但此名是Parallel字段组Derived，非ServerArgs实际字段，引发AttributeError。失败发生于参数解析/加载权重之前；原合成fixture人为添加了该属性，故6项测试未发现。新增serving_probe尚在编写，未标通过；保留失败日志，随后修复累计下一轮。

### 候选5/10：修复真实参数解析并验收服务

- 目的：消除LoopVQ兼容hook与当前ServerArgs字段结构不匹配，补足真实参数对象验收。
- 步骤：核对parallel输入/Derived及其余hook属性，以正确dcp_size或已定义runtime helper检查DCP；更新fixture并增加真实ServerArgs解析检查，完成候选4待验的HTTP服务probe。
- 范围：arg_groups/kv_cache_hook.py、test_loopvq_wiring.py、候选4新增serving_probe.py/test_serving_probe.py及本记录；不改变码本或attention算术。
- 成功标准：真实完整Ouro+最终bundle参数解析成功，DCP>1仍被拒绝；原6项及新检查通过；随后真实eager/full模型服务按候选4矩阵验收，不把解析通过代替服务正确。
- 风险：其他分层配置Derived字段也可能与简化fixture不同；同时审计所有hook访问，不放宽不支持配置的拒绝条件。
- 验证：本地静态/协议测试，A10实际ServerArgs与启动，记录SHA/命令/退出码/日志；失败再次修改计入下一轮。

- 正式训练完成：calibration/fit/FT均exit0；fit351.8s，FT303.0s，48updates/4epochs，heldout KL曲线0.06782355→0.05621675→0.05229405→0.04873301→0.04768510。最终bundle codebook.pt/vq_v_codebook.pt各607379165B，两个identity rotation各12640793B，build_info1628B。训练源固定bf1cf18；完成并结束恢复诊断后，A10 LoopVQ/test已ff-only到790e69e用于评测。
- 实际最终bundle与官方loader的encode/decode表及code_layer逐元素一致；校准seq0/12/15、base0/23/47、17边界位置、全部heads/groups共156672个K/V编码，target/official/FP64最近邻全部同索引，raw HP写入、prefill/flush和dummy保护通过exit0。少量极小BF16值转FP16改变，但这批数据未改变索引。evidence/a10/real-bundle-encoding.jsonl保留结果；不替代真实服务生命周期。
- 五任务数据online/offline复核exit0：AIME24 30/AIME25 30/GPQA Diamond198/HumanEval+164/MBPP+378；GPQA固定83022ce，AIME分别8d88b287/563bb840；EvalPlus GT已预热，manifest已存evidence。真实BF16 HTTP审计aime24+HumanEval、seed0/1（每cell1题64token）全部exit0，payload/template/IDs/seed/评分链路通过；这4个截断审计分数不能作为benchmark准确率。
- 候选5本地：hook32个字段schema审计、py_compile/diff检查exit0，serving_probe新增9项真实本机mock HTTP/SSE协议测试exit0；A10真实wiring/解析/服务待执行，不能报通过。

- 候选5 A10测试：实际ServerArgs wiring单元与eager/full真实模型参数resolve_once全部exit0（round5-real-resolution.log）；最终bundle eager服务30542/GPU0已健康，full graph30544/GPU3启动中，真实全suite执行中。当前两遍reference实现仍不能作为论文优化性能交付。

### 候选6/10：移植官方纯LoopVQ优化decode

- 目的：完成用户要求的官方SGLang适配/实际加速复现，避免仅交付用于正确性核对的慢速两遍实现。
- 步骤：提取d5d58c83实际G16默认vqwide_g MODE1 CUDA(R1/R2/PC1)，官方identity HP pre-RoPE stage1和统一LSE reducer；加入稳定metadata/scratch及预编译，再接线显式profile。保留hf_reference供差异定位，新增official_cuda用于正式性能测试。
- 范围：kernels/ops/loopvq/{official_quant_cuda,official_hp,official_reduce,official_metadata,official_attention}.py；loopvq_backend.py及必要args/hook、测试和记录。pool/allocator/训练码本不重写；候选5真实服务若发现缺陷在本候选首次验证前一并修复并记录。
- 成功标准：quant/HP/reducer对官方原模块同输入同split数值通过；真实bundle及2K/8K/约35K、split边界/空quant/nonaligned、bs1/2/4/8/16+padding3/5/15通过；profile内eager/full及回收复用通过；短kernel和真实HTTP性能已测后才启动全矩阵。
- 风险：官方FP32 RoPE/softmax及HP FP16 MMA与hf_reference BF16舍入不同，绝不混为逐位相同算法/结果。纯G16 CAP2048、R2每split1024、48quant splits覆盖49152，必须硬拒绝超容量避免shared越界；A100 shared约97.14KiB/block。CUDA13/Torch2.14 JIT需实编，缓存隔离，capture前完成编译和shared属性设置。
- 验证：同一A10环境固定架构8.0与最终bundle，先对官方kernel的GPU数值/graph再真实服务生命周期；性能注明profile、硬件、代码、实际batch，BF16无LoopVQ仍保留已验证参考。对官方数学profile的速度差异不能仅归因于KV压缩，不跨profile混合统计。

- 候选6首次验证前范围澄清（性能公平性审计）：统一公开ouro_decode_profile={hf_reference,optimized}。optimized下VQ用官方CUDA分阶段decode；BF16只将forward_decode切到当前已有generic Triton split-K，不切prefill、不全局关闭ouro_bf16_rounding，保持两方法共有的HF norm/RoPE/SiLU/fullR4。默认hf_reference保持现行为。正式速度主表使用optimized BF16 vs optimized LoopVQ，原两遍结果仅诊断；新BF16必须重跑HF/官方32token及teacher-forced/长上下文/graph验收。该profile不是整个模型的官方逐位复刻，当前generic QK还有FP32 reduction差异。记录BF16 splits8/VQ quant48+HP12的完整实现配置差异，不能把速度差全归因KV压缩。
- 训练恢复诊断已exit0：从epoch1/step12恢复到epoch4/step48，recipe一致、Adam step48；另一A100上最终bestKL0.04756755 vs连续训练0.04768510，差0.00011755；FP32码本K/V平均绝对差0.00046731/0.00051596、最大0.02083/0.02164。GPU散射/原子累加不保证位确定性，报告为恢复成功且存在已量化数值差异，不宣称逐位等价；正式产物仍选连续训练目录。

- 候选5真实最终bundle服务全部通过：eager GPU0/30542和fullgraph GPU3/30544均完成17case（包含abort）并exit0，16个有输出case共72请求，eager/graph output IDs与停止位置全部一致；short HP与HF也一致。覆盖4096输入、63/64/65/319/320/321/327/328/333/383边界、bs1/2/4/8/16、padding3/5/15、600token多圈ring、ragged结束/owned abort、复用及先验idle后flush；均无空闲KV泄漏。原始结果A10 results/serving-vq-{eager,graph}与serving-eager-graph-comparison.json。
- 候选6本地冻结：5个官方算子/metadata/adapter和GPU测试AST通过；相关11文件py_compile、serving9/9、throughput11/11及diff检查exit0。CUDA源MODE1与d5d58c8只差宏头、split/CAP断言和空白；尚未GPU编译，不能报数值/性能通过。Lean上层capture选择虽可为true，最底层实际仅HIP执行，A10旧路径并未使用Lean；optimized仍显式固定split契约并省掉无用arena。

- 候选6 A10 source85cd80c5：真实wiring9/9、官方GPU3/3 exit0；22个算子case各eager/graph共44次，partial最大误差0、merged BF16逐元素相同；8种batch的metadata CPU oracle及变长graph通过。证据round6-official-kernels.json。optimized BF16三条32token greedy/停止和96个teacher-forced argmax与HF全一致；top20 logprob最大差仍0.343647/0.430154/0.499999，不声称概率逐位一致。
- optimized BF16与VQ各自完成eager/full全suite并exit0；每方法72请求output IDs/停止位置全一致，abort/idle/free/reuse/flush全通过，证据round6-serving-comparison.json。默认参考profile的上一组72请求另存，不混作optimized证据。所有真实模型仍完整R4；LoopSpec未联合启用。
- 训练配方审计补充：官方及本次builder/FT沿用固定HF checkpoint的outer model(...).logits，threshold=1.0阈值选层，骨干无条件执行4轮；不是mixture。BF16 gate/pdf/cumsum代码本身不能严格保证选中第4轮，目前无实际提前选层频率证据。线上服务和HF probe则显式最终第4轮LM head。保持官方训练配方，不把两种readout称为代码层面严格等价。
- 未启用且不作支持声明：enable_deterministic_inference、triton_attention_split_tile_size、SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS；正式准确率manifest逐项核实默认false/None/absent。35K decode算子通过不等于35K fresh prefill容量通过；跨轮raw K/V暂存约1.125MiB/输入token，须另测长输入峰值。
- 候选6正式common-shape BF16 bs16预热失败：pool40960恰等于16×(2048+512)，但scheduler per_req_token_overhead=page_size=8，未覆盖预留；实际511条bs15随后511条bs1，target_batch_fraction=0、status=batch_not_verified。原始8192输出/62.664s的130.729 tok/s不作为bs16正式吞吐。日志和失败summary保留于A10 results/throughput-round6/common-shape/bf16/bs16；无CUDA OOM、无retraction，不伪称OOM。

### 候选7/10：给吞吐服务保留调度分页余量

1. 目的：让common-shape bs16真正同时驻留，消除精确容量边界造成15+1顺序执行。
2. 步骤：本地新增可复用A10吞吐启服脚本；common两方法pool41984，比完整token需求多1024，覆盖16×8页预留及严格比较余量；按Git同步，在同GPU5重跑BF16与VQ。paper-long在GPU6另配对，BF1640960、VQ557568（16×34816+512），后者窗口8/64；各协议单独成表。
3. 范围：a10_throughput_server.sh及本记录。推理算子、调度器、评分器、训练和既有准确率配置不改；已有准确率服务固定source85cd80c5。
4. 成功标准：native Thinking输入2048/output512、bs1/2/4/8/16不变；每cell2预热+3测量，原90%实际target batch和至少8连续decode门槛不放宽；同GPU/环境/profile/代码SHA配对。paper-long BF16 bs2以上保存容量N/A，不发送会串行的请求、不编造OOM。
5. 风险：common BF16多占1.5GiB显存；启服前核对资源。长协议VQ逻辑池约55.8万token仍需实际显存与高batch验收。所有失败原样保留，未启动不标通过。
6. 验证：本地bash -n和diff检查，push/pull loopspec-test；A10新启动readback、import/SHA、health、实际bs16完整输出；成功后完成同一固定候选的全部计划内测量。此配置修改计第7轮，不借环境修复免计。

- 候选7 common BF16 bs16容量修正已通过：2预热+3测量均status=ok，测量178.430/178.501/178.170 output tok/s，真实batch审计通过；GPU5随后切VQ进行相邻配对。吞吐sourcefeba71a892，accuracy source85cd80c5；两提交python树同为6ca971e4043dca1e7362daf5a7af49da6a540822（后者仅新增runner/记录）。六个正式accuracy worker各负责一个method/seed，concurrency1，五任务共30cells/4800generation，均已完成10条HTTP审计后启动；完整结果尚未完成。所有服务和client都在A10 tmux，W&B各run独立记录。
- 补充非VQ回归：test_unified_radix_lock_ref.py 1test/1error、test_dllm_fdfo_kv_reuse.py 6test/4errors（exit1）。新增LoopVQ检测在默认路径无条件调用get_kvcache，原有SimpleNamespace/_FakeAllocator未实现该接口。真实BaseTokenToKVPoolAllocator有该方法；不能由此声称生产allocator缺接口。test_decode_radix_lock_ref初次因CUDA_VISIBLE_DEVICES空字符串触发test_utils端口计算IndexError（基础设施），改为不存在的GPU99隔离GPU后原样复验；其他相关CPU回归继续执行。失败日志round7-regression-*完整保留，未修改fixture。

### 候选8/10：保持非量化分配器原有接口

1. 目的：LoopVQ显式选择专用路径，普通分配/释放不额外要求量化缓存接口，防止未约束Mock属性truthy误入。
2. 步骤：LoopVQKVAllocator增加明确类标记；alloc_for_extend、alloc_for_decode、release_kv_cache仅在该标记严格为True时路由专用路径；其余路径保留原行为，不吞接口异常或改测试fixture。
3. 范围：mem_cache/loopvq_allocator.py、allocation.py、common.py及任务记录；不改算术、码本、采样、评分或当前已冻结测量负载。
4. 成功标准：原失败CPU测试及其余相关默认路径回归通过；真实LoopVQ分配器继续正确进入专用分支，已有核心/wiring测试与真实HTTP生命周期通过。
5. 风险：错误标记会把VQ请求路由到普通分配器，可能破坏物理slot所有权；因此必须验证真实VQ申请、完成、abort和复用，不能仅凭CPU mock成功判定。
6. 验证：本地语法/diff审查，Git loopspec-test同步；A10 GPU99隔离的现有CPU测试，实际GPU VQ核心/服务生命周期复验，保存命令/退出码/提交。固定feba71a892的已启动吞吐测量不与新源码混跑；精确记录运行时版本，修复不追认旧回归失败为通过。

- 候选8本地三文件AST/diff检查exit0。候选7补充CPU回归中其余9文件共131测试全部exit0；decode锁测试用GPU99重试后13test/5errors，确认为MagicMock未配置属性truthy误入VQ，与前两个接口失败同源。为保持六个准确率服务和两套性能任务源码冻结，A10新建独立sglang-loopvq-validation clone，仅跟踪用户origin/loopspec-test，通过Git pull同步候选8；原sglang-loopspec-tp2保持feba71a892直至其作业完成。新验证进程显式PYTHONPATH和sglang.__file__指向验证clone，不修改旧进程源文件。

- 候选8远端独立clone90bbb757ada完成：unittest163通过、2个CUDA-only跳过；backend pytest23通过、10subtests通过，全部exit0。新代码真实full-graph服务17case/72输出请求exit0，与候选6optimized graph的全部token/停止一致，abort/idle/释放/复用/flush通过；round8-serving-vs-round6.json保存比较。只在GPU4与本任务accuracy短时共卡做正确性验证，时间区间存round8-correctness-overlap.json；不计性能或ETA样本，验证服务已停止。GPU5性能服务始终独占。

### 候选9/10：长实验自动衔接与扩展数据冻结

1. 目的：让已授权的长输出完整协议在tmux中自动完成BF16→LoopVQ配对及汇总，并准备后续Thinking扩展的固定数据；不依赖交互终端持续手工切换。
2. 步骤：读取已启动的BF16 paper-long完成标记/summary，验证真实PID、命令、空闲状态及GPU归属后仅停止该自有服务；确认GPU空闲，调用已冻结feba71a892仓库的a10_throughput_server.sh，在同GPU6启动VQ8/64/557568，实际测五档并汇总。容量不足的BF16 bs2以上明确requests_sent=0/N/A。另一独立数据程序下载GSM8K、MATH500、RULER corpus及LongBench指定revision并离线复读。
3. 范围：SGLang任务目录a10_paperlong_pipeline.py及有必要的进程安全测试；LoopVQ/test的experiments/loopvq_reproduction/freeze_extension_data.py；本记录。推理源码保持候选8，实际性能服务的固定sourcefeba71a892保持不动，两个fork仅各自测试分支同步。
4. 成功标准：监督器能识别等待/成功/失败，不覆盖已有测量、只管理已核实的自有PID、不把未运行当成功；真实原长度32768及2预热3测量不变，同GPU/代码/环境校验后生成JSON/CSV/W&B，容量N/A可追溯。数据revision/hash/行数和离线读取一致，原始语料不入Git/W&B。
5. 风险：长任务中途崩溃、PID复用、他人后来占用GPU、网络中断；任一身份或资源检查不满足必须fail closed保留现场，不杀未知进程、不降长度、不升级环境包。64K扩展仍受当前实现限制，此轮不宣称支持。
6. 验证：本地语法和进程/状态守卫的有意义测试、diff；经Git同步至独立验证clone，A10只读inspect已运行BF16状态，再在tmux启动监督器。数据线上下载/实际内容核验/新进程离线复验；后续实际切换、五档结果和退出码独立核实，不能把inspect当完整长协议通过。

- 候选9数据冻结已通过：LoopVQ/test b5f821d（两端一致），AST/diff检查exit0；A10原eval环境在独立tmux依次online/offline均exit0，manifest状态verified_offline。GSM8K train7473/test1319，MATH500 500，RULER 6500中取官方顺序前40条niah_single_2构建语料，LongBench16任务3750条；全部revision、文件/内容hash及固定3-shot示例hash保存在evidence/a10/extensions-data-manifest.json。未生成扩展评测结果；jieba/fuzzywuzzy/rouge当前未安装，只记录缺项，不在运行中的accuracy环境改包。新增文件不属于现有run_evaluation recipe的source清单，既有驱动/任务/包/manifest未变。

- 候选7正式common-shape矩阵已全部通过：同一A10 GPU5（GPU-82009e07-07c1-3a6c-5613-df2e45f63980）、固定feba71a892、同环境/2048输入fixture、512输出、prefix64/recent256、pool41984、optimized profile；每cell2预热+3测量，10cells共50次batch请求全部ok，所有测量target_batch_fraction=1.0。主指标为完整HTTP batch端到端output tok/s的三次中位数（包含prefill），不是纯decode吞吐。

  | bs | BF16 tok/s | LoopVQ tok/s | LoopVQ/BF16 |
  |---:|---:|---:|---:|
  | 1 | 27.7824 | 25.4731 | 0.916882× |
  | 2 | 51.1903 | 44.6318 | 0.871882× |
  | 4 | 83.3295 | 77.6614 | 0.931979× |
  | 8 | 128.5608 | 119.1218 | 0.926580× |
  | 16 | 178.4257 | 157.9593 | 0.885295× |

  全部加速比小于1，当前移植在这组短输出负载未取得速度提升；不由此推断尚未完成的论文长输出协议。测得whole-device显存峰值BF16约70542–70544MiB、VQ约32232–32234MiB；这是当前固定pool、模型与临时缓冲合计，不是单token KV压缩率。汇总exit0，W&B run bwuna24m；证据evidence/a10/common-shape-aggregate/{throughput.json,throughput.csv,wandb.json}与common-shape-matrix-provenance.json。最终服务PID1607168在身份、GPU归属与空闲指标核验后已正常停止。
- 准确率仍在运行：2026-11-23T15:03:04Z快照共30个已保存正式响应、0失败，其中3条达到32768输出上限；未有完整benchmark cell可交付。evidence/a10/accuracy-progress-snapshot.json仅为进度，不能当准确率或最终ETA；VQ seed0还包含此前GPU4正确性短时共卡区间，不用该耗时做公平性能比较。完整30cells/4800 generations保持原计划；paper-long BF16首轮预热32768token已ok（2152.16s），其余重复和VQ尚在运行/等待。

- 候选9监督器本地首次验证通过：`PYTHONDONTWRITEBYTECODE=1 python3 research/loopspec/experiments/2026-11-23-loopvq-reproduction/test_a10_paperlong_pipeline.py -v`，14/14、exit0；两新文件AST与diff检查exit0。覆盖PID复用、未知GPU进程、准确argv/模型/端口、pidfd仅向已核实父进程发SIGTERM、空闲指标、完整原长度结果、容量N/A、排他锁、只读inspect及拒绝覆盖未完成状态。旧BF16 manifest未封存start_ticks，监督器明确从首次接管开始记录身份，不补造启动时证据；实际A10接管及后续自动切换仍须分别验收。

- 候选9 A10接管已验收：监督器提交4593859828e5d64f9998bdbf39e5788d10370ea1，仅独立validation clone同步；原性能/accuracy checkout仍为feba71a892。只读`--inspect` exit0、返回waiting。首次使用loopvq-build的Conda Python被Linux pidfd前置检查拒绝（exit1，日志paperlong-pipeline-r9.log，无服务操作、无status.json）；原样源码改用现场确认具备两个pidfd接口的`/usr/bin/python3` 3.12.3，未改包/推理环境/源码，已通过实际PID/端口/GPU/环境/SHA核查并进入waiting_for_bf16。tmux窗口paperlong-pipeline-r9-system，监督器PID1619623；状态A10 results/throughput-round7/paper-long/pipeline/status.json，日志logs/paperlong-pipeline-r9-system.log。此状态只证明已接管等待，BF16→VQ切换和完整长矩阵仍待运行验收；不标为已完成。evidence/a10/round9-{handoff,bf16-adoption}.json封存本次快照。六个主accuracy服务全部health200、client尚无退出标记；GPU5已释放。之后不要pull修改进入正在运行的冻结checkout；优先读取原tmux/进程/状态再续办，禁止重复启动。

### 性能差距诊断（既有候选只读分析与剖析，未修改推理源码）

用户要求解释论文加速而当前common-shape小于1的原因，并核实是否移植不足。

1. 目的：区分论文协议差异、短上下文固有成本与当前实现的可修复额外开销；不能用工程正确性通过替代性能支持完备。
2. 步骤：核对原论文图表与官方脚本；从现有raw记录比较TTFT/TPOT；在已释放的GPU5用冻结feba71a892服务做BF16/VQ、bs1/16的相同请求剖析，拆分kernel、H2D/同步与GPU空隙。
3. 范围：现有serve/HTTP profiler接口及外部Nsight工具，诊断日志另存；六个accuracy worker与GPU6长协议继续原样运行，不改源码/量化/模型/评分。
4. 成功标准：每个原因标明代码或实测证据与影响范围，明确尚未量化部分；profiler数据只用于归因，不能覆盖无profiler的正式吞吐表。
5. 风险：profiling本身扰动延迟、CUDA graph只显示聚合launch、首次JIT与warmup混入；先预热、采集graph节点并单列结果，必要时缩小诊断采集范围而不改变正式协议。
6. 验证：同一GPU/源码/环境、fixture与输出长度；保存实际服务PID/config、profile请求/退出码、trace及逐类汇总。任何后续源码优化才进入下一候选，预算当前仍9/10。

协议纠正：重新核对论文图14后，Ouro-2.6B Thinking图展示2K输入/4K、8K、16K输出；当前名为paper-long的2048/32768来自官方run_efficiency.sh默认，不是逐图同形状复现。保留已启动实验与路径名，但报告必须标为“官方脚本长输出设置”，不能把32768称作图14唯一论文长度。当前2048/512表本来即独立common-shape协议，不能与图中峰值直接对比。

**论文图表、bit与峰值口径（已核原PDF图像）。**

- [PDF第17页 Fig.14](https://arxiv.org/pdf/2611.05304#page=17)：Ouro-2.6B-Thinking、单RTX 6000 Ada 48GB，横轴为batch size，三个形状是2K输入/4K、8K、16K输出；效率窗口为8 sink/64 recent。
- 图上5.0×、5.2×、6.1×均标在红色实线菱形的**0.25-bit**峰值，量化batch分别64、64、32。它们比较各方法容量内的峰值，不是同batch倍数，也不是本次0.5-bit的成绩或验收目标；在论文该GPU和这组三种长度下，BF16只容纳1至3条长请求。
- [PDF第16页 Fig.13](https://arxiv.org/pdf/2611.05304#page=16)同模型、同RTX 6000 Ada、decode batch1给出下表；论文并未声称短context必然加速。

| bit | 1K context延迟变化 | 8K | 16K |
| --- | --- | --- | --- |
| 0.5 | +3% | −18% | −35% |
| 0.25 | +3% | −19% | −36% |

论文A100补充实验是**1.4B-Thinking/A100 40GB**（PDF第15–16页 Fig.11–12），不能当作当前2.6B-Thinking/A100 80GB的同硬件结果。
官方[run_efficiency.sh:L83](../../../../../LoopVQ/experiments/loopvq_efficiency/scripts/run_efficiency.sh#L83)默认2048/32768，但外层[run.sh:L43–57](../../../../../LoopVQ/experiments/loopvq_efficiency/scripts/run.sh#L43)默认2048/4096，并以注释称32768为最终长度；**脚本注释与发布图14实际展示形状不等同**。
本节据原图纠正前文依据该注释所写的“最终figure为2K→32K”：已启动`paper-long`的2048/32768不变，报告名称为“官方脚本长输出设置”，不声称逐图同形状复现；common-shape仍是独立2048/512、64/256协议。

**计时口径与已完成实测。**

官方[bench_point.py:L11–14](../../../../../LoopVQ/experiments/loopvq_efficiency/scripts/bench_point.py#L11)以`bs*(input+output)/HTTP wall`为E2E throughput；当前以`bs*output/HTTP wall`为主指标。同形状配对的speedup相同，不能用分子定义差异消掉当前小于1的结果。
当前五档speedup为0.916882/0.871882/0.931979/0.926580/0.885295，实际target batch覆盖率均1.0；[完整汇总](evidence/a10/common-shape-aggregate/throughput.json)的首尾档如下（BF16→VQ）：

| batch | TTFT（秒） | 客户端TPOT（毫秒） |
| --- | --- | --- |
| 1 | 0.685730 → 0.716016 | 34.7225 → 37.9324 |
| 16 | 8.554697 → 8.965762 | 73.1569 → 83.9468 |

所有batch的VQ客户端TPOT也更慢，故**不能只归因prefill，也不能归因未凑齐目标batch**。TTFT/TPOT是客户端SSE观察，不是GPU阶段耗时；不能据此虚构kernel占比。当前2次同形状预热+3次median，与官方常规1024/16预热后单次目标测量也须分别记录。

**源码事实与待验证归因。**

- 官方BF16 prefill选FA3；VQ的Triton wrapper在fresh prefill调用同类FlashAttention varlen：[配置](../../../../../LoopVQ/experiments/loopvq_efficiency/scripts/run_efficiency.sh#L206)、[调用](../../../../../LoopVQ/python/sglang/srt/layers/attention/triton_backend.py#L683)。当前两边prefill仍是保留QK/scale/normalized-prob BF16舍入的[两遍attention](../../../../python/sglang/kernels/ops/attention/ouro_attention.py#L70)，`optimized`未整体替换prefill。
- 当前[Ouro模型](../../../../python/sglang/srt/models/ouro.py#L30)保留native RoPE和分步SiLU乘法；RMSNorm已经是带`cast_x_before_out_mul=True`的fused kernel，不能笼统称为全部未融合。共同成本会稀释attention收益，但共同加同样成本本身不能解释从加速翻成减速。
- BF16最大split8/VQ最大split48沿官方配置；VQ已经包含R1/R2共享RoPE/codebook路径。最大split数不等于有效split数，实际按context、batch与SM数量决定：[attention分派](../../../../python/sglang/kernels/ops/loopvq/official_attention.py#L51)、[metadata](../../../../python/sglang/kernels/ops/loopvq/official_metadata.py#L47)。
- 官方[FA3 wrapper](../../../../../LoopVQ/python/sglang/jit_kernel/flash_attention_v3.py#L72)允许CUDA≥12.3的SM8/SM9；不能依据旧“Hopper+”注释断言A100/Ada未用FA3。具体community kernel或sgl_kernel fallback须有运行时证据，源码配置不证明论文二进制来源。
- 采样前的待验证假设：common只遍历约2K–2.5K context，64/256窗口名义保留320而非72 tokens，量化查表/RoPE、HP attention、reduce、编码/flush等成本可能尚未被历史KV读取节省覆盖。当时尚无独立GPU计时；下方node/graph实测更新此判断，不将未量化机制直接认作根因。
- 归因顺序：GPU5配对profile拆分上述VQ特有成本、BF16 attention、共同算子和CPU间隙；再检查较长context的转折。官方reference与target同卡同shape/window配对、各方法容量峰值测试分别报告，不改写当前common结果；FA/模型算术切换须作为后续独立候选，不能替换冻结准确率路径。

**Nsight node四cell实测诊断（更新以上待验证假设）。**

A10 GPU5、Nsight Systems 2026.3.2，BF16/VQ分别采集batch1和16的2048输入/512输出请求，每cell先做2次同形状预热。两臂固定推理提交`feba71a8927502fd334a94a86a6d312ff3f4e543`、同GPU UUID/模型/环境、完整R4、prefix64/recent256、`optimized`及full decode graph；实际请求均验证512token、finish length与`[DONE]`。服务manifest、请求验证及原始CSV见[evidence/a10/perf-diagnosis](evidence/a10/perf-diagnosis/)；[kernel-comparison.json](evidence/a10/perf-diagnosis/kernel-comparison.json)保存逐类汇总、调用数、来源SHA256及口径限制。

下表单位为秒，全部是**Nsight node采样内的累计GPU kernel时长，不是请求wall time，也不是无profile吞吐**。矩阵类含cuBLAS split-K归并；它不能误计为attention reduce。公共算子类含矩阵计算、HF norm、HF prefill及逐元素/RoPE/SiLU符号族，少量同名调度算子无法仅凭符号分离。

| kernel类别 | BF16 bs1 | VQ bs1 | BF16 bs16 | VQ bs16 |
| --- | ---: | ---: | ---: | ---: |
| Decode attention合计 | 3.191 | 4.361 | 19.482 | 23.203 |
| 其中VQ quant主核 | — | 3.107 | — | 17.580 |
| 其中VQ HP attention | — | 0.794 | — | 4.986 |
| VQ encoder（prefill＋flush） | — | 0.057 | — | 0.697 |
| VQ tier metadata | — | 0.0037 | — | 0.0054 |
| KV写入 | 0.182 | 0.196 | 0.278 | 0.263 |
| 公共模型算子＋prefill | 15.675 | 15.690 | 26.670 | 26.606 |
| 全部kernel累计 | 19.094 | 20.365 | 46.494 | 51.108 |

- BF16 decode attention为stage1＋stage2；VQ为quant＋HP＋reduce＋LSE清零。bs1增加1.171秒、36.7%，解释全部kernel增量的**92.1%**；bs16增加3.721秒、19.1%，解释全部kernel增量的**80.6%**。
- bs16的quant主核较BF16 stage1节省1.501秒，但HP attention花费4.986秒，超过这部分节省。因此这两个短context cell的attention消费者尚无净收益，不能只归因编码器或prefill。
- Encoder仅占VQ全部kernel时间的0.28%/1.36%，metadata也很小；先前对其为主要热点的猜测未获此次采样支持。公共模型算子两臂基本持平：共享HF算术会限制可加速部分，但不能解释这里的配对增量。
- 四cell均有511次graph replay；decode各attention核均98,112次，即511×192个R4 slot。bs1有1次prefill，bs16有4次prefill；VQ encoder分别222次（96次prefill＋63×2次flush）、510次（4×96＋63×2），与完整请求生命周期吻合。SQLite确认各graph节点数分别为BF16 6,919/7,303、VQ 7,305/7,689，不能将本次结果归因为graph未生效或遗漏循环层。
- 调度额外操作确实存在：bs16 H2D次数52→8,276，普通CUDA kernel启动30,072→110,135；但H2D本身累计GPU时间只有9.1毫秒。VQ新增大量stream同步，同时event同步API累计驻留时间减少（调用次数未减少）；同步API时长包含GPU执行，不能相加或直接认作新增CPU开销。

[BF16时间线](evidence/a10/perf-diagnosis/bf16-timeline-summary.json)和[VQ时间线](evidence/a10/perf-diagnosis/vq-timeline-summary.json)保留graph内/外kernel、GPU busy union和graph间隔。graph外应按全部kernel减graph kernel计算，Nsight的非graph行`graphId`为NULL，不能用`graphId=0`筛选。**Node tracing会放大stream同步相关扰动，graph间隔还包含sampling、分配和编码工作，不能直接当作正常推理CPU损失；GPU busy union/event span也不等于客户端wall time。**采集/导出区间不能作为生成耗时。后续graph粒度低开销bs16复核已完成，见下表和light-comparison.json，仍不将间隔直接认作正常CPU损失。正式无profile的五档speedup表保持原结果，不能用上述kernel比值替换。

**补充验证计划：官方服务栈同卡反事实。** 继续使用已释放的GPU5，冻结官方LoopVQ `test` 的原始 `python/` 与 `serve_method.sh`（b5f821d的这两个范围与d5d58c83一致），reference独立环境；两臂同模型/bundle、完整R4、同一2048/512的bs16请求、64/256窗口、41984容量、40960声明context、8192 prefill预算及decode graph。官方BF16 prefill配置FA3、VQ配置Triton wrapper，split8/48；每臂2预热＋3次无profiler测量，保存SSE、curl wall、实际config/PID、日志区间与batch证据。清理外部VQ/stash/split覆盖，固定官方默认GPU stash与R2门槛；不把target专用manifest字段伪造到reference。该对照比较完整官方服务栈与当前移植，Torch版本、模型算术和prefill仍不同，不能将全部差异归于单一kernel。无源码/正式运行配置变更，候选预算仍9/10；准确率与长协议继续使用原冻结进程。

**低开销graph trace复核已完成（bs16）。** 两臂重新启动、2次同形状预热、1次采集；Nsight、client及导出均exit0，SSE完整16×512、两臂511次decode graph。使用`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`与本次capture的`cudaGraphLaunch` runtime按correlationId关联，排除Nsight在手动capture前也保留的warmup graph记录；不能直接统计整张GRAPH_TRACE表。原始trace/SQLite留在A10，紧凑证据为`{bf16,vq}-light-{server-manifest,request-validation,timeline-summary}.json`及curl记录。

| graph级采集指标 | BF16 | LoopVQ |
| --- | ---: | ---: |
| 511次graph跨度累计（秒） | 35.7413 | 39.3404 |
| 相邻graph间隔累计（秒） | 0.0355 | 4.0564 |
| 相邻graph间隔中位数（毫秒） | 0.0696 | 7.4339 |
| H2D次数 | 52 | 8276 |
| 同一进程两次未启用capture的预热wall（秒） | 45.8159 / 45.8625 | 53.5306 / 53.4272 |
| 采集请求wall（秒） | 46.0716 | 54.2296 |

graph跨度包含内部调度空隙，与node表的kernel累计时间不是同一指标。两臂对照、正式median与15项来源hash见[light-comparison.json](evidence/a10/perf-diagnosis/light-comparison.json)。

该trace中VQ graph之间确有更长间隔，并非仅node采集才出现；但VQ采集wall仍比正式无profiler中位数约51.8614秒高4.6%，因此4.0564秒不能作为正常运行的纯CPU损失，间隔还包含采样、编码、分配及GPU等待。BF16采集wall较正式约45.9127秒高0.35%。node trace的6.935秒VQ间隔降为graph trace的4.056秒，与采样扰动的影响一致，但这是跨run差异，并非隔离测出的profiler开销；正式speedup不改写。H2D自身总耗时约9毫秒，数据拷贝本身不足以解释秒级差距；逐请求操作与同步串行化是待干预验证的候选机制。

源码复核：当前`official_hp.py`与官方identity HP默认均为BLOCK_N32/BLOCK_H4、warps4/stages1、12 splits、FP32 RoPE及FP16 QK MMA，未发现HP关键优化漏移植。common 64/256稳态保留320–327行，对应10–11个非空HP split；论文8/64为72–79行、3个非空split。窗口是数值/协议变量，不能当作修bug静默替换。明确的调度移植差异在`loopvq_allocator.py:494–508`：提前wait、循环中逐请求新建CUDA索引并调用已有向量allocator；官方`mem_cache/common.py:831–929`批量分配、GPU规划并延后wait。是否及如何修改须进入第10轮计划，当前仍未改推理源码。

**官方服务栈同卡反事实已完成（不是源码单变量实验）。** A10 GPU5，原始官方`python/`及`serve_method.sh`、reference Torch2.9.1+cu128环境；同一模型/bundle、完整R4、bs16、2048/512、64/256窗口、41984池、显式40960 context，BF16 FA3 prefill/VQ Triton wrapper、decode Triton、split8/48、graph `[1,2,4,8,16]`、无radix/chunked prefill。两臂均2预热＋3正式请求，SSE完整16×512、finish length、DONE；日志区间内所有非零decode batch均16、graph=True。原始请求直接复用target `common-shape/bf16/bs16/measure-01.request.json`，重复使用相同内容；官方摘要与manifest位于A10 `results/perf-diagnosis/reference-{bf16,vq}-common-{summary,manifest}.json`。

| 服务栈（bs16） | BF16中位wall（秒） | LoopVQ中位wall（秒） | BF16输出tok/s | LoopVQ输出tok/s | 同栈speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 官方reference | 40.237467 | 39.959264 | 203.591344 | 205.008781 | 1.006962× |
| 当前移植正式common结果 | 45.912664 | 51.861446 | 178.425716 | 157.959345 | 0.885295× |

官方BF16三次测量的已验证中位数为40.237467秒，VQ三次为39.959272/39.959264/39.958586秒；最终以摘要median为准，不把预热计入。**同卡同短请求下，官方接近持平而当前移植明显减速，因此不能将当前小于1全部归于硬件/短上下文。性能适配未完成。** 当前移植相对官方，BF16约多5.675秒、VQ约多11.902秒；两臂差额之差约6.227秒描述完整服务栈差异，不能直接归因allocator某一行。Torch/compiler、prefill、模型算术和调度均有差异，尚缺同栈干预实验来隔离各项净影响。官方1.007×也不是短请求已获得显著加速的证据，论文长context/8+64/容量峰值仍须单独复现。

16:16 UTC原X5/A10会话断开；确认旧session与ControlMaster均失效后，经已有X5别名恢复持久跳板与A10会话（session21643/63365）。恢复后官方VQ客户端exit0，六路accuracy及长输出服务health200，长流程仍waiting_for_bf16；tmux任务未因这次断线中断。随后A10会话再次确认断开，而X5通过实际hostname响应确认仍在线；复用同一X5 ControlMaster重建A10 session9260，没有绕过跳板或重复启动实验。官方BF16/VQ两臂在全部请求验证完成后，经PID/argv/UID/start_ticks/GPU归属与空闲指标核验发送SIGTERM；服务日志均记载remaining requests=0后退出，wrapper exit137，不将其误报为服务自然exit0或请求失败。正式客户端两臂均exit0。原始摘要、manifest、launch记录和[reference-stack-comparison.json](evidence/a10/perf-diagnosis/reference-stack-comparison.json)已完整取回本地并校验。

**官方node补充采样未成功导出，不能用于kernel归因。** 单独启动同配置官方VQ服务（PID1724943），2预热＋1采集生成均完整16×512且HTTP200，采集请求wall40.521373秒；但`stop_profile`等待Nsight导出，300秒后curl超时exit28，Nsight report停在20%且始终0字节。此40.521373只作为诊断请求记录，不进入正式median，也不证明profiling控制流程成功。核实所有请求DONE、used_tokens/queue均0、已验证PID/argv/UID/start_ticks及GPU进程归属后结束该诊断服务；running gauge仍残留最后batch的16，明确未通过原先“三个gauge均0”的正常空闲检查，日志随后确认remaining requests=0。Nsight还等待重挂父进程，SIGINT后agent shutdown RPC超时；仅针对已核实本诊断UUID/父子归属的Nsight agent/importer清理，保留请求产物和失败日志；工具退出时自行移除了空报告。未取得可用官方kernel trace，不能从这里推出quant/HP相对target的耗时差；官方无profiler对照、target有效node/light采样均不受影响。

剩余因果边界：当前node表的3.721秒是target VQ对target BF16的attention增量，不是target对官方VQ的差额。两栈除了调度外还有HP→quant与quant→HP执行顺序、int64/int32 loc、实际编译器和BF16分母实现差异；没有单变量实测前，不分配各项贡献百分比。源码复核也确认官方decode仍对非空K scratch执行fused rotary，不能声称官方完全省去K RoPE而target遗漏了这一数学捷径。下一候选优先评估已有向量allocator的batch调用、保留原等待顺序和量化数学，并做同栈改前/改后对照；尚未开始第10轮，预算仍9/10。

诊断收尾快照为2026-11-23T16:43:29Z：官方诊断服务及三个Nsight进程均已消失，GPU5无计算进程；六路accuracy和GPU6长输出服务health200，长流程仍waiting_for_bf16。详见evidence/a10/perf-diagnosis/{reference-vq-node-status,diagnostic-final-status}.json。所有有效诊断产物与失败状态均归档，未改变冻结推理源码、准确率配置或正式吞吐结果。

## 已实现代码导航

本节描述本地 `90bbb757a` 的 standalone LoopVQ 实现，不新增运行行为或修改轮次。官方算法/源码、当前移植、算子对照、真实服务与完整 benchmark 是不同证据：代码导航说明实现位置，实测结论及运行时 SHA 以前面的实验记录和对应原始结果为准。已启动准确率、吞吐进程分别冻结在其记录的版本，不能把本地 HEAD 追认为运行版本。以下链接均相对于本记录，指向 SGLang 仓库内文件。

### 入口与核心调用链

```text
--loopvq-codebook-dir + --ouro-decode-profile
  → 参数兼容检查 → 固定显存预算 → LoopVQKVPool / LoopVQKVAllocator
  → 加载 K/V bundle → 编码表、按深度切片的解码表、共享索引映射

新请求：alloc_for_extend → 完整 R4 prefill → live BF16 attention
        → 每层保存 raw K/V → 同一基础层四轮到齐 → coupled encode

生成：alloc_for_decode → 分配 HP recent slot / 必要时 demote 最旧8个token
      → 先编码 K/V，再发布位置映射 → 模型完整 R4 decode
      → hf_reference 或 optimized attention → 第4轮 LM head

结束/abort：等待先前 forward 的 stream event → 释放完整 owned row
          → 回收 quant/prefix pages → 重置请求 ring → 复用 request slot
```

| 环节 | 关键文件 | 阅读重点 |
| --- | --- | --- |
| 显式启用与拒绝不支持配置 | [model.py](../../../../python/sglang/srt/arg_groups/fields/model.py)、[kv_cache_hook.py](../../../../python/sglang/srt/arg_groups/kv_cache_hook.py) | bundle目录、prefix/recent窗口、两种decode profile；TP/PP/DP/CP/DCP=1、BF16、page-size=8、无speculation/radix/chunked prefill等约束 |
| 显存预算与实例化 | [pool_configurator.py](../../../../python/sglang/srt/model_executor/pool_configurator.py)、[kv_cache_configurator.py](../../../../python/sglang/srt/mem_cache/kv_cache_configurator.py) | `LoopVQPoolConfigurator`先扣固定HP窗口与码本，再计算quant容量；构造专用pool/allocator |
| bundle布局与加载 | [loopvq_pack.py](../../../../python/sglang/srt/mem_cache/loopvq_pack.py)、[loopvq_codebook.py](../../../../python/sglang/srt/mem_cache/loopvq_codebook.py) | `code_layer_of`、`step_of`、`slot_codebooks`与严格identity/G16验证 |
| 模型接线 | [ouro.py](../../../../python/sglang/srt/models/ouro.py)、[loopvq_backend.py](../../../../python/sglang/srt/layers/attention/loopvq_backend.py) | query沿用native RoPE；缓存保存raw K；prefill/decode分别分派，模型仍执行完整4轮 |
| 缓存所有权与producer | [loopvq_pool.py](../../../../python/sglang/srt/mem_cache/loopvq_pool.py)、[loopvq_allocator.py](../../../../python/sglang/srt/mem_cache/loopvq_allocator.py)、[encode.py](../../../../python/sglang/kernels/ops/loopvq/encode.py) | 共享quant索引、独立HP数据、跨轮暂存、ring demotion与编码后发布 |
| attention消费者 | [official_attention.py](../../../../python/sglang/kernels/ops/loopvq/official_attention.py)、[official_metadata.py](../../../../python/sglang/kernels/ops/loopvq/official_metadata.py)、[attention.py](../../../../python/sglang/kernels/ops/loopvq/attention.py) | optimized的分段metadata与60-slot scratch；保留HF参考两遍路径用于核对 |

### G16 / c4 为什么是0.5 bit

Ouro有48个基础层、4轮，逻辑KV slot为 `s = depth * 48 + layer`，共192个。一个LoopVQ向量由同一token、基础层和head的4个channel跨4轮组成，维数 `G = c * Tc = 4 * 4 = 16`；256个centroid对应一个uint8索引，所以索引payload是 `8 / 16 = 0.5 bit/标量`。page-size=8则表示分配/flush的token粒度，与8-bit索引不是同一概念。

向量排列是channel-major、depth-minor：深度 `m` 读取centroid位置 `m, m+4, m+8, m+12`。四轮共享同一个索引，但读取不同分量，不能理解为四轮共享相同KV值。加载后encode表为 `[48,16,32,256,16]`，decode表为 `[192,16,32,256,4]`；quant arena的物理层数只有48，HP arena仍有192个独立深度。

K/V索引合计每token `48 * 16 * 32 * 2 = 49,152 bytes`，是完整BF16 KV的1/32。该比例不包含BF16 prefix/recent窗口、768MiB的K/V encode+decode表、模型权重、metadata和prefill暂存，不能直接当作整个服务的显存压缩比。

### 训练产物到服务加载

训练在用户的独立 `LoopVQ/test` 仓库完成：`experiments/loopvq_reproduction/run_training.py` 固定数据、模型和配方，调用 `calibration/build_ouro_bundle.py` 收集各slot的K/V与梯度、逆RoPE并拟合Fisher加权k-means；`calibration/finetune_codebook.py` 冻结模型权重，仅通过clean/quant KL更新K/V码本。nearest-centroid assignment不求导，重建通过可微gather和两阶段backward传回centroid。

最终bundle的 `codebook.pt` / `vq_v_codebook.pt` 是 `torch.save` 字典，包含FP16 centroid、布局、identity forward/inverse和zero mean；两个identity rotation文件及 `build_info.json` 保留官方导出格式。另存的恢复checkpoint包含FP32训练码本、Adam、RNG、epoch/step和best state，不是服务加载文件。

Target实际只读取K/V两份码本；loader在CPU校验完整slot/head覆盖、256×16 FP16 centroid和纯G16/c4/R4约束，再建立GPU表。TaSQ变换、非零mean、归一化、额外scale等被显式拒绝。训练沿用官方HF外层threshold=1.0 readout；线上服务与HF probe明确取最终第4轮LM head，两者不作代码层面严格等价声明。

### Prefill、decode与缓存生命周期

Prefill只接受完整新prompt，分配逻辑顺序为protected prefix、quant中段、recent尾部。当前attention读取live BF16 K/V，量化缓存用于后续前向；前3轮的compact raw K/V留在GPU，到第4轮同一基础层到齐后共同编码并释放暂存。该暂存峰值约1.125MiB/输入token，因此长decode算子通过不等于同长度fresh prefill容量通过。

Decode每个请求每步增加一个HP token。recent ring比配置窗口多8个位置；达到阈值时，将最旧8个token的所有深度联合编码。`alloc_for_decode`先写完整K/V索引，随后更新请求位置映射；分配和flush位于CUDA graph之外。`encode.py`直接读取四轮张量或HP arena并scatter索引，不展开完整历史解压KV。

通用入口在 [allocation.py](../../../../python/sglang/srt/mem_cache/allocation.py) 仅在专用allocator类标记严格为True时切入LoopVQ。完成/abort在 [common.py](../../../../python/sglang/srt/mem_cache/common.py) 按完整owned row释放，避免quant尾部半页按逻辑位置对齐造成重复释放或泄漏；allocator按物理页去重回收quant/prefix，recent是请求专属ring，释放时重置游标。当前不支持共享prefix、CPU KV offload或缓存迁移。

### Attention与CUDA graph边界

`hf_reference` 使用自写两遍混合attention，保留QK、scaled score、normalized probability的BF16舍入边界。`optimized` 使用移植的官方G16 MODE1 quant CUDA、identity HP stage1及统一LSE reducer：[official_quant_cuda.py](../../../../python/sglang/kernels/ops/loopvq/official_quant_cuda.py)、[official_hp.py](../../../../python/sglang/kernels/ops/loopvq/official_hp.py)、[official_reduce.py](../../../../python/sglang/kernels/ops/loopvq/official_reduce.py)。其算术与HF参考profile不同；norm、query RoPE、SiLU和prefill仍保持本仓库的HF语义，并未整体替换为官方模型实现。

优化路径预分配HP12＋quant48共60个split slots；每层将LSE重置为负无穷，空段不会复用旧输出。slot0在GPU上将位置有序的请求KV列表拆成quant与HP两段，其他191个slot复用metadata；quant与HP保留各自正确的绝对RoPE位置。B=1用R1，其余用R2，实际split数按组内最长序列和SM数量确定。固定sm80、CAP2048和当前context≤40960约束，JIT及shared-memory attribute设置必须在capture之前完成。

[decode_cuda_graph_runner.py](../../../../python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py) 将capture/padding写入保留的HP dummy page，metadata把dummy请求视为空序列。[scheduler.py](../../../../python/sglang/srt/managers/scheduler.py) 记录forward完成event，在可能demote/free/reuse缓存前建立stream等待；pool的 `wait_pending_forward` 是GPU stream依赖，不是每步强制CPU同步。[recurrent_runner.py](../../../../python/sglang/srt/model_executor/recurrent_runner.py) 显式排除LoopVQ，当前服务不联合启用LoopSpec。

### 官方优化尚未整体移植的部分

- Prefill保留本仓库的HF语义两遍kernel；未切换为官方服务脚本默认的FA3或官方Triton单遍在线softmax。
- Producer是专用Triton Tensor Core融合实现，并非官方encoder原样移植。当前16-token tile、现场计算centroid norm、一次FP16 MMA并直接scatter；官方G16 producer使用64-token tile、预存norm、FP16高低两部分MMA及后续scatter。已测真实数据编码一致不代表所有极小值/near-tie输入都逐位等价。
- 未整体移植官方GPU flush plan/apply；当前分配和demotion含CPU逐请求规划。也未移植CPU stash预算、chunked-prefill shadow、多bit宽、多模型及TaSQ路径。
- R1/R2与动态split受batch/配对长度影响，不承诺跨batch逐位相同。deterministic/split-tile/static-split选项未纳入当前支持声明，不能套用generic BF16路径的batch-invariance保证。

因此common E2E包含两方法共同的prefill成本，以及LoopVQ特有的暂存、编码、周期flush和调度成本。正式加速比表示当前移植的同机配对结果，不能等同于纯decode kernel加速，也不能把所有差异都归因KV压缩；未有独立阶段GPU计时时，不根据HTTP总时长推断各环节占比。

### 验证入口与证据范围

[test_loopvq_core.py](test_loopvq_core.py) 覆盖布局、loader、ring/free/reuse及参考算子；[test_loopvq_wiring.py](test_loopvq_wiring.py) 覆盖参数解析和预算；[test_loopvq_official_kernels.py](test_loopvq_official_kernels.py) 使用真实码本、原官方模块及CPU metadata oracle核对优化算子/graph。上述算子测试不替代真实模型准确率。

[baseline_probe.py](baseline_probe.py) 明确HF最终第4轮readout并保存token/logprob；[serving_probe.py](serving_probe.py) 核对真实服务的eager/full、padding、边界、abort和复用；[benchmark_throughput.py](benchmark_throughput.py) 校验实际batch、固定长度、graph状态和同机配对。已通过/失败/未运行状态均沿用本记录中的具体提交、命令和证据，不把接口测试、短输出审计或算子通过写成完整benchmark完成。


## 2026-11-24 candidate 10/10: context sweep and loop-level attribution

User confirmed 1K/2K/4K/8K means visible decode history; HP window stays 8+64.
1. Purpose: separate attention, shared model kernels and scheduling overhead, while auditing official differences.
2. Steps: frozen-inference unprofiled HTTP matrix, then short Nsight node traces and four-loop attribution.
3. Scope: experimental measurement/analysis helpers and this record only; no inference source changes. Restore failed long-output supervisor using a new, independently hashed metadata manifest and new state directory, preserving previous failure evidence.
4. Acceptance: all four contexts and bs1/2/4/8/16 measured or explicitly capacity limited; paired inputs, actual graph/batch evidence; each loop's attention/common GPU costs reconciled with all graph kernels.
5. Risks: client TTFT is not server prefill; short E2E dominated by prefill; node profiling perturbs CPU/wall time. Keep primary throughput unprofiled. BF16 capacity prevents some large-context pairs. Generic pointwise kernels attributed by execution position, not invented fine-grained names.
6. Validation: A10 GPU5, full R4 Thinking, frozen feba inference, same model/environment and HP8+64. Input L-1, 33 output tokens gives 32 decode steps at histories L..L+31. Profile uses 9 output tokens/8 decode steps at L..L+7, bs1 and4. Two warmups, three unprofiled measurements. Verify 772 norms/192 layers/4 loops per graph and preserve trace/schema/provenance. Use tmux throughout. Candidate10 starts here; no performance fix is attempted.


### 完整服务路径对齐审计（冻结 feba，官方 d5d58c83 的 Python 树）

这里的“官方”指作者仓库实际服务代码，不等同于论文每张图的配置。“移植”指本轮真正运行的 frozen feba，不以开发分支 HEAD 代替推理提交。以下覆盖本实验从请求到释放的活动路径；不声称整个两个 SGLang 仓库逐行相同。

| 环节 | 官方实现 / 配置 | 当前移植状态 | 性能或正确性含义 |
|---|---|---|---|
| HTTP、tokenizer、scheduler、输出传输 | 作者 vendored SGLang | 基于 v0.5.22 的 fork；服务框架版本不同 | 整栈差异不能全归因某个内核；本次两臂同机同框架、相同 input_ids、增量 SSE |
| 模型与循环 | Ouro 48 层共享权重、4 次循环、深度独立 KV | 对齐完整 R4，未启用 early exit 或 LoopSpec | 没有通过少算 loop 获得速度；不能跨深度直接共享 BF16 KV |
| 码本与压缩 | 四循环耦合、G16、256 centroid；HP sink/recent、其余索引 | 0.5bit noTaSQ、相同训练 bundle；K/V FP16 centroid、8-bit 索引 | 0.5bit 是对原始四循环标量的有效位宽，不是每个索引半比特；无全历史解压中间张量 |
| Q/K RoPE | fused RoPE；FP32 cache/product | native Torch，保持 HF 中间 BF16 舍入；有 index/cast/mul/add/cat 多个小 kernel | 两臂共同成本；不是完全官方模型算子路径。VQ decode 还计算未使用的 rotated K；官方 scratch 也仍参与 K RoPE，不能说官方完全不算 K |
| RMSNorm / residual | 官方 fused 实现及其舍入语义 | 已使用 HF 语义 fused RMSNorm，cast_x_before_out_mul；residual 单独 add | 不是“全部 norm 未融合”；每图 772 次 norm 可用于核对四 loop |
| MLP | SiluAndMul 融合 | SiLU 后 BF16 舍入，再独立 multiply | 公共模型路径更多小 kernel；直接换 fused 会改变数值，需单独正确性验收 |
| Fresh prefill attention | 脚本默认 FA3；VQ wrapper 的 fresh 路径也可走 FA3 | 两臂保留 HF 舍入的两遍 ouro_attention | 明确未对齐，长输入 TTFT/E2E 可能受影响；不能用 client TTFT 直接充当 server prefill GPU 时间 |
| Prefill KV 暂存与编码 | 官方 stash、producer、布局与预算 | 四 loop 原始 KV 暂存，专用融合 producer，完成 K/V 后发布索引 | 含独立内存预算和额外编码；新上下文测试记录此图外成本 |
| Encoder | G16 的 64-token tile，预存 norm，高低两部分 MMA，后续 scatter | 16-token tile，现场 norm，一次 FP16 MMA，融合 gather/scatter | 不同实现；已有实际数据编码一致不构成 near-tie 全域逐位证明；旧 trace 中不是主要耗时 |
| Decode BF16 baseline | 官方 generic Triton split stage1/2，split8 | 同类 split8，但显式 FP32 product/reduction 有差别 | baseline 也非逐行官方，不能只检验 VQ 一臂 |
| Decode VQ quant | 官方 MODE1 G16 CUDA、动态 split、R1/R2 | 已移植这一计算核心，split 上限48、bs1 R1，其余 R2 | 不是仍在用逐元素参考解码。算术核对通过不等于 CUDA 编译产物/访存布局性能一致 |
| HP attention | BF16 HP，经 FP32 RoPE 与 FP16 MMA，12 个 split slots | 核心算法、默认 tile/warp/stage 对齐 | HP 不是免费旁路；64+256 时有10–11非空段，8+64通常3段。优化空段是进一步优化建议，并非漏掉已有官方功能 |
| Quant/HP 顺序与合并 | HP→quant→统一 LSE reduce | quant→HP→同类 reduce；每层先 LSE fill(-inf) | 顺序不同，潜在 cache 影响尚未做单变量验证。LSE fill 和 reduce 是真实额外开销 |
| Metadata | 官方 tier index / split 元数据 | slot0 更新一次、191个后续 slot复用；int32 地址，固定最大 context 网格，debug assert | 并非每层重建；旧 trace 占比很小。官方部分地址int64，当前 unit scale 零stride，布局/模板不同 |
| HP recent 分配 | 一次批量 alloc_hp_recent | CPU 遍历请求，每请求创建 CUDA id tensor，再调用分配 | 明确缺少官方批量路径；小 H2D/同步次数增加，不能把 CPU 同步等待和 GPU 耗时相加 |
| Decode flush 调度 | GPU plan / free 先做，后 wait_forward，再 apply，尽量与旧 forward 重叠 | alloc_for_decode 入口先 wait，CPU逐请求规划，每8 token批量 encode后更新映射 | 明确未对齐，可能减少 CPU/GPU 重叠；必须保持前一轮读完才覆盖的正确性约束，不能简单删除等待 |
| 缓存发布/生命周期 | 官方完整 mixed allocator/radix 生命周期 | 两种码本都编码后才发布；demote/free/reuse 有 stream 依赖；abort/reuse 已做真实服务验证 | 当前独立 AR 范围已验证；性能修复不能破坏拒绝/释放/复用顺序 |
| CUDA graph | 官方支持 decode graph | 当前完整 graph，固定桶1/2/4/8/16；元数据缓冲持久化，分配在 graph 外 | 图外 CPU 循环没有被 graph 自动消除；每请求记录真实 batch 与 graph，不仅看启动参数 |
| Memory / batch admission | 官方可按容量获得更大 batch，CPU stash /相关预算更完整 | 当前独立 pool / stash预算；BF16和VQ容量不同 | 相同 batch 的速度与最大容量吞吐是两种问题；容量不够的格子记录 N/A，不跨batch拼加速比 |
| Radix / chunked prefill / prefix reuse | 官方 mixed 路径具有相关实现 | 当前显式禁用或拒绝；TP/PP/DP均1；无 speculative | 当前功能边界，不把“已禁用功能”说成本次活动路径的直接耗时；开启这些功能要另做移植和正确性验证 |
| Sampling / stopping | 官方脚本 pytorch sampler，可随机采样 | 本次温度0、ignore_eos、固定 token数；准确率任务另用原设定 | 吞吐不受不同提前停止干扰；随机采样不能宣称同seed逐token一致 |
| 编译与运行时 | 参考 Torch2.9.1+cu128、独立环境 | Torch2.14.1+cu130、Triton3.8，sm80 A100 | 同GPU整栈对比仍含运行时差异；不是 allocator 的严格因果消融 |
| 论文口径 | Fig13为0.5bit bs1，Fig14高峰为0.25bit且容量batch不同；8+64窗口 | 旧common测试2K/512、64+256；本轮改为指定history长度、8+64 | 旧测试不能直接否定论文；本轮短decode扫描也不能冒充Fig14长输出复现 |

关键源码：当前 `python/sglang/srt/mem_cache/loopvq_allocator.py:482`，官方 `LoopVQ/python/sglang/srt/mem_cache/common.py:831,866,929`；当前 `python/sglang/srt/models/ouro.py:29,40,53`、`python/sglang/srt/layers/attention/loopvq_backend.py:8`、`python/sglang/kernels/ops/loopvq/official_attention.py:37`、`official_metadata.py:73`、`encode.py`。这些代码证据分别支持批量分配、wait位置、模型舍入和attention dispatch，不替代运行时间测量。


兼容性边界补充（`arg_groups/kv_cache_hook.py:34`）：当前还明确拒绝 CP/DCP>1、hierarchical/unified cache、KV sharding、LoRA、two-batch overlap、mixed chunk、torch.compile、memory saver、PD disaggregation、priority preemption、post-capture KV resizing、RoPE scaling 和非 Ouro-2.6B R4 几何；decode仅 full graph/eager，prefill graph关闭。它们均未在本次配对启用，故不能列为当前实际慢下来的已证实原因。普通 scheduler overlap 与 two-batch overlap 不是同一个开关。正式性能修复优先级应由下面的时间测量决定，而不是把所有功能都打开。

Candidate10 validation so far: local py_compile and git diff --check exit0; commit 80c873d3815edb6446166e6ac1e197190acb5903 pushed and pulled into validation checkout. Frozen inference checkout stays feba. On A10, old BF16 and VQ bs1 node traces both pass exact 511 graph / 772 norm / 192 layer accounting. Their mean graph kernel sums are 35.8717485 ms and 38.2891168 ms; these are the OLD 64+256-window 2K+512 traces, not the new 8+64 sweep. New unprofiled run and queued short profiles use tmux on GPU5. Original long-output recovery verified exact K/V hashes and reached VQ bs1 measurement on GPU6; old failed pipeline directory remains unchanged.


调度审计进一步定位：最早的等待不只在 allocator。当前 `scheduler.py:2014` 在 `get_next_batch_to_run` 之前已经调用 pool.wait_pending_forward；其内部是 schedule stream 的 wait_event 并清除 pending event，因此 allocator:494 的第二次调用通常已无待处理 event。官方 scheduler 的 `maybe_wait_mixed_kv_forward_done`（约1473行）仅 stash event，真正 wait 延后到 common.py:929 的 apply 前。当前把依赖提前到了整轮 batch planning 前，而不仅是 flush 前；随后逐请求 CUDA tensor创建/同步可能被排在上一轮forward之后。这是明确未对齐的调度位置，不能称为“只是alloc函数里多一次wait”。wait_event本身不等于CPU同步；是否变成host stall由后续同步和stream依赖决定。


## 2026-11-24 user protocol correction; candidate 11/20

User explicitly changed the primary requirement to DECODE throughput only, then clarified fixed 2048 input tokens and 1024/2048/4096/8192 NEW output tokens, ignore_eos=True. User explicitly authorized ten additional candidates, cumulative 11–20; prior count is preserved. The context-20261124 short-output experiment is auxiliary evidence only, NOT this requested matrix. Its controller PID2392663/start346283832 was verified and SIGSTOP'ed before the VQ arm; all 17 feasible BF16 cells completed, 3 capacity rows saved. No short-output E2E ratio will be substituted for requested decode throughput.

Six-part plan:
1. Purpose: measure pure, full-batch decode throughput under the corrected long-output protocol, with four-loop kernel attribution.
2. Steps: archive short sweep; fixed native-chat 2048-token fixture; 2 short warmups plus 3 fixed-length measured runs per feasible cell; independent Nsight short capture windows during real 8192-token generation; automatic paired summaries/W&B.
3. Scope: experiment launch wrapper (new decode-sweep protocol), context launch utility optional parameters, new decode sweep/analysis helper. Frozen inference remains feba; no model, cache or scheduling fixes in this candidate.
4. Acceptance: outputs exactly requested length and actual full batch/graph verified; primary tokens/s uses earliest common emitted count >=2 through final output for all requests, all prefill excluded; raw SSE timestamps retained. Counts must start <=8, otherwise attribution fails. Same physical GPU and input IDs per BF/VQ pair. Missing capacity explicitly N/A. Each captured graph passes 772 norm /192 layer accounting.
5. Risks: long matrix takes hours; SSE adds host/transport jitter, so profile reports GPU time separately. Nsight only runs in the separate profile arm. Long-output history grows from ~2K to 3K/4K/6K/10K; do not call those fixed-context tests. Capacity pools differ; reserve a BF16 safety margin by using the already validated 41984-token pool.
6. Validation: static checks; pure event-analysis checks against completed real short-run evidence; initial actual long-output cell plus all planned matrix cells; preserve failures, return codes, GPU identity, source and environment provenance. Profile samples near output 1K/2K/4K/8K on bs1 and4 during actual fixed-2K generation, requesting eight steps per capture and recording actual captured count. Completed results are never inferred from a running tmux window.
