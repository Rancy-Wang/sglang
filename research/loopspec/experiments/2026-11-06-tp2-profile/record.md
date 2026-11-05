# A-series TP2 AR / DFlash / single-proposal LoopSpec

## 身份与预算

- 用户明确要求：DFlash修复后在A系列单机TP2重新比较AR/DFlash/LoopSpec，输出加速比、acceptance及详细Profile；LoopSpec仅一个proposal位置。
- 基线：935d48ddbf0a0f6763f6afcd00cded318f206ad2（Wave撤除已通过X5回归）。分支loopspec-test，正式loopspec不动。
- 状态：进行中，候选3/10。TP2接入是新的用户任务，不重开原DAS-Wave优化预算。
- 口径：bs1/2/4/8/16、每格全30题AIME25、max_new_tokens1024，Thinking模板、R4、BF16、greedy、CUDA Graph全开。LoopSpec depths=[1]，只保留第一proposal分支；流水线仍可有不同token的在途lane。用户已确认完整batch矩阵，详细Profile聚焦bs1。

## 修改前计划

1. 目的：当前LoopSpec/CDB只接受TP1，新增TP2才能完成用户要求；保持DFlash已验证导出/feature语义。
2. 步骤：核实A机器和环境，新增TP2接线与分布式graph capture、同步rank间决策；驱动参数化模型/TP；短测正确性与graph后运行full30，另取三题三位置32token窗口Profile。
3. 范围：model_executor/recurrent_runner.py、speculative/loopspec/{config.py,worker.py}，公共run_aime25.py及utils/recurrent_trace.py导出TP rank标签；新增本目录的测试/TP2 trace分析与报告。必要时修正Profile rank标签和本次发现的TP2接口问题，记录各轮。
4. 标准：TP2两rank正常capture/replay且无死锁/错误，单proposal，所有方法30唯一题与输出/停止完整；配对吞吐及TTFT/decode/E2E，acceptance分母可追溯；Profile区分两个rank和通信，不能将双卡kernel时间相加当延迟。
5. 风险：collective捕获、LMHead词表分片、rank随机决策分歧、KV映射/图复用、BF16归约导致argmax差异、Profile跨rank重复统计。不能把长序列数值差异隐瞒为无损通过。
6. 验证：CPU/静态及已有LoopSpec调度测试；A系列同SHA下AR TP1/TP2短prompt与逐token检查，DF/LS对TP2 AR短测、bs1图运行、两rank一致；然后三算法full30×1024，Profile q0/q2/q20的0/50/90%轨迹各32token，统计AR每token、DF每周期和LS每loop/每输出的prepare/commit/forward/attention/通信等分布。性能与Profile分开运行，同GPU同配置，保存完整prompts/IDs/launch/config和原始trace。若数值不一致，保存首分歧并单列为诊断性性能结果。

## 环境选择

- 持久X5 session24353（shell1079580，ControlPath /tmp/loopspec-remove-ruoxi-x5.mit.edu-22）；A系列均通过此master转发，本地密钥认证。
- A10 session88097：8卡A100全部被其他人的计算任务占用，根盘722MiB可用；跳过。
- A11 session63925：8卡A100全部被其他人的计算任务占用，swap满；跳过。
- A12 session11376，ruoxi07@a12.mit.edu，shell3171588：8×A100-SXM4-80GB全部0MiB/0%，GPU0/1 NV12互联；主机available1.9TiB、swap0，home共享NFS可用2.7TiB。选择GPU0/1，UUID分别5a664f81-ad65-8947-c6c2-cbb7b4b50d4e、c078abcf-b59b-9999-9e5f-1c89b8ae654c。
- 现有~/repo/sglang是用户ddtree分支，不能切换/污染；将建独立clone到~/repo/sglang-loopspec-tp2。现有spec/dfl环境是Torch2.11、SGLang0.5.14/0.5.15；需本任务隔离环境，不升级它们。无docker/sudo权限。

## 候选1/10

- 开始前证据：LoopSpec config明确TP=1；recurrent executor selector亦TP=1；RecurrentCoreGraph仅裸torch.cuda.graph，未进入distributed.graph_capture。实际worker用executor.finish/LogitsProcessor（已支持词表all-gather），非旧readout_recurrent局部lm_head接口。
- 验证：待执行。

- 环境检查：A12驱动570.195.03，已有cu130环境实际CUDA初始化失败。将从X5已验证容器复制依赖到本任务独立venv，并使用容器自带CUDA13 compat用户态库，仅设置任务LD_LIBRARY_PATH，不改驱动或系统依赖。模型/数据独立复制并校验hash，代码仍仅经Git。

- 跨本地SSH中转资产传输实测约1MiB/s，不适合22GiB依赖；停止本任务传输通道，保留SSH master和部分文件。增加serve_artifacts.py作为一次性TLS白名单资产流：仅A12源IP+随机Bearer认证、固定四个只读路由，证书/凭据通过已认证SSH传递；A12直接从X5读取环境/模型/数据，完成后停止服务。源码仍通过Git同步。
- 本地候选1静态检查：AST、git diff --check、Profile区间重叠union/通信分类/request+decode过滤断言通过。

## 候选2/10：加入用户确认的batch矩阵

- 候选1提交428bf92ab4169d4c598f6414054b82201ec2f277；本地静态及X5既有CDB/LoopSpec CPU测试9项通过，退出0。A12 GPU验证尚未执行，环境仍在传输，不标通过。
- 用户确认重跑bs1/2/4/8/16，详细Profile聚焦bs1。因此在同一TP2任务加入cohort并发驱动与矩阵统计，开始候选2/10；运行时代码仍沿用候选1。
- 新增范围：run_aime25.py支持batch-size，按cohort记录wall time，分母为并发组实际elapsed而不是逐请求延迟之和；新增summarize_matrix.py输出同batch相对AR加速、逐请求TPOT、acceptance和实际峰值batch。full30用统一context65536、KVcapacity40000（是否可用需GPU短测），避免旧X5 capacity导致bs16缩批。
- A12模型/数据hash核对通过：target c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080，draft d7757a0958fe3d62df63baf3451457df4f0f05f0a3f949cf9ca70e44cbf9067c，dataset b1cb6f70eb12b5a0e68f9ea88be9a35c6a3c66f0d00112b5b26160b042326b19。

### 候选2环境与检查进展

- SHA af7eb8ee264ed88ca45752c53d69a35077acd08b 已push并A12 pull同SHA。矩阵合成测试3方法×30请求：并发wall分母得到1/2/4加速、两个spec accepted/tested=.5、AR IDs30/30，退出0；AST/diff检查通过。
- 独立环境：`/homes/ruoxi07/experiments/loopspec/tp2-20261106/environment/venv`，依赖来自X5验证容器，Torch2.14.1+cu130 / Transformers5.19.0 / Triton3.8.0 / FlashInfer0.7.1 / SGLang0.5.22。PYTHONPATH指向A12独立clone/python。
- 用户态compat为CUDA13的580.126.20库，仅通过本任务LD_LIBRARY_PATH启用，A12系统570.195.03驱动不变。实际Torch CUDA张量执行通过。用户目录安装Rust1.92.0及ninja1.13.2（版本与源环境一致），不改全局环境。
- 传输结束：临时TLS server PID1094385已终止、30443不再监听；token/private key已删除，A12凭据文件已删除。模型与数据hash匹配，完整依赖传输exit0。
- 初次AR TP1短测`round2/ar-tp1`失败：Triton driver.c编译找不到系统Python.h，server退出-9/driver退出1；无模型输出。保留日志，GPU进程已退出。这是环境阻塞，未改源代码。补充X5独立CPython3.12.15及headers到任务目录并重定向新venv，重试新目录。

- 第二次AR TP1短测`round2/ar-tp1-v2`在JIT链接失败：系统没有`c++`命令；nvcc编译本身通过。设置任务CXX=/usr/bin/g++-13、CC=/usr/bin/gcc，并在独立venv/bin添加编译器链接，未改系统。
- 第三次`round2/ar-tp1-v3`通过：2/2题各64 tokens，driver退出0，decode日志cuda graph=True；运行后server已清理。仍为同一af7候选，环境修复不计源码轮次。
- 启动`round2/tp2-bs1`三算法短测，TP2、capacity40000、context65536、单proposal、每题64输出；GPU0/1启动前均空闲。

- `round2/tp2-bs1`：AR两题完成，图开启；相对TP1输出在0-based index2/index20首分歧，停止一致。不能称逐token等价。DFlash启动因FlashInfer采样JIT使用CUDA12.8头文件缺少cuda::maximum/minimum而失败，driver退出1；LoopSpec未运行。另发现Custom AllReduce IPC JIT编译失败回退，需要补齐工具链后复核。
- 任务环境安装CUDA13.0.88 nvcc/runtime/crt/nvvm及CCCL13.0.85（X5 nvcc同版本）到environment/cuda13，CUDA_HOME与PATH/LD_LIBRARY_PATH仅本session设置。`round2/tp2-bs1-cuda13`重新从AR起测，源码仍af7，同候选2。

- CUDA13工具链重试`round2/tp2-bs1-cuda13`完成，三方法各2×64输出，driver退出0。三者均图开启；Custom AllReduce不再编译失败。LoopSpec所有warmup/实测请求两rank计数严格相等，D3 accepted/proposed均0。DF与AR两题均分歧(index2/20)，LS与AR一题分歧(index2)；短测统计保存在results/tp2-bs1-smoke.json，不当作full30性能。
- `round2/tp2-bs16-cuda13`短测进行中：每方法16并发warmup，再2题64输出；后者仅2并发，不能冒充16题正确性矩阵。全30正式cohort将验证真实16并发及尾批14。

- bs16短测三方法均完成，driver退出0。正式矩阵开始：A12 GPU0/1、SHA af7eb8ee264ed88ca45752c53d69a35077acd08b，matrix/bs{1,2,4,8,16}，每格三方法全30题、max_new_tokens1024、context65536、KV40000、TP2、greedy、R4、BF16、单D1proposal。循环遇失败即停止，日志matrix-driver.log及每bs-driver.log。当前仅启动，不标完成。

### 正式矩阵进展（未完成）

- A12持久session11376内启动串行矩阵wrapper PID3234857，依次bs1/2/4/8/16，各ar/dflash/loopspec1；每个子进程失败即停止后续。所有输出在`/homes/ruoxi07/experiments/loopspec/tp2-20261106/matrix`，总日志同级matrix-driver.log。
- bs1 AR于2026-11-05 18:01 UTC左右完成30/30，30,720输出tokens、0 retractions，cohort吞吐28.2892939433 tokens/s，decode加权35.3131600729 ms/token。DFlash仍运行，其他cell未完成；不能用短测填补正式结果。
- 最终任务工具链CUDA_HOME=environment/cuda13/nvidia/cu13，PATH加入其bin，LD_LIBRARY_PATH加入其lib；lib64为lib的任务内链接。其余Python/compat/Rust变量沿用上述独立环境。

- bs1三方法全30题完成，bs=1 driver退出0，均30,720输出tokens/0重调度。DF吞吐46.4074595947 tok/s、相对AR1.6404601574，accepted/proposed0.08203155246、accepted/tested0.55166465975、accepted/verify1.23047328688、摊销decode47.8884962179 ms/verify。LS吞吐61.6486533280 tok/s、相对AR2.1792220566，accepted/proposed0.67598302827、accepted/tested0.89289670903、1.32404692082新core/decode-token、decode16.1574158198 ms/token。多batch与详细Profile尚待完成。
- 统计注意：LoopSpec scheduler按拆分后的单请求batch打印#running-req，不等于全局在途请求数；应另从客户端sent/end事件计算并发，不能把此日志peak=1解释为bs16未并发。CLI并发设置不证明每一步固定batch；正式cohort尾批按30题余数执行。

- bs1完整性：AR自身IDs30/30，DF/LS对AR IDs均0/30，stop均30/30；实测cohort日志cuda graph=False均0。既有数值不等价仍未解决，不能声称无损加速。
- bs2全三方法30/30完成、driver退出0、均0重调度：AR51.5438878498 / DF84.0120112117 / LS98.4640328989 tok/s；DF1.62991219165×、LS1.91029503218×。
- bs4全三方法30/30完成、driver退出0、均0重调度：AR90.8164925401 / DF119.0472192982 / LS149.3006673569 tok/s；DF1.31085462528×、LS1.64398187136×。
- bs8 AR/DF均30/30及0重调度，吞吐160.9822952303 /184.5407309462 tok/s，DF1.14634178052×；LS进行中。DF decode日志观测峰值7且有排队，不能把客户端并发8当作每步固定8。不在基准中途修改调度参数；报告需保留实际排队、尾批和cohort等待。
- summarize_matrix.py用analysis-check/bs1（指向已完成正式bs1）执行退出0；可从全30原始输出重算吞吐/接受率/图覆盖。analysis-check并非完整15格结果。

## 候选3/10：正式数据后处理修正

- 候选2推理矩阵全部完成：bs1/2/4/8/16各三方法全30题，五个driver退出0。详细Profile已在同GPU0/1启动，PID3336816，输出profile-bs1，运行源码af7不变。
- 失败证据：summarize_matrix.py glob(bs*)误匹配同目录bs1-driver.log，报NotADirectoryError，退出1。正式请求数据不受影响。
- 六项计划：1目的：可靠汇总正式矩阵并区分客户端并发与scheduler日志batch；2步骤：只遍历目录，按sent/end事件统计客户端峰值及平均在途请求；3范围：summarize_matrix.py及本记录/结果，不改推理运行时；4标准：完整读15单元450请求，bs1已确认数值保持一致，客户端峰值等于设置并发；5风险：客户端在途含服务器排队，不能冒充GPU固定batch，平均值分母须使用cohort墙钟；6验证：本地读取本次正式原始数据、assert15格/450题/图与重调度计数、静态检查。Profile期间不pull更改运行目录，结束后再同步后处理代码。
- 机器状态补充：Profile启动前GPU0/1均0MiB空闲，GPU2–7已出现其他用户任务；未触碰。A12为共享主机，最终报告应记录并发环境，不能声称整机独占。每格manifest保存启动时nvidia-smi快照。

- 候选3本地后处理验证退出0：15格450请求/460800输出tokens，全部length=1024，全部0 retraction、实测日志0 graph=False，客户端峰值=设置bs；bs1既有吞吐比逐值一致。五个batch下LS两rank各30请求计数一致，D3 accepted/proposed均0。结果见results/matrix-summary.{json,csv}及matrix-validation.json。AST/diff检查通过。
- 修正日志解读：AR大batch日志峰值也会少于客户端总在途请求，LS按拆分视图记录；7/14等数值只能称scheduler日志观测值，不能据此断言GPU固定batch或硬性上限。客户端在途含排队，亦不是GPU固定batch。本矩阵定义为cohort并发测试。
