# VQ+ INT4 残差 attention 与 SGLang 接入计划

状态：仅设计，未修改推理源码、未启动新算子实验。用户要求停止并删除上一轮准确率评测，然后制定本计划。历史残差训练/独立评测研究已用10/10轮，保持原计数；本计划不借改名继续旧任务的第11轮。后续实施预算在实施授权时明确。

更新：用户已批准本计划并要求每轮验证后重新审视性能瓶颈。开始新的SGLang VQ+算子实现，按项目默认最多10轮执行；旧的独立准确率任务仍为已取消10/10，不重启。本实现的完整交付范围为上文批准的缓存、编码、融合attention、生命周期及真实serve配对验证，不以局部kernel通过代替完成。

### 候选1/10

结果：c6ba757f63953cede2ca254ba8bd79c74c47bc60，A10 GPU4，loopvq-build Torch2.8.0+cu128/Triton3.4.0，check_residual.py退出0；远端r1.json/r1.log/r1.exit在/homes/ruoxi07/experiments/loopvq/2026-11-26-vqplus。codec逐bit通过，attention至32K最大绝对差0.000244140625。图内单层8K耗时batch1=1.221871ms，batch16=9.294439ms，性能不通过。只读profiler：主stage1=1.217ms/次，占99.66%；reduce=4.181us/次；Triton metadata n_regs255、n_spills0、shared16896B。不能声称发生spill；优先消除重复K重建和全局码本gather。完整serve/生命周期尚未验证。服务环境加载既有compat库后Torch2.14.1+cu130可正常初始化CUDA，未修改驱动。

### 候选2/10

目的：降低已测得的主stage1开销。步骤：新增shared-codebook CUDA stage1，K中心+残差+RoPE+BF16舍入后QK；FP32softmax；V中心+残差+BF16舍入后PV；HP使用同一数值契约，全部split覆盖LSE；保留Triton实现用于诊断。范围为residual_attention_cuda.py、residual_attention.py及同目录check_residual.py。预期shared表代替重复global gather、每lane只保存4/16维数值以降低寄存器；成功标准为原容差不变并显著低于R1图内耗时。风险为shared占用、softmax归约、非连续query/HP索引。验证同A10GPU4，同一独立oracle含32K及空行、graphpadding，图内不同B/长度与R1比较；JIT在capture前。CUDA延迟收益不替代完整模型验收。

目的：建立可测的packed残差存储、group32编码和直接融合split-KV路径。范围：新增residual_encode.py/residual_attention.py，修改pool、pool_configurator、KV构造、参数hook和attention分派，增加独立GPU oracle/计时脚本。采用Triton融合HP/quant的stage1加FP32 LSE reducer作为首个可审计实现，不预先宣称其快于原CUDA VQ。风险为舍入、掩码、四loop地址、编码额外launch及寄存器压力。成功标准为pack/scale与独立oracle逐bit一致，attention在预先固定atol=0.002/rtol=0.01下匹配显式FP32 softmax重建参考，空序列和污染scratch也正确；量化新增开销并以证据规划下一轮。测试包括prefill/flush编解码、mixed HP/quant随机物理页、CUDA graph重放和基本延迟。A10 GPU4/5首选，持久PTY10517经X5；/repo/sglang为其他任务ddtree，另建隔离/repo/sglang-vqplus，经loopspec-test同步。首轮通过后继续真实模型及生命周期验证，不宣布整个目标完成。

代码基点：Rancy-Wang/sglang，loopspec-test，80549ace9271dd55ef20baab5072f735625bd33b。LoopVQ训练参考为test分支dc3af341ba4593f580272470da803a5db839a423。工作区原有未跟踪snapshots目录保留。正式分支不变。

清理：已停止A10/A12/A13七个cell_worker及其所属server/client/子进程，关闭评测自动跟进loopvq；删除本轮远端jobs/replicas/workers/dispatch、准确率输出/日志/表格及本地对应结果，删除24个W&B评测run。group中剩余两个run均为训练run dcbebws8、9dvetlbu，状态finished，训练日志和artifacts保留。取消/删除操作清单保留于原任务根目录cancel-*.json；不清除其他研究的历史实验。

## 1. 修改目的与边界

定义VQ+：原LoopVQ联合四个loop的码本索引，加每个loop独立的对称INT4残差、group32 FP16 scale。首版固定Ouro-2.6B-Thinking、四个loop、TP2、HP8+64、page8；不改码本训练、KL目标、模型深度、RoPE或MLP策略。目标是能直接读取压缩表示的attention，消除整段BF16历史KV还原及通用SDPA路径。先以当前optimized SGLang路径为共同控制变量，再评估新增残差。

不宣称最优性能已经成立。需要在吞吐、误差和显存之间实测选择；原始LoopVQ、BF16和VQ+均同机同卡对比较。首版不叠加LoopSpec，不扩大到prefix reuse/chunked prefill/offload等当前拒绝的功能。

## 2. 数学与存储契约

对每个真实token、base-layer、head，原码本索引覆盖4个channel×4个loop；每个loop从该索引对应的中心取自己的4个channel。残差不跨loop共享。

训练兼容的中心解码定义为b=float32(bfloat16(exported_fp16_centroid))。对每个loop、head、token的32个连续channel：e=x-b；s=max(fp16(max(abs(e))/7),fp16.tiny)；z=clamp(round_to_nearest_even(e/float32(s)),-7,7)。必须使用已舍入到FP16的s计算z，不能使用未舍入scale。两条有符号4bit码打包进一个uint8，-8编码不使用。

K残差定义在pre-RoPE空间，V残差在原V空间。训练兼容attention参考值为：

- k_served = BF16(RoPE_position(b_k+s_k*z_k))。
- v_served = BF16(b_v+s_v*z_v)。
- logits = dot(q_rope,k_served)/sqrt(128)。
- 对HP和VQ+的全部token做同一个softmax，再求加权V。

上述BF16舍入发生在寄存器内，不需要写回全量KV。若把RoPE移到query侧、或把V累计拆成p*b和p*r，忽略中间BF16舍入会改变数值契约；只能作为单独验证的优化候选，不能默认为与训练逐bit等价。现有optimized原始VQ从FP16中心直接计算，不能直接复用这个细节冒充训练兼容残差路径。

残差模式可在加载时一次性将decode码本转成BF16，维持16bit表大小，避免每次lookup重复FP16→BF16转换；encoder仍读取其约定的原始中心。此处是匹配已有训练解码语义，不是另加码本低比特量化实验。保留原始VQ的FP16 decode表路径作为独立对照，不能全局替换其语义。

另外两项已有差异必须先量化：训练assign用torch.cdist，当前Triton encoder用FP16输入dot-minus-norm；训练从post-RoPE BF16逆变换取得K，SGLang保留原始pre-RoPE K。独立oracle需接收同一输入，验证索引、tie和重建；区分输入定义差异与kernel错误。原码本不因部署布局被重训或覆盖。

建议初始布局（H为TP本地head数，Q为quant物理slot数）：

| 数据 | Shape | dtype |
| --- | --- | --- |
| K/V索引，各一份 | [48,Q,H,32] | uint8，四个loop共享 |
| K/V残差，各一份 | [192,Q,H,64] | uint8，每byte两个INT4 |
| K/V scale，各一份 | [192,Q,H,4] | FP16 |
| HP K/V | 延续现有每loop独立arena | BF16 |

残差sidecar与原索引采用同一个physical quant slot，复用页表，不增加第二套token映射；不同loop的sidecar不能alias。优先让同一head内channel连续，支持32/64/128bit向量加载；是否改为page内tile布局由访问合并实测决定。

持久量化区的平均bit/原始元素：0.5+4+16/32=5。D128下，单kind/单有效层/head平均80B（共享索引摊销8B、残差64B、scale8B），BF16为256B。TP2每rank每token合计：原VQ24KiB、VQ+240KiB、BF16768KiB。这里不含固定码本、HP区、页对齐和scratch；对应3.2倍BF16存储压缩、原VQ的10倍容量开销。共享索引会被不同loop重复读取，因此持久容量比例不等于attention带宽比例，更不是速度承诺。

## 3. 修改步骤：写入、attention、调度

### A. 先建立独立数值oracle

用训练codec和保存的exported book固定参考。覆盖zero residual、全部零group、极小scale、FP16 scale上溢、rounding ties、负数pack/unpack及四loop交错布局。codec/packed bytes/scale应逐bit一致；中心选择若因dot与cdist接近边界而不同，保留最早分歧并决定显式数值协议，不静默略过。

### B. 编码与残差记录

prefill：沿用新请求完整prefill路径。只有某base-layer的四个loop K/V全部就绪后，才对量化区写联合索引和四份残差；prefix/recent仍写真实BF16。不把depth3的残差复用于depth0..2。

decode：当前token先写HP ring。旧完整8-token page离开HP窗口时，从HP读取四个loop的真实K/V，生成联合索引、残差和scale；全部完成后再发布req_to_token映射。不要每生成一token重复编码全部历史。

现有encoder每CTA只处理4channel×4loop，但group32残差scale跨8个这样的VQ group；不能在原CTA直接算完整scale。首选可实现结构是：

1. 联合索引pass，保持最近中心搜索，K/V可用额外grid维合并提交。
2. 残差pass，以(token,head,group32,base-layer,kind)为工作单元，读取8个code以及真实4loop×32channel值，逐loop归约max、生成FP16 scale、RNE量化、pack并写sidecar；不生成全历史residual float临时tensor。

先允许明确的两pass实现，测清编码耗时。进一步评估把八个VQ group放进同CTA的一pass方案，只有不造成寄存器/shared-memory占用恶化时才采用。K/V合并grid不保证执行更快，必须与分开launch对比。

### C. VQ+ fused attention

新算子直接从packed残差、scale、VQ索引读取数据；不在global memory写出解量化K/V。以现有CUDA vqg_stage1的三个phase为结构参考：

1. QK：加载同一token的K中心与INT4残差，在寄存器中相加；融合RoPE及所需BF16舍入，再计算完整logit。可就地覆盖中心寄存器以控制存活变量。
2. Softmax：基于已经包含K残差的logit做FP32稳定softmax/LSE。绝不能分别计算base-attention与residual-attention后相加。
3. PV：读取V中心和packed V残差，融合scale乘法、相加及BF16舍入，再使用同一softmax权重累计。V残差不是输出端另加一个向量。

长期路径保留split-KV维度，使batch1×8heads也能形成足够CTA。残差不是新的softmax tier；只有HP和量化区两个数据tier。HP输出与VQ+输出仍用统一LSE合并，不能对两个独立归一化输出取平均。

首个正确性版本沿用现有quant-stage1 + HP-stage1 + reduce。优化版本评估：

- 合并HP/VQ+的stage1提交，block-uniform路径分别处理HP或VQ+ split，随后一个reduce；避免为每层额外运行残差kernel。
- reducer直接使用本次有效split mask/长度，或保证producer逐项覆写全部LSE槽位，从而移除lse.fill_。必须覆盖空量化区、短batch、图padding和复用后旧值；未经证明不删reset。
- 很短上下文评估单CTA完成HP+VQ+并直接输出，省去reduce；batch1可能并行度不足，必须与多split比较。
- 计算真实有效长度，使用tile内mask；不让8K历史按16K完整attention计算。

调参维度先小规模：tokens/split 128/256/512/1024，CTA128/256/512 threads，R1/R2请求分组，UA/UC unroll，HP splits1/3/6/12。根据B、H、有效长度、寄存器/共享内存和GPU SM数选型，不能固定沿用原始VQ的CAP2048、512线程、HP12、R2。scale在group32内重用；不把完整残差搬进shared memory。双缓冲预取只在占用率与stall数据支持时引入。

INT4在这里首先是存储格式，并不自动使attention适合INT4 Tensor Core：query/softmax权重仍为浮点，残差scale还随token/group变化。首版用向量load、寄存器unpack和浮点FMA；Tensor Core变体必须有适合的tile复用和单独收益证据。

### D. SGLang缓存生命周期与图

pool、allocator、pool_configurator同时改：token容量必须计入residual和scale，get_kv_size_bytes/日志也必须返回真实占用，避免仍按0.5bit分配导致OOM。共享索引与所有loop sidecar作为一个页单位分配、释放、复用。

保留既有scheduler-stream/forward-stream的事件依赖：上一forward读完→encode/index+residual+scale→publish映射→下一forward可读。禁止先更新页表再异步写残差；禁止HP ring在旧值未读完时覆盖。失败和请求abort不得留下仍可见的半写页。首版继续拒绝prefix cache复制、KV迁移、offload和speculative组合。

所有sidecar地址、partial/LSE、metadata、flush descriptor和dummy页在capture前分配；init/JIT/SM查询不放在decode热路径。主forward采用已有full CUDA graph。周期性flush先使用持久plan buffer和独立小图，避免新增逐层Python提交；是否把带mask的flush放进每步forward图由实测决定，不能为节省一次CPU判断而每步重复量化。

dummy page的残差为0，scale为合法有限值，valid长度为0；padding不得写入真实请求页。动态batch缩放只更新固定设备buffer，不做GPU .item()/.cpu()读回以决定split。当前层0更新的metadata供其余191层复用。

## 4. 具体修改范围

以下路径相对sglang/python/sglang/；新文件仅为计划名。

| 文件/模块 | 修改 |
| --- | --- |
| srt/arg_groups/fields/model.py、kv_cache_hook.py | residual bits/group/format及兼容性检查；原VQ默认路径不变 |
| srt/mem_cache/loopvq_codebook.py | 校验残差schema与中心解码dtype；固定codebook hash；不修改原权重 |
| srt/mem_cache/loopvq_pool.py | packed K/V residual与scale arena、每loop视图、写入接口和容量统计 |
| srt/model_executor/pool_configurator.py | 5bit有效容量、固定HP/码本/scratch预算 |
| srt/mem_cache/loopvq_allocator.py | 完整页flush事务和sidecar共同释放/重用 |
| kernels/ops/loopvq/encode.py + residual_encode.py（新） | joint-index与group32 residual pack pass |
| kernels/ops/loopvq/residual_attention_cuda.py（新） | QK残差+RoPE、softmax、PV残差融合kernel |
| kernels/ops/loopvq/official_attention.py、official_metadata.py、official_reduce.py | 显式残差分派、有效split、HP融合与LSE处理；保留原路径作为对照 |
| srt/layers/attention/loopvq_backend.py、triton_backend.py | 传入residual/scale视图和配置，支持TP2本地heads |
| srt/model_executor/runner/decode_cuda_graph_runner.py | 固定buffer、padding和graph replay验证 |
| srt/models/ouro.py | 原始K位置与4loop时序核对；仅在接口确需时改，不另改MLP/RoPE数值策略 |
| research/loopspec/experiments/2026-11-26-vqplus-residual/ | 独立oracle、生命周期测试、微基准与profile脚本、每轮记录 |

## 5. 预期效果、风险与成功标准

确定的结构收益：无完整BF16量化区缓存；没有decode每层单独的residual解码launch；TP2仅本地head处理；group32每32channel一个scale；真实历史长度驱动split。VQ+相比原VQ会增加大量访存、unpack和FMA，不能承诺几乎零代价，尤其长上下文。相比BF16是否更快取决于上述融合和并行度，短上下文可能仍慢。性能目标是得到经同机配对验证的Pareto结果，而非把理论压缩比当加速比。

正确性门槛：codec bytes/scale逐bit匹配独立oracle；关闭残差必须回到原VQ；零残差的新增路径不能制造额外修正；每loop身份、TP head切片与物理页映射严格正确。attention与显式重建oracle分别比较logit/LSE/output；在调优前固定容差并报告max/RMS误差，不能用调宽容差消除失败。完整模型检查同profile基线与新实现，区分浮点归约差异、量化误差和代码错误。

重点风险：训练BF16中心/FP16导出与服务FP16中心差异；post/pre-RoPE输入差异；INT4符号扩展/scale顺序；group32跨CTA归约；所有有效split未覆写引起旧LSE污染；压缩页跨stream发布；batch1占用率与R2寄存器压力；5bit比0.5bit容量大10倍；prefill四loop stash的暂态峰值。

## 6. 验证与交付顺序

实施后按SGLang loopspec-test本地commit/push、远端pull同步；核实运行路径和SHA。长任务tmux；先检查A10>A11>A12>A13空闲GPU，不干扰其他作业。默认A100 TP2，同机同卡对配对，保持R4、HP8+64及同一模型/采样设置。

1. Codec/布局：CPU独立oracle与CUDA pack/unpack/encode；涵盖K/V、四loop、TP2、极值、全零group、page边界和随机物理页。
2. Kernel：batch1/2/4/8/16，真实历史64/72/79/80/127/128/1024/2048/4096/8192/16384/32768，混合长度、空quant、尾page、非连续位置；比较显式重建oracle，检查mask与LSE。
3. 生命周期：连续多次flush、batch进出、abort、页回收复用、graph padding、eager/graph一致性及请求相互污染。无GPU主机读回、无隐藏fallback；用sanitizer定位越界/竞态。
4. 微基准：BF16、原VQ、VQ+三者同硬件，额外加“显式解量化VQ+”仅作诊断对照；量化区与HP各自计时，flush单列且报告摊销和尖峰，CUDA event计时与Nsight详细profile分开。
5. 完整forward/decode：warmup后多次交错配对，报告wall-time、GPU kernel总时、CPU提交/等待、QK/softmax/PV/HP/reduce、metadata、flush、TP通信、其余相同步骤。测试CUDA graph开启；bs1/2/4/8/16和1K/2K/4K/8K历史，长上下文补16K/32K。若做长输出，则固定2K输入、输出1K/2K/4K/8K、ignore EOS，仅主报纯decode并单列prefill。
6. 不恢复已取消的五benchmark完整评测。先交付：可复核kernel正确性报告、真实显存账单、各bs/长度的decode收益与退化区间、完整step分解和剩余瓶颈。只有用户随后要求时，另行重启准确率评测。

建议实施顺序：协议与oracle→packed pool与flush→直接融合VQ+ attention→图与split/HP调优→真实serve配对验收。每个候选修改+验证按项目规则记录轮次；不能在同一次改动里同时更换训练方法、算子数值协议和调度策略而失去归因。

候选2结果：f7dbe9c4a，A10 GPU4，服务环境Torch2.14.1+cu130/Triton3.8.0，r2.exit=0，独立oracle至32K最大差1.90735e-6，dynamic graph/padding通过。8K单层B1=0.125713ms、B16=1.02656ms；1KB1=0.034475ms、B16=0.242722ms。cuobjdump显示62寄存器、STACK/LOCAL=0，显著低于首轮255；仍有固定32split冗余码本装载和低occupancy疑点。R1不同环境的加速比仅暂估，同服务环境复测另记。r2-wiring.exit=1：旧mock配置未含residual_bits；8项其余测试通过。NCU拒绝GPU计数器(ERR_NVGPUCTRPERM)，不修改系统权限；NCU下产生的延迟不能用作性能数据。

### 候选3/10

目的：改善split调度及memory-latency隐藏，修复默认关闭残差时的兼容回归。范围：CUDA stage1删除shared loc缓存、直接读取只读页表，提供128/256/512线程编译变体；按B/H/长度在设备端选择有效split(上限32、每split<=1024)，空split仍覆盖；ResidualAttention初始化配置，容量配置getattr默认0及显式32K入口校验；测试增加可选线程与原固定split对照、CPU容量测试。预期大B短上下文显著减少重复码本加载，512线程可借减少shared loc存储保持两CTA/SM；成功标准仍为同oracle容差，并比较各线程/动态split的配对延迟而非凭occupancy推断。风险为低B并行度下降、寄存器/共享内存变化。验证A10GPU4逐一运行完整kernel/graph套件，各候选GPU计时不重叠，CPU回归；随后用选定参数做真实TP2服务验证。数值合同和训练不变。

候选3结果：2a430fda8，A10 GPU4，128/256/512线程和512固定32split全部通过codec/output/LSE/动态graph检查（各exit0），512动态在8KB1=0.0712ms、B16=0.7742ms，128/256显著更慢；固定32split在B2的8K略快，但大B短长度显著落后，尚未证明动态策略全局最优。旧wiring九测试通过；残差CPU contract入口和TP2容量账单通过。真实TP2在GPU4/5、端口30664，训练residual-ft，HP8+64，context32768，capacity65536，max-prefill16384，full graph bs1/2/4/8/16启动成功。128token连续decode、pad3、bs16、中止及页复用后相同batch1的128token逐ID一致。每rank固定1.478GiB+245760B/quant token；65536容量K/V各8.24GiB（日志显示GB）。r3-serve.log、r3-client.log/exit、r3-lifecycle.json均保留远端新任务根。

R1同服务环境复测r1-serving-env.exit=0，8KB1=0.262724ms、B16=2.140877ms；与R2代码配对收益约2.1x，不能使用跨Triton版本的9.7x宣传代码收益。

R3真实TP2 trace：r3-native-bs1，2K输入、1088输出，约输出928处采10个decode step。CPU/GPU profiling会扰动时间，不用该请求43.78tokens/s作最终吞吐。TP0图span均值22.6493ms，attention+reduce6.1873ms；TP1图span22.3031ms，attention6.1963ms；正常图间隔约0.08ms、包含flush的一次约0.35ms。10step内残差编码2次合计0.0813ms，原joint-index编码2次合计0.174ms。cudaEventSynchronize的CPU等待与GPU执行重叠，不能相加成额外149ms。未观测到大幅CPU提交空洞，后续优先GPU attention。原始trace和校正后按cudaGraphLaunch correlation配对的摘要已同步到本地reports/vqplus-20261126/r3-native-bs1。

### 候选4/10（计划，待R3无profile基线结束后验证）

目的：针对主attention的访存等待与指令数优化，补真实服务配对测试工具。步骤：CUDA QK/PV增加可选unroll(1/2/4)，仅在已经完成BF16舍入后的dot/PV累积使用显式FMA，RoPE和残差重建舍入不变；以128/256/512线程资源和延迟选择，而非假定展开越大越好。微基准使用一个graph内多次算子以消除Python连续replay供给不足，单独保留原计时口径；增加真实page8布局计时，与逐token随机最坏布局区分。范围residual_attention_cuda.py、residual_attention.py、check_residual.py和独立服务验证工具。风险为register增加导致occupancy下降/spill、FP32归约微差、短上下文退化。成功标准为原pack逐bit和output/LSE容差全部不变、graph/lifecycle通过，并有同卡配对改善；失败候选留证据，不自动选用。真实serve全R4 TP2+full graph，BF16/原VQ/VQ+同卡对，时间与profile分开，不恢复准确率评测。

R4资源安排：A10空闲卡对4/5正运行R3同卡BF16/原VQ/VQ+完整服务对照；其余六卡为他人任务。A11八卡均有2.1–2.4GiB任务，跳过。A12八卡现场空闲，使用GPU0做R3/R4 kernel配对；通过同一X5持久会话PTY82582访问。为不改变A10正在运行的R3源码，另建sglang-vqplus-kernel clone，仍仅从用户fork的loopspec-test拉取。新测试提供显式--kernel-root，可用同一计时脚本读取旧R3 checkout；不混用跨机加速比。

R4验证前补充计时控制：R1–R3的合成微基准不同request会随机复用相同physical KV，不能当作真实独立请求的带宽结论。R4计时为每request分配独立quant区域及HP区域，支持page8和随机token两种布局；每B重新分配足够的单层arena，直接生成合法INT4码避免测试初始化物化大量FP32数据。数值oracle仍使用旧随机fixture以保持回归可比。R3与R4一律用这份新计时脚本在同卡重测，不能把测试口径变化算成实现加速。

R4结果：323079e01d2b3200a929708a2a5205e7f9ea3543（首个实现eac9c1ce7，GPU验证前补齐独立请求计时）。A12首次编译因/usr/bin/g++不存在失败，日志保留a12-r4/512-1.log；改用机器已有gcc-13/g++-13及独立torch-extension缓存，不修改源码或系统依赖，属于环境修复。gcc13目录的R3基线及512U1/U2/U4、256U2/U4均exit0，codec/output/LSE/graph通过。8K的B1/2/4/8/16，R3为0.0548/0.1509/0.2534/0.3960/0.7410ms；R4-512U1为0.0615/0.1469/0.2476/0.3854/0.7189ms。U2/U4未带来稳定收益，256明显更慢。小B需更充分重复估计时钟波动，不宣称所有shape改善。每次graph含16个完整attention，KV按page8且请求区域独立，不能与旧共享随机fixture直接算加速。

### 候选5/10

目的：针对大batch residual attention瓶颈，评估两个request共用码本/RoPE的R2分组。源码分析表明R1每request重复加载同位置FP32cos/sin；R2可复用系数与64KiB表，但不会共享不同请求的真实KV或softmax。范围新增residual_attention_grouped_cuda.py，ResidualAttention分派及GPU测试。每CTA256线程、每请求CAP512、64split，两个request的长度/位置/概率/输出独立；batch1保留已有R1。风险为寄存器增加、group中长度不一致、末尾奇数request和empty split；必须完整覆写LSE且不越界。预期在B>=2减少重复加载，成功标准仍为原oracle容差、奇数B和动态graph通过，并与同A12GPU0的R4按统一page8/disjoint协议比较。配置保留R1作基线；只有实测改善才选用R2。真实服务继续TP2全R4，无准确率重启。计划通过后还需极值codec/训练编码差异、长上下文与完整性能验收，当前未宣称完成。

R5结果：4432365729f7eca9277cb7b738da944bfc53e8c7，A12 GPU0，a12-r5/r4.exit与r5.exit均0，codec/oracle/奇数及不同长度batch/graph通过。8K B1/2/4/8/16，R4=0.0512/0.1470/0.2474/0.3856/0.7190ms；R5=0.0512/0.1195/0.3364/0.5175/0.9232ms。R2在B2获益、大B退化，不设默认；256线程和两个PV accumulator同时存活使并行度/资源成为下一候选。

### 候选6/10

目的：减少R2同时存活的PV累加器并提高延迟隐藏。步骤：PV按request顺序处理、复用WARPS×128 shared partial，512线程、launch_bounds(512,2)约束；保留数值合同和R1默认。范围residual_attention_grouped_cuda.py。预期比R5大B降低延迟；风险是更多barrier或编译器spill抵消收益。验证同A12 GPU0，完整codec/oracle/LSE/奇数batch/graph套件和page8独立请求图内计时，对比同卡R4/R5，检查编译资源。只有改善才采用，不因测试通过宣称全局最优。

R6结果：ade7985e8a7fc83d00409adc3e3aa3996bbcf596，A12 GPU0，a12-r6/r6.exit=0，codec/output/LSE/graph/奇数及不同长度batch通过。8K B2/4/8/16为0.0943/0.2299/0.3586/0.6743ms，相对R4分别1.559/1.076/1.075/1.066x；1K与2K的B4/8仍约慢3.5–7%。cuobjdump在A10读取同一so：64registers，STACK/LOCAL0；动态shared78992B（约77.14KiB），容许每SM两CTA。B1未改路径，计时波动不能算代码退化。

### 候选7/10

目的：采用有证据的batch分派并补齐codec边界安全。范围ResidualAttention、residual_encode.py、check_residual.py。步骤：默认按capture batch选B2及B>=16的R2，其余R1；保留强制R1/R2诊断选项。FP16 scale溢出或非有限输入在GPU编码时assert，避免悄悄写坏页，不引入CPU读回；增加round-to-even ties、tiny scale与独立子进程非法输入测试。风险是图padding后选路不同产生归约微差、device assert额外成本及poisoned CUDA context，所以非法测试隔离。验证同卡kernel套件、codec拒绝、TP2生命周期/eager对照和真实decode/profile；按性能复查新增检查是否只在prefill/flush而非每层decode执行。数值容差不变。

R7本地语法和git diff --check通过；本地系统Python缺torch，CPU contract尚未在本地执行，转已部署远端环境验证，不计为通过。

R7 kernel/codec结果：f36a4c29ea790800181a4ad0155e2c1e6674fb88，A12 GPU0完整套件exit0，新增ties/tiny逐bit匹配；nan/inf/overflow各独立进程exit0且rejected=true；18项CPU检查通过。真实TP2服务验证仍在执行，A10 GPU4/5做2K与生命周期/profile；A12 GPU2/3、4/5、6/7分别做1K、4K、8K，三方法在各自同一卡对顺序配对。A11忙跳过。每组capacity81920、128新token、3repeats、full graph，8K B16在BF16单rank所需KV超过80GiB，不运行也不外推。

### 候选8/10

目的：隐藏码本shared加载等待，继续针对已定位GPU attention瓶颈。步骤：增加可选cp.async表搬运，K加载等待后QK；QK完成全CTA同步后预取V，与独立prob/reduce区域内softmax重叠，PV前wait_group及全CTA同步。范围R1/R2 CUDA源码及其prepare配置、ResidualAttention和GPU验证参数。风险为shared重用竞态、copy提交/完成顺序、资源变化；数值合同完全不变。成功标准同oracle/LSE/graph全部通过且同A12GPU0比同步R7更快；保存强制同步对照，不预设收益。验证另建冻结worktree，不修改正在服务的R7目录；sanitizer检查async边界。

R8初测：de17e27a73775fd6a1d88d917ec0fa61c2e21e98，A12 GPU0，a12-r8/r7与r8均exit0且原oracle/graph通过。R1 B1 1K/2K/8K分别约1.13/1.087/1.084x，R1 B4/B8短长度约1.02–1.03x；R2各形状约0.99–1.01x，无有效收益。小B时钟波动仍需同进程交错复测；memcheck/racecheck已启动，未完成不标通过。

### 候选9/10

目的：收敛部署默认值并完成数值来源审计，不再扩大算法变量。范围ResidualAttention默认分派、KV兼容性入口、独立配对/训练协议诊断脚本。步骤：R1默认异步、R2同步，显式配置仍可强制两者；检查bundle声明的残差bits/group/scale/codes；同一进程同一KV交错同步/异步计时排除小B时钟偏差；用Thinking原生模板和真实HF前向采集选定层/heads四loop K/V，比较GPU assignment与训练FP32 cdist，以及raw pre-RoPE和inverse-BF16-RoPE的影响，明确不是benchmark准确率。风险为profile变体归约及HF版本接口、默认预取收益不稳定；保留R7同步路径和失败证据。验证独立CPU/GPU合同、sanitizer、真实TP2 eager/graph生命周期与配对decode；若R1异步无稳定收益，最后一轮回退默认而不夸大优化。

R9初步结果：a427732609d57dc766d19f28be611b0ad500ca1b，A12 GPU1，CPU合同19项、完整GPU oracle/codec/graph及同进程配对均exit0。交错同步/异步：R1 B1 1K约1.05x、2K约1.007x、8K约1.002x；B4 1K约1.038x，B8 1K约1.014x；R2 B16 2K约0.990x。纠正R8单进程分离运行受时钟影响的较大收益估计。真实HF审计服务env TF5因create_causal_mask签名失败，改用已有训练env Torch2.8/cu128、TF4.56.2后exit0；未改checkpoint。抽样3层×4head×256token，K/V每种98304联合索引相同输入GPU vs FP32 cdist均0差异；K输入定义差异引起549/98304(0.558%)索引变化，V为0。该审计encoder在训练环境运行，不能替代服务环境逐bit验证；服务环境的独立oracle另外已通过。原始forward capture未写入当前脚本，后续需要端到端跨环境索引审计时须单独导出输入。

### 候选10/10（最后一轮）

目的：检验大batch stage1是否受每CTA完整shared码本搬运/占用限制。步骤：新增可选global码本lookup，省掉64KiB shared表及K/V复制，以只读全局缓存加载中心；保留QK/softmax/PV数学及现有分派。默认仍shared，提供显式--loopvq-residual-codebook=global用于验证/选择，不自动覆盖现有有效路径。范围R1/R2 CUDA与分派、模型参数/兼容hook/backend接线；GPU测试新增--global-table和--correctness-only（sanitizer不重复计时）；同进程pair支持global/shared。风险为非合并global gather和L1/L2压力反而更慢。成功标准原oracle/LSE/图全部不变、同卡交错收益及真实TP2生命周期/纯decode可复现；失败则保留shared，不放宽标准。此轮后不再改源码/测试，若有剩余问题如实交付并等待新预算，不以新任务名规避。
