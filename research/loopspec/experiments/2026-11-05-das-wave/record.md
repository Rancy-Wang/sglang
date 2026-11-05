> 历史归档（2026-11-06）：DAS-Wave 已按用户要求撤除；本文描述撤除前的实现和结果，不代表当前可用算法。DFlash/LoopSpec 公共工具与修复保留。见 [撤除记录](../2026-11-06-remove-das-wave/record.md)。

# DAS-Wave 复现与 AIME25 对比任务

日期：2026-11-05（Asia/Shanghai）。

## 身份与预算

- 用户授权：将当前 loopspec-test 与 loopspec 的所有系统代码差异整理成正式提交，标题 `[update] Support LoopSpec for bs>1`；随后设计 DAS-Wave 的原生算法选项、bs=1/多 batch 验证和 AIME25 全30题四算法对比及详细 profile。
- 原分支：loopspec-test；工作树干净。
- 起始 SHA：loopspec-test `2eaa51692`；loopspec `237f4d44a`。
- 官方 LoopSpec 参考 SHA：`41aa710ed7c9ce908cffa03597be5d1df19d7b56`，只读。
- 晋升候选：1/10；新 DAS-Wave 实现预算：0/10，尚未修改算法。
- 历史 CDB/批量决策任务计数独立保留，不重置。

## 晋升修改前六项计划

1. 目的：晋升用户指定快照中的多 batch 系统代码。
2. 步骤：核对分支和差异，提取运行时文件，创建单个正式提交，按确认的 Git 关系连接实验分支。
3. 范围：差异中的全部25个 python/sglang/srt 文件（含旧 graph.py 删除）；不含 research、测试、README、JSON 等实验文件。
4. 成功标准：正式分支 runtime tree 与 2eaa51692 完全相同；只有一个指定标题的代码提交；实验分支现有内容保留。
5. 风险：漏依赖、误带实验文件、分支关系方向错误；核对清单、tree 和 parents，不重写旧提交。
6. 验证：git diff --check、全部晋升 Python 文件 AST、完整提交文件清单与 tree 相等检查、远端 refs 检查。此为已有代码晋升，不将历史 GPU 验证称作新验证。

## 晋升候选 1/10

开始前已阅读 scheduler/CDB、timeline、recurrent runner、Ouro、LoopSpec worker/pipeline/sampling/utils 与 trace 的差异及历史验证记录。
用户要求的“向上连接”正在核对：预期 squash 后正常 merge 回 loopspec-test，merge 的第一父为原实验分支，第二父为新正式提交，合并前后实验 tree 一致。

### 晋升结果

- 正式提交 `82a851f6fe12c6e17354054fd7b160594f203b66`，parent `237f4d44aee9f9751234758c86d3154421c852e0`，已push origin/loopspec。
- 正常merge `099cfd389c4c02e478d45b22c974ed658a7e874a`，parents依次为 `2eaa51692ff3f6da104a7f1b7b5cc8996dbb2439` 和 `82a851f6fe12c6e17354054fd7b160594f203b66`，已push origin/loopspec-test。
- 25个runtime文件，1501插入/688删除；全部存在Python文件AST通过；cached diff --check exit0；晋升runtime tree对原实验SHA完全相同；merge后 `git diff --exit-code 2eaa51692 HEAD` exit0。
- 原分支已恢复loopspec-test；未改写历史，未带入research/测试/README/JSON。
- 晋升1/10完成；无新GPU测试，历史数值限制详见plan.md。

### 只读研究与环境核查

- 论文v1与本地全文：§3、附录A/B及消融部分，计划见plan.md。
- X5持久SSH session=61213，host=x5、user=ruoxi、shell PID=665415，已建立并复用；未主动断开。
- X5源码初始与本地2eaa51692一致且干净；GPU快照无计算进程，未启动GPU实验。
- DFlash现有目录为训练态。CPU weights_only读取training_state.pt成功：step10000、5层、b16、mask16、feature IDs=(1,12,23,34,45)。无config/safetensors；导出与Ouro feature语义核对列入前置工作。
- 首次CPU metadata探测使用mmap失败（host torch2.0.1不支持），去掉mmap后成功；均为只读，不计入修改轮次。
- DAS-Wave runtime、测试和benchmark未实施；新任务0/10。当前先交付详细设计；异步询问的实施范围如有新回复，以用户最新指示推进。

## 实施授权与候选1/10

用户于2026-11-05明确要求根据计划实施DAS-Wave，并要求V2及高性能；允许经X5访问A10/A11/A12/A13核对DFlash。继续本地修改→loopspec-test push→X5 pull，不直接在远端修源码。

- 目的：原生V2 DAS_WAVE，先建立合批eager路径与正确KV所有权。
- 步骤：算法/CLI注册、CDB接入、独立draft prefix映射、并行carry refinement、完整R4 target verify、批量决策、native结果提交；CPU与X5 bs1验证后再验多请求。
- 范围：plan.md列明的spec_info/arg fields/hook/tokenizer validation、recurrent runner/timeline、spec_prepare_for_decode、新das_wave包及本目录测试脚本。
- 标准：固定target、no bonus on full accept、greedy/stop等价、无KV跨depth/请求污染和泄漏，真实V2接口；eager性能仅作初始诊断，不冒充最终优化完成。
- 风险：V2发布顺序、提前EOS、临时slot回收、并行位置的causal索引、FP16/BF16 shape差异；独立状态机与同前缀对照定位。
- 验证：AST/diff/CPU契约；X5隔离容器同SHA，GPU/端口核对，AR与DAS同Thinking输入、smoke/生命周期/多batch，再进行graph候选。
- 已用轮次：开始1/10，未验证。
- 原SSH句柄61213已由工具确认不存在；建立新长连接92524，x5/ruoxi，shell691889，ControlPath=/tmp/loopspec-das-%r-%h-%p；后续复用。
- X5可直接读取用户指定脚本及训练JSON/YAML，尚未访问A10–A13，也未运行训练脚本。

### 候选1结果与候选2/10

- SHA1df94409c412ec4c192e962fc11a2d78891c51d0，本地AST/diff通过，5项CPU契约通过。
- X5同SHA，新容器loopspec-das-wave-20261105，GPU0；显式PYTHONPATH=/workspace/sglang/python后确认导入当前源码。
- 首次测试退出1：容器Git safe.directory缺失，已仅在新隔离容器配置。重试退出1：AutoTokenizer.apply_chat_template返回BatchEncoding无法JSON序列化；未启动GPU模型，不视为算法失败或通过。
- 候选2修改范围：verify_server.py显式return_dict=False，保留原日志。目的/标准/风险/验证沿用第一轮，同SHA重跑bs1和后续矩阵；启动2/10。

### 候选2结果与候选3/10

- SHA5e55a7bd1，GPU AR启动被缺失Rust扩展阻塞；隔离容器设置SGLANG_RUST_BUILD_MODE=auto后编译成功（环境修复，未改源码）。
- 同候选AR和DAS均完成真实32-token Thinking请求；DAS V2日志cycles16/proposed74/tested31/accepted16/emitted32，尚非完整正确性验收。
- 测试因立即GET flush收到400失败；原生flush支持timeout等待，旧项目生命周期脚本已有同用法。保留round-2全部日志。
- 候选3：测试脚本改用POST /flush_cache?timeout=15，显式--return-output-ids，逐cohort保存partial以保留失败证据；Rust自动编译环境写入复现命令。runtime代码不变，继续bs1后多batch，严格token IDs/stop验证；启动3/10。
- A10经现有X5 ControlMaster转发访问，会话97155、账号ruoxi07、shell3448157，未启动A10 GPU任务。训练fallback transformers.py SHA256=1dafbdcf828a62c1d60e5ba1760187e927c7e03aec00920b0c023ed4b81aaf5c。
- 已核实capture hook按物理layer ID覆盖存储，最终保留最后循环的layer输出，循环末norm之前。配置SHA74cb...与训练元数据相同，X5 target config/tokenizer_config SHA与训练provenance相同。
- SpecForge参考SHA af1c3f3868e0ad1d1bc4a3bc4c577f2c9d3fa7b5；CPU materialize_draft严格加载58/58 tensors通过，全BF16；fc shape2048×10240。未导出、未运行DFlash推理。

### 候选3结果与候选4/10

- SHA41d6e408d，round-3/bs1.exit=1；AR完成64 token请求、flush成功，但HTTP客户端将空成功body当作JSON解析失败。不是算法验收通过。
- 第4轮范围仅verify_server.py：允许空HTTP body，比较finish_reason，并在token/text/stop不等价时assert失败，避免假阳性exit0。
- 目的/步骤：恢复配对验收，先bs1、再bs1/2/4/8/16及overlap；运行时不变。成功标准、风险与环境沿用实施计划，保持严格token ID等价；启动4/10。

### 候选4结果与候选5/10六项计划

- SHAed497e478，round-4/bs1.exit=1；flush返回的是纯文本200，不是空body。上一轮判断不完整；HTTP接口源码已核对，不把AR完成请求误记为配对通过。
- 目的：恢复可信验收，减少Wave逐层CPU launch开销，并将已核对训练语义的DFlash接入真正推理。
- 步骤：按端点区分文本响应并做本地HTTP契约测试；Wave draft更新+readout、独立R4 verify+readout捕获；Ouro最终循环物理层特征捕获；SpecForge严格导出和逐tensor核对。
- 范围：verify_server.py/test_http.py、das_wave/graph.py/worker.py、models/ouro.py、新export_dflash_checkpoint.py；不改原模型或训练文件。
- 标准：eager/graph分别与AR核对IDs/stop，graph实际命中；多请求无KV告警；导出58项逐元素相等且metadata/hash可追溯；feature捕获保持target logits。
- 风险：共享graph pool输出覆盖、padding污染、导出配置不兼容、feature归一化位置错误；独立clone输出、discard slot和严格权重核对，保留全部失败。
- 验证：CPU契约/AST/diff，本地push→X5同SHA；先eager bs1后graph及overlap/多batch；DFlash CPU导出后GPU启动与配对。同一候选测多矩阵不重计，后续源码修改再进入6/10。启动5/10。

### 候选5已取得证据与候选6/10计划

- SHAfdab7708798527136b03012e53ad78f8a9d805fe；CPU5项算法+1项HTTP契约通过，AST11文件通过。
- X5 round-5/bs1：两次64-token eager请求IDs/text/stop全部等于AR，exit0。
- multi-eager --batches 1 2 4 8 16 --tokens64 --overlap：62请求全部完成，24严格匹配，38不匹配，exit1。bs1/2通过；bs4开始有分歧，不能称多batch正确性通过。
- bs1-graph --tokens128 --overlap：图桶1..5实际命中，每次主请求62cycles，310次replay；与AR序列有差异，exit1。eager128对照及DFlash启动测试仍在收尾。
- DFlash严格导出exit0，58/58 BF16 tensors逐元素一致；新artifact /results/dflash-inference-step10000；原checkpoint未变。A10和X5 target model.safetensors SHA256均c506a79247dc51fc0400d789365c3d43932f718abce9810f3606ace47d0a3080。
- 新隔离容器loopspec-dflash-audit-20261105使用GPU1、同镜像/源码、port30216，仅正确性测试；不得把双GPU并行诊断时长用于性能结论。
- AIME旧文件30个答案与MathArena对应，但存在LaTeX控制字符错误和题面/图示表示差异；source-check保留，正式实验需冻结公开原始题面并记录版本，不能默默修旧文件。
- 第6轮目的：定位严格差分失败，独立验证KV与feature语义。步骤：同一已物化prefix下串行R4与并行verify比较logits/KV，graph同shape比较，DAS draft写隔离检查；对照训练HF最终循环feature。
- 范围：新增probe_wave.py、verify_dflash_features.py，生产运行时暂不修改。
- 成功标准：graph/eager同shape输出、prefix写隔离、feature最终物理层语义通过；逐token数值翻转如实记录，不放宽原严格门槛。风险：诊断重建prefix与原线上KV轨迹不同，必须明确它是隔离实验，不能替代线上失败。
- 验证：AST/CPU后Git同步同SHA，单独GPU运行探针；记录差异/argmax margin和每层KV误差，保留原候选日志。启动6/10（第5轮已发出的作业继续用原SHA完成后才pull）。
- 候选5收尾：DFLASH native V2/eager/overlap两次64tokens严格等于AR，exit0；bs1-eager128仍与AR不一致，exit1。AR自身graph/eager128也不一致，Wave亦不一致；graph中首次AR/Wave差异在0-based token104。这提示shape数值问题，但尚非根因证明。

### 候选6结果与候选7/10六项计划

- SHAde497046da3f4fe7f9080dc76a94daa4daebe04a；wave-probe完成24组(bs1/4 × width1/5/16 ×4prompts)。同shape graph/eager logits全部逐位一致，width1单行与串行logits/KV逐位一致，所检查成熟prefix KV未被draft改写。
- 跨shape logits最大绝对差0.5，1处argmax翻转；exit1保留严格失败。此为重新prefill同一prefix的隔离诊断，不替代线上轨迹。
- dflash-features完成native捕获与hook一致、capture on/off logits一致的断言，但HF构造失败：Transformers5.19删除未显式设置的pad_token_id默认属性。尚未保存完整HF对照report。
- A10训练环境Transformers5.12.1；target modeling_ouro.py只与X5相差causal-mask API参数拼写/位置参数，配置代码相同。权重/config/tokenizer匹配不等于HF代码版本一致，区别记录在此。
- 第7轮目的：完成HF特征对照与V2生命周期验收。步骤：仅诊断中补回历史pad=None默认值，显式传入完整causal mask以跨版本调用原模型；增加零/短尾/stop/取消/slot复用/unsupported-request拒绝测试。
- 范围：verify_dflash_features.py、verify_server.py、新verify_wave_lifecycle.py；不改权重、原始config或生产模型数学算子。
- 标准：native特征为最终循环物理层输出且target logits不变，HF误差完整报告；生命周期完成后flush且无严格KV错误；多batch数值失败不因此抹去。
- 风险：HTTP取消时序、旧HF masking API、chunk/padding导致shape不同；记录所有payload/output，诊断与主性能计时分开。
- 验证：本地HTTP/算法契约和AST，Git同步后X5 GPU0 lifecycle eager/graph overlap，GPU1 feature对照与DFlash graph smoke；启动7/10。

### 候选7结果与候选8/10六项计划

- SHAba0e772d79f4914b2804c48155c27d8779a6c82c；Wave lifecycle eager和graph均exit0：两组8请求不同长度(含0/1/短尾)、token stop、两次取消与survivor、sampling/logprob拒绝、重复flush，严格KV检查未失败。
- DFlash native feature与最终物理层hook逐位相同，开启capture不改target logits；HF4个检查点argmax一致，数值误差完整保留round-7/dflash-features/report.json，exit0。HF不是逐位相同，不能误报。
- DFlash graph服务在首次prefill触发KeyError1，exit1。确定根因：breakable prefill只replay内部OuroModel body，Python属性字典不恢复，而顶层forward清空字典后查找feature。
- 第8轮目的：使aux feature成为prefill body图的真实输出。步骤：OuroModel在启用capture时返回(hidden, aux)，顶层消费显式返回值；不依赖replay恢复Python side effect。
- 范围：models/ouro.py及本记录。成功标准：DFLASH prefill/verify/draft graph启动并完成真实请求；eager特征语义和target logits不变；保留跨shape数值差异。
- 风险：嵌套body输出结构、timeline手动路径；保持无capture时原返回类型，timeline分支显式组装aux。
- 验证：AST/diff/CPU契约；Git同步同SHA后DFlash graph bs1与多请求生命周期，重跑feature对照；Wave未改数值路径，按需服务回归。启动8/10。

### 候选8结果和当前交付状态

- 验证SHA fa023257486d319c9be18c0673e5ff56664f6447，本地/origin/X5一致；生产源码工作树干净。
- round-8/dflash-graph exit0：prefill breakable、target verify full、draft full图均运行，两个64-token请求与同配置AR的IDs/text/stop严格相同。
- round-8/dflash-lifecycle exit0：graph/overlap bs8，零/短输出、stop、取消、survivor、flush/复用通过；round-8/dflash-features exit0，capture hook契约和target logits不变，HF四点argmax一致。
- 推理checkpoint落于X5 /home/ruoxi/models/Ouro-2.6B-Thinking-DFlash-inference-step10000。首次host copytree因容器root创建的safetensors读取权限失败；随后仅用docker cp复制该导出artifact，host逐项hash核对全部通过。未更改源训练文件或其权限。
- 数值口径已向用户异步澄清：保留原BF16并披露分歧，或先要求确定性逐token一致。尚未收到答复，不能将超时当作批准；AIME25完整30题和性能profile尚未启动。
- 冻结MathArena数据artifact SHA256 b1cb6f70eb12b5a0e68f9ea88be9a35c6a3c66f0d00112b5b26160b042326b19，HF revision c94da77eb22bbd6439e62a323bec18493a421302，源题面原样保留。旧aime.jsonl未改，数据不进Git。
- 轻量证据已复制至results/：validation-summary、Wave固定prefix探针、DFlash feature/export manifest及数据manifest。完整tokens/logs/native features留在X5 raw root，不把大文件提交Git。
- 收尾检查：GPU0/1均8MiB，无compute processes；30215/30216无监听。本任务服务均退出，两个隔离容器仅sleep；X5会话92524和A10会话97155未主动断开。
- 已用8/10轮，剩余2轮；没有重置计数，也未把DAS-Wave自动晋升到loopspec。后续文档/证据提交不改变已验证runtime tree。

## 第9/10轮：规范模型路径、全30题与独立Profile

用户已明确要求继续完整AIME25和Profile，并将正确推理模型替换到规范名称。保留R4/BF16，严格token数值验收仍标为未通过，不再等待口径确认。

1. 目的：完成全部30题的bs1实测并补齐可复算的接受率和阶段时间。
2. 步骤：核对导出hash并备份替换规范目录；补齐运行器/轻量计数/可关闭trace；短运行验证；同GPU串行完整负载；独立Profile与汇总。
3. 范围：本目录run_aime25.py、summarize_aime25.py；loopspec/utils.py和worker.py的计数；dflash_worker_v2.py的opt-in阶段span；不改变算法数学。
4. 标准：各method重复覆盖30个唯一题ID并保存全部token和终止原因；规范DFlash目录实际加载；指标分母可审计；Profile区分inclusive span与kernel busy，LS depth1/3分别报告。
5. 风险：长前缀显存、BF16分歧、8192上限截断、profile开销；保留失败和截断，不能声称无损或所有题自然结束。
6. 验证：本地语法及统计契约检查，提交push/pull同SHA；X5 GPU0同容器短生成和profile smoke；正式run_aime25各算法30题，profiling单独运行，raw保存/results。

开始候选9/10。X5规范替换已完成：旧训练目录备份为`/home/ruoxi/models/Ouro-2.6B-Thinking-DFlash-training-backup-20261105T115931Z`；导出目录rename到`/home/ruoxi/models/Ouro-2.6B-Thinking-DFlash`，config和safetensors SHA256替换前后均通过。原训练文件、raw导出均保留。

第9轮验证：提交`79806c4943caa152c408eab1fdab0482ece5e22a`，本地py_compile、6项unittest、5项统计契约退出0。X5同SHA、当前源码import核实。`run_aime25.py --output /results/round9-smoke --smoke --tokens 64`五方法各2题完成，退出0；`summarize_aime25.py`退出0。规范DFlash路径已实际加载。独立`--profile-reference /results/round9-smoke --profile-ids 0 --profile-tokens 16 --tokens 12288 --context 14336 --capacity 13312`五方法各3窗口完成，退出0；GPU图和trace可用。raw位于round9-smoke和round9-profile-smoke。短负载不是全30题结果。

## 第10/10轮：长输出测量开销与Profile归因

1. 目的：避免累计SSE输出的二次方传输影响长答案计时，补齐明确的DFlash prepare/commit与kernel阶段归因。
2. 步骤：增量SSE拼接并逐事件核对token总数；添加关闭时无wrapper的prepare/commit spans；用CUDA correlation关联runtime launch和kernel，再按CPU span包含关系分类；短生成/Profile验证后完整30题。
3. 范围：run_aime25.py、summarize_profile.py，dflash_worker_v2.py、dflash_info.py的诊断标记；无算法变化。
4. 标准：增量输出长度与completion_tokens逐事件相等；各方法输出/计数可汇总；所有graph kernels通过correlation归因并明确无法映射部分；全部30题每方法完整保存。
5. 风险：SSE最后事件重复/空增量，GPU多stream无法按CPU墙钟直接切分，Profiler扰动。使用API增量开关和严格长度断言；kernel归属按launch correlation而非时间重叠；保留未归属项。
6. 验证：本地语法、现有6项测试，X5同步同SHA五方法64token，独立16token Profile；正式统一max_new=12288/context14336/KV13312，同GPU0串行。若此轮仍有源码问题，不进行第11轮。

开始候选10/10。

第10轮短验证：`da86ebecb1bccfa6aea1e73812e38cefa5c2a05a`，本地py_compile、6项unittest退出0；X5五方法各2×64 tokens生成及summarize退出0。AR/DFlash/LoopSpec1增量输出与第9轮IDs/text逐项相同。DFlash新增prepare_block/prepare_verify的3个Profile窗口退出0、标记实际存在。kernel归因分析退出0，未映射kernel计数保留（不是全部映射率100%）。正式运行于2026-11-05约12:18 UTC开始：`run_aime25.py --output /results/aime25-full-v1 --tokens 12288 --context 14336 --capacity 13312`，GPU0，五方法按固定顺序各30题。截止记录时仍在运行，不标完成。

只读性能线索（非新的实现/验证轮）：当前Ouro attention为保留checkpoint的BF16 QK/scale/normalized-probability舍入，执行两遍prefix扫描。`python/sglang/kernels/ops/attention/ouro_attention.py`的grid为(batch_size, heads, ceil(query_len/16))，decode bs1时只有16个head程序，未沿KV长度split；每程序以N=32遍历完整KV，M=16但decode仅1个有效query。这为长前缀AR速率下降提供了代码层面的假设，须用正式Profile核实。DFlash block query packing、LoopSpec分支并行和Wave多行会改变有效工作/occupancy，因此加速比包含当前后端执行形状收益，不能外推到优化split-KV基线。此处不擅自替换舍入语义或在10轮外改kernel。

第10轮短负载的正确性口径补充：执行/流式契约通过不等于严格AR一致。AIME前两题各64 tokens中，四个spec配置都在index2和index20首次与AR分歧，详见results/round10-smoke.json；各算法的增量/累计接口对照仍一致。该结果延续既有BF16 shape数值问题，未把它改标为无损通过。正式实验遵照用户最新要求继续，报告吞吐比、完整输出差异、准确率和截断，不能把不同输出的耗时比解释为同工作量无损加速。

## 用户调整实验口径：总上下文1024，所有方法CUDA Graph

用户最新明确要求限制上下文1024并全部开启CUDA Graph。停止原长上下文运行（保留aime25-full-v1原始目录，其不计作全30题完成）。不改源码，沿用第10轮候选`da86ebecb1bccfa6aea1e73812e38cefa5c2a05a`。30题最长prompt804 tokens，统一max_new_tokens128，为verify临时窗口保留至少92 tokens；context_length1024，capacity4096，bs1。所有方法启动参数仍为`--cuda-graph-max-bs-decode 1`，无disable-cuda-graph；实际capture/replay由server日志和Profile核实。新输出aime25-context1024-graph-v1，不能把128token截断测试称为完整自然解题或与旧长上下文结果混合。

## 用户澄清：每题最多输出1024 tokens（主口径）

上一条“上下文1024”被助手误解为总上下文上限；用户纠正为每题最多输出1024 tokens。已承认理解错误，停止128-output运行。该目录仅保留诊断数据，不能作为最终主结果。正确主参数：max_new_tokens1024，context_length2048（最长prompt804+输出1024+verify余量32<2048），capacity4096，bs1，完整R4/BF16，全方法CUDA Graph开启。沿用同一第10轮候选，无源码修改。新目录aime25-output1024-graph-v1，先完成五种配置全30题，再按计划检查重复与独立Profile。


## 最新主实验口径与 bs=2/4/8/16 扩展（同一候选，无新增修改轮）

用户进一步明确“不要管总上下文多少”：仅限制每题 max_new_tokens=1024。模型原生 max_position_embeddings=65536，实验不额外缩短上下文；物理 KV 池是 GPU 容量约束，不作为题目截断策略。bs1 已用 unchanged run_aime25.py --tokens 1024 --context 65536 --capacity 13312 --repeat 1 启动，目录 aime25-output1024-graph-bs1/run，host driver PID926309，X5 同 SHA da86ebecb1bccfa6aea1e73812e38cefa5c2a05a。

扩展实验计划（不改运行时代码或测试工具）：
1. 目的：测完整30题在并发/调度上限 bs=2/4/8/16 下的同配置 AR、DFlash、LoopSpec depth1、LoopSpec depth1/3、DAS-Wave 对比。
2. 步骤：复用已有 profile_workload.py 的 --external-server 通用 HTTP cohort；30题同时提交，由实际 server --max-running-requests B 控制，记录排队和实际活跃 batch；同 GPU 串行各方法；独立 Profile。
3. 范围：现有工具 CLI 调用、原始实验产物及研究记录；无推理源码/脚本修改。外部服务模式下旧 harness 的 algorithm 参数仅用于构造未执行的 command，真实算法以另存实际 launch、get_server_info 和 server.log 为准，汇总不可把占位 ar 当实际算法。
4. 标准：每配置30个唯一题目完整 HTTP 响应、每题至多1024输出、原始输入 IDs/输出 IDs/停止条件和最后 meta_info 可复算，CUDA Graph 实际执行，和相同 B 的 AR 对比吞吐/延迟/接受率。
5. 风险：32GiB 卡的完整 R4 KV 在 B16 可能不够；实际动态 batch、回退/重新调度、CUDA Graph 覆盖、失败须显式报告；已有跨 shape BF16 分歧不视为已解决。
6. 验证：核对 prompts.json 与 bs1 完全相同，所有响应数/唯一题 ID/finish/max tokens；日志实际图/批大小/计数，指标按 sum(output_tokens)/cohort makespan，TTFT 含 server 排队，单请求 decode 与总体吞吐分开。详细 Profile 与无 Profile 性能隔离。

执行资源补充：X5 GPU0 跑 bs1，空闲 GPU1 跑 bs2/4/8/16；每个 B 的五方法在同一 GPU 上顺序配对。两卡共享 CPU/主机，记录为并行独立 GPU 条件，不能把它当隔离单卡测量。GPU1 外部服务不传 --context-length，使用模型默认65536。KV pool13312 全方法一致；profile 仍单独进行。


Profile 分组只读核对：pipeline._child 在 parent.depth=d 时创建 depth0 分支；根完成 R4 时该分支已执行4-d次。故接受 draft depth1 后下一输出验证区间 entry_depth3，仅需1次新core；接受 draft depth3 后 entry_depth1，需3次新core；拒绝/启动 entry_depth0，需4次。analyze_profile.py 的 entry_depth 是待验证 token 已计算深度，不是 proposal origin。最终报告须明确区间条件是“前一个输出接受哪个草稿后”的流水线增量成本，不能将 entry_depth3 标为 draft depth3。bs1 LS13 全30题第二候选接受次数29–181/题，默认q0/14/29的短窗口有望覆盖该路径，最终按实际样本数报告。

正式Profile抽样选定q0/q2/q20（输入54/137/804 tokens），覆盖短、典型和最长输入；每方法在自身1024-token主轨迹的0%/50%/90%位置，独立重建prefix后测32输出token，共45窗口。替代工具默认q0/q14/q29，仅改变已有CLI选择，不改源码。Profile前缀重建是独立诊断，不替代主运行的接受率/耗时，也不保证与增量KV轨迹逐位相同。

为复用空闲资源，bs1完成后GPU0执行bs16五配置；GPU1继续bs4/8。每个B内部仍同物理GPU配对，跨B的绝对吞吐曲线需标注GPU分配（B1/16=GPU0，B2/4/8=GPU1），不能把跨卡差异当纯batch扩展收益。双卡同时运行条件与telemetry一并保留。详细bs1 Profile仍在GPU0，且与其主吞吐运行分离。


bs16 默认调度失败及现有参数实验：ar-b16 driver退出1，`Incomplete output IDs for run-0-q17`；server出现多次KV满重调度，原始目录完整保留，不能将该cell计为成功。只读检查发现LS/Wave上下文统计也会在重调度后重置，因此不能把最后一段计数冒充完整请求接受率。

1. 目的：在32GiB卡完整R4/BF16容量限制下完成可审计的限并发16负载，避免重调度导致的流式/计数问题。
2. 步骤：沿用同一未修改候选，使用现有环境参数SGLANG_INIT_NEW_TOKEN_RATIO=1及SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1，对全部五方法预留完整1024输出预算，重新从AR开始配对。
3. 范围：仅已有server CLI启动环境与新的raw目录aime25-output1024-matrix-reserved；不改源码/权重/输出上限，原失败目录不覆盖。
4. 标准：全30题完整IDs和stop，0重调度，实际admission/batch分布公开；此数据明确标为并发上限16且受显存约束，不能当全程固定bs16。
5. 风险：物理KV无法容纳16条完整序列，实际活跃请求会低于16；与bs2/4/8的默认admission策略不同，跨B扩展曲线不能无条件解释；如仍失败保持失败证据。
6. 验证：核对进程环境中上述两项为1、同SHA和CUDA Graph启用；所有响应IDs数量等于completion_tokens且<=1024，metrics num_retractions为0；各方法同GPU0、相同容量13312、同admission设置。该操作为已有候选的运行参数实验，没有新增代码修改轮，预算仍10/10。


bs16资源复测：13312 KV池下，Wave在verify CUDA Graph capture时报OOM，尚未生成；当时GPU31.25GiB占用，图私有池3.55GiB，原日志保留。预留预算AR虽然30题IDs完整，仍有1次retraction，最后metrics快照为0不能替代每请求累计num_retractions；原“0重调度”标准未满足，未改标为通过。DFlash/LS1/LS13则无retraction且完整30题通过。

现有CLI参数计划：1目的：保留全方法CUDA Graph并建立共同可容纳的bs16容量条件；2步骤：先核实Wave捕获能在12000-token KV池完成，再同GPU0重测五配置全30题；3范围：新raw目录aime25-output1024-matrix-b16-kv12000、max-total-tokens=12000，保留full-budget admission环境，源码不变；4标准：全30题严格IDs/stop通过，图开启，计数覆盖可审计，实际并发披露；5风险：实际并发低于16、仍可能retraction，原13312结果只作资源诊断，不能混入新矩阵；6验证：相同候选、dtype/R4/template/输出上限1024，全部五方法用相同12000 KV容量，按每请求num_retractions和原始日志判断，而非最后Gauge快照。预算仍10/10，无新源码候选。


## 全30题主矩阵完成（沿用候选10/10）

最终25组全部完成，5方法×B1/2/4/8/16，每组30题、每题1024 tokens，共750请求/768000输出tokens。bs1 raw `aime25-output1024-graph-bs1/run`；B2/4/8 raw `aime25-output1024-matrix`；最终B16 raw `aime25-output1024-matrix-b16-kv12000`。各driver exit0，逐请求最终meta核对0 retraction、完整IDs和length stop，所有输入IDs与冻结Thinking prompts一致。B16旧13312容量结果和失败不混入主表。实际B16活跃生成峰值AR/Wave/LS1/LS13为10、DFlash为9，不能称全程固定batch16。

B1 E2E tokens/s：AR30.5069，DF52.5318，LS1 71.1782，LS13 75.9316，Wave24.5868。相同B吞吐比完整见benchmark-report.md和results/aime25-output1024-matrix.csv。所有输出均1024截断，长输出严格AR等价未通过，未改标为无损。单pass无重复置信区间；部分时段另一GPU同时负载，后期GPU1空闲，跨B扩展性能存在资源/调度混杂。

图日志复核：external harness会在计时前访问/health，源码health_generate(input_ids=[0], max_new_tokens=1)。DFlash的B2/4/8/16各一次False恰对应这个1-token prefill，早于GET/get_server_info与正式cohort提交；20组正式请求阶段日志均True。证据aime25-output1024-graph-audit.json保留原False行，不能删除它们或谎报整份日志全部True。20组manifest同SHA da86ebecb1bccfa6aea1e73812e38cefa5c2a05a、正确源码import及数据hash通过，actual-launch与server_info另存环境audit。

详细Profile于主测完成且GPU0清空后启动，host driver1041636，沿用已记录命令与q0/2/20×三个轨迹位置×32输出，独立目录aime25-output1024-profile-bs1。分析仅后处理既有trace/日志，不改运行时代码或测试工具；仍10/10。


## Profile与结果后处理完成

run_aime25 Profile driver1041636退出0，45窗口全部32输出，无retraction。summarize_profile退出0；进一步只读后处理按request ID过滤CPU区间和CUDA launch，显式补齐scheduler.forward下AR finish的decode归因。发现LoopSpec首窗口含各7个其他rid的启动探活decode输出，原trace/summary完整保留，过滤后每方法279个decode输出区间；D1/3的D3条件样本16个。Wave150次callback中4次为空，周期均值只计146个真实draft/verify；DFlash132周期。GPU kernel未归因时间DFlash5.381%，其他方法<=0.825%，未硬分配未知项。全部算法trace实际包含CUDA Graph launch。详见profile-report.md和profile-aggregate-filtered.json，未修改任何运行时/测试源码。

关键结果：AR主测32.755ms/token；DF42.797ms/周期；Wave78.007ms/周期。Profile D1/3接受D1后的GPU区间12.486ms（252样本），接受D3后34.993ms（16样本），重启49.395ms（11样本）。DF target verify占其GPU stream周期约97.04%；Wave draft/verify占51.43%/47.65%。已归因decode kernel中GEMM/GEMV约47–51%、attention约26–30%，未将attention单独描述为全部瓶颈。

收尾：主测与Profile进程结束，telemetry930126已停止；两卡均8MiB/0%且30215/30216无监听，两个容器仅保留本任务环境，SSH会话84278保持。轻量JSON/CSV/PNG与报告本地核对后提交loopspec-test，原始tokens/trace留X5。10/10轮预算未重置，DAS-Wave严格无损与实际加速仍未通过；不进行第11轮、不自动晋升正式分支。

交付前只读校验退出0：25个唯一method/B组合、每组30×1024、0 retraction；45个Profile窗口、各方法graph launches>0、过滤后的AR/LS各279个decode区间；全部JSON解析与报告相对链接通过；两张PNG已渲染检查。仅提交本任务报告和轻量结果，不改runtime tree。
