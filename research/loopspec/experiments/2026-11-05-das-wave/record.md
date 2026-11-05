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
