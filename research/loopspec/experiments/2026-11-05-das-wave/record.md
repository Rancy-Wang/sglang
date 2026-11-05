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
