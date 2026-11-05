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
