# A-series TP2 AR / DFlash / single-proposal LoopSpec

## 身份与预算

- 用户明确要求：DFlash修复后在A系列单机TP2重新比较AR/DFlash/LoopSpec，输出加速比、acceptance及详细Profile；LoopSpec仅一个proposal位置。
- 基线：935d48ddbf0a0f6763f6afcd00cded318f206ad2（Wave撤除已通过X5回归）。分支loopspec-test，正式loopspec不动。
- 状态：进行中，候选1/10。TP2接入是新的用户任务，不重开原DAS-Wave优化预算。
- 口径：默认bs1、全30题AIME25、max_new_tokens1024，Thinking模板、R4、BF16、greedy、CUDA Graph全开。LoopSpec depths=[1]，只保留第一proposal分支；流水线仍可有不同token的在途lane。batch矩阵已异步询问用户，未答时先bs1。

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
