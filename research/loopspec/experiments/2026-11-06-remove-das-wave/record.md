# Remove DAS-Wave after 099cfd389

## 身份与预算

- 用户请求：删除 `099cfd389` 后所有 DAS-Wave 方法的修改。
- 基线：`d61635dce7f171f1a33baaf8d07cb59cf8ee07e6`；恢复参照：`099cfd389c4c02e478d45b22c974ed658a7e874a`。
- 分支：loopspec-test；正式 loopspec 保持 `82a851f6fe12c6e17354054fd7b160594f203b66`。
- 状态：通过；本次撤除第 1/10 轮。原 DAS-Wave 实现任务的 10/10 记录保留；本任务不继续优化该方法。

## 修改前计划

1. 目的：撤除 DAS-Wave 的运行行为，保留混合提交中的 DFlash 修复和 LoopSpec 统计。
2. 步骤：核对 099..HEAD；恢复 Wave-only 文件、删除专用模块和测试；清理公共脚本依赖；标记历史记录；本地验证后提交/push，再 X5 pull 和短测。
3. 范围：恢复 arg_groups/{field_order.py,fields/spec.py}、managers/{tokenizer_manager.py,scheduler_components/metrics_reporter.py}、mem_cache/timeline_pool.py、model_executor/recurrent_runner.py、speculative/{spec_info.py,spec_registry.py,spec_utils.py}。删除 speculative/das_wave 六文件及研究目录 probe_wave.py、test_wave.py。公共生命周期工具改中性名称；调整 verify_server.py、verify_dflash_features.py、run_aime25.py。既有报告标为历史，保留结果与只读解析器。
4. 成功标准：系统运行代码无 DAS-Wave 注册、配置、分发和实现；9 个恢复文件逐字节等于 099；Ouro、DFlash 两文件、LoopSpec 两文件逐字节等于撤除前 HEAD；默认运行列表仅 AR/DFlash/LoopSpec；X5 CUDA Graph 短测仍能完成。
5. 风险：混合提交不能整体 revert；DFlash feature 检查引用 probe_wave.error，删除前须迁移独立误差函数。历史失败证据不得销毁。通过新增撤除 commit 保留公开历史，不 force push。
6. 验证：git diff --check、Python AST、HTTP 单测、字节级基线比对、运行代码引用扫描；沿用用户此前指定的 X5 实验环境，核对 SHA/包路径/GPU/内存/端口，在 Ouro-2.6B-Thinking R=4 BF16 Triton 下跑 AR、DFlash、LoopSpec depth1、depth1/3，CUDA Graph 开启，AIME25 前两题每题最多64输出token；与旧 round10-smoke 对应算法逐 token/停止条件比较。仅撤除回归测试，不重新声称跨算法无损或性能改善。

## 候选 1/10

- 假设：选择性恢复 Wave-only 接线不会改变其他算法；移除 Wave 依赖后 DFlash 工具仍独立可用。
- 本地检查：`git diff --check`、`python3 -m unittest discover -s research/loopspec/experiments/2026-11-05-das-wave -p test_http.py`（1 项）、所有保留/修改研究脚本 AST、9 个恢复文件与 099 字节比对、5 个保留运行时文件与 d616 字节比对、python/ 下 Wave 引用扫描全部通过，退出码0。`run_aime25.py --help`仅列四种保留方法。
- X5 检查：通过，详见下方结果。历史 session 84278 已失效且旧 ControlMaster 不存在，已重建持久 session 24353；不修改远端源码。
- 正式分支：不晋升。

## 验证结果与完成

- 本地/origin/X5 测试提交均为 `82ad8f604718010e6e77d5c9182dcd854b2c86e1`，源码 clean。相对于099，python运行代码仅余 Ouro/DFlash/LoopSpec 五文件的独立改动。
- X5 算法导入检查：NONE、DFLASH、LOOPSPEC正常，DAS_WAVE解析抛出ValueError，退出0。
- 容器 `loopspec-das-wave-20261105`（名称为历史名称），镜像 `81726dc9e63a9c6778396c8176a7203716a527f28b2bbb6b0eac661a0b050afb`；Python `/opt/sglang/bin/python`，sglang `/workspace/sglang/python/sglang/__init__.py`，Torch2.14.1+cu130。只读挂载X5 repo。
- GPU0 RTX5090 UUID `GPU-449e057b-787e-b46a-7a4c-727bfe1bd515`；启动前双卡8MiB/0%、无计算进程，30215空闲，主机available180GiB、swap使用3.4GiB。
- 命令：`docker exec -e PYTHONPATH=/workspace/sglang/python -w /workspace/sglang loopspec-das-wave-20261105 /opt/sglang/bin/python research/loopspec/experiments/2026-11-05-das-wave/run_aime25.py --output /results/remove-das-wave-20261106-smoke --smoke --tokens 64 --context 14336 --capacity 13312 --port 30215`。采用旧round10-smoke相同容量以逐token回归，退出0。
- 四方法各两题×64输出tokens，共8请求；对旧round10-smoke的同算法输出ID及停止条件8/8一致。所有方法正式prefill/decode日志均有CUDA Graph True。
- Raw：X5 `/home/ruoxi/experiments/loopspec/2026-11-05-das-wave/remove-das-wave-20261106-smoke`，含manifest、prompts、逐题输出、server日志、removal-regression.json。
- 一次host分析在断言通过后写容器root所有的目录遇到PermissionError；改为同容器执行只读比较并保存摘要，退出0。无源码修订，不计新候选。
- 这是撤除回归，不是全30题新基准或TP2验证。长期跨算法数值等价问题沿用历史结论。
- 达到本任务成功标准；1/10轮。正式loopspec未改动，公开历史未重写。
