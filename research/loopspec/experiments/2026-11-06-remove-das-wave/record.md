# Remove DAS-Wave after 099cfd389

## 身份与预算

- 用户请求：删除 `099cfd389` 后所有 DAS-Wave 方法的修改。
- 基线：`d61635dce7f171f1a33baaf8d07cb59cf8ee07e6`；恢复参照：`099cfd389c4c02e478d45b22c974ed658a7e874a`。
- 分支：loopspec-test；正式 loopspec 保持 `82a851f6fe12c6e17354054fd7b160594f203b66`。
- 状态：进行中；本次撤除第 1/10 轮。原 DAS-Wave 实现任务的 10/10 记录保留；本任务不继续优化该方法。

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
- X5 检查：待执行。历史 session 84278 已失效且旧 ControlMaster 不存在，已重建持久 session 24353；不修改远端源码。
- 正式分支：不晋升。
