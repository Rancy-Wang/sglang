# Thinking AIME batch profile

## Identity and budget

New user-requested benchmark task; previous CDB implementation task remains completed at 18/20.
Baseline ee77d73ffdfbd9ca3ed2f1d79ef095268732499a; loopspec-test only. Round 1/10, in progress.
No formal branch promotion authorized. SSH session 60877 reused, X5 ruoxi.

## Six-part plan

1. Compare current native AR/CDB and LoopSpec V2 on the newly downloaded Thinking checkpoint, bs=1/2/4/8/16, simultaneous and staggered arrivals.
2. Validate full R=4 native reference and serving outputs, then collect separate warmed unprofiled latency and detailed profiles for all 20 cells. Diagnose first token divergences rather than claiming exact equivalence.
3. Experiment scripts only: extend previous profile_workload.py with opt-in Thinking/template, graph/context/memory and compact streaming fields; new matrix driver and indexed analyzer here; independent full-R numerical probe here. No planned production edits.
4. Success: all cells have complete streams and recorded finish causes, identical AR/LS input IDs and sampling settings, per-request and cohort E2E distributions, depth-path counts and inclusive stage CPU/GPU intervals with shared batch ownership. Performance is observational; slower cases must be retained. Independent native computation precedes interpreting performance. Numerics calibrated on this checkpoint, not copied from base model.
5. Risks: batch-shape BF16 divergence, EOS-dependent work, KV retraction under B16, huge traces/profiler overhead. Preserve failing logs; distinguish unprofiled timing, profiled timing, shared stream intervals, kernel busy time, and host wall time. Do not sum nested/shared spans as exclusive costs.
6. X5 GPU0 RTX5090, isolated image lmsysorg/sglang:v0.5.22-cu130, readonly source/model/data, source SHA verified after Git sync. Model /home/ruoxi/models/Ouro-2.6B-Thinking revision f1edd81e7ac41355db670500ceaf204e0f73af68. BF16 Triton, R4, greedy, d1=1, native overlap+CUDA graphs, max requests/graph batch16. AIME first16, checkpoint chat template enable_thinking=True/add_generation_prompt=True; prompt IDs/template saved. Provisional 512 output tokens pending optional user preference; natural EOS retained and truncations labelled. Two warmups, five measured repeats; each request scheduled at common t0 or t0+100*i ms. Separate one full cohort trace per cell after warmup. Output /home/ruoxi/repo/test/results/loopspec/2026-10-29-thinking-batch-profile. CPU compile/diff checks, Git commit/push/pull, container import/config check, independent native AR and LoopSpec correctness, then complete matrix. Local checks are not GPU verification.

## Round 1/10

Hypothesis: existing serving code supports Thinking weights without production changes; existing trace metadata suffices to assign token paths and shared stages. New benchmark tooling is needed for template consistency, capacity16, repeated matrix, and linear-time trace grouping.
Validation pending.

Round 1 validation: local compileall/diff-check exit0; local/origin/X5 SHA 17b55e656d6461269ef478e4f7ddc38f6325bd2d. New container loopspec-thinking-20261029, GPU0, import /workspace/sglang/python/sglang/__init__.py. native_reference.py --output /results/native-reference-r1 --steps 64 exit0: 16 single-request references, 2048 independent B16/B64 teacher-forced positions, 10 argmax flips (0.488%), full-R ordinary KV path explicitly asserted. Calibrated per-depth bounds stored before LoopSpec evaluation. max_total_num_tokens14342. New isolated /opt/thinking-hf Transformers4.54.1 installed exit0, without changing serving environment.

## Round 2/10

Add checkpoint-owned HF eager spot checks (q1/q16 at prefixes0/15/63), explicit trust_remote_code for tokenizer to avoid interactive fallback, and final reproducible CSV/Markdown report generator. This extends the original validation/report scope; production unchanged. Validate HF/native on identical inputs, LoopSpec graph16 against independently calibrated native references, strict HTTP smoke1/16, then run all performance/profile cells. Full latency measurements remain pending. Default512 retained after optional clarification window.
