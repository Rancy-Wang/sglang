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

Round 2: SHA314f62a6da32c07607555cb1403246e2a79564cf. HF exited1 (isolated environment cannot find torch). LoopSpec actual graph/16 owners/64 positions exited1: six numerical envelope limits exceeded (depth1 maxRMSE .3362, depth2 maxabs2.5625/RMSE .5661, depth3 maxabs2.875, logits maxabs6.90625/RMSE1.00565); retain full report. This is not a performance pass.

## Round 3/10

Independent control omitted padded prefill: graph pads total tokens to next power-of-two, whereas native calibration used unpadded M. Extend ONLY independent native calibration to graph-equivalent prefill padding and decode M16/32/64, retaining B1 reference unchanged; then rerun unchanged LoopSpec verification against independently determined envelope. Correct HF isolation by ordered PYTHONPATH of HF site-packages then serving site-packages, without modifying serving dependencies or checkpoint. No production or algorithm changes. Preserve prior failed results.

Round 3: SHA6c42e8be31b489d50957d4c8f0c288dbd9f64331. Independent padded controls exit0, 3072 positions/14 low-margin flips. Rechecking unchanged LoopSpec report removes five of six failures; depth1 RMSE .336233 remains at q9 prefix30 (target argmax identical, margin9.125). HF environment fails in new kernels package with old hub strict dataclass compatibility; no GPU HF result yet.

## Round 4/10

Complete independent calibration for ALL requested batch sizes, not only16: contiguous cohorts covering all16 questions at bs1/2/4/8 and physical row widths1/4 with graph-equivalent prefill padding; full native forward, identical reference prefixes, no LoopSpec computation. This specifically covers small/mixed cohort shapes omitted before. Retain numeric check and observed outlier. Pin kernels0.9.0/huggingface_hub0.34.4 only in independent HF environment to match its Transformers4.54.1; rerun unchanged HF probe. Production unchanged. Then strict HTTP smoke before performance interpretation.

Round 4: SHA1817d6a4045fe1bea74264654985a99f6bd07696. HF eager6positions exit0/all final argmax equal, maxlogitabs .3359375/maxRMSE .0742701. Native expanded controls exit0,11264positions/48flips; independent maxdepth1RMSE .350301 =>25%headroom .437876, covers prior .336233 LoopSpec outlier. Strict HTTP smoke did not execute requests: client failed Git dubious ownership; its server independently OOM during prefill graph capture at memfraction.85 (pool+graphs left90MiB). Both failures preserved. Container-specific Git safe.directory set; no global host change. No living GPU process/port after server self-cleanup. Synthetic AR/LS analyzer4-loop association checks exit0.

## Round 5/10

For BOTH algorithms set memfraction.78/max_total_tokens12000, same R4/context2048/graphmaxdecode16. Workload3218prompt+8192maxoutput=11410 logical tokens plus small branch headroom, so capacity should cover B16. Verify no retractions rather than assuming. Compute Git SHA before spawning server so metadata failure cannot orphan it. Strict smoke1/16, verify unchanged LoopSpec with new independently measured envelope, then full20-cell performance and separate profile. No production edits.
