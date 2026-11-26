# VQ+ page-path implementation follow-up

## Authorization and budget

2026-11-26: the user explicitly authorized a new maximum of 10 optimization rounds, preserving the existing training method, pre-RoPE quantization format and trained codebooks, implementing the page-level data path and verifying four controls on A100. The previous implementation remains completed at 10/10; this follow-up has its own explicitly authorized budget. No post-RoPE codec, retraining, INT2 residual, or changed rounding is in scope.

Baseline: SGLang `2eb4f0d380effc82cd4008fcf8096276098cdcbe`, branch `loopspec-test`. Use the existing task workspace and preserve unrelated snapshots and LoopVQ training record edits.

## Implementation plan

1. Purpose: reduce repeated index/scale/address instructions and inefficient head-strided token reads without changing decoded values.
2. Steps: freeze the environment; capture actual model Q/pre-RoPE K/V and codebooks; implement page/group32 reader; compare current VQ+, materialized-equivalent native BF16 attention, scalar INT4, and paged VQ+; then change scheduling/pipelining based on measured evidence; integrate the winning path and verify serving/flush/lifecycle.
3. Scope: new `residual_attention_paged_cuda.py`, task-local capture/diagnostic scripts, followed by encoder/pool/dispatch changes once the reader is proven. Existing CUDA kernels remain baseline controls. No training/model-weight changes.
4. Criteria: codec values and BF16 boundaries unchanged; output atol=.002/rtol=.01 and LSE atol=rtol=2e-5. Seek >=15% lower B8/B16 attention latency versus current VQ+ before full serving validation; report BF16 and scalar INT4 comparisons honestly. The proposed broader target is BF16-relative throughput geomean>=1.10 and per-shape>=.95; failure must be reported, not hidden by redefining the target.
5. Risks: page indexing, mixed HP/quant, partial pages, warp shuffle participation, register/smem pressure, stale graph scratch, flush publication, random versus real code distributions. Scalar INT4 is only a separate diagnostic codec; it must not replace the authorized VQ+ production codec silently.
6. Validation: local syntax/diff checks; local commit and push only task files to `loopspec-test`; remote Git synchronization; A100 correctness before timing, interleaved CUDA-event/graph timing, real-model snapshots, NCU if permitted, sanitizer, TP2 serving with unchanged prompts and full four loops. All costs including encoder/flush and conversion are reported. Offline materialization is diagnostic only.

## Round 1/10 — page/group32 layout and four-way harness

Status: implementation preparation; no GPU result yet.

Hypothesis: reorganizing token-major storage into `[page,head,group32,token8,payload]` plus cooperative scale loads reduces read/address instruction work and improves transactions. Retain the existing split/local-softmax baseline in the first candidate to isolate data path changes. Measure both layout and cooperative-load effects with explicit variants if needed.

Files: new paged CUDA module and task-local capture/diagnostic harness. Production encoder/pool wiring is deferred until correctness and kernel evidence establish the reader contract.

Connection: X5 persistent exec session `90355`, socket `/tmp/vqpage-20261126-x5.sock`, hostname `x5`, account `ruoxi`, shell PID1878419. Initial A10 attempt failed (exit255); X5 remained live. One bounded A10 retry uses explicit ProxyCommand before the config's ProxyJump, preserving X5 forwarding.

Candidate source: `c1ab6cb725d67614feefef8424a887eccad9a741`, pushed to origin and verified on the remote clean `loopspec-test` checkout. Local py_compile and staged diff checks passed.

Resource selection: A10 GPU4/5 initially idle but became occupied by another user's jobs before measurement. Our capture PID2506340 was terminated; exit143, nine captured requests retained under `inputs-2k`. No timing was taken there. A11 has another task resident on all GPUs, so it was skipped for isolated timing. A12 was verified with zero compute processes and idle A100 GPUs; selected GPU0 for capture and GPU1 for diagnostics.

A12 connection: persistent session24610, socket `/tmp/vqpage-20261126-a12.sock`, shell PID3411470, account ruoxi07, via X5. A11 session78913 is retained but not used for compute. The A12 `/homes` checkout, models and artifacts were verified, not assumed. Remote source `/homes/ruoxi07/repo/sglang-vqplus-kernel`, output root `/homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage`.

Capture command: `env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 /homes/ruoxi07/anaconda3/envs/loopvq-build/bin/python research/loopspec/experiments/2026-11-26-vqplus-residual/capture_paged_inputs.py --model /homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking --book /homes/ruoxi07/experiments/loopvq/2026-11-26-residual/artifacts/residual-ft --out /homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage/a12-inputs-2k --length 2048 --requests 16 --layers 0 23 47`. Exit0. Capture environment Torch2.8.0+cu128/Transformers4.56.2; all four loops asserted. FP32 serving RoPE applied to actual Q projections; raw pre-RoPE K/V preserved.

Service/diagnostic environment: existing `sglang-loopvq-test` Python, Torch2.14.1+cu130, Triton3.8.0; `sglang.__file__` verified in the current candidate checkout. Build cache is task-owned `extensions-a12`, MAX_JOBS=2. First smoke job PID3412536, wrapper3412535; uses partial A10 capture only for mathematical correctness, not a cross-host timing baseline. It compares old auto/R1, plain reader, paged reader, scalar INT4, and native materialized attention.

Results and decision: GPU smoke/compilation pending. Additional capture wrapper3414152 is collecting 1K/4K/8K on GPU0, with per-context exit files. No new source revision/optimization round was introduced by resource relocation.

Environment correction (same candidate/round): the first smoke exited1 because A12 has `/usr/bin/g++-13`, not `/usr/bin/g++`. Retry with `CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 TORCH_CUDA_ARCH_LIST=8.0`, wrapper3415621/Python3415622, `r1-smoke-gcc13.{log,json,exit}`, using the complete A12 2K capture. B1/layer23/loop0 passed all six output checks; B16 still compiling the old grouped baseline at this checkpoint. All A12 captures (1024/2048/4096/8192,16 requests,3 layers,4 loops) exited0.

A12 lacked NCU. Copied the existing A10 NCU2026.3.1 CLI target and sections (about141MB) into the task-owned shared output directory `tools/ncu-2026.3.1`, without system changes. A12 `--version` and `--list-sections` succeeded. Counter permission and SM80 section support remain to be tested. No performance conclusion yet.

Round1 initial results: smoke exit0, all six controls passed B1/B16 N2048 layer23 loop0; paged max_abs<=3.8147e-6 vs independent oracle. Interleaved graph medians at B16/N2048 (us): current226.51, old_R1 230.23, plain_reader219.83, paged258.71, scalar_INT4 218.73, materialized107.24. B8:120.48/120.51/127.70/132.22/90.23/62.29 respectively. N1024 paged B16 135.60us vs current130.34us. Reject the first page-reader as a performance improvement. Multi-context run still finishing; preserve all raw results. NCU2026.3.1 permission probe failed ERR_NVGPUCTRPERM; no counter evidence. cuobjdump reports64registers/thread, zero stack/local memory,512threads; dynamic smem is separately set by source.

## Round 2/10 — compact bounded streaming splits

Plan before edits: retain round1 as control, add a fixed capture-static CTA budget derived from B*H, at most32 splits. Each CTA processes its assigned interval in <=1024-token tiles and merges tile softmax statistics/output in FP32; only launched splits participate in stage2. This removes empty CTA writes/dispatch for short contexts without reading lengths on the host or limiting dynamic graph lengths. For B16/H8 the grid shrinks from4096 to256 CTAs. This is a source-derived hypothesis, not measured bottleneck attribution. Also isolate page-only and cooperation-only variants because round1's combined change regressed.

Scope: paged CUDA module, diagnostic harness, new boundary/graph/LSE harness. Keep old kernels and codecs unchanged. Validate exact layout inverse, lengths0..32768, ragged/odd batches, padding, noncontiguousQ, stale scratch, LSE atol/rtol2e-5 and output .002/.01, then real-KV paired timings. Risks: tile softmax rounding/overflow, loop barriers, stale unused splits, graph replay with changed lengths. Promotion requires correctness plus lower measured large-batch latency; a losing variant is retained only as an explicit experiment. No production path promotion until integration and serving tests.

Round1 completed: all four context runs exited0 (N1024/2048/4096/8192 × B1/2/4/8/16, layer23/loop0), output checks passed before each timing. B16 current/paged/materialized medians(us): N1024 130.34/135.60/70.59; N2048 226.51/258.71/107.24; N4096 357.66/396.07/193.93; N8192 649.60/714.60/349.21. Every large-batch page result fails the proposed15% improvement threshold. Local raw JSON copies are in `evidence/paged/r1/`; the complete remote logs remain under the output root. These are single-layer hot-cache graph timings, not full-model serving measurements.
