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

Results and decision: pending.
