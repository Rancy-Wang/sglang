# VQ+ page-path continuity

Status: bounded implementation and validation complete, new budget10/10 used. No further source, test or behavior changes without an explicitly renewed budget. Performance gates were not met; do not describe the result as globally optimal or faster than BF16 at large batch.

## Target and authorization

Local source: `/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang`, branch `loopspec-test`. Tested code revision: `7d1c54b5ab30a98ffda5b4cd5bbd0b797479b064`; any subsequent commit for this task is documentation/results only. User authorized preserving training/pre-RoPE codec, implementing a page data path and measuring four-way A100 controls, with at most10 new rounds. The prior10/10 implementation record remains separate and unchanged. Formal `loopspec` branch promotion was not authorized and did not occur.

Original linked chat:01a1f8df-2e9d-74f3-b471-0ec6577f4ee2 on remote-control:env_e_6a68ec597c9c8325a8035cb714e0bc83. Relevant design and results were read. Original analysis and design plan remain in draft.md/plan.md; their post-RoPE/INT2 proposals were outside the subsequent implementation scope.

## Contract and final implementation

Ouro-2.6B-Thinking, full4loops×48base layers, TP2/local8heads, MHA D128, SM80, context<=32768. Joint index codes encode4loops×4channels,C256. Per-loop INT4[-7,7] residuals/group32 FP16 scales; BF16 centers, FP32 reconstruction/RoPE followed by unchanged BF16 composition rounding. HP prefix8/recent64,page8. No prefix reuse, chunked prefill, speculative decode or offload support added.

Final opt-in page-head storage is [page,head,token8,payload], directly encoded by prefill and flush. Global CUDA reader,512threads, R2 onlyB2 and R1 otherwise; capture-static dispatch, existing dynamic effective splits and FP32 partial/LSE. Token-layout defaults and original dispatch remain unchanged. Rejected group32/cooperative/staged/remapped/cache/Triton variants remain explicit experiments.

## Results and evidence

Authoritative report: ../paged-results.md; structured results plus evidence hashes: ../paged-results.json;20shape matrix: ../paged-micro.csv; all10plans/commits/failures: ../paged-record.md. Raw local evidence: ../evidence/paged/{r1..r5,r6-r8,r7,r8,r9,r10}; large snapshots/traces remain on A12 as well.

Final checks:20CPU contracts,100boundary/LSE/dynamicgraph cases,144real cross-layer/loop/length/B1/B2/B16 cases all passed. Both direct encoders inverse-match token arrays bitwise including >2GiB int32-destination addressing. Final real-reader max_abs .0009765625 with unchanged output .002/.01 and LSE2e-5. Round9 forcedR2 memcheck0errors covers unchanged finalR2page kernel; historical round2 memcheck and round3 staged racecheck are separately labeled. Final TP2 lifecycle/abort/reuse/greedy outputIDs match tokenreference.

Same-card-pair full serving at2048input/128output, three repeats, complete active-batch/fullgraph evidence: B16 BF16420.824,tokenVQ+276.476,finalpage294.213tokens/s (+6.416% vsVQ+, still only~70% ofBF16). B8 final189.825 vsold186.322/BF16263.584. B2 regression from round8 was fixed. Final B16 single-layer latency reductions over1K/2K/4K/8K are~12/2/4/3%, not a broad15% improvement. Full serving outside2K was not retested in this budget. Small-B micro timings include clock warmup variation; no sub-percent or global-optimum claim.

NCU2026.3.1 failed ERR_NVGPUCTRPERM; no measured hardware-counter attribution. Static resources, PyTorch profiler and paired CUDA-event timings are available. Final TP0 B16 fullgraph53.865ms,mainattention35.340ms vsold57.490/38.602 andBF1638.316/19.937. Flush extra amortization~.324ms/step cannot explain remaining attention gap. Scalar-only is a diagnostic approximation, not a production replacement or a quality-certified codec.

## Remote environment and cleanup

A12 selected after A10 became busy and A11 had resident jobs. Remote repo `/homes/ruoxi07/repo/sglang-vqplus-kernel`; output `/homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage`. Trained book `/homes/ruoxi07/experiments/loopvq/2026-11-26-residual/artifacts/residual-ft`; model `/homes/ruoxi07/experiments/loopspec/tp2-20261106/models/Ouro-2.6B-Thinking`.

A100-SXM4-80GB,driver570.195.03,Torch2.14.1+cu130,Triton3.8.0,CUDA13,GCC13. ServicePython `/homes/ruoxi07/experiments/loopvq/2026-11-23-reproduction/envs/sglang-loopvq-test/bin/python`; PYTHONPATH repo/python,CC=/usr/bin/gcc-13,CXX=/usr/bin/g++-13,MAX_JOBS2,task extensions-a12. Capture usedTorch2.8cu128/Transformers4.56.2. GPU clocks were not locked; idle210,warmup1140,sustained service1410MHz observed.

All full-serving runs used GPUs2,3 UUID70859fd0-1839-71a2-2ad4-76f495f0d055 and39c78864-d732-8bfa-43f0-8180d77e086c,TP2,optimized,triton,pytorch,page8,max-running16,max-total65536,max-prefill16384,context32768,disable radix,chunkedprefill-1,fullgraphs1/2/4/8/16,prefillgraphsdisabled;BF16 maxsplits8,VQ48. Microtests usedGPU1,correctnessGPU0. No standalone performance ran during final service timing.

Final server parent3501172 and workers3502205/3502295 stopped after all clients passed; server exit0,processes absent, nvidia-smi compute-process list empty. No remaining GPU jobs for this task. Persistent SSH sessions intentionally retained: X5 exec90355/socket/tmp/vqpage-20261126-x5.sock; A12 exec24610/socket/tmp/vqpage-20261126-a12.sock; A10exec6622 andA11exec78913 also retained. Recheck liveness before reuse; do not assume current hardware occupancy.

## Preserved work and next action

Unrelated untracked reproduction evidence/tp2/snapshots preserved. LoopVQ training repo's pre-existing experiments/residual_ablation/record.md edit untouched. No model weights, codebooks or large snapshots committed. All source edits local, synchronization by authorized Git push/pull only.

No required implementation/verification work remains in this budget. Future first experiment: obtain authorized performance-counter access on comparableA100, use existing diagnose_paged.py --profile-one page_head_auto/materialized_serving for B16/N2048, inspect supported NCU sections and collect memory/scheduler/warp/source evidence before new kernel edits. New format or lower-bit residual work additionally needs scope and quality criteria. Do not resume canceled broad accuracy evaluation implicitly.
