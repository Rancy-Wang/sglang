> Continuation status: all20 rounds are now consumed; R19/R20 failed full-model wide greedy and default dispatch was restored to R13. See [R14–R20 review](continuation-review.md) and [current plan](plan.md) for the latest validation status. The text below records the earlier R13 stage.

# ResidualQuant / VQ+ investigation

Date: 2026-11-27. New user authorization: up to 20 additional implementation/validation rounds. Earlier completed 10-round records remain unchanged. No global-optimum claim is possible from a bounded search.

## Contract

Continue the existing Ouro-2.6B-Thinking, four-loop, TP2/local8-head, D128, page8, SM80 decode path. Preserve trained joint codebook, pre-RoPE K, INT4 residuals/group32 FP16 scales, BF16 composition boundaries, HP prefix8/recent64, API and supported shapes. Output tolerance .002/.01, LSE2e-5, codec bitwise parity, serving greedy-ID and lifecycle parity. No broad quality evaluation or new training format is implied.

## Evidence before new profiling

Target baseline 7833b357a8d4554f90cd4eba309b64282c0b83ce (previous tested implementation 7d1c54b5ab30a98ffda5b4cd5bbd0b797479b064). Prior A100 results and limitations: ../..//2026-11-26-vqplus-residual/paged-results.md. The page reader remained slower than native BF16 at B16. Hardware counters were unavailable there; source-level hypotheses were not measured bottleneck findings.

Official reference: https://arxiv.org/pdf/2610.10381 and https://github.com/heejunkim00/ResidualQuant, local read-only checkout /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/ResidualQuant at f4c3cccb76cb6df37eb6233fa92c7aaa054d6695.

ResidualQuant uses a different anchor/residual representation and numerical path. Its attention/compute.cuh assigns eight lanes per token, keeps BF16 vector query/code arithmetic, applies scale/offset to grouped dot products, pipelines packed K/V planes and streams partial softmax/PV. attention/launch.cuh bases work capacity on measured kernel occupancy. These are source observations, not performance measurements on our workload.

Our residual_attention_cuda.py stage1 reconstructs FP32 centers+INT4 residual, applies per-position RoPE and BF16 rounding, then FP32 QK; stores split scores in shared memory, performs block softmax, loads V centroids/residuals, then FP32 PV with 16 scalar output accumulators per lane. Page-head changes addressing only. Important candidates for attribution are dependent centroid loads, scalar conversions/bit extraction, QK RoPE work, PV register pressure, block barriers and split reduction. Logical compressed bytes do not determine actual cache-sector traffic.

## Fair stage correspondence

| Stage | Current VQ+ | ResidualQuant reference | Required evidence |
|---|---|---|---|
| Cache write | joint index + per-loop residual; periodic flush | anchor/residual quantization | amortized write time separately |
| Query/preparation | query staging, RoPE inside K path | prepare query rotation/live score | total launch time, separate preparation |
| QK | indexed centers + residual + position RoPE + BF16 round + FP32 dot | grouped low-bit dot and affine corrections | source-PC stalls, instruction counts, load sectors |
| Softmax | split-wide shared scores + block reductions | streaming lane/token statistics | barriers, eligible warps, dependency stalls |
| PV | indexed centers + residual + BF16 round + FP32 accumulation | packed residual restoration/BF16 pair accumulation | registers, spills, sectors, source attribution |
| Merge | FP32 split partials/LSE | FP32 partition merge, optional inverse rotation | launch and scratch cost |

Do not transplant BF16 accumulation or move RoPE/rounding across sums under the existing contract. Position-independent Q·codebook LUT is not algebraically valid for position-dependent pre-RoPE K. A V code-index histogram also changes rounding unless correction is included; assess its synchronization and temporary storage rather than assuming free codebook reuse across requests.

### A useful byte-accounting correction (derived, not measured traffic)

At D128/group32 and schedule[2,2,2,4], official cache.cuh allocates74B per K or V anchor (64B codes+8B FP8 scale/offset+2B alpha) and42B per residual (32+8+2). async_copy.cuh omits the anchor's constant2B alpha when staging: nonanchor loops read72+42=114B per K/V, final anchor loop72B. Combined K+V staged payload averages (3*228+144)/4=207B per token/head/loop. Current VQ+ reads2*(32B index+64B residual+8B scale)=208B, before centroid fetches, RoPE, HP regions, cache-sector inflation and reuse. Thus these two specific formats have nearly equal average sequential payload reads despite different persistent storage. RQ's capacity advantage alone cannot explain a same-shape kernel gap; sparse centroid traffic and decode/arithmetic schedules must be measured. RQ persistent K+V storage averages100B per loop versus VQ+160B (shared index amortized across four loops), excluding side data/codebooks. Different loop schedules/group sizes change these figures.

Source detail: RQ anchor softmax maintains independent strided token streams per8-lane group, and only merges streams after scanning pages. This avoids a block-wide softmax reduction for every page. Its PV stream is BF16-pair arithmetic, whereas our FP32 accumulation is part of the current contract. An FP32 stream implementation could borrow the ownership/barrier design while retaining precision, but register pressure and per-token rescaling are unmeasured tradeoffs.

RQ prepare.cuh::rotate_queries does use Tensor Cores: shared FP16 rotation weights and up to8 BF16 queries are processed with mma.sync m16n8k8 TF32 operands/FP32 accumulation. Both source operand precisions fit the TF32 mantissa. This is separate from the packed-history attention's BF16 SIMT path. It is a useful example of putting shared dense work on MMA; it does not establish that single-query MHA history QK/PV should use padded MMA, nor that VQ's position-dependent RoPE can be replaced by one shared query rotation.

## Hardware

User-named BUS hosts are A800-SXM4-80GB, driver535.161.08. InfiniAI-BUS-2 currently has all eight GPUs idle; sudo authenticated and root identity verified. CUDA12.2/NCU2023.2.2 installed. Root counter probes on all three user-named hosts returned ERR_NVGPUCTRPERM; container capabilities omit SYS_ADMIN. No hardware counter report exists. Use a compatible task environment and remeasure every baseline on this host. Do not alter system driver settings or touch other jobs.


Continuation: user requires rounds14–20; no counter-access wait. First hypothesis is fixed32-grid empty CTA overhead; follow with measured QK/PV instruction and scheduling ablations, then full-serving acceptance against frozen R13 and native BF16. Runtime/compiler/inputs were revalidated on BUS2.
