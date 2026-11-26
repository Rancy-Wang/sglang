# VQ+ redesign continuity

Updated: 2026-11-26 (Asia/Singapore).

## Active implementation update

The user subsequently authorized a new maximum of 10 rounds: preserve existing training and pre-RoPE codec, implement the page path, and run four-way A100 controls. Post-RoPE/INT2/retraining proposals in the design plan are now out of scope. Follow `../paged-record.md` for the new budget and current execution evidence. The previous 10/10 round record remains historical.

A10 GPU4/5 were verified idle A100-SXM4-80GB at initial inspection. X5 exec session90355/socket `/tmp/vqpage-20261126-x5.sock`; A10 session6622/socket `/tmp/vqpage-20261126-a10.sock`. A10 shell PID2497326, account ruoxi07; jump via the live X5 master. Reuse these handles and recheck liveness/resources. Target checkout `/homes/ruoxi07/repo/sglang-vqplus-kernel`, clean `loopspec-test`, baseline2eb4f0d38.

Current execution host A12, session24610/socket `/tmp/vqpage-20261126-a12.sock`, shell3411470, through live X5 session90355. A10 skipped busy, A11 resident tasks. Output `/homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage`. Rounds1..5 complete, correctness passed but no large-batch improvement; memcheck round2 zeroerrors, racecheck round3 zerohazards. NCU2026.3.1 blocked ERR_NVGPUCTRPERM. Round6 f58bd57e1 integrated direct page encoders/pool/CLI/backend, passed bitwise codec/legacy codec/>2GiB/CLI tests; new Triton reader compile failed due2D max_contiguous hint. CUDA/native audit passed: B16/N2048 current226.40us,page233.27us,native serving max8 actual5splits102.96us,fixed48native107.06us. Round7 source9d6e92a330f36451e99cbe0635b759ade83359ea changes hint to[1,4], clean remote synced. Jobs just launched: boundary3460354/GPU0(r7-boundary), diagnostics3460358/GPU1(r7-timing-2k; current/page_only/triton_page/triton_scalar/scalar_int4/materialized_serving), BF16 TP2 server wrapper3460362/GPU2,3/port30664(r7-bf16-server.log). Server readiness and child PID NOT yet checked. GPU2 UUID70859fd0-1839-71a2-2ad4-76f495f0d055; GPU3 UUID39c78864-d732-8bfa-43f0-8180d77e086c. Use this SAMEpair for all full-serving baselines/candidates. Only initialization concurrent with isolated kernel tests; wait for other benchmark jobs before final paired serving measurements. Environment CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 TORCH_CUDA_ARCH_LIST=8.0, task extensions-a12. Three later rounds remain8..10. Token layout stays default; page flags requireINT4/global, no formalbranch promotion. Recheck all exits/PIDs. Read paged-record.md for commands, failures and criteria.

## Original design-review scope and authorization

The user requested a deep review of the linked LoopVQ chat and a more efficient VQ+ attention design, especially for large batch decode. This turn is analysis/planning only. No inference/training source changes, GPU runs, commits, pushes, or accuracy evaluation restarts were performed. Existing implementation budget remains 10/10; this design review does not reset it.

## Target and working tree

- Source: `/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang`, branch `loopspec-test`, HEAD `2eb4f0d380effc82cd4008fcf8096276098cdcbe`; tested implementation `4f9f8398d0955824e7c64dfcf63a826bbb70dc05`.
- Existing untracked data preserved: `research/loopspec/experiments/2026-11-23-loopvq-reproduction/evidence/tp2/snapshots/`.
- Training reference: `/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/LoopVQ`, HEAD `dc3af341ba4593f580272470da803a5db839a423`; pre-existing modification to `experiments/residual_ablation/record.md` preserved.
- Chat: `01a1f8df-2e9d-74f3-b471-0ec6577f4ee2`, host `remote-control:env_e_6a68ec597c9c8325a8035cb714e0bc83`, title `克隆 LoopVQ 并梳理代码`. Read relevant original residual request, training-scope correction, VQ+ plan and final performance reports. No need to repeat unrelated canceled accuracy-job history.

## Contract

Ouro-2.6B-Thinking, full 4 loops × 48 base layers; A100/SM80, TP2, local 8 heads, MHA single-token decode, D128, page8, HP prefix8/recent64. Index codes jointly encode 4 loops × 4 channels with C256. INT4 residuals and group32 FP16 scales are independent per loop. K is pre-RoPE; current served K/V include BF16 composition rounding. Current supported context <=32768; prefix reuse, chunked prefill, speculative decode and offload remain unsupported.

## Evidence

- `../results.md`, `../results.json`, `../record.md`: historical measurements and implementation rounds.
- `analysis/source-manifest.json`: SHA256 of inspected source/harness/report files; recheck before implementation.
- `analysis/algebra-check.json`: CPU NumPy synthetic algebra verification only, seed20261126. K-LUT and V histogram additive identities agree within ~7e-16. Naive pre-RoPE position-independent LUT, separate softmax, and skipping BF16 composition rounding are not generally equivalent. No GPU speed or real-model quality claim.
- Raw existing results: `/Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/reports/vqplus-20261126/`.
- Historical target host paths and exact server/client commands are in `r10-global-command.txt` and `r10-global-client-command.txt` there. Remote environment/resource availability was not rechecked in this turn.

## Decisions

1. Diagnose with real Q/K/V snapshots and strong native BF16 attention, plus direct scalar INT4 as a required quality/cost competitor.
2. First implementation candidate A: page/head/group32 packed layout, cooperative loading, bounded tile pipeline, compact valid-task scheduling, old codec and rounding semantics.
3. Research candidate B: post-RoPE K encoding enables query-dependent center-logit LUT plus packed residual dot. Initial V path stays direct gather; histogram aggregation is a separate conditional optimization.
4. New linear decomposition requires explicit rounding schema and quality validation. Do not silently replace `BF16(C+r)` with separate accumulations.
5. Explore INT2 residuals only after isolating format changes; allow asymmetric K/V codecs. Keep original Fisher initialization + final-output KL training framework, changing one variable at a time.
6. Do not assume batch MHA shares KV or that INT4 storage enables a mixed BF16×INT4 Tensor Core instruction. No Hopper/Blackwell-specific kernel prescription for SM80.

## Unknowns and next action

Unknown: real snapshot completeness, current NCU permission, true hardware bottleneck, direct-INT4 quality comparison, post-RoPE quality loss, viable INT2 precision, code histogram distribution, accepted quality budget.

Next action after user starts implementation: follow remote-session/project instructions, freeze environment and capture/replay real B16/H8/N≈2K data. First compare current VQ+ with materialized-equivalent native BF16 as diagnostic only, then add scalar INT4 and candidate A. Full measured candidate costs include conversion, LUT build, scheduling and flush. The plan's proposed acceptance thresholds are not measured outcomes or preapproved quality tolerances.

Deliverables: [draft.md](draft.md), [plan.md](plan.md), and the analysis JSON files. All are task-local; no tracked KDA reference files were changed.
