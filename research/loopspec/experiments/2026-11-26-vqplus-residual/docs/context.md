# VQ+ redesign continuity

Updated: 2026-11-26 (Asia/Singapore).

## Active implementation update

The user subsequently authorized a new maximum of 10 rounds: preserve existing training and pre-RoPE codec, implement the page path, and run four-way A100 controls. Post-RoPE/INT2/retraining proposals in the design plan are now out of scope. Follow `../paged-record.md` for the new budget and current execution evidence. The previous 10/10 round record remains historical.

A10 GPU4/5 were verified idle A100-SXM4-80GB at initial inspection. X5 exec session90355/socket `/tmp/vqpage-20261126-x5.sock`; A10 session6622/socket `/tmp/vqpage-20261126-a10.sock`. A10 shell PID2497326, account ruoxi07; jump via the live X5 master. Reuse these handles and recheck liveness/resources. Target checkout `/homes/ruoxi07/repo/sglang-vqplus-kernel`, clean `loopspec-test`, baseline2eb4f0d38.

Current host A12, persistent session24610/socket `/tmp/vqpage-20261126-a12.sock`, shell3411470 through X5 master. A10 skipped busy, A11 resident jobs. Output `/homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage`; repo `/homes/ruoxi07/repo/sglang-vqplus-kernel`. CC=/usr/bin/gcc-13,CXX=/usr/bin/g++-13,task extensions-a12,SM80. NCU2026.3.1 blocked ERR_NVGPUCTRPERM; no cache/DRAM/occupancy counter claims.

Rounds1..5 correct but no large-batch improvement. Round2 memcheck zeroerrors; round3 racecheck zerohazards. Round6 integrated direct page encode/pool/CLI/backend with bitwise parity incl>2GiB; new Triton compile hint failed. Round7 fixed hint,100boundary/LSE/graph cases passed, but Triton slower. Native BF16 diagnostic corrected from fixed48 to actual serving max8/dynamic (B16 actual5splits); old fixed48 numbers retain label.

Round8 `32a69ef16d72f6e1dcc294937715f591b5b09286` adds head-local layout2 `[page,head,token8,channels]`/CLI `page-head`, keeping group32 layout1 `page` and default0 `token`. Local syntax and all generated-source transforms passed. Both layout encoders pass bitwise parity/>2GiB; CUDA layout2 100boundary/LSE/graph tests pass. Triton boundary and real micro timing underway (wrapper3471370/GPU0 and3471372/GPU1). Last structural candidate; rounds9/10 reserved necessary fixes/acceptance.

Full-serving fixed GPUs2,3 UUID70859fd0-1839-71a2-2ad4-76f495f0d055 /39c78864-d732-8bfa-43f0-8180d77e086c. BF16 at9d6e92a33 lifecycle/timing/profile all exit0; B16/2K decode419.96/421.51/420.82tokens/s, fullgraph/active-batch verified, abort/reuse passed; own server stopped and GPUs verified free. Token VQ+ now running at32a69ef16 parent3471375/wrapper3471374,port30664,ready200; lifecycle wrapper3474659 underway. Wait for standalone GPU tests before token timing/profile; later run selected page CUDA on samepair/config and compare lifecycle outputIDs againsttoken reference. All serving: full4loop TP2,2048input/128output,HP8+64,page8,max-running16,max-total65536,max-prefill16384,context32768,optimized,triton,pytorch,disable radix,chunkedprefill-1,full decode graphs1/2/4/8/16,prefillgraphsdisabled. Native maxsplits8; VQ48. No formalbranch promotion. Raw BF16 results copying locally into evidence/paged/r7. Recheck all job exits/PIDs; never kill other tasks or close persistent SSH sessions.

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
