# R14–R20 final disposition and follow-up plan

All20 optimization rounds have been used. R19/R20 passed kernel tests but failed the predeclared wide full-model greedy gate. Default dispatch was restored in `dedf55fff` to accepted R13; experimental adaptive code remains explicit opt-in. No R21 is authorized. [Continuation review](continuation-review.md) supersedes the historical status below. Restoration validation, all matched resource/output-budget scenarios, evidence archival and own-service cleanup are complete. The performance/generalization objective remains unmet; no new optimization is authorized.

## Completed validation sequence

1. Frozen R13 A/B/A wide greedy against its saved reference: completed, all cases pass.
2. R20 original lifecycle: pass; wide B32/N1024: fail. Save failed outputs; do not run it as an accepted throughput candidate.
3. Rollback source: compare retained payload/encoder bytes to R13, check disabled adaptive default; run unchanged `run_isolated.py check_paged.py --layout 2 --group-size 0` and fresh full-server lifecycle/wide greedy. Run B16/2K timing only after GPU boundary checks finish.
4. Match BF16 maxrunning64/KV65536/context32768 for the 2K reference. Earlier R13/BF16 wide1K/B32,512/B64,8K/B1/4 completed.
5. Smaller capacity: KV32768/context16384/maxrunning16/graphs1,2,4,8,16, same GPU2/3, VQ retained vs BF16. Existing `check_serving.py --suite timing --input-length 1024 --batches 1 4 16 --output-length 128`, then B16 with `--output-length 32` and `256`; three repeats each. Existing lifecycle check first. No profiler or other GPU jobs during pure timing.
6. Archive source identities, raw SSE, batch/Graph evidence, failures, sanitizer/resources/SASS and summaries. Stop only own test services; retain persistent SSH. Commit only task source/docs on loopspec-test, never promote formal loopspec.

## First future experiment (requires a new optimization budget)

Use an explicit logical reduction topology while varying physical CTA size. Candidate location: `python/sglang/kernels/ops/loopvq/residual_attention_adaptive_cuda.py::source_for/prepare`; retained control: `residual_attention_payload_cuda.py` and original fallback in `ResidualAttention.forward`. Preserve each accumulator's token order, logical warp reduction and split merge order. Start with256 physical threads emulating the512-thread control's logical groups. Resource-derived physical scheduling must not silently redefine floating-point grouping. Performance is unknown; extra logical-state storage can erase occupancy benefits.

Before performance, add task-local bit-pattern diagnostics against frozen R13 for output/LSE and partials with identical logical splits. Keep independent tolerance/finite checks unchanged. Existing real-input harness commands from the target repository (RUN is the existing remote `/dev/shm/wangruoxi-vqrq-20261127`, TASK is this task's absolute directory):

```sh
CUDA_VISIBLE_DEVICES=0 "$RUN/serve-env/bin/python" "$TASK/integration_probe.py" \
  --snapshot "$RUN/2k" --book "$RUN/books" --output "$RUN/future-topology-2k.json" \
  --batches 16 --loops 0 1 2 3
CUDA_VISIBLE_DEVICES=0 "$RUN/serve-env/bin/python" "$TASK/integration_probe.py" \
  --snapshot "$RUN/1024" --book "$RUN/books" --output "$RUN/future-topology-1k.json" \
  --batches 32 --loops 0 --repeat-requests
```

These harnesses exist; the new topology and bit-pattern diagnostic are proposed, not implemented. Follow with derived2080, all existing boundary/uneven/Graph/padding/memcheck checks, then unchanged full-service lifecycle and `wide_greedy.py --reference "$RUN/serving-r13-wide/wide-greedy"`. Reject any contract failure before acceptance timing. Final promotion still needs ≥5%B16 full-serving gain and ≤5% regression across matched validated scenarios, using complete service decode throughput,3repeats/warmup and original scheduler proof. Expand other GPUs only after compiling/querying their actual resources and validating complete backend support; synthetic occupancy invariants cannot stand in for hardware validation.

Essential unknowns: exact logit-level cause of R19/R20 greedy divergence; whether topology-preserving physical remapping saves enough resources; real performance on any GPU other than A800; hardware cache/warp counters currently unavailable. NCU is helpful but not a prerequisite for the proposed controlled experiment. Post-RoPE retraining, altered rounding, and relaxed accumulation are outside the current contract.

---

The following is the historical plan and R13 acceptance record, retained for provenance.

# Generalization requirement (user steering during R17/R18)

No final B16/108-SM-only policy. Fixed splits are ablations. Select geometry/splits from actual device resources and shape; preserve unverified-device fallback. Validate B1/2/3/4/8/16/32/64, short/long and uneven lengths, graph padding, and capacity limits where memory permits. B>16 fixtures explicitly repeat verified captured requests. Extend matched full-service scenarios to B32/N1K and B64/N512 with matching max-running/graph/KV limits. Cross-GPU portability is a design requirement; performance remains unverified without hardware measurement. Current accessible measured hardware is A800/SM80. No relaxed trained or numerical contract.

# Continuation plan: rounds14–20

User requests all remaining7 rounds, targeting B16’s11.4%full-serving deficit toBF16. Current round14 compact-grid experiment is specified in record.md. KeepR13/source4743e9430 as immutable comparison, preserve trainedpre-RoPE/INT4/FP32/BF16contract. No NCU prerequisite. Screen with CUDA13/int64 real-input matched ablations, reject failures before performance; reserve final integration/guard and fullservingvalidation withinround20. Adaptive nextcandidate depends onmeasuredresults; eachsource/test/behaviorchangegetsnextnumberand six-partplan. Full-servinggo/no-go usesunchangedcheck_serving.py lifecycle/tokenIDs andB1/2/4/8/16 timing, sameGPU2/3/model/input/seed/environment. Historicalcompletedplanbelowisretainedasevidence, notcurrentstoppingrule.

# Executable plan and acceptance record

Status: **complete for the retained R13 implementation**, after 13 of the newly authorized maximum20 rounds. Runtime source `4743e9430`; original-production control `56fbb58de`. Earlier budgets and rejected candidates remain in `../record.md`. The best measured candidate passes the agreed kernel and full-serving gates; it is not a global-optimum claim and B16 throughput remains below native BF16.

## Contract

Ouro-2.6B-Thinking, TP2, 48 layers ×4 loops, local H8/D128, SM80 A800108SM. Preserve trained pre-RoPE codebooks, UINT8 joint indices, INT4/group32 FP16 residuals, BF16 rounding points, FP32 accumulation, HP prefix8/recent64, page8, API/Graph behavior. No training-format change, precision relaxation or broad quality evaluation.

## Concrete changes

1. `python/sglang/kernels/ops/loopvq/residual_attention_payload_cuda.py`: `pair8` shares the low/high RoPE-pair page address and metadata ownership; `pv16` reads one packed index/residual unit and one scale per16 output channels. Exact finite FP16×INT4 reconstruction FMA reduces live registers while preserving the original RoPE and dot summation. Use512 threads, minimum3 resident blocks, original register-resident decode/consume path. Do not stage complete codebooks in shared memory: measured bank-aware asynchronous staging regressed.
2. `_split_policy` and constant `PAYLOAD_SPLITS[13][33]`: precompute occupancy-based partition policy once; GPU lengths select a429-byte table, with original CAP/short-length clamps. No host length synchronization. Exhaustive425997 policy cases match R12 exactly.
3. `residual_attention.py::ResidualAttention`: prepare once and query actual occupancy capacity; select payload only for measured layout2/default group/dynamic splits/H8/108SM/capacity324/B4..16. Preserve original small-batch and unsupported configurations. `payload=False` is the actual production control, not the older microbench's R1-only label.
4. Task-only integration/stage/payload/shared probes retain independent oracles and raw samples. `analyze_serving.py` reads existing evidence; it does not change the evaluator.

## Correctness and measurement gates

- Output atol .002/rtol .01; LSE atol/rtol2e-5; encoder unchanged. Exact reconstruction property check:131858432 finite cases.
- Original `check_paged.py`:100 boundary/uneven-length/LSE cases plus dynamic Graph and padding, passed. Compute Sanitizer memcheck passed with0 errors for R12 and R13.
- `integration_probe.py`: original/candidate/native paired warmed Graph timing,30 samples×32 inner calls, B1/2/3/4/5/8/15/16 and N1K/2K/4K/8K; all4loops at2K. Actual B2 grouped dispatch included. R13 B16/N2K125.121μs vs221.404μs original; B1/2/3 within1%. CUDA12/13 candidate resources both40 registers and0 stack/local for both LOC widths.
- Full serving on the same BUS2 GPU2/3 and CUDA13 environment: unchanged `../2026-11-26-vqplus-residual/check_serving.py`, input2048/output128, B1/2/4/8/16,3 repeats after warmup. All lifecycle/greedy-ID/abort/reuse checks passed. Each sample verifies full target batch and CUDA Graph from scheduler logs. Candidate B16 throughput371.748 vs293.573 token/s (+26.63%); BF16 reference419.645 token/s. All acceptance thresholds met: ≥10% kernel gain, no>5% small-batch regression, ≥5% full-serving gain.
- Serving traces are CUPTI activities, not NCU counters. Both ranks show1920 historical-reader calls for10steps×192layers. Profile throughput is excluded from acceptance. Full traces use growing lengths>2048, int64 LOC and CUDA13; do not mix their absolute times with fixedN2048 CUDA12 microbenchmarks.

## Reproduction and evidence

Task workspace is this experiment directory. BUS2 temporary RUN `/dev/shm/wangruoxi-vqrq-20261127`; micro Python `RUN/env/bin/python`, serving Python `RUN/serve-env/bin/python`. Remote candidate worktree `/share/wangruoxi/repo/sglang-vqplus-r13`, baseline `/share/wangruoxi/repo/sglang-vqplus-baseline`. Exact server arguments are in each archived server log/identity; source code is synchronized only through Git. Model revision/hash and book hashes are in evidence/compat and input manifests. Reproduce the summary from repository root:

```sh
python3 research/loopspec/experiments/2026-11-27-residualquant-profile/analyze_serving.py \
  research/loopspec/experiments/2026-11-27-residualquant-profile/evidence \
  --output research/loopspec/experiments/2026-11-27-residualquant-profile/evidence/serving-analysis.json
```

Raw artifacts: `evidence/r1`–`r13`, `evidence/serving-{baseline,candidate,bf16}`, `evidence/compat`; acceptance hashes in `evidence/acceptance-manifest.json`. Large evidence is deliberately not committed. All task test servers stopped and GPUs released; persistent SSH masters retained. Formal `loopspec` branch unchanged.

## Recommended first follow-up and unresolved evidence

First obtain a host/container with working GPU performance counters. Root probes on all three named BUS hosts failed ERR_NVGPUCTRPERM because container capabilities lack SYS_ADMIN; there are no NCU reports to interpret. Do not change host driver settings as an incidental task. Re-run the exact existing counter probe documented in `profile/20261127-bus2-counter-probe/harness/README.md`, check installed sections, then collect matched original/payload source counters with the same input and partition controls. A diagnostic capture command after counter access is available (RUN and TASK denote the absolute paths above):

```sh
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name-base function \
  --kernel-name 'regex:payload_stage1' --launch-count 1 --set full \
  --export "$RUN/payload-full" \
  "$RUN/env/bin/python" "$TASK/integration_probe.py" \
  --snapshot "$RUN/2k" --book "$RUN/books" \
  --output "$RUN/payload-ncu-harness.json" --batches 16 --loops 0 --repeats 1
```

This captures a diagnostic launch, not acceptance timing. Verify the selected kernel and collect original-reader/control reports separately; use unprofiled warmed timing for promotion. Prioritize QK centroid-load sectors versus RoPE/conversion instruction dependencies and eligible warps; inspect PV metadata and any local-memory traffic. A further mapping/worklist candidate must target the measured dominant cost, be recorded as a new round, preserve independent tests, and exceed the same promotion gates. Reject superficial codebook/shared-memory changes without a measured benefit. Post-RoPE retraining or changed accumulation is a different algorithmic contract, not a free optimization of the current representation.
