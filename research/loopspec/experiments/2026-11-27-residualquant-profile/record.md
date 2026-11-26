# Additional optimization budget

User: “我再批准20轮优化”. Baseline7833b357a8d4554f90cd4eba309b64282c0b83ce, branchloopspec-test. Earlier10/10 batches retained. Current1/20 started; no production kernel changes in this round.

## Round1/20 — isolated measurement entry

1. Purpose: replay existing production CUDA/Triton functions on BUS2 without importing unrelated model/server dependencies; establish counter access and real-input baseline.
2. Steps: add a thin task-local bootstrap for existing diagnose_paged/check_paged/check_residual scripts; expose real package paths; provide only hardware-query utilities needed by decode_attention; run unchanged harness and unchanged kernels. Add reproducible ncu_report metric/source extraction, not implementation modifications.
3. Files: task-local run_isolated.py and analyze_ncu.py. No production math/dispatch edits.
4. Success: real snapshot/book hashes validated by existing harness, oracle pass at unchanged .002/.01, NCU report collected and parsed. No performance improvement claimed for measurement infrastructure.
5. Risks: bypassing package initialization could alter environment; record all supplied utilities, use real environ module, assert SM80/CUDA, and compare source kernel functions unchanged. CUDA12.1Torch/Triton3.1 may expose source compatibility gaps; fix in a separately counted round if required. NCUroot may still lack container capabilities.
6. Validation: Python compile, source review, Git push/pull, isolated real B16/N2048 baseline and boundary tests; sameGPU matched native/page profiles; full/source sections supported by NCU2023, no fabricated PM sampling. Store unique profile/<run> dirs under task workspace with harness/reports/analysis. Include versions and GPU UUID in logs.

Environment observation: existing akv environment uses Torch2.13cu130 and cannot initialize on driver535. New isolated Torch2.5.1cu121 environment is being prepared without modifying existing environments. BUS2 /share/wangruoxi currently resolves to overlay with only4.5GiB free; install paused before completion to choose mounted storage.

Before first GPU validation, extend this same measurement-infrastructure candidate with counter_probe.cu: a checked, tiny CUDA launch to test NCU counter permission independently of the ongoing Python download. It is not a performance benchmark. Compile locally authored source remotely through Git using nvcc sm80; compare non-root/root NCU basic report status. Storage selected: task-owned /dev/shm/wangruoxi-vqrq-20261127 (executable tmpfs,381GiB free, host816GiB available), since persistent shared mounts are read-only; retain results locally before teardown. No system changes.

Round1 counter result: a95c8b1014e39a1ddfaf4a5192798ac6aaa7cb9f; nvcc probe succeeds, root NCU exits1 ERR_NVGPUCTRPERM on BUS1 GPU0, BUS2 GPU0 and Teng GPU4. No report/metric was collected. All root CapEff0xa80c65fb omit CAP_SYS_ADMIN. Official NVIDIA documentation requires host enablement or container SYS_ADMIN for driver535. Counter failure is infrastructure, not kernel failure. Local Python syntax passed; full real replay pending dependency/data setup. Measurement bootstrap retained.

## Round2/20 — stage cycles and matched arithmetic control (planned before edits)

1. Purpose: obtain stage-level evidence despite unavailable hardware counters. Distinguish compressed reconstruction/RoPE cost from the existing thread mapping/softmax/reduction cost.
2. Steps: create task-only full/scalar-diagnostic/dense-matched CUDA readers using the unchanged production source; dense control reads materialized post-RoPE BF16 and removes RoPE, preserving QK/PV ownership/reductions. Add an optional separate instrumented compilation writing clock64 timestamps at existing QK/softmax/PV synchronization boundaries. Compare instrumented vs uninstrumented graph timing; report overhead, not NCU stalls or traffic.
3. Scope: task-local stage_profile.py only. No production dispatch/codec changes. Real trained-codebook input and existing oracle/timing helpers reused. Preserve the original native Triton reader as an additional whole-attention control.
4. Success: full and dense controls satisfy unchanged output tolerance, scalar against its own independent reconstruction oracle; stage cycles from active CTAs only. Timing speedups from uninstrumented readers only. If instrumentation changes latency materially (>10%), treat cycle fractions as perturbative evidence and require independent ablations before optimization conclusions.
5. Risks: clock/global-store overhead, register allocation differences, cycle counts are CTA residence time not additive global latency; scalar differs in numerical values and is not a deployment candidate. Same-source dense control is not the production BF16 baseline. Setup/materialization excluded and explicitly labeled.
6. Validation: local syntax/generated-source checks, Git synchronization, real B16/N2048 layer23loop0 output checks, paired warmed graphs and repeated CTA-clock samples, compiled resource extraction. Extend layers/loops after a useful hypothesis emerges. Primary performance comparison remains actual native serving kernel and final full serving; no speed claim from instrumentation alone.
