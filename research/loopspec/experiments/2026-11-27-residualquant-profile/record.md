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
