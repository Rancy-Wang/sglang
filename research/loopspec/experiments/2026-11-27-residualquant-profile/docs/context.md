# Continuity

Active goal: paper/code research, same-stage NCU attribution and up to20 newly authorized optimization rounds. Current count1/20 in progress: measurement bootstrap and counter probe; production kernels unchanged. Prior two completed10-round records unchanged. Source baseline7833b357a8d4554f90cd4eba309b64282c0b83ce on loopspec-test. Local root /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang. See draft.md and plan.md for contract and gates.

Skills applied: loopspec-code-change, x5-persistent-session, x5-loopspec-experiment, loop-transformer-research, ncu-report-skill, PDF reading workflow. User overrides default10-round limit with20 and specifies BUS hosts.

Reference checkout /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/ResidualQuant, clean f4c3cccb76cb6df37eb6233fa92c7aaa054d6695. Research provenance: arxiv2610.10381v2 and official repo; numeric paths differ, no speed claim transferred.

Persistent SSH: BUS1 exec29655/socket/tmp/vqrq-20261127-bus1.sock; BUS2 exec36805/socket/tmp/vqrq-20261127-bus2.sock; Teng exec61593/socket/tmp/vqrq-20261127-teng.sock. BUS2 sudo/root verified, NCU2023.2.2 CUDA12.2, all8 A800 idle at probe. Do not close sessions. Do not record authentication secrets. Teng onlyGPU4 idle at probe; others occupied. BUS1 idle but BUS2 selected for existing CUDA and verified sudo.

Previous A12 data remains /homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage/a12-inputs-2k (plus1024/4096/8192), book /homes/ruoxi07/experiments/loopvq/2026-11-26-residual/artifacts/residual-ft. A12 exec24610/socket/tmp/vqpage-20261126-a12.sock via X5 exec90355/socket/tmp/vqpage-20261126-x5.sock verified alive this turn. Data transfer permitted over verified multiplex channels; source sync exclusively Git. Prior A12 Torchcu130 cannot be assumed compatible with BUSdriver535.

Current source a95c8b1014e39a1ddfaf4a5192798ac6aaa7cb9f, synchronized to BUS2 /share/wangruoxi/repo/sglang-vqplus-kernel. All three named hosts authenticate sudo but root NCU probes return ERR_NVGPUCTRPERM, CUDA itself passes. Effective root capabilities0xa80c65fb omit CAP_SYS_ADMIN; driver535 requires host/container permission. Teng needed native rebuild due GLIBC2.31 vs transferred binary2.34. No hardware-counter report exists.

BUS2 runtime/data/extensions now under /dev/shm/wangruoxi-vqrq-20261127 (381GiB free at setup; host816GiB available). /share/public and /share/h_public read-only; overlay only4.4GiB free. Existing cu130Torch cannot run driver535. Torch2.5.1cu121 wheel downloaded with aria2; dependency download ongoing using PyPI JSON SHA256 and aria2. Do not reinstall/alter old env. Remote shell vars VQRQ_RUN/VQRQ_REPO/VQRQ_TASK point at these paths.

Local immutable inputs at task/evidence/inputs/{2k,books}; all16 real layer23 rows and both607379165-byte books SHA256 match original manifest. Other layers/lengths remain on A12. SCP uploads ongoing BUS2: exec42063 for2k,66709 forbooks. Existing A12 exec24610 write_stdin was automatically rejected (approval required, Never policy); read-only SFTP data copies through verified master succeed, no target process actions retried. Native A12 data remain intact.

Next action: complete environment/data setup; validate isolated B16/N2048 baseline and old boundary suite; if counters remain unavailable use a separately planned/counted instrumented stage-cycle reader and explicit ablations, with instrument overhead checked, to guide optimization. Keep NCU limitation visible and do not fabricate stall/traffic attribution.
