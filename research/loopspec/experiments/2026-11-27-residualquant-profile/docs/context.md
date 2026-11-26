# Continuity

Active goal: paper/code research, same-stage NCU attribution and up to20 newly authorized optimization rounds. Current count0/20; no new kernel or harness edits yet. Prior two completed10-round records unchanged. Source baseline7833b357a8d4554f90cd4eba309b64282c0b83ce on loopspec-test. Local root /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/sglang. See draft.md and plan.md for contract and gates.

Skills applied: loopspec-code-change, x5-persistent-session, x5-loopspec-experiment, loop-transformer-research, ncu-report-skill, PDF reading workflow. User overrides default10-round limit with20 and specifies BUS hosts.

Reference checkout /Users/x.puppet/Desktop/Undergraduate/Project/LoopSpec/ResidualQuant, clean f4c3cccb76cb6df37eb6233fa92c7aaa054d6695. Research provenance: arxiv2610.10381v2 and official repo; numeric paths differ, no speed claim transferred.

Persistent SSH: BUS1 exec29655/socket/tmp/vqrq-20261127-bus1.sock; BUS2 exec36805/socket/tmp/vqrq-20261127-bus2.sock; Teng exec61593/socket/tmp/vqrq-20261127-teng.sock. BUS2 sudo/root verified, NCU2023.2.2 CUDA12.2, all8 A800 idle at probe. Do not close sessions. Do not record authentication secrets. Teng onlyGPU4 idle at probe; others occupied. BUS1 idle but BUS2 selected for existing CUDA and verified sudo.

Previous A12 data remains /homes/ruoxi07/experiments/loopvq/2026-11-26-vqpage/a12-inputs-2k (plus1024/4096/8192), book /homes/ruoxi07/experiments/loopvq/2026-11-26-residual/artifacts/residual-ft. A12 exec24610/socket/tmp/vqpage-20261126-a12.sock via X5 exec90355/socket/tmp/vqpage-20261126-x5.sock verified alive this turn. Data transfer permitted over verified multiplex channels; source sync exclusively Git. Prior A12 Torchcu130 cannot be assumed compatible with BUSdriver535.

Next action: inspect BUS2 existing Python environment, prove NCU access with existing harness or task-local counted replay, synchronize data/source, collect baseline before choosing implementation.
