# Cache policy ablations

Implementation authorization: `APPROVED_PLAN_ID=PLAN-CS-20261117-R1`,
`APPROVED_ROUND=3`. The approved file allowlist covers the two new compilers,
Context IR/wire/occurrence/recovery, frontend admission, Radix selection,
scheduler/PD integration, dedicated tests and BCP runner, and this document.
Existing policies retain their wire formats and execution behavior.

Status: CPU implementation checkpoint; real-model numerical and BCP acceptance
remain pending. This is not a claim of production or ablation validation.

`text_drop_with_pe` retains full-history position effects but recomputes the
active text at the latest effective Drop. C records carry the position used to
compute each hidden state; TOKEN records carry the stored K position. Retry
matching may ignore only the latter. Historical deleted text is provenance,
not a KV-bearing Radix node or a recoverable attention hole.

`message_drop_without_repos` recomputes the full logical history on an effective
Drop. Survivors have compact positions; removed tokens retain their position
immediately before their first effective Drop. Historical visibility expires
after each trigger. Only exact matching is permitted, including for cache
publication. Dropped history can have resident KV and be recovered.

Both policies have isolated cache namespaces. Events run after their trigger;
Drop precedes Reposition at the same boundary. Insert and hybrid Linear/Mamba
are excluded for these new policies. The existing policies are unaffected.

Validation requires independent CPU semantic oracles, numerical comparisons,
and five frozen BCP cases with rolling keep-12 tool Drop. Text policy adds
96 Ki-token Reposition. Both policies run ordinary and PD deployments (20 full
trajectories), using generated assistant turns and recorded real tool feedback.
Cold/hot/retry/eviction diagnostics use identical payloads and separate fixed
continuations; formal trajectories use their recorded generation budgets.

## Interface and compilation

```json
{"drop_rule": "text_drop_with_pe", "drop_message": {"2": [1]}, "reposition": [3]}
```

```json
{"drop_rule": "message_drop_without_repos", "drop_message": {"3": [1]}}
```

Both receive the complete messages, including dropped messages. Nonempty
`reposition` is rejected for the second rule. Existing object-valued rules and
top-level `drop_message` without a string rule keep their original semantics.

The text compiler first obtains the existing full-history position tape. It
projects the final active tokens, folds all rotations strictly before the last
effective Drop into the new compute positions, and keeps the remaining R
transitions with execution-coordinate boundaries. A same-boundary R remains
after the reset. Every real token has `[C, compute_position, -1]` immediately
before `[TOKEN, token_id, stored_position]`; C has no physical KV slot. Prior
deleted contents and irrelevant computation history are absent from the key.

The frozen-death compiler uses visibility runs and a weighted Fenwick tree.
Runs with the same death boundary are queried before any of them are removed.
For each token `i`, its position is `i` minus the count of preceding tokens
that died strictly earlier. Survivors have infinite death. Work is
`O(N + R log R)`, where R counts visibility runs, without a token-by-event loop.
All tokens are born in one execution stage at those recomputed positions.

Transport versions 9 (tensor IPC) and 10 (binary JSON) carry `cache_policy` and
an execution-to-full provenance map. Generated tokens have provenance `-1`.
Old versions 3/4/7/8 omit the new fields. Generated-key append and retraction
both preserve C records. Cache namespaces separate the two policies from each
other and from all legacy/native paths.

Exact-only matching still returns position, Full residency, holes and SWA
residency. Recovery uses the existing general occurrence reader graph for the
new policies, splitting at nonconsecutive compute positions. It does not use
`searchsorted` on the frozen-death position sequence. Final PD transport sends
only active KV; its signature includes policy and final decode cursor.

An all-dropped execution is rejected before scheduling: at least one active
token is required for the next-token distribution. The full native template
usually supplies an undropped generation prompt, but the compiler does not
silently invent one.

## Current validation evidence

Local Python: `/opt/anaconda3/envs/myenv/bin/python`, PyTorch 2.10.0, macOS CPU.
The combined focused suite completed with **505 passed, 31 skipped**. Skips
include Linux runtime and opt-in CUDA cases. New independent tests cover 800
random histories, 1,200 multilayer prefix-identity histories, random Full/SWA
holes, native ragged gather selection, generated append/retraction identity,
wire corruption, legacy schema isolation and PD metadata. Existing IR, rules,
Insert, recovery, usage, occurrence ownership, decode, raw storage and staged
plan tests are included. New files pass Ruff; `git diff --check` passes.

The standalone transfer suite additionally encountered a local missing
`orjson` dependency when importing native SRT; that environment limitation is
not counted as a passed check. Real allocator/frontend tests require Linux.

Remote read-only preflight found occupied GPUs on all three configured hosts.
BUS-2's existing checkout has unrelated changes and a missing object alternate;
it must not be repaired or overwritten by this task. Zhangyudong's system disk
had about 156 MB free. BUS-1 has the BCP dataset and expanded BF16 GPT-OSS-120B
weights, but its GPUs need an available allocation before model validation.

Runner entry points (use a new output directory for every invocation):

```sh
python benchmark/context_system/cache_policy_bcp.py preflight \
  --dataset DATASET --model BF16_MODEL --output PREFLIGHT
python benchmark/context_system/cache_policy_bcp.py run \
  --dataset DATASET --model BF16_MODEL --frozen PREFLIGHT/frozen.json \
  --policy text_drop_with_pe --url SERVER/v1/chat/completions \
  --gpu-mapping 0,1,2,3 --output RUN
```

Run both policies on ordinary and PD deployments; PD additionally takes
`--prefill-url` and `--bootstrap-port`. Preflight rejects insufficient event
coverage. Formal runs also reject insufficient actual coverage, shortened
generation, modified data and template mismatches. Snapshot request files are
saved for first Drop, first R, multiple Drops and final-turn diagnostics.

BUS-1 Linux tests on `568428dacf` completed with **30 passed, 12 skipped**,
including the real UnifiedRadixCache/allocator and public schema. The frozen
BF16 GPT-OSS tokenizer cohort is **223, 231, 238, 775, 844**. Original candidates
771/778/792 failed the required 96K coverage; replacements were the first
qualifying case IDs in ascending numeric order, selected using input histories
only. All five pass incremental historical-boundary checks, have 27–32 Drop
events, one R at 98,562–100,601 positions, and a Drop after R. Their final full
inputs contain 114,172–122,699 tokens. No model outputs informed this selection.

BUS-1 single-thread timings on those full inputs (10 samples after two warmups):

| Policy | Compile median range | Key construction median range |
|---|---:|---:|
| Original message Drop + R | 25.3–29.6 ms | 3.42–3.77 ms |
| Text Drop with PE | 31.6–36.7 ms | 1.45–1.69 ms |
| Message Drop without repos | 29.1–31.9 ms | 3.75–4.14 ms |

These measure compilation from already-rendered native provenance; they exclude
tokenization, JIT warmup, inference and network time. Text mode executes only
40,294–43,971 tokens in these final snapshots. This shows a modest compilation
overhead, not identical latency to the old compiler.

The expanded Linux regression uncovered a legacy fixture missing
`swa_evict_floor`. Running its exact test on the original `3d1f6734e1` worktree
reproduced the same failure. It is recorded separately, not patched as part of
these policies. An admission mock regression introduced by this change was
fixed by restricting new policy inspection to wire versions 9/10.
At `297ff16c2e`, that Linux regression finished with **196 passed, 33 skipped,
1 pre-existing failure**. The remaining failure is the same missing fixture
field reproduced on the baseline checkout.

Round 2 adds a bounded independent real-model oracle that uses native model
projections with independent FP32 attention and complex RoPE rotation, without
reading the production compiler or KV pool. Its rotation math passes CPU checks
for both styles and policies; its actual model test is still GPU-gated. It is
not a scalable full-BCP oracle. Evidence acceptance tests require all four real
run reports, frozen numerical limits, per-token logits, layer/read validation,
actual retrigger for A and zero retrigger for B. Missing evidence skips tests;
it never constitutes acceptance. Run benchmark tests separately from registered
tests because their approved filenames share a Python module basename.

Round 3 adds `simulate_compact` in
`test/registered/context_system/cache_policy_reference.py:119`: independent
flat-array event replay with linear memory, compared against the original
sequential reader graph over 200 randomized histories. Its 128K synthetic
geometry tests check final boundary visibility without allocating all reader
rows. `chunked_reference_attention_forward` in the same file at line 185
uses tiled FP32 online softmax and bounded visibility scratch space. It matches
the row-wise oracle and a separate complex-number calculation on CPU for both
policies, both RoPE styles, Full/SWA attention, sinks and soft caps. Runtime
tests locally finish with **30 passed, 18 skipped**. This supplies a bounded
memory attention reference, but its real-model/TP driver, GPU execution and
BCP diagnostics remain unvalidated; arithmetic for Full attention is still
quadratic.

Additional read-only resource preflight on `Tengxuan-BUS-2` found eight A800
80GB GPUs using 78,944–79,250 MiB each, with their compute processes invisible
in the current container. System disk free space was about 8.7GB. No existing
process was stopped, and no model validation was launched there.

Still required before acceptance: CUDA validation, independent full-model
numerical execution (including the BCP reference/diagnostic driver),
fixed-continuation cold/hot/retrigger/partial-eviction ordinary/PD comparison,
and all 20 complete BCP trajectories.
