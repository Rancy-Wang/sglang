# Cache policy ablations

Implementation authorization: `APPROVED_PLAN_ID=PLAN-CS-20261117-R1`,
`APPROVED_ROUND=25`. The approved file allowlist covers the two new compilers,
Context IR/wire/occurrence/recovery, frontend admission, Radix selection,
scheduler/PD integration, dedicated tests and BCP runner, and this document.
Existing policies retain their wire formats and execution behavior.

Status: GPU validation in progress; complete numerical and BCP acceptance
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

On resumption, Tengxuan's independent checkout at
`/home/wangruoxi/cache-policies-r1-20261117/sglang` passed **34 CPU tests** with
**14 skipped** at `37a6543bea`. Its five-case tokenizer preflight reproduced
the frozen input hashes, model metadata fingerprints, thresholds and event
coverage. The NVIDIA PIDs map through `/proc/*/status`'s `Ngid` to a SGLang
process family and another GPU process owned by `renzhongqi`. An authorized
SIGTERM attempt failed; noninteractive sudo requires a password. No task was
terminated. Complete model execution still requires an available allocation.

Small CUDA tests fit in the remaining GPU 0 memory with the PyTorch allocator
capped at 256 MiB. The first run had **8 passed, 4 failed**; peak PyTorch
allocated/reserved memory was 849,920/2,097,152 bytes (CUDA context overhead is
additional). All failures used a zero SWA window. That fixture incorrectly
assumed self-only attention: both current and frozen native Triton enable SWA
only for positive windows, and Context recovery rejects zero. Round 4 changes
the kernel fixture to the smallest valid window, one, and explicitly tests
zero-window rejection. No production kernel or legacy policy changes are
needed for this fixture correction. The mathematical reference still tests
zero as a boundary of its own explicit visibility definition.

Still required before acceptance: CUDA validation, independent full-model
numerical execution (including the BCP reference/diagnostic driver),
fixed-continuation cold/hot/retrigger/partial-eviction ordinary/PD comparison,
and all 20 complete BCP trajectories.

## Resumed GPU validation, 2026-11-17

The user subsequently authorized releasing the identified Tengxuan jobs and
provided administrative access. Only the verified GPU processes were stopped;
all eight GPUs were confirmed free before starting this experiment. The earlier
resource-blocked statements above describe the preflight at that time.

The isolated Tengxuan checkout runs the expanded BF16 GPT-OSS-120B model with
TP=4, Triton attention and Triton MoE, page size 1 and disabled CUDA graphs.
Ordinary A uses GPUs 4–7 while B uses GPUs 0–3. The production source tree has
not changed during these validation-tool iterations. Recorded commit IDs are
retained; acceptance resolves their `python/sglang` Git trees to verify equality
when only diagnostic commits differ between long-running jobs.

Evidence as of 14:50 host time:

- Native small CUDA attention/zero-window tests: 14 passed. Independent
  complex-number attention reference tests on GPU: 24 passed.
- Bounded real-model logits pass the predeclared 2× no-event calibration gate
  for both policies. Native BF16/MoE variation is substantial (one no-event
  control has max error 2.1171875); this is not bitwise precision evidence.
- The independent streaming reference has exactly identical logits for full
  versus 17-token chunks on both policies under batch-invariant model GEMMs,
  including three externally fixed continuation tokens, on all four TP ranks.
  Its FP32 attention uses fixed per-query arithmetic shapes for this check.
- Context Triton explicitly rejects deterministic candidate execution. The
  initial combined invocation therefore failed two candidate tests while both
  reference chunk-invariance tests passed. Candidate attention layer checks
  are now run in the ordinary supported mode; their result remains pending.
- Ordinary A completed cases 223, 231 and 238 (126 turns), each with actual
  96 Ki-token Reposition and subsequent Drop. Ordinary B completed case 223
  (41 turns). Remaining ordinary cases and all PD trajectories are pending.
- Latest local policy checks: 56 passed, 23 skipped. Runner/artifact checks:
  7 passed. Environment-gated skips never count as acceptance.

`cache_policy_bcp.py reference-jobs` freezes native-template token ownership,
independently translated Drop/R boundaries, and an eight-token continuation
from completed reports. `--include-calibration` prepends a no-event control
before policy comparisons. `CONTEXT_REFERENCE_JOBS` and
`CONTEXT_REFERENCE_OUTPUT` enable the opt-in real-model reference artifact test;
the reference owns its KV and does not read candidate compilation or cache state.
Saving these artifacts is explicitly separate from numerical acceptance.

`cache_policy_bcp.py diagnostics` records cold/hot/retrigger/partial-eviction
fixed-token requests. The explicitly enabled probe inspects all 36 layers and
all TP ranks, checks independently expected physical decode reads and local
FP32 attention, and uses native cache eviction only after source leases are
released. Legacy requests without these diagnostic parameters are unaffected.
The original formal trajectories never enable the probes or forced sampling.
`compare-artifacts` verifies payload/continuation/runtime-source identity and
retains per-token error metrics even when comparison fails. These tools have
not yet completed the full BCP numerical matrix; their existence is not evidence
that the matrix passed.

## CUDA Graph quick replay, 2026-11-17

At the user's request, the eager long-run queue was stopped and the service
restarted with decode Full CUDA Graph (batch size 1) and prefill Breakable CUDA
Graph. Neither phase was disabled. Model, attention/MoE backends, TP=4, page
size and chunk size remain unchanged. The native one-token hot-prefill path
can fall below the smallest capture bucket; long prefill and decode replay
are confirmed by runtime logs, not just startup arguments.

`cache_policy_bcp.py quick --jobs JOBS --model MODEL --url URL --output RUN`
replays the five frozen real final histories under both policies, plus case
223's first Drop and first 96 Ki-token Reposition under the text policy. It
uses 64 generated tokens per request, cold/hot pairs, and independently checks
execution lengths. The run at `b270ebee0b` completed all 24 HTTP requests in
135.61 seconds of request time, excluding model startup. Local runner checks:
`python -m pytest -q benchmark/context_system/test_cache_policy_bcp.py`:
9 passed. The command now rejects a server with either graph phase disabled.

For the five final snapshots, median cold/hot latency was 5.65/1.42 seconds
for `text_drop_with_pe` and 16.06/1.53 seconds for
`message_drop_without_repos`. Hot requests reported positive cache hits.
Ten of twelve cold/hot free-generation pairs were token-identical. Case 231
diverged at generated token 34 for text policy and token 14 for KV policy.
This is a functional/performance smoke result, not precise-reuse acceptance.

A separate Graph-enabled case-231 diagnostic forced each policy's cold
64-token continuation on both cold and hot paths. Raw-logit max/mean/p99
differences (maximum over the 64 rows) were respectively:

| Path | Max absolute | Mean absolute | p99 absolute |
| --- | ---: | ---: | ---: |
| Native no-drop control | 4.6875 | 0.530116 | 1.718750 |
| Text policy | 2.96875 | 0.417898 | 1.535156 |
| KV policy | 2.71875 | 0.308439 | 1.234375 |

Each path had 62/64 identical raw-logit argmaxes. At the text divergence the
top two candidates were separated by 0.125 in the cold run and swapped in the
hot run; at the KV divergence the hot candidates tied. Policy differences
are below the native control on these aggregate metrics, but this diagnostic
does not establish their cause or replace the independent oracle matrix.

The normal-mode bounded same-QKV layer test failed its strict tolerance on
the no-drop baseline, before policy events. That numerical observation is
retained; the user subsequently accepted logits differences and set semantic
correctness as the acceptance criterion. It is no longer a numerical-equality
gate. Ordinary text free generation completed
all five cases (211 turns); the eager KV run was interrupted for this user
requested Graph run. Full Graph-enabled KV trajectories and PD acceptance
are still incomplete. Do not combine the shortened snapshot replay with the
original full-generation-budget acceptance reports.

## Semantic verification with CUDA Graph enabled

The opt-in `serialized_probe(graph_metadata=True)` observer runs after the
native decode graph runner loads the request. It compares the actual Full/SWA
index buffers used by graph replay with independently simulated event readers
on every TP rank. This observer is outside capture: the older per-layer Python
hook does not execute for every replay and cannot establish this coverage.
The observer does not disable or recapture the model graphs. Ordinary benchmark
requests omit `context_graph_read_probe` and pay no diagnostic tensor readback.
The request-state probe also checks active projection, computation and stored
positions, visibility expiry and final keep masks. KV policy diagnostics require
zero retrigger tokens. Numerical differences remain recorded separately.

The first graph observer attempt (`7d54383361`) incorrectly expected sampling
parameters on the backend's reduced replay view and caused the diagnostic
server to exit. `99198d68f3` moves observation to the full request loader and
adds a regression test for native loading without diagnostic parameters. This
was a test instrumentation failure; production policy code was unchanged.

The corrected real-service run `graph-semantic-99198d6` completed 26 cold/hot
requests: both policies at first Drop and the five frozen final BCP histories,
plus the text policy at first 96 Ki-token Reposition. All 1,456 physical-reader
checks passed (seven decode inputs, two attention pools, four TP ranks per
request), and all request-state comparisons passed. All 12 KV-policy requests
reported zero retry tokens. These are actual graph-replay buffer observations,
not equality of generated text or a numerical-tolerance inference. The histories
are from the earlier completed text trajectories, replayed as real requests
under each policy; this is separate from generating complete KV trajectories.

Focused IR/Radix/recovery/runtime tests passed 53 tests locally (21 skipped)
and 69 tests on BUS-2 with `CUDA_VISIBLE_DEVICES=0 RUN_CONTEXT_GPU=1` (5 skipped).
The GPU run includes native attention against independent event readers.
`graph-kv-99198d6` follows the semantic run and uses the unchanged five cases,
original per-turn output budgets, rolling drops and the Graph-enabled service
on GPUs 4–7. Its full-trajectory result remains pending. Neither this semantic
check nor the unit-level transfer tests establish full PD deployment acceptance.
