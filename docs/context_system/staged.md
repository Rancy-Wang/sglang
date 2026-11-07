# Unified staged Drop / Reposition

Approved plan: `PLAN-CS-20261107-R2`, implementation round 1. Baseline:
`2997bc2ba466d047a4aaa8f6af4ae07fa60492f7` on `system-test`.

`--context-execution-mode staged` selects one scheduler-owned executor for Drop,
Reposition and their combination. The default is `occurrence`. Ordinary requests
have no staged state. There is no hit-ratio threshold or frontend probe/warmup.

The existing full-template `ContextProgram` and extended Radix match determine
the raw prefix, source positions and Full/SWA residency. `prepare_context_recovery`
closes necessary missing dependencies; resident gaps remain reusable. `StagePlan`
cuts each missing interval at Drop and effective Reposition boundaries and the
native admission budget. A match within a message resumes at that token, without
rendering the message again. Native generation still leaves a final query token.

Each nonterminal stage materializes only the positions needed at that stage.
The occurrence ownership machinery copies/rotates borrowed sources into private
destinations and retires old private versions after the batch completion barrier.
Drop removes dead owners without renumbering survivors. Reposition changes their
positions. Trailing zero-query events are materialized in the final forward;
no synthetic query token is inserted. Native middle-chunk handling discards the
padding sample and advances neither output nor grammar state.

Intermediate versions are held privately by the request, retaining the original
Radix source lease. They are not published under the complete program's final
position key. Final publication uses the existing sparse Drop-proof checks and
ownership transfer. An abort before final materialization disables insertion and
uses native private-page cleanup. This choice avoids introducing a second set of
stage-specific cache identities, but other requests cannot reuse intermediate
private work before the final publication.

First-release scope: text-only Full Attention and GPT-OSS Full/SWA, page size 1,
ordinary autoregressive TP serving on the existing supported native backends.
Staged Insert, hybrid Linear attention and PD are rejected explicitly. Existing
occurrence capability checks remain in force. CUDA Graph and overlap are retained.
Revert operationally by selecting `occurrence`; do not reset shared history.

## Validation

CPU tests compare ordered visible `(raw token, position)` pairs against an
independent simulator, recycle physical slots immediately after completion, and
cover cold/partial/hot matching, sparse repair and terminal-only transitions.

```sh
python -B -m pytest -o addopts= -p no:cacheprovider -q \
  test/registered/context_system/test_staged_plan.py \
  test/registered/context_system/test_planner.py \
  test/registered/context_system/test_occurrence_ownership.py
```

Native integration tests exercise the real CLI, admission and cache wrapper.
`mini_staged_oracle.py` runs mini's production Drop warmups and Reposition
open/step/ack protocol in its resident-model test pump. It keeps one model and
tokenizer initialized, separates formal logits from internal samples, checks
mini cache integrity and records its exact commit. It does not replace attention
with a dense reference or modify mini's source.

BCP uses `bcp_numeric_fixture.load_fixture`, with an explicit frozen input and
model, followed by the existing native raw-logits probe. Compare cold Drop,
cold Drop+Reposition, exact reuse, Retry and consecutive Reposition. The
`test_staged_runtime.py::test_bcp_staged_reference` entry can run either SGLang
mode against the same staged oracle. Fixed-token logits and free generation
are separate evidence. Existing numerical failures are not waived.

```sh
CONTEXT_MINI_ROOT=/verified/mini-checkout \
CONTEXT_SERVER_MODEL=/verified/model \
CONTEXT_BCP_FIXTURE=/frozen/bcp.json \
CONTEXT_ORACLE_OUTPUT=/isolated/mini-staged.json \
python test/registered/context_system/mini_staged_oracle.py

CONTEXT_SERVER_MODEL=/verified/model \
CONTEXT_BCP_ORACLE=/isolated/mini-staged.json \
CONTEXT_TRACE_DIR=/isolated/staged-traces \
CONTEXT_SERVER_EXTRA_ARGS='["--context-execution-mode","staged"]' \
python -B -m pytest -o addopts= -p no:cacheprovider -q \
  test/registered/context_system/test_staged_runtime.py
```

Validation status at the first checkpoint: local CPU suite 20 passed, 6 GPU-only
skipped. Native integration and BCP are pending; this is not a completed GPU or
numerical acceptance claim. The local command used the existing `myenv` Python
and the project's existing TVM FFI/ninja dependency directory, without changing
repository dependencies.
