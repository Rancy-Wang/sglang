"""48K policy transactions, protocol boundaries and real summary validation."""
import copy

import pytest
from insert_summary48k import (
    History,
    native_logit_bounds,
    process_summary,
    select_summary,
)


def test_numeric_bounds_use_only_native_controls_and_reject_invalid_controls():
    control = {"max_abs": 2., "mean_abs": .3, "p99_abs": .9}
    assert native_logit_bounds([control]) == {"max_abs": 4., "mean_abs": .6, "p99_abs": 1.8}
    assert native_logit_bounds([{k: 0. for k in control}]) == {
        "max_abs": .15, "mean_abs": .02, "p99_abs": .0625}
    for invalid in ([], [dict(control, max_abs=float("nan"))], [dict(control, mean_abs=-1)]):
        with pytest.raises(ValueError):
            native_logit_bounds(invalid)


def test_transaction_drops_before_insert_and_preserves_stateless_history():
    history = History()
    history.append([{"role": "user", "content": "question"}, {"role": "assistant", "content": "old"},
                    {"role": "user", "content": "old"}, {"role": "assistant", "content": "keep"}])
    trigger = history.append([{"role": "user", "content": "pending"}])
    history.commit([1, 2], 3, trigger, "summary")
    assert history.drop_message == {"3": [1, 2]}
    assert history.insert == {"4": -1} and history.reposition == [-1]
    assert [m["index"] for m in history.messages] == [0, 1, 2, -1, 3, 4]
    assert [m["index"] for m in history.active()] == [0, -1, 3, 4]
    history.append([{"role": "assistant", "content": "answer"}])
    trigger = history.append([{"role": "user", "content": "pending"}])
    history.commit([-1, 3, 4], 5, trigger, "updated summary")
    assert history.insert == {"4": -1, "6": -2}
    assert [m["index"] for m in history.active()] == [0, -2, 5, 6]


def test_protocol_groups_are_not_split():
    messages = [{"index": i, "role": r} for i, r in enumerate(
        ("user", "assistant", "tool", "assistant", "tool", "tool"))]
    selected, anchor = select_summary(messages, [i for i in range(6) for _ in range(10)], keep_recent=12)
    assert [m["index"] for m in selected] == [1, 2]
    assert anchor == 3


def test_failed_summary_does_not_mutate_history():
    history = History()
    history.append([{"role": "user", "content": "question"}])
    old = copy.deepcopy(history.payload())
    with pytest.raises(ValueError):
        process_summary({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    assert history.payload() == old


@pytest.mark.parametrize("reason,completion,valid", [
    ("abort", 0, False), ("abort", 1, False), ("error", 1, False),
    ("stop", 0, False), ("stop", 1, True), ("length", 64, True), ("tool_calls", 1, True),
])
def test_http_success_is_not_generation_success(reason, completion, valid):
    from insert_summary48k import validate_completion
    body = {"choices": [{"finish_reason": reason}], "usage": {"completion_tokens": completion}}
    if valid:
        validate_completion(body)
    else:
        with pytest.raises(RuntimeError, match="did not complete"):
            validate_completion(body)


def test_oversized_pending_retains_born_anchor_then_can_be_summarized():
    from insert_summary48k import THRESHOLD, validate_summary_size
    messages = [{"index": i, "role": role} for i, role in enumerate(
        ("user", "assistant", "tool", "assistant", "tool"))]
    owners = [i for i, n in enumerate((323, 30000, 3000, 536, 51762)) for _ in range(n)]
    selected, anchor = select_summary(messages, owners, trigger=4)
    assert [m["index"] for m in selected] == [1, 2] and anchor == 3
    assert validate_summary_size(85625, 53680, 51762)
    # On the next turn the huge tool result is computed history, so the 20K
    # target must yield to a later complete group that fits the 48K trigger.
    messages = [{"index": i, "role": role} for i, role in zip(
        (0, -1, 3, 4, 5, 6), ("user", "user", "assistant", "tool", "assistant", "tool"))]
    owners = [i for i, n in enumerate((323, 1000, 536, 51762, 200, 3000)) for _ in range(n)]
    selected, anchor = select_summary(messages, owners, trigger=6)
    assert [m["index"] for m in selected] == [-1, 3, 4] and anchor == 5
    assert not validate_summary_size(57000, 4800, 3000)
    with pytest.raises(ValueError, match="below 48K"):
        validate_summary_size(60000, THRESHOLD, 3000)
    with pytest.raises(ValueError, match="shorten"):
        validate_summary_size(60000, 60001, 52000)


def test_large_pending_user_does_not_become_an_unborn_insert_anchor():
    messages = [{"index": i, "role": role} for i, role in enumerate(
        ("user", "assistant", "user", "assistant", "user"))]
    owners = [i for i, n in enumerate((10, 2000, 2000, 20, 52000)) for _ in range(n)]
    selected, anchor = select_summary(messages, owners, trigger=4)
    assert [m["index"] for m in selected] == [1, 2]
    assert anchor == 3
