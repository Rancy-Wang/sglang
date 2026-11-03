"""48K policy transactions, protocol boundaries and real summary validation."""
import copy

import pytest
from insert_summary48k import History, process_summary, select_summary


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
