"""Independent audit must detect lost frontiers and altered virtual records."""

from array import array
from types import SimpleNamespace

from test_ir import ROOT, load_file

audit = load_file(
    "insert_state_audit", ROOT / "benchmark/context_system/insert_state_audit.py"
)


class Key:
    def __init__(self, records, start=0, end=None):
        flat = array("i", [x for row in records for x in row])
        mapping = [i for i, row in enumerate(records) if row[0] == 0]
        self.context = SimpleNamespace(records=flat, token_to_record=mapping)
        self.context_start = start
        self.length = (len(mapping) if end is None else end) - start
        self.extra_key = self.cache_salt = None

    def __len__(self):
        return self.length


def node(key=None, children=(), evicted=False):
    return SimpleNamespace(key=key, children=dict(enumerate(children)), evicted=evicted)


def test_retry_ignores_only_token_positions():
    rows = [
        (0, 50, 0),
        (1, 1, 4),
        (2, 0, 0),
        (3, 7, 0),
        (0, 99, 3),
        (4, 0, 0),
        (0, 70, 4),
    ]
    key = Key(rows)
    rotated = Key([(k, a, b + 30 if k == 0 else b) for k, a, b in rows])
    root = node(children=[node(rotated)])
    assert audit.brute_frontier(root, key)[0] == 0
    assert audit.brute_frontier(root, key, retry=True)[0] == 3
    for i, (kind, a, b) in enumerate(rows):
        if kind == 0:
            continue
        changed = rows.copy()
        changed[i] = (kind, a + 1, b)
        assert (
            audit.brute_frontier(node(children=[node(Key(changed))]), key, retry=True)[
                0
            ]
            < 3
        )


def test_split_edges_and_eviction_are_real_frontiers():
    rows = [(0, 1, 0), (3, 2, 0), (0, 2, 5), (4, 0, 0), (0, 3, 6)]
    key = Key(rows)
    leaf = node(Key(rows, 2))
    middle = node(Key(rows, 1, 2), [leaf])
    root = node(children=[node(Key(rows, 0, 1), [middle])])
    assert audit.brute_frontier(root, key)[0] == 3
    leaf.evicted = True
    assert audit.brute_frontier(root, key)[0] == 2
    assert audit.token_units(Key(rows, 1, 2)) == [((3, 2, 0), (0, 2, 5))]


def test_frontier_examines_all_branches_and_namespace():
    target = Key([(0, 1, 0), (0, 2, 1), (0, 3, 2)])
    short = Key([(0, 1, 0), (0, 9, 1)])
    full = Key([(0, 1, 10), (0, 2, 11), (0, 3, 12)])
    root = node(children=[node(short), node(full)])
    assert audit.brute_frontier(root, target)[0] == 1
    assert audit.brute_frontier(root, target, retry=True)[0] == 3
    full.cache_salt = "other"
    assert audit.brute_frontier(root, target, retry=True)[0] == 1


def test_native_rounding_control_removes_only_dropped_indices():
    messages = [
        {"role": "user", "content": str(i), **({"index": i} if i < 0 else {})}
        for i in (0, 1, -1, 2, 3)
    ]
    payload = {"messages": messages, "drop_message": {"1": [1], "3": [1, 2, 2]}}
    assert [m["content"] for m in audit.active_messages(payload)] == ["0", "-1", "3"]
