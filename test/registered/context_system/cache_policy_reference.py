"""Slow specification, intentionally independent of production IR/compiler.

Every query records (logical token, key position) in causal order. A Drop in
text mode discards the old query graph and recomputes the entire live prefix;
in KV mode it recomputes full history at frozen-death/current-live positions.
"""

from dataclasses import dataclass


@dataclass
class Reference:
    active: list
    positions: list
    compute_positions: list
    readers: list
    next_position: int


def simulate(tokens, drops, reposition=(), *, policy):
    alive, positions, birth, readers = [], [], [], []
    expiry, cursor = [], 0
    for boundary in range(len(tokens) + 1):
        remove = {i for a, b in drops.get(boundary, ()) for i in range(a, b)}
        effective = [i for i in alive if i in remove]
        if effective:
            for i in effective:
                expiry[i] = boundary
            alive = [i for i in alive if i not in remove]
            if policy == "text_drop_with_pe":
                for rank, i in enumerate(alive):
                    birth[i] = positions[i]
                    readers[i] = [(j, positions[j]) for j in alive[: rank + 1]]
            else:
                for rank, i in enumerate(alive):
                    positions[i] = rank
                cursor = len(alive)
                for i in range(boundary):
                    birth[i] = positions[i]
                    readers[i] = [
                        (j, positions[j]) for j in range(i + 1) if i < expiry[j]
                    ]
        if boundary in reposition:
            if policy != "text_drop_with_pe":
                raise ValueError("KV policy has no reposition")
            if not alive:
                raise ValueError("No live tokens to reposition")
            # Match the existing definition of an effective R: a changed K.
            if any(positions[i] != rank for rank, i in enumerate(alive)):
                for rank, i in enumerate(alive):
                    positions[i] = rank
                cursor = len(alive)
        if boundary < len(tokens):
            i = boundary
            positions.append(cursor)
            birth.append(cursor)
            cursor += 1
            expiry.append(2**31 - 1)
            alive.append(i)
            readers.append([(j, positions[j]) for j in alive])
    return Reference(alive, positions, birth, readers, cursor)


def hidden_fingerprints(tokens, reference, layers=3):
    """Symbolic multilayer dependency graph, includes query/key PE separately."""
    hidden = list(tokens)
    for _ in range(layers):
        hidden = [
            (
                hidden[i],
                reference.compute_positions[i],
                tuple((hidden[j], p) for j, p in read),
            )
            for i, read in enumerate(reference.readers)
        ]
    return hidden
