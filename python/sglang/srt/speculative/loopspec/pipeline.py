"""Incremental LoopSpec tree. One call yields exactly one verified token.

The executor owns tensors/KV. This state machine never emits a proposal and
never conflates accepting the first proposal with accepting the second.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Lane:
    uid: int
    index: int
    token: int
    path: tuple[int, ...]
    prefix: tuple[int, ...]
    history: tuple[int, ...]
    parent: int | None = None
    origin: int | None = None
    depth: int = 0
    slot: int | None = None
    hidden: Any = None
    proposals: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, executor, first_token, prefix, max_new_tokens, total, depths):
        self.executor = executor
        self.total, self.depths = total, depths
        self.max_new_tokens = max_new_tokens
        self.next_uid = 1
        self.committed = 1
        self.cycles = self.reused = self.restarts = 0
        self.lanes = [Lane(0, 1, first_token, (0,), tuple(prefix), (first_token,))]

    def _child(self, parent, token, origin):
        uid = self.next_uid
        self.next_uid += 1
        return Lane(
            uid,
            parent.index + 1,
            token,
            parent.path + (uid,),
            parent.prefix + (parent.slot,),
            parent.history + (token,),
            parent=parent.uid,
            origin=origin,
        )

    def advance_until_commit(self):
        while True:
            self.cycles += 1
            following, children, final = [], [], None
            ordered = sorted(self.lanes, key=lambda lane: -lane.depth)
            if hasattr(self.executor, "advance_many"):
                self.executor.advance_many(ordered, self.depths[0])
            else:
                for lane in ordered:
                    self.executor.advance(lane, self.depths[0])
            # Ancestors run first, guaranteeing prefix KV exists at this depth.
            for lane in ordered:
                if lane.depth == self.total:
                    if final is not None or lane.index != self.committed:
                        raise RuntimeError("LoopSpec committed out of order")
                    final = lane, self.executor.verify(lane)
                    continue
                if lane.depth in self.depths:
                    proposal = self.executor.propose(lane)
                    if proposal is not None:
                        token, distribution = proposal
                        lane.proposals[lane.depth] = (token, distribution)
                        if lane.index + 1 < self.max_new_tokens:
                            children.append(self._child(lane, token, lane.depth))
                following.append(lane)
            following.extend(children)
            if final is None:
                self.lanes = following
                continue
            lane, (token, origin, logits) = final
            self.committed += 1
            selected = next(
                (
                    child
                    for child in following
                    if child.parent == lane.uid
                    and child.origin == origin
                    and child.token == token
                ),
                None,
            )
            if self.committed < self.max_new_tokens:
                if selected is None:
                    selected = self._child(lane, token, self.total)
                    following.append(selected)
                    self.restarts += 1
                else:
                    self.reused += int(selected.depth > 0)
                following = [child for child in following if selected.uid in child.path]
            else:
                following = []
            self.lanes = following
            # Ownership transfer happens before pruning shared prefix slots.
            self.executor.commit(lane)
            self.executor.retain(following)
            return token, origin, logits
