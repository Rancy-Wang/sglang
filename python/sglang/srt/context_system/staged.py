"""One event frontier for Drop and Reposition, starting at the actual cache miss.

The immutable program remains the cache identity. Intermediate KV belongs to
the request and is not published under that program's final-position key.
Recovery supplies missing query intervals, including holes before the match;
this planner only splits those intervals at effective visibility/position events.
"""

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StagePlan:
    boundaries: tuple[int, ...]
    length: int

    @classmethod
    def from_program(cls, program):
        layout = program.layout
        # The frozen old ContextLayout supports Drop/Reposition only.
        n = len(layout.positions)
        if n < 1:
            raise ValueError("Staged execution requires a query token")
        # birth_stages may jump across several zero-query Reposition events.
        # Their ordered transitions are retained by the immutable layout.
        changes = np.flatnonzero(np.diff(layout.birth_stages.numpy())) + 1
        drops = layout.drop_insert_offsets.numpy()
        boundaries = sorted({int(x) for x in (*changes, *drops) if 0 < x < n})
        return cls(tuple([*boundaries, n]), n)

    def next_end(self, start: int, limit: int) -> int:
        if not 0 <= start < limit <= self.length:
            raise ValueError("Staged query window is outside the program")
        return min(limit, self.boundaries[bisect_right(self.boundaries, start)])


def has_unpublished_stage(req):
    """Do not insert intermediate positions under the final Radix identity."""
    if not getattr(req, "context_staged", False) or req.context_state is None:
        return False
    program = req.context_recompute_program or req.context_program
    return len(req.context_state.canonical_rows) < len(program.layout.positions)
