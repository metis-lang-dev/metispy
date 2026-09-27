"""A ground program: staged clause sets, quiescence links, initial state.

This is the ONE object of the architecture — proof theory, derivations
and the measure are readouts of it (docs/ARCHITECTURE.md). Stage order is
significant only for event indexing (stable enumeration for policies and
event distributions), never for semantics."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .clause import Clause, Link

# a state of the process: (active stage, multiset as a frozenset of
# (atom, count>0) pairs) — hashable, order-free
State = Tuple[str, frozenset]


def pack_counts(counts: Dict[str, int]) -> frozenset:
    return frozenset((a, n) for a, n in counts.items() if n > 0)


@dataclass
class Program:
    stages: Dict[str, List[Clause]]
    links: List[Link] = field(default_factory=list)
    init_stage: str = ""
    init: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.init_stage:
            self.init_stage = next(iter(self.stages))

    @property
    def stage_names(self) -> List[str]:
        return list(self.stages)

    def events(self):
        """Stable event table: clauses in stage order, then links —
        the enumeration policies and event distributions index by."""
        ev = [("clause", s, c) for s in self.stages
              for c in self.stages[s]]
        ev += [("link", lk.pre, lk) for lk in self.links]
        return ev

    def initial_state(self) -> State:
        return (self.init_stage, pack_counts(self.init))
