"""Ground clauses and quiescence links — the unit the measure is built
from.

A Clause is a ground linear-logic rule
    c₁·a₁ ⊗ … ⊗ !p₁·b₁ ⊗ … ⊸ d₁·r₁ ⊗ …
with `consume` the linear premises (decremented), `persist` the
!-premises (read, never decremented), `produce` the products.

`weight` is the clause's PRIOR — metis's one deliberate extension over
the ceptre/ll2pg semantics, where priors lived only in the policy hook.
The CHOICE probability of a clause is proportional to
weight × transition_count (see weight.py). With every weight = 1 the
measure is bit-identical to the ll2pg oracle (parity mode; the oracle
tests run this way)."""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(slots=True)
class Clause:
    name: str
    consume: Dict[str, int] = field(default_factory=dict)
    produce: Dict[str, int] = field(default_factory=dict)
    persist: Dict[str, int] = field(default_factory=dict)
    weight: float = 1.0


@dataclass(slots=True)
class Link:
    """qui ⊗ stage(pre) ⊗ <consume/persist> ⊸ stage(post) ⊗ <produce>.

    Fires only at quiescence of `pre` (no clause of the stage enabled);
    switches the active stage to `post`. Guard atoms it needs are carried
    in consume/persist like any clause."""
    name: str
    pre: str
    post: str
    consume: Dict[str, int] = field(default_factory=dict)
    produce: Dict[str, int] = field(default_factory=dict)
    persist: Dict[str, int] = field(default_factory=dict)
    weight: float = 1.0
