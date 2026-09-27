"""First-order clause schemas over typed finite domains.

A schema is the Π-quantified form of a clause:
    Π X:τ … . consume(X̄) ⊗ !persist(X̄) ⊸ produce(X̄)   [guards, distinct]

Patterns are (predicate, arg-tuple); an arg is a declared variable (a key
of `vars`) or a constant. Grounding (ground.py) instantiates every
variable over its type's constants — INCLUDING variables that only occur
in produce/guards: functional outputs (e.g. the successor cell) are
declared vars whose value the guards pin down by pruning. Guards are
premises on BACKWARD predicates (Horn-derivable, horn.py); they prune
substitutions at ground time and vanish from the multisets (MORPHISM §C′
GUARD rule) — they are never resources.

`weight` is the clause prior, inherited by every ground instance."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Pattern = Tuple[str, Tuple[str, ...]]


def pat(pred: str, *args: str) -> Pattern:
    return (pred, tuple(args))


def ground_atom(pattern: Pattern, subst: Dict[str, str]) -> str:
    """Render a ground atom; identical format to the ll2pg/fol oracle
    (pred(a,b)), which the parity tests pin."""
    pred, args = pattern
    if not args:
        return pred
    return f"{pred}({','.join(subst.get(a, a) for a in args)})"


def multiset(patterns, subst) -> Dict[str, int]:
    """Accumulating duplicates is what makes substitution collapse
    (X=Y) price as multiplicity 2 -> falling factorial x^(2)."""
    out: Dict[str, int] = {}
    for p in patterns:
        a = ground_atom(p, subst)
        out[a] = out.get(a, 0) + 1
    return out


@dataclass(frozen=True)
class Signature:
    """type name -> its constants (finite domains)."""
    types: Dict[str, List[str]]


@dataclass
class ClauseSchema:
    name: str
    stage: str
    vars: Dict[str, str] = field(default_factory=dict)   # var -> type
    consume: List[Pattern] = field(default_factory=list)
    produce: List[Pattern] = field(default_factory=list)
    persist: List[Pattern] = field(default_factory=list)
    guards: List[Pattern] = field(default_factory=list)  # bwd premises
    distinct: List[Tuple[str, str]] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class LinkSchema:
    """Quiescence link, Π-quantified like a clause."""
    name: str
    pre: str
    post: str
    vars: Dict[str, str] = field(default_factory=dict)
    consume: List[Pattern] = field(default_factory=list)
    produce: List[Pattern] = field(default_factory=list)
    persist: List[Pattern] = field(default_factory=list)
    guards: List[Pattern] = field(default_factory=list)
    distinct: List[Tuple[str, str]] = field(default_factory=list)
    weight: float = 1.0
