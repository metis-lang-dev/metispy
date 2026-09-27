"""Π-grounding: schemas × signature × Horn guards -> ground clauses.

Semantics pinned by the parity tests against ll2pg/fol: one ground
clause per substitution (same `name[X=c,Y=d]` tag format), multisets
accumulated so substitution collapse prices as falling factorials,
`distinct` side-conditions prune θ, guards prune θ by Horn derivability
and vanish from the multisets. All instances of a schema share the
schema's prior weight — CHOICE weight W_r = Σ_θ W_rθ is preserved
because each substitution is its own ground clause (MORPHISM §C′)."""

import itertools
from typing import List, Optional

from .clause import Clause, Link
from .horn import HornDB
from .program import Program
from .schema import ClauseSchema, LinkSchema, Signature, ground_atom, \
    multiset


def _substitutions(vars_, distinct, guards, sig: Signature,
                   horn: Optional[HornDB]):
    names = list(vars_)
    domains = [sig.types[vars_[v]] for v in names]
    for combo in itertools.product(*domains):
        subst = dict(zip(names, combo))
        if any(subst[x] == subst[y] for x, y in distinct):
            continue
        if guards:
            if horn is None:
                raise ValueError("guarded schema needs a HornDB")
            goals = ((g[0], tuple(subst.get(a, a) for a in g[1]))
                     for g in guards)
            if not all(horn.derivable(g) for g in goals):
                continue
        yield names, subst


def _tag(names, subst) -> str:
    return ",".join(f"{v}={subst[v]}" for v in names)


def ground_schema(schema: ClauseSchema, sig: Signature,
                  horn: Optional[HornDB] = None) -> List[Clause]:
    out = []
    for names, subst in _substitutions(schema.vars, schema.distinct,
                                       schema.guards, sig, horn):
        name = f"{schema.name}[{_tag(names, subst)}]" if names \
            else schema.name
        out.append(Clause(name,
                          consume=multiset(schema.consume, subst),
                          produce=multiset(schema.produce, subst),
                          persist=multiset(schema.persist, subst),
                          weight=schema.weight))
    return out


def ground_link(schema: LinkSchema, sig: Signature,
                horn: Optional[HornDB] = None) -> List[Link]:
    out = []
    for names, subst in _substitutions(schema.vars, schema.distinct,
                                       schema.guards, sig, horn):
        name = f"{schema.name}[{_tag(names, subst)}]" if names \
            else schema.name
        out.append(Link(name, pre=schema.pre, post=schema.post,
                        consume=multiset(schema.consume, subst),
                        produce=multiset(schema.produce, subst),
                        persist=multiset(schema.persist, subst),
                        weight=schema.weight))
    return out


def ground_program(stage_order, schemas, links, sig: Signature,
                   horn: Optional[HornDB], init_stage: str,
                   init_atoms) -> Program:
    """Assemble a ground staged Program. `init_atoms` is a list of
    ground Patterns (or a ready count dict)."""
    stages = {s: [] for s in stage_order}
    for sc in schemas:
        stages[sc.stage].extend(ground_schema(sc, sig, horn))
    glinks = [g for lk in links for g in ground_link(lk, sig, horn)]
    init = dict(init_atoms) if isinstance(init_atoms, dict) else \
        multiset(list(init_atoms), {})
    return Program(stages=stages, links=glinks,
                   init_stage=init_stage, init=init)
