"""Π-grounding: schemas × signature × Horn guards -> ground clauses.

Semantics pinned by the parity tests against ll2pg/fol: one ground
clause per substitution (same `name[X=c,Y=d]` tag format), multisets
accumulated so substitution collapse prices as falling factorials,
`distinct` side-conditions prune θ, guards prune θ by Horn derivability
and vanish from the multisets. All instances of a schema share the
schema's prior weight — CHOICE weight W_r = Σ_θ W_rθ is preserved
because each substitution is its own ground clause (MORPHISM §C′)."""

import os
from typing import List, Optional

from .clause import Clause, Link
from .horn import HornDB, HornNonground
from .program import Program
from .schema import ClauseSchema, LinkSchema, Signature, ground_atom, \
    multiset


def _values(horn: HornDB, g, v, subst, memo):
    """The constants of variable v that make guard g derivable, the other
    variables of g bound by subst: one Horn query (memoized per guard and
    binding); None when the answer is not a set of constants (a quantified
    fact leaves v open: the caller checks each constant instead)."""
    args = tuple("?" + v if a == v else subst.get(a, a) for a in g[1])
    key = (g[0], args)
    if key not in memo:
        try:
            memo[key] = {s[v] for s in horn.solutions([(g[0], args)])}
        except HornNonground:
            memo[key] = None
    return memo[key]


def _substitutions(vars_, distinct, guards, sig: Signature,
                   horn: Optional[HornDB]):
    """The substitutions of the schema's variables over their types that
    keep every `distinct` pair apart and make every guard derivable — the
    same ones, in the same order, as filtering the full product of the
    types (itertools.product), found depth first: each guard is answered
    when its last variable is bound, as that variable's values (one Horn
    query, kept in the type's order), and a distinct pair is checked there
    too, so a failed prefix is never extended (a planner instance checked
    440k guards one by one to keep 28k clauses; a specialist 171k)."""
    names = list(vars_)
    pos = {v: i for i, v in enumerate(names)}
    domains = [list(sig.types[vars_[v]]) for v in names]
    if guards and horn is None:
        if all(domains):          # as before: raised at the first combination
            raise ValueError("guarded schema needs a HornDB")
        return
    at = [([], []) for _ in names]    # per position: the guards, the pairs it completes
    early = [[] for _ in names]       # per position: fact-table guards it occurs in
    closed = []                       # guards with no variable: checked once
    variables = set(names)
    for g in guards or ():
        vs = sorted({pos[a] for a in g[1] if a in pos})
        if vs and horn is not None and horn.fact_table(g[0]) is not None:
            # a guard on a ground fact table filters EVERY position it
            # names, from the index, as soon as that position is bound —
            # the guard cut eliminated first, not at its last variable:
            # a prefix no row can complete is never extended (the same
            # substitutions, in the same order: domains are only filtered)
            for i in vs:
                early[i].append(g)
        elif vs:
            at[vs[-1]][0].append(g)
        else:
            closed.append(g)
    for x, y in distinct or ():
        at[max(pos.get(x, 0), pos.get(y, 0))][1].append((x, y))
    if names and not all(domains):
        return
    if any(not horn.derivable(g) for g in closed):
        return
    subst, memo = {}, {}

    def walk(i):
        if i == len(names):
            yield names, dict(subst)
            return
        v, (gs, ds) = names[i], at[i]
        dom = domains[i]
        for g in early[i]:
            key = (g, v, tuple(subst.get(a) for a in g[1]))
            vals = memo.get(key)
            if vals is None:
                vals = memo[key] = horn.fact_table(g[0]).values(
                    g[1], v, variables, subst)
            dom = [c for c in dom if c in vals]
        checks = []
        for g in gs:
            vals = _values(horn, g, v, subst, memo)
            if vals is None:
                checks.append(g)
            else:
                dom = [c for c in dom if c in vals]
        for c in dom:
            subst[v] = c
            if any(subst[x] == subst[y] for x, y in ds):
                continue
            if checks and not all(horn.derivable(
                    (g[0], tuple(subst.get(a, a) for a in g[1])))
                    for g in checks):
                continue
            yield from walk(i + 1)
        subst.pop(v, None)

    try:
        yield from walk(0)
    finally:
        # walk is a closure over itself: break the cycle so the memo (one
        # entry per guard and prefix) goes with the generator, not with
        # the next cyclic collection
        memo.clear()
        walk = None


def _tag(names, subst) -> str:
    return ",".join(f"{v}={subst[v]}" for v in names)


class GroundBudget(ValueError):
    """A grounding past METIS_GROUND_BUDGET events (unset: no budget). The
    policy loop sets it: a candidate rule whose variables range over whole
    types (`connected(T1, T2)` read by nothing else, 167^2 terms) is refused
    with its words, not materialised until a memory scope kills the run."""


def ground_budget() -> Optional[int]:
    b = os.environ.get("METIS_GROUND_BUDGET")
    return int(b) if b else None


def _over(schema, sig, limit):
    sizes = ", ".join(f"{v}: {t} {len(sig.types.get(t, ()))}"
                      for v, t in schema.vars.items())
    return GroundBudget(
        f"ground-budget: {schema.name} grounds more than {limit} events "
        f"over its variables ({sizes}); bind them on the left with a fact "
        f"the state holds, or drop the atom that reads them alone")


def ground_schema(schema: ClauseSchema, sig: Signature,
                  horn: Optional[HornDB] = None,
                  atoms=None, limit: Optional[int] = None) -> List[Clause]:
    out = []
    for names, subst in _substitutions(schema.vars, schema.distinct,
                                       schema.guards, sig, horn):
        if limit is not None and len(out) >= limit:
            raise _over(schema, sig, limit)
        name = f"{schema.name}[{_tag(names, subst)}]" if names \
            else schema.name
        out.append(Clause(name,
                          consume=multiset(schema.consume, subst, atoms),
                          produce=multiset(schema.produce, subst, atoms),
                          persist=multiset(schema.persist, subst, atoms),
                          weight=schema.weight))
    return out


def ground_link(schema: LinkSchema, sig: Signature,
                horn: Optional[HornDB] = None,
                atoms=None) -> List[Link]:
    out = []
    for names, subst in _substitutions(schema.vars, schema.distinct,
                                       schema.guards, sig, horn):
        name = f"{schema.name}[{_tag(names, subst)}]" if names \
            else schema.name
        out.append(Link(name, pre=schema.pre, post=schema.post,
                        consume=multiset(schema.consume, subst, atoms),
                        produce=multiset(schema.produce, subst, atoms),
                        persist=multiset(schema.persist, subst, atoms),
                        weight=schema.weight))
    return out


def ground_program(stage_order, schemas, links, sig: Signature,
                   horn: Optional[HornDB], init_stage: str,
                   init_atoms) -> Program:
    """Assemble a ground staged Program. `init_atoms` is a list of
    ground Patterns (or a ready count dict)."""
    stages = {s: [] for s in stage_order}
    atoms = {}                        # the atom table, for this grounding only
    budget, n = ground_budget(), 0
    for sc in schemas:
        g = ground_schema(sc, sig, horn, atoms,
                          None if budget is None else budget - n)
        n += len(g)
        stages[sc.stage].extend(g)
    glinks = [g for lk in links for g in ground_link(lk, sig, horn, atoms)]
    init = dict(init_atoms) if isinstance(init_atoms, dict) else \
        multiset(list(init_atoms), {})
    return Program(stages=stages, links=glinks,
                   init_stage=init_stage, init=init)
