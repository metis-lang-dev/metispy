"""Π-grounding: schemas × signature × Horn guards -> ground clauses.

Semantics pinned by the parity tests against ll2pg/fol: one ground
clause per substitution (same `name[X=c,Y=d]` tag format), multisets
accumulated so substitution collapse prices as falling factorials,
`distinct` side-conditions prune θ, guards prune θ by Horn derivability
and vanish from the multisets. All instances of a schema share the
schema's prior weight — CHOICE weight W_r = Σ_θ W_rθ is preserved
because each substitution is its own ground clause (MORPHISM §C′)."""

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
    closed = []                       # guards with no variable: checked once
    for g in guards or ():
        vs = sorted({pos[a] for a in g[1] if a in pos})
        if vs:
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
