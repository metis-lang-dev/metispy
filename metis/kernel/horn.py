"""Backward-chaining Horn solver — the guard side of the semantics.

Backward predicates are NOT resources: they form a closed Horn world
(facts and rules whose bodies are backward predicates only — typing.py
enforces the separation). Guards on clause schemas are ground goals by
grounding time (every guard argument is a declared, instantiated
variable or a constant); derivability prunes the substitution, exactly
the MORPHISM §C′ GUARD rule:

    Σ ⊢_Horn gθ  for every bwd premise g of r
    ─────────────────────────────────────────
    rθ ∈ ground(r),  with the g's deleted

The solver is standard SLD resolution with unification (clause bodies
may chain through variables), depth-bounded. Domains encode relations —
arithmetic included — as fact tables (e.g. adv(c1,v2,c3)); no builtin
evaluation lives in the kernel."""

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from .schema import Pattern

_DEPTH = 256


@dataclass(frozen=True)
class HornClause:
    name: str
    head: Pattern
    body: Tuple[Pattern, ...] = ()
    vars: Tuple[str, ...] = ()      # variable names used in head/body


def fact(pred: str, *args: str) -> HornClause:
    return HornClause(f"{pred}({','.join(args)})", (pred, tuple(args)))


class HornDB:
    def __init__(self, clauses):
        self.by_pred: Dict[str, list] = {}
        for c in clauses:
            self.by_pred.setdefault(c.head[0], []).append(c)
        self._fresh = 0

    # -- unification over renamed-apart variables ("?v@k") ---------------

    def _rename(self, cl: HornClause) -> HornClause:
        if not cl.vars:
            return cl
        self._fresh += 1
        m = {v: f"?{v}@{self._fresh}" for v in cl.vars}
        r = lambda p: (p[0], tuple(m.get(a, a) for a in p[1]))
        return HornClause(cl.name, r(cl.head),
                          tuple(r(b) for b in cl.body))

    @staticmethod
    def _walk(t: str, s: Dict[str, str]) -> str:
        while t.startswith("?") and t in s:
            t = s[t]
        return t

    @classmethod
    def _unify(cls, pa: Pattern, pb: Pattern,
               subst: Dict[str, str]) -> Optional[Dict[str, str]]:
        if pa[0] != pb[0] or len(pa[1]) != len(pb[1]):
            return None
        s = dict(subst)
        for x, y in zip(pa[1], pb[1]):
            x, y = cls._walk(x, s), cls._walk(y, s)
            if x == y:
                continue
            if x.startswith("?"):
                s[x] = y
            elif y.startswith("?"):
                s[y] = x
            else:
                return None
        return s

    def _solve(self, goals, subst, depth) -> Iterator[Dict[str, str]]:
        if not goals:
            yield subst
            return
        if depth <= 0:
            return
        g0 = goals[0]
        g = (g0[0], tuple(self._walk(a, subst) for a in g0[1]))
        for cl in self.by_pred.get(g[0], ()):
            r = self._rename(cl)
            s2 = self._unify(r.head, g, subst)
            if s2 is None:
                continue
            yield from self._solve(r.body + tuple(goals[1:]), s2,
                                   depth - 1)

    def derivable(self, goal: Pattern) -> bool:
        return next(self._solve((goal,), {}, _DEPTH), None) is not None

    def solutions(self, goals) -> Iterator[Dict[str, str]]:
        """Deduplicated bindings of the ?-variables appearing in
        `goals` (a conjunction), SLD enumeration order."""
        vs = sorted({a for _, args in goals for a in args
                     if a.startswith("?")})
        seen = set()
        for s in self._solve(tuple(goals), {}, _DEPTH):
            key = tuple(self._walk(v, s) for v in vs)
            if key in seen:
                continue
            seen.add(key)
            yield {v[1:]: c for v, c in zip(vs, key)}
