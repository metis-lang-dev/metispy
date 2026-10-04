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
may chain through variables), depth-bounded, with VARIANT TABLING: the
distinct answers of each call pattern (constants kept, variables
abstracted by first occurrence) are computed once, in SLD order, and
replayed. Observable results are SLD's — `derivable` is existence and
`solutions` dedups first-wins, so per-call dedup in first-found order
preserves both the answer set and its order — while recomputation of
shared subgoals (plus/mult over unary nats: ~n^6 before) disappears.
A call that re-enters a variant still being tabled falls back to plain
SLD; a tabled call gets its own depth budget (the bound is per call,
not per whole derivation — indistinguishable short of 256-deep proofs). Domains encode relations —
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
        self._table: Dict[Pattern, list] = {}   # complete variants
        self._active: set = set()                # variants being tabled

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

    def _resolve(self, g, rest, subst, depth):
        """Plain SLD step on goal g: one branch per matching clause."""
        for cl in self.by_pred.get(g[0], ()):
            r = self._rename(cl)
            s2 = self._unify(r.head, g, subst)
            if s2 is None:
                continue
            yield from self._solve(r.body + rest, s2, depth - 1)

    @staticmethod
    def _variant(args) -> Tuple[Tuple[str, ...], Dict[str, str]]:
        """Abstract variables by first occurrence: ?#0, ?#1, …"""
        m: Dict[str, str] = {}
        out = []
        for a in args:
            if a.startswith("?"):
                a = m.setdefault(a, f"?#{len(m)}")
            out.append(a)
        return tuple(out), m

    def _complete(self, g, key) -> list:
        """All distinct answers (g's args, instantiated, variables
        canonicalized) in SLD order — the first only, for a ground g."""
        self._active.add(key)
        answers, seen = [], set()
        try:
            for s3 in self._resolve(g, (), {}, _DEPTH):
                ans, _ = self._variant(
                    tuple(self._walk(a, s3) for a in g[1]))
                if ans not in seen:
                    seen.add(ans)
                    answers.append(ans)
                    if not key[1] or all(not a.startswith("?")
                                         for a in key[1]):
                        break              # ground: existence suffices
        finally:
            self._active.discard(key)
        self._table[key] = answers
        return answers

    def _solve(self, goals, subst, depth) -> Iterator[Dict[str, str]]:
        if not goals:
            yield subst
            return
        if depth <= 0:
            return
        g0 = goals[0]
        g = (g0[0], tuple(self._walk(a, subst) for a in g0[1]))
        rest = tuple(goals[1:])
        cargs, _ = self._variant(g[1])
        key = (g[0], cargs)
        answers = self._table.get(key)
        if answers is None:
            if key in self._active:          # re-entrant: plain SLD
                yield from self._resolve(g, rest, subst, depth)
                return
            answers = self._complete(g, key)
        for ans in answers:
            self._fresh += 1
            inst = tuple(f"{a}@{self._fresh}" if a.startswith("?")
                         else a for a in ans)
            s2 = self._unify((g[0], inst), g, subst)
            if s2 is None:
                continue
            yield from self._solve(rest, s2, depth - 1)

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
            if any(c.startswith("?") for c in key):
                # a quantified fact var answered the query without
                # being pinned: enumerable only through its type,
                # which the Horn world does not carry
                raise ValueError(
                    f"non-ground Horn solution for {list(goals)}: a "
                    f"fact variable is left unbound — guard it with "
                    f"a predicate that enumerates it")
            if key in seen:
                continue
            seen.add(key)
            yield {v[1:]: c for v, c in zip(vs, key)}
