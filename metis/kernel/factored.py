"""The factored evaluator — EFD cut-discipline variable elimination
for the sync class (docs/efd-comparison.md), v2: numpy factor tables
with broadcast-aligned products and MIN-FILL elimination (the step-1
profile ranking), plus the ADMISSION-TIME COST CERTIFICATE.

Real-time defense (the design decision): programs are rejected or
downgraded BEFORE evaluation, never killed mid-flight. Induced width
and total factor-ops are graph properties of (scopes, ticks, query) —
`certificate()` computes them symbolically with no tables, at the
same place contextual typing runs (admission). The runtime budget is
expressed in FACTOR-OPS (platform-independent), converted to time by
a measured per-platform constant; over-budget queries fall back down
the evaluator ladder (exact-support -> factored -> sampled-with-
fixed-N, whose cost is deterministic by construction)."""

import itertools
from typing import Callable, Dict, Iterable, List, Set, Tuple

import numpy as np


class Factor:
    __slots__ = ("vars", "table")

    def __init__(self, vars_: Tuple[str, ...], table: np.ndarray):
        self.vars = vars_
        self.table = table          # shape (2,)*len(vars)

    @staticmethod
    def point(var: str, value: int) -> "Factor":
        t = np.zeros(2)
        t[value] = 1.0
        return Factor((var,), t)

    def multiply(self, other: "Factor") -> "Factor":
        nv = self.vars + tuple(v for v in other.vars
                               if v not in self.vars)
        a = self.table.reshape(self.table.shape
                               + (1,) * (len(nv) - len(self.vars)))
        perm = [other.vars.index(v) if v in other.vars else None
                for v in nv]
        src = [i for i, p in enumerate(perm) if p is not None]
        b = np.transpose(other.table,
                         [perm[i] for i in src])
        shape = [other.table.shape[p] if p is not None else 1
                 for p in perm]
        b = b.reshape(shape)
        return Factor(nv, a * b)

    def sum_out(self, var: str) -> "Factor":
        i = self.vars.index(var)
        return Factor(self.vars[:i] + self.vars[i + 1:],
                      self.table.sum(axis=i))


def _min_fill_order(scopes: List[Set[str]], keep: Set[str]):
    adj: Dict[str, Set[str]] = {}
    for s in scopes:
        for v in s:
            adj.setdefault(v, set()).update(s - {v})
    order = []
    todo = set(adj) - keep
    while todo:
        def fill(v):
            nb = adj[v] - {v}
            return sum(1 for x, y in itertools.combinations(nb, 2)
                       if y not in adj.get(x, ()))
        # lexicographic tie-break: hash-randomization independent
        v = min(todo, key=lambda u: (fill(u), u))
        nb = adj[v] - {v}
        for x in nb:
            adj[x].update(nb - {x})
            adj[x].discard(v)
        order.append((v, len(nb)))
        del adj[v]
        todo.discard(v)
    return order


def _eliminate(factors: List[Factor], keep: Set[str]) -> List[Factor]:
    order = _min_fill_order([set(f.vars) for f in factors], keep)
    for v, _w in order:
        touching = [f for f in factors if v in f.vars]
        if not touching:
            continue
        rest = [f for f in factors if v not in f.vars]
        prod = touching[0]
        for f in touching[1:]:
            prod = prod.multiply(f)
        rest.append(prod.sum_out(v))
        factors = rest
    return factors


class FactoredSync:
    """Query-time elimination over the full k-tick unrolling (never a
    maintained belief state — Boyen-Koller entanglement, see v1)."""

    def __init__(self, units: List[str],
                 parents: Dict[str, List[str]],
                 p_on: Callable[[str, Dict[str, int], bool], float],
                 init_on: Iterable[str]):
        self.units = list(units)
        self.parents = parents
        self.p_on = p_on
        on = set(init_on)
        self.t = 0
        self._init = [Factor.point(f"{u}@0", 1 if u in on else 0)
                      for u in self.units]

    def _scope(self, u):
        return tuple(dict.fromkeys(self.parents.get(u, []) + [u]))

    def _cpt(self, u: str, t: int) -> Factor:
        scope = self._scope(u)
        nv = tuple(f"{v}@{t}" for v in scope) + (f"{u}@{t + 1}",)
        table = np.empty((2,) * len(nv))
        for bits in itertools.product((0, 1), repeat=len(scope)):
            snap = dict(zip(scope, bits))
            p = self.p_on(u, snap, bool(snap[u]))
            table[bits + (1,)] = p
            table[bits + (0,)] = 1.0 - p
        return Factor(nv, table)

    def tick(self, k: int = 1):
        self.t += k
        return self

    def _unrolled(self):
        fs = list(self._init)
        for t in range(self.t):
            fs += [self._cpt(w, t) for w in self.units]
        return fs

    def marginal(self, u: str) -> float:
        var = f"{u}@{self.t}"
        fs = _eliminate(self._unrolled(), {var})
        prod = fs[0]
        for f in fs[1:]:
            prod = prod.multiply(f)
        z = prod.table.sum()
        on = prod.table.sum() - prod.table.take(0, prod.vars.index(var)).sum()
        return float(on / z) if z else 0.0

    # -- the admission-time cost certificate ---------------------------

    def certificate(self, query_unit: str) -> Dict[str, int]:
        """Induced width + total factor-ops for marginal(query_unit),
        computed SYMBOLICALLY (scopes only, no tables) — the
        admission gate: reject/downgrade before evaluating."""
        scopes = [{f"{u}@0"} for u in self.units]
        for t in range(self.t):
            for u in self.units:
                scopes.append({f"{v}@{t}" for v in self._scope(u)}
                              | {f"{u}@{t + 1}"})
        order = _min_fill_order(scopes, {f"{query_unit}@{self.t}"})
        width = max((w for _, w in order), default=0)
        ops = sum(2 ** (w + 1) for _, w in order)
        return {"width": width, "ops": ops}
