"""Junction tree over factor scopes — IR item 2 (docs/ir-plan.md):
the elimination structure is computed ONCE per compiled program
(min-fill + running-intersection construction) and CALIBRATED per
evidence; one two-pass calibration yields every variable's marginal,
where plain elimination pays per query. This is the AOT analysis the
pack artifact will carry for the C++ interpreter: the tree and the
schedule are compile-time objects; evaluation is table ops along a
fixed structure.

Generic over factored.Factor (categorical dims); knows nothing about
programs — factorize.FactoredProgram feeds it the emitted scopes."""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from .factored import Factor


class JunctionTree:
    """Built from SCOPES only (symbolic — the admission side);
    calibrate() takes the actual factors. Cliques come from min-fill
    elimination: clique_i = {v_i} ∪ neighbors(v_i) at elimination
    time; parent_i = the clique of the earliest-eliminated separator
    var (running intersection holds by the fill edges)."""

    def __init__(self, scopes: Sequence[Set[str]],
                 cards: Dict[str, int]):
        self.cards = cards
        adj: Dict[str, set] = {}
        for s in scopes:
            for v in s:
                adj.setdefault(v, set()).update(s - {v})
        self.cliques: List[Tuple[str, frozenset, frozenset]] = []
        self.pos: Dict[str, int] = {}
        todo = set(adj)
        while todo:
            def fill(v):
                nb = adj[v] - {v}
                return sum(1 for x, y in
                           itertools.combinations(nb, 2)
                           if y not in adj.get(x, ()))
            # lexicographic tie-break: the AOT tree must be the same
            # structure in every process (hash-randomization free)
            v = min(todo, key=lambda u: (fill(u), u))
            nb = frozenset(adj[v] - {v})
            self.pos[v] = len(self.cliques)
            self.cliques.append((v, nb | {v}, nb))
            for x in nb:
                adj[x].update(nb - {x})
                adj[x].discard(v)
            del adj[v]
            todo.discard(v)
        self.parent = [min((self.pos[u] for u in sep), default=-1)
                       for _v, _cl, sep in self.cliques]
        self.children: List[List[int]] = [[] for _ in self.cliques]
        for i, p in enumerate(self.parent):
            if p >= 0:
                self.children[p].append(i)
        # component root of each clique (forests are fine)
        self.root = list(range(len(self.cliques)))
        for i, p in enumerate(self.parent):
            if p >= 0:
                self.root[i] = None
        for i in range(len(self.cliques)):
            j = i
            while self.parent[j] >= 0:
                j = self.parent[j]
            self.root[i] = j

    def calibrate(self, factors: Sequence[Factor]) -> "Calibration":
        """Two-pass sum-product. Every factor lands in the clique of
        its earliest-eliminated var (which contains its whole scope —
        the elimination-clique property)."""
        n = len(self.cliques)
        assigned: List[List[Factor]] = [[] for _ in range(n)]
        scalar = 1.0
        for f in factors:
            if not f.vars:
                scalar *= float(f.table)
                continue
            assigned[min(self.pos[v] for v in f.vars)].append(f)

        def product(fs):
            prod = fs[0]
            for f in fs[1:]:
                prod = prod.multiply(f)
            return prod

        def sum_to(f, keep):
            for u in [u for u in f.vars if u not in keep]:
                f = f.sum_out(u)
            return f

        pots: List[Factor] = [None] * n
        ups: List[Factor] = [None] * n
        for i, (v, cl, sep) in enumerate(self.cliques):
            fs = assigned[i] + [ups[c] for c in self.children[i]]
            if not fs:
                fs = [Factor((v,), np.ones(self.cards.get(v, 2)))]
            pots[i] = product(fs)
            ups[i] = sum_to(pots[i], sep)
        zroot = {i: (float(ups[i].table) if not ups[i].vars
                     else float(ups[i].table.sum()))
                 for i in range(n) if self.parent[i] < 0}
        downs: List[Factor] = [None] * n
        beliefs: List[Factor] = [None] * n
        for i in reversed(range(n)):
            b = pots[i]
            if downs[i] is not None:
                b = b.multiply(downs[i])
            beliefs[i] = b
            for c in self.children[i]:
                # division-free: rebuild the message from the
                # siblings instead of dividing c's contribution out
                fs = assigned[i] + [ups[s] for s in self.children[i]
                                    if s != c]
                if downs[i] is not None:
                    fs = fs + [downs[i]]
                if not fs:
                    v = self.cliques[i][0]
                    fs = [Factor((v,),
                                 np.ones(self.cards.get(v, 2)))]
                m = product(fs)
                downs[c] = sum_to(m, self.cliques[c][2])
        return Calibration(self, beliefs, zroot, scalar)


@dataclass
class Calibration:
    tree: JunctionTree
    beliefs: List[Factor]
    zroot: Dict[int, float]
    scalar: float

    @property
    def z(self) -> float:
        z = self.scalar
        for zr in self.zroot.values():
            z *= zr
        return z

    def marginal(self, var: str) -> np.ndarray:
        """Normalized marginal over `var` (within its component —
        components are independent, so this IS the joint marginal)."""
        i = self.tree.pos[var]
        b = self.beliefs[i]
        for u in [u for u in b.vars if u != var]:
            b = b.sum_out(u)
        zr = self.zroot[self.tree.root[i]]
        return b.table / zr
