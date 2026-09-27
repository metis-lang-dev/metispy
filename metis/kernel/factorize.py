"""Factor-graph emission from an ARBITRARY ground program — IR item 1
(docs/ir-plan.md): the k-step unrolling of a 0/1 program becomes an
occurrence net of CHOICE sites (categorical outcome variables with
CPTs) and SSA atom occurrences (binary variables), on which the
existing min-fill elimination and the admission cost certificate run
unchanged. This generalizes factored.FactoredSync — no binary-unit
assumption, no hand-provided p_on: sites, scopes and CPTs are DERIVED
from the program, its weight-port hooks and their declared read-scopes
(the `reads {...}` contract, metis.lang.factor_hooks).

How it works, per micro-step: constants propagate (control atoms —
chain tokens, stage tokens — fold away; nothing distinguishes control
from state but branch-independence itself). Where enabled candidates
or binding scopes touch branch-dependent atoms, a site variable is
emitted whose CPT rows are the normalized CHOICE weights, bindings
evaluated on synthesized snapshots restricted to their DECLARED scope
(reads outside it raise — the contract has teeth). Writes become SSA
versions: deterministic factors from the site outcome.

The emitted class (admission; everything else raises FactorizeError):
  - all multiset counts are 0/1 (no falling factorials);
  - no event may produce an atom that is possibly already present;
  - the stage schedule is branch-independent (competing links agree
    on their post stage);
  - tier mixing is not emitted: if on some REACHABLE branch a step's
    clause candidates are all disabled (falling through to links/done
    in the interleaved semantics), the lost mass is detected by the
    Z==1 check at evaluation and reported — reject, never silently
    approximate.

The construction pass is SYMBOLIC (no tables, no binding calls), so
certificate() prices a query at admission time exactly as
factored.FactoredSync.certificate does — same doctrine: reject or
downgrade the evaluator ladder before running, never kill mid-flight."""

import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .factored import Factor
from .jtree import JunctionTree
from .program import Program


class FactorizeError(ValueError):
    pass


class ScopedCounts(dict):
    """The binding's view of a synthesized snapshot: EXACTLY the
    declared read-scope. Present atoms map to 1 (multiset style);
    membership or lookup outside the scope raises."""
    def __init__(self, scope, present):
        super().__init__(present)
        self.scope = scope

    def _check(self, k):
        if k not in self.scope:
            raise FactorizeError(f"binding read outside its declared "
                                 f"scope: '{k}'")

    def __contains__(self, k):
        self._check(k)
        return super().__contains__(k)

    def __getitem__(self, k):
        self._check(k)
        return super().__getitem__(k)

    def get(self, k, default=None):
        self._check(k)
        return super().get(k, default)


# hooks(event_name) -> None | (fn(counts)->float, scope_atoms, compl
#                              [, (factor_name, ground_args)])
# the optional 4th element is serialization metadata (the IR wire)
Hooks = Callable[[str], Optional[tuple]]


@dataclass
class _Site:
    var: str
    parents: Tuple[str, ...]          # binary atom-version variables
    # per candidate: (event, dep vars that must be 1, folded hook)
    # folded hook: (fn, scope frozenset, base present, (atom, var)...)
    recs: List[Tuple[object, Tuple[str, ...], Optional[tuple]]]


@dataclass
class _Write:
    new: str
    x: str                            # site variable
    card: int
    old: Optional[str]                # None = old value was constant
    vals: List[Optional[int]]         # per outcome; None = keep old


class FactoredProgram:
    def __init__(self, program: Program, steps: int,
                 hooks: Optional[Hooks] = None,
                 unknowns: Tuple[str, ...] = ()):
        """`unknowns`: init atoms lifted to VARIABLES — the compiled
        structure (tree, tables, certificate) then serves EVERY tick
        of a receding-horizon loop; per tick only the init
        point-factors change (marginal(atom, init={...}))."""
        self.program = program
        self.steps = steps
        self.unknowns = tuple(sorted(unknowns))
        self._hooks = hooks or (lambda name: None)
        for name, counts in self._multisets(program):
            for a, n in counts.items():
                if n > 1:
                    raise FactorizeError(
                        f"{name}: multiplicity {n} on '{a}' — the "
                        f"emission covers 0/1 programs")
        self._env: Dict[str, object] = {a: 1 for a in program.init}
        self._cards: Dict[str, int] = {}
        self._sites: List[_Site] = []
        self._writes: List[_Write] = []
        self._unknown_vars: Dict[str, str] = {}
        for a in self.unknowns:
            v = f"{a}#i"
            self._cards[v] = 2
            self._env[a] = v
            self._unknown_vars[a] = v
        self._tables = None
        self._jt: Optional[JunctionTree] = None
        self._cal = None
        self._cal_key = None
        self._maybe_stall = False      # some site MAY have empty rows
        # (site var, card, old var) -> outcomes that PRODUCE the atom
        # while it may still be present: indicator factors zeroing the
        # (outcome ∧ present) configs — count-2 is impossible in the
        # emitted class, so reachable mass there fails the Z check
        self._guards: Dict[Tuple[str, int, str], set] = {}
        stage = program.init_stage
        for t in range(steps):
            stage = self._step(t, stage)
            if stage is None:              # done tier: counts frozen
                break

    @staticmethod
    def _multisets(program):
        yield "init", program.init
        for kind, _s, ev in program.events():
            for part in (ev.consume, ev.produce, ev.persist):
                yield ev.name, part

    # -- the symbolic pass (no tables, no binding calls) ---------------

    def _deps(self, ev):
        """Parent vars an event needs at 1, or None if it is
        definitely disabled (some premise is constant-absent)."""
        deps = []
        for a in list(ev.consume) + list(ev.persist):
            v = self._env.get(a, 0)
            if v == 0:
                return None
            if v != 1:
                deps.append(v)
        return tuple(dict.fromkeys(deps))

    def _step(self, t, stage):
        cands = [(ev, d) for ev in self.program.stages[stage]
                 if (d := self._deps(ev)) is not None]
        is_link = not cands
        if is_link:
            cands = [(lk, d) for lk in self.program.links
                     if lk.pre == stage
                     and (d := self._deps(lk)) is not None]
            if not cands:
                return None
            posts = {lk.post for lk, _ in cands}
            if len(posts) > 1:
                raise FactorizeError(
                    f"step {t}: branch-dependent stage schedule "
                    f"{sorted(posts)}")
        recs = []
        for ev, deps in cands:
            h = self._hooks(ev.name)
            hook = None
            if h is not None:
                fn, scope, compl = h[0], h[1], h[2]
                base = frozenset(a for a in scope
                                 if self._env.get(a, 0) == 1)
                varats = tuple((a, self._env[a]) for a in scope
                               if isinstance(self._env.get(a, 0), str))
                hook = (fn, frozenset(scope), base, varats, compl,
                        h[3] if len(h) > 3 else None)
            recs.append((ev, deps, hook))
        parents = tuple(dict.fromkeys(
            [v for _, deps, _ in recs for v in deps]
            + [v for _, _, h in recs if h for _, v in h[3]]))

        clean = all(self._env.get(a, 0) == 0 or a in recs[0][0].consume
                    for a in recs[0][0].produce)
        if len(recs) == 1 and not parents and clean:   # deterministic
            ev = recs[0][0]
            for a in ev.consume:
                self._env[a] = 0
            for a in ev.produce:
                self._env[a] = 1
            return ev.post if is_link else stage

        xvar = f"X{t}"
        self._cards[xvar] = len(recs)
        self._sites.append(_Site(xvar, parents, recs))
        depvars = tuple(dict.fromkeys(v for _, deps, _ in recs
                                      for v in deps))
        if len(depvars) > 16:
            self._maybe_stall = True
        else:
            for bits in itertools.product((0, 1), repeat=len(depvars)):
                asn = dict(zip(depvars, bits))
                if not any(all(asn[d] == 1 for d in deps)
                           for _, deps, _ in recs):
                    self._maybe_stall = True   # if reachable, Z < 1
                    break

        written = dict.fromkeys(a for ev, _, _ in recs
                                for a in list(ev.consume)
                                + list(ev.produce))
        for a in written:
            old = self._env.get(a, 0)
            vals: List[Optional[int]] = []
            for j, (ev, _, _) in enumerate(recs):
                if a in ev.produce:
                    if a not in ev.consume and old == 1:
                        raise FactorizeError(
                            f"{ev.name}: produces '{a}' which may "
                            f"already be present")
                    if a not in ev.consume and isinstance(old, str):
                        self._guards.setdefault(
                            (xvar, len(recs), old), set()).add(j)
                        self._maybe_stall = True
                    vals.append(1)
                elif a in ev.consume:
                    vals.append(0)
                else:
                    vals.append(None)          # keeps its old value
            if isinstance(old, int):
                folded = [old if v is None else v for v in vals]
                if len(set(folded)) == 1:
                    self._env[a] = folded[0]
                    continue
                new = self._new_var(a, t)
                self._writes.append(_Write(new, xvar, len(recs),
                                           None, folded))
            else:
                if all(v is None for v in vals):
                    continue
                new = self._new_var(a, t)
                self._writes.append(_Write(new, xvar, len(recs),
                                           old, vals))
            self._env[a] = new
        return recs[0][0].post if is_link else stage

    def _new_var(self, atom, t):
        v = f"{atom}#{t}"
        self._cards[v] = 2
        return v

    # -- the admission-time cost certificate ---------------------------

    def certificate(self, atom: str) -> Dict[str, int]:
        """Induced width + factor-ops for marginal(atom), from the
        symbolic structure only — no tables exist, no binding has
        been called. The admission gate."""
        v = self._env.get(atom, 0)
        if isinstance(v, int):
            return {"width": 0, "ops": 0}
        return _elim_cost(self._structure(), self._cards, {v})

    def _structure(self):
        """The emitted factor scopes — the symbolic object the
        certificate, the junction tree and the IR serialization all
        read."""
        scopes = [set(s.parents) | {s.var} for s in self._sites]
        scopes += [{w.x, w.new} | ({w.old} if w.old else set())
                   for w in self._writes]
        scopes += [{x, old} for x, _c, old in self._guards]
        scopes += [{v} for v in self._unknown_vars.values()]
        return scopes

    # -- evaluation ----------------------------------------------------

    def _build(self):
        fs = []
        for s in self._sites:
            card = len(s.recs)
            table = np.zeros((2,) * len(s.parents) + (card,))
            for bits in itertools.product((0, 1),
                                          repeat=len(s.parents)):
                assign = dict(zip(s.parents, bits))
                ws = []
                for ev, deps, hook in s.recs:
                    if any(assign[d] != 1 for d in deps):
                        ws.append(0.0)
                        continue
                    w = ev.weight
                    if hook is not None:
                        fn, scope, base, varats, compl = hook[:5]
                        present = {a: 1 for a in base}
                        present.update(
                            {a: 1 for a, pv in varats
                             if assign[pv] == 1})
                        f = float(fn(ScopedCounts(scope, present)))
                        w *= (1.0 - f) if compl else f
                    ws.append(w if w > 0 else 0.0)
                tot = sum(ws)
                if tot > 0:
                    for j, w in enumerate(ws):
                        table[bits + (j,)] = w / tot
                # else: stall row — zero mass; the Z check reports it
                # if any upstream branch actually reaches it
            fs.append(Factor(s.parents + (s.var,), table))
        for w in self._writes:
            if w.old is None:
                table = np.zeros((w.card, 2))
                for j, v in enumerate(w.vals):
                    table[j, v] = 1.0
                fs.append(Factor((w.x, w.new), table))
            else:
                table = np.zeros((w.card, 2, 2))
                for j, v in enumerate(w.vals):
                    if v is None:
                        table[j, 0, 0] = table[j, 1, 1] = 1.0
                    else:
                        table[j, 0, v] = table[j, 1, v] = 1.0
                fs.append(Factor((w.x, w.old, w.new), table))
        for (x, card, old), outcomes in self._guards.items():
            table = np.ones((card, 2))
            for j in outcomes:
                table[j, 1] = 0.0
            fs.append(Factor((x, old), table))
        return fs

    def _calibration(self, init=None):
        """Calibrate the junction tree (IR item 2: the tree is built
        ONCE; per evidence only the messages rerun). `init` sets the
        unknown atoms' point factors — the per-tick input. The Z==1
        check runs here: Z < 1 means a stalled-tier row is REACHABLE
        — reject, never approximate."""
        if init is not None:
            bad = set(init) - set(self.unknowns)
            if bad:
                raise FactorizeError(
                    f"init overrides non-unknown atoms: {sorted(bad)}")
        key = frozenset(
            (a, int(bool((init or {}).get(
                a, self.program.init.get(a, 0)))))
            for a in self.unknowns)
        if self._cal is not None and self._cal_key == key:
            return self._cal
        if self._jt is None:
            self._jt = JunctionTree(self._structure(), self._cards)
            self._tables = self._build()
        points = [Factor.point(self._unknown_vars[a], val)
                  for a, val in sorted(key)]
        cal = self._jt.calibrate(self._tables + points)
        if abs(cal.z - 1.0) > 1e-6:
            raise FactorizeError(
                f"probability mass {cal.z:.6g} != 1 — a stalled "
                f"clause tier is reachable; the program falls outside "
                f"the emitted class")
        self._cal, self._cal_key = cal, key
        return cal

    def marginal(self, atom: str, init=None) -> float:
        """P(atom present after the horizon)."""
        v = self._env.get(atom, 0)
        if isinstance(v, int):
            if self._maybe_stall:
                self._calibration(init)
            return float(v)
        m = self._calibration(init).marginal(v)
        return float(m[1])

    def marginals(self, atoms, init=None) -> Dict[str, float]:
        """All queries from ONE calibration — the junction-tree
        payoff over per-query elimination."""
        return {a: self.marginal(a, init) for a in atoms}

    def likelihood(self, observed: Dict[str, int],
                   init=None) -> float:
        """P(observed) — evidence-clamp point factors on the final
        occurrences and read the total mass. Zero IS the verdict:
        the runtime surprise op (docs/learning.md leg 1) — an
        unmodeled observation has no support and no parameter can
        give it any."""
        clamps = []
        for a, v in observed.items():
            ev = self._env.get(a, 0)
            want = 1 if v else 0
            if isinstance(ev, int):
                if ev != want:
                    return 0.0
                continue
            clamps.append((ev, want))
        if not clamps:
            return 1.0
        cal = self._calibration(init)      # builds tables; Z check
        points = [Factor.point(self._unknown_vars[a], val)
                  for a, val in sorted(self._cal_key)]
        points += [Factor.point(v, val) for v, val in clamps]
        return self._jt.calibrate(self._tables + points).z


_CACHE: Dict[tuple, FactoredProgram] = {}


def factored_for(program: Program, steps: int,
                 hooks: Optional[Hooks] = None,
                 unknowns: Tuple[str, ...] = (), salt=None,
                 cache: Optional[dict] = None) -> FactoredProgram:
    """The cross-tick seam (IR item 2): same program_key + horizon +
    unknowns -> the SAME compiled structure (tree, tables,
    certificate); per tick only the init point-factors change.
    `salt` must distinguish registries when one program runs under
    different bindings — hooks are not hashable."""
    from .nets import program_key
    c = _CACHE if cache is None else cache
    key = (program_key(program), steps, tuple(sorted(unknowns)), salt)
    fp = c.get(key)
    if fp is None:
        fp = c[key] = FactoredProgram(program, steps, hooks, unknowns)
    return fp


def _elim_cost(scopes, cards, keep):
    """Min-fill replay with per-variable cardinalities: induced width
    and total factor-ops (Π of the eliminated cluster's dims), plus
    the ORDER itself — the schedule the IR serialization ships to
    the C++ interpreter. The cardinality-aware sibling of
    factored._min_fill_order."""
    adj: Dict[str, set] = {}
    for s in scopes:
        for v in s:
            adj.setdefault(v, set()).update(s - {v})
    width, ops, order = 0, 0, []
    todo = set(adj) - keep
    while todo:
        def fill(v):
            nb = adj[v] - {v}
            return sum(1 for x, y in itertools.combinations(nb, 2)
                       if y not in adj.get(x, ()))
        # lexicographic tie-break: the shipped schedule and the
        # certified ops must not depend on hash randomization
        v = min(todo, key=lambda u: (fill(u), u))
        order.append(v)
        nb = adj[v] - {v}
        width = max(width, len(nb))
        cost = cards.get(v, 2)
        for u in nb:
            cost *= cards.get(u, 2)
        ops += cost
        for x in nb:
            adj[x].update(nb - {x})
            adj[x].discard(v)
        del adj[v]
        todo.discard(v)
    return {"width": width, "ops": ops, "order": order}
