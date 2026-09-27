"""Probability notching — leg 2 of the learning doctrine
(docs/learning.md): the emission's sites ARE the sufficient-statistic
tables (paper Thm 4), so learning a port is counting site outcomes
per declared-scope row and taking the Dirichlet posterior mean.
alpha encodes the hand prior AND its strength; results ship as an
OVERLAY registry (experience over base — never editing the catalog)
plus the human-reviewable ROW SHEET (prior -> posterior with the
evidence beside it).

Site classes handled (checked structurally, refused otherwise):
- PAIRED: every candidate hooked, each factor appearing exactly as a
  complement pair (@w f / @w ~f, same args). The fired candidate
  attributes a Bernoulli draw to ITS pair — sound when pairs' guards
  are mutually exclusive (the snapshot discipline's shape; assumed,
  documented).
- CATEGORICAL: hooked candidates non-complemented with distinct
  factors, plus at least one UNHOOKED reference. Outcome-probability
  Dirichlet, factors recovered relative to the reference
  (f_j = (q_j / w_j) / (q_ref / w_ref)); the reference pins the
  scale CHOICE normalization would otherwise absorb.

Two data regimes: `site_counts` replays fully OBSERVED runs;
`expected_site_counts` takes exact posterior joints
(Conditioned.state_event_posterior) — the E-step, free because the
smoother is exact — and `em_notch` iterates it (MAP-EM, prior
anchored at the base registry)."""

import math
from typing import Callable, Dict, List, Sequence, Tuple

from .factorize import FactoredProgram
from .filter import ForwardProjection, apply_event
from .posterior import Conditioned

Row = Tuple[Tuple[str, int], ...]           # sorted (atom, bit)


def _row(scope, m) -> Row:
    return tuple(sorted((a, 1 if m.get(a, 0) > 0 else 0)
                        for a in scope))


def _site_plan(fp: FactoredProgram):
    """step -> ('paired'|'cat', site); plus (factor, args) -> scope."""
    plan, scopes = {}, {}
    for s in fp._sites:
        hooks = [h for _, _, h in s.recs]
        metas = [h[5] if h is not None else None for h in hooks]
        if any(h is not None and m is None
               for h, m in zip(hooks, metas)):
            continue                        # unserializable hook
        for h in hooks:
            if h is not None:
                scopes[h[5]] = h[1]
        by_factor: Dict[tuple, List[int]] = {}
        for j, h in enumerate(hooks):
            if h is not None:
                by_factor.setdefault(h[5], []).append(j)
        if not by_factor:
            continue
        if all(h is not None for h in hooks) and all(
                len(js) == 2 and hooks[js[0]][4] != hooks[js[1]][4]
                for js in by_factor.values()):
            plan[int(s.var[1:])] = ("paired", s)
        elif all(len(js) == 1 and not hooks[js[0]][4]
                 for js in by_factor.values()) and \
                any(h is None for h in hooks):
            plan[int(s.var[1:])] = ("cat", s)
    return plan, scopes


def _sig(site):
    return tuple((h[5] if h is not None else None,
                  round(ev.weight, 12))
                 for ev, _, h in site.recs)


def _accumulate(counts, kind, site, fired_name, m, w):
    """Add posterior weight w for the event named fired_name firing
    at `site` under marking m."""
    for j, (ev, _deps, h) in enumerate(site.recs):
        if ev.name != fired_name:
            continue
        if kind == "paired":
            _, scope, _, _, compl, meta = h
            key = ("b", meta[0], meta[1], _row(scope, m))
            c = counts.setdefault(key, [0.0, 0.0])
            c[0] += w
            c[1] += 0.0 if compl else w
        else:
            scope = frozenset().union(
                *(h2[1] for _, _, h2 in site.recs
                  if h2 is not None))
            key = ("c", _sig(site), _row(scope, m))
            c = counts.setdefault(key, [0.0] * len(site.recs))
            c[j] += w
        return


def site_counts(fp: FactoredProgram,
                traces: Sequence[Sequence[int]]) -> Dict:
    """Sufficient statistics from fully OBSERVED runs (event-index
    traces), replayed against the emission's site structure."""
    events = fp.program.events()
    plan, _ = _site_plan(fp)
    counts: Dict = {}
    for trace in traces:
        stage, kappa = fp.program.initial_state()
        m = dict(kappa)
        for t, e in enumerate(trace):
            if t in plan:
                kind, site = plan[t]
                _accumulate(counts, kind, site, events[e][2].name,
                            m, 1.0)
            stage, kappa = apply_event(events, stage, m, e)
            m = dict(kappa)
    return counts


def expected_site_counts(fp: FactoredProgram,
                         conds: Sequence[Conditioned]) -> Dict:
    """The exact E-step: expected sufficient statistics under the
    evidence posteriors (state_event_posterior joints)."""
    events = fp.program.events()
    plan, _ = _site_plan(fp)
    counts: Dict = {}
    for c in conds:
        for t in plan:
            kind, site = plan[t]
            if t >= c.pr.steps:
                continue
            for (s, e), w in c.state_event_posterior(t).items():
                if e is None:
                    continue
                m = dict(s[1])
                _accumulate(counts, kind, site,
                            events[e][2].name, m, w)
    return counts


def _estimate(fp: FactoredProgram, counts: Dict,
              base: Dict[str, object], alpha: float):
    """Posterior-mean rows + overlay registry + row sheet, prior
    anchored at the BASE registry (MAP-EM keeps it fixed)."""
    _, scopes = _site_plan(fp)
    learned: Dict = {}
    for key in sorted(counts, key=repr):
        if key[0] == "b":
            _tag, factor, args, row = key
            n, s = counts[key]
            p0 = float(base[factor](args, {a: 1 for a, b in row
                                           if b}))
            post = (alpha * p0 + s) / (alpha + n)
            learned[(factor, args, row)] = (p0, post, n)
        else:
            _tag, sig, row = key
            n = counts[key]
            present = {a: 1 for a, b in row if b}
            w0, q0 = [], []
            for meta, wgt in sig:
                f0 = 1.0 if meta is None else \
                    float(base[meta[0]](meta[1], dict(present)))
                w0.append(wgt)
                q0.append(wgt * f0)
            z0 = sum(q0)
            q0 = [q / z0 for q in q0]
            tot = alpha + sum(n)
            qhat = [(alpha * q0[j] + n[j]) / tot
                    for j in range(len(sig))]
            ref_q = sum(qhat[j] for j, (mt, _) in enumerate(sig)
                        if mt is None)
            ref_w = sum(w for mt, w in sig if mt is None)
            for j, (meta, wgt) in enumerate(sig):
                if meta is None:
                    continue
                fhat = (qhat[j] / wgt) / (ref_q / ref_w)
                f0 = q0[j] * z0 / wgt
                learned[(meta[0], meta[1], row)] = \
                    (f0, fhat, sum(n))

    by_factor: Dict[str, Dict] = {}
    for (factor, args, row), (_p0, post, _n) in learned.items():
        by_factor.setdefault(factor, {})[(args, row)] = post

    def make(factor, basefn):
        scope_of = {args: scopes[(f, args)]
                    for (f, args) in scopes if f == factor}
        rows = by_factor.get(factor, {})

        def fn(args, counts_, _rows=rows, _scope=scope_of,
               _base=basefn):
            sc = _scope.get(tuple(args))
            if sc is not None:
                v = _rows.get((tuple(args), _row(sc, counts_)))
                if v is not None:
                    return v
            return _base(args, counts_)
        return fn

    overlay = {f: make(f, b) for f, b in base.items()}
    lines = []
    for (factor, args, row), (p0, post, n) in sorted(learned.items()):
        head = factor if not args else f"{factor}({','.join(args)})"
        cond = " ".join(f"{a}={b}" for a, b in row)
        lines.append(f"{head} | {cond} : {p0:.3f} -> {post:.3f}   "
                     f"(n={n:.6g}, a0={alpha:g})")
    return overlay, learned, "\n".join(lines) + "\n"


def notch(fp: FactoredProgram, traces: Sequence[Sequence[int]],
          registry: Dict[str, object], alpha: float = 10.0):
    """(overlay registry, learned rows, row sheet) from fully
    observed runs. The overlay is a plain registry — weight_policy /
    factor_hooks consume it unchanged; shippable as a pack."""
    return _estimate(fp, site_counts(fp, traces), registry, alpha)


def em_notch(program, steps: int, registry: Dict[str, object],
             evidence_list: Sequence, hooks_factory: Callable,
             policy_factory: Callable, alpha: float = 10.0,
             iters: int = 8):
    """MAP-EM over PARTIALLY observed episodes (per-episode evidence
    predicates). Per iteration: exact E-step (state_event_posterior
    under the current registry), M-step = _estimate with the prior
    anchored at the ORIGINAL base. Returns (overlay, learned, sheet,
    log-likelihood trace) — the LL trace is monotone up to floating
    point, the standard EM guarantee."""
    reg = registry
    lls: List[float] = []
    out = (reg, {}, "")
    for _ in range(iters):
        pol = policy_factory(reg)
        fp = FactoredProgram(program, steps, hooks_factory(reg))
        conds, ll = [], 0.0
        for ev in evidence_list:
            c = Conditioned(ForwardProjection(program, steps,
                                              policy=pol), ev)
            ll += math.log(c.likelihood)
            conds.append(c)
        lls.append(ll)
        counts = expected_site_counts(fp, conds)
        out = _estimate(fp, counts, registry, alpha)
        reg = out[0]
    return out[0], out[1], out[2], lls
