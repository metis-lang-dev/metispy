"""The Bayesian leg — reading the measure backward.

The forward filter computes the predictive (generative-direction)
measure. This module adds the queries the original vision called
"evaluate which inference path to follow":

  Conditioned          forward-backward over the reachable state graph
                       given per-slice evidence predicates: evidence
                       likelihood, smoothed state marginals gamma_t,
                       posterior event distributions xi_t
  behavior_posterior   model selection over hypothesis measures (e.g.
                       per-agent behavior rows as clause-weight
                       variants): P(h | evidence) ∝ prior(h)·L_h —
                       the filtered-belief replacement for static
                       priors; feed each posterior back as the next
                       tick's prior
  map_derivation       Viterbi over the same graph: the most probable
                       derivation (proof) of the horizon, optionally
                       conditioned — the explanation object
  surprise             likelihood-ratio upgrade of wayplan's
                       check_surprise support test: unmodeled (zero
                       mass) vs rare (mass ratio to the modal state
                       below threshold)

All passes reuse the projection's OWN transition machinery
(_choice + apply_event), so conditioning can never disagree with
filtering by construction. Evidence is per-slice hard predicates on
states; zero-likelihood evidence raises."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .filter import DONE, ForwardProjection, State, apply_event

Evidence = Dict[int, Callable[[State], bool]]


def _passes(evidence: Optional[Evidence], t: int, s: State) -> float:
    """Evidence weight: predicates may return bool (hard) or float
    (soft/likelihood evidence — the TODO one-liner, landed)."""
    if not evidence:
        return 1.0
    pred = evidence.get(t)
    if pred is None:
        return 1.0
    v = pred(s)
    return float(v)


class Conditioned:
    def __init__(self, pr: ForwardProjection,
                 evidence: Optional[Evidence] = None):
        self.pr = pr
        K = pr.steps
        init = pr.program.initial_state()

        # alpha (filtered, evidence-gated)
        w0 = _passes(evidence, 0, init)
        alpha: List[Dict[State, float]] = [
            {init: w0} if w0 > 0 else {}]
        trans: List[Dict[State, List[Tuple[object, float, State]]]] = []
        for t in range(K):
            step: Dict[State, List[Tuple[object, float, State]]] = {}
            nxt: Dict[State, float] = {}
            for s, p in alpha[t].items():
                stage, kappa = s
                counts = dict(kappa)
                outs = []
                for e, pe in pr._choice(stage, counts):
                    s2 = apply_event(pr.events, stage, counts, e)
                    outs.append((e, pe, s2))
                    w = _passes(evidence, t + 1, s2)
                    if w > 0:
                        nxt[s2] = nxt.get(s2, 0.0) + p * pe * w
                step[s] = outs
            trans.append(step)
            alpha.append(nxt)
        self.likelihood = sum(alpha[K].values())
        if self.likelihood <= 0.0:
            raise ValueError("evidence has zero likelihood")

        # beta (backward)
        beta: List[Dict[State, float]] = [dict() for _ in range(K + 1)]
        beta[K] = {s: 1.0 for s in alpha[K]}
        for t in range(K - 1, -1, -1):
            b: Dict[State, float] = {}
            for s in alpha[t]:
                tot = 0.0
                for e, pe, s2 in trans[t][s]:
                    w = _passes(evidence, t + 1, s2)
                    if w > 0:
                        tot += pe * w * beta[t + 1].get(s2, 0.0)
                b[s] = tot
            beta[t] = b
        self._alpha, self._beta, self._trans = alpha, beta, trans
        self._evidence = evidence

    def gamma(self, t: int) -> Dict[State, float]:
        """Smoothed marginal P(state_t | evidence)."""
        g = {s: p * self._beta[t].get(s, 0.0) / self.likelihood
             for s, p in self._alpha[t].items()}
        return {s: p for s, p in g.items() if p > 0}

    def event_posterior(self, t: int) -> Dict[object, float]:
        """P(event at slice t | evidence); key None = quiesce/done."""
        xi: Dict[object, float] = {}
        for (_s, e), w in self.state_event_posterior(t).items():
            xi[e] = xi.get(e, 0.0) + w
        return xi

    def state_event_posterior(self, t: int) -> Dict[tuple, float]:
        """P(state_t = s, event at t = e | evidence) — the joint the
        EM E-step needs: expected counts must attribute an event to
        the SCOPE VALUATION it fired under, which lives in s."""
        xi: Dict[tuple, float] = {}
        for s, p in self._alpha[t].items():
            for e, pe, s2 in self._trans[t][s]:
                ev = _passes(self._evidence, t + 1, s2)
                w = p * pe * ev * self._beta[t + 1].get(s2, 0.0)
                if w > 0:
                    xi[(s, e)] = xi.get((s, e), 0.0) \
                        + w / self.likelihood
        return xi


def behavior_posterior(builders: Dict[object,
                                      Callable[[], ForwardProjection]],
                       evidence: Evidence,
                       prior: Optional[Dict[object, float]] = None):
    """Posterior over hypothesis measures given shared evidence.
    Returns (posterior, likelihoods); hypotheses with zero-likelihood
    evidence get posterior 0 (not an error — losing IS the signal)."""
    liks: Dict[object, float] = {}
    for h, build in builders.items():
        try:
            liks[h] = Conditioned(build(), evidence).likelihood
        except ValueError:
            liks[h] = 0.0
    post = {h: liks[h] * (prior or {}).get(h, 1.0) for h in builders}
    z = sum(post.values())
    if z <= 0.0:
        raise ValueError("evidence impossible under every hypothesis")
    return {h: p / z for h, p in post.items()}, liks


@dataclass
class MapResult:
    prob: float                # exp(logp): may underflow to 0.0 on
    events: Tuple              # very long horizons — logp is exact
    final: State
    logp: float = 0.0


def _map_tiekey(pr, s, e):
    """Canonical order among equal-probability predecessors: the
    event's LABEL (stage/name, stable under event-table reordering
    and session history; quiesce sorts last), then the predecessor
    state. Dict/insertion order is NOT canonical — the same scene
    reached through different session histories enumerates it
    differently, and the MAP must not depend on that."""
    if e is None:
        return ("￿(quiesce)", s)
    _, stage, ev = pr.events[e]
    return (f"{stage}/{getattr(ev, 'name', '?')}", s)


def map_derivation(pr: ForwardProjection,
                   evidence: Optional[Evidence] = None) -> MapResult:
    """Viterbi: the most probable event sequence over the horizon,
    optionally evidence-conditioned. Log-space (a long horizon's
    product underflows linear doubles near step ~1000 and every
    candidate then ties at 0), back-pointers per step instead of a
    path tuple per state (O(Σ support) total, not O(support·t²)),
    and canonical tie-breaks (see _map_tiekey)."""
    from math import exp, log
    init = pr.program.initial_state()
    if _passes(evidence, 0, init) <= 0:
        raise ValueError("evidence rejects the initial state")
    delta: Dict[State, float] = {init: 0.0}          # state -> logp
    back: list = []            # per step: state -> (prev_state, e)
    for t in range(pr.steps):
        nxt: Dict[State, float] = {}
        par: Dict[State, Tuple[State, object]] = {}
        for s in sorted(delta):
            lp = delta[s]
            stage, kappa = s
            counts = dict(kappa)
            for e, pe in pr._choice(stage, counts):
                s2 = apply_event(pr.events, stage, counts, e)
                w = _passes(evidence, t + 1, s2)
                if w <= 0:
                    continue
                cand = lp + log(pe) + (log(w) if w != 1.0 else 0.0)
                if s2 not in nxt or cand > nxt[s2] or (
                        cand == nxt[s2]
                        and _map_tiekey(pr, s, e)
                        < _map_tiekey(pr, *par[s2])):
                    nxt[s2], par[s2] = cand, (s, e)
        if not nxt:
            raise ValueError("evidence has zero likelihood")
        delta = nxt
        back.append(par)
    best_lp = max(delta.values())
    best = min(s for s in delta if delta[s] == best_lp)
    logp, path, s2 = delta[best], [], best
    for par in reversed(back):
        s2, e = par[s2]
        path.append(e)
    return MapResult(exp(logp), tuple(reversed(path)), best, logp)


def surprise(pr: ForwardProjection, t: int, observed: State,
             rare_ratio: float = 0.05):
    """Likelihood-ratio surprise of an observed state against the
    slice-t predictive distribution. kind: 'unmodeled' (zero mass),
    'rare' (mass / modal mass < rare_ratio), or None."""
    dist = pr.history[t]
    p = dist.get(observed, 0.0)
    p_modal = max(dist.values())
    modal = max(dist, key=lambda s: dist[s])
    ratio = p / p_modal if p_modal > 0 else 0.0
    kind = "unmodeled" if p == 0.0 else \
        ("rare" if ratio < rare_ratio else None)
    return {"kind": kind, "p": p, "ratio": ratio,
            "support": len(dist), "modal": modal,
            "p_modal": p_modal}
