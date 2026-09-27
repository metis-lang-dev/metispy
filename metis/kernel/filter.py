"""Exact forward filtering — evaluation of the program's measure in the
generative direction.

UPDATE is deterministic given the chosen event, so each slice's joint is
supported on the REACHABLE states only; filtering the distribution over
that support is exact inference on the denoted k-step DBN (the ll2pg
three-evaluator equality, carried here as the oracle-parity tests).

CHOICE per state: three tiers —
  1. enabled clauses of the active stage, P ∝ prior × transition count,
     reweightable/prunable by the policy hook;
  2. at quiescence (tier 1 empty): enabled links of the stage;
  3. else: done — absorbing, counts frozen.

The policy hook `policy(stage, counts, candidates)` receives
[(event_index, weight)] and returns the reweighted list (renormalized
here). It is the measure-level plug for safety shields, behavior rows and
drama managers — interventions on the SAME measure, keeping inference
exact."""

from typing import Dict, List, Optional

from .program import Program, State, pack_counts
from .weight import choice_weight

DONE = "(done)"


def apply_event(events, stage: str, counts: Dict[str, int], e) -> State:
    """Apply event index e (or None = quiesce) to a state; THE
    transition function — shared by the filter, the smoother and the
    Viterbi pass so they cannot drift."""
    if e is None:
        return (DONE, pack_counts(counts))
    kind, _, ev = events[e]
    knew = dict(counts)
    for q, n in ev.consume.items():
        knew[q] = knew.get(q, 0) - n
    for q, n in ev.produce.items():
        knew[q] = knew.get(q, 0) + n
    return (ev.post if kind == "link" else stage, pack_counts(knew))


class ForwardProjection:
    def __init__(self, program: Program, steps: int, policy=None):
        self.program = program
        self.steps = steps
        self.policy = policy
        self.events = program.events()
        self.event_dists: List[List[float]] = []

        # per (stage, kind): events indexed by their RAREST consume atom,
        # so a state only inspects events that could possibly be enabled
        # (plus the few with no linear premises). Pure optimization of
        # the same CHOICE — the ll2pg oracle's tier index, ported after
        # profiling on the golden corpus (4-car blocks: 84 s -> tractable).
        self._tier_index = {}
        for stage in program.stage_names:
            for kind in ("clause", "link"):
                tier = [(i, ev) for i, (k, s, ev) in enumerate(self.events)
                        if k == kind and s == stage]
                freq: Dict[str, int] = {}
                for _, ev in tier:
                    for a in ev.consume:
                        freq[a] = freq.get(a, 0) + 1
                index, rest = {}, []
                for i, ev in tier:
                    if ev.consume:
                        key = min(ev.consume, key=lambda a: freq[a])
                        index.setdefault(key, []).append((i, ev))
                    else:
                        rest.append((i, ev))
                self._tier_index[(stage, kind)] = (index, rest)

        dist: Dict[State, float] = {program.initial_state(): 1.0}
        self.history = [dist]
        for _ in range(steps):
            dist = self._step(dist)
            self.history.append(dist)
        self.final = dist

    # -- CHOICE ----------------------------------------------------------

    def _choice(self, stage: str, counts: Dict[str, int]):
        """[(event index or None, probability)] for one state."""
        if stage == DONE:
            return [(None, 1.0)]

        def candidates(kind):
            index, rest = self._tier_index[(stage, kind)]
            out = list(rest)
            for a in counts:
                out.extend(index.get(a, ()))
            return out

        for tierno, kind in ((0, "clause"), (1, "link")):
            ws = [(i, choice_weight(ev, counts))
                  for i, ev in candidates(kind)]
            ws = [(i, w) for i, w in ws if w > 0]
            if ws and self.policy is not None and tierno == 0:
                ws = self.policy(stage, counts, ws)
            tot = sum(w for _, w in ws)
            if tot > 0:
                return [(i, w / tot) for i, w in ws if w > 0]
        return [(None, 1.0)]

    def _step(self, dist):
        out: Dict[State, float] = {}
        evd = [0.0] * (len(self.events) + 1)   # last slot: done/quiesce
        for (stage, kappa), p in dist.items():
            counts = dict(kappa)
            for e, pe in self._choice(stage, counts):
                nxt = apply_event(self.events, stage, counts, e)
                evd[e if e is not None else len(self.events)] += p * pe
                out[nxt] = out.get(nxt, 0.0) + p * pe
        self.event_dists.append(evd)
        return out

    # -- queries (API-compatible with the ll2pg oracle) ------------------

    def final_distribution(self, pred: str) -> List[float]:
        by_count: Dict[int, float] = {}
        for (_, kappa), p in self.final.items():
            c = dict(kappa).get(pred, 0)
            by_count[c] = by_count.get(c, 0.0) + p
        card = max(2, max(by_count) + 1)
        return [by_count.get(c, 0.0) for c in range(card)]

    def stage_distribution(self, t: Optional[int] = None) -> List[float]:
        assert t is None or t == self.steps, "final slice only"
        by_stage: Dict[str, float] = {}
        for (s, _), p in self.final.items():
            by_stage[s] = by_stage.get(s, 0.0) + p
        return [by_stage.get(s, 0.0)
                for s in self.program.stage_names] + \
            [by_stage.get(DONE, 0.0)]

    def p_done(self) -> float:
        return self.stage_distribution()[-1]

    def event_distribution(self, t: int) -> List[float]:
        return list(self.event_dists[t])

    def n_reachable(self) -> int:
        return len(self.final)
