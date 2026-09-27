"""The sampling evaluator — the scale bridge past the exact cliff.

SampledProjection is a drop-in for ForwardProjection wherever the
consumer reads `final` / `history` / `final_distribution` /
`p_done`: same programs, same catalogs, same weight-port policies,
same scorers (expected_on, marginal_on, crash-mass readers all
duck-type) — only the evaluator swaps. Where exact support
enumeration dies at ~2^14 states, this trades exactness for O(n)
trajectories with O(1/sqrt(n)) error.

It shares the filter's transition machinery (choice_weight +
apply_event) and the sampler's policy semantics, so the three
evaluators (exact filter, trace sampler, this) draw from ONE
measure — preservation-tested."""

import random
from typing import Dict, List

from .filter import DONE, apply_event
from .program import Program, State
from .weight import choice_weight


class SampledProjection:
    def __init__(self, program: Program, steps: int, n: int = 1000,
                 seed: int = 7, policy=None, history: bool = False):
        self.program = program
        self.steps = steps
        self.n = n
        events = program.events()
        by_tier: Dict[str, list] = {}
        for i, (k, s, ev) in enumerate(events):
            by_tier.setdefault((k, s), []).append((i, ev))
        rng = random.Random(seed)
        final: Dict[State, int] = {}
        hist: List[Dict[State, int]] = \
            [dict() for _ in range(steps + 1)] if history else []
        for _ in range(n):
            stage, kappa = program.initial_state()
            counts = dict(kappa)
            if history:
                hist[0][(stage, kappa)] = \
                    hist[0].get((stage, kappa), 0) + 1
            state = (stage, kappa)
            for t in range(1, steps + 1):
                if stage != DONE:
                    chosen = None
                    for kind in ("clause", "link"):
                        ws = [(i, choice_weight(ev, counts))
                              for i, ev in by_tier.get((kind, stage),
                                                       ())]
                        ws = [(i, w) for i, w in ws if w > 0]
                        if ws and policy is not None \
                                and kind == "clause":
                            ws = [(i, w) for i, w
                                  in policy(stage, counts, ws)
                                  if w > 0]
                        if ws:
                            chosen = self._pick(rng, ws)
                            break
                    state = apply_event(events, stage, counts, chosen)
                    stage, kappa = state
                    counts = dict(kappa)
                if history:
                    hist[t][state] = hist[t].get(state, 0) + 1
            final[state] = final.get(state, 0) + 1
        self.final = {s: c / n for s, c in final.items()}
        self.history = [{s: c / n for s, c in d.items()}
                        for d in hist] if history else None

    @staticmethod
    def _pick(rng, ws):
        tot = sum(w for _, w in ws)
        x = rng.random() * tot
        for i, w in ws:
            x -= w
            if x <= 0:
                return i
        return ws[-1][0]

    # -- the ForwardProjection query subset ------------------------------

    def final_distribution(self, pred: str) -> List[float]:
        by_count: Dict[int, float] = {}
        for (_, kappa), p in self.final.items():
            c = dict(kappa).get(pred, 0)
            by_count[c] = by_count.get(c, 0.0) + p
        card = max(2, max(by_count) + 1)
        return [by_count.get(c, 0.0) for c in range(card)]

    def p_done(self) -> float:
        return sum(p for (s, _), p in self.final.items() if s == DONE)

    def n_reachable(self) -> int:
        return len(self.final)
