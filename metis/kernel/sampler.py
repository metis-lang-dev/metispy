"""The seeded reference interpreter — one weighted-picker trajectory at a
time.

This is the operational side of the preservation theorem: the filter's
measure must equal the distribution this sampler draws from (the role
`ceptre --samples` played for ll2pg, brought in-repo so preservation is
CI-testable forever, with no external binary). Same three-tier CHOICE,
same weights, same policy hook semantics as the filter (tier-0
reweighting; an emptied tier falls through to links) — so sampled and
exact evaluation draw from ONE measure."""

import random
from typing import Dict, Tuple

from .program import Program, State, pack_counts
from .weight import choice_weight

DONE = "(done)"


def _pick(rng: random.Random, ws):
    tot = sum(w for _, w in ws)
    x = rng.random() * tot
    for i, w in ws:
        x -= w
        if x <= 0:
            return i
    return ws[-1][0]


def run_trace(program: Program, steps: int, rng: random.Random,
              policy=None):
    """One trajectory, returning (event-index trace, final state). The
    trace records real events only (indices into program.events());
    quiescence/DONE ends recording. `policy` reweights tier-0
    candidates exactly as in the filter."""
    events = program.events()
    stage, kappa = program.initial_state()
    counts = dict(kappa)
    trace = []
    for _ in range(steps):
        if stage == DONE:
            break
        chosen = None
        for kind in ("clause", "link"):
            ws = [(i, choice_weight(ev, counts))
                  for i, (k, s, ev) in enumerate(events)
                  if k == kind and s == stage]
            ws = [(i, w) for i, w in ws if w > 0]
            if ws and policy is not None and kind == "clause":
                ws = [(i, w) for i, w in policy(stage, counts, ws)
                      if w > 0]
            if ws:
                chosen = _pick(rng, ws)
                break
        if chosen is None:
            stage = DONE
            continue
        trace.append(chosen)
        kind, _, ev = events[chosen]
        for q, n in ev.consume.items():
            counts[q] = counts.get(q, 0) - n
        for q, n in ev.produce.items():
            counts[q] = counts.get(q, 0) + n
        counts = {q: n for q, n in counts.items() if n > 0}
        if kind == "link":
            stage = ev.post
    return tuple(trace), (stage, pack_counts(counts))


def run_once(program: Program, steps: int, rng: random.Random,
             policy=None) -> State:
    return run_trace(program, steps, rng, policy=policy)[1]


def sample_final(program: Program, steps: int, n: int,
                 seed: int = 7, policy=None) -> Dict[State, float]:
    """Empirical final-state distribution over n seeded runs."""
    rng = random.Random(seed)
    freq: Dict[State, int] = {}
    for _ in range(n):
        s = run_once(program, steps, rng, policy=policy)
        freq[s] = freq.get(s, 0) + 1
    return {s: c / n for s, c in freq.items()}
