"""The PORTABLE reference sampler (docs/rewrite/06-sampler-wire.md)
— the evaluator ladder's bottom rung, specified for byte-identical
traces across Python, OCaml and C++, on the FULL language (no
class-F restriction: multiplicities, quiescence, branch-dependent
schedules all sample here).

Same measure as kernel/sampler.py (identical tier and weight
semantics — Thm 1's distribution); different, spec-pinned
mechanics: canonical event order (presentation-invariant traces)
and splitmix64. The kernel sampler stays as-is for the pinned
in-repo gates; this one is the cross-language artifact rung."""

from typing import Dict, List, Optional, Tuple

from .nets import _clause_line, _link_line
from .program import Program, State, pack_counts
from .weight import choice_weight

DONE = "(done)"
_M64 = (1 << 64) - 1


class SplitMix64:
    """splitmix64, normative constants (06-sampler-wire §3)."""

    def __init__(self, seed: int):
        self.state = seed & _M64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _M64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
        return z ^ (z >> 31)

    def unit(self) -> float:
        return (self.next() >> 11) * 2.0 ** -53


def canonical_events(program: Program) -> List[Tuple]:
    """The normative event order: per stage in declared order,
    clauses sorted bytewise by canonical line; then links sorted
    bytewise. Traces index into THIS list — presentation-invariant
    by construction."""
    ev = []
    for s in program.stages:
        for c in sorted(program.stages[s], key=_clause_line):
            ev.append(("clause", s, c))
    for lk in sorted(program.links, key=_link_line):
        ev.append(("link", lk.pre, lk))
    return ev


def run_trace(program: Program, steps: int, seed: int,
              policy=None) -> Tuple[Tuple[int, ...], State]:
    """One seeded trajectory: (canonical event indices, final
    state). Selection per the normative step (06-sampler-wire §4):
    left-to-right sum, threshold = unit() * total, first candidate
    with threshold < cumsum. `policy` reweights tier 0 exactly as
    the filter's hook (production bindings enter here; the parity
    gates run prior-only)."""
    events = canonical_events(program)
    rng = SplitMix64(seed)
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
                total = 0.0
                for _, w in ws:
                    total += w
                threshold = rng.unit() * total
                acc = 0.0
                chosen = ws[-1][0]
                for i, w in ws:
                    acc += w
                    if threshold < acc:
                        chosen = i
                        break
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


class TraceSampler:
    """run_trace for many runs of one program (the watch's sampled tier,
    rddl-watch s-cfe94dca): the same draws — each step's candidates in
    canonical order, the same weights (an event's prior times its
    integer transition count, as choice_weight), the same policy and
    threshold — so `run(steps, seed)` is run_trace's final state; but
    the event table is compiled once and a step weighs only the events
    whose rarest premise is present (the filter's tier index), not
    every event of the stage."""

    def __init__(self, program: Program, policy=None):
        self.program = program
        self.events = canonical_events(program)
        self.policy = policy
        freq: Dict[str, int] = {}
        for _, _, ev in self.events:
            for a in set(ev.consume) | set(ev.persist):
                freq[a] = freq.get(a, 0) + 1
        # (stage, kind) -> (no premise, {rarest premise: [compiled event]});
        # a compiled event: (index, prior, consume items, persist items)
        index = {}
        for i, (kind, stage, ev) in enumerate(self.events):
            always, by = index.setdefault((stage, kind), ([], {}))
            c = (i, ev.weight, tuple(ev.consume.items()), tuple(ev.persist.items()))
            prem = set(ev.consume) | set(ev.persist)
            if not prem:
                always.append(c)
            else:
                by.setdefault(min(prem, key=lambda a: (freq[a], a)), []).append(c)
        self.index = index

    def _tier(self, stage, kind, counts):
        always, by = self.index.get((stage, kind), ((), {}))
        cand = list(always)
        for a in counts:
            got = by.get(a)
            if got:
                cand.extend(got)
        cand.sort()
        get = counts.get
        ws = []
        for i, prior, cons, pers in cand:
            n = 1                       # weight.transition_count, inlined
            for a, k in cons:
                c = get(a, 0)
                for j in range(k):
                    n *= c - j
                    if n <= 0:
                        break
                if n <= 0:
                    break
            if n > 0:
                for a, k in pers:
                    n *= get(a, 0) ** k
                    if not n:
                        break
            if n > 0:
                w = prior * n
                if w > 0:
                    ws.append((i, w))
        return ws

    def run(self, steps: int, seed: int) -> State:
        rng = SplitMix64(seed)
        stage, kappa = self.program.initial_state()
        counts = dict(kappa)
        events, policy = self.events, self.policy
        for _ in range(steps):
            if stage == DONE:
                break
            chosen = None
            for kind in ("clause", "link"):
                ws = self._tier(stage, kind, counts)
                if ws and policy is not None and kind == "clause":
                    ws = [(i, w) for i, w in policy(stage, counts, ws)
                          if w > 0]
                if ws:
                    total = 0.0
                    for _, w in ws:
                        total += w
                    threshold = rng.unit() * total
                    acc = 0.0
                    chosen = ws[-1][0]
                    for i, w in ws:
                        acc += w
                        if threshold < acc:
                            chosen = i
                            break
                    break
            if chosen is None:
                stage = DONE
                continue
            kind, _, ev = events[chosen]
            for q, n in ev.consume.items():
                left = counts[q] - n
                if left > 0:
                    counts[q] = left
                else:
                    del counts[q]
            for q, n in ev.produce.items():
                counts[q] = counts.get(q, 0) + n
            if kind == "link":
                stage = ev.post
        return stage, pack_counts(counts)


def final_line(seed: int, state: State) -> str:
    """The F-line rendering of a final state (sorted atom:count)."""
    stage, kappa = state
    items = sorted(f"{a}:{n}" for a, n in kappa)
    tail = (" " + " ".join(items)) if items else ""
    return f"F {seed} {stage}{tail}"


def sample_final(program: Program, steps: int, n: int,
                 seed: int = 1, policy=None) -> Dict[State, float]:
    """Empirical final-state distribution over n runs seeded
    seed..seed+n-1 (independent streams — splitmix64's design
    use)."""
    freq: Dict[State, int] = {}
    for k in range(n):
        _, s = run_trace(program, steps, seed + k, policy=policy)
        freq[s] = freq.get(s, 0) + 1
    return {s: c / n for s, c in freq.items()}
