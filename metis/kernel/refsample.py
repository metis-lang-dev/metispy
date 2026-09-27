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
