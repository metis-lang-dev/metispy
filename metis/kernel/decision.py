"""The generic decision loop — planning as evaluation of intervened
measures.

Per tick, a domain supplies:
  candidates   the SAFE action set (the domain's hard shield — RSS for
               road — runs BEFORE the measure; an empty set is `doomed`
               and the domain's fallback action is returned unscored)
  do(a)        the intervention: commit candidate a, returning the
               mid-situation the horizon is projected from (typically
               an initial-multiset edit)
  project(sit) build + filter the intervened measure (a
               ForwardProjection or anything score() can read)
  score(pr)    the lexicographic score tuple; by convention component 0
               is the hard-safety mass (P(crash)), then the domain
               objectives (larger-is-better entries negated)

Every component is quantized with round9 — tie-breaking must never
hinge on float summation-order noise (python round(x, 9), the exact
oracle discipline; found by lolli's differential harness). Ties resolve
to the earliest candidate, matching the oracle's dict-order min()."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


def round9(x: float) -> float:
    return round(float(x), 9)


@dataclass
class PlanResult:
    action: Any
    doomed: bool
    dof: int                              # |candidates| (freedom now)
    scores: Dict[Any, Tuple[float, ...]]
    projections: Dict[Any, Any]


def plan(candidates: Iterable[Any],
         do: Callable[[Any], Any],
         project: Callable[[Any], Any],
         score: Callable[[Any], Tuple[float, ...]],
         doomed_action: Optional[Any] = None) -> PlanResult:
    candidates = list(candidates)
    if not candidates:
        return PlanResult(doomed_action, True, 0, {}, {})
    scores: Dict[Any, Tuple[float, ...]] = {}
    projections: Dict[Any, Any] = {}
    best = None
    for a in candidates:
        pr = project(do(a))
        s = tuple(round9(x) for x in score(pr))
        scores[a] = s
        projections[a] = pr
        if best is None or s < scores[best]:
            best = a
    return PlanResult(best, False, len(candidates), scores, projections)


def cached_projector(cache, steps: int, policy_builder=None,
                     policy_key=None):
    """A plan()-shaped `project` over a ProjectionCache — the
    receding-horizon seam: candidates whose intervened program was
    projected before (same program_key, steps, policy_key) are served
    from the cache; on misses the policy is built AGAINST THE
    CANONICAL program (index safety). policy_key must name the
    policy's CONTENT; None disables caching (function identity is
    not content identity)."""
    def project(program):
        return cache.project(program, steps, policy_key=policy_key,
                             policy_builder=policy_builder)
    return project


class ClampView:
    """The projection surface plan() sees on the FACTORED path: one
    shared compiled structure (FactoredProgram), one evidence
    assignment per candidate — do() as a clamp, never a re-ground
    (decision-over-IR: the C++ IrEval mirrors this exactly)."""

    def __init__(self, fp, init):
        self.fp = fp
        self.init = dict(init)

    def marginal(self, atom: str) -> float:
        return self.fp.marginal(atom, init=self.init)

    def likelihood(self, observed: Dict[str, int]) -> float:
        return self.fp.likelihood(observed, init=self.init)


def clamp_domain(fp, candidates_fn, evidence_fn, score_fn,
                 doomed=None):
    """The fw.Evaluator domain seam over ONE FactoredProgram:
    `evidence_fn(sit, a)` maps a candidate to an assignment over the
    declared unknowns; project wraps it as a ClampView; score reads
    marginals. The factored analog of fw.catalog_domain."""
    def factory(sit):
        return (candidates_fn(sit),
                lambda a: evidence_fn(sit, a),
                lambda init: ClampView(fp, init),
                lambda vw: score_fn(vw, sit),
                doomed)
    return factory
