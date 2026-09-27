"""The surprise record — leg 1's logging contract (docs/learning.md):
evaluation is DETERMINISTIC given (artifact, init, observation), so
the runtime logs pointers, not traces; the offline side re-derives
the full context (support, posteriors, MAP explanation) by re-running
the exact machinery — here.

A record is JSON-able and small: the artifact identity (program_key +
caller-supplied meta such as pack versions), the tick's init
multiset, the partially-observed outcome (atom -> count over the
OBSERVED vocabulary only; absent atoms may be asserted with count 0),
the horizon, and the verdict computed at logging time. Zero
likelihood is the STRUCTURAL signal (no parameter can create missing
support — a missing behavior is a missing rule).

`recover` replays a record against a CANDIDATE artifact — typically
the base catalog plus a shadow-admitted pack — returning likelihood,
the smoothed posterior and the MAP derivation (the explanation
object). The shadow-pack mode is a usage pattern, not an API:
condition and explain with the candidate, keep plan() on the base
until the pack is approved."""

from typing import Dict, Optional

from .filter import ForwardProjection
from .nets import program_key
from .posterior import Conditioned, map_derivation


def _evidence(observed: Dict[str, int], steps: int):
    def pred(s):
        counts = dict(s[1])
        return all(counts.get(a, 0) == n for a, n in observed.items())
    return {steps: pred}


def surprise_record(program, steps: int, observed: Dict[str, int],
                    policy=None, rare_ratio: float = 0.05,
                    meta: Optional[dict] = None) -> dict:
    """Log an observed (partial) outcome against the deployed
    artifact. kind: 'unmodeled' (zero likelihood — the leg-1
    trigger), 'rare' (likelihood below rare_ratio of the modal
    matching mass... below rare_ratio outright), or None."""
    pr = ForwardProjection(program, steps, policy=policy)
    like = sum(p for s, p in pr.final.items()
               if _evidence(observed, steps)[steps](s))
    kind = "unmodeled" if like == 0.0 else \
        ("rare" if like < rare_ratio else None)
    return {
        "format": "metis-surprise", "version": 1,
        "program_key": program_key(program),
        "steps": steps,
        "init": dict(program.init),
        "observed": dict(observed),
        "likelihood": like,
        "kind": kind,
        "meta": dict(meta or {}),
    }


def recover(record: dict, program, steps: Optional[int] = None,
            policy=None) -> dict:
    """Replay a record against a candidate artifact (offline). If the
    candidate is the RECORDED artifact, verify the key; a different
    key means a shadow candidate is being scored — allowed, that is
    the point. Returns the full re-derived context; likelihood 0
    keeps explanation None (the candidate does not explain it
    either)."""
    steps = steps if steps is not None else record["steps"]
    evidence = _evidence(record["observed"], steps)
    pr = ForwardProjection(program, steps, policy=policy)
    try:
        cond = Conditioned(pr, evidence)
    except ValueError:
        return {"likelihood": 0.0, "explains": False,
                "posterior": None, "explanation": None}
    m = map_derivation(pr, evidence)
    names = [pr.events[e][2].name for e in m.events if e is not None]
    return {"likelihood": cond.likelihood, "explains": True,
            "posterior": cond, "explanation": m,
            "explanation_events": names}
