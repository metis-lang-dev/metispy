"""Embedded interpreter directives (00-language-spec §8).

A `.llp` may carry its own case as `#`-directive lines — the Ceptre
lineage's in-file `#trace`, restored as a comment class: the lexer
drops `#...` like a `%` comment, so the compiled catalog and its
canonical key are byte-identical with or without them. A directive is
a LINE whose first non-blank char is '#'; its body is one
case-manifest line, same vocabulary as the metisc manifest driver
(init/steps/seeds/unknown/query/lik). At most one `init` line per
atom. Certified expectations stay in manifests — this channel is for
interactive discovery (metisc --run, the editor's Send to
Interpreter)."""

from typing import Dict, List, Tuple


class DirectiveError(ValueError):
    pass


def _clamp(tok: str) -> Tuple[str, int]:
    atom, eq, val = tok.partition("=")
    if not eq:
        raise DirectiveError(f"bad clamp: {tok}")
    return (atom, int(val))


def embedded_case(text: str) -> Dict:
    """Scan a .llp source for '#'-directive lines; return the case as
    {init, steps, seeds, unknowns, queries, liks}."""
    case = {"init": {}, "steps": 0, "seeds": 0,
            "unknowns": [], "queries": [], "liks": []}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("#"):
            continue
        parts = line[1:].split()
        if not parts:
            continue
        key, args = parts[0], parts[1:]
        if key == "init" and len(args) == 2:
            case["init"][args[0]] = int(args[1])
        elif key == "steps" and len(args) == 1:
            case["steps"] = int(args[0])
        elif key == "seeds" and len(args) == 1:
            case["seeds"] = int(args[0])
        elif key == "unknown" and len(args) == 1:
            case["unknowns"].append(args[0])
        elif key == "query" and len(args) == 1:
            case["queries"].append(args[0])
        elif key == "lik" and args:
            case["liks"].append([_clamp(t) for t in args])
        else:
            raise DirectiveError(f"bad directive: {line}")
    return case


def render_run(name: str, case: Dict, prog) -> str:
    """The --run replay, mirroring metisc byte-for-byte on the shared
    lines (I/T/F: same reference sampler, same seeds), plus the exact
    presence probabilities the python referee can add (metisc's Q is
    the sampled hit count; here it carries `exact=` when the filter
    is tractable)."""
    from metis.kernel.refsample import (canonical_events, final_line,
                                        run_trace)
    steps = case["steps"]
    if not steps:
        raise DirectiveError("no #steps directive")
    seeds = case["seeds"] or 1
    out = [f"case {name} steps={steps} seeds={seeds}"]
    for atom, n in case["init"].items():
        out.append(f"I {atom} {n}")
    events = canonical_events(prog)
    hits = {q: 0 for q in case["queries"]}
    for seed in range(1, seeds + 1):
        trace, final = run_trace(prog, steps, seed)
        names = " ".join(events[i][2].name for i in trace) or "-"
        out.append(f"T {seed} {names}")
        out.append(final_line(seed, final))
        _, kappa = final
        for q in hits:
            if any(a == q and n > 0 for a, n in kappa):
                hits[q] += 1
    exact = _exact_presence(prog, steps, list(hits))
    for q, k in hits.items():
        line = f"Q {q} {k}/{seeds}"
        if q in exact:
            line += f" exact={exact[q]:.6g}"
        out.append(line)
    return "\n".join(out)


_EXACT_EVENT_CAP = 512


def _exact_presence(prog, steps: int, queries: List[str]) -> Dict:
    """Presence marginals from the exact filter, where tractable —
    best-effort: any failure (or a large grounding) just drops the
    `exact=` annotation, never the run."""
    if not queries or len(prog.events()) > _EXACT_EVENT_CAP:
        return {}
    try:
        from metis.kernel import ForwardProjection
        pr = ForwardProjection(prog, steps)
        return {q: sum(p for (_, k), p in pr.final.items()
                       if any(a == q and n > 0 for a, n in k))
                for q in queries}
    except Exception:
        return {}
