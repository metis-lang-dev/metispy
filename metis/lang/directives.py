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


def squash_atoms(line: str) -> str:
    """`#init at(e0, f0) 1` -> `#init at(e0,f0) 1`: whitespace inside an
    atom's parentheses is formatting, never a token boundary (models
    and people write the spaced form naturally)."""
    import re
    return re.sub(r"\(([^()]*)\)",
                  lambda m: "(" + re.sub(r"\s+", "", m.group(1)) + ")",
                  line)


def embedded_case(text: str) -> Dict:
    """Scan a .llp source for '#'-directive lines; return the case as
    {init, steps, seeds, unknowns, queries, liks}."""
    case = {"init": {}, "steps": 0, "seeds": 0,
            "unknowns": [], "queries": [], "liks": [],
            "interactive": [], "plugin": "", "notch": []}
    for raw in text.splitlines():
        line = squash_atoms(raw.strip())
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
        elif key == "interactive" and len(args) == 1:
            # the additive &: the stage's CHOICE is environment-
            # resolved under metisc --run; in this REPL the prompt
            # itself is the external chooser, so the directive is
            # recorded but drives nothing here.
            case["interactive"].append(args[0])
        elif key == "plugin" and len(args) == 1:
            # scene-level like #init: the port plugin that binds the
            # catalog's weight ports and drives its env
            # (PLUGIN-PORTS.md). Recorded here; only the interactive
            # TERMINAL shell imports it (D4) — every other reader of
            # the case ignores it.
            case["plugin"] = args[0]
        elif key == "notch" and len(args) == 1:
            # a fitted overlay file (<stem>.notch.json) shipped on the
            # plugin's registry at serve, as `notch load` ships it; the
            # catalog stays the law-discovery's prior when left out
            case["notch"].append(args[0])
        else:
            raise DirectiveError(f"bad directive: {line}")
    return case


def _pretty_trace(seed: int, rows) -> List[str]:
    """A trace as its Foata beats — the naturally induced grouping
    (nets.foata_levels, the layering run_key hashes): one block per
    causal level, events within a beat are pairwise independent
    (concurrent; their order carries no information). Rule names
    column-aligned."""
    from metis.kernel.nets import foata_levels
    if not rows:
        return [f"T {seed} -"]
    levels = foata_levels(rows)
    beats = max(levels) + 1
    out = [f"T {seed} — {len(rows)} events, {beats} beats"]
    name_w = 0
    split = []
    for _, _, ev in rows:
        base, bra, rest = ev.name.partition("[")
        split.append((base, bra + rest))
        name_w = max(name_w, len(base))
    beat_w = len(str(beats))
    # reorder into the normal form proper: one block per layer (a
    # legitimate reordering — Theorem 4: same trace class, same
    # weight); within a beat, firing order
    for lvl in range(beats):
        first = True
        for (base, binds), l in zip(split, levels):
            if l != lvl:
                continue
            tag = f"{lvl + 1:>{beat_w}}" if first else " " * beat_w
            out.append(f"  {tag} | {base:<{name_w}} {binds}")
            first = False
    return out


def state_table(kappa) -> List[str]:
    """A multiset tabulated by predicate: one row per pred, `(args)`
    terms (`:n` when n > 1, `·` for a bare 0-ary presence), wrapped
    and aligned. The shared rendering of every state the shell shows
    (trace finals, `state`, `map`); zero counts are not state."""
    by_pred: Dict[str, List[str]] = {}
    for atom, n in sorted(kappa):
        if n <= 0:
            continue
        pred, bra, rest = atom.partition("(")
        item = (bra + rest) if bra else ""
        if n > 1:
            item += f":{n}"
        by_pred.setdefault(pred, []).append(item if item else "·")
    out: List[str] = []
    if not by_pred:
        return out
    pred_w = max(len(p) for p in by_pred)
    for pred, items in by_pred.items():
        line = f"  {pred:<{pred_w}} "
        indent = " " * len(line)
        for it in items:
            if len(line) + len(it) + 1 > 78 and line.strip() != pred:
                out.append(line.rstrip())
                line = indent
            line += " " + it
        out.append(line.rstrip())
    return out


def _pretty_state(seed: int, stage: str, kappa) -> List[str]:
    return [f"F {seed} {stage}"] + state_table(kappa)


def render_run(name: str, case: Dict, prog, pretty: bool = False,
               notes=None) -> str:
    """The --run replay. Default (wire) mode mirrors metisc
    byte-for-byte on the shared lines (I/T/F: same reference sampler,
    same seeds) — the parity surface, pinned by tests. `pretty=True`
    is the human surface (REPL, editor): T becomes the trace's Foata
    beats, F a per-predicate state table; I/Q and the exact=
    annotations are identical in both modes."""
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
        if pretty:
            rows = [events[i] for i in trace]
            out += _pretty_trace(seed, rows)
            stage, kappa = final
            out += _pretty_state(seed, stage, kappa)
        else:
            names = " ".join(events[i][2].name for i in trace) or "-"
            out.append(f"T {seed} {names}")
            out.append(final_line(seed, final))
        _, kappa = final
        for q in hits:
            if any(a == q and n > 0 for a, n in kappa):
                hits[q] += 1
    exact, why = _exact_presence(prog, steps, list(hits))
    if why is not None and notes is not None:
        from metis.kernel.diagnostics import note
        qs = ", ".join(hits)
        notes.append(note(
            "exact-unreachable",
            f"exact presence out of reach for {qs} at {steps} steps: {why}",
            "trace", (f"trace ({steps} steps; #query {qs})",),
            (("queries", qs), ("reason", why))))
    for q, k in hits.items():
        line = f"Q {q} {k}/{seeds}"
        if q in exact:
            line += (f" exact={exact[q]:.6g}" if exact[q] is not None
                     else " exact=(out of exact reach)")
        out.append(line)
    return "\n".join(out)


_EXACT_EVENT_CAP = 512


def _exact_presence(prog, steps: int, queries: List[str]) -> Dict:
    """Presence marginals from the exact filter, where tractable —
    best-effort, but never silent about WHY: over the state budget
    (or out of memory) every query maps to None and the Q line says
    so, instead of the annotation just vanishing. Returns (marginals,
    reason): reason names why exact numbers are missing, else None."""
    if not queries:
        return {}, None
    if len(prog.events()) > _EXACT_EVENT_CAP:
        # say so, don't vanish
        return ({q: None for q in queries},
                f"grounding of {len(prog.events())} events exceeds the "
                f"exact cap of {_EXACT_EVENT_CAP}")
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded)
    try:
        pr = ForwardProjection(prog, steps, max_states=STATE_BUDGET,
                               keep_history=False)
        return ({q: sum(p for (_, k), p in pr.final.items()
                        if any(a == q and n > 0 for a, n in k))
                 for q in queries}, None)
    except StateBudgetExceeded as e:
        return {q: None for q in queries}, f"state budget: {e}"
    except MemoryError:
        return {q: None for q in queries}, "out of memory"
    except Exception:
        return {}, None
