"""The wrapper wire: how the web console (and any other UI) speaks
to the one kernel REPL. Two faces, one dispatcher:

  - `run_intent(shell, line)` — a chat/page-suggested ':verb arg'
    line under the module policy (the whitelist lives HERE and no
    wrapper invents its own). Excluded by design: accept (approval
    is human; a session accept must be a deliberate prompt action),
    export/sink/load (file paths from page viewers), and any llm/
    formalize spec naming a command (shell execution).
  - `serve_json(shell)` — the JSON-lines wire (`metis --json FILE`):
    one {op, arg} per line in, one JSON result per line out.

Ops map to the shell's loop verbs (metis.loop.verbs cmd_*) plus
thin wire readouts of the runtime session (trace/map/marginals/
init/steps/seed/help), all returning dicts."""
import json
import sys
from dataclasses import replace
from typing import Dict

INTENT_VERBS = frozenset(
    ["records", "record", "replay", "mine", "text", "init", "steps",
     "seed", "gate", "revise", "trace", "map", "marginals", "show",
     "formalize", "llm", "narrate", "graph", "help"])


def parse_intent(line: str):
    """One suggested ':verb arg' line -> (op, arg), or None if the
    policy refuses it."""
    line = (line or "").strip()
    if not line.startswith(":"):
        return None
    op, _, arg = line[1:].partition(" ")
    if op not in INTENT_VERBS:
        return None
    if op in ("formalize", "llm") and \
            arg.strip().startswith("command"):
        return None          # shell execution never from page intents
    return op, arg


# -- wire-only readouts of the runtime session ----------------------

def _note(shell, d) -> None:
    """Record a runtime diagnostic on the shell's session notes (spec 08
    step 3; listed by `findings`)."""
    shell.__dict__.setdefault("notes", []).append(d)


def session_findings(shell) -> list:
    """The serving world's current diagnostics, in listing order: the
    served catalog's own (warnings), the last gate/revise report's, and
    the runtime notes the instruments spoke this session."""
    out = list(getattr(shell.doc, "diagnostics", []) or [])
    rep = getattr(getattr(shell, "loop", None), "report", None)
    if rep is not None:
        out += list(getattr(rep, "diagnostics", []) or [])
    return out + list(shell.__dict__.get("notes", []))


def _op_findings(shell, arg):
    """findings [N]: every current diagnostic as its JSON object, or the
    N-th (1-based) alone."""
    import json
    from metis.kernel.diagnostics import to_json
    ds = session_findings(shell)
    if arg.strip():
        try:
            k = int(arg)
        except ValueError:
            return {"ok": False, "error": "usage: findings [N]"}
        if not 1 <= k <= len(ds):
            return {"ok": False, "error": f"no finding {k} "
                                          f"(have {len(ds)})"}
        ds = [ds[k - 1]]
    return {"ok": True, "diagnostics": [json.loads(to_json(d))
                                        for d in ds]}


def _scene_prog(shell, init=None):
    """The scene's ground program, cached per (catalog object, init):
    the read-only instruments (map/marginals/trace) re-serve an
    unchanged scene without re-grounding. reload/accept/revise swap
    the catalog OBJECT, so the cache cannot go stale. (Catalog.ground
    itself stays uncached: its callers may mutate what it returns.)"""
    init = dict(init if init is not None else shell.case["init"])
    cat = shell.doc.catalog
    key = tuple(sorted(init.items()))
    hit = shell.__dict__.get("_scene_cache")
    if hit is not None and hit[0] is cat and hit[1] == key:
        return hit[2]
    prog = cat.ground(init)
    shell.__dict__["_scene_cache"] = (cat, key, prog)
    return prog


def _op_trace(shell, arg):
    from metis.kernel.refsample import canonical_events, run_trace
    prog = _scene_prog(shell)
    seed = int(arg) if arg.strip() else 1
    idx, (stage, kappa) = run_trace(prog, shell.case["steps"] or 6,
                                    seed)
    events = canonical_events(prog)
    return {"ok": True, "seed": seed,
            "events": [f"{events[i][1]}/{events[i][2].name}"
                       for i in idx],
            "final_stage": stage,
            "final": {a: n for a, n in sorted(kappa)}}


def _op_map(shell, arg):
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded, map_derivation)
    try:
        steps = int(arg) if str(arg).strip() \
            else (shell.case["steps"] or 6)
    except ValueError:
        return {"ok": False, "error": "usage: map [STEPS]"}
    try:
        pr = ForwardProjection(_scene_prog(shell), steps,
                               max_states=STATE_BUDGET,
                               keep_history=False)
    except StateBudgetExceeded as e:
        from metis.kernel.diagnostics import budget_note
        retry = f"map {max(1, e.step - 1)}"
        _note(shell, budget_note(e, "map", f"map {steps}", retry))
        return {"ok": False, "error": f"{e} (try: {retry})"}
    m = map_derivation(pr)

    def lbl(e):
        if e is None:
            return "(quiesce)"
        _, s, ev = pr.events[e]
        return f"{s}/{getattr(ev, 'name', '?')}"
    return {"ok": True, "prob": m.prob,
            "events": [lbl(e) for e in m.events],
            "final": {a: n for a, n in sorted(m.final[1])},
            "final_stage": m.final[0]}


def _op_marginals(shell, arg):
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded)
    try:
        pr = ForwardProjection(_scene_prog(shell),
                               shell.case["steps"] or 6,
                               max_states=STATE_BUDGET,
                               keep_history=False)
    except StateBudgetExceeded as e:
        from metis.kernel.diagnostics import budget_note
        n = shell.case["steps"] or 6
        _note(shell, budget_note(e, "marginals", f"marginals ({n} steps)",
                                 f"#steps {max(1, e.step - 1)}"))
        return {"ok": False, "error": str(e)}
    marg: Dict[str, float] = {}
    for (_, kappa), p in pr.final.items():
        for a, n in kappa:
            if n > 0:
                marg[a] = marg.get(a, 0.0) + p
    want = arg.split() if arg.strip() else None
    return {"ok": True, "support": len(pr.final),
            "marginals": {a: round(p, 12)
                          for a, p in sorted(marg.items())
                          if want is None or a in want}}


def _op_init(shell, arg):
    from metis.loop.verbs import LoopVerbs  # noqa: F401 (doc anchor)
    init = {}
    part, depth = "", 0
    for ch in arg + ",":
        if ch == "," and depth == 0:
            if part.strip():
                k, _, v = part.partition("=")
                init[k.strip()] = int(v or 1)
            part = ""
            continue
        depth += ch == "("
        depth -= ch == ")"
        part += ch
    shell.case["init"] = init
    return {"ok": True, "init": init}


def _op_steps(shell, arg):
    shell.case["steps"] = int(arg)
    return {"ok": True, "steps": shell.case["steps"]}


def _op_seed(shell, arg):
    shell.do_seed(arg)
    return {"ok": True, "seed": int(arg)}


def _op_llm(shell, arg):
    from .llm import DEFAULT_SPEC, resolve
    a = arg.strip()
    if not a:
        return {"ok": True,
                "llm": shell.llm or f"(default: {DEFAULT_SPEC})"}
    resolve(a)
    shell.llm = a
    return {"ok": True, "llm": a}


def _op_narrate(shell, arg):
    from metis.kernel.refsample import canonical_events, run_trace
    from . import prompts
    from .narrate import narrate, user_prompt
    override = getattr(shell, "sys_prompts", {}).get("narrate")
    parts = str(arg).split()
    show = parts.pop(0) if parts and parts[0] in ("system",
                                                  "user") else None
    if show == "system":           # the doctrine half, no model call
        text, src = prompts.resolve("narrate", override)
        return {"ok": True, "role": "system", "source": src,
                "prompt": text}
    prog = _scene_prog(shell)
    seed = int(parts[0]) if parts else 1
    idx, (stage, kappa) = run_trace(prog, shell.case["steps"] or 6,
                                    seed)
    events = canonical_events(prog)
    rows = [events[i] for i in idx]
    if show == "user":             # the certified material, ditto
        return {"ok": True, "role": "user", "seed": seed,
                "events": len(rows),
                "prompt": user_prompt(rows, stage, kappa,
                                      shell.text, seed=seed)}
    text = narrate(shell.llm or "ollama", rows, stage, kappa,
                   shell.text, seed=seed, system=override)
    return {"ok": True, "seed": seed, "events": len(rows),
            "llm": shell.llm or "ollama", "narration": text}


def _op_graph(shell, arg):
    """Spec 09 P2: the causality graph of the session as JSON (the
    console draws it). No file writes on the wire — paths are the
    REPL's."""
    from metis.lang.causality import as_obj
    g, _ = shell.causal()
    return {"ok": True, **as_obj(g, shell.stage, shell.key)}


def _op_help(shell, arg):
    return {"ok": True, "ops": sorted(INTENT_VERBS)}


_WIRE_OPS = {"trace": _op_trace, "map": _op_map,
             "marginals": _op_marginals, "init": _op_init,
             "steps": _op_steps, "seed": _op_seed, "llm": _op_llm,
             "narrate": _op_narrate, "help": _op_help,
             "findings": _op_findings, "graph": _op_graph}


def dispatch(shell, op: str, arg: str = "") -> dict:
    """One wire op against the shell: a loop verb (cmd_*) or a wire
    readout."""
    fn = getattr(shell, "cmd_" + op, None) if op not in _WIRE_OPS \
        else None
    try:
        if op in _WIRE_OPS:
            return _WIRE_OPS[op](shell, arg)
        if fn is not None:
            return fn(arg)
        return {"ok": False, "error": f"unknown op '{op}'",
                "ops": sorted(set(_WIRE_OPS)
                              | {n[4:] for n in dir(shell)
                                 if n.startswith("cmd_")})}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def run_intent(shell, line: str) -> dict:
    """Execute one page/chat-queued intent under the module policy."""
    p = parse_intent(line)
    if p is None:
        return {"ok": False, "cmd": line,
                "error": "intent refused — allowed verbs: " +
                         " ".join(sorted(INTENT_VERBS)) +
                         " (no command: specs)"}
    return dispatch(shell, *p)


def serve_json(shell, stdin=None, stdout=None) -> int:
    """The JSON-lines wire: {op, arg} per line in, one JSON result
    per line out (`metis --json FILE`)."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            print(json.dumps({"ok": False, "error": str(e)}),
                  file=stdout, flush=True)
            continue
        print(json.dumps(dispatch(shell, req.get("op", ""),
                                  req.get("arg", ""))),
              file=stdout, flush=True)
    return 0
