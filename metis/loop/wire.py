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
     "formalize", "llm", "narrate", "graph", "why", "causes",
     "ctx", "findings", "run", "step", "events", "reset", "undo",
     "state", "log", "list", "help",
     # spec 10 P5: interventions — session-local, no paths, no
     # approval; masks/the shelf are session state
     "do", "fork", "back", "whatif", "necessity", "sufficiency"])


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
    notes = shell.__dict__.setdefault("notes", [])
    if d not in notes:          # a repeated instrument adds no row
        notes.append(d)


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
                                        for d in ds],
            "text": _text(shell, "findings", arg)}


def _text(shell, cmd, arg=""):
    """The REPL's own rendering of a read-only command — the text half
    of a result posting (CONSOLE §4); None for a shell without it."""
    cap = getattr(shell, "capture", None)
    return cap(cmd, str(arg)) if cap else None


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
    out = {"ok": True, "seed": seed,
           "events": [f"{events[i][1]}/{events[i][2].name}"
                      for i in idx],
           "final_stage": stage,
           "final": {a: n for a, n in sorted(kappa)}}
    if not arg.strip():
        # bare `trace` = the REPL's: the file's '#'-case replayed (all
        # seeds, queries, exact= where tractable), its runtime notes
        # recorded for `findings` — the console shows what a terminal
        # user sees; `trace SEED` stays the one seeded readout above
        text = _text(shell, "trace")
        if text is not None:
            out["text"] = text
    return out


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
    k0 = getattr(shell, "_base", (0,))[0]
    live = list(getattr(shell, "fired_rows", [])[k0:])
    if not parts and live:
        # bare `narrate` tells THIS session's run since the graph base
        # (the REPL's own reading — the console narrates what its
        # stream shows); `narrate SEED` a fresh seeded trace
        source, seed, rows = "session", 0, live
        # spec 10: interventions travel with the material (never told
        # as chance)
        hows = [h["how"] for h in getattr(shell, "fired_how", [])[k0:]] \
            or None
        bdo = tuple(getattr(shell, "_base_do", ()))
        stage = shell.stage
        kappa = [(q, n) for q, n in sorted(shell.counts.items()) if n > 0]
    else:
        prog = _scene_prog(shell)
        seed = int(parts[0]) if parts else 1
        idx, (stage, kappa) = run_trace(prog, shell.case["steps"] or 6,
                                        seed)
        events = canonical_events(prog)
        rows = [events[i] for i in idx]
        source = "seed"
        hows, bdo = None, ()
    if show == "user":             # the certified material, ditto
        return {"ok": True, "role": "user", "seed": seed,
                "events": len(rows),
                "prompt": user_prompt(rows, stage, kappa,
                                      shell.text, seed=seed,
                                      marks=hows, base_do=bdo)}
    text = narrate(shell.llm or "ollama", rows, stage, kappa,
                   shell.text, seed=seed, system=override,
                   marks=hows, base_do=bdo)
    return {"ok": True, "source": source, "seed": seed,
            "events": len(rows), "llm": shell.llm or "ollama",
            "narration": text}


def _op_graph(shell, arg):
    """Spec 09 P2: the causality graph of the session as JSON (the
    console draws it). No file writes on the wire — paths are the
    REPL's."""
    from metis.lang.causality import as_obj, dot, header
    g, n = shell.causal()
    k = shell._base[0]
    title = header(shell.path.name, shell.key, shell.stage, n) + (
        f" (since #{k}: base #0 = the state then)" if k else "")
    mk = shell.marks() if hasattr(shell, "marks") else {}
    bd = getattr(shell, "_base_do", [])
    return {"ok": True, **as_obj(g, shell.stage, shell.key, mk, bd),
            "dot": dot(g, title, mk, bd), "text": _text(shell, "graph")}


def _op_why(shell, arg):
    """Spec 09 P3: the cone as JSON — target, cone, pruned."""
    res = shell._why(str(arg))
    if isinstance(res, str):
        return {"ok": False, "error": res}
    title, members, g = res
    keep = set(members)
    return {"ok": True, "target": title, "cone": members,
            "pruned": [nd.i for nd in g.nodes if nd.i not in keep],
            "text": _text(shell, "why", arg)}


def _op_causes(shell, arg):
    """Spec 09 P3: causes A B as JSON rows + verdict."""
    from metis.lang.causality import causes
    parts = str(arg).split()
    if len(parts) != 2:
        return {"ok": False, "error": "usage: causes A B"}
    g, _ = shell.causal()
    return {"ok": True, **causes(g, shell._occ_matcher(parts[0]),
                                 shell._occ_matcher(parts[1])),
            "text": _text(shell, "causes", arg)}


def _op_ctx(shell, arg):
    """CONSOLE §4: the aggregate context as data (the band and the
    sections render from it) + its text block; `ctx tokens|rules|
    trace` narrows the text to one section, the data stays whole."""
    a = str(arg).strip()
    if a and a not in shell._CTX_SECTIONS:
        return {"ok": False, "error": "usage: ctx [tokens|rules|trace]"}
    return {"ok": True, **shell.ctx(),
            "text": "\n".join(shell.ctx_lines(a))}


def _op_help(shell, arg):
    return {"ok": True, "ops": sorted(INTENT_VERBS)}


_WIRE_OPS = {"trace": _op_trace, "map": _op_map,
             "marginals": _op_marginals, "init": _op_init,
             "steps": _op_steps, "seed": _op_seed, "llm": _op_llm,
             "narrate": _op_narrate, "help": _op_help,
             "findings": _op_findings, "graph": _op_graph,
             "why": _op_why, "causes": _op_causes, "ctx": _op_ctx}


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


# -- session execution (CONSOLE §2a chips): the REPL command runs ONCE;
#    its printed text is kept, the structured half is read off the
#    session after. No paths, no approval — the same verbs a terminal
#    user types.

def _fired_since(shell, k):
    """Occurrences k+1.. as the console draws them: beat, episode and
    the linear effect, from the causality graph (stable: an event's
    beat depends only on earlier events)."""
    if not hasattr(shell, "causal"):
        return []
    g, _ = shell.causal()
    by = {nd.i: nd for nd in g.nodes}
    out = []
    for j, (kind, stage, ev) in enumerate(shell.fired_rows[k:], k + 1):
        nd = by.get(j)
        out.append({"i": j, "name": ev.name, "kind": kind,
                    "stage": stage if kind == "clause" else ev.pre,
                    "post": ev.post if kind == "link" else None,
                    "beat": nd.beat if nd else None,
                    "episode": nd.episode if nd else None,
                    "consume": dict(ev.consume),
                    "produce": dict(ev.produce),
                    "persist": dict(ev.persist)})
    return out


def _exec(cmd):
    def op(shell, arg):
        k = len(shell.fired)
        text = shell.capture(cmd, str(arg))
        k = min(k, len(shell.fired))        # undo/reset may shrink
        return {"ok": True, "text": text,
                "fired": _fired_since(shell, k),
                "stage": shell.stage,
                "counts": dict(sorted(shell.counts.items()))}
    op.__doc__ = f"CONSOLE §2a: `{cmd}` on the session, text + data."
    return op


def _op_events(shell, arg):
    cands = shell._candidates()
    total = sum(w for _, w in cands) or 1.0
    return {"ok": True, "text": shell.capture("events"),
            "enabled": [{"k": k, "name": shell.events_tbl[i][2].name,
                         "kind": shell.events_tbl[i][0], "w": w,
                         "p": w / total}
                        for k, (i, w) in enumerate(cands, 1)]}


def _op_cf(cmd):
    """whatif / necessity / sufficiency: the comparison report (text —
    the console shows it preformatted, v1) + the session's
    intervention state, which the command provably leaves as it was."""
    def op(shell, arg):
        text = shell.capture(cmd, str(arg))
        out = {"ok": True, "text": text, "stage": shell.stage,
               "fired": len(shell.fired)}
        if hasattr(shell, "_ctx_interventions"):
            out.update(shell._ctx_interventions())
        return out
    return op


def _op_read(cmd):
    def op(shell, arg):
        return {"ok": True, "text": shell.capture(cmd, str(arg)),
                "stage": shell.stage,
                "counts": dict(sorted(shell.counts.items())),
                "fired": len(shell.fired)}
    return op


_WIRE_OPS.update({"run": _exec("run"), "step": _exec("step"),
                  "reset": _exec("reset"), "undo": _exec("undo"),
                  "events": _op_events, "state": _op_read("state"),
                  "log": _op_read("log"), "list": _op_read("list"),
                  # spec 10: state/choice surgery reports what fired
                  # (do EVENT) and the state after; the counterfactual
                  # instruments are read-only reports (+ ctx's do data)
                  "do": _exec("do"), "fork": _exec("fork"),
                  "back": _exec("back"),
                  "whatif": _op_cf("whatif"),
                  "necessity": _op_cf("necessity"),
                  "sufficiency": _op_cf("sufficiency")})


def intent_posting(shell, line: str, session=None) -> dict:
    """What a loop session posts back for one queued intent (SPEC §6,
    CONSOLE §4/§5): {status: done|refused|error, result: the text
    rendering, json: the structured result} — sections render from
    `json`, never from scraped text. `session` (CONSOLE §5a) is
    echoed so every posting names its session."""
    from .verbs import fmt
    res = run_intent(shell, line)
    if not res.get("ok"):
        status = ("refused" if str(res.get("error", ""))
                  .startswith("intent refused") else "error")
    else:
        status = "done"
    text = res.pop("text", None)
    out = {"status": status, "result": text if text is not None
           else fmt(res), "json": res}
    if session is not None:
        out["session"] = session
    return out


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
