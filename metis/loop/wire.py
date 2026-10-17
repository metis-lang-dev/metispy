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

from ..lang.repl_api import REPL_API

# The orchestrator's verbs, read from the REPL API (repl-tools step 4: one
# source, declared there and never typed here): its reads (served_by repl)
# and its surprise verbs (served_by watch). Each is a wire op below; a
# declared verb the wire cannot dispatch is a test failure.
DECLARED = {t["name"]: t for t in REPL_API
            if t["served_by"] in ("repl", "watch")}

# The loop's own design verbs (the orchestrator does not call them).
LOOP_VERBS = frozenset(
    ["records", "record", "replay", "mine", "text", "init", "steps",
     "seed", "gate", "revise", "trace", "map", "marginals", "show",
     "formalize", "llm", "narrate", "graph", "causes",
     "run", "step", "reset", "undo", "log", "list", "help",
     # spec 10 P5: interventions — session-local, no paths, no
     # approval; masks/the shelf are session state
     "do", "fork", "back", "whatif", "necessity", "sufficiency"])

INTENT_VERBS = LOOP_VERBS | frozenset(DECLARED)


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


def _scene_policy(shell, prog, canonical=False):
    """The session's weight ports over a scene program (None without
    ports): the scene readouts price like the instruments do.
    canonical=True for run_trace (it indexes canonical_events)."""
    pricer = getattr(shell, "pricer", None)
    return (pricer.policy(prog, canonical=canonical)
            if pricer is not None else None)


def _op_trace(shell, arg):
    """trace [SEED]: one sampled run of the scene (#init, #steps) as
    JSON — the events fired, the final stage and multiset; bare, also
    the REPL's `trace` text (the file's '#'-case replayed from #init,
    = metisc --run). The session is untouched."""
    from metis.kernel.refsample import canonical_events, run_trace
    prog = _scene_prog(shell)
    seed = int(arg) if arg.strip() else 1
    idx, (stage, kappa) = run_trace(prog, shell.case["steps"] or 6,
                                    seed, policy=_scene_policy(shell, prog,
                                                        canonical=True))
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
    """map [STEPS]: the modal derivation (Viterbi) of the scene from
    #init — its probability, events and final state (exact; a blown
    state budget is refused with a shorter horizon to try)."""
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded, map_derivation)
    try:
        steps = int(arg) if str(arg).strip() \
            else (shell.case["steps"] or 6)
    except ValueError:
        return {"ok": False, "error": "usage: map [STEPS]"}
    try:
        prog = _scene_prog(shell)
        pr = ForwardProjection(prog, steps,
                               max_states=STATE_BUDGET,
                               keep_history=False,
                               policy=_scene_policy(shell, prog))
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
    """marginals [ATOM...]: exact presence marginals at the scene's
    horizon (#steps from #init): P(atom present) per atom, or for the
    atoms named."""
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded)
    try:
        prog = _scene_prog(shell)
        pr = ForwardProjection(prog,
                               shell.case["steps"] or 6,
                               max_states=STATE_BUDGET,
                               keep_history=False,
                               policy=_scene_policy(shell, prog))
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
    """init ATOM=N, ATOM(ARGS)=N, ...: set the scene's #init (the
    multiset the scene readouts — map, marginals, trace, narrate SEED —
    start from); N defaults to 1. The live session is not touched."""
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
    """steps N: set the scene's horizon (#steps) for map, marginals,
    trace and narrate SEED."""
    shell.case["steps"] = int(arg)
    return {"ok": True, "steps": shell.case["steps"]}


def _op_seed(shell, arg):
    """seed N: reseed the session's sampler (step, run)."""
    shell.do_seed(arg)
    return {"ok": True, "seed": int(arg)}


def _op_llm(shell, arg):
    """llm [SPEC]: show, or set and validate, the LLM endpoint formalize
    and narrate use: ollama[:MODEL] | anthropic[:MODEL] | command:ARGV."""
    from .llm import DEFAULT_SPEC, resolve
    a = arg.strip()
    if not a:
        return {"ok": True,
                "llm": shell.llm or f"(default: {DEFAULT_SPEC})"}
    resolve(a)
    shell.llm = a
    return {"ok": True, "llm": a}


def _op_narrate(shell, arg):
    """narrate [SEED] | narrate system | narrate user: this session's
    fired events since the last reset as prose through the llm endpoint
    (Foata beats, the final state, the catalog's %% docs); with SEED a
    fresh trace sampled with that seed; `system` / `user` show the
    prompt halves without a model call."""
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
                                        seed,
                                        policy=_scene_policy(
                                            shell, prog, canonical=True))
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
    """graph: the session's causality graph (spec 09 P2) as JSON — its
    events in beats with their covering parents, the frontier marked
    (the console draws it). No file writes on the wire — paths are
    the REPL's."""
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
    """causes A B: for each occurrence of B, is an occurrence of A among
    its causes (spec 09 P3) — JSON rows and the verdict yes / some /
    no."""
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
    trace` narrows the text to one section, the data stays whole.
    `ctx --mirror` (G-E): the tree mirror instead — one line per
    domain: stage, status, surprise state, #query marginals."""
    a = str(arg).strip()
    if a == "--mirror":
        from .watch import mirror, mirror_line
        rows = mirror(shell, shell.__dict__.get("_watch"))
        lines = [mirror_line(r) for r in rows]
        return {"ok": True, "mirror": rows, "lines": lines,
                "text": "\n".join(lines)}
    if a and a not in shell._CTX_SECTIONS:
        return {"ok": False,
                "error": "usage: ctx [tokens|rules|trace|--mirror]"}
    return {"ok": True, **shell.ctx(),
            "text": "\n".join(shell.ctx_lines(a))}


def _op_watch(shell, arg):
    """G-D: `watch [beats=N] [every=SECONDS]` — the runtime drives the
    served world under its host and the wire pushes event/quiescence/
    surprise lines, unpolled, in order (serve_json interleaves them
    with the answers to other ops). `watch off` stops it, `watch
    status` reads it. One watch per session; a new `watch` replaces a
    finished one."""
    from .watch import Watch
    a = str(arg).strip()
    w = shell.__dict__.get("_watch")
    if a in ("off", "status"):
        if w is None:
            return {"ok": True, "watch": "none"}
        if a == "off":
            w.done = True
        return {"ok": True, "watch": "off" if w.done else "on",
                **w.state()}
    opts = {}
    for tok in a.split():
        k, eq, v = tok.partition("=")
        if not eq or k not in ("beats", "every", "arm", "operator",
                                "scope", "explain_repeat_notice"):
            return {"ok": False, "error": "usage: watch [beats=N] "
                    "[every=SECONDS] [arm=checked|blind] [operator=BEATS]"
                    " [scope=order|session] [explain_repeat_notice=N]"
                    " | watch off | watch status"}
        if k == "scope":
            if v not in ("order", "session"):
                return {"ok": False, "error": f"watch: scope={v} — order "
                        "(default) or session"}
            opts[k] = v
            continue
        if k == "arm":
            if v not in ("checked", "blind"):
                return {"ok": False, "error": f"watch: arm={v} — "
                        "checked (B/C) or blind (A)"}
            opts[k] = v
            continue
        try:
            opts[k] = float(v) if k == "every" else int(v)
        except ValueError:
            return {"ok": False, "error": f"watch: bad {k}={v}"}
    if w is not None and not w.done:
        return {"ok": False, "error": "watch: already on (watch off "
                "first)", **w.state()}
    w = Watch(shell, beats=opts.get("beats"),
              blind=opts.get("arm") == "blind",
              operator=opts.get("operator"),
              scope=opts.get("scope", "order"),
              explain_repeat_notice=opts.get("explain_repeat_notice", 2))
    w.every = opts.get("every", 0.5)
    sched = getattr(shell.plugin, "schedule", None)
    return {"ok": True, "watch": "on", "path": w.path,
            "host": shell.plugin_spec or None, "every": w.every,
            "beats": w.beats, "arm": "blind" if w.blind else "checked",
            "operator": w.operator, "scope": w.scope,
            "schedule": sched() if callable(sched) else None,
            "stage": shell.stage,
            "counts": dict(sorted(shell.counts.items()))}


def _op_intent(shell, arg):
    """runtime-wire step 3: one intent down — `explain RECORD CARD MOVE`,
    `reset RECORD`, `escalate RECORD TEXT` — consumed exactly once: the
    answer is its id, the outcome returns as a `result` signal."""
    w = shell.__dict__.get("_watch")
    if w is None:
        return {"ok": False, "error": "intent: no watch — intents act on "
                "the records a watch opened"}
    return w.intent(arg)


def _op_cards(shell, arg):
    """The catalog cards offerable at the served node (provenance text,
    extends/sibling, the moves each adds, binding slots)."""
    from .cards import card_line, cards
    cs = cards(shell)
    return {"ok": True, "cards": cs,
            "text": "\n".join(card_line(c) for c in cs)}


def _op_script(shell, arg):
    """The script of the served node (metis-agent RAG §11.1), handed
    over once at attach: per card, per clause its %% comment, premises
    and effects (in the atoms' %% words) and cross-domain target; per
    atom its %% doc. The surprise and result shapes are not script."""
    from .cards import script, script_line
    ss = script(shell)
    return {"ok": True, "script": ss,
            "text": "\n".join(script_line(s) for s in ss)}


def _op_ledger(shell, arg):
    """The runtime ledger of the watched run: records (gap, resolution,
    refusals, mined_text) and every intent result, in order."""
    from .watch import ledger
    return {"ok": True, **ledger(shell, shell.__dict__.get("_watch"))}


def describe(shell, op: str) -> str:
    """The words for one wire verb: the REPL API's declaration, else the
    docstring of what serves it — the wire's own op, the loop verb
    (metis.loop.verbs cmd_*), the REPL command (metis.lang.repl do_*),
    the first that has one. Its first paragraph, on one line."""
    import inspect
    if op in DECLARED:
        return DECLARED[op]["description"]
    w = _WIRE_OPS.get(op)
    cmd = getattr(w, "runs", op)
    for fn in (w, getattr(shell, "cmd_" + op, None),
               getattr(shell, "do_" + cmd, None)):
        doc = inspect.cleandoc(getattr(fn, "__doc__", None) or "")
        if doc:
            return " ".join(doc.split("\n\n")[0].split())
    return ""


def _op_help(shell, arg):
    """help [ops | VERB...]: the wire's verbs, each with its description
    — for the declared ones the line the orchestrator's model reads, for
    the rest the docstring of what serves it (a human sees the same
    words). `help ops` is the verb list alone (a ping that fits a bus
    envelope); `help VERB...` those verbs' words."""
    ops = sorted(INTENT_VERBS)
    want = str(arg).split()
    if want == ["ops"]:
        return {"ok": True, "ops": ops}
    unknown = [w for w in want if w not in INTENT_VERBS]
    if unknown:
        return {"ok": False, "error": f"no verb {' '.join(unknown)} — "
                                      f"`help ops` lists them"}
    return {"ok": True, "ops": ops,
            "descriptions": {n: describe(shell, n) for n in want or ops}}


_WIRE_OPS = {"trace": _op_trace, "map": _op_map,
             "marginals": _op_marginals, "init": _op_init,
             "steps": _op_steps, "seed": _op_seed, "llm": _op_llm,
             "narrate": _op_narrate, "help": _op_help,
             "findings": _op_findings, "graph": _op_graph,
             "why": _op_why, "causes": _op_causes, "ctx": _op_ctx,
             "watch": _op_watch, "intent": _op_intent,
             "cards": _op_cards, "script": _op_script,
             "ledger": _op_ledger}


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
                    "stage": stage if kind in ("clause", "exo") else ev.pre,   # exo: a host beat (no link)
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
    op.runs = cmd       # its words are the REPL command's (describe)
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


def _op_surprise(verb):
    """A declared surprise verb as a wire op: `explain RECORD CARD MOVE`
    is `intent explain RECORD CARD MOVE`, down to the running watch."""
    def op(shell, arg):
        return _op_intent(shell, f"{verb} {arg}".strip())
    op.__doc__ = f"the surprise verb `{verb}`, on the running watch."
    return op


_reset_session = _exec("reset")


def _op_reset(shell, arg):
    """`reset RECORD` while a watch runs: the declared surprise verb (the
    reading was noise). Bare: the loop's own reset, the session back to
    the file's #init (the console's chip)."""
    if str(arg).strip() and shell.__dict__.get("_watch") is not None:
        return _op_intent(shell, "reset " + str(arg).strip())
    return _reset_session(shell, arg)


_WIRE_OPS.update({"run": _exec("run"), "step": _exec("step"),
                  "reset": _op_reset, "undo": _exec("undo"),
                  "explain": _op_surprise("explain"),
                  "escalate": _op_surprise("escalate"),
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
    per line out (`metis --json FILE`). While a `watch` runs, its
    signal lines are written between the answers, one driver step
    every `every` seconds (0 = as fast as the world moves); at end of
    input a running watch is driven to its end before the wire
    closes."""
    import queue
    import threading
    import time
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    lines: "queue.Queue" = queue.Queue()
    eof = object()

    def reader():
        for line in stdin:
            lines.put(line)
        lines.put(eof)

    threading.Thread(target=reader, daemon=True).start()

    def emit(obj):
        print(json.dumps(obj), file=stdout, flush=True)

    def flush():                    # intent results, in seq order
        w = shell.__dict__.get("_watch")
        while w is not None and w.outbox:
            emit(w.outbox.pop(0))

    closed, due = False, 0.0
    while True:
        w = shell.__dict__.get("_watch")
        active = w is not None and not w.done
        if closed and not active:
            return 0
        item = None
        if not closed:
            wait = None if not active else max(0.0, due - time.monotonic())
            try:
                item = lines.get(timeout=wait) if wait != 0.0 \
                    else lines.get_nowait()
            except queue.Empty:
                item = None
        if item is eof:
            closed = True
            continue
        if item is not None:
            line = item.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except ValueError as e:
                emit({"ok": False, "error": str(e)})
                continue
            emit(dispatch(shell, req.get("op", ""), req.get("arg", "")))
            flush()
            continue
        if active:
            if closed and w.every:
                time.sleep(max(0.0, due - time.monotonic()))
            try:
                sigs = w.step()
            except Exception as e:       # the wire outlives its driver
                w.done = True
                sigs = [{"signal": "error", "path": w.path,
                         "error": f"{type(e).__name__}: {e}"}]
            for sig in sigs:
                emit(sig)
            flush()
            due = time.monotonic() + w.every
