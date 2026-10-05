"""metis — the interactive shell over a .llp catalog.

The Ceptre lineage's `#interactive`, upgraded: a stateful session
(current stage + multiset) you can step, sample, assert into and
query exactly, over a catalog you keep editing in your editor
(`reload` re-compiles; a broken edit keeps the previous catalog
serving — the mis-deployed-pack doctrine). Stepping uses the
REFERENCE sampler's normative selection (canonical event order,
splitmix64, left-to-right sum), so an unperturbed seeded session
replays `metisc --run` byte-for-byte, and `save` writes the session
state back as '#'-directives (00-language-spec §8) — a discovered
state is immediately re-runnable by either toolchain.

Discovery only: certified expectations stay in manifests."""

import argparse
import atexit
import io
import os
import re
import sys
from cmd import Cmd
from dataclasses import replace
from pathlib import Path

from metis.kernel import (STATE_BUDGET, ForwardProjection,
                          StateBudgetExceeded, program_key)
from metis.kernel.refsample import (DONE, SplitMix64, canonical_events,
                                    choice_weight, run_trace)
from metis.lang import compile_catalog, parse, resolve_includes
from metis.loop.verbs import LoopState, LoopVerbs, fmt as _loop_fmt
from metis.loop.wire import _note
from metis.lang.directives import (DirectiveError, embedded_case,
                                   render_run, state_table)

_HELP = """\
events              enabled transitions of the firing tier, numbered
step [k|name]       fire candidate k (or by name); no arg = sampled
run N               sample N steps (stops at quiescence)
state               current stage + multiset
list                file, frame (stage, steps, state) and the
                    current stage's rules with source lines:
                    * enabled now · ◂ the present rests on it (#k,
                    why's cone) · · fired, no longer matters
                    (presence-only, like why) · ? waiting on atoms
                    some rule here makes (* = enabled) · – spent:
                    an atom nothing here makes; then other stages'
                    rules the present rests on
assert ATOM [N]     add N (default 1) tokens          [alias: +]
retract ATOM [N]    remove N (default 1) tokens       [alias: -]
query ATOM [STEPS]  exact presence marginal from HERE (filter)
seed N              reseed the sampler
undo                back one mutating command
reset               back to the file's #init directives
reload              re-read + recompile the file; state carries over
edit [RULE]         open the file (VS Code terminal: `code`; else
                    $VISUAL/$EDITOR, reloading on exit; else
                    xdg-open), at RULE's line if given
trace               replay the file's embedded '#'-case, fresh from
                    #init (= metisc --run); session state untouched
save [PATH]         session state as '#'-directives (stdout or file)
log                 fired events this session
ctx [tokens|rules|trace]
                    the context at this stop (Pwndbg's context; the
                    console's band): catalog, key, stage, beat,
                    episode, status running|quiescent|done, tokens,
                    enabled — then the sections (all, or one)

causality (spec 09) — the run as a partial order, since the last
reset/assert/retract/#init (node #0 is the state then):
graph [F.dot|F.json] beats: each event with its covering parents
                    (i<j, flow atoms in parens), * = frontier
                    ("where we are"); or write DOT (`dot -Tsvg`)
                    / JSON
why [ATOM|K]        causal cone with rule lines + what is pruned:
                    bare = what the current state rests on; ATOM =
                    its last producer (else the base); K = event K
                    (dependence on the multiset — atoms are
                    indistinguishable, no token lineage)
causes A B          per occurrence of B: is some A among its causes;
                    yes / some / no (TeLLer's x)
map [STEPS]         modal derivation (Viterbi) of the file scene
                    (exact; a blown state budget says so — lower
                    the horizon rather than wait)

the loop (text -> candidate -> gates -> world; one kernel REPL,
every UI a wrapper):
sink FILE.jsonl     load surprise records     records    list them
record [N]          select one (scene + text adopted from it)
replay              recover() it against the world (explain/verify)
mine [TEXT]         the gap to mine / set + stamp the approved unit
text <<EOF          set the approved doc text (heredoc)
llp <<EOF           set the candidate LLP source (heredoc)
load FILE.llp       candidate from file
formalize [SPEC]    draft via the llm endpoint (repair loop)
gate                gate candidate as a PACK against the world
revise              gate as a BASE EDIT (accept replaces the world)
accept              admitted candidate joins the served world LIVE
show [what]         loop state     export [DIR]   world as files
llm [SPEC]          show/set + VALIDATE the LLM endpoint
                    (ollama[:MODEL] | anthropic[:MODEL] |
                    command:ARGV; seeded by the config dot file:
                    $METIS_CONFIG | .metis.toml |
                    ~/.config/metis/config.toml)
narrate [SEED]      the session so far (or a seeded trace) as prose:
                    Foata beats + %% docs through the llm endpoint
narrate system|user show the system / certified-material prompt
                    (no model call; the UI halves). `narrate system
                    <<EOF` sets a session override; the persistent
                    home is ~/.config/metis/prompts/narrate.txt
                    (likewise `formalize system [<<EOF]` /
                    prompts/formalize.txt — narrators differ)
key                 canonical program_key of the serving catalog
version             what is running: package, source+git, python,
                    config dot file, llm endpoint
quit                leave (also: exit, ctrl-d)

Commands are bare words — no prefix; 'help' or '?' lists them.
'#'-directive lines (sent from the buffer with shift+enter) apply to
the live session: #init ATOM N assigns the count (assert ADDS),
#query runs, #steps/#seeds/#unknown/#lik update the case; '%'
comments pass silently. Rule text is not REPL input: edit the file,
save, 'reload'."""


def _fmt_counts(counts):
    return " ".join(f"{a}:{n}" for a, n in sorted(counts.items())) \
        or "(empty)"


_RULE_HEAD = re.compile(r"^\s*(?:qui\s+)?([\w-]+)\s*\[")
_NOT_RULES = frozenset(["catalog", "provenance", "type", "pred",
                        "stage", "stages", "layers", "namespace"])


def _source_name(name, spans):
    """An event's rule as written in the source: alternative-head
    branches (`blow_stick-1`, `-2`) belong to `blow_stick`."""
    if name in spans:
        return name
    base = re.sub(r"-\d+$", "", name)
    return base if base in spans else name


class Repl(LoopVerbs, Cmd):
    prompt = "llp> "

    def __init__(self, path, seed=1, **kw):
        super().__init__(**kw)
        self.path = Path(path)
        self.rng = SplitMix64(seed)
        self.undo_stack = []
        self.fired = []
        self.loop = LoopState()
        self.fired_rows = []   # (kind, stage, ev) per fired event
        # the causality graph's base (spec 09): (log length, stage,
        # counts) at the last edit no event explains — start, reset,
        # assert/retract, #init; the graph covers fired_rows after it
        self._base = (0, "", {})
        # runtime/loop diagnostics the instruments spoke this session
        # (spec 08 step 3) — listed by `findings`
        self.notes = []
        from metis.loop import config
        self.llm = config.get("llm", "endpoint", "")  # spec (loop.llm)
        self.sys_prompts = {}  # per-seat session overrides (prompts)
        self._load(initial=True)
        flat = _fmt_counts(self.counts)
        if len(flat) > 60:              # tragedy wraps 5 lines flat:
            toks = sum(self.counts.values())
            flat = (f"{len(self.counts)} atoms, {toks} tokens "
                    f"— 'state' tabulates them")
        warn = "".join(f"warning: {w}\n" for w in self.doc.warnings)
        self.intro = warn + (
            f"metis shell — {self.path.name} "
            f"(catalog {self.doc.catalog.name}, key {self.key[:12]}…)\n"
            f"stage {self.stage} | {flat}\n"
            "type 'help' for commands, 'events' to see what can fire")

    # -- catalog / session plumbing ------------------------------------

    def _compile(self):
        text = self.path.read_text()
        ast = parse(text, file=str(self.path))
        doc = compile_catalog(resolve_includes(ast, self.path.parent))
        return text, doc, embedded_case(text)

    def _load(self, initial=False):
        text, doc, case = self._compile()
        # a new world: the old world's runtime notes no longer speak
        self.__dict__["notes"] = []
        self.text, self.doc, self.case = text, doc, case
        self.prog = doc.catalog.ground(dict(case["init"]))
        # seed the read-only scene cache (wire._scene_prog) with the
        # program just grounded: the first map/trace costs no ground
        self._scene_cache = (doc.catalog,
                             tuple(sorted(dict(case["init"]).items())),
                             self.prog)
        self.events_tbl = canonical_events(self.prog)
        self.key = program_key(self.prog)
        if initial or self.stage not in (
                list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(case["init"])
            self._rebase()
        self.atoms = set(self.case["init"])
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    def _reserve(self, catalog):
        """Re-serve a merged/revised catalog LIVE (loop accept):
        new ground, events, key, atoms; counts and stage carry over
        exactly as `reload` does."""
        self.doc = replace(self.doc, catalog=catalog)
        self.__dict__["notes"] = []     # a new world: old notes retire
        self.prog = catalog.ground(dict(self.case["init"]))
        self.events_tbl = canonical_events(self.prog)
        self.key = program_key(self.prog)
        if self.stage not in (list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(self.case["init"])
            self._rebase()
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    def _mark(self):
        self.undo_stack.append(
            (self.stage, dict(self.counts), len(self.fired), self._base))

    def _rebase(self):
        """The current state becomes the causality graph's base: an
        edit no fired event explains (spec 09 — the graph is a pure
        function of base + the rows fired since)."""
        self._base = (len(self.fired), self.stage, dict(self.counts))

    def causal(self):
        """(graph, events since base) — spec 09's object over the
        session, derived on demand: nothing persists, so undo/reset/
        reload stay correct."""
        from metis.kernel.nets import causal_graph
        k, stage, counts = self._base
        rows = self.fired_rows[k:]
        return causal_graph(rows, counts, stage, first=k + 1), len(rows)

    def _candidates(self):
        """The firing tier, exactly as the reference sampler sees it:
        enabled clauses of the current stage, else its links."""
        if self.stage == DONE:
            return []
        for kind in ("clause", "link"):
            ws = [(i, choice_weight(ev, self.counts))
                  for i, (k, s, ev) in enumerate(self.events_tbl)
                  if k == kind and s == self.stage]
            ws = [(i, w) for i, w in ws if w > 0]
            if ws:
                return ws
        return []

    def _fire(self, i):
        kind, _, ev = self.events_tbl[i]
        for q, n in ev.consume.items():
            self.counts[q] = self.counts.get(q, 0) - n
        for q, n in ev.produce.items():
            self.counts[q] = self.counts.get(q, 0) + n
        self.counts = {q: n for q, n in self.counts.items() if n > 0}
        if kind == "link":
            self.stage = ev.post
        self.fired.append(ev.name)
        self.fired_rows.append(self.events_tbl[i])
        return ev.name

    def _sample(self):
        """One normative selection step (06-sampler-wire §4)."""
        ws = self._candidates()
        if not ws:
            self.stage = DONE
            return None
        total = 0.0
        for _, w in ws:
            total += w
        threshold = self.rng.unit() * total
        acc = 0.0
        chosen = ws[-1][0]
        for i, w in ws:
            acc += w
            if threshold < acc:
                chosen = i
                break
        return self._fire(chosen)

    def _say(self, s):
        self.stdout.write(s + "\n")

    # -- commands ------------------------------------------------------

    def do_events(self, arg):
        """enabled transitions of the firing tier, numbered"""
        ws = self._candidates()
        if not ws:
            self._say(f"quiescent (stage {self.stage})")
            return
        kind = self.events_tbl[ws[0][0]][0]
        total = sum(w for _, w in ws)
        for k, (i, w) in enumerate(ws, 1):
            ev = self.events_tbl[i][2]
            self._say(f"  {k}. {ev.name}  w={w:g}  p={w / total:g}")
        if kind == "link":
            self._say("  (quiescence links — firing switches stage)")

    def do_step(self, arg):
        """fire candidate k or by name; no argument = sampled"""
        ws = self._candidates()
        if not ws:
            self.stage = DONE
            self._say("quiescent — nothing can fire")
            return
        self._mark()
        if not arg.strip():
            name = self._sample()
            self._say(f"fired (sampled): {name}")
            return
        a = arg.strip()
        pick = None
        if a.isdigit() and 1 <= int(a) <= len(ws):
            pick = ws[int(a) - 1][0]
        else:
            hits = [i for i, _ in ws
                    if self.events_tbl[i][2].name.startswith(a)]
            if len(hits) == 1:
                pick = hits[0]
            elif hits:
                self.undo_stack.pop()
                self._say(f"ambiguous: {len(hits)} candidates match "
                          f"{a!r} — 'events' to list them")
                return
        if pick is None:
            self.undo_stack.pop()
            self._say(f"no enabled candidate {a!r} — 'events' lists "
                      f"1..{len(ws)}")
            return
        self._say(f"fired: {self._fire(pick)}")

    def do_run(self, arg):
        """sample N steps (stops at quiescence)"""
        try:
            n = int(arg.strip() or "1")
        except ValueError:
            self._say("usage: run N")
            return
        self._mark()
        names = []
        for _ in range(n):
            name = self._sample()
            if name is None:
                break
            names.append(name)
        self._say(" ".join(names) if names else "(quiescent)")
        self._say(f"stage {self.stage} | {_fmt_counts(self.counts)}")

    def do_state(self, arg):
        """current stage + multiset, tabulated by predicate"""
        rows = state_table(self.counts.items())
        self._say("\n".join([f"stage {self.stage}"]
                            + (rows or ["  (empty)"])))

    def _adjust(self, arg, sign, usage):
        parts = arg.split()
        if not parts or len(parts) > 2:
            self._say(usage)
            return
        atom = parts[0]
        try:
            n = int(parts[1]) if len(parts) == 2 else 1
        except ValueError:
            self._say(usage)
            return
        if atom not in self.atoms:
            self._say(f"note: {atom} appears in no rule of this "
                      "catalog (typo?)")
        self._mark()
        c = self.counts.get(atom, 0) + sign * n
        if c < 0:
            self._say(f"only {self.counts.get(atom, 0)} present; "
                      "removed all")
            c = 0
        self.counts[atom] = c
        if c == 0:
            self.counts.pop(atom, None)
        self._rebase()
        self._say(f"{atom}:{c}")

    def do_assert(self, arg):
        """assert ATOM [N] — add N (default 1) tokens"""
        self._adjust(arg, +1, "usage: assert ATOM [N]")

    def do_retract(self, arg):
        """retract ATOM [N] — remove N (default 1) tokens"""
        self._adjust(arg, -1, "usage: retract ATOM [N]")

    def do_query(self, arg):
        """query ATOM [STEPS] — exact presence marginal from here"""
        parts = arg.split()
        if not parts or len(parts) > 2:
            self._say("usage: query ATOM [STEPS]")
            return
        atom = parts[0]
        steps = int(parts[1]) if len(parts) == 2 \
            else (self.case["steps"] or 1)
        if self.stage == DONE:
            p = float(self.counts.get(atom, 0) > 0)
            self._say(f"P[{atom} present | done] = {p:g}")
            return
        here = replace(self.prog, init_stage=self.stage,
                       init=dict(self.counts))
        try:
            pr = ForwardProjection(here, steps,
                                   max_states=STATE_BUDGET,
                                   keep_history=False)
        except StateBudgetExceeded as e:
            from metis.kernel.diagnostics import budget_note
            retry = f"query {atom} {max(1, e.step - 1)}"
            _note(self, budget_note(e, "query",
                                          f"query {atom} {steps}", retry))
            self._say(f"{e} (try: {retry})")
            return
        except Exception as e:
            self._say(f"filter failed: {e!r}")
            return
        p = sum(q for (_, k), q in pr.final.items()
                if any(a == atom and n > 0 for a, n in k))
        self._say(f"P[{atom} present after {steps} steps] = {p:.6g}")

    def do_seed(self, arg):
        """seed N — reseed the sampler"""
        try:
            self.rng = SplitMix64(int(arg.strip()))
            self._say(f"seed {arg.strip()}")
        except ValueError:
            self._say("usage: seed N")

    def do_undo(self, arg):
        """back one mutating command"""
        if not self.undo_stack:
            self._say("nothing to undo")
            return
        self.stage, self.counts, k, self._base = self.undo_stack.pop()
        del self.fired[k:]
        del self.fired_rows[k:]
        self.do_state("")

    def do_reset(self, arg):
        """back to the file's #init directives"""
        self._mark()
        self.stage = self.prog.init_stage
        self.counts = dict(self.case["init"])
        self._rebase()
        self.do_state("")

    def do_reload(self, arg):
        """re-read + recompile the file; session state carries over"""
        old_key = self.key
        try:
            self._load()
        except Exception as e:
            findings = getattr(e, "findings", None) or [str(e)]
            self._say("reload failed — previous catalog still "
                      "serving:\n  " + "\n  ".join(findings))
            return
        if self.key != old_key and (self.loop.accepted
                                    or self.loop.base_src):
            self.loop.accepted = []
            self.loop.base_src = ""
            self._say("note: accepted packs/revision dropped — the "
                      "re-read file is the world now")
        for w in self.doc.warnings:
            self._say(f"warning: {w}")
        what = "unchanged" if self.key == old_key else "NEW key"
        self._say(f"reloaded {self.path.name} ({what} "
                  f"{self.key[:12]}…); "
                  f"stage {self.stage} | {_fmt_counts(self.counts)}")

    def _rule_spans(self):
        """rule name -> (first, head, last) 1-based source lines, the
        %% doc block above the head included; a rule ends at the first line
        whose code (comment stripped) ends with '.'."""
        lines = self.text.splitlines()
        spans = {}
        for n, l in enumerate(lines):
            m = _RULE_HEAD.match(l)
            if not m or m.group(1) in _NOT_RULES:
                continue
            first = n
            while first > 0 and lines[first - 1].lstrip().startswith("%%"):
                first -= 1
            last = n
            while (last < len(lines) - 1
                   and not lines[last].split("%", 1)[0].rstrip()
                   .endswith(".")):
                last += 1
            spans.setdefault(m.group(1), (first + 1, n + 1, last + 1))
        return spans

    def do_list(self, arg):
        """the current file, frame (stage, steps, state) and the
        rules of the current stage with source lines, marked: * enabled
        now, ◂ the present rests on it (why's cone), · fired but no
        longer matters, ? waiting (on what); then the rules of other
        stages the present rests on"""
        self._say(f"file   {self.path}  (catalog "
                  f"{self.doc.catalog.name}, key {self.key[:12]}…)")
        last = self.fired[-1] if self.fired else "—"
        self._say(f"frame  stage {self.stage} | {len(self.fired)} "
                  f"fired, last {last}")
        for row in state_table(self.counts.items()) or ["  (empty)"]:
            self._say("     " + row)
        self._say("\n".join(self._rules_lines()))

    def _rules_lines(self, rel=None):
        """list's rules block (also `ctx rules`): the current stage's
        rules with source lines and relevance marks, then what the
        present rests on in other stages."""
        rel = rel or self._relevance()
        out = []
        spans, lines = self._rule_spans(), self.text.splitlines()
        width = len(str(len(lines)))
        if self.stage == DONE:
            out.append("rules  none can fire — the run is done ('reset' "
                       "to start over)")
        else:
            out.append(f"rules  of stage {self.stage}  (* enabled now ×N "
                       "· ◂ the present rests on #k · · fired, no longer "
                       "matters · ? waiting on ← maker · – spent: no X "
                       "and nothing here makes it)")
        for name in sorted(rel["stage"],
                           key=lambda r: spans.get(r, (1e9,))[0]):
            mark, tail = self._list_marks(name, rel)
            if name not in spans:
                out.append(f"  {mark} {'?':>{width}}  {name}{tail}")
                continue
            a, head, b = spans[name]
            for k in range(a, b + 1):
                lead = f"  {mark if k == head else ' '} {k:>{width}}  "
                out.append(f"{lead}{lines[k - 1]}"
                           f"{tail if k == head else ''}")
        if rel["elsewhere"]:
            out.append("rests on  (rules of other stages the present "
                       "depends on)")
            for stage, items in rel["elsewhere"]:
                out.append(f"  ◂ {stage}: " + " · ".join(
                    f"{n} {' '.join(f'#{i}' for i in occ)}"
                    for n, occ in items))
        if rel["pruned_elsewhere"]:
            out.append(f"  · {rel['pruned_elsewhere']} more fired "
                       "event(s) in other stages no longer matter")
        return out

    def _relevance(self):
        """list's relevance (spec 09 D3 follow-up): per rule, enabled
        now (×N), the occurrences in the current-state cone (what the
        present rests on — `why`'s cone) and the fired-but-pruned ones
        since the graph base; for the current stage's idle rules, what
        they wait on. Presence-only like `why`: pruned = nothing it
        touched is still present, not 'caused nothing'."""
        from metis.kernel.nets import cone, live_roots
        spans = self._rule_spans()

        def src(name):
            return _source_name(name.partition("[")[0], spans)
        enabled = {}
        for i, _ in self._candidates():
            n = src(self.events_tbl[i][2].name)
            enabled[n] = enabled.get(n, 0) + 1
        g, _ = self.causal()
        k0, base_stage, base_counts = self._base
        stage = None if self.stage == DONE else self.stage
        keep = set(cone(g, live_roots(self.fired_rows[k0:], base_counts,
                                      base_stage, self.counts, stage,
                                      first=k0 + 1)))
        in_cone, pruned, where = {}, {}, {}
        for nd in g.nodes:
            if nd.kind == "init":
                continue
            n = src(nd.name)
            (in_cone if nd.i in keep else pruned).setdefault(n, []) \
                .append(nd.i)
            where.setdefault(n, "links" if nd.kind == "link"
                             else nd.stage)
        names = []
        for kind, s, ev in self.events_tbl:
            n = src(ev.name)
            if s == self.stage and n not in names:
                names.append(n)
        elsewhere = {}
        for n, occ in in_cone.items():
            if n not in names:
                elsewhere.setdefault(where[n], []).append((n, occ))
        order = {s: k for k, s in enumerate(self.prog.stages)}
        return {
            "stage": names, "enabled": enabled, "cone": in_cone,
            "pruned": pruned,
            "elsewhere": sorted(elsewhere.items(),
                                key=lambda kv: order.get(kv[0], 1e9)),
            "pruned_elsewhere": sum(len(o) for n, o in pruned.items()
                                    if n not in names),
        }

    def _waiting_on(self, name):
        """What an idle rule of the current stage lacks, from its
        nearest ground instance: `? atom ← maker` when every missing
        atom has a producer in the stage (`rule*` = enabled now), else
        `– no atom` — spent: nothing in this stage makes it again."""
        spans = self._rule_spans()
        best = None
        for kind, s, ev in self.events_tbl:
            if s != self.stage or _source_name(
                    ev.name.partition("[")[0], spans) != name:
                continue
            need = dict(ev.consume)
            for a, n in ev.persist.items():
                need[a] = need.get(a, 0) + n
            miss = sorted(a for a, n in need.items()
                          if self.counts.get(a, 0) < n)
            if best is None or len(miss) < len(best[1]):
                best = (kind, miss)
        if best is None:
            return ""
        kind, miss = best
        if not miss:
            return ("? quiescence (fires when no clause can)"
                    if kind == "link" else "? nothing — weight 0 here")
        cand = [self.events_tbl[i][2] for i, _ in self._candidates()]

        def maker(a):
            for ev in cand:
                if a in ev.produce:
                    return _source_name(ev.name.partition("[")[0],
                                        spans) + "*"
            for _, s, ev in self.events_tbl:
                if s == self.stage and a in ev.produce:
                    return _source_name(ev.name.partition("[")[0], spans)
            return None
        makers = {a: maker(a) for a in miss}
        dead = [a for a in miss if makers[a] is None]
        if dead:          # spent: nothing in this stage makes it again
            more = f" (+{len(dead) - 3})" if len(dead) > 3 else ""
            return "– no " + ", ".join(dead[:3]) + more
        more = f" (+{len(miss) - 3})" if len(miss) > 3 else ""
        return "? " + "; ".join(f"{a} ← {makers[a]}"
                                for a in miss[:3]) + more

    def _list_marks(self, name, rel):
        """(mark, tail) for one rule of the current stage."""
        en = rel["enabled"].get(name, 0)
        occ, gone = rel["cone"].get(name, []), rel["pruned"].get(name, [])
        tail = []
        if en > 1:
            tail.append(f"×{en}")
        if occ:
            tail.append("◂ " + " ".join(f"#{i}" for i in occ))
        if gone:
            tail.append("· " + " ".join(f"#{i}" for i in gone))
        wait = self._waiting_on(name) \
            if not en and self.stage != DONE else ""
        if wait:
            tail.append(wait)
        mark = ("*" if en else "◂" if occ else "·" if gone
                else wait[:1] if wait else " ")
        return mark, ("   " + "  ".join(tail) if tail else "")

    _CTX_STATE = {"*": "fireable", "◂": "rests_on", "·": "fired_pruned",
                  "?": "waiting", "–": "spent", " ": "idle"}

    def ctx(self):
        """The aggregate context (CONSOLE §4, Pwndbg's `context`):
        list's relevance analysis plus the header facts, as data —
        the console's band and sections render from this, never from
        text. status: running (a clause can fire) | quiescent (only a
        stage link can: the stage has settled) | done (nothing can)."""
        rel = self._relevance()
        g, _ = self.causal()
        spans = self._rule_spans()
        cands = self._candidates()
        total = sum(w for _, w in cands) or 1.0
        kinds = {self.events_tbl[i][0] for i, _ in cands}
        status = ("done" if self.stage == DONE or not cands
                  else "quiescent" if kinds == {"link"} else "running")
        rules = []
        lines = self.text.splitlines()
        for name in sorted(rel["stage"],
                           key=lambda r: spans.get(r, (1e9,))[0]):
            mark, tail = self._list_marks(name, rel)
            a, _, b = spans.get(name, (0, 0, -1))
            rules.append({
                "src": "\n".join(lines[a - 1:b]) if a else "",
                "name": name, "state": self._CTX_STATE[mark],
                "detail": tail.strip(),
                "line": spans[name][1] if name in spans else None,
                "enabled": rel["enabled"].get(name, 0),
                "rests_on": rel["cone"].get(name, []),
                "pruned": rel["pruned"].get(name, [])})
        return {
            "catalog": self.doc.catalog.name, "key": self.key,
            "run_key": self._run_key(),
            "file": str(self.path), "stage": self.stage,
            "status": status, "fired": len(self.fired),
            "base": self._base[0],
            "beat": max((nd.beat for nd in g.nodes), default=0),
            "episode": sum(1 for nd in g.nodes if nd.kind == "link"),
            "counts": dict(sorted(self.counts.items())),
            "enabled": [{"name": self.events_tbl[i][2].name, "w": w,
                         "p": w / total} for i, w in cands],
            "rules": rules,
            "last_event": self.fired[-1] if self.fired else None,
            "rests_on_stages": [
                {"stage": st, "rules": [{"name": n, "occurrences": occ}
                                        for n, occ in items]}
                for st, items in rel["elsewhere"]],
            "pruned_elsewhere": rel["pruned_elsewhere"],
        }

    def _run_key(self):
        """run_key of the session since the graph base (spec 09: the
        Foata class, hashed) — None if the events predate a reload."""
        from metis.kernel.nets import run_key
        pos = {id(ev): k for k, (_, _, ev) in enumerate(self.prog.events())}
        try:
            return run_key(self.prog, [pos[id(ev)] for _, _, ev
                                       in self.fired_rows[self._base[0]:]])
        except KeyError:
            return None

    _CTX_SECTIONS = ("tokens", "rules", "trace")

    def ctx_lines(self, section=""):
        """The text ctx: the band (what the console's context band
        shows), then the sections — all three bare, or the one asked
        for (`ctx tokens|rules|trace`)."""
        from metis.lang.causality import beats_text
        c = self.ctx()
        dots = "  ".join(
            f"{a} {'●' * n if n <= 5 else f'●×{n}'}"
            for a, n in c["counts"].items()) or "(empty)"
        n_rules = sum(1 for r in c["rules"] if r["state"] == "fireable")
        out = [f"CATALOG {c['catalog']}  KEY {c['key'][:12]}  STAGE "
               f"{c['stage']}  BEAT {c['beat']}  EPISODE {c['episode']}"
               f"  {c['status'].upper()}  {c['fired']} fired"
               + (f" · last {c['last_event']}" if c["last_event"]
                  else ""),
               f"TOKENS  {dots}",
               f"ENABLED {len(c['enabled'])} event(s) · {n_rules} of "
               f"{len(c['rules'])} rule(s) fireable"]
        want = [section] if section else list(self._CTX_SECTIONS)
        if "tokens" in want:
            out += ["── tokens"] + (state_table(self.counts.items())
                                   or ["  (empty)"])
        if "rules" in want:
            out += ["── rules"] + self._rules_lines()
        if "trace" in want:
            g, _ = self.causal()
            out += ["── trace"] + beats_text(g)
        return out

    def do_ctx(self, arg):
        """ctx [tokens|rules|trace]: the context at this stop (Pwndbg's
        `context`) — the band (catalog, key, stage, beat, episode,
        status, tokens, enabled), then all sections or the one named"""
        a = arg.strip()
        if a and a not in self._CTX_SECTIONS:
            self._say("usage: ctx [tokens|rules|trace]")
            return
        self._say("\n".join(self.ctx_lines(a)))

    def capture(self, cmd, arg=""):
        """A read-only command's text output, as the shell would print
        it (the text half of a console result posting)."""
        out, self.stdout = self.stdout, io.StringIO()
        try:
            getattr(self, "do_" + cmd)(arg)
            return self.stdout.getvalue().rstrip("\n")
        finally:
            self.stdout = out

    def do_edit(self, arg):
        """open the file in your editor: in a VS Code terminal `code`,
        else $VISUAL/$EDITOR, else xdg-open. edit RULE jumps to the
        rule. A terminal editor blocks; a changed file reloads on
        exit."""
        import shlex
        import shutil
        import subprocess
        line = None
        if arg.strip():
            span = self._rule_spans().get(arg.strip())
            if span is None:
                self._say(f"no rule {arg.strip()!r} in {self.path.name}"
                          " — 'list' shows the current stage's rules")
                return
            line = span[1]
        f = str(self.path.resolve())
        ed = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        in_vscode = (os.environ.get("TERM_PROGRAM") == "vscode"
                     or "VSCODE_IPC_HOOK_CLI" in os.environ)
        if in_vscode and shutil.which("code"):
            # the file is already a VS Code buffer: open it there, not
            # in a second editor inside the REPL terminal
            cmd = ["code", "-g", f"{f}:{line}"] if line else ["code", f]
            blocking = False
        elif ed:
            cmd = shlex.split(ed) + ([f"+{line}", f] if line else [f])
            blocking = True
        elif shutil.which("xdg-open"):
            cmd, blocking = ["xdg-open", f], False
        else:
            self._say("no editor — set $EDITOR (or $VISUAL)")
            return
        before = self.path.stat().st_mtime_ns
        try:
            if blocking:
                subprocess.call(cmd)
            else:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
        except OSError as e:
            self._say(f"cannot start editor {cmd[0]!r}: {e.strerror}")
            return
        if blocking and self.path.stat().st_mtime_ns != before:
            self.do_reload("")
        elif not blocking:
            self._say(f"opened {self.path.name} in {cmd[0]} — save, "
                      "then 'reload' here")

    def _heredoc(self, arg):
        """`text <<EOF` convention: read lines from stdin until the
        tag; a plain arg passes through."""
        a = arg.strip()
        if not a.startswith("<<"):
            return arg
        tag = a[2:].strip() or "EOF"
        lines = []
        while True:
            line = self.stdin.readline()
            if not line or line.rstrip("\n") == tag:
                break
            lines.append(line)
        return "".join(lines)

    def _loop_say(self, res):
        self._say(_loop_fmt(res))

    def do_text(self, arg):
        """set the approved doc text (multi-line: text <<EOF)"""
        self._loop_say(self.cmd_text(self._heredoc(arg)))

    def do_llp(self, arg):
        """set the candidate LLP source (multi-line: llp <<EOF)"""
        self._loop_say(self.cmd_llp(self._heredoc(arg)))

    def do_load(self, arg):
        """load a candidate .llp file (its header comment becomes
        the text when none is set)"""
        self._loop_say(self.cmd_load(arg))

    def _sys_prompt(self, kind, rest):
        """Show or set a seat's system prompt. Bare: the active
        prompt and where it came from (session | file | builtin).
        `<<EOF`: a session override (empty heredoc clears it) — the
        persistent home is ~/.config/metis/prompts/<kind>.txt."""
        from metis.loop import prompts
        if rest.strip():
            text = self._heredoc(rest).strip()
            if text:
                self.sys_prompts[kind] = text
            else:
                self.sys_prompts.pop(kind, None)
            src = "session" if text else "cleared"
            self._say(f"{kind} system prompt: {src}"
                      + (f" ({len(text)} chars)" if text else ""))
            return
        text, src = prompts.resolve(kind, self.sys_prompts.get(kind))
        self._say(f"({src})\n{text}")

    def do_formalize(self, arg):
        """draft the candidate from the text via the llm endpoint
        (repair loop; findings feed back). `formalize system [<<EOF]`
        shows/sets the formalizer's system prompt."""
        parts = arg.split(None, 1)
        if parts and parts[0] == "system":
            return self._sys_prompt("formalize",
                                    parts[1] if len(parts) > 1 else "")
        self._loop_say(self.cmd_formalize(arg))

    def do_findings(self, arg):
        """findings [N]: the serving world's current diagnostics, one
        line each (catalog warnings, the last gate/revise report, the
        session's runtime notes); `findings N` shows one in full —
        subject, context, hint (spec 08)"""
        from metis.kernel.diagnostics import render_text
        from metis.loop.wire import session_findings
        ds = session_findings(self)
        a = arg.strip()
        if a:
            try:
                k = int(a)
            except ValueError:
                self._say("usage: findings [N]")
                return
            if not 1 <= k <= len(ds):
                self._say(f"no finding {k} (have {len(ds)})")
                return
            self._say(render_text(ds[k - 1]).rstrip("\n"))
            return
        if not ds:
            self._say("no findings")
            return
        w = len(str(len(ds)))
        self._say("\n".join(
            f"{k:>{w}}. {render_text(d).splitlines()[0]}"
            for k, d in enumerate(ds, 1)))

    def _report_say(self, out):
        """A gate/revise report: the report fields, then its diagnostics
        through the SAME text renderer `findings` uses (spec 08 D4) —
        the wire keeps the JSON objects."""
        from metis.kernel.diagnostics import render_text
        out = dict(out)
        out.pop("diagnostics", None)
        self._loop_say(out)
        rep = self.loop.report
        for d in (getattr(rep, "diagnostics", None) or []):
            self._say(render_text(d).rstrip("\n"))

    def do_gate(self, arg):
        """gate the candidate as a PACK against the served world"""
        self._report_say(self.cmd_gate(arg))

    def do_revise(self, arg):
        """gate the candidate as a BASE EDIT (accept then replaces
        the world; measure diff shown)"""
        self._report_say(self.cmd_revise(arg))

    def do_accept(self, arg):
        """admitted candidate joins the served world — re-served
        LIVE (new key/events; counts and stage carry over)"""
        self._loop_say(self.cmd_accept(arg))

    def do_sink(self, arg):
        """load surprise records from a JSONL sink"""
        self._loop_say(self.cmd_sink(arg))

    def do_records(self, arg):
        """list loaded surprise records"""
        self._loop_say(self.cmd_records(arg))

    def do_record(self, arg):
        """select a record; its init/steps become the scene, its
        mined text the approved text"""
        self._loop_say(self.cmd_record(arg))

    def do_replay(self, arg):
        """recover() the record against the served world
        (explanation before a fix, verification after)"""
        self._loop_say(self.cmd_replay(arg))

    def do_mine(self, arg):
        """bare: the observed-vs-modeled gap; with TEXT: set + stamp
        the approved unit"""
        self._loop_say(self.cmd_mine(self._heredoc(arg)))

    def do_show(self, arg):
        """loop state: show [state|text|base|llp]"""
        self._loop_say(self.cmd_show(arg))

    def do_export(self, arg):
        """accepted world as files: base.llp + packs + text.md"""
        self._loop_say(self.cmd_export(arg))

    def do_map(self, arg):
        """the modal derivation (Viterbi) + final state, from the
        file scene"""
        from metis.loop.wire import _op_map
        res = _op_map(self, arg)
        if not res.get("ok"):
            return self._loop_say(res)
        evs = res["events"]
        w = len(str(len(evs)))
        lines = [f"map p={res['prob']:g}"]
        lines += [f"  {k:>{w}} | {e}" for k, e in enumerate(evs, 1)]
        lines.append(f"final {res['final_stage']}")
        lines += state_table(res["final"].items()) or ["  (empty)"]
        self._say("\n".join(lines))

    def _llm_status(self, spec):
        """One line on the endpoint, VALIDATED where that is cheap:
        for ollama, the /api/tags URI says whether the server is up
        and the model pulled — a 404 at narrate time is too late."""
        from metis.loop.llm import DEFAULT_SPEC, resolve
        ep = resolve(spec or DEFAULT_SPEC)
        shown = spec or f"(default: {DEFAULT_SPEC})"
        if ep.backend != "ollama":
            return f"llm: {shown}"
        from metis.loop import ollama
        host = ollama.default_host()
        model = ep.arg or ollama.default_model()
        from metis.kernel.diagnostics import note as dnote

        def llm_note(text, problem):
            _note(self, dnote(
                "llm-endpoint", text, "llm", (f"llm {shown}",),
                (("endpoint", str(shown)), ("host", host),
                 ("problem", problem), ("model", model))))
        try:
            have = ollama.tags(host)
        except OSError as e:
            text = (f"llm: {shown} — ollama UNREACHABLE at {host} "
                    f"({getattr(e, 'reason', e)})")
            llm_note(text, f"ollama unreachable at {host}")
            return text
        ok = model in have or f"{model}:latest" in have
        note = "" if ok else f" — '{model}' NOT pulled"
        if not ok:
            llm_note(f"llm: {shown} @ {host}{note}",
                     f"model {model} not pulled")
        return (f"llm: {shown} @ {host} "
                f"(installed: {', '.join(have) or 'none'}){note}")

    def do_llm(self, arg):
        """show/set the LLM endpoint (loop.llm): ollama[:MODEL] |
        anthropic[:MODEL] | command:ARGV; seeded by [llm] endpoint
        in the config dot file"""
        from metis.loop.llm import resolve
        a = arg.strip()
        if not a:
            self._say(self._llm_status(self.llm))
            return
        try:
            resolve(a)
        except ValueError as e:
            self._say(str(e))
            return
        self.llm = a
        self._say(self._llm_status(a))

    def do_narrate(self, arg):
        """tell this session's fired events (or 'narrate SEED' for a
        fresh seeded trace) as prose: Foata beats + final state + the
        catalog's %% docs through the configured llm endpoint — the
        loop's narrate-out boundary. `narrate system` prints the
        doctrine prompt, `narrate user [SEED]` the certified
        material, neither calls the model."""
        from metis.loop.narrate import narrate, user_prompt
        parts = arg.split()
        show = parts.pop(0) if parts and parts[0] in ("system",
                                                      "user") else None
        if show == "system":
            return self._sys_prompt("narrate", " ".join(parts))
        a = " ".join(parts)
        if a:
            try:
                seed = int(a)
            except ValueError:
                self._say("usage: narrate [system|user] [SEED]")
                return
            idx, (stage, kappa) = run_trace(self.prog,
                                            self.case["steps"] or 6,
                                            seed)
            rows = [self.events_tbl[i] for i in idx]
        else:
            seed = 0
            rows = list(self.fired_rows)
            stage = self.stage
            kappa = [(q, n) for q, n in sorted(self.counts.items())
                     if n > 0]
        if not rows:
            self._say("nothing to narrate — step/run first, or "
                      "'narrate SEED' for a sampled trace")
            return
        if show == "user":
            self._say(user_prompt(rows, stage, kappa, self.text,
                                  seed=seed))
            return
        try:
            text = narrate(self.llm or "ollama", rows, stage, kappa,
                           self.text, seed=seed,
                           system=self.sys_prompts.get("narrate"))
        except Exception as e:
            self._say(f"narrate failed ({type(e).__name__}): {e}")
            return
        self._say(text)

    def do_trace(self, arg):
        """replay the file's embedded '#'-case (= metisc --run):
        fresh ground from #init, seeded I/T/F/Q lines + exact= where
        tractable. Leaves the interactive session untouched."""
        try:
            from metis.loop.wire import _scene_prog
            prog = _scene_prog(self)
            spoke = []
            self._say(render_run(self.path.stem, self.case, prog,
                                 pretty=True, notes=spoke))
            for d in spoke:
                _note(self, d)
        except DirectiveError as e:
            self._say(f"cannot trace: {e} — e.g. '#steps 6' in the "
                      "file or as a sent line")

    def _directive_block(self):
        lines = [f"#init {a} {n}"
                 for a, n in sorted(self.counts.items())]
        if self.case["steps"]:
            lines.append(f"#steps {self.case['steps']}")
        if self.case["seeds"]:
            lines.append(f"#seeds {self.case['seeds']}")
        lines += [f"#query {q}" for q in self.case["queries"]]
        return lines

    def do_save(self, arg):
        """session state as '#'-directives (stdout, or a .llp file)"""
        block = self._directive_block()
        path = arg.strip()
        if not path:
            for l in block:
                self._say(l)
            return
        body = [l for l in self.text.splitlines()
                if not l.lstrip().startswith("#")]
        while body and not body[-1].strip():
            body.pop()
        Path(path).write_text("\n".join(body + block) + "\n")
        self._say(f"wrote {path} (catalog + {len(block)} directives) "
                  f"— replay: metisc --run {path}")

    def do_log(self, arg):
        """fired events this session"""
        self._say("\n".join(f"  {k}. {n}"
                            for k, n in enumerate(self.fired, 1))
                  or "(nothing fired yet)")

    def do_graph(self, arg):
        """the causality graph (spec 09) since the last reset/assert/
        retract/#init: beats with covering parents, * = frontier;
        graph F.dot / F.json writes the DOT / JSON mirror"""
        from metis.lang.causality import as_json, beats_text, dot, \
            header
        g, n = self.causal()
        k = self._base[0]
        title = header(self.path.name, self.key, self.stage, n) + (
            f" (since #{k}: base #0 = the state then)" if k else "")
        a = arg.strip()
        if not a:
            self._say("\n".join(beats_text(g, title)))
            return
        if a.endswith(".dot"):
            text = dot(g, title)
        elif a.endswith(".json"):
            text = as_json(g, self.stage, self.key)
        else:
            self._say("usage: graph [FILE.dot | FILE.json]")
            return
        try:
            Path(a).expanduser().write_text(text)
        except OSError as e:
            self._say(f"cannot write {a}: {e.strerror}")
            return
        self._say(f"wrote {a} ({len(g.nodes)} nodes, {len(g.edges)} "
                  f"edges, frontier {' '.join(map(str, g.frontier))})")

    def _occ_matcher(self, word):
        """Event-name matcher for why/causes: the ground name, the
        rule name, or the source rule of an ⊕ branch."""
        spans = self._rule_spans()

        def match(name):
            base = name.partition("[")[0]
            return word in (name, base, _source_name(base, spans))
        return match

    def _rule_head(self, name):
        """(lineno, head text) of an event's rule, or None."""
        spans = self._rule_spans()
        span = spans.get(_source_name(name.partition("[")[0], spans))
        if span is None:
            return None
        return span[1], self.text.splitlines()[span[1] - 1]

    def _why(self, arg):
        """(title, cone, graph) for `why [ATOM|K]`, or an error str."""
        from metis.kernel.nets import cone, live_roots
        g, n = self.causal()
        k0, base_stage, base_counts = self._base
        a = arg.strip()
        if not a:
            stage = None if self.stage == DONE else self.stage
            roots = live_roots(self.fired_rows[k0:], base_counts,
                               base_stage, self.counts, stage,
                               first=k0 + 1)
            what = "the current state"
        elif a.isdigit():
            k = int(a)
            if not (k == 0 or k0 < k <= len(self.fired)):
                return (f"no event {k} in the graph — it covers "
                        f"{k0 + 1}..{len(self.fired)} (0 = the base)")
            roots, what = [k], f"event {k}"
        else:
            rows = self.fired_rows[k0:]
            prod = [k0 + 1 + j for j, (_, _, ev) in enumerate(rows)
                    if a in ev.produce]
            if prod:
                roots = prod[-1:]
                what = f"{a} (last produced by event {prod[-1]})"
            elif base_counts.get(a, 0) > 0:
                roots, what = [0], f"{a} (from the base state)"
            else:
                return (f"{a} was never produced and is not in the base "
                        "state — 'state' lists atoms, 'log' events")
        if a and not a.isdigit() and self.counts.get(a, 0) == 0:
            what += " — not present now"
        members = cone(g, roots)
        title = (f"why {what}: {len(members)} of {len(g.nodes)} "
                 f"occurrences")
        return title, members, g

    def do_why(self, arg):
        """why [ATOM|K]: the causal cone — bare: what the current state
        rests on (events that wrote a PRESENT atom or the stage, and
        their causes; absences do not count); ATOM: the last event
        that produced it; K: event K. Atoms are indistinguishable:
        this is dependence, not token lineage."""
        from metis.lang.causality import cone_text
        res = self._why(arg)
        if isinstance(res, str):
            self._say(res)
            return
        title, members, g = res
        self._say("\n".join(cone_text(g, members, title,
                                      self._rule_head)))

    def do_causes(self, arg):
        """causes A B: for each occurrence of B, is an occurrence of A
        among its causes (TeLLer's x) — rows + yes/some/no"""
        from metis.lang.causality import causes
        parts = arg.split()
        if len(parts) != 2:
            self._say("usage: causes A B   (event or rule names)")
            return
        g, _ = self.causal()
        res = causes(g, self._occ_matcher(parts[0]),
                     self._occ_matcher(parts[1]))
        if res["verdict"] == "none":
            self._say(f"{parts[1]} has not fired since the base — "
                      "'log' lists events")
            return
        for r in res["rows"]:
            got = (" ".join(map(str, r["a"])) if r["a"] else "no")
            self._say(f"  {r['b']} {r['name']}  <- {parts[0]}: {got}")
        self._say(f"{res['verdict']}: {parts[0]} causes "
                  f"{sum(1 for r in res['rows'] if r['a'])} of "
                  f"{len(res['rows'])} {parts[1]}")

    def do_version(self, arg):
        """version — what exactly is running: package, source
        checkout (+ git), python, config dot file, llm endpoint"""
        self._say("\n".join(version_lines()
                            + [self._llm_status(self.llm)]))

    def do_key(self, arg):
        """canonical program_key of the serving catalog"""
        self._say(self.key)

    def do_help(self, arg):
        if arg:
            super().do_help(arg)
        else:
            self._say(_HELP)

    def onecmd(self, line):
        """Ctrl-C cancels the RUNNING command (a long `query`, a
        stuck backend), never the shell — exit stays quit/ctrl-d. A
        command that raises is reported, and the session survives."""
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            self._say("(interrupted)")
        except Exception as e:
            self._say(f"error ({type(e).__name__}): {e}")

    def do_quit(self, arg):
        """leave the shell"""
        return True

    do_exit = do_quit
    do_EOF = do_quit

    def emptyline(self):            # never repeat-last (cmd default):
        pass                        # blank lines in a sent buffer
                                    # selection must be inert

    def default(self, line):
        s = line.strip()
        if not s or s.startswith("%"):
            return                      # comments pass silently, so a
                                        # buffer selection sends whole
        if s.startswith("#"):
            return self._directive(s)
        if self._looks_like_source(s):
            self._say("that is LLP source — rules live in the FILE: "
                      "edit it, save, then 'reload' here (session "
                      "state carries over)")
            return
        if s.startswith("+"):
            return self.do_assert(s[1:])
        if s.startswith("-"):
            return self.do_retract(s[1:])
        self._say(f"unknown command: {s.split()[0]!r} — commands are "
                  "bare words; 'help' lists them ('#' only prefixes "
                  "directive lines)")

    _DECL_WORDS = frozenset(
        ["catalog", "provenance", "extends", "include", "layers",
         "stages", "type", "namespace", "pred", "bwd", "weight",
         "guard", "input", "output", "stage", "qui"])

    def _looks_like_source(self, s):
        """A buffer selection of RULE text sent here (shift+enter)
        deserves a pointer to the save+reload flow, not per-line
        'unknown command' noise — and `-o ...` must never fall into
        the '-ATOM' retract shorthand."""
        parts = s.split()
        return (parts[0] in self._DECL_WORDS
                or "-o" in parts
                or s in ("{", "}")
                or bool(re.match(r"[a-z][A-Za-z0-9_-]*\s*\[", s)))

    def _directive(self, s):
        """A '#'-directive line sent from the buffer (the editor's
        shift+enter) applies to the LIVE session: #init is an
        assignment (sets the count, unlike assert's add), #query
        runs, #steps/#seeds/#unknown/#lik update the case that
        `query` defaults and `save` write out."""
        parts = s[1:].split()
        if not parts:
            return
        key, args = parts[0], parts[1:]
        try:
            if key == "init" and len(args) == 2:
                atom, n = args[0], int(args[1])
                self._mark()
                if n > 0:
                    self.counts[atom] = n
                else:
                    self.counts.pop(atom, None)
                self._rebase()
                self.case["init"][atom] = n
                self._say(f"{atom}:{n}")
            elif key == "steps" and len(args) == 1:
                self.case["steps"] = int(args[0])
                self._say(f"steps {args[0]}")
            elif key == "seeds" and len(args) == 1:
                self.case["seeds"] = int(args[0])
                self._say(f"seeds {args[0]}")
            elif key == "query" and len(args) == 1:
                if args[0] not in self.case["queries"]:
                    self.case["queries"].append(args[0])
                self.do_query(args[0])
            elif key == "unknown" and len(args) == 1:
                if args[0] not in self.case["unknowns"]:
                    self.case["unknowns"].append(args[0])
                self._say(f"unknown {args[0]}")
            elif key == "lik" and args:
                from metis.lang.directives import _clamp
                self.case["liks"].append([_clamp(t) for t in args])
                self._say(f"lik {' '.join(args)}")
            else:
                self._say(f"bad directive: {s}")
        except ValueError:
            self._say(f"bad directive: {s}")

    # -- completion (readline) -----------------------------------------

    def _complete_atoms(self, text):
        return sorted(a for a in self.atoms if a.startswith(text))

    complete_assert = complete_retract = complete_query = \
        lambda self, text, *rest: self._complete_atoms(text)

    def complete_step(self, text, *rest):
        return sorted({self.events_tbl[i][2].name
                       for i, _ in self._candidates()
                       if self.events_tbl[i][2].name.startswith(text)})


def _history():
    base = Path(os.environ.get("XDG_STATE_HOME",
                               Path.home() / ".local/state"))
    d = base / "metis"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d / "history"


def version_lines():
    """Toolchain provenance — what exactly is running."""
    import platform
    import metis as _m
    from importlib import metadata
    from metis.loop import config
    try:
        v = metadata.version("metispy")
    except metadata.PackageNotFoundError:
        v = "(not installed)"
    src = Path(next(iter(_m.__path__))).resolve()
    git = ""
    try:
        import subprocess
        r = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            git = f" (git {r.stdout.strip()})"
    except OSError:
        pass
    cfg = config.path()
    return [f"metispy {v} @ {src}{git}",
            f"python {platform.python_version()} — {sys.executable}",
            f"config {cfg}" if cfg else
            "config (none: .metis.toml / ~/.config/metis/config.toml)"]


# the tabula-rasa world (SPEC §4): served when no file is given;
# `revise` + `accept` replace it, `save`/`export` keep the result.
_SCRATCH = """\
catalog scratch 1.
layers (dyn).
stages (z).
stage z {
}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="metis",
        description="interactive shell over a .llp catalog "
                    "(line editing, history and tab completion via "
                    "readline)")
    ap.add_argument("llp", nargs="?",
                    help="the catalog to serve; none serves a blank "
                         "scratch world — build it in-session "
                         "(revise/accept, or load a candidate) and "
                         "save/export what you keep")
    ap.add_argument("--seed", type=int, default=1,
                    help="sampler seed (default 1)")
    ap.add_argument("--version", action="store_true",
                    help="toolchain provenance (package, source+git, "
                         "python, config), then exit")
    ap.add_argument("--json", action="store_true",
                    help="the JSON-lines wrapper wire: one {op, arg} "
                         "per stdin line, one JSON result per line "
                         "(the web console's channel)")
    args = ap.parse_args(argv)
    if args.version:
        print("\n".join(version_lines()))
        return 0
    if not args.llp:
        import tempfile
        fd, scratch = tempfile.mkstemp(suffix=".llp",
                                       prefix="metis-scratch-")
        os.write(fd, _SCRATCH.encode())
        os.close(fd)
        args.llp = scratch
        print(f"no catalog given — serving a blank scratch world "
              f"({scratch}); revise/accept to build it, save/export "
              f"to keep it", file=sys.stderr)
    if args.json:
        from metis.loop.wire import serve_json
        try:
            shell = Repl(args.llp, seed=args.seed)
        except Exception as e:
            import json as _json
            findings = getattr(e, "findings", None) or [str(e)]
            print(_json.dumps({"ok": False, "findings": findings}),
                  file=sys.stderr)
            return 1
        return serve_json(shell)
    try:
        import readline
        hist = _history()
        if hist:
            try:
                readline.read_history_file(hist)
            except OSError:
                pass
            readline.set_history_length(1000)
            atexit.register(lambda: readline.write_history_file(hist))
    except ImportError:
        pass                        # line editing degrades gracefully
    try:
        repl = Repl(args.llp, seed=args.seed)
    except Exception as e:
        findings = getattr(e, "findings", None) or [str(e)]
        print("cannot serve %s:\n  %s"
              % (args.llp, "\n  ".join(findings)), file=sys.stderr)
        return 1
    while True:
        try:
            repl.cmdloop()
            return 0
        except KeyboardInterrupt:
            # Ctrl-C at the prompt: drop the line, keep the session.
            print("^C ('quit' or ctrl-d to leave)")
            repl.intro = None


if __name__ == "__main__":
    sys.exit(main())
