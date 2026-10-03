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
from metis.lang.directives import (DirectiveError, embedded_case,
                                   render_run, state_table)

_HELP = """\
events              enabled transitions of the firing tier, numbered
step [k|name]       fire candidate k (or by name); no arg = sampled
run N               sample N steps (stops at quiescence)
state               current stage + multiset
assert ATOM [N]     add N (default 1) tokens          [alias: +]
retract ATOM [N]    remove N (default 1) tokens       [alias: -]
query ATOM [STEPS]  exact presence marginal from HERE (filter)
seed N              reseed the sampler
undo                back one mutating command
reset               back to the file's #init directives
reload              re-read + recompile the file; state carries over
                    (alias: edit — rule changes go file -> reload)
trace               replay the file's embedded '#'-case, fresh from
                    #init (= metisc --run); session state untouched
save [PATH]         session state as '#'-directives (stdout or file)
log                 fired events this session
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
        from metis.loop import config
        self.llm = config.get("llm", "endpoint", "")  # spec (loop.llm)
        self.sys_prompts = {}  # per-seat session overrides (prompts)
        self._load(initial=True)
        flat = _fmt_counts(self.counts)
        if len(flat) > 60:              # tragedy wraps 5 lines flat:
            toks = sum(self.counts.values())
            flat = (f"{len(self.counts)} atoms, {toks} tokens "
                    f"— 'state' tabulates them")
        self.intro = (
            f"metis shell — {self.path.name} "
            f"(catalog {self.doc.catalog.name}, key {self.key[:12]}…)\n"
            f"stage {self.stage} | {flat}\n"
            "type 'help' for commands, 'events' to see what can fire")

    # -- catalog / session plumbing ------------------------------------

    def _compile(self):
        text = self.path.read_text()
        ast = parse(text)
        doc = compile_catalog(resolve_includes(ast, self.path.parent))
        return text, doc, embedded_case(text)

    def _load(self, initial=False):
        text, doc, case = self._compile()
        self.text, self.doc, self.case = text, doc, case
        self.prog = doc.catalog.ground(dict(case["init"]))
        self.events_tbl = canonical_events(self.prog)
        self.key = program_key(self.prog)
        if initial or self.stage not in (
                list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(case["init"])
        self.atoms = set(self.case["init"])
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    def _reserve(self, catalog):
        """Re-serve a merged/revised catalog LIVE (loop accept):
        new ground, events, key, atoms; counts and stage carry over
        exactly as `reload` does."""
        self.doc = replace(self.doc, catalog=catalog)
        self.prog = catalog.ground(dict(self.case["init"]))
        self.events_tbl = canonical_events(self.prog)
        self.key = program_key(self.prog)
        if self.stage not in (list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(self.case["init"])
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    def _mark(self):
        self.undo_stack.append(
            (self.stage, dict(self.counts), len(self.fired)))

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
            self._say(f"{e} (try: query {atom} {max(1, e.step - 1)})")
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
        self.stage, self.counts, k = self.undo_stack.pop()
        del self.fired[k:]
        del self.fired_rows[k:]
        self.do_state("")

    def do_reset(self, arg):
        """back to the file's #init directives"""
        self._mark()
        self.stage = self.prog.init_stage
        self.counts = dict(self.case["init"])
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
        what = "unchanged" if self.key == old_key else "NEW key"
        self._say(f"reloaded {self.path.name} ({what} "
                  f"{self.key[:12]}…); "
                  f"stage {self.stage} | {_fmt_counts(self.counts)}")

    do_edit = do_reload                 # the sketch's name, kept

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

    def do_gate(self, arg):
        """gate the candidate as a PACK against the served world"""
        self._loop_say(self.cmd_gate(arg))

    def do_revise(self, arg):
        """gate the candidate as a BASE EDIT (accept then replaces
        the world; measure diff shown)"""
        self._loop_say(self.cmd_revise(arg))

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
        try:
            have = ollama.tags(host)
        except OSError as e:
            return (f"llm: {shown} — ollama UNREACHABLE at {host} "
                    f"({getattr(e, 'reason', e)})")
        ok = model in have or f"{model}:latest" in have
        note = "" if ok else f" — '{model}' NOT pulled"
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
            prog = self.doc.catalog.ground(dict(self.case["init"]))
            self._say(render_run(self.path.stem, self.case, prog,
                                 pretty=True))
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
        stuck backend), never the shell — exit stays quit/ctrl-d."""
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            self._say("(interrupted)")

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
