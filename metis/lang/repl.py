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
from metis.lang.cone import NotSettled
from metis.lang.measure import (Arm, cone_names, footer, measure,
                                parse_tail, pval, readout_text,
                                structural)
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
transcript [FILE]   lab-notebook log of this session (terminal only;
                    'transcript off' closes it)
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
narrate [SEED]      the session so far as prose — or, with a number, a
                    fresh trace SAMPLED with that seed (not N beats):
                    Foata beats + %% docs through the llm endpoint.
                    Catalog %% conventions: `%% [fold]` bookkeeping
                    (not told), `%% [fold P] words` (a run of it is
                    one line + the net-new P atoms), `%% a | b` on a
                    ⊕ rule (branch N tells phrase N); `% evidence:`
                    lines never reach the narrator. A narration cut at
                    the endpoint's cap ends on a sentence and says so
narrate system|user show the system / certified-material prompt
                    (no model call; the UI halves). `narrate system
                    <<EOF` sets a session override; the persistent
                    home is ~/.config/metis/prompts/narrate.txt
                    (likewise `formalize system [<<EOF]` /
                    prompts/formalize.txt — narrators differ)
ports (PLUGIN-PORTS.md) — real values and time behind the catalog;
operator code, so TERMINAL ONLY (D4):
plugin [SPEC|off]   show / attach (pkg.mod:factory | file.py:factory)
                    / detach the port plugin (also '#plugin SPEC' in
                    the file, or [ports] plugin in the dot file);
                    '#notch FILE' in the file ships a fitted overlay
                    at serve, as `notch load FILE` (`notch pop` drops it)
ports               declared weight ports: bound?, read scope, the
                    rules referencing them
port F(ARGS)        evaluate one binding against the current counts
                    (scoped: an out-of-scope read raises)
tick [N]            advance the env N beats by hand (its do-lines join
                    the trace as [env] rows); step/run tick on their
                    own when the world is quiet — quiet, not dead
notch [ALPHA]       fit the ports from this session's sampled steps
                    (Dirichlet per read-scope row), print the row
                    sheet; `notch accept` ships it as an OVERLAY
                    (never a catalog edit), `notch pop` drops one;
                    `notch load FILE` ships a fitted overlay file
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


def _ground_args(name):
    """`rule[X=a,Y=b]` -> ('a', 'b'): a fired event's ground arguments,
    as on_event receives them."""
    if "[" not in name:
        return ()
    tag = name.split("[", 1)[1].rstrip("]")
    return tuple(kv.split("=", 1)[-1] for kv in tag.split(",") if kv)


class _EnvPricer:
    """The session pricer bound to ONE env snapshot: before a binding
    runs, the plugin's env is restored to the arm's snapshot (lazily —
    only when another arm's env is loaded). Port values are constants
    per arm: no tick happens inside a measure."""

    def __init__(self, repl, snap):
        self.repl, self.snap = repl, snap
        self.base = repl.pricer

    def _ensure(self):
        r = self.repl
        if r.plugin is not None and self.snap is not None and \
                r.__dict__.get("_env_live") is not self.snap:
            r.plugin.restore(self.snap)
            r._env_live = self.snap

    def site(self, name):
        return self.base.site(name)

    def raw(self, name, counts):
        self._ensure()
        return self.base.raw(name, counts)

    def factor(self, name, counts):
        self._ensure()
        return self.base.factor(name, counts)

    def policy(self, program):
        pol = self.base.policy(program)

        def bound(stage, counts, ws):
            self._ensure()
            return pol(stage, counts, ws)
        return bound


class _Tee(io.TextIOBase):
    """stdout for one logged command: written through AND kept."""

    def __init__(self, under):
        self.under, self.buf = under, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.under.write(s)

    def flush(self):
        self.under.flush()

    def getvalue(self):
        return self.buf.getvalue()


def _now_s():
    import time as _time
    return _time.strftime("%Y-%m-%d %H:%M:%S")


class Repl(LoopVerbs, Cmd):
    prompt = "llp> "

    def __init__(self, path, seed=1, terminal=False, plugin=None,
                 registry=None, host=False, **kw):
        super().__init__(**kw)
        # the interactive TERMINAL shell (main() passes True): the only
        # session that writes the operator's disk (transcript) or
        # imports operator code (plugin) — D4
        self.terminal = terminal
        # host=True: a RUNTIME process the operator spawned with
        # `metis --json --host SPEC` (runtime-wire, INTEGRATION.md):
        # the constructor's plugin — named on the operator's argv,
        # never by the file or the wire — attaches off the terminal;
        # the file's #plugin and the `plugin` verb stay refused (D4)
        self._host = bool(host and plugin)
        # port plugins (PLUGIN-PORTS.md): the plugin, its spec, the
        # registry it bound, the session's pricer (None = no ports);
        # _envs[j] = the env snapshot after j fired rows (kept beside
        # the world so fork/back/undo/whatif restore it); _env0 = the
        # env as attached (reset returns to it)
        # `registry`: an embedding HOST's in-process bindings (no
        # operator code imported — allowed off the terminal); a plugin,
        # when one attaches, replaces it
        self.plugin, self.plugin_spec = None, ""
        self.registry, self.pricer = registry, None
        self._envs, self._env0 = [], None
        self._notch = None              # a fitted, not yet shipped fit
        self._overlays = []             # shipped notch overlays (names)
        self._overlay_names = []        # per overlay: "notch fit" or its file
        self._overlay_srcs = []         # per overlay: (resolved file, ports) — a file ships once
        # `plugin SPEC|off` (or the constructor) pins the spec over the
        # file's #plugin / the dot file's [ports] plugin; None = unpinned
        self._plugin_pin = plugin
        self._port_warns = []
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
        # spec 10 (interventions): per fired row HOW it was chosen —
        # {"how": "sampled"|"forced"|"do", "p": its tier probability};
        # forced and do rows are conditioning, not sampling, and are
        # excluded from the trace likelihood. masks block events/rules
        # out of the tier (`do ~EVENT`); the ledger records the do's;
        # _base_do names the state do's the current base is made of;
        # shelf = the one shelved world while forked (D1).
        self.fired_how = []
        self.masks = set()
        self.ledger = []
        self._base_do = []
        self.shelf = None
        self.fork_at = None
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
        warn += "".join(f"warning: {w}\n" for w in self._port_warns)
        if self.plugin is not None:
            warn += (f"plugin {self.plugin_spec} "
                     f"({type(self.plugin).__name__}) attached — "
                     f"{len(self.doc.ports.weights)} weight port(s); "
                     "the env ticks when the world is quiet\n")
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
        # ports bind BEFORE anything is committed: an unbound port
        # refuses to serve (initial) / keeps the old catalog (reload)
        self._bind(doc, case, initial=initial)
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
        self._ship_pending_notch()
        if initial or self.stage not in (
                list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(case["init"])
            self._rebase()
        self.atoms = set(self.case["init"])
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    def _reserve(self, catalog, ports=None):
        """Re-serve a merged/revised catalog LIVE (loop accept):
        new ground, events, key, atoms; counts and stage carry over
        exactly as `reload` does. `ports`: the new world's port
        surface (a pack's merged over the base's)."""
        doc = replace(self.doc, catalog=catalog,
                      **({"ports": ports} if ports is not None else {}))
        self._bind(doc, self.case)
        self.doc = doc
        self.__dict__["notes"] = []     # a new world: old notes retire
        self.prog = catalog.ground(dict(self.case["init"]))
        self.events_tbl = canonical_events(self.prog)
        self.key = program_key(self.prog)
        self._ship_pending_notch()
        if self.stage not in (list(self.prog.stages) + [DONE]):
            self.stage = self.prog.init_stage
            self.counts = dict(self.case["init"])
            self._rebase()
        for _, _, ev in self.events_tbl:
            self.atoms |= set(ev.consume) | set(ev.produce) \
                | set(ev.persist)

    # -- ports: the plugin, the registry, the pricer (PLUGIN-PORTS.md) --

    def _bind(self, doc, case, initial=False):
        """Resolve the plugin spec (pin > #plugin > [ports] plugin),
        import + bind it if it changed, and build the session pricer
        for `doc`'s weight ports. Raises (PluginError, or PortError on
        an unbound port — fail fast, the BT.cpp rule) WITHOUT touching
        the session; commits only on success. Plugins are operator
        code: a non-terminal session never imports one (D4)."""
        from metis.fw.plugin import load_plugin
        from metis.fw.registry import PortError, Registry
        from metis.lang.compiler import LangError, PortPricer
        from metis.loop import config
        warns = []
        spec = self._plugin_pin
        if spec is None:
            spec = case.get("plugin") or config.get("ports", "plugin", "")
        spec = (spec or "").strip()
        if spec and not self.terminal and not (
                self._host and spec == (self._plugin_pin or "").strip()):
            warns.append(f"plugin {spec} not attached — plugins are "
                         "terminal-only (D4: this session never imports "
                         "operator code)")
            spec = ""
        plugin, reg = self.plugin, self.registry
        if spec != self.plugin_spec or (spec and plugin is None):
            plugin, reg = None, None
            if spec:
                plugin = load_plugin(spec, case, self.path.parent)
                reg = Registry()
                plugin.bind(reg)
        pricer = None
        if doc.ports.weight_refs:
            try:
                pricer = PortPricer(
                    doc.ports, reg.weight_bindings() if reg else {},
                    horn=doc.catalog.horn_db())
            except LangError as e:
                missing = sorted({r.factor for r in
                                  doc.ports.weight_refs.values()}
                                 - set(reg.weight_bindings() if reg
                                       else ()))
                err = PortError(missing)
                found = None if spec else self._plugin_beside()
                how = ("plugin " + spec + " does not bind it" if spec
                       else ("no plugin binds it — one is beside the "
                             f"catalog: serve it with `{found[0]}`, or add "
                             f"'{found[1]}' to {self.path.name}" + found[2])
                       if found
                       else "no plugin binds it: serve it with `metis "
                       "--plugin pkg.mod:factory FILE` (catalog-served "
                       "files), or put '#plugin pkg.mod:factory' in a "
                       "scene file")
                how += ("" if spec or self.terminal else
                        " (plugins attach in the terminal shell only)")
                err.findings = [f"unbound weight port '{m}' — {how}"
                                for m in missing] or list(
                                    getattr(e, "findings", [str(e)]))
                raise err from None
            for f in pricer.opaque:
                warns.append(f"weight port '{f}' declares no `reads` — "
                             "opaque: it sees the whole multiset (not "
                             "factorizable)")
        attach = plugin is not self.plugin
        self.plugin, self.registry, self.pricer = plugin, reg, pricer
        self.plugin_spec = spec if plugin is not None else ""
        self._port_warns = warns
        if attach:
            self._env_attach()
            # a fresh registry carries no overlay: the file's #notch
            # lines ship again on it (as `notch load` would)
            self._overlays, self._overlay_names, self._overlay_srcs = [], [], []
            self._notch_pending = list(case.get("notch") or []) \
                if plugin is not None else []

    def _seeded_trace(self, seed):
        """`narrate SEED`'s fresh trace: the reference sampler from the
        initial state, priced by THIS session's weight ports — the
        plugin's base binding and every notch overlay shipped on it
        (`notch accept`, `notch load`). A prior-flat base port prices
        flat until a fitted overlay is shipped."""
        idx, (stage, kappa) = run_trace(
            self.prog, self.case["steps"] or 6, seed,
            policy=self.pricer.policy(self.prog, canonical=True)
            if self.pricer else None)
        return [self.events_tbl[i] for i in idx], stage, kappa

    def _plugin_beside(self):
        """A port plugin by convention beside the catalog — <stem>_plugin.py
        or <stem>/<stem>_plugin.py, its factory `make` (else its first
        `def NAME(case`) — as (the serve command, the #plugin line, a
        hint for a fitted overlay <stem>.notch.json beside it); None when
        there is none."""
        d, stem = self.path.parent, self.path.stem
        for f in (d / f"{stem}_plugin.py", d / stem / f"{stem}_plugin.py"):
            if not f.exists():
                continue
            src = f.read_text(errors="replace")
            m = re.search(r"^def (\w+)\(case", src, re.M)
            fac = "make" if re.search(r"^def make\(", src, re.M) or not m \
                else m.group(1)
            spec = f"{f.relative_to(d).as_posix()}:{fac}"
            notch = d / f"{stem}.notch.json"
            return (f"metis --plugin {spec} {self.path.name}",
                    f"#plugin {spec}",
                    (f"; then `notch load {notch.name}` for the fitted "
                     "world" if notch.exists() else ""))
        return None

    def _ship_pending_notch(self):
        """The file's #notch overlays, once the catalog is compiled and
        its plugin's fresh registry is bound (serve, reload, plugin)."""
        pending, self._notch_pending = getattr(self, "_notch_pending", []), []
        for f in pending:
            if not self._notch_load(f, quiet=True):
                self._port_warns.append(f"#notch {f}: not shipped")

    def _pricing(self):
        """What prices the session's weight ports, in one line: the
        plugin's base binding (its `name`, else its spec) and the
        overlays shipped on it — '' for a catalog without weight ports."""
        if self.pricer is None:
            return ""
        base = getattr(self.plugin, "name", "") or self.plugin_spec or "no plugin"
        return (f"priced by {base}; overlays: "
                + (", ".join(self._overlay_names) or "none"))

    def _notch_load(self, arg, quiet=False):
        """notch load FILE: a fitted overlay file (the recorder's
        --overlay format, {"rows": [{port, args, row, notched}]}) shipped
        on the registry as `notch accept` ships a fit."""
        if not arg:
            self._say("usage: notch load FILE (a fitted overlay, e.g. "
                      f"{self.path.stem}.notch.json)")
            return False
        if self.registry is None:
            self._say("no port plugin attached — an overlay rides on the "
                      "plugin's registry; serve the catalog with it first")
            return False
        f = Path(arg).expanduser()
        if not f.is_absolute():
            f = next((c for c in (self.path.parent / f, Path.cwd() / f)
                      if c.exists()), self.path.parent / f)
        if not f.exists():
            self._say(f"no overlay file {arg}")
            return False
        from metis.lang.overlay import overlay_table as _overlay_table
        try:
            tables = _overlay_table(f)
        except (ValueError, KeyError, TypeError) as e:
            self._say(f"not an overlay file ({type(e).__name__}: {e}) — "
                      "want {\"rows\": [{port, args, row, notched}]}")
            return False
        ports = {r.factor for r in self.doc.ports.weight_refs.values()}
        unknown = sorted(set(tables) - ports)
        if unknown:
            self._say(f"{f.name} prices port(s) this catalog does not "
                      f"declare: {', '.join(unknown)} — not loaded")
            return False
        src = (str(f.resolve()), tuple(sorted(tables)))
        if src in self._overlay_srcs:
            # the #notch directive (or an earlier load) shipped this file on
            # these ports: a second copy would price the catalog twice
            if not quiet:
                self._say(f"overlay {f.name} already shipped on {', '.join(sorted(tables))}"
                          f" (depth {len(self._overlays)}) — not loaded twice; `notch pop` drops it")
            return True
        for port, ov in sorted(tables.items()):
            self.registry.overlay(port, ov)
        self._overlays.append(sorted(tables))
        self._overlay_names.append(f.name)
        self._overlay_srcs.append(src)
        self._rebuild_pricer()
        if quiet:
            return True
        self._say(f"overlay {f.name} shipped on {', '.join(sorted(tables))}"
                  f" (depth {len(self._overlays)}) — the catalog is "
                  f"unchanged (key {self.key[:12]}…); `notch pop` drops it")
        return True

    def _rebuild_pricer(self):
        """After the registry's overlay chain changed (notch)."""
        from metis.lang.compiler import PortPricer
        if self.doc.ports.weight_refs and self.registry is not None:
            self.pricer = PortPricer(self.doc.ports,
                                     self.registry.weight_bindings(),
                                     horn=self.doc.catalog.horn_db())

    def _env_snap(self):
        """The env as the neutral JSON env (metis.kernel.record): every
        position's snapshot, the fork's and the record's, is JSON data,
        and a restore is handed that form (a host that is not JSON
        fails here, at attach)."""
        if self.plugin is None:
            return None
        from metis.kernel.record import env_json
        return env_json(self.plugin.snapshot())

    def _env_attach(self):
        """A (new) plugin: its env as attached is the env of every
        position so far (the honest best: it was not there before)."""
        snap = self._env_snap()
        self._env0 = snap
        self._envs = ([snap] * (len(self.fired) + 1)
                      if self.plugin is not None else [])

    def _env_to(self, snap):
        if self.plugin is not None and snap is not None:
            self.plugin.restore(snap)

    def _env_push(self):
        if self.plugin is not None:
            self._envs.append(self._env_snap())

    def _tick(self, quiet_rows=True):
        """One env beat (the INPUT face): plugin.tick(counts) -> its
        do-lines applied as ONE exogenous row (kind "exo", how "exo":
        conditioning, excluded from the likelihood like a do), the env
        snapshot recorded beside it. A tick REVIVES a quiescent world
        (G6: a shop with no customers is quiet, not dead). -> the row
        name, or None when the env's episode is over (tick -> None).
        quiet_rows=False (the runtime's watch): a beat with no input
        adds no row — the env moves on and the head's env snapshot
        follows it; -> "" for that quiet beat."""
        from metis.fw.plugin import exo_event
        if self.plugin is None:
            return None
        dos = self.plugin.tick(dict(self.counts))
        if dos is None:
            return None
        if not dos and not quiet_rows:
            if self._envs:
                self._envs[-1] = self._env_snap()
            return ""
        ev, after = exo_event(list(dos), self.counts)
        if self.stage == DONE:
            self.stage = self.__dict__.get("_quiet_stage") \
                or self.prog.init_stage
        self.counts = after
        self.fired_how.append({"how": "exo", "p": None})
        self.fired.append(ev.name)
        self.fired_rows.append(("exo", self.stage, ev))
        self._env_push()
        return ev.name

    def _latch_done(self):
        """Nothing can fire: remember the stage (a tick may revive it),
        then latch DONE."""
        if self.stage != DONE:
            self._quiet_stage = self.stage
        self.stage = DONE

    def _mark(self):
        self.undo_stack.append(
            (self.stage, dict(self.counts), len(self.fired), self._base,
             frozenset(self.masks), len(self.ledger), list(self._base_do),
             self.fork_at, self._env_snap()))

    def _rebase(self):
        """The current state becomes the causality graph's base: an
        edit no fired event explains (spec 09 — the graph is a pure
        function of base + the rows fired since)."""
        self._base = (len(self.fired), self.stage, dict(self.counts))
        self._base_do = []
        self.fork_at = None          # a new base: the cut leaves the graph

    def _masked(self, name):
        """D2: a mask names a ground event exactly, or a rule — all its
        ground instances and ⊕ branches."""
        if not self.masks:
            return False
        rule = name.partition("[")[0]
        return (name in self.masks or rule in self.masks
                or re.sub(r"-\d+$", "", rule) in self.masks)

    def likelihood(self):
        """(L, sampled, forced): the trace likelihood is the product of
        the SAMPLED steps' tier probabilities; forced/do steps are
        conditioning, not sampling (spec 07 P2, spec 10 P2), and are
        excluded — never priced as if chance had chosen them. Env
        (exo) rows are exogenous input: excluded too, and counted in
        neither (`log` says how many)."""
        lik, ns, nf = 1.0, 0, 0
        for h in self.fired_how:
            if h["how"] == "sampled":
                lik *= h["p"] if h["p"] is not None else 0.0
                ns += 1
            elif h["how"] != "exo":
                nf += 1
        return lik, ns, nf

    def marks(self):
        """{occurrence: "forced"|"do"|"fork"} for the graph renderers
        (spec 10 P2/P1), over the graph's occurrences (since base)."""
        k0 = self._base[0]
        out = {j: h["how"] for j, h in
               enumerate(self.fired_how[k0:], k0 + 1)
               if h["how"] != "sampled"}
        if self.fork_at is not None and self.fork_at >= k0:
            j = self.fork_at if self.fork_at > k0 else 0   # 0 = the base
            out[j] = (out[j] + "+fork") if j in out else "fork"
        return out

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
                  if k == kind and s == self.stage
                  and not self._masked(ev.name)]
            ws = [(i, w) for i, w in ws if w > 0]
            if ws and self.pricer is not None and kind == "clause":
                from metis.lang.cone import priced
                ws = priced(ws, lambda i: self.events_tbl[i][2].name,
                            self.counts, self.pricer)
            if ws:
                return ws
        return []

    def _fire(self, i, how="sampled", ws=None):
        """Fire event i; `ws` = the tier it was chosen from (computed
        here when the caller has not), for its recorded probability."""
        ws = self._candidates() if ws is None else ws
        total = sum(w for _, w in ws)
        w = dict(ws).get(i)
        h = {"how": how, "p": (w / total) if w and total else None}
        if self.masks:            # the tier this p was drawn from
            h["masks"] = sorted(self.masks)
        kind, _, ev = self.events_tbl[i]
        site = self.pricer.site(ev.name) if self.pricer else None
        if site is not None and how == "sampled":
            # notch's sufficient statistic: the port, its read-scope
            # row and its value at the moment chance chose this event
            factor, args, compl, scope = site
            row = tuple(sorted((a, int(self.counts.get(a, 0) > 0))
                               for a in (scope or ())))
            h["site"] = (factor, args, compl, row,
                         self.pricer.raw(ev.name, self.counts))
        self.fired_how.append(h)
        for q, n in ev.consume.items():
            self.counts[q] = self.counts.get(q, 0) - n
        for q, n in ev.produce.items():
            self.counts[q] = self.counts.get(q, 0) + n
        self.counts = {q: n for q, n in self.counts.items() if n > 0}
        if kind == "link":
            self.stage = ev.post
        self.fired.append(ev.name)
        self.fired_rows.append(self.events_tbl[i])
        if self.plugin is not None:
            # the OUTPUT face: the fired event actuates the env
            self.plugin.on_event(ev.name, _ground_args(ev.name))
            self._env_push()
            beat = getattr(self.plugin, "beat_stage", None)
            if kind == "link" and beat and ev.post == beat:
                self._tick()          # settle entry: one env beat
        return ev.name

    def _sample(self):
        """One normative selection step (06-sampler-wire §4)."""
        ws = self._candidates()
        if not ws:
            if self.plugin is not None:
                name = self._tick()       # quiet, not dead (G6)
                if name is not None:
                    return name
            self._latch_done()
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
        return self._fire(chosen, "sampled", ws)

    def _say(self, s):
        self.stdout.write(s + "\n")

    # -- commands ------------------------------------------------------

    def do_events(self, arg):
        """enabled transitions of the firing tier, numbered"""
        ws = self._candidates()
        if not ws:
            self._say(f"quiescent (stage {self.stage})"
                      + (f" — the env ticks on step/run (plugin "
                         f"{self.plugin_spec}); quiet, not dead"
                         if self.plugin is not None else ""))
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
            if self.plugin is not None:
                self._mark()
                name = self._tick()
                if name is not None:
                    self._say(f"quiet — the env ticked: {name}  [env]")
                    return
                self.undo_stack.pop()
                self._latch_done()
                self._say("quiescent, and the env's episode is over "
                          "(tick returned None) — nothing can fire")
                return
            self._latch_done()
            self._say("quiescent — nothing can fire")
            return
        self._mark()
        if not arg.strip():
            name = self._sample()
            self._say(f"fired (sampled): {name}")
            return
        pick, err = self._resolve(arg.strip(), ws)
        if pick is None:
            self.undo_stack.pop()
            self._say(err)
            return
        self._say(f"fired: {self._fire(pick, 'forced', ws)}")

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
        self._say(f"stage {self.stage} | {_fmt_counts(self.counts)}"
                  + (f"\n({self._pricing()})" if self.pricer else ""))

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
        return True

    def do_assert(self, arg):
        """assert ATOM [N] — add N (default 1) tokens"""
        self._adjust(arg, +1, "usage: assert ATOM [N]")

    def do_retract(self, arg):
        """retract ATOM [N] — remove N (default 1) tokens"""
        self._adjust(arg, -1, "usage: retract ATOM [N]")

    # -- interventions (spec 10) ----------------------------------------

    def do_do(self, arg):
        """do +ATOM [N] | -ATOM [N] | EVENT | ~EVENT — an intervention,
        recorded as one: state surgery (assert/retract), choice surgery
        (force an enabled EVENT — excluded from the likelihood), or a
        block (~ masks a ground event, or a rule's every instance, out
        of the tier; the measure renormalizes). Bare `do` lists them."""
        a = arg.strip()
        if not a:
            k0 = self._base[0]
            n_exo = sum(1 for h in self.fired_how if h["how"] == "exo")
            n_chosen = sum(1 for h in self.fired_how if h["how"] == "forced")
            rows = [f"  {d['k']:>3}  do {d['op']}" for d in self.ledger]
            self._say("\n".join(
                ["interventions this session (after occurrence k):"]
                + (rows or ["  (none)"])
                + ([f"chosen steps (step EVENT — conditioning, not "
                    f"interventions): {n_chosen} — 'log' marks them "
                    "[forced]"] if n_chosen else [])
                + [f"masks: {' '.join(sorted(self.masks)) or '(none)'}",
                   f"base #{k0}: " + (
                       "do " + ", ".join(self._base_do)
                       if self._base_do else "not an intervention")]
                + ([f"exogenous (env) rows: {n_exo} — 'log' marks "
                    "them [env]"] if n_exo else [])))
            return
        if a[0] in "+-":
            prev = list(self._base_do) \
                if self._base[0] == len(self.fired) else []
            if not self._adjust(a[1:], +1 if a[0] == "+" else -1,
                                "usage: do +ATOM [N] | do -ATOM [N]"):
                return
            op = a[0] + " ".join(a[1:].split())
            self._base_do = prev + [op]
            self.ledger.append({"k": len(self.fired), "op": op})
            self._say(f"do {op} — an intervention (state surgery): "
                      "the graph's base is now this edit")
            return
        if a[0] == "~":
            name = a[1:].strip()
            n = sum(1 for _, _, ev in self.events_tbl
                    if ev.name == name or ev.name.partition("[")[0]
                    == name or re.sub(r"-\d+$", "",
                                      ev.name.partition("[")[0]) == name)
            if not name or not n:
                self._say(f"no event or rule {name!r} — 'events' and "
                          "'list' name them")
                return
            self._mark()
            self.masks.add(name)
            self.ledger.append({"k": len(self.fired), "op": "~" + name})
            self._say(f"do ~{name} — blocked ({n} ground event"
                      f"{'' if n == 1 else 's'}); the tier renormalizes "
                      "without it until reset/back")
            return
        ws = self._candidates()
        if not ws:
            self._say("quiescent — nothing can be forced")
            return
        pick, err = self._resolve(a, ws)
        if pick is None:
            self._say(err)
            return
        self._mark()
        name = self._fire(pick, "do", ws)
        self.ledger.append({"k": len(self.fired), "op": name})
        self._say(f"do {name} — forced (an intervention: conditioning, "
                  "not sampling; excluded from the likelihood)")

    def _resolve(self, a, ws):
        """An enabled candidate by number or unique name prefix (the
        `step` rule) -> (index, None) | (None, why)."""
        if a.isdigit() and 1 <= int(a) <= len(ws):
            return ws[int(a) - 1][0], None
        hits = [i for i, _ in ws
                if self.events_tbl[i][2].name.startswith(a)]
        if len(hits) == 1:
            return hits[0], None
        if hits:
            return None, (f"ambiguous: {len(hits)} candidates match "
                          f"{a!r} — 'events' to list them")
        return None, (f"no enabled candidate {a!r} — 'events' lists "
                      f"1..{len(ws)}")

    def _snapshot_world(self):
        """The whole session world, deep-copied (the fork's shelf)."""
        import copy
        return copy.deepcopy({
            "stage": self.stage, "counts": self.counts,
            "fired": self.fired, "fired_rows": self.fired_rows,
            "fired_how": self.fired_how, "base": self._base,
            "base_do": self._base_do, "undo": self.undo_stack,
            "masks": self.masks, "ledger": self.ledger,
            "rng": self.rng, "envs": self._envs,
            "env": self._env_snap()})

    def do_fork(self, arg):
        """fork [K] — branch at occurrence K (default: here): the actual
        world is shelved, the session becomes exactly post-K by
        deterministic prefix replay, the graph marks the cut; `back`
        restores it. One shelved world at a time (D1)."""
        if self.shelf is not None:
            self._say(f"already forked at #{self.fork_at if self.fork_at is not None else '?'} "
                      "— 'back' first (one alternative world at a time; "
                      "parallel worlds are parallel sessions)")
            return
        k0, n = self._base[0], len(self.fired)
        a = arg.strip()
        try:
            k = int(a.lstrip("#")) if a else n
        except ValueError:
            self._say("usage: fork [K]")
            return
        if not k0 <= k <= n:
            self._say(f"fork K needs #{k0} <= K <= #{n}: before #{k0} "
                      "lies an edit no event explains (undo/reset to "
                      "reach it)")
            return
        self.shelf = self._snapshot_world()
        self._fork_to(k)
        self._say(f"forked at #{k} — the actual world (#{n}) is shelved; "
                  "'back' returns to it. The future from here is "
                  "re-randomized, not replayed.")
        self.do_state("")

    def _fork_to(self, k):
        """The session becomes exactly post-#k: deterministic prefix
        replay from the graph's base (the caller checked base <= k <=
        tip and shelved what it must)."""
        n = len(self.fired)
        k0, stage, counts = self._base
        counts = dict(counts)
        for kind, _, ev in self.fired_rows[k0:k]:
            for q, m in ev.consume.items():
                counts[q] = counts.get(q, 0) - m
            for q, m in ev.produce.items():
                counts[q] = counts.get(q, 0) + m
            if kind == "link":
                stage = ev.post
        self.counts = {q: m for q, m in counts.items() if m > 0}
        self.stage = stage if not (k == n and self.stage == DONE) \
            else self.stage
        del self.fired[k:]
        del self.fired_rows[k:]
        del self.fired_how[k:]
        self.ledger = [d for d in self.ledger if d["k"] <= k]
        self.undo_stack = []
        self.fork_at = k
        if self.plugin is not None:
            # the counterfactual sees the env AS IT WAS at #k
            del self._envs[k + 1:]
            self._env_to(self._envs[k])

    # the INSTRUMENTS' private shelf (spec 10 D1 amendment): whatif /
    # necessity / sufficiency fork and come back on their own, so they
    # work inside a user fork (mask-and-wander, then ask) — the
    # user-facing `fork` stays one-shelf.
    def _ifork(self, k):
        """-> None, or the refusal (K outside [base, tip])."""
        k0, n = self._base[0], len(self.fired)
        if not k0 <= k <= n:
            return (f"fork K needs #{k0} <= K <= #{n}: before #{k0} lies "
                    "an edit no event explains (undo/reset to reach it)")
        self._ishelf = (self._snapshot_world(), self.shelf, self.fork_at)
        self._fork_to(k)
        return None

    def _iback(self):
        w, shelf, fork_at = self.__dict__.pop("_ishelf")
        self.stage, self.counts = w["stage"], w["counts"]
        self.fired, self.fired_rows = w["fired"], w["fired_rows"]
        self.fired_how, self._base = w["fired_how"], w["base"]
        self._base_do, self.undo_stack = w["base_do"], w["undo"]
        self.ledger, self.rng = w["ledger"], w["rng"]
        self.masks = w["masks"]
        self.shelf, self.fork_at = shelf, fork_at
        self._envs = w["envs"]
        self._env_to(w["env"])

    def do_back(self, arg):
        """back — leave the fork: restore the shelved actual world
        exactly (masks set inside the fork are cleared with it)"""
        if self.shelf is None:
            self._say("not forked — nothing to go back to")
            return
        w, self.shelf, self.fork_at = self.shelf, None, None
        self.stage, self.counts = w["stage"], w["counts"]
        self.fired, self.fired_rows = w["fired"], w["fired_rows"]
        self.fired_how, self._base = w["fired_how"], w["base"]
        self._base_do, self.undo_stack = w["base_do"], w["undo"]
        self.ledger, self.rng = w["ledger"], w["rng"]
        self.masks = w["masks"]       # the fork's own masks are gone
        self._envs = w["envs"]
        self._env_to(w["env"])        # and the actual world's env
        self._say(f"back in the actual world (#{len(self.fired)})")
        self.do_state("")

    _WHATIF = re.compile(
        r"^\s*#?(?P<k>\d+)\s+(?P<dos>(?:do\(\s*[^()]+\)\s*)+)"
        r"(?P<mode>query|run)\s+(?P<rest>.+?)\s*$")

    def _quiet(self, line):
        """Run a command with its output captured (whatif's inner
        steps) -> the text it would have said."""
        keep, self.stdout = self.stdout, io.StringIO()
        try:
            self.onecmd(line)
            return self.stdout.getvalue()
        finally:
            self.stdout = keep

    def do_whatif(self, arg):
        """whatif K do(X) [do(Y)...] query ATOM [@N|@frame|N] [exact|
        sampled|full] | run N — fork at #K, apply the do's, snapshot,
        back (the session is untouched), then answer and compare with
        the actual world. query reads at @N cone steps by default (N =
        the actual remainder's cone events — spec 11 D1), exact on the
        cone below the budget, sampled ± ci above it; run is one
        sampled draw. do(+A [N]) do(-A) do(EVENT) do(~EVENT)."""
        m = self._WHATIF.match(arg)
        if not m:
            self._say("usage: whatif K do(X) [do(Y)...] query ATOM "
                      "[@N|@frame|N] [exact|sampled|full] | "
                      "whatif K do(X)... run N")
            return
        k = int(m["k"])
        dos = [d.strip() for d in
               re.findall(r"do\(\s*([^()]+?)\s*\)", m["dos"])]
        rest = m["rest"].split()
        n = len(self.fired)
        try:
            if m["mode"] == "query":
                if not rest:
                    raise ValueError
                atom = rest[0]
                ro, tier = parse_tail(rest[1:])
            else:
                if len(rest) != 1:
                    raise ValueError
                atom, horizon = None, int(rest[0])
        except ValueError:
            self._say("usage: whatif K do(X)... query ATOM [@N|@frame|N] "
                      "[exact|sampled|full] | run N")
            return
        if atom and atom.lstrip("!") not in self.atoms:
            self._say(f"note: {atom} appears in no rule of this catalog "
                      "(atoms are ground — 'state' lists them)")
        actual = {"counts": dict(self.counts), "n": n,
                  "rows": list(self.fired[k:]),
                  "how": [h["how"] for h in self.fired_how[k:]]}
        refused = self._ifork(k)
        if refused:
            self._say(refused)
            return
        try:
            if m["mode"] == "run":
                return self._whatif_run(k, dos, horizon, actual)
            got = self._apply_dos(k, dos)
        finally:
            self._iback()
        if isinstance(got, str):
            self._say(got)
            return
        base, cf, done = got
        self._whatif_query(k, dos, atom, ro, tier, base, cf, done)

    def _apply_dos(self, k, dos):
        """Inside the fork: (baseline arm, with-do arm, the do's done) or
        a refusal str. The arms are SNAPSHOTS (stage, counts, masks,
        forced) — measured after `back`, on data, not on the session."""
        base = self._snap()
        forced = []
        for d in dos:
            before = len(self.ledger)
            said = self._quiet("do " + d)
            if len(self.ledger) == before:
                return (f"whatif: do({d}) did not apply at #{k} — "
                        + said.strip())
            if d[0] not in "+-~":
                forced.append(self.fired[-1])
        done = [f"#{e['k']} do {e['op']}" if e["op"][0] not in "+-~"
                else f"do {e['op']}"
                for e in self.ledger if e["k"] >= k][-len(dos):]
        return base, self._snap(forced), done

    def _snap(self, forced=()):
        return Arm(self.stage, dict(self.counts), frozenset(self.masks),
                   len(forced), tuple(forced), self._env_snap())

    def _remainder_readout(self, f, default=None):
        """D1: @N with N = the actual remainder's (events after #f) cone
        events — a function of the question's cone, resolved inside
        `measure`."""
        rest = [ev.name for _, _, ev in self.fired_rows[f:]]

        def ro(names):
            n = sum(1 for x in rest if x in names)
            return ("cone", n if rest else (default or
                                            self.case["steps"] or 1))
        return ro

    def _fact_at(self, f, ans, outcome):
        """The actual world at the answer's readout, from fork point #f:
        @N — after the N-th cone event of the remainder (out-of-cone
        events cannot move a cone atom); plain N — after #f+N; @frame —
        the tip. -> (position, counts)."""
        ro = ans.readouts[0]
        n = len(self.fired)
        if ro[0] == "global":
            j = min(n, f + ro[1])
        elif ro[0] == "cone":
            names = cone_names(self.prog, ans.preps[0].cone)
            j, seen = f, 0
            for idx in range(f, n):
                if seen >= ro[1]:
                    break
                if self.fired_rows[idx][2].name in names:
                    seen += 1
                j = idx + 1
            if seen < ro[1]:
                j = n
        else:
            j = n
        return j, self._counts_at(j)

    def _measure(self, arms, outcome, ro, tier, line=""):
        """measure() with the session's cone cache; raises
        StateBudgetExceeded (a forced exact over budget) or NotSettled
        (an @frame with no answer) — the callers say either."""
        cache = self.__dict__.setdefault("_cone_cache", {})
        if cache.get("_key") != self.key:      # a new world: retire
            cache.clear()
            cache["_key"] = self.key
        if self.pricer is None:
            ans = measure(self.prog, arms, outcome, ro, tier,
                          STATE_BUDGET, cache)
        else:
            # each arm prices under ITS env snapshot (a counterfactual
            # arm sees the forked env); the live env comes back after
            keep = self._env_snap()
            try:
                ans = measure(self.prog, arms, outcome, ro, tier,
                              STATE_BUDGET, cache,
                              price=[_EnvPricer(self, a.env)
                                     for a in arms])
            finally:
                self._env_to(keep)
                self.__dict__.pop("_env_live", None)
        if ans.refusal is not None and line:
            # spec 08 step 3 stands: the exact tier's refusal is a
            # runtime note even when the sampled tier then answered
            from metis.kernel.diagnostics import budget_note
            _note(self, budget_note(ans.refusal, line.split()[0], line,
                                    line + "  (answered by the sampled "
                                    "tier)"))
        return ans

    def _exact_refused(self, e, line):
        """`exact`/`full` forced and over budget: say it (budget
        doctrine) — the auto tier would have sampled instead."""
        from metis.kernel.diagnostics import budget_note
        _note(self, budget_note(e, "query", line,
                                line.replace(" exact", "")
                                .replace(" full", "")))
        self._say(f"{e} — `exact` was forced; drop it to let the "
                  "sampled tier answer (± ci)")

    def _whatif_query(self, k, dos, atom, ro, tier, base, cf, done):
        names = " ".join(f"do({d})" for d in dos)
        try:
            ans = self._measure([base, cf], atom,
                                ro or self._remainder_readout(k), tier,
                                f"whatif {k} {names} query {atom}")
        except StateBudgetExceeded as e:
            return self._exact_refused(
                e, f"whatif {k} {names} query {atom} {tier}")
        except NotSettled as e:
            self._say(str(e))
            return
        at, fc = self._fact_at(k, ans, atom)
        pb, pc = ans.ps
        cb, cc = ans.cis
        y = self._y(atom)
        vals = [self._fact(fc, atom), f"P = {pval(pc, cc)}",
                f"P = {pval(pb, cb)}"]
        w = max(len(v) for v in vals) + 3
        forced = len(cf.spent_names)
        rows = [
            f"WHATIF at #{k}  {names}  ·  query {atom}  ·  "
            f"{readout_text(ans.readouts[0])} after #{k}",
            f"  in fact      {vals[0]:<{w}}the actual world, #{at}",
            f"  with do      {vals[1]:<{w}}counterfactual"
            + (f" ({forced} forced)" if forced else "")
            + (f" — {structural(ans.preps[1], atom)}"
               if structural(ans.preps[1], atom) else ""),
            f"  by chance    {vals[2]:<{w}}no intervention: same fork, "
            "same readout"
            + (f" — {structural(ans.preps[0], atom)}"
               if structural(ans.preps[0], atom) else ""),
            "  interventions: " + "; ".join(done)]
        delta = pc - pb
        tol = max(1e-12, (cb or 0) + (cc or 0))
        trend = ("raises" if delta > tol else "lowers"
                 if delta < -tol else "leaves")
        how = (f"{trend} P({y}) from {pval(pb, None)} to {pval(pc, None)}"
               if trend != "leaves" else
               f"leaves P({y}) at {pval(pc, None)}"
               + (" (within the sampling error)" if cb else ""))
        fact = self._holds(fc, atom)
        rows.append(f"  ⇒ had {names} at #{k}, P({y}) = {pval(pc, cc)}; "
                    f"by chance from there {pval(pb, cb)}; in fact {y} "
                    f"{'holds' if fact else 'does not hold'}. "
                    f"The intervention {how}.")
        self._say("\n".join(rows + footer(ans, ["chance", "with do"])))

    def _whatif_run(self, k, dos, horizon, actual):
        names = " ".join(f"do({d})" for d in dos)
        got = self._apply_dos(k, dos)
        if isinstance(got, str):
            self._say(got)
            return
        _, _, done = got
        self._quiet(f"run {horizon}")
        rows = list(self.fired[k:])
        how = [h["how"] for h in self.fired_how[k:]]
        self._say("\n".join(self._whatif_run_report(
            k, names, horizon, actual, rows, how, dict(self.counts),
            done)))

    @staticmethod
    def _p(x):
        """A probability for the reports: float noise within 1e-12 of 0
        or 1 prints as the exact value (no "PN ≈ 1.11e-16")."""
        return pval(x, None)

    def _whatif_run_report(self, k, names, n_run, actual, rows, how,
                           counts, done):
        tag = {"forced": " [forced]", "do": " [do]"}
        left = [f"{k + 1 + j}. {e}{tag.get(h, '')}"
                for j, (e, h) in enumerate(zip(actual["rows"],
                                               actual["how"]))]
        right = [f"{k + 1 + j}. {e}{tag.get(h, '')}"
                 for j, (e, h) in enumerate(zip(rows, how))]
        head = "in fact (the actual world)"
        w = max([len(x) for x in left] + [len(head)])
        out = [f"WHATIF at #{k}  {names}  ·  run {n_run} (ONE sampled "
               "draw, not a measure — 'query' gives the probability)",
               f"  {head:<{w}}   with do"]
        for j in range(max(len(left), len(right))):
            a = left[j] if j < len(left) else ""
            b = right[j] if j < len(right) else ""
            out.append(f"  {a:<{w}}   {b}")
        fa, fb = actual["counts"], counts
        gained = sorted(a for a in fb if fb[a] > fa.get(a, 0))
        lost = sorted(a for a in fa if fa[a] > fb.get(a, 0))
        out.append("  final  " + ("differs: "
                                  + " ".join([f"+{a}" for a in gained]
                                             + [f"-{a}" for a in lost])
                                  if gained or lost else "the same state"))
        out.append("  interventions: " + "; ".join(done))
        return out

    # -- probabilities of causation (spec 10 P4; spec 11 tiers) ---------

    def _cause_args(self, verb, arg, need_atom):
        """K [ATOM] [@N|@frame|N] [exact|sampled|full] -> (k, name, atom,
        readout|None, tier) or an error str. K = an occurrence of the
        actual trace; the fork is at K-1 (in [base, tip))."""
        parts = arg.split()
        usage = (f"usage: {verb} K {'ATOM' if need_atom else '[ATOM]'} "
                 "[@N|@frame|N] [exact|sampled|full]")
        if not parts or not parts[0].lstrip("#").isdigit():
            return usage
        k, n, k0 = int(parts[0].lstrip("#")), len(self.fired), \
            self._base[0]
        if not k0 < k <= n:
            return (f"{verb} K needs an event of the graph: "
                    f"#{k0 + 1}..#{n} (the fork is at K-1)")
        rest = parts[1:]
        atom = None
        if rest and not (rest[0].startswith("@") or rest[0].isdigit()
                         or rest[0] in ("exact", "sampled", "full")):
            atom, rest = rest[0], rest[1:]
        try:
            ro, tier = parse_tail(rest)
        except ValueError:
            return usage
        if atom is None and need_atom:
            return usage
        if atom is None:
            cands = self._forward_cone_atoms(k)
            if len(cands) != 1:
                return (f"{verb} {k} needs an ATOM — present atoms in "
                        f"#{k}'s forward cone: "
                        + (" ".join(cands) if cands else "(none)"))
            atom = cands[0]
        return k, self.fired[k - 1], atom, ro, tier

    def _counts_at(self, j):
        """The actual world's multiset after occurrence j (deterministic
        replay from the graph base; j clipped to the tip) — the "in
        fact" at a readout short of the actual tip."""
        k0, _, counts = self._base
        if j >= len(self.fired):
            return dict(self.counts)
        c = dict(counts)
        for _, _, ev in self.fired_rows[k0:max(k0, j)]:
            for q, m in ev.consume.items():
                c[q] = c.get(q, 0) - m
            for q, m in ev.produce.items():
                c[q] = c.get(q, 0) + m
        return {q: m for q, m in c.items() if m > 0}

    def _forward_cone_atoms(self, k):
        """Present atoms whose last producer's causal cone contains #k
        (spec 09 cones: what #k is an ancestor of, now)."""
        from metis.kernel.nets import cone
        g, _ = self.causal()
        k0 = self._base[0]
        last = {}
        for j, (_, _, ev) in enumerate(self.fired_rows[k0:], k0 + 1):
            for q in ev.produce:
                last[q] = j
        return sorted(q for q, m in self.counts.items()
                      if m > 0 and q in last and k in cone(g, [last[q]]))

    def _cause_arms(self, k, op):
        """Fork at k-1: (baseline arm, the arm after `do op`) or a
        refusal str; the session comes back untouched."""
        refused = self._ifork(k - 1)
        if refused:
            return refused
        try:
            base = self._snap()
            before = len(self.ledger)
            said = self._quiet("do " + op)
            if len(self.ledger) == before:
                return said.strip()
            forced = () if op[0] == "~" else (self.fired[-1],)
            return base, self._snap(forced)
        finally:
            self._iback()

    def do_necessity(self, arg):
        """necessity K [ATOM] [@N|@frame|N] [exact|sampled|full] — PN:
        the actual trace fired #K and ATOM holds at the readout; fork at
        K-1, block #K's ground event, read P(ATOM | do(not K)) at the
        same readout (default @N cone steps, N = the remainder's cone
        events), beside the fact and chance. Exact on the cone when it
        fits the budget, else sampled ± ci (spec 11). Without ATOM: the
        present atoms in #K's forward cone (used if exactly one)."""
        got = self._cause_args("necessity", arg, False)
        if isinstance(got, str):
            self._say(got)
            return
        k, ev, atom, ro, tier = got
        arms = self._cause_arms(k, "~" + ev)
        if isinstance(arms, str):
            self._say(f"necessity: cannot block #{k} at #{k - 1}: {arms}")
            return
        try:
            ans = self._measure(list(arms), atom,
                                ro or self._remainder_readout(k - 1), tier,
                                f"necessity {arg.strip()}")
        except StateBudgetExceeded as e:
            return self._exact_refused(e, f"necessity {arg.strip()}")
        except NotSettled as e:
            self._say(str(e))
            return
        at, fc = self._fact_at(k - 1, ans, atom)
        y = self._y(atom)
        if not self._holds(fc, atom):
            self._say(f"{y} does not hold in fact at #{at} (the readout "
                      f"{readout_text(ans.readouts[0])}) — necessity asks "
                      "about an outcome that happened ('sufficiency' "
                      "asks the other way)")
            return
        (pb, pc), (cb, cc) = ans.ps, ans.cis
        pn = 1.0 - pc
        hi = pn + (cc or 0)
        lo = pn - (cc or 0)
        # PN alone cannot tell necessity from the outcome's own
        # uncertainty at the readout: when blocking #K does not LOWER
        # P(Y) below chance, nothing of Y is attributable to #K (the
        # Tian-Pearl lower bound (P(y) - P(y|do not-x)) / P(x,y) is 0) —
        # and if blocking RAISES it, #K worked against Y
        tol = max(1e-12, (cb or 0) + (cc or 0))
        inhibits = pc > pb + tol
        word = ("not necessary" if inhibits or pc >= pb - tol
                else "necessary" if lo >= 0.95 else "not necessary"
                if hi <= 0.05 else "partly necessary")
        vals = [self._fact(fc, atom), f"P = {pval(pc, cc)}",
                f"P = {pval(pb, cb)}"]
        w = max(len(v) for v in vals) + 3
        lw = max(len(f"without #{k}"), 11) + 2
        self._say("\n".join([
            f"NECESSITY of #{k} {ev} for {y}  ·  "
            f"{readout_text(ans.readouts[0])} after #{k - 1}",
            f"  {'in fact':<{lw}}{vals[0]:<{w}}#{k} fired; the actual "
            f"world, #{at}",
            f"  {'without #' + str(k):<{lw}}{vals[1]:<{w}}fork at "
            f"#{k - 1}, do ~{ev} (blocked through the readout)"
            + (f" — {structural(ans.preps[1], atom)}"
               if structural(ans.preps[1], atom) else ""),
            f"  {'by chance':<{lw}}{vals[2]:<{w}}from #{k - 1}, nothing "
            "blocked"
            + (f" — {structural(ans.preps[0], atom)}"
               if structural(ans.preps[0], atom) else ""),
            f"  ⇒ PN ≈ {pval(pn, cc)}: #{k} {ev} was {word} for {y} — "
            f"had it not fired, {y} would fail with probability "
            f"{pval(pn, cc)} (it holds in fact; by chance from #{k - 1}: "
            f"{pval(pb, cb)})."
            + (f" Blocking #{k} RAISES P({y}) from {pval(pb, cb)} to "
               f"{pval(pc, cc)}: #{k} worked AGAINST {y} here — the PN "
               f"is {y}'s own uncertainty at the readout, not necessity."
               if inhibits else
               f" Blocking #{k} leaves P({y}) at chance: the PN is "
               f"{y}'s own uncertainty at the readout, not necessity."
               if pc >= pb - tol and pn > tol else ""),
            "  (interventional, prefix-fixed: the past to "
            f"#{k - 1} is the actual one, the future re-randomized; "
            "`why` gives the structural cone beside it)"]
            + footer(ans, ["chance", "without"])))

    def do_sufficiency(self, arg):
        """sufficiency K ATOM [@N|@frame|N] [exact|sampled|full] — the
        INTERVENTIONAL PROXY for PS (D3): fork at K-1, force #K, P(ATOM)
        beside the unforced baseline at the same readout. Exact PS needs
        the posterior over the worlds where K did not fire and ATOM did
        not hold — deferred."""
        got = self._cause_args("sufficiency", arg, True)
        if isinstance(got, str):
            self._say(got)
            return
        k, ev, atom, ro, tier = got
        arms = self._cause_arms(k, ev)
        if isinstance(arms, str):
            self._say(f"sufficiency: cannot force #{k} at #{k - 1}: "
                      f"{arms}")
            return
        try:
            ans = self._measure(list(arms), atom,
                                ro or self._remainder_readout(k - 1), tier,
                                f"sufficiency {arg.strip()}")
        except StateBudgetExceeded as e:
            return self._exact_refused(e, f"sufficiency {arg.strip()}")
        except NotSettled as e:
            self._say(str(e))
            return
        at, fc = self._fact_at(k - 1, ans, atom)
        (pb, pc), (cb, cc) = ans.ps, ans.cis
        y = self._y(atom)
        vals = [self._fact(fc, atom), f"P = {pval(pc, cc)}",
                f"P = {pval(pb, cb)}"]
        w = max(len(v) for v in vals) + 3
        lw = max(len(f"forced #{k}"), 11) + 2
        lift = pc - pb
        self._say("\n".join([
            f"SUFFICIENCY (interventional PROXY, not Pearl's PS) of #{k} "
            f"{ev} for {y}  ·  {readout_text(ans.readouts[0])} after "
            f"#{k - 1}",
            f"  {'in fact':<{lw}}{vals[0]:<{w}}the actual world, #{at}",
            f"  {'forced #' + str(k):<{lw}}{vals[1]:<{w}}fork at "
            f"#{k - 1}, do {ev}",
            f"  {'by chance':<{lw}}{vals[2]:<{w}}from #{k - 1}, unforced",
            f"  ⇒ forcing #{k} {ev} at #{k - 1} gives P({y}) = "
            f"{pval(pc, cc)} against {pval(pb, cb)} by chance "
            f"({'+' if lift >= 0 else '−'}{pval(abs(lift), None)}).",
            "  PROXY: true PS = P(ATOM | do(K), in the worlds where K did "
            "not fire and ATOM did not hold) needs the posterior over "
            "those ¬K worlds — exact PS is deferred (spec 10 D3)."]
            + footer(ans, ["chance", "forced"])))

    def do_query(self, arg):
        """query ATOM [@N|@frame|N] [exact|sampled|full] — the outcome's
        probability from here (masks applied): default @N cone steps
        with N = #steps; exact on the cone below the budget, else
        sampled ± ci (spec 11); a plain N reads N global events
        (scheduling-sensitive, validation)."""
        parts = arg.split()
        if not parts:
            self._say("usage: query ATOM [@N|@frame|N] "
                      "[exact|sampled|full]")
            return
        atom = parts[0]
        try:
            ro, tier = parse_tail(parts[1:])
        except ValueError:
            self._say("usage: query ATOM [@N|@frame|N] "
                      "[exact|sampled|full]")
            return
        if self.stage == DONE:
            p = float(self._holds(self.counts, atom))
            self._say(f"P[{atom} | done] = {p:g}")
            return
        try:
            ans = self._measure([self._snap()], atom,
                                ro or ("cone", self.case["steps"] or 1),
                                tier, f"query {arg.strip()}")
        except StateBudgetExceeded as e:
            return self._exact_refused(e, f"query {arg.strip()}")
        except NotSettled as e:
            self._say(str(e))
            return
        except Exception as e:
            if getattr(e, "diagnostic", None) is not None:
                raise                 # a structured finding (G5): onecmd
            self._say(f"filter failed: {e!r}")
            return
        under = (f" | do ~{', ~'.join(sorted(self.masks))}"
                 if self.masks else "")
        proof = structural(ans.preps[0], atom)
        self._say("\n".join(
            [f"P[{self._y(atom)} at {readout_text(ans.readouts[0])}"
             f"{under}] = {pval(ans.ps[0], ans.cis[0])}"
             + (f"  ({proof})" if proof else "")]
            + footer(ans, ["here"])))

    def _here_prog(self):
        """The program from the session's state, the active masks
        applied (spec 10 P2: a block prunes the tier, the measure
        renormalizes — the same object for the exact tier)."""
        prog = replace(self.prog, init_stage=self.stage,
                       init=dict(self.counts))
        if self.masks:
            prog = replace(
                prog,
                stages={st: [c for c in cs if not self._masked(c.name)]
                        for st, cs in prog.stages.items()},
                links=[lk for lk in prog.links
                       if not self._masked(lk.name)])
        return prog

    # an OUTCOME is an atom (it holds when present) or `!atom` (it
    # holds when absent — "the wolf is spent"): spec 10's measures
    # answer for either.
    @staticmethod
    def _holds(counts, y):
        return (counts.get(y.lstrip("!"), 0) > 0) != y.startswith("!")

    @staticmethod
    def _y(y):
        return "¬" + y[1:] if y.startswith("!") else y

    @staticmethod
    def _fact(counts, y):
        a = y.lstrip("!")
        return f"{a} {'PRESENT' if counts.get(a, 0) > 0 else 'absent'}"

    def _presence(self, atom, steps):
        """P[atom present after `steps`] from here, exactly (raises
        StateBudgetExceeded on a blowup; DONE = presence now); `!atom`
        = P[absent]."""
        if atom.startswith("!"):
            return 1.0 - self._presence(atom[1:], steps)
        if self.stage == DONE or steps <= 0:
            return float(self.counts.get(atom, 0) > 0)
        prog = self._here_prog()
        pr = ForwardProjection(prog, steps, max_states=STATE_BUDGET,
                               keep_history=False,
                               policy=self.pricer.policy(prog)
                               if self.pricer else None)
        return sum(q for (_, k), q in pr.final.items()
                   if any(a == atom and n > 0 for a, n in k))

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
        (self.stage, self.counts, k, self._base, masks, nl, base_do,
         self.fork_at, env) = self.undo_stack.pop()
        del self.fired[k:]
        del self.fired_rows[k:]
        del self.fired_how[k:]
        if self.plugin is not None and env is not None:
            del self._envs[k + 1:]
            self._envs[k] = env
            self._env_to(env)
        self.masks, self._base_do = set(masks), list(base_do)
        del self.ledger[nl:]
        self.do_state("")

    def do_reset(self, arg):
        """back to the file's #init directives"""
        self._mark()
        self.stage = self.prog.init_stage
        self.counts = dict(self.case["init"])
        self.masks = set()
        self._rebase()
        if self.plugin is not None:   # the env returns to as attached
            self._env_to(self._env0)
            self._envs[-1] = self._env0
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
        status = ("done" if self.stage == DONE
                  or not cands and self.plugin is None
                  else "quiescent" if kinds == {"link"} or not cands
                  else "running")
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
            # spec 10: interventions, masks, the shelf, the honest
            # likelihood (sampled steps only)
            **self._ctx_interventions(),
        }

    def _ctx_interventions(self):
        lik, ns, nf = self.likelihood()
        under = sorted({m for h in self.fired_how
                        if h.get("how") == "sampled"
                        for m in h.get("masks", ())})
        # a chosen step (`step EVENT`, how=forced) is conditioning but not
        # an intervention: `do` lists the ledger only, so the band counts
        # the two apart (specialist step 2a: '1 forced/do' beside a `do`
        # that listed none)
        chosen = sum(1 for h in self.fired_how if h["how"] == "forced")
        return {"likelihood": {"value": lik, "sampled": ns,
                               "forced": nf, "chosen": chosen,
                               "do": nf - chosen, "under_masks": under},
                "masks": sorted(self.masks),
                "base_do": list(self._base_do),
                "fork": ({"at": self.fork_at,
                          "actual": len(self.shelf["fired"])}
                         if self.shelf is not None else None)}

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
        lk, fk = c["likelihood"], c["fork"]
        if lk["forced"] or c["masks"] or c["base_do"] or fk:
            out.append("DO      " + "  ".join(x for x in (
                ", ".join(x for x in (
                    f"{lk['chosen']} chosen step"
                    f"{'' if lk['chosen'] == 1 else 's'}"
                    if lk["chosen"] else "",
                    f"{lk['do']} do" if lk["do"] else "") if x)
                + " (out of the likelihood)" if lk["forced"] else "",
                "base do " + ", ".join(c["base_do"]) if c["base_do"]
                else "",
                "blocked " + " ".join(c["masks"]) if c["masks"] else "",
                f"FORKED at #{fk['at']} (actual #{fk['actual']} shelved"
                " — back)" if fk else "") if x))
        if self.pricer is not None:
            out.append("PRICED  " + self._pricing())
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
        if self.__dict__.get("_tlog") is not None:
            self.__dict__.setdefault("_theredocs", []).append(
                "".join(lines))
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
        LIVE (new key/events; counts and stage carry over). With no
        candidate admitted and a notch fit pending, ships the fit as an
        overlay (`notch accept`)."""
        rep = self.loop.report
        if self._notch is not None and not (rep and rep.ok):
            return self._notch_accept()
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
        """tell this session's fired events since the last reset/edit
        (or 'narrate SEED' for a fresh trace sampled with that seed —
        a seed, not a number of beats) as prose: Foata beats + final
        state + the catalog's %% docs through the configured llm
        endpoint — the loop's narrate-out boundary. `narrate system`
        prints the doctrine prompt, `narrate user [SEED]` the
        certified material, neither calls the model. The catalog's %%
        conventions (metis/loop/narrate.py): [fold] / [fold P] rows
        are machinery, folded; `a | b` gives a ⊕ rule's branches their
        phrases; `% evidence:` stays out. A narration the endpoint cut
        at its cap ends on its last sentence and says so."""
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
            rows, stage, kappa = self._seeded_trace(seed)
            hows, bdo = None, ()          # a fresh sampled trace
        else:
            seed = 0
            rows = list(self.fired_rows[self._base[0]:])   # since base
            hows = [h["how"] for h in self.fired_how[self._base[0]:]]
            stage = self.stage
            kappa = [(q, n) for q, n in sorted(self.counts.items())
                     if n > 0]
            bdo = tuple(self._base_do)
        if not rows:
            self._say("nothing to narrate — step/run first, or "
                      "'narrate SEED' for a sampled trace")
            return
        if show == "user":
            self._say(user_prompt(rows, stage, kappa, self.text,
                                  seed=seed, marks=hows, base_do=bdo)
                      + (f"\n({self._pricing()})" if self.pricer else ""))
            return
        try:
            text = narrate(self.llm or "ollama", rows, stage, kappa,
                           self.text, seed=seed,
                           system=self.sys_prompts.get("narrate"),
                           marks=hows, base_do=bdo)
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
        lines = [f"#plugin {self.plugin_spec}"] if self.plugin_spec \
            else []
        lines += [f"#notch {f}" for f in self.case.get("notch", [])]
        lines += [f"#init {a} {n}"
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
        # the ACCEPTED base (a revision replaced the served file's
        # world — a blank world's file is not what was born)
        body = [l for l in self._base_src().splitlines()
                if not l.lstrip().startswith("#")]
        while body and not body[-1].strip():
            body.pop()
        Path(path).write_text("\n".join(body + block) + "\n")
        packs = len(self.loop.accepted)
        self._say(f"wrote {path} (catalog + {len(block)} directives) "
                  f"— replay: metisc --run {path}"
                  + (f" — NOTE: {packs} accepted pack(s) are not in it; "
                     "`export DIR` writes base + packs" if packs else ""))

    def do_log(self, arg):
        """fired events this session; forced/do rows and the state do's
        are marked, and the likelihood counts sampled steps only"""
        if not self.fired and not self.ledger:
            self._say("(nothing fired yet)")
            return
        tag = {"forced": "  [forced]", "do": "  [do]", "exo": "  [env]"}
        state = {}
        for d in self.ledger:
            if d["op"][0] in "+-~":
                state.setdefault(d["k"], []).append(d["op"])
        out = []
        for k in range(len(self.fired) + 1):
            for op in state.get(k, []):
                out.append(f"     do {op}")
            if k == self.fork_at and self.shelf is not None:
                out.append(f"     ── fork at #{k} (actual world shelved:"
                           f" #{len(self.shelf['fired'])}) ──")
            if k < len(self.fired):
                h = self.fired_how[k] if k < len(self.fired_how) \
                    else {"how": "sampled"}
                out.append(f"  {k + 1}. {self.fired[k]}"
                           + tag.get(h["how"], ""))
        lik, ns, nf = self.likelihood()
        n_exo = sum(1 for h in self.fired_how if h["how"] == "exo")
        # the masks that renormalized the sampled tiers: the number is
        # the INTERVENED measure's, so say under which interventions
        under = sorted({m for h in self.fired_how
                        if h.get("how") == "sampled"
                        for m in h.get("masks", ())})
        out.append(f"likelihood {lik:.6g} over {ns} sampled step"
                   f"{'' if ns == 1 else 's'}"
                   + (f" under do ~{', ~'.join(under)}" if under else "")
                   + (f" ({nf} forced/do excluded — conditioning, "
                      "not sampling)" if nf else "")
                   + (f" ({n_exo} env row{'' if n_exo == 1 else 's'} — "
                      "exogenous input, not chance)" if n_exo else ""))
        self._say("\n".join(out))

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
        mk, bd = self.marks(), self._base_do
        if not a:
            self._say("\n".join(beats_text(g, title, mk, bd)))
            return
        if a.endswith(".dot"):
            text = dot(g, title, mk, bd)
        elif a.endswith(".json"):
            text = as_json(g, self.stage, self.key, mk, bd)
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

    # -- port plugins (PLUGIN-PORTS.md) — terminal only (D4) ----------

    def _terminal_only(self, verb):
        if getattr(self, "terminal", False):
            return False
        self._say(f"{verb} is terminal-only: this session is not the "
                  "interactive `metis` shell (D4 — the web/shared console "
                  "never imports operator code)")
        return True

    def do_plugin(self, arg):
        """plugin [SPEC|off] — show the attached port plugin, attach
        one (pkg.mod:factory | file.py:factory; re-imports it, fresh
        env), or detach it. Refused when the catalog's ports would be
        left unbound. TERMINAL ONLY (D4)."""
        if self._terminal_only("plugin"):
            return
        a = arg.strip()
        if not a:
            if self.plugin is None:
                self._say("no plugin attached"
                          + (" — the catalog declares weight ports; "
                             "they are unbound" if self.doc.ports.weights
                             else ""))
                return
            beat = getattr(self.plugin, "beat_stage", None)
            self._say(f"plugin {self.plugin_spec} "
                      f"({type(self.plugin).__name__}); ticks at "
                      "quiescence" + (f" and on entry to stage {beat}"
                                      if beat else "")
                      + f"; {len(self._envs)} env snapshot(s) kept; "
                      f"{len(self._overlays)} overlay(s) shipped")
            return
        pin = "" if a == "off" else a
        old = (self._plugin_pin, self.plugin_spec)
        self._plugin_pin = pin
        if pin and pin == self.plugin_spec:
            self.plugin_spec = ""          # same spec: re-import, fresh
        try:
            self._bind(self.doc, self.case)
        except Exception as e:
            self._plugin_pin = old[0]
            self.plugin_spec = old[1]
            findings = getattr(e, "findings", None) or [str(e)]
            self._say("plugin not changed:\n  " + "\n  ".join(findings))
            return
        self._overlays, self._overlay_names, self._overlay_srcs = [], [], []
        self._ship_pending_notch()
        self._notch = None
        for w in self._port_warns:
            self._say(f"warning: {w}")
        self._say(f"plugin {self.plugin_spec} attached — env snapshot "
                  "taken; reset returns to it" if self.plugin is not None
                  else "plugin detached")

    def do_ports(self, arg):
        """ports — the declared weight ports: bound or not, the
        overlay depth, the declared read scope, the rules referencing
        them"""
        ports = self.doc.ports
        if not ports.weights:
            self._say("no weight ports declared")
            return
        bound = set(self.registry.weight_bindings()) if self.registry \
            else set()
        users = {}
        for rule, ref in sorted(ports.weight_refs.items()):
            users.setdefault(ref.factor, []).append(
                ("~" if ref.complement else "") + rule)
        out = []
        for f, params in sorted(ports.weights.items()):
            head = f + (f"({','.join(params)})" if params else "")
            depth = len(self.registry._overlays.get(f, ())) \
                if self.registry else 0
            reads = ports.reads.get(f)
            scope = ("{" + ", ".join(_read_text(r) for r in reads) + "}"
                     if reads is not None else "(opaque — no reads)")
            out.append(f"  {head:<18} {'bound' if f in bound else 'UNBOUND'}"
                       + (f" +{depth} overlay(s)" if depth else "")
                       + f"  reads {scope}  ·  @w by "
                       + (" ".join(users.get(f, [])) or "(no rule)"))
        self._say("\n".join(
            [f"weight ports ({self.plugin_spec or 'no plugin'}):"] + out))

    def do_port(self, arg):
        """port F(ARGS) — evaluate one binding against the current
        counts, through its declared scope (a debug probe; an
        out-of-scope read raises, as in every tier)"""
        from metis.lang.compiler import ground_reads
        m = re.match(r"^\s*([\w-]+)\s*(?:\(([^()]*)\))?\s*$", arg)
        if not m:
            self._say("usage: port F(ARGS)")
            return
        f = m[1]
        args = tuple(x.strip() for x in (m[2] or "").split(",")
                     if x.strip())
        if f not in self.doc.ports.weights:
            self._say(f"no weight port {f!r} — 'ports' lists them")
            return
        binds = self.registry.weight_bindings() if self.registry else {}
        if f not in binds:
            self._say(f"port {f} is UNBOUND")
            return
        want = len(self.doc.ports.weights[f])
        if len(args) != want:
            self._say(f"port {f} takes {want} argument(s)")
            return
        counts = dict(self.counts)
        scope = None
        if f in self.doc.ports.reads:
            from metis.kernel.factorize import ScopedCounts
            scope = ground_reads(self.doc.ports, f, args,
                                 self.doc.catalog.horn_db())
            counts = ScopedCounts(frozenset(scope),
                                  {a: n for a, n in self.counts.items()
                                   if a in scope})
        v = binds[f](args, counts)
        self._say(f"{f}({','.join(args)}) = {v:g}   (scope: "
                  + (" ".join(f"{a}={self.counts.get(a, 0)}"
                              for a in scope) if scope else
                     "{}" if scope is not None else "opaque")
                  + (f"; plugin {self.plugin_spec}" if self.plugin else "")
                  + ")")

    def do_tick(self, arg):
        """tick [N] — advance the env N beats by hand: each beat's
        do-lines join the trace as one [env] row (exogenous input:
        conditioning, out of the likelihood)"""
        if self.plugin is None:
            self._say("no plugin attached — nothing ticks")
            return
        try:
            n = int(arg.strip() or "1")
        except ValueError:
            self._say("usage: tick [N]")
            return
        self._mark()
        names = []
        for _ in range(n):
            name = self._tick()
            if name is None:
                names.append("(env episode over)")
                break
            names.append(name)
        self._say("  ".join(names))
        self._say(f"stage {self.stage} | {_fmt_counts(self.counts)}")

    def do_notch(self, arg):
        """notch [ALPHA] — fit the weight ports from this session's
        SAMPLED steps (forced/do/env rows are conditioning, never
        evidence): per (port, args, read-scope row), a Bernoulli on the
        complement pair, prior = the mean value the base predicted,
        posterior mean (alpha*p0 + s)/(alpha + n). Prints the row
        sheet; `notch accept` (or `accept` with no candidate pending)
        ships it as an OVERLAY on the registry — never a catalog edit;
        `notch load FILE` ships a fitted overlay file the same way (the
        recorder's --overlay format, e.g. <stem>.notch.json beside the
        catalog); `notch pop` drops the newest overlay."""
        a = arg.strip()
        if a == "accept":
            return self._notch_accept()
        if a == "load" or a.startswith("load "):
            return self._notch_load(a[4:].strip())
        if a == "pop":
            if not self._overlays:
                self._say("no notch overlay shipped")
                return
            for f in self._overlays.pop():
                self.registry.pop_overlay(f)
            if self._overlay_names:
                self._overlay_names.pop()
            if self._overlay_srcs:
                self._overlay_srcs.pop()
            self._rebuild_pricer()
            self._say(f"overlay dropped ({len(self._overlays)} left)")
            return
        if self.pricer is None:
            self._say("no weight ports to notch")
            return
        try:
            alpha = float(a) if a else 10.0
        except ValueError:
            self._say("usage: notch [ALPHA] | notch accept | notch load FILE | notch pop")
            return
        fit = notch_rows(self.fired_how, alpha, self.doc.ports)
        if not fit["rows"]:
            self._say("nothing to fit — no SAMPLED step chose a paired "
                      "priced event this session (step/run first)"
                      + "".join(f"\n  refused {r}" for r in fit["refused"]))
            return
        self._notch = fit
        self._say(fit["sheet"].rstrip("\n"))
        self._say(f"({fit['n']} sampled exposure(s); "
                  + (f"refused: {'; '.join(fit['refused'])}; "
                     if fit["refused"] else "")
                  + "`notch accept` ships it as an overlay)")

    def _notch_accept(self):
        fit = self._notch
        if fit is None:
            self._say("no notch fit pending — `notch` first")
            return
        shipped = []
        for f, table in sorted(fit["tables"].items()):
            def ov(args, counts, _t=table, _f=f):
                row = tuple(sorted((x, int(counts.get(x, 0) > 0))
                                   for x in fit["scopes"][(_f, args)]))
                return _t.get((tuple(args), row))
            self.registry.overlay(f, ov)
            shipped.append(f)
        self._overlays.append(shipped)
        self._overlay_names.append("notch fit")
        self._overlay_srcs.append(("notch fit", tuple(sorted(shipped))))
        self._notch = None
        self._rebuild_pricer()
        self._say(f"overlay shipped on {', '.join(shipped)} (depth "
                  f"{len(self._overlays)}) — the catalog is unchanged "
                  f"(key {self.key[:12]}…); `notch pop` drops it")

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
        command that raises is reported, and the session survives.
        With a transcript open, each TOP-LEVEL command is logged with
        its output (an instrument's inner fork/do/back is not)."""
        tlog = self.__dict__.get("_tlog")
        depth = self.__dict__.get("_tdepth", 0)
        verb = line.strip().split(" ", 1)[0]
        logging = tlog is not None and depth == 0 and verb not in (
            "transcript", "quit", "exit", "EOF")
        self._tdepth = depth + 1
        if logging:
            self._theredocs = []
            keep, tee = self.stdout, _Tee(self.stdout)
            self.stdout = tee
        try:
            return super().onecmd(line)
        except KeyboardInterrupt:
            self._say("(interrupted)")
        except Exception as e:
            d = getattr(e, "diagnostic", None)
            if d is not None:         # e.g. G5 reads-undeclared: listed
                _note(self, d)        # by `findings`
            self._say(f"error ({type(e).__name__}): {e}"
                      + (" — a weight binding read outside its declared "
                         "`reads` (G5); `ports` shows the scopes"
                         if type(e).__name__ == "ScopeViolation" else ""))
        finally:
            self._tdepth = depth
            if logging:
                self.stdout = keep
                self._tlog_command(line, verb, tee.getvalue())

    # -- the lab notebook (terminal only) -------------------------------

    def do_transcript(self, arg):
        """transcript [FILE] | transcript off — a lab-notebook log of the
        session: every command + output, heredoc bodies, the system
        prompts in effect and their changes, candidate snapshots after
        llp/load/formalize, gate reports and accepts. FILE defaults to
        metis-transcript-YYYYmmdd-HHMMSS.md here. TERMINAL ONLY — the
        web/shared console never writes the operator's disk (D4)."""
        if not getattr(self, "terminal", False):
            self._say("transcript is terminal-only: this session is not "
                      "the interactive `metis` shell (D4 — the web "
                      "console never writes the operator's disk)")
            return
        a = arg.strip()
        tlog = self.__dict__.get("_tlog")
        if a == "off":
            if tlog is None:
                self._say("no transcript open")
                return
            tlog.write(f"\n---\n_transcript closed {_now_s()}_\n")
            tlog.close()
            self._say(f"transcript closed: {self._tpath}")
            self._tlog = None
            return
        if not a and tlog is not None:
            self._say(f"transcript open: {self._tpath} — "
                      "'transcript off' closes it")
            return
        if tlog is not None:
            tlog.close()
        import time as _time
        path = Path(a or _time.strftime(
            "metis-transcript-%Y%m%d-%H%M%S.md")).expanduser()
        f = open(path, "a", buffering=1)
        from metis.loop import prompts
        f.write(f"# metis transcript — {self.path.name}\n\n"
                f"- opened {_now_s()}\n"
                f"- catalog {self.doc.catalog.name}, key {self.key}\n"
                f"- llm seat: {self.llm or '(default)'}\n"
                f"- plugin: " + (f"{self.plugin_spec} "
                                 f"({type(self.plugin).__name__})"
                                 if self.plugin is not None else "(none)")
                + "\n"
                f"- stage {self.stage}, {len(self.fired)} fired\n")
        for kind in prompts.KINDS:
            self._tlog_prompt(f, kind, "in effect")
        self._tlog, self._tpath = f, path
        self._say(f"transcript open: {path}")

    def _tlog_prompt(self, f, kind, why):
        from metis.loop import prompts
        text, src = prompts.resolve(kind, self.sys_prompts.get(kind))
        f.write(f"\n### system prompt `{kind}` ({why}; source: {src})\n\n"
                "```text\n" + text.rstrip() + "\n```\n")

    def _tlog_command(self, line, verb, out):
        f = self._tlog
        if f is None:
            return
        f.write(f"\n## {_now_s()} — `{line.strip()}`\n")
        for body in self.__dict__.get("_theredocs", []):
            f.write("\nheredoc:\n\n```text\n" + body.rstrip("\n")
                    + "\n```\n")
        if out.strip():
            f.write("\n```text\n" + out.rstrip("\n") + "\n```\n")
        rest = line.strip()[len(verb):].strip()
        if verb in ("formalize", "narrate") and rest.startswith("system") \
                and "<<" in rest:
            self._tlog_prompt(f, verb, "changed")
        elif verb in ("llp", "load", "formalize") and self.loop.cand:
            f.write("\n#### candidate snapshot (after "
                    f"`{verb}`)\n\n```llp\n"
                    + self.loop.cand.rstrip("\n") + "\n```\n")
        elif verb == "accept":
            f.write(f"\n**accepted** — the world is now key {self.key}\n")
        elif verb == "llm":
            f.write(f"\n_llm seat: {self.llm}_\n")

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
        from metis.lang.directives import squash_atoms
        s = squash_atoms(s)
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
            elif key == "plugin" and len(args) == 1:
                self.do_plugin(args[0])
                if self.plugin_spec == args[0]:
                    self.case["plugin"] = args[0]
            elif key == "notch" and len(args) == 1:
                if self._notch_load(args[0]):
                    self.case.setdefault("notch", []).append(args[0])
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


def _read_text(r):
    """A read pattern as written: `atom(args) if guard(args)`."""
    def atom(a):
        ts = ",".join(str(t.value) for t in a.terms)
        return f"{a.pred}({ts})" if a.terms else a.pred
    return atom(r.atom) + (" if " + ", ".join(atom(g) for g in r.guards)
                           if r.guards else "")


def notch_rows(fired_how, alpha=10.0, ports=None):
    """The session notch (PLUGIN-PORTS.md integration point 6): the
    PAIRED-site estimator of kernel.notch over the session's recorded
    sampled steps — each `site` (factor, args, complement, scope row,
    the port's value then) is one Bernoulli draw of its complement
    pair: s counts the uncomplemented outcome. Prior p0 = the mean
    value the BASE predicted at those exposures (an env-driven port
    varies per step; the row sheet says what the base believed on
    average), posterior mean (alpha*p0 + s) / (alpha + n). Data is
    captured at sampling time, so undo/fork/back keep it honest.
    Only PAIRED ports (referenced as both `@w f` and `@w ~f`) are fitted
    — a categorical site needs kernel.notch's reference-relative
    estimator; those are listed as refused, never fitted as Bernoulli."""
    paired = None
    if ports is not None:
        pol = {}
        for ref in ports.weight_refs.values():
            pol.setdefault(ref.factor, set()).add(ref.complement)
        paired = {f for f, cs in pol.items() if cs == {True, False}}
    acc, refused = {}, set()
    for h in fired_how:
        site = h.get("site")
        if h.get("how") != "sampled" or site is None:
            continue
        factor, args, compl, row, f = site
        if paired is not None and factor not in paired:
            refused.add(f"{factor}: not a complement pair (@w {factor} / "
                        f"@w ~{factor}) — categorical sites need "
                        "kernel.notch")
            continue
        n, s, fsum = acc.get((factor, args, row), (0, 0, 0.0))
        acc[(factor, args, row)] = (n + 1, s + (0 if compl else 1),
                                    fsum + f)
    rows, tables, scopes, lines = {}, {}, {}, []
    total = 0
    for (factor, args, row), (n, s, fsum) in sorted(acc.items()):
        p0 = fsum / n
        post = (alpha * p0 + s) / (alpha + n)
        rows[(factor, args, row)] = (p0, post, n, s)
        tables.setdefault(factor, {})[(args, row)] = post
        scopes[(factor, args)] = tuple(a for a, _ in row)
        total += n
        head = factor if not args else f"{factor}({','.join(args)})"
        cond = " ".join(f"{a}={b}" for a, b in row) or "(no scope)"
        lines.append(f"{head} | {cond} : {p0:.3f} -> {post:.3f}   "
                     f"(n={n}, s={s}, a0={alpha:g})")
    return {"rows": rows, "tables": tables, "scopes": scopes,
            "sheet": "\n".join(lines) + "\n", "n": total,
            "refused": sorted(refused)}


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
    ap.add_argument("--plugin", metavar="SPEC",
                    help="attach a port plugin (pkg.mod:factory or "
                         "file.py:factory, relative to the catalog's "
                         "folder) — binds the catalog's weight ports and "
                         "drives its env; overrides the file's #plugin. "
                         "Terminal only (D4)")
    ap.add_argument("--host", metavar="SPEC",
                    help="with --json: the runtime process's host plugin "
                         "(pkg.mod:factory or file.py:factory) — the env "
                         "the served world runs against, driven by the "
                         "wire's `watch` op (runtime-wire). Named on the "
                         "command line only; the wire never names one")
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
    if args.json and args.plugin:
        print("--plugin is terminal-only: the --json wire never imports "
              "operator code (D4)", file=sys.stderr)
        return 2
    if args.host and not args.json:
        print("--host is the --json runtime's host plugin; the terminal "
              "shell attaches with --plugin", file=sys.stderr)
        return 2
    if args.json:
        from metis.loop.wire import serve_json
        try:
            shell = Repl(args.llp, seed=args.seed, plugin=args.host,
                         host=bool(args.host))
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
        # terminal=True: `transcript` writes the OPERATOR's own disk and
        # `plugin`/#plugin imports operator code — the interactive
        # terminal only (never the --json wire / web console: D4)
        repl = Repl(args.llp, seed=args.seed, terminal=True,
                    plugin=args.plugin)
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
