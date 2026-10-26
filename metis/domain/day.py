"""metis.domain.day — a package's day and its golden (dagent-fixtures s-86bbd1e7 step 2): the package
run as every domain agent runs it, at the package's level only, so the package can say what it does.

The golden is part of the package (domains/<d>/goldens/, the user's ownership rule of 2026-10-26):
the scripted day IN, the package's stream OUT, written by this REPL and checked by every runtime
that serves the package (metis-catalog's packages job against this, metis-agent's Go and py
agents against the same files). Neither runtime is the reference; the package is.

    goldens/day.jsonl          the day: one line a step, as a domain agent takes it —
                               {"clock": "beat"}, an intent {"id", "domain", "verb", "args"},
                               {"world": [do-lines], "id"}, {"ports": {}, "id"}, a read
                               {"tool": state | ctx | events | why | findings, "args"}
    goldens/day.out.jsonl      line 1 {"ready": {domain, beat}, "script": the narrator's script};
                               then per input {"in": line, "out": [the replies, as the reply
                               table has them (metis.domain.reply), no `voice`]}; last
                               {"end": {stage, counts, fired, records}}
    goldens/intents.jsonl      {"verb", "args"} a line: the intents the package's card is held to
    goldens/intents.out.jsonl  {"in": the intent, "out": {atom, act, used, dropped} | {error}}

The day is the package's: the program and its extensions, its host in the replay case (the person
as intents, the world as world lines), the watch (metis.loop.watch) at the domain's beat, the
mapper and the replier over intent.yaml, the fragment's words for what has no %% line. It is not
the agent's: no narrator (every text is the %% line or the fragment's ref in words), no consult,
no surprise verbs (reset, explain, escalate), no records sink, no actor, no scored beat, no
acks. Those are a domain agent's own machinery and its own tests. A read's output is the REPL's
capture as it is, but for the host's plugin spec, said as <host> (a runtime's import path); an
agent scrubs what it serves (paths, markup, its cap), so a runtime compares its read to the
golden's text scrubbed its way.

What a domain agent does at this level is what this does, line for line (metis-agent's
DomainAgent: intent, beat, signal, ports, world, the reads): `project(stream)` turns an agent's
stream into this shape (its acks and `voice` out), so an agent's day is the golden's.

    python -m metis.catalog golden plant            # write domains/plant/goldens/*.out.jsonl
    python -m metis.catalog golden plant --check    # regenerate in memory, diff (CI): exit 1 on a difference
"""
import contextlib
import importlib.abc
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

from . import package
from .fragments import from_signal, script_text, words
from .mapper import Mapped, Mapper
from .replier import Replier
from .reply import reply

GOLDENS = "goldens"
FILES = ("day.jsonl", "day.out.jsonl", "intents.jsonl", "intents.out.jsonl")

# a domain agent's lines a package day does not take (its own machinery)
AGENT_TOOLS = ("consult", "reset", "explain", "escalate", "activate", "deactivate")


class NoHost:
    beat = 0

    def submit(self, *a):
        raise RuntimeError("this domain has no host: it takes no intents")

    retract = submit


class Now:
    def __init__(self, counts):
        self.counts = counts

    def get(self, a):
        return self.counts.get(a, 0)


PREFIX = "metis_domain_day__"
_NAMESPACES = {}           # module key -> the namespace its module holds (the finder serves it)


class _Loader(importlib.abc.Loader):
    def __init__(self, key):
        self.key = key

    def create_module(self, spec):
        return None

    def exec_module(self, mod):
        mod.__dict__.update(_NAMESPACES[self.key])


class _Finder(importlib.abc.MetaPathFinder):
    """metis_domain_day__<name>: the namespace a Day built, importable (the kernel attaches the
    host by `module:make`, and reloads the module as it does)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in _NAMESPACES:
            return importlib.util.spec_from_loader(fullname, _Loader(fullname))
        return None


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.append(_Finder())


def namespace(d):
    """The domain's namespace (what a domain agent reads of a domain module) for the catalog's
    Domain `d`, as an importable module so the kernel can attach its host (`PLUGIN`)."""
    if d.agent is not None:
        raise ValueError("domain %s has an agent module of its own (%s): its day is that agent's" % (d.name, d.agent[0]))
    key = PREFIX + d.name
    ns = dict(NAME=d.name, DISPLAY=d.display, VOICE=d.manifest.get("voice", ""), CATALOG=d.program)
    if d.intent.get("spec") or d.manifest.get("host"):
        ns.update(package.build(d.name, display=d.display, voice=d.manifest.get("voice", ""), dir=d.dir,
                                module=key, fresh=True, host=d.manifest.get("host")))
    elif d.serves():
        ns.update(package.bare(d.name, display=d.display, voice=d.manifest.get("voice", "")))
    else:
        raise ValueError("domain %r has no program, agent module or intent.spec (%s)" % (d.name, d.dir))
    ns["MANIFEST"] = d
    served = d.serves()
    ns["CATALOG"], ns["EXTENSIONS"] = (served[0], list(served[1:])) if served else (ns["CATALOG"], [])
    _NAMESPACES[key] = ns
    sys.modules.pop(key, None)
    return importlib.import_module(key)


class Day:
    """One package's day: its program and extensions on the REPL, its host in the replay case, the
    watch, the mapper and the replier. `handle(line)` → the line's replies."""

    BEAT_STEPS = 200          # a domain agent's watch steps a beat may take (the clock's steps overrides)
    HISTORY = 2000            # the fired rows the REPL keeps (the clock's history overrides; 0 = all)
    EXACT_STATES, SAMPLES = 10000, 500

    def __init__(self, module, seed=1):
        os.environ.setdefault("METIS_CONFIG", "")      # the kernel's llm endpoint is never called
        from metis.lang.repl import Repl
        from metis.loop.watch import Watch
        self.mod, self.name = module, module.NAME
        plugin = getattr(module, "PLUGIN", None)
        clock = getattr(getattr(module, "MANIFEST", None), "clock", None) or {}
        history = int(clock.get("history", self.HISTORY))
        self.repl = Repl(str(module.CATALOG), seed=seed, plugin=plugin, host=bool(plugin), history=history or None,
                         stdout=io.StringIO())
        self.extensions = list(getattr(module, "EXTENSIONS", None) or [])
        package.apply_extensions(self.repl, self.extensions)
        self.host = self.repl.plugin
        if plugin and self.host is None:
            raise RuntimeError("day: host %s did not attach" % plugin)
        if self.host is None:
            self.host = NoHost()
        inner = getattr(self.host, "inner", self.host)
        if hasattr(inner, "reseed"):
            inner.reseed(seed)
        lockstep, follow = bool(clock.get("lockstep")), bool(clock.get("follow"))
        tier = None
        if lockstep or follow:
            tier = {"max_states": int(clock.get("exact_states", self.EXACT_STATES)),
                    "samples": int(clock.get("samples", self.SAMPLES)), "seed": seed}
        self.watch = Watch(self.repl, beats=getattr(module, "BEATS", None), lockstep=lockstep, tier=tier, follow=follow)
        self.mapper = Mapper(module.CARD)
        handles = getattr(module, "HANDLE_NAMES", {})
        self.display_of = lambda v: handles.get(v, words(v))
        docs = package.rule_docs(module.CATALOG)
        for p in self.extensions:
            docs.update(package.rule_docs(p))
        self.replier = Replier(self.name, docs, module.OUTCOMES, self.mapper.commands,
                               getattr(module, "alternatives", None), package.mirror_atoms(self.repl.text),
                               getattr(module, "PAIRS", ()), self.display_of)
        self.own_voice = getattr(module, "VOICE", "") == "own"
        self.host.last_row = lambda: self.repl.fired_rows[-1]
        if hasattr(self.host, "replay"):
            self.host.replay()
        elif not getattr(module, "OWN_DAY", False) and not isinstance(self.host, NoHost):
            raise ValueError("domain %s has no replay case (its host keeps its own day)" % self.name)
        self.over = False
        self.held = []
        self.display = getattr(module, "DISPLAY", self.name)
        self.exo_events = getattr(module, "EXO_EVENTS", False)
        self.quiescence = getattr(module, "QUIESCENCE", "episode")
        self.beat_steps = int(getattr(module, "BEAT_STEPS", 0) or clock.get("steps") or self.BEAT_STEPS)
        self.script = package.scripts_of(self.repl, self.extensions)
        displays = dict(getattr(module, "DISPLAYS", {}), **{self.name: self.display})
        self.names = lambda p: displays.get(p, words(p))
        if hasattr(inner, "sync"):
            self.sync_world(inner)

    def beat_no(self):
        return getattr(self.host, "beat", self.watch.beat)

    def ready(self):
        return {"clock": "ready", "domain": self.name, "beat": self.beat_no()}

    def script_text(self):
        """The script the narrator gets: the cards as the agent holds them, in display names."""
        return script_text(self.name, self.script, self.names)

    # ── a line ──
    def handle(self, msg):
        if "clock" in msg:
            if msg["clock"] != "beat":
                return [reply(self.name, "error", error="a package day beats (clock: beat), not %r" % msg["clock"])]
            return self.beat() + [{"clock": "beat", "beat": self.beat_no(), "over": self.over}]
        if "tool" in msg:
            return self.tool(msg)
        if "verb" in msg:
            return self.intent(msg)
        if "world" in msg:
            return self.world(msg)
        if "ports" in msg:
            return self.ports(msg)
        return [reply(self.name, "error", id=msg.get("id"), error="not a line of a package's day: %s" % json.dumps(msg))]

    def intent(self, msg):
        iid = msg.get("id")
        if msg.get("domain", self.name) != self.name:
            return [reply(self.name, "error", id=iid, error="not this domain: %r" % msg.get("domain"))]
        if self.over:
            return [reply(self.name, "error", id=iid, error="the host's episode is over")]
        if self.watch.open is not None:
            # a surprise is open: taken now, refused at the next beat (no new orders until it closes)
            self.held.append(self.surprise_open_refusal(iid))
            return [reply(self.name, "result", id=iid, status="accepted", ref={"beat": self.beat_no()}, text="Accepted.")]
        if not self.mapper.commands:
            return [reply(self.name, "error", id=iid, error=package.NO_INTENTS)]
        m = self.mapper.map(msg)
        atom = m.atom if isinstance(m, Mapped) else None
        inner = getattr(self.host, "inner", self.host)
        if atom is not None and hasattr(inner, "intent_atom"):
            atom = inner.intent_atom(atom)
        if not isinstance(m, Mapped):
            return [reply(self.name, "error", id=iid, error="mapper: " + m.error)]
        self.host.submit(iid, atom, m.dropped)
        self.replier.submitted(iid, atom, m.act)
        return [reply(self.name, "result", id=iid, status="accepted", ref={"beat": self.beat_no()}, text="Accepted.")]

    def surprise_open_refusal(self, iid):
        k = self.watch.open
        try:
            diff = dict(self.repl.loop.records[k]["meta"].get("diff") or {})
        except (IndexError, KeyError, TypeError):
            diff = {}
        dtext = ", ".join("%s %+d" % (a.replace("_", " "), n) for a, n in sorted(diff.items()))
        text = "surprise #%s open%s: I take no new orders until it is closed" % (k, " (%s)" % dtext if dtext else "")
        return reply(self.name, "result", id=iid, status="refused", reason="surprise_open", blocker=None,
                     ref={"seq": len(self.repl.fired), "beat": self.beat_no()}, text=text)

    def beat(self):
        """One beat: the watch steps until it would tick again (at most beat_steps), then the
        commands no rule took."""
        w, out = self.watch, self.held
        self.held = []
        if self.over:
            return out
        b0, f0, n = w.beat, len(self.repl.fired_rows), 0
        inner = getattr(self.host, "inner", self.host)
        lock = getattr(w, "lockstep", False)
        while not w.done:
            if getattr(inner, "done", False):
                w.done = True
                break
            would_tick = w.observe_only or (w.quiet and not self.repl._candidates()) or getattr(w, "held", False)
            if would_tick and (w.beat > b0 or (lock and len(self.repl.fired_rows) > f0)):
                break
            if n >= self.beat_steps:
                out += self.unsettled(f0, n)
                break
            n += 1
            for s in w.step():
                out += self.signal(s)
        if w.done:
            self.over = True
        late, gone = self.replier.untaken(self.repl.counts, self.beat_no())
        for atom in gone:
            self.host.retract(atom)
        return out + late

    def unsettled(self, f0, n):
        times = {}
        for kind, _, ev in self.repl.fired_rows[f0:]:
            if kind != "exo":
                times[ev.name] = times.get(ev.name, 0) + 1
        rules = sorted(times.items(), key=lambda kv: (-kv[1], kv[0]))
        text = "%s did not settle: %d steps in one beat, and %s kept firing; the beat stopped there" % (
            self.display, n, ", ".join("%s (%d)" % kv for kv in rules[:5]) or "no rule")
        return [reply(self.name, "quiescence", goal_pending=False, unsettled=[list(kv) for kv in rules],
                      ref={"seq": len(self.repl.fired), "beat": self.beat_no()}, text=text)]

    def signal(self, s):
        """One watch signal → its replies: a fired clause through the replier (its %% line), an
        outside input or a reading where the domain reports them, a quiescence where it reports
        every one, a surprise."""
        kind, beat = s["signal"], self.beat_no()
        if kind == "event" and s.get("kind") == "clause":
            k = s["fired"]
            ev = self.repl.fired_rows[k - 1][2]
            return self.replier.fired(ev, k, beat, self.repl.counts, None)
        if kind == "event":
            if s.get("kind") == "exo" and not self.exo_events:
                return []
            f = from_signal(s, Now(self.repl.counts))[0]
            ref = {"seq": len(self.repl.fired), "beat": beat, "event": f.ref}
            return [reply(self.name, "event", ref=ref, counts=self.replier.counts(self.repl.counts), text=words(f.ref))]
        if kind == "quiescence":
            if s.get("episode_over"):
                self.over = True
            elif self.quiescence != "every":
                return []
            f = from_signal(s, None)[0]
            return [reply(self.name, "quiescence", goal_pending=False, over=True if s.get("episode_over") else None,
                          ref={"seq": len(self.repl.fired), "beat": beat},
                          text="the day is over" if s.get("episode_over") else words(f.ref))]
        if kind == "surprise":
            f = from_signal(s, None)[0]
            return [reply(self.name, "surprise", record=s["record"], difference=s.get("diff", {}),
                          cards=package.other_cards(self.repl), ref={"seq": len(self.repl.fired), "beat": beat},
                          text=words(f.ref))]
        return []

    def tool(self, msg):
        name, args, iid = msg.get("tool"), msg.get("args") or {}, msg.get("id")
        if name in package.READS:
            arg = args.get("atom") or args.get("arg") or ""
            return [reply(self.name, "read", tool=name, output=self.repl.capture(name, arg) or "(no output)",
                          ref={"seq": len(self.repl.fired), "beat": self.beat_no()})]
        if name in AGENT_TOOLS:
            return [reply(self.name, "error", id=iid, error="tool %s: a domain agent's, not a package day's" % name)]
        return [reply(self.name, "error", id=iid, error="no tool %r" % name)]

    def world(self, msg):
        if not hasattr(self.host, "world"):
            return [reply(self.name, "error", id=msg.get("id"), error="this domain's host takes no world lines")]
        lines = msg["world"] if isinstance(msg["world"], list) else [msg["world"]]
        try:
            self.host.world(lines)
        except ValueError as e:
            return [reply(self.name, "error", id=msg.get("id"), error=str(e))]
        inner = getattr(self.host, "inner", self.host)
        if hasattr(inner, "sync"):
            self.sync_world(inner)
        return []

    def sync_world(self, inner):
        self.repl.plugin.tick = inner.sync
        try:
            self.repl._tick(quiet_rows=False)
        finally:
            del self.repl.plugin.tick
        self.watch._checkpoint()
        self.watch.quiet = False
        if not getattr(inner, "done", False):
            self.watch.done, self.over = False, False

    def ports(self, msg):
        pairs = [list(p) for p in getattr(self.mod, "PAIRS", ())]
        sides = {x for p in pairs for x in p}
        held = sorted(a for a, n in self.repl.counts.items() if n > 0 and a.partition("(")[0] in sides)
        kinds = (getattr(self.mod, "CARD", None) or {}).get("handles", {})
        standing = set(getattr(self.mod, "STANDING", ()))
        handles = {k: [{"id": h, "display": self.display_of(h)} for h, _ in hs if h not in standing]
                   for k, hs in kinds.items()}
        inner = getattr(self.host, "inner", self.host)
        extra = {"host": inner.ports()} if hasattr(inner, "ports") else {}
        return [reply(self.name, "ports", world=hasattr(self.host, "world"), pairs=pairs,
                      reports=dict(getattr(self.mod, "REPORTS", {})), handles=handles, held=held,
                      ref={"seq": len(self.repl.fired), "beat": self.beat_no()}, **extra)]

    def end(self):
        """The day's end state: the stage, the counts, every fired name, the surprise records."""
        return end_state(self.repl, getattr(self.mod, "PLUGIN", None))


HOST = "<host>"       # a read's words for the host's plugin spec: the runtime's import path, not the package's


def project(stream, plugin=None):
    """A domain agent's stream (its replies and acks, a line's at a time) in a golden's shape: the
    acks out, each reply's `voice` out, and in a read the runtime's plugin spec (`plugin`, the
    module:factory it attached the host by) said as <host>."""
    out = []
    for r in stream:
        if "ack" in r:
            continue
        r = {k: v for k, v in r.items() if k != "voice"}
        if plugin and r.get("kind") == "read" and isinstance(r.get("output"), str):
            r["output"] = r["output"].replace(plugin, HOST)
        out.append(r)
    return out


def end_state(repl, plugin=None):
    """A day's end on a REPL: {stage, counts, fired, records} — the records as JSON, the host's
    plugin spec in them (meta.host) said as <host>."""
    text = json.dumps(repl.loop.records, default=str)
    if plugin:
        text = text.replace(json.dumps(plugin)[1:-1], HOST)
    return {"stage": repl.stage, "counts": dict(sorted(repl.counts.items())), "fired": list(repl.fired),
            "records": json.loads(text)}


def _dump(rows):
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


def run_day(d, lines, seed=1):
    """The package `d`'s day over `lines` → the golden's rows (ready + script, a row a line, end)."""
    with contextlib.redirect_stderr(io.StringIO()):
        day = Day(namespace(d), seed=seed)
    rows = [{"ready": day.ready(), "script": day.script_text()}]
    for msg in lines:
        with contextlib.redirect_stderr(io.StringIO()):
            out = day.handle(msg)
        rows.append({"in": msg, "out": project(out, getattr(day.mod, "PLUGIN", None))})
    rows.append({"end": day.end()})
    return rows


def map_intents(d, lines):
    """The package `d`'s card over its intents → {"in", "out"} rows."""
    import yaml
    f = Path(d.dir) / "intent.yaml"
    spec = (yaml.safe_load(f.read_text()) or {}) if f.exists() else {}
    mp = Mapper(package.card(spec))
    rows = []
    for msg in lines:
        m = mp.map(msg)
        out = ({"atom": m.atom, "act": m.act, "used": list(m.used), "dropped": dict(m.dropped)}
               if isinstance(m, Mapped) else {"error": m.error})
        rows.append({"in": msg, "out": out})
    return rows


def _read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def golden(d, seed=1):
    """{file name: text} of the package `d`'s generated goldens (the .out files of the inputs it has)."""
    g = Path(d.dir) / GOLDENS
    out = {}
    if (g / "day.jsonl").is_file():
        out["day.out.jsonl"] = _dump(run_day(d, _read_jsonl(g / "day.jsonl"), seed))
    if (g / "intents.jsonl").is_file():
        out["intents.out.jsonl"] = _dump(map_intents(d, _read_jsonl(g / "intents.jsonl")))
    return out


def check(d, seed=1):
    """[(file, problem)] — the package's committed goldens against a fresh generation."""
    import difflib
    g = Path(d.dir) / GOLDENS
    bad = []
    gen = golden(d, seed)
    if not gen:
        return [(str(g), "no goldens/day.jsonl or goldens/intents.jsonl")]
    for name, text in gen.items():
        f = g / name
        if not f.is_file():
            bad.append((name, "not written (python -m metis.catalog golden %s)" % d.name))
        elif f.read_text() != text:
            diff = list(difflib.unified_diff(f.read_text().splitlines(True), text.splitlines(True),
                                             "committed " + name, "generated", n=0))
            bad.append((name, "differs from what the package makes now:\n" + "".join(l[:400] + ("\n" if len(l) > 400 else "")
                                                                                       for l in diff[:20])))
    return bad
