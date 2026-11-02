"""metis.trace — a composition's trace (metis-lang spec/17 amendment 7; trace-replay s-c111ff41): the sequence of its
ticks with what each member applied, the rules that fired and the outputs, on a checkpoint. Given the checkpoint a
trace determines the run, so the model replays it with no world, no agent and no LLM. A member is replayed ALONE: copy
semantics, its inputs are all it had.

    TRACE/seg-<k0>.jsonl           a segment: its header, then its ticks
    TRACE/checkpoints/k<k>.json    each segment starts at one, so dropping the oldest segment is the ring

    header      {"trace": 1, "k0": 0, "checkpoint": "checkpoints/k0.json", "clock": "tick", "composition": ...,
                 "image": ..., "protocol": <protocol_key()>, "commands": [the world's commands],
                 "clocks": [{"domain", "n"}: the order Live beats them in], "live": <the key of the Live that wrote it>,
                 "members": {member: {"key": <the program's sha256>, "seed": 1, "weights": [], "out": [preds]}}}
    tick        {"k": 1, "t": 0.5, "members": {member: {"why": "change", "copy": {"+": [...], "-": [...]},
                                                         "seq": [...], "out": [...]}}}
                ("between": true: what came between two ticks, before a segment's end. A header's "cut": "reload": its
                checkpoint does not follow from the last segment: a session was replaced)
    seq         in order: {"beat": [do-lines], "fired": ["rule[X=a]", ...]} (its inputs, then its rules; "left": true when
                the beat ended with rules still enabled: a world that took its tick, a step cap);
                {"take": [do-lines], "by": origin} (between beats; it fires at the next beat, the stub's run: false; by: relay,
                deliver: regenerated in a composition's replay, the agent carries them; world, intent, tool: its inputs);
                {"step": [rules], "by": origin} (fired out of any beat); {"reload": true} (a new session: the segment ends with this tick)
                A member that beat with nothing in or fired, and took nothing, has its `why` and no seq. `out` is every
                out port the tick's rules said (outputs below): the out atoms, and each world command as {"rule", "kind",
                "args"}, so a candidate's first differing output is the first real divergence.
    checkpoint  {"k": 0, "sessions": {member: {"stage", "quiet", "rng": "<decimal>", "counts": {atom: n}}},
                 "copies": {member: {atom: n}}, "world": <the world's own reference>}

`in` (the do-lines a member applied) is the truth for the determinism check; `copy` is what a candidate's replay syncs
from, since a do-line is a delta against the recorded session. A candidate is exact up to its first differing output.

The driver of metis-loop's policy has no ticks: its header says clock: firing, its lines are its own (pi/policy/tracing.py), and an
acting episode replays by `replay_firings`: its session's start, its seed, {name, lines} per firing.

    w = Segments(dir, every=200, ring=6000)             the writer: open_segment(header, checkpoint), tick(rec)
    state = session_state(shell)                         a session as a checkpoint holds it; restore_session(shell, state)
    tr = read(dir)                                       the segments still there, oldest first
    replay(dir, "planner", "planner.llp")               one member: → Replay(.ok, .ticks, .beats, .rules, .divergence)
    replay_firings("nh.llp", start, seed, firings)       an acting episode: → Replay
    case(dir, "planner", k, after=[...], out=[...])      the surprise at tick k as a case of the package (the stub's YAML): → its text
    blame(dir, "planner", k, atoms)                      who last moved what is missing or stale: the member's rule, the world, or a take
    goldens(domain, seeds=(1, 2, 3))                     the package's cases on several seeds: → (ok, [the cases that fail, named])
    apply_pack(unit, pack)                               a hand-written pack on a unit: → (the unit's new text, {added, dropped, whole})
    duplicates(pack, texts)                              a pack's rules that repeat one already there, by name or by its premises and effects

    python -m metis.trace replay DIR MEMBER PROGRAM
    python -m metis.trace case DIR MEMBER --at K [--after ATOM ...] [--out ATOM ...] [--into CASES_DIR]
    (the composition's replay and the fired diff are metis_agent.trace's: they run the agent's own relay code)
"""
import hashlib
import io
import json
import contextlib
import os
import shutil
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

FORMAT = 1
STEPS = 200                         # the most rules a beat replays to quiescence: past it, a beat that never settles
GROUND_BUDGET = 1000000             # a session's events unless METIS_GROUND_BUDGET says: about 900 MB (NetHack's measured 248,254 events, 220 MB)


class TraceError(ValueError):
    pass


class OutOfTime(BaseException):
    """A judgement past its time budget: not an `except Exception`'s to swallow on the way out, and not a verdict: "not judged"."""

    def __init__(self, label="", seconds=None):
        super().__init__("%s: past its budget of %s s" % (label, seconds))
        self.label, self.seconds = label, seconds


@contextlib.contextmanager
def deadline(seconds, label=""):
    """OutOfTime in the body after SECONDS of wall clock (None or 0: no limit; off the main thread: no limit, SIGALRM is the main thread's). An outer deadline still holds: the
    nearer one fires, and the outer goes on counting after."""
    if not seconds or threading.current_thread() is not threading.main_thread():
        yield
        return

    def fire(_sig, _frame):
        raise OutOfTime(label, seconds)
    outer_handler = signal.signal(signal.SIGALRM, fire)
    outer_left = signal.getitimer(signal.ITIMER_REAL)[0]
    t0 = time.monotonic()
    signal.setitimer(signal.ITIMER_REAL, min(seconds, outer_left) if outer_left else seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, outer_handler)
        if outer_left:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, outer_left - (time.monotonic() - t0)))


# ── a session as a checkpoint holds it ──
def session_state(shell):
    """The session's counts, stage, quiet stage and RNG (K3's fields, metis-live 1). The RNG as a decimal string: it is
    past 2^53, and a page's JSON would round it."""
    return {"stage": shell.stage, "quiet": shell.__dict__.get("_quiet_stage"), "rng": str(shell.rng.state),
            "counts": {a: n for a, n in sorted(shell.counts.items()) if n > 0}}


def restore_session(shell, state):
    shell.stage, shell.counts = state["stage"], dict(state["counts"])
    shell.rng.state = int(state["rng"])
    if state.get("quiet") is not None:
        shell._quiet_stage = state["quiet"]
    else:
        shell.__dict__.pop("_quiet_stage", None)


def program_key(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_of(shell):
    return sorted(str(getattr(w, "name", w)) for w in shell.doc.ports.weights)


def protocol_key():
    """The protocol's standard units as this toolchain has them. A trace from before a protocol change does not replay:
    its checkpoints lack what the change added (tick__gen, say)."""
    d = Path(__file__).resolve().parent / "catalog" / "iface"
    h = hashlib.sha256()
    for f in sorted(d.glob("*.llp")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()[:16]


# ── an output: a world command (spec/17 amendment 6's test) or an out atom ──
def pred_of(atom):
    return atom.split("(")[0]


def command_entry(name, said=None):
    """A world command as `out` holds it: {"rule": the event, "kind", "args"}. A case may assert the KIND a norm requires
    (expect.out): a yield and an accel leave the same tokens. SAID: the host's (kind, args) for it, as its restrictor emits
    them; else the rule's own name and bindings."""
    base, _, rest = name.partition("[")
    if said is not None:
        return {"rule": name, "kind": said[0], "args": [a for a in said[1:] if a is not None]}
    return {"rule": name, "kind": base, "args": [b.split("=", 1)[-1] for b in rest.rstrip("]").split(",") if b]}


def outputs(ev, commands, out_preds, said=None):
    """What the fired event EV says outside its member: the world commands of its rule (a rule named P or P_…, or P in
    its head; command_entry, SAID(name) its host's words) and the atoms it produced of the member's out preds."""
    base, out = ev.name.split("[")[0], []
    heads = {pred_of(a) for a in ev.produce}
    if any(base == p or base.startswith(p + "_") or p in heads for p in commands):
        out.append(command_entry(ev.name, said(ev.name) if said else None))
    out += sorted(a for a in ev.produce if pred_of(a) in out_preds)
    return out


def out_name(x):
    """An output as a replay compares it: a host's words on a command are not the model's to reproduce."""
    return x["rule"] if isinstance(x, dict) else x


# ── the writer: segments on disk, the ring ──
class Segments:
    """Lines to TRACE/seg-<k0>.jsonl, one file a segment. EVERY: the ticks of a segment (0: one segment). RING: the ticks
    kept at least (0: all of them, a test run's); the oldest segments go with their checkpoints, whole."""

    def __init__(self, path, every=200, ring=0, keep=False):
        """KEEP: a resumed run goes on after what is there (last_k); else a start begins anew."""
        self.dir, self.every, self.ring = Path(path), int(every or 0), int(ring or 0)
        if self.dir.exists() and not keep:
            shutil.rmtree(self.dir)
        (self.dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.fh = None
        self.k0s = sorted(int(p.stem[4:]) for p in self.dir.glob("seg-*.jsonl"))

    def last_k(self):
        """The last line's k in what is there, or -1."""
        if not self.k0s:
            return -1
        lines = (self.dir / ("seg-%d.jsonl" % self.k0s[-1])).read_text().splitlines()
        return json.loads(lines[-1]).get("k", self.k0s[-1] - 1) if len(lines) > 1 else self.k0s[-1] - 1

    def checkpoint(self, k, data):
        """TRACE/checkpoints/k<k>.json; → its name, as a line says it."""
        (self.dir / "checkpoints" / ("k%d.json" % k)).write_text(json.dumps(dict(data, k=k), sort_keys=True))
        return "checkpoints/k%d.json" % k

    def due(self, k):
        """A segment ends before tick K."""
        return self.fh is None or (self.every and k - self.k0s[-1] >= self.every)

    def open_segment(self, k, header, checkpoint):
        if self.fh:
            self.fh.close()
        name = self.checkpoint(k, checkpoint)
        self.fh = open(self.dir / ("seg-%d.jsonl" % k), "w")
        self.k0s.append(k)
        self.line(dict(header, trace=FORMAT, k0=k, checkpoint=name))
        while self.ring and len(self.k0s) > 1 and k - self.k0s[1] >= self.ring:
            old = self.k0s.pop(0)
            (self.dir / ("seg-%d.jsonl" % old)).unlink(missing_ok=True)
            (self.dir / "checkpoints" / ("k%d.json" % old)).unlink(missing_ok=True)

    def line(self, rec):
        self.fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


# ── the reader ──
@dataclass
class Segment:
    k0: int
    header: dict
    checkpoint: dict
    ticks: List[dict]


def read(path):
    """The segments still there, oldest first."""
    d, out = Path(path), []
    for f in sorted(d.glob("seg-*.jsonl"), key=lambda p: int(p.stem[4:])):
        lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if not lines:
            continue
        h = lines[0]
        if h.get("trace") != FORMAT:
            raise TraceError("%s: trace format %r, this reads %d" % (f, h.get("trace"), FORMAT))
        out.append(Segment(h["k0"], h, json.loads((d / h["checkpoint"]).read_text()), lines[1:]))
    if not out:
        raise TraceError("%s: no segment" % d)
    return out


# ── the replay: one member, no world ──
@dataclass
class Replay:
    member: str
    ticks: int = 0
    beats: int = 0
    rules: int = 0
    inputs: int = 0
    fired: dict = field(default_factory=dict)         # {k: [rule names fired at k]}
    out: dict = field(default_factory=dict)           # {k: [outputs at k]}
    divergence: Optional[dict] = None                 # the first: {k, what, want, got}

    @property
    def ok(self):
        return self.divergence is None


def open_session(program, seed=1):
    """A session on PROGRAM, under a grounding budget: a rule that grounds past it is refused in its own words
    (metis.kernel.ground.GroundBudget), before a replay eats the memory scope."""
    from metis.lang.repl import Repl as Shell
    set_here = "METIS_GROUND_BUDGET" not in os.environ
    if set_here:
        os.environ["METIS_GROUND_BUDGET"] = str(GROUND_BUDGET)
    try:
        return Shell(str(program), seed=seed, stdout=io.StringIO())
    finally:
        if set_here:
            del os.environ["METIS_GROUND_BUDGET"]


def _apply(sh, lines):
    from metis.fw.plugin import exo_event
    from metis.kernel.refsample import DONE
    _, after = exo_event(list(lines), sh.counts)
    if sh.stage == DONE:                          # a tick revives a quiet world (Repl._tick)
        sh.stage = sh.__dict__.get("_quiet_stage") or sh.prog.init_stage
    sh.counts = after


def _fire_all(sh, want, left):
    got = []
    while sh._candidates() and len(got) < (len(want) if left else STEPS):
        got.append(sh._sample())
    return got


def _state_differs(sh, state):
    """Where the session is not as the checkpoint says, or None."""
    have = session_state(sh)
    for f in ("counts", "rng", "stage"):
        if have[f] != state[f]:
            if f == "counts":
                a, b = have[f], state[f]
                return {"what": "counts", "want": sorted(set(b.items()) - set(a.items()))[:8], "got": sorted(set(a.items()) - set(b.items()))[:8]}
            return {"what": f, "want": state[f], "got": have[f]}
    return None


def replay_segment(seg, member, program, r, check_key=True):
    """SEG for MEMBER on PROGRAM into R: from its checkpoint. → False at a divergence, or at a reload (the next segment
    starts from its own checkpoint)."""
    m = seg.header["members"].get(member)
    if m is None:
        raise TraceError("no member %r in the trace (it has: %s)" % (member, ", ".join(sorted(seg.header["members"]))))
    if m.get("weights"):
        raise TraceError("%s declares weight ports (%s): a trace does not carry their values yet" % (member, ", ".join(m["weights"])))
    have = seg.header.get("protocol")
    if have and have != protocol_key():
        raise TraceError("this trace is of another protocol (%s, this toolchain's %s): its checkpoints lack what changed since; "
                         "it does not replay across a protocol change" % (have, protocol_key()))
    if check_key and m.get("key") and m["key"] != program_key(program):
        raise TraceError("%s: %s is not the program the trace was written from" % (member, program))
    commands, out_preds = seg.header.get("commands") or (), set(m.get("out") or ())
    sh = open_session(program, m.get("seed", 1))
    restore_session(sh, seg.checkpoint["sessions"][member])
    events = {ev.name: ev for _, _, ev in sh.events_tbl}

    def stop(k, what, want, got):
        r.divergence = {"k": k, "what": what, "want": want, "got": got}
        return False

    for t in seg.ticks:
        k, e = t["k"], t["members"].get(member)
        r.ticks += 1
        if e is None:
            continue
        seq = e.get("seq")
        if seq is None:
            if "why" not in e:                     # its reading changed, it did not beat
                continue
            seq = [{"beat": [], "fired": []}]      # it beat with nothing: nothing may fire
        fired, out = [], []
        for item in seq:
            if "reload" in item:
                return False
            if "take" in item:
                _apply(sh, item["take"])
                r.inputs += 1
            elif "beat" in item:
                if item["beat"]:
                    _apply(sh, item["beat"])
                    r.inputs += 1
                want = item.get("fired") or []
                got = _fire_all(sh, want, item.get("left"))
                r.beats += 1
                r.rules += len(got)
                if got != want:
                    return stop(k, "fired", want, got)
                fired += got
            elif "step" in item:
                got = [sh._sample() for _ in item["step"]]
                r.rules += len(got)
                if got != item["step"]:
                    return stop(k, "step", item["step"], got)
                fired += got
        for name in fired:
            ev = events.get(name)
            if ev is not None:
                out += outputs(ev, commands, out_preds)
        want_out = [out_name(x) for x in e.get("out") or []]
        if [out_name(x) for x in out] != want_out:
            return stop(k, "out", want_out, [out_name(x) for x in out])
        if fired:
            r.fired[k] = fired
        if out:
            r.out[k] = out
    return sh


def replay(path, member, program, check_key=True):
    """MEMBER alone over the trace at PATH, on PROGRAM (its .llp). Each beat's inputs applied, then its rules to
    quiescence: nothing says how many to fire, and a beat that fired nothing must fire nothing. At each next segment's
    checkpoint the session must be as that checkpoint says. → Replay; .divergence names the first."""
    segs = read(path)
    r = Replay(member)
    for i, seg in enumerate(segs):
        sh = replay_segment(seg, member, program, r, check_key)
        if r.divergence:
            return r
        following = i + 1 < len(segs) and "cut" not in segs[i + 1].header and segs[i + 1].k0 in (seg.ticks[-1]["k"], seg.ticks[-1]["k"] + 1) if seg.ticks else False
        if sh is not False and following:
            bad = _state_differs(sh, segs[i + 1].checkpoint["sessions"][member])
            if bad:
                r.divergence = dict(bad, k=segs[i + 1].k0, what="checkpoint " + bad["what"])
                return r
    return r


def replay_firings(program, start, seed, firings, member=""):
    """An acting episode of metis-loop's policy (pi/policy/acting.py), the session alone: START its {stage, counts}, SEED the
    episode's random.Random, FIRINGS its {name, lines}. Each firing chosen as the loop chooses (the seeded rng over the enabled
    rules), the world's lines applied after it. → Replay; .divergence names the first."""
    import random
    from metis.fw.plugin import exo_event
    r, rng = Replay(member), random.Random(seed)
    sh = open_session(program)
    sh.stage, sh.counts = start["stage"], dict(start["counts"])
    for k, f in enumerate(firings):
        ws = sh._candidates()
        r.ticks += 1
        if not ws:
            r.divergence = {"k": k, "what": "fired", "want": [f["name"]], "got": []}
            return r
        name = sh._fire(rng.choices([i for i, _ in ws], weights=[x for _, x in ws])[0], ws=ws)
        r.rules += 1
        if name != f["name"]:
            r.divergence = {"k": k, "what": "fired", "want": [f["name"]], "got": [name]}
            return r
        r.fired[k] = [name]
        if f.get("lines"):
            _, sh.counts = exo_event(list(f["lines"]), sh.counts)
            r.inputs += 1
    return r


# ── a surprise as a case of the package ──
def _segment_of(segs, k, k0=None):
    """The segment the window starts in: holding K0, else the one holding K."""
    at = k if k0 is None else k0
    seg = next((x for x in reversed(segs) if x.k0 <= at), None)
    if seg is None:
        raise TraceError("tick %d is before the trace (it starts at %d: the ring)" % (at, segs[0].k0))
    return seg


def case(path, member, k, after=(), out=(), k0=None, note=None, units=()):
    """The window to tick K (the trace's k, as `at` is: a world log's own tick may be behind it) of MEMBER as a case: the checkpoint its segment starts from, one boundary per beat or take, and at K the
    observed outcome the rules did not produce: `after` (what the world did) and `out` (the command kinds a norm requires). No rule is
    named, so the case survives the rule that passes it. → the YAML's text; it fails in the package's harness until a rule passes it.
    Out's kinds match RULE names, with globs: the law is written in rules, and a host's rendered kind is the world's accounting (rulecheck). A case runs one
    member alone on recorded takes: a golden; the composition's judgement is metis_agent.trace.judge, which holds the same expectation at the tick."""
    import yaml
    if not after and not out:
        raise TraceError("a case asserts something: --after (what the world did) or --out (the command a norm requires)")
    segs = read(path)
    start = _segment_of(segs, k, k0)
    st = start.checkpoint["sessions"].get(member)
    if st is None:
        raise TraceError("no member %r in the trace" % member)
    bounds, last = [], None
    for seg in segs:
        if seg.k0 < start.k0:
            continue
        for t in seg.ticks:
            if t["k"] > k:
                break
            if k0 is not None and t["k"] < k0:
                continue
            e = t["members"].get(member) or {}
            seq = e.get("seq")
            if seq is None and "why" in e:
                seq = [{"beat": []}]
            for item in seq or ():
                if "take" in item:
                    bounds.append({"lines": list(item["take"]), "run": False})
                elif "beat" in item:
                    bounds.append({"lines": list(item["beat"])})
                    last = bounds[-1]
    if last is None:
        raise TraceError("%s did not beat in the window to tick %d" % (member, k))
    # what the window's last beat must show, the takes after it being what brought the member there
    last.setdefault("expect", {})
    if after:
        last["expect"]["after"] = list(after)
    if out:
        last["expect"]["out"] = list(out)
    doc = {"case": "surprise at tick %d: %s" % (k, member), "note": note or "written by metis.trace case from a trace; it fails until a rule passes it",
           **({"units": list(units)} if units else {}), "member": member, "at": k, "seed": start.header["members"][member].get("seed", 1), "rng": st["rng"], "stage": st["stage"],
           "init": ["%s %d" % (a, n) if n != 1 else a for a, n in sorted(st["counts"].items())], "boundaries": bounds}
    return yaml.safe_dump(doc, sort_keys=False, width=200)


# ── whose is it: on the baseline, before any candidate ──
def blame(path, member, k, atoms, program=None):
    """At tick K, who last moved each of ATOMS (`atom`: it should hold; `atom 0`: it should not) in MEMBER's session, over all the trace holds: a rule of the member, its world
    (`beat`), or a take (relay, deliver: the director's, carried by the agent). → {"atoms": [{atom, want, have, last: {k, by, what}, whose}], "verdict": "this package's" | "not this
    package's"}. Not this package's when what is missing or stale was last moved by a take or by the world, or never moved and its pred is an in-port or a world reading: no rule of
    the member left it so (a projection's lie, a protocol's stale token). `whose` says which: a person's to read, never the loop's to fix."""
    from metis.fw.plugin import parse_do_line
    segs = read(path)
    if k < segs[0].k0:
        raise TraceError("tick %d is before the trace (it starts at %d: the ring)" % (k, segs[0].k0))
    start = next((x for x in reversed(segs) if x.k0 <= k and "cut" in x.header), segs[0])     # a new session's state follows from no row
    m = start.header["members"].get(member)
    if m is None:
        raise TraceError("no member %r in the trace" % member)
    counts, last = dict(start.checkpoint["sessions"][member]["counts"]), {}
    world_preds = {pred_of(a) for a in (start.checkpoint.get("copies") or {}).get(member, ())}
    events = {}
    if program:
        events = {ev.name: ev for _, _, ev in open_session(program, m.get("seed", 1)).events_tbl}

    def op(line, tick, by):
        sign, a, n = parse_do_line(line)
        counts[a] = counts.get(a, 0) + n if sign == "+" else max(0, counts.get(a, 0) - n)
        last[a] = {"k": tick, "by": by, "what": line}

    for seg in segs:
        if seg.k0 < start.k0:
            continue
        for t in seg.ticks:
            if t["k"] > k:
                break
            e = t["members"].get(member) or {}
            copied = e.get("copy") or {}
            world_preds |= {pred_of(a) for a in list(copied.get("+", ())) + list(copied.get("-", ()))}
            for item in e.get("seq") or ():
                for line in item.get("take", ()):
                    op(line, t["k"], item.get("by") or "take")
                for line in item.get("beat", ()):
                    op(line, t["k"], "world")
                for name in list(item.get("fired") or ()) + list(item.get("step") or ()):
                    ev = events.get(name)
                    if ev is None:
                        continue
                    for a, n in ev.consume.items():
                        counts[a] = max(0, counts.get(a, 0) - n)
                        last[a] = {"k": t["k"], "by": "rule", "what": name}
                    for a, n in ev.produce.items():
                        counts[a] = counts.get(a, 0) + n
                        last[a] = {"k": t["k"], "by": "rule", "what": name}
    in_preds, out, others = set(m.get("in") or ()), [], False
    for spec in atoms:
        a, _, n = str(spec).strip().partition(" ")
        want = int(n) if n else 1
        have = counts.get(a, 0)
        row = {"atom": a, "want": want, "have": have, "last": last.get(a), "whose": None}
        if have != want:
            mover = (last.get(a) or {}).get("by")
            if mover == "rule":
                row["whose"] = "the member's rule"
            elif mover == "world" or (mover is None and pred_of(a) in world_preds):
                row["whose"] = "the world's reading"
            elif mover in ("relay", "deliver", "take") or (mover is None and pred_of(a) in in_preds):
                row["whose"] = "a take from the director"
            elif mover is not None:
                row["whose"] = "from outside: %s" % mover
            else:
                row["whose"] = "the member's: nothing ever moved it"
            others = others or row["whose"] in ("the world's reading", "a take from the director")
        out.append(row)
    return {"atoms": out, "verdict": "not this package's" if others else "this package's"}


# ── the goldens: a refusal names them ──
def family_files(domain):
    """The cases of the family package of DOMAIN's name under METIS_HOME."""
    home = os.environ.get("METIS_HOME")
    family = Path(home) / Path(domain.dir).name / "cases" if home else None
    return sorted(family.glob("*.yaml")) if family and family.is_dir() and family.resolve() != (Path(domain.dir) / "cases").resolve() else []


def golden_files(domain, skip=()):
    """The package's cases, and those of its family (spec/18: a consumer runs its family's cases on its own program), but those named in SKIP."""
    from metis.catalog import stub
    return list(stub.cases_of(domain)) + [f for f in family_files(domain) if f.name not in set(skip)]


def _included(program):
    """The unit names PROGRAM includes, through its includes, wherever they resolve (its directory, then METIS_HOME)."""
    import re
    home, seen, todo = os.environ.get("METIS_HOME"), set(), [Path(program)]
    while todo:
        f = todo.pop()
        for name in re.findall(r'^include "([^"]+)"', f.read_text(), re.M):
            if name in seen:
                continue
            seen.add(name)
            for d in [f.parent] + ([Path(home)] if home else []):
                if (d / name).is_file():
                    todo.append(d / name)
                    break
    return {Path(n).name for n in seen}


def family_skips(domain):
    """The family's cases that are not this consumer's: [{"file", "why"}]. A family case is keyed to the unit it exercises, and no case says for whom: a case whose expected rule
    (expect.fired, an expect.out kind) is not in the consumer's program, or that lists `units` its program does not include, is skipped by name and listed, never counted as failing.
    Only a family's: a package's own case naming a rule it lacks is a real failure. Read on the program as it is: a candidate that drops a rule does not excuse the case that wants it."""
    import fnmatch
    from metis.catalog import stub
    files = family_files(domain)
    if not files:
        return []
    rules = {ev.name.split("[")[0] for _, _, ev in open_session(domain.program).events_tbl}
    units, out = _included(domain.program), []
    for f in files:
        c = stub.load_case(f)
        why = None
        missing = sorted(set(c.get("units") or ()) - units)
        if missing:
            why = "it exercises %s, which this program does not include" % ", ".join(missing)
        for b in c["boundaries"]:
            e = (b or {}).get("expect") or {}
            for r in e.get("fired") or ():
                why = why or (None if r in rules else "it expects %s, which this program does not have" % r)
            for x in e.get("out") or ():
                if isinstance(x, dict) and x.get("kind"):
                    why = why or (None if any(fnmatch.fnmatchcase(r, x["kind"]) for r in rules) else "it expects a command of kind %s, which this program does not have" % x["kind"])
        if why:
            out.append({"file": f.name, "why": why})
    return out


def goldens(domain, seeds=(1, 2, 3), skip=None, budget=None):
    """Every case of the package (golden_files), on each of SEEDS (one sampler's draw moves when a clause is merely enabled), but the family's that are not this consumer's
    (SKIP: their file names; default family_skips on the program as it is). BUDGET: seconds a run of one case may take. → (ok, failed): failed names each case, its seed, and what it
    says: a person may overrule a golden, so the refusal has to say which. A run past its budget says "not_judged": true: it neither passed nor failed."""
    from metis.catalog import stub
    failed = []
    if skip is None:
        skip = [x["file"] for x in family_skips(domain)]
    session, start = None, None
    for f in golden_files(domain, skip):
        c = stub.load_case(f)
        for seed in ([c.get("seed", 1)] if c.get("rng") is not None else seeds):
            try:
                with deadline(budget, Path(f).name):
                    if session is None and not c.get("ports"):     # one compile for the package: a big grounding takes seconds, and there are seeds times cases
                        session = stub.Stub(domain.program, domain.manifest).repl
                        start = session_state(session)
                    if session is not None:
                        restore_session(session, start)
                    r = stub.run_case(domain, f, seed=seed, session=session)
            except OutOfTime as x:
                failed.append({"case": c.get("case") or Path(f).stem, "file": Path(f).name, "seed": seed, "not_judged": True, "failures": [str(x)]})
                continue
            if not r.passed:
                failed.append({"case": r.name, "file": Path(f).name, "seed": seed,
                               "failures": [x for b in r.boundaries for x in b.failures][:4]})
    return not failed, failed


def apply_pack(unit, pack):
    """A hand-written PACK on a unit's TEXT: a whole program (it has `catalog`) replaces it; else its `drop NAME` lines take rules out and the rest is
    appended: its `pred` lines and `stage S { … }` blocks as written. → (text, {"whole", "added", "dropped"}). A drop of a rule the unit has not is said."""
    import re
    if re.search(r"^catalog\s", pack, re.M):
        return pack, {"whole": True, "added": sorted(rules_of(pack)), "dropped": sorted(set(rules_of(unit)) - set(rules_of(pack)))}
    lines, drops = [], []
    for line in pack.splitlines():
        m = re.match(r"^\s*drop\s+([A-Za-z_][A-Za-z0-9_]*)\s*\.?\s*$", line)
        if m:
            drops.append(m.group(1))
        else:
            lines.append(line)
    text = unit
    for name in drops:
        new = re.sub(r"(^[ \t]*%%[^\n]*\n)*^[ \t]*" + re.escape(name) + r"\s*(?:\[[^\]]*\])?\s*:[^.]*?-o[^.]*\.[^\n]*\n?", "", text, count=1, flags=re.M)
        if new == text:
            raise TraceError("the pack drops %s, which this unit does not have" % name)
        text = new
    body = "\n".join(lines).strip()
    return text.rstrip("\n") + "\n\n" + body + "\n" if body else text, {"whole": False, "added": sorted(rules_of(body)), "dropped": drops}


def duplicates(pack, texts):
    """The rules of PACK that repeat one of TEXTS' ({where: text}): [{rule, where, other, how}]: by name, or by the same premises and effects under another name."""
    def body(statement):
        return statement.split(":", 1)[1].strip().rstrip(".")
    out, mine = [], rules_of(pack)
    for name, st in mine.items():
        for where, text in texts.items():
            for other, o in rules_of(text).items():
                if other == name and body(o) != body(st):
                    continue                   # the same name otherwise: the compiler's rule-duplicate (or a replacement the pack drops)
                if other == name or body(o) == body(st):
                    out.append({"rule": name, "where": where, "other": other, "how": "the same rule" if other == name else "the same premises and effects"})
    return out


def holds(expect, counts, fired, outs=()):
    """A case's expectation at a tick of a replay: its failures. The stub's own `after` and `out` entries, on a session's COUNTS and the rules FIRED there."""
    from metis.catalog import stub
    res = stub.BoundaryResult(0, [], [], [], [{"event": x, "rule": x.split("[")[0], "p": None} for x in fired], list(outs), [], "running")
    return stub._score(res, {k: v for k, v in expect.items() if k in ("after", "out")}, counts)


def rules_of(text):
    """{rule: its statement}: what `new_rules` compares."""
    import re
    return {m.group(1): " ".join(m.group(0).split()) for m in re.finditer(r"^\s*([a-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?\s*:[^.]*?-o[^.]*\.", text, re.M)}


def new_rules(candidate, current):
    """The rules CANDIDATE has that CURRENT has not, or has otherwise."""
    a, b = rules_of(candidate), rules_of(current)
    return sorted(r for r, x in a.items() if b.get(r) != x)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m metis.trace", description="a composition's trace: one member's replay, a surprise as a case")
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("replay", help="one member alone over the trace, no world")
    r.add_argument("trace")
    r.add_argument("member")
    r.add_argument("program")
    c = sub.add_parser("case", help="the window to tick K as a case of the member's package")
    c.add_argument("trace")
    c.add_argument("member")
    c.add_argument("--at", type=int, required=True)
    c.add_argument("--from", dest="k0", type=int)
    c.add_argument("--after", action="append", default=[])
    c.add_argument("--out", action="append", default=[], help="an atom, or kind:PATTERN, not:PATTERN")
    c.add_argument("--note")
    c.add_argument("--unit", action="append", default=[], help="a family's case: a unit it exercises (a consumer without it skips the case)")
    c.add_argument("--into", help="the package's cases/: the file surprise-<member>-<K>.yaml there; else stdout")
    a = ap.parse_args(argv)
    if a.verb == "replay":
        res = replay(a.trace, a.member, a.program)
        print(json.dumps({"ok": res.ok, "ticks": res.ticks, "beats": res.beats, "rules": res.rules, "divergence": res.divergence}))
        return 0 if res.ok else 1
    out = [{x.split(":", 1)[0]: x.split(":", 1)[1]} if x.split(":", 1)[0] in ("kind", "not") else x for x in a.out]
    text = case(a.trace, a.member, a.at, after=a.after, out=out, k0=a.k0, note=a.note, units=a.unit)
    if a.into:
        f = Path(a.into) / ("surprise-%s-%d.yaml" % (a.member, a.at))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        print(f)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
