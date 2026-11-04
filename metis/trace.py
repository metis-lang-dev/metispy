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

The instruments (metis-loop patterns/CHECKS.md §5), each from the trace alone, each printing rows a ruling can quote
(its notes, a header line, one row a line). A tick row carries no counts: `timeline` reconstructs a member's state from
each segment's checkpoint through the do-lines applied and, with --program, the rules' consume/produce (blame's accounting).

    timeline(dir, member, program=None)                  → ([Tick(k, counts, fired, made)], exact)
    python -m metis.trace blame DIR --member M --k K --atoms a,b [--program P]       who last moved what (Blame first)
    python -m metis.trace count DIR ATOM [--by ATOM ...] [--member M]                 the cross-tab (Measure the class first)
    python -m metis.trace families DIR [--family p,q,r ...] [--program P]             exactly-one per tick, and the flicker (Exclusive family; Steady reading)
    python -m metis.trace holds DIR ATOM --max N [--member M]                         windows held past N ticks, the rules within (Bounded yield)
    python -m metis.trace diff DIR_A DIR_B [--member M]                               the first tick the fired rows or the state differ (Sim race in the log)
    python -m metis.trace outcomes DIR --commit A --done B --cancel C [--member M]    each commitment window's end (Time to spare; Committed action)
    python -m metis.trace fired DIR RULE --seeds N --program P --member M             the seeds the rule fires on (Search the seeds for a firing)
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
STAND_IN = 0.5                      # a weight port's value when the trace does not carry it (its host, which binds it, is not run in a
                                    # replay): strictly inside (0, 1). At 1.0, ~p is 0, so the second clause of a choice site on the
                                    # port (p | ~p) is never enabled, and a record that chose it would read as a divergence; at 0, the first.
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
    note: Optional[str] = None                        # stood_in_note: the member's weight ports stood in

    @property
    def ok(self):
        return self.divergence is None


def open_session(program, seed=1, weights=()):
    """A session on PROGRAM, under a grounding budget: a rule that grounds past it is refused in its own words
    (metis.kernel.ground.GroundBudget), before a replay eats the memory scope. WEIGHTS: the member's weight ports the
    trace does not carry (its header's `weights`), each stood in at STAND_IN so the program opens (stood_in_note says so)."""
    from metis.lang.repl import Repl as Shell
    kw = {}
    if weights:
        from metis.fw.registry import Registry
        reg = Registry()
        for w in weights:
            reg.weight(str(w), STAND_IN)
        kw["registry"] = reg
    set_here = "METIS_GROUND_BUDGET" not in os.environ
    if set_here:
        os.environ["METIS_GROUND_BUDGET"] = str(GROUND_BUDGET)
    try:
        return Shell(str(program), seed=seed, stdout=io.StringIO(), **kw)
    finally:
        if set_here:
            del os.environ["METIS_GROUND_BUDGET"]


def weights_in(header, member):
    """MEMBER's weight ports as a segment's HEADER names them (none: [])."""
    return list(((header or {}).get("members") or {}).get(member, {}).get("weights") or ())


def stood_in_note(weights):
    """The one line every verb says for a member opened with its weight ports stood in, or None."""
    if not weights:
        return None
    return "%s stood in at %s: this member's choices are the record's, not evidence" % (", ".join(weights), STAND_IN)


def trace_note(path, member):
    """stood_in_note for MEMBER of the trace at PATH, as a list of notes ([] when it declares no weight port)."""
    n = stood_in_note(weights_in(read(path)[0].header, member))
    return [n] if n else []


def force_recorded(sh, queue):
    """A member opened with its weight ports stood in draws the record's choice: its next sampled firing is QUEUE's next
    rule when that rule is enabled (one draw of the stream spent, as the record spent it), else its own (and a replay
    then says where it left the record)."""
    own = sh._sample

    def sample():
        ws = sh._candidates()
        if queue and ws:
            hit = next((i for i, _ in ws if sh.events_tbl[i][2].name == queue[0]), None)
            if hit is not None:
                queue.pop(0)
                sh.rng.unit()
                return sh._fire(hit, "sampled", ws)
        return own()
    sh._sample = sample


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
    weights = list(m.get("weights") or ())
    have = seg.header.get("protocol")
    if have and have != protocol_key():
        raise TraceError("this trace is of another protocol (%s, this toolchain's %s): its checkpoints lack what changed since; "
                         "it does not replay across a protocol change" % (have, protocol_key()))
    if check_key and m.get("key") and m["key"] != program_key(program):
        raise TraceError("%s: %s is not the program the trace was written from" % (member, program))
    commands, out_preds = seg.header.get("commands") or (), set(m.get("out") or ())
    sh = open_session(program, m.get("seed", 1), weights)
    restore_session(sh, seg.checkpoint["sessions"][member])
    events = {ev.name: ev for _, _, ev in sh.events_tbl}
    queue = []                                     # the record's draws, forced where enabled (a weight port stood in)
    if weights:
        r.note = stood_in_note(weights)
        force_recorded(sh, queue)

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
                queue[:] = list(want) if weights else []
                got = _fire_all(sh, want, item.get("left"))
                r.beats += 1
                r.rules += len(got)
                if got != want:
                    return stop(k, "fired", want, got)
                fired += got
            elif "step" in item:
                queue[:] = list(item["step"]) if weights else []
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


def rule_base(name):
    """A fired event's rule name: an alternative head fires as <name>-<k>, a ground event as name[binding]."""
    return name.split("[", 1)[0].rsplit("-", 1)[0] if "[" in name or "-" in name else name


def generated_rules(doc):
    """The rule names inside the protocol stub's generated block (BEGIN … END marks) of each source file of the compiled DOC: the protocol's rules, which a member's author never
    wrote (Stub-owned echo). Read from the files the rules' locations name; a file that cannot be read contributes nothing."""
    from metis.catalog.protocol_stub import END, MARK
    from metis.lang import ast
    rules = [r for d in (doc.source.decls if doc.source else ()) if isinstance(d, ast.Stage) for r in d.rules if r.file and r.line]
    spans = {}
    for f in {r.file for r in rules}:
        try:
            lines = open(f).read().split("\n")
        except OSError:
            continue
        begun = [k for k, line in enumerate(lines) if line.startswith(MARK)]
        if begun and END in lines[begun[0]:]:
            spans[f] = (begun[0] + 1, lines.index(END, begun[0]) + 1)
    return {ast.rule_base(r.name) for r in rules if r.file in spans and spans[r.file][0] <= r.line <= spans[r.file][1]}


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
    program = session_program(start, member, program)
    weights = list(m.get("weights") or ())
    counts, last = dict(start.checkpoint["sessions"][member]["counts"]), {}
    world_preds = {pred_of(a) for a in (start.checkpoint.get("copies") or {}).get(member, ())}
    events, generated = {}, set()
    if program:
        sh = open_session(program, m.get("seed", 1), weights)
        events = {ev.name: ev for _, _, ev in sh.events_tbl}
        generated = generated_rules(sh.doc)

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
            if mover == "rule" and rule_base((last.get(a) or {}).get("what") or "") in generated:
                # patterns open item 10: a rule inside the stub's generated block is the protocol's, not the member's
                row["whose"] = "the protocol's generated block (the stub's, not the member's)"
            elif mover == "rule":
                row["whose"] = "the member's rule"
            elif mover == "world" or (mover is None and pred_of(a) in world_preds):
                row["whose"] = "the world's reading"
            elif mover in ("relay", "deliver", "take") or (mover is None and pred_of(a) in in_preds):
                row["whose"] = "a take from the director"
            elif mover is not None:
                row["whose"] = "from outside: %s" % mover
            else:
                row["whose"] = "the member's: nothing ever moved it"
            others = others or row["whose"] in ("the world's reading", "a take from the director", "the protocol's generated block (the stub's, not the member's)")
        out.append(row)
    res = {"atoms": out, "verdict": "not this package's" if others else "this package's"}
    if weights:
        res["note"] = stood_in_note(weights)
    return res


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


# ── the trace instruments (metis-loop patterns/CHECKS.md §5): rows a ruling can quote ──
@dataclass
class Table:
    """Rows to print: NOTES (summary lines) first, then the HEADER line, then one row a line, columns padded. A cell that is a
    list, set or tuple is comma-joined, an empty one or None is `-`, a bool is yes/no."""
    header: List[str]
    rows: List[list] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @staticmethod
    def cell(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (list, tuple, set, frozenset)):
            return ",".join(str(x) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)) or "-"
        return str(v)

    def lines(self):
        grid = [[str(h) for h in self.header]] + [[self.cell(v) for v in r] for r in self.rows]
        widths = [max(len(r[i]) for r in grid if i < len(r)) for i in range(max(len(r) for r in grid))]
        out = list(self.notes)
        for r in grid:
            out.append("  ".join(c.ljust(widths[i]) if i < len(r) - 1 else c for i, c in enumerate(r)).rstrip())
        return out

    def __str__(self):
        return "\n".join(self.lines())


@dataclass
class Tick:
    """One tick of a member as `timeline` reconstructs it: its COUNTS after the tick, the rules FIRED at it (full names, in order), and MADE: the atoms that went up during it
    (a take or beat's `+`, a rule's produce), so an atom consumed again within the tick is still seen to have appeared."""
    k: int
    counts: dict
    fired: list = field(default_factory=list)
    made: set = field(default_factory=set)


def timeline(path, member, program=None):
    """MEMBER's state at every tick of the trace at PATH, reconstructed: a tick row carries no counts (only each segment's checkpoint does), so the counts are replayed from
    the segment's checkpoint through the do-lines applied (takes, beats) and, when PROGRAM is given, the consume/produce of each rule fired (blame's accounting; without a
    program the rules' effects are unknown and the state is exact only at each segment's start). The counts are re-synced at every segment's checkpoint. A row repeating a
    tick's k (`between`) merges into it. → (ticks: [Tick], exact: whether the rules' effects were applied)."""
    from metis.fw.plugin import parse_do_line
    segs = read(path)
    m = segs[0].header["members"].get(member)
    if m is None:
        raise TraceError("no member %r in the trace (it has: %s)" % (member, ", ".join(sorted(segs[0].header["members"]))))
    program = session_program(segs[0], member, program)
    events = {}
    if program:
        events = {ev.name: ev for _, _, ev in open_session(program, m.get("seed", 1), m.get("weights") or ()).events_tbl}
    ticks = []
    for seg in segs:
        counts = dict((seg.checkpoint.get("sessions") or {}).get(member, {}).get("counts") or {})
        for t in seg.ticks:
            k = t["k"]
            if ticks and ticks[-1].k == k:
                cur = ticks[-1]
            else:
                cur = Tick(k, {})
                ticks.append(cur)
            e = t["members"].get(member) or {}
            for item in e.get("seq") or ():
                for line in list(item.get("take", ())) + list(item.get("beat", ())):
                    sign, a, n = parse_do_line(line)
                    if sign == "+":
                        counts[a] = counts.get(a, 0) + n
                        cur.made.add(a)
                    else:
                        counts[a] = max(0, counts.get(a, 0) - n)
                for name in list(item.get("fired") or ()) + list(item.get("step") or ()):
                    cur.fired.append(name)
                    ev = events.get(name)
                    if ev is None:
                        continue
                    for a, n in ev.consume.items():
                        counts[a] = max(0, counts.get(a, 0) - n)
                    for a, n in ev.produce.items():
                        counts[a] = counts.get(a, 0) + n
                        cur.made.add(a)
            cur.counts = {a: n for a, n in counts.items() if n > 0}
    return ticks, bool(program)


def held(spec, counts):
    """The ground instances of SPEC present in COUNTS: SPEC with parentheses is one exact atom; a bare name is a pred, every instance of it."""
    if "(" in spec:
        return [spec] if counts.get(spec, 0) > 0 else []
    return sorted(a for a, n in counts.items() if n > 0 and pred_of(a) == spec)


def _windows(ticks, spec):
    """The maximal runs of consecutive ticks on which SPEC holds: [(first index, last index)]."""
    out, start = [], None
    for i, t in enumerate(ticks):
        if held(spec, t.counts):
            if start is None:
                start = i
        elif start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(ticks) - 1))
    return out


def _toggles(labels, least=4):
    """The runs of indices where LABELS alternate with period <= 2 (a-b-a-b: each differs from the one before and equals the one two back) over more than LEAST ticks:
    [(first, last)]."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        alternating = i < len(labels) and labels[i] != labels[i - 1] and (i - start < 2 or labels[i] == labels[i - 2])
        if not alternating:
            if i - start > least:
                out.append((start, i - 1))
            start = i if i >= len(labels) or labels[i] == labels[i - 1] else i - 1      # a b c: the run b c b c may start at b
    return out


SESSION_LIVE = "metis-loop-console"          # the header's `live` of a console REPL session's trace (metis-loop SessionTrace)


def session_program(seg, member, program=None):
    """The program the verbs read a console session's trace with: PROGRAM when given, else the member's own from the
    header when the trace is a REPL session's (live: metis-loop-console) and the file is there. A session steps the
    rules by tool with no world, so without its program the rules' effects are unknown (the alf drive: /why said
    'never moved' for an atom a rule made). A composition's trace keeps the program it is given."""
    if program or (seg.header or {}).get("live") != SESSION_LIVE:
        return program
    p = ((seg.header.get("members") or {}).get(member) or {}).get("program")
    return p if p and Path(p).is_file() else None


def _note_state(exact):
    return "state: from each segment's checkpoint through the do-lines and the rules' effects" if exact else \
        "state: from each segment's checkpoint through the do-lines only (no --program: the rules' effects are not applied)"


def blame_rows(path, member, k, atoms, program=None):
    """Blame first: the CLI of `blame`, its result as rows (whose / atom / want / have / last mover) under the verdict line, nothing new in the logic. → Table."""
    r = blame(path, member, k, atoms, program)
    t = Table(["whose", "atom", "want", "have", "last_k", "last_by", "last_what"], notes=["verdict: %s" % r["verdict"]] + ([r["note"]] if r.get("note") else []))
    for row in r["atoms"]:
        last = row["last"] or {}
        t.rows.append([row["whose"] or "nobody's: it holds as it should", row["atom"], row["want"], row["have"], last.get("k"), last.get("by"), last.get("what")])
    return t


def count(path, atom, by=(), member=None, program=None):
    """Measure the class first; A norm with its domain: the cross-tab of the ticks on which ATOM holds (count > 0: any instance of a pred name, or one exact ground atom) by
    the value of each BY atom at the same tick (an exact atom: holds / absent; a pred name: the instances present, or absent). The notes say the ticks and those with ATOM;
    one row per by-combination, with the ticks with and without ATOM in it, most first. → Table."""
    ticks, exact = timeline(path, member or _only_member(path), program)
    table = {}
    for t in ticks:
        key = tuple(_value(b, t.counts) for b in by)
        with_, without = table.get(key, (0, 0))
        table[key] = (with_ + 1, without) if held(atom, t.counts) else (with_, without + 1)
    total = len(ticks)
    n = sum(w for w, _ in table.values())
    out = Table(list(by) + ["with_%s" % atom, "without"], notes=["ticks: %d" % total, "ticks with %s: %d" % (atom, n), _note_state(exact)] + trace_note(path, member or _only_member(path)))
    for key, (w, wo) in sorted(table.items(), key=lambda kv: (-kv[1][0], kv[0])):
        out.rows.append(list(key) + [w, wo])
    return out


def _value(spec, counts):
    inst = held(spec, counts)
    if not inst:
        return "absent"
    return "holds" if inst == [spec] else ",".join(inst)


def _only_member(path):
    """The member when none is named: the trace's one member, else a refusal naming them."""
    members = sorted(read(path)[0].header["members"])
    if len(members) != 1:
        raise TraceError("--member: the trace has %s" % ", ".join(members))
    return members[0]


def families(path, fams=(), member=None, program=None):
    """Exclusive family + Steady reading: for each family (a set of pred names, from --family and from the program's declared `families` when it has them) whether EXACTLY ONE
    member holds at every tick: a summary row per family (ticks ok / zero / two_or_more), one row per violating tick with what held; then the flicker rows: a family whose
    held member, or a rule that fired, alternates with period <= 2 for more than 4 consecutive ticks (a-b-a-b), one row per run with its first and last tick and the pair.
    → [Table summary, Table violations, Table flicker]."""
    member = member or _only_member(path)
    ticks, exact = timeline(path, member, program)
    fams = [("+".join(f), list(f)) for f in fams]
    if program:
        doc = getattr(open_session(program, 1, weights_in(read(path)[0].header, member)), "doc", None)
        for name, members in getattr(doc, "families", None) or ():
            fams.append((str(getattr(name, "name", name)), [str(getattr(x, "name", x)) for x in members]))
    summary = Table(["family", "members", "ticks", "ok", "zero", "two_or_more"], notes=["ticks: %d" % len(ticks), _note_state(exact)] + trace_note(path, member))
    bad = Table(["family", "k", "held"])
    flicker = Table(["what", "name", "first_k", "last_k", "ticks", "pair"])
    for name, members in fams:
        ok = zero = many = 0
        labels = []
        for t in ticks:
            present = [m for m in members if held(m, t.counts)]
            labels.append(tuple(present))
            if len(present) == 1:
                ok += 1
            elif not present:
                zero += 1
                bad.rows.append([name, t.k, "none"])
            else:
                many += 1
                bad.rows.append([name, t.k, present])
        summary.rows.append([name, members, len(ticks), ok, zero, many])
        for a, b in _toggles(labels):
            pair = sorted({labels[a], labels[a + 1]}, key=str)
            flicker.rows.append(["family", name, ticks[a].k, ticks[b].k, b - a + 1, " <-> ".join("+".join(p) or "none" for p in pair)])
    names = sorted({pred_of(x) for t in ticks for x in t.fired})
    fired = {n: [any(pred_of(x) == n for x in t.fired) for t in ticks] for n in names}
    for n in names:
        for a, b in _toggles(fired[n]):
            other = next((o for o in names if o != n and all(fired[o][i] != fired[n][i] for i in range(a, b + 1))), None)
            if other and other < n and (a, b) in _toggles(fired[other]):
                continue                                  # the same run, already told from its partner
            flicker.rows.append(["rule", n, ticks[a].k, ticks[b].k, b - a + 1, "%s <-> %s" % (n, other or "idle")])
    return [summary, bad, flicker]


def holds_windows(path, atom, max_ticks, member=None, program=None):
    """Bounded yield: every window where ATOM is held (count > 0) for more than MAX_TICKS consecutive ticks; one row per window with its first and last tick, its length and the
    rules (names) that fired on the member within it, so a reader sees whether a release ever fired. → Table."""
    ticks, exact = timeline(path, member or _only_member(path), program)
    out = Table(["first_k", "last_k", "ticks", "rules_fired"], notes=["ticks: %d" % len(ticks), "windows of %s past %d ticks:" % (atom, max_ticks), _note_state(exact)] + trace_note(path, member or _only_member(path)))
    for a, b in _windows(ticks, atom):
        if b - a + 1 > max_ticks:
            out.rows.append([ticks[a].k, ticks[b].k, b - a + 1, sorted({pred_of(x) for t in ticks[a:b + 1] for x in t.fired})])
    out.notes[1] = "windows of %s past %d ticks: %d" % (atom, max_ticks, len(out.rows))
    return out


def diff(path_a, path_b, member=None, program=None):
    """Sim race in the log: two runs of the same command, tick by tick: the first tick where the fired rules differ and the first where the state atoms differ, both sides'
    rows at that tick (for the state, the atoms one side has and the other has not); identical: said, with the tick count. Every member, or MEMBER. → [Table fired, Table state]."""
    members = [member] if member else sorted(read(path_a)[0].header["members"])
    fired = Table(["side", "k", "member", "fired"])
    state = Table(["side", "k", "member", "atoms_only_here"])
    tl = {m: (timeline(path_a, m, program)[0], timeline(path_b, m, program)[0]) for m in members}
    n = min(len(a) for a, _ in tl.values()), min(len(b) for _, b in tl.values())
    for table, what in ((fired, "fired"), (state, "state")):
        found = None
        for i in range(min(n)):
            for m in members:
                a, b = tl[m][0][i], tl[m][1][i]
                if (a.fired != b.fired) if what == "fired" else (a.counts != b.counts):
                    found = (i, m)
                    break
            if found:
                i, m = found
                a, b = tl[m][0][i], tl[m][1][i]
                table.notes.append("%s differ first at tick %s (%s)" % (what, a.k, m))
                if what == "fired":
                    table.rows += [["A", a.k, m, a.fired], ["B", b.k, m, b.fired]]
                else:
                    only = lambda x, y: sorted(set(x.counts.items()) - set(y.counts.items()))
                    table.rows += [["A", a.k, m, ["%s %d" % kv for kv in only(a, b)]], ["B", b.k, m, ["%s %d" % kv for kv in only(b, a)]]]
                break
        if not found:
            table.notes.append("%s identical over %d ticks" % (what, min(n)) + ("" if n[0] == n[1] else " (A has %d ticks, B has %d)" % n))
    for m in members:
        fired.notes += ["%s: %s" % (m, x) for x in trace_note(path_a, m)]
    return [fired, state]


def outcomes(path, commit, done, cancel, member=None, program=None):
    """Time to spare / Committed action: for a commitment atom COMMIT (held while an action runs), its windows and how each ended: with DONE appearing (at the window's last tick
    or the one after), with CANCEL appearing, or with neither (the trace ended on it, or it was released otherwise: the row says which). Three counts in the notes, one row per
    window. → Table."""
    ticks, exact = timeline(path, member or _only_member(path), program)
    out = Table(["first_k", "last_k", "ticks", "ended"], notes=[_note_state(exact)] + trace_note(path, member or _only_member(path)))
    tally = {"done": 0, "cancel": 0, "neither": 0}
    for a, b in _windows(ticks, commit):
        end = ticks[b:b + 2]
        seen = lambda spec: any(held(spec, t.counts) or any(pred_of(x) == spec or x == spec for x in t.made) for t in end)
        if seen(done):
            how = "done"
        elif seen(cancel):
            how = "cancel"
        else:
            how = "neither: the trace ended" if b == len(ticks) - 1 else "neither"
        tally[how.split(":")[0]] += 1
        out.rows.append([ticks[a].k, ticks[b].k, b - a + 1, how])
    out.notes[:0] = ["windows of %s: %d" % (commit, len(out.rows)), "done: %d" % tally["done"], "cancel: %d" % tally["cancel"], "neither: %d" % tally["neither"]]
    return out


def fired_on_seeds(path, rule, seeds, program, member):
    """Search the seeds for a firing: MEMBER replayed over the trace on PROGRAM once per seed 1..SEEDS (the session opened on that seed, its counts and stage from the checkpoint,
    not its rng), the recorded inputs applied beat by beat and the rules fired to quiescence as `replay` fires them, without holding the run to the recorded firings: a seed
    moves the sampler, so the run may leave the recorded one. One row per seed: whether RULE (a rule's name, or one firing with its bindings) fired, and the first tick. The
    program's key is not checked: a candidate's rule is what one searches for. → Table."""
    segs = read(path)
    m = segs[0].header["members"].get(member)
    if m is None:
        raise TraceError("no member %r in the trace (it has: %s)" % (member, ", ".join(sorted(segs[0].header["members"]))))
    out = Table(["seed", "fired", "first_k"])
    for seed in range(1, seeds + 1):
        sh, first = None, None
        for seg in segs:
            if sh is None or "cut" in seg.header:
                sh = open_session(program, seed, m.get("weights") or ())
                st = seg.checkpoint["sessions"][member]
                sh.stage, sh.counts = st["stage"], dict(st["counts"])
            for t in seg.ticks:
                e = t["members"].get(member)
                if e is None or first is not None:
                    continue
                seq = e.get("seq")
                if seq is None:
                    seq = [{"beat": [], "fired": []}] if "why" in e else []
                for item in seq:
                    if "take" in item:
                        _apply(sh, item["take"])
                        got = []
                    elif "beat" in item:
                        if item["beat"]:
                            _apply(sh, item["beat"])
                        got = _fire_all(sh, item.get("fired") or [], item.get("left"))
                    elif "step" in item:
                        got = [sh._sample() for _ in item["step"]]
                    else:
                        got = []
                    if any(x and (x == rule or rule_base(x) == rule) for x in got):    # a ground event is go[P=door]
                        first = t["k"]
                        break
        out.rows.append([seed, first is not None, first])
    out.notes.append("%s fired on %d of %d seeds" % (rule, sum(1 for r in out.rows if r[1]), seeds))
    out.notes += trace_note(path, member)
    return out



# ── sort: each response violation of a run, by why it is one (pascal's road sort, lifted: CHECKS.md §5; checks-open s-17e557f0) ──
SORT_CLASSES = {1: "checker", 2: "projection", 3: "competing", 4: "responder", 5: "gap", 6: "unseen, the other's duty"}
SORT_BOUND = {"brake_min": -4.0, "brake_min_correct": -3.0}     # the road's passenger ego: the bound each response gives
SORT_RECENT = 60                                                 # rows looked back for a release's lane change (settled_to holds 2 s)


def _sort_bound(pairs, bounds):
    return min((bounds[p["resp"]] for p in pairs if p["resp"] in bounds), default=None)


def sort_violations(rows, bounds=SORT_BOUND):
    """(the row a command was decided on, the row it was applied in), where the applied acceleration is above the bound of the TRUTH's
    dangerous pairs and the car moves (a standing car meets any brake). ROWS: the log's world rows, in order."""
    return [(a, b) for a, b in zip(rows, rows[1:]) if b["a"] is not None and _sort_bound(a["truth"], bounds) is not None
            and b["v"] > 0 and b["a"] > _sort_bound(a["truth"], bounds) + 1e-9]


def _edge(lane):
    return lane.rsplit("_", 1)[0]


def changed_lane(rows, pair):
    """Did the truth see PAIR's body change lanes on one road in ROWS? True, False, or None when no row names it."""
    car = pair["id"].split(":")[0]
    seen = [p["lane"] for r in rows for p in r["truth"] if p["id"].split(":")[0] == car and p["lane"]]
    return any(a != b and _edge(a) == _edge(b) for a, b in zip(seen, seen[1:] + [pair["lane"]])) if seen else None


def sort_row(before, row, facts, recent=(), bounds=SORT_BOUND, danger_pred="resp_lon", release_pred="settled_to",
             holder=":xing_ego_prio"):
    """One violation → (class, the reason in words). FACTS: the member's atoms once BEFORE's projection reached it; RECENT: the rows
    before BEFORE. Class 2 is the sorter's call on the truth rows: it says what the belief lacked, not why. HOLDER marks a pair id
    whose way is the ego's (class 6, the other's duty when the sensor does not see it)."""
    ids = lambda r: ", ".join(p["id"] for p in r["truth"] if p["resp"] in bounds)
    cmd = row["cmd"][0] if row["cmd"] else None
    braking = [a for a in facts if a.startswith("%s(brake" % danger_pred)]
    believed = [p for p in before["danger"] if p["resp"] in bounds]
    seen = before.get("seen_ids")
    bound_pairs = [p for p in before["truth"] if p["resp"] in bounds]
    if seen is not None and bound_pairs and all(holder in p["id"] and p["id"].split(":")[0] not in seen for p in bound_pairs):
        return 6, ("the truth has %s dangerous, the ego has the way over it and the sensor does not see it: the yielder's violation, "
                   "listed, as sumo_host.account does" % ids(before))
    if cmd == "brake" and row["cmd"][1] in bounds:
        return 4, "the command is the shield's, and the acceleration is above the bound only by rounding or the car's limit"
    if braking:
        return 3, "the responder's fact held (%s) and the command is %s: another rule held the controls" % (braking[0], cmd)
    if believed:
        settled = [a for a in facts if a.startswith("%s(%s," % (release_pred, believed[0]["role"]))]
        real = next((p for p in before["truth"] if (p["lane"], p["s"]) == (believed[0]["lane"], believed[0]["s"])), None)
        if settled and real is not None and changed_lane(recent, real) is False:
            return 2, "the fold released %s on %s, and %s is false: the truth saw that car change no lane" % (believed[0]["id"], settled[0], settled[0])
        if settled:
            return 1, ("the belief has %s dangerous, the fold released it (%s): the projection's ruled release, which the truth's fold "
                       "does not have" % (believed[0]["id"], settled[0]))
        return 5, "the belief has %s dangerous, no response in the facts and no release: nothing read it" % believed[0]["id"]
    return 2, "the truth has %s dangerous, the belief no dangerous pair: the projection's facts are false against the world" % ids(before)


def member_atoms(path, member, ticks):
    """MEMBER's atoms at each of TICKS: its checkpoint, then its takes and beats, tick by tick."""
    wanted, out = set(ticks), {}
    segs = read(path)
    counts = dict(segs[0].checkpoint["sessions"][member]["counts"])
    for seg in segs:
        for t in seg.ticks:
            for item in (t["members"].get(member) or {}).get("seq") or ():
                for line in list(item.get("take", ())) + list(item.get("beat", ())):
                    if line[0] == "+":
                        counts[line[1:].split(" ")[0]] = 1
                    else:
                        counts.pop(line[1:].split(" ")[0], None)
            if t["k"] in wanted:
                out[t["k"]] = sorted(counts)
    return out


def sort(path, log, member="responder", shift=1, bounds=SORT_BOUND, danger_pred="resp_lon", release_pred="settled_to",
         holder=":xing_ego_prio", cases=None):
    """The verdict rows of a run, one per response violation: {tick, norm, class, name, why, command, a, v, zone, truth, belief}.
    LOG: the world's rows (each with tick, cmd, a, v, zone, danger and truth: pairs {id, role, resp, lane, s}, and seen_ids); the
    facts are MEMBER's on the trace at the decided row's tick + SHIFT (a row's facts reach the trace when the member beats on them).
    CASES: a directory: a surprise case per violation, the member's `brake_lon*` asked at the tick."""
    rows = [r for r in map(json.loads, open(log)) if "resp" in r]
    found = sort_violations(rows, bounds)
    facts = member_atoms(path, member, [a["tick"] + shift for a, _ in found])
    at = {r["tick"]: i for i, r in enumerate(rows)}
    out = []
    for a, b in found:
        n, why = sort_row(a, b, facts.get(a["tick"] + shift, ()), rows[max(0, at[a["tick"]] - SORT_RECENT):at[a["tick"]]], bounds,
                          danger_pred, release_pred, holder)
        out.append({"tick": b["tick"], "norm": "rss_response", "class": n, "name": SORT_CLASSES[n], "why": why, "command": b["cmd"],
                    "a": b["a"], "v": a["v"], "zone": a["zone"], "truth": [p["id"] for p in a["truth"] if p["resp"] in bounds],
                    "belief": [p["id"] for p in a["danger"] if p["resp"] in bounds]})
    if cases:
        Path(cases).mkdir(parents=True, exist_ok=True)
        for r in out:
            (Path(cases) / ("surprise-%d.yaml" % r["tick"])).write_text(
                case(path, member, r["tick"], out=["brake_lon*"], note="class %d %s: %s" % (r["class"], r["name"], r["why"])))
    return out


def sort_summary(rows):
    return {"violations": len(rows), "by_class": {SORT_CLASSES[n]: sum(r["class"] == n for r in rows) for n in SORT_CLASSES}}


def parser():
    """The verbs' argparse: the CLI's, and the one source of metis/trace_tools.json (tools/gen_trace_tools.py)."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m metis.trace", description="a composition's trace: one member's replay, a surprise as a case, and the instruments "
                                 "(blame, count, families, holds, diff, outcomes, fired): rows a ruling can quote, from the trace alone")
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
    c.add_argument("--into", help="NAME.yaml: that file; any other name: the package's cases/ directory, the file "
                                  "surprise-<member>-<K>.yaml there; else stdout")
    # the instruments (CHECKS.md §5): each prints its notes, a header line, then one row a line
    state_help = "the member's .llp: with it the rules' effects are applied to the state; without, only the do-lines from each segment's checkpoint"
    b = sub.add_parser("blame", help="who last moved what is missing or stale at tick K: the member's rule, the world, or a take (Blame first)")
    b.add_argument("trace")
    b.add_argument("--member", required=True)
    b.add_argument("--k", type=int, required=True)
    b.add_argument("--atoms", required=True, help="comma-separated: `atom` should hold, `atom 0` should not, `atom 3` should hold thrice")
    b.add_argument("--program", help=state_help)
    n = sub.add_parser("count", help="the cross-tab: the ticks on which ATOM holds, by the value of each --by atom at the same tick (Measure the class first; A norm with its domain)")
    n.add_argument("trace")
    n.add_argument("atom", help="a pred name (any instance), or one ground atom with its parentheses")
    n.add_argument("--by", action="append", default=[], help="an atom to tabulate by: holds / absent, or a pred's instances present")
    n.add_argument("--member", help="the member whose state is read (the trace's one member when it has one)")
    n.add_argument("--program", help=state_help)
    f = sub.add_parser("families", help="exactly one of each family holds at every tick, and the flicker: a family or a rule toggling a-b-a-b (Exclusive family; Steady reading)")
    f.add_argument("trace")
    f.add_argument("--family", action="append", default=[], help="p,q,r: pred names of which exactly one holds; the program's declared families are read too")
    f.add_argument("--member")
    f.add_argument("--program", help=state_help)
    h = sub.add_parser("holds", help="every window where ATOM is held past N consecutive ticks, with the rules that fired within it (Bounded yield)")
    h.add_argument("trace")
    h.add_argument("atom")
    h.add_argument("--max", type=int, required=True, help="the ticks a hold may last")
    h.add_argument("--member")
    h.add_argument("--program", help=state_help)
    d = sub.add_parser("diff", help="two runs of the same command: the first tick where the fired rules differ, and where the state differs (Sim race in the log)")
    d.add_argument("trace_a")
    d.add_argument("trace_b")
    d.add_argument("--member", help="one member; else every member of the trace")
    d.add_argument("--program", help=state_help)
    o = sub.add_parser("outcomes", help="the windows of a commitment atom and how each ended: done, cancel, or neither (Time to spare; Committed action)")
    o.add_argument("trace")
    o.add_argument("--commit", required=True, help="the atom held while the action runs")
    o.add_argument("--done", required=True)
    o.add_argument("--cancel", required=True)
    o.add_argument("--member")
    o.add_argument("--program", help=state_help)
    so = sub.add_parser("sort", help="each response violation of a run's log, by why it is one: checker, projection, competing, responder, gap, unseen (Sort before rule)")
    so.add_argument("trace")
    so.add_argument("log", help="the world's log: rows with tick, cmd, a, v, zone, danger, truth, seen_ids")
    so.add_argument("--member", default="responder")
    so.add_argument("--shift", type=int, default=1, help="ticks from a row to its facts on the trace (the road's: 1)")
    so.add_argument("--bound", action="append", default=[], metavar="RESP=A", help="a response and its acceleration bound (the road's: brake_min=-4, brake_min_correct=-3)")
    so.add_argument("--danger-pred", default="resp_lon")
    so.add_argument("--release-pred", default="settled_to")
    so.add_argument("--holder", default=":xing_ego_prio", help="the mark in a pair id of a way the ego holds")
    so.add_argument("--cases", help="a directory: a surprise case per violation")
    s = sub.add_parser("fired", help="how many of seeds 1..N the rule fires on when the member is replayed over the trace's inputs (Search the seeds for a firing)")
    s.add_argument("trace")
    s.add_argument("rule", help="a rule's name, or one firing with its bindings: rule[X=a]")
    s.add_argument("--seeds", type=int, required=True)
    s.add_argument("--program", required=True)
    s.add_argument("--member", required=True)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    if a.verb == "replay":
        res = replay(a.trace, a.member, a.program)
        print(json.dumps({"ok": res.ok, "ticks": res.ticks, "beats": res.beats, "rules": res.rules, "divergence": res.divergence,
                          **({"note": res.note} if res.note else {})}))
        return 0 if res.ok else 1
    tables = None
    if a.verb == "blame":
        tables = [blame_rows(a.trace, a.member, a.k, [x.strip() for x in a.atoms.split(",") if x.strip()], a.program)]
    elif a.verb == "count":
        tables = [count(a.trace, a.atom, a.by, a.member, a.program)]
    elif a.verb == "families":
        tables = families(a.trace, [[x.strip() for x in fam.split(",") if x.strip()] for fam in a.family], a.member, a.program)
    elif a.verb == "holds":
        tables = [holds_windows(a.trace, a.atom, a.max, a.member, a.program)]
    elif a.verb == "diff":
        tables = diff(a.trace_a, a.trace_b, a.member, a.program)
    elif a.verb == "outcomes":
        tables = [outcomes(a.trace, a.commit, a.done, a.cancel, a.member, a.program)]
    elif a.verb == "sort":
        bounds = {k: float(v) for k, _, v in (b.partition("=") for b in a.bound)} or SORT_BOUND
        rows = sort(a.trace, a.log, a.member, a.shift, bounds, a.danger_pred, a.release_pred, a.holder, a.cases)
        for r in rows:
            print(json.dumps(r))
        print(json.dumps(sort_summary(rows)))
        return 0
    elif a.verb == "fired":
        tables = [fired_on_seeds(a.trace, a.rule, a.seeds, a.program, a.member)]
    if tables is not None:
        print("\n\n".join(str(t) for t in tables))
        return 0
    out = [{x.split(":", 1)[0]: x.split(":", 1)[1]} if x.split(":", 1)[0] in ("kind", "not") else x for x in a.out]
    text = case(a.trace, a.member, a.at, after=a.after, out=out, k0=a.k0, note=a.note, units=a.unit)
    if a.into:
        f = Path(a.into)
        if f.suffix == ".yaml":                   # a file's name: that file (the pi loop gave one, and got a directory)
            if f.is_dir():
                import sys
                print("metis.trace case: --into %s is a directory, not the case's file: name a NAME.yaml that is "
                      "not a directory, or give the directory itself" % f, file=sys.stderr)
                return 2
        else:
            f = f / ("surprise-%s-%d.yaml" % (a.member, a.at))
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        print(f)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
