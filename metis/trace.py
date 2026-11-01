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
"""
import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

FORMAT = 1
STEPS = 200                         # the most rules a beat replays to quiescence: past it, a beat that never settles
GROUND_BUDGET = 1000000             # a session's events unless METIS_GROUND_BUDGET says: about 900 MB (NetHack's measured 248,254 events, 220 MB)


class TraceError(ValueError):
    pass


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
