"""Generator for the live goldens (runtime-binding K1 + K2): metispy's live
REPL — the reference agent/py runs on — seeded, on a served program with a
scripted host whose ticks bring do-lines, dumps every row and the counts
after it. metis::Live (metis-lang runtime/include/metis/live.hpp) replays
them byte for byte (runtime/tools/live_run).

The REPL path is the live one exactly: each sample is Repl._sample() — a
normative selection over _candidates() (canonical order, port-priced), or,
when nothing is enabled, the host's tick applied as one exogenous row
(_tick: fw/plugin.py exo_event; a latched stage revives); a tick of None
ends the host's episode.

The golden (live_run's input):
  L <name> <seed> <samples>
  G <stage name> …                              (the sidecar's stages: the
                                                wire carries indices only)
  O <canon of wire event 0> …                   (the sidecar's canon)
  B <port> const <v> | B <port> if <atom> <then> <else>
  T <line> ; <line> … | T -                     the ticks at quiescence, in order
  A <k> <line> ; <line> …                       a tick mid-activity: applied
                                                before sample k (Repl._tick, as
                                                the watch ticks every beat)
  F <k> <event>                                 a forced row before sample k (after
                                                its A): Repl._fire(i, "forced")
  Q <stage>                                     the host's episode is over and the
                                                world latched (_latch_done); with an
                                                A still ahead the run goes on, and
                                                that tick revives the stage
  R <kind> <canon> <stage> | <name> | <a:n …>   one per sample
  E <stage>

    python tools/gen_live_goldens.py DIR             the public cases (metis-lang
                                                     corpus programs): live_<n>.wire
                                                     + live_<n>.txt in DIR
    python tools/gen_live_goldens.py --domains DIR   the catalog domains' goldens
                                                     (<n>.txt; their wires are the
                                                     artifact goldens')

The projection goldens (K4, proj_<n>): P(observed | the program, from a
state, over h steps) by kernel/record.py surprise_record — the watch's
surprise test — priced as the REPL prices; the cases are the agents'
recorded surprises (RECORDS) and windows of the live trace (from the
state before a run of h program rows: the counts after it in full, one
atom of them, and the full counts with one atom off by one):
  L <name> · G <stages> · B <bindings>
  P <stage> | <counts> | <h> | <observed, zeros kept> | <likelihood, 12 digits>

gen_smoke_fixtures.py writes the public ones into metis-lang
runtime/fixtures with the provenance header; tests/test_live_goldens.py
pins the domains' (METIS_BLESS=1 regenerates)."""

import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from langpath import LANG  # noqa: E402

# the kitchen's weight ports as metis/domains/kitchen/host.py _weights binds
# them (tests/test_live_goldens.py checks the two agree)
KITCHEN = [("w_cook", "const", 1.0), ("w_fetch_dish", "if", "cooking", 1.0, 0.0),
           ("w_fetch_onion", "if", "cooking", 0.05, 1.0), ("w_load", "const", 1.0),
           ("w_plate", "const", 1.0), ("w_serve", "const", 1.0),
           ("w_partner", "const", 0.1)]
# the orders: pot space up front, so the onion's load competes with the
# second cook's partner_load and, once cooking, fetch_onion with fetch_dish
ORDERS = [["+pot_space 9"], [], ["+hand_free"], ["+pot_space 3", "+held_onion"],
          ["+cooking"], [], ["+hand_free 2"], ["+pot_space 6"], [],
          ["-pot_space 4", "+held_dish"], ["+hand_free"], []]
# mid-activity: the second cook never goes quiet (fetch_onion, partner_load)
AT = {0: ["+pot_space 9"], 20: ["+pot_space 3", "+held_dish"], 45: ["+hand_free"],
      70: ["+pot_space 6"], 100: ["-cooking", "+hand_free"]}

PUBLIC = {
    # two stages, a link, constant and read-scoped ports, complements; a
    # latched stage revived by a tick
    "cascade": dict(program=LANG / "corpus/cascade.llp", seed=11,
                    bindings=[("pa", "const", 0.3), ("pb", "if", "a", 0.8, 0.2)],
                    ticks=[["+ta"], [], ["+tb"], ["+ta", "+tb"], ["-za 5"],
                           ["+ta 2"], ["+tb 2"], ["do +ta"], ["+tb 0"], [],
                           ["+ta", "-a"], ["+tb"]],
                    at={0: ["+ta 3"], 3: ["+tb 3"], 6: ["+tb 2", "+ta"],
                        28: ["+tb 2"]},
                    force={1: "aon", 5: "boff"}, samples=40),
    # fractional rule weights at full precision (the wire's repr weights)
    "weights": dict(program=ROOT / "tools/live_weights.llp", seed=23,
                    samples=120, bindings=[],
                    ticks=[["+tok"]] * 30 + [["+tok 3"]] * 10),
    # multiplicities: a one-stage world sampled at length
    "love_triangle": dict(program=LANG / "corpus/love_triangle.llp", seed=5,
                          samples=60, bindings=[],
                          ticks=[[], ["+jealous(helena)"], [], []]),
}

DOMAINS = {
    "plant": dict(seed=7, samples=40, bindings=[],
                  ticks=[["+sun"], ["+cmd_check"], [], ["+cmd_water"],
                         ["+sun", "+cmd_check"], ["+visitor"],
                         ["+sun 2", "+cmd_check 2"],
                         ["-checked(dry) 5", "-checked(moist) 5"],
                         ["+cmd_water", "+visitor"], ["do +sun"], ["+moist 0"],
                         ["+sun", "+cmd_water", "+visitor", "+cmd_check"]]),
    "dog": dict(seed=3, samples=80, bindings=[],
                ticks=[["+cmd_go(kitchen)", "+cmd_sit", "+cmd_find(keys)"], [],
                       ["+cmd_stand", "+cmd_follow"],
                       ["+cmd_go(sofa)", "+cmd_paw", "+cmd_lie"], ["-battery_ok"],
                       ["+cmd_go(door)", "+cmd_stop"], ["+battery_ok"],
                       ["+cmd_lie", "+cmd_cancel", "+cmd_stand"],
                       ["+cmd_go(window_chair)", "+cmd_find(bottle)"],
                       ["+cmd_sit 2", "+cmd_follow"], ["+cmd_cancel 2"]]),
    "kitchen": dict(seed=13, samples=150, bindings=KITCHEN, ticks=ORDERS,
                    at=AT),
    "kitchen_shared": dict(seed=17, samples=150, bindings=KITCHEN, ticks=ORDERS,
                           at=AT),
}


def py_binding(spec):
    if spec[1] == "const":
        return spec[2]
    _, _, atom, yes, no = spec
    return lambda args, counts: yes if counts.get(atom, 0) > 0 else no


def binding_line(spec):
    return "B " + " ".join(repr(x) if isinstance(x, float) else str(x)
                           for x in spec)


class ScriptHost:
    """The scripted host: its ticks in order, then None (the episode is
    over); it binds the case's weight ports. The case comes through
    METIS_LIVE_CASE (JSON): the Repl imports this file afresh as the
    plugin, so a module global would not reach it."""

    def __init__(self, case):
        self.ticks, self.bindings = json.loads(os.environ["METIS_LIVE_CASE"])
        self.i = 0
        self.pending = None          # a tick the driver forces (`at`)

    def bind(self, reg):
        for spec in self.bindings:
            reg.weight(spec[0], py_binding(spec))

    def tick(self, counts):
        if self.pending is not None:
            lines, self.pending = self.pending, None
            return lines
        if self.i >= len(self.ticks):
            return None
        self.i += 1
        return list(self.ticks[self.i - 1])

    def on_event(self, name, args):
        pass

    def snapshot(self):
        return {"i": self.i}

    def restore(self, snap):
        self.i = snap["i"]


def make(case):
    return ScriptHost(case)


def trace(name, program, packs, seed, samples, ticks, bindings, at=None,
          force=None):
    """(ground.wire, golden text) of one case; `at`: {sample k: do-lines}
    ticked before sample k; `force`: {sample k: event} fired before it."""
    at, force = at or {}, force or {}
    from metis.kernel.artifact import build, served_doc
    from metis.lang.repl import Repl
    files = build(program, packs)
    side = json.loads(files["sidecar.json"])
    os.environ["METIS_LIVE_CASE"] = json.dumps([ticks, bindings])
    r = Repl(str(program), seed=seed, plugin=f"{Path(__file__).resolve()}:make",
             host=True, stdout=io.StringIO())
    if packs:                    # the served program: the program plus its packs
        doc, _, _ = served_doc(program, packs)
        r._reserve(doc.catalog, doc.ports)
    assert r.key == side["key"], f"{name}: the REPL serves another program"
    canon = {id(t): j for j, t in enumerate(r.events_tbl)}
    out = [f"L {name} {seed} {samples}",
           "G " + " ".join(side["stages"]),
           "O " + " ".join(str(e["canon"]) for e in side["events"])]
    out += [binding_line(b) for b in bindings]
    out += ["T " + (" ; ".join(t) if t else "-") for t in ticks]
    out += [f"A {k} " + " ; ".join(at[k]) for k in sorted(at)]
    out += [f"F {k} {force[k]}" for k in sorted(force)]

    def row():
        kind, _, ev = r.fired_rows[-1]
        c = " ".join(f"{a}:{n}" for a, n in sorted(r.counts.items())) or "-"
        out.append(f"R {kind} {canon.get(id(r.fired_rows[-1]), -1)} "
                   f"{r.stage} | {ev.name} | {c}")
    for k in range(samples):
        if k in at:
            r.plugin.pending = list(at[k])
            r._tick()
            row()
        if k in force:
            ws = r._candidates()
            i = [j for j, _ in ws if r.events_tbl[j][2].name == force[k]]
            assert i, f"{name}: {force[k]} is not enabled before sample {k}"
            r._fire(i[0], "forced", ws)
            row()
        if r._sample() is None:
            if not any(j > k for j in list(at) + list(force)):
                break
            if not out[-1].startswith("Q "):  # latched; a tick ahead revives
                out.append(f"Q {r.stage}")
            continue
        row()
    out.append(f"E {r.stage}")
    return files["ground.wire"], "\n".join(out) + "\n"


# the agents' recorded surprises (metis-catalog domains/<n>/records, local
# files; their values copied here): the plant's visitor — before its pack
# (session 20261015-110203, record 0) and with the visit announced
# (20261015-114403, +visitor, the pack's trigger) — and the first cook's
# tape when the second cook loads its pot (kitchen 20261015-095716 rec 0)
RECORDS = {
    "plant": [dict(stage="main", init={"dry": 1}, steps=1,
                   observed={"moist": 1, "dry": 0}),
              dict(stage="main", init={"dry": 1, "visitor": 1}, steps=1,
                   observed={"moist": 1, "dry": 0})],
    "kitchen": [dict(stage="task", steps=1,
                     init={"served": 1, "pot_space": 2, "hand_free": 1,
                           "pot_onion": 1},
                     observed={"hand_free": 0, "held_onion": 1,
                               "held_dish": 0, "held_soup": 0,
                               "pot_space": 1, "pot_onion": 2, "cooking": 0,
                               "ready": 0, "served": 1})],
}


def likelihood(doc, bindings, stage, counts, steps, observed):
    """The watch's record (watch.py _record): the catalog grounded on the
    checkpoint, the REPL's pricer, surprise_record's likelihood."""
    from metis.fw.registry import Registry
    from metis.kernel.record import surprise_record
    from metis.lang.compiler import PortPricer
    prog = doc.catalog.ground(dict(counts), init_stage=stage)
    policy = None
    if doc.ports.weight_refs:
        reg = Registry()
        for b in bindings:
            reg.weight(b[0], py_binding(b))
        policy = PortPricer(doc.ports, reg.weight_bindings(),
                            horn=doc.catalog.horn_db()).policy(prog)
    return surprise_record(prog, steps, dict(observed),
                           policy=policy)["likelihood"]


def _counts(text):
    return {} if text == "-" else {
        a: int(n) for a, n in (t.rsplit(":", 1) for t in text.split())}


def _text(counts):
    return " ".join(f"{a}:{n}" for a, n in sorted(counts.items())) or "-"


def projection(name, program, packs, bindings, golden, records=()):
    """The projection golden of one case, from its live golden's rows."""
    from metis.kernel.artifact import served_doc
    doc, _, _ = served_doc(program, packs)
    side_stages = next(l for l in golden.splitlines() if l.startswith("G "))
    rows = []                        # (kind, stage after, counts after)
    for line in golden.splitlines():
        if line.startswith("R "):
            head, _, c = line.rpartition(" | ")
            kind, _, stage = head.split(" | ")[0].split()[1:4]
            rows.append((kind, stage, _counts(c)))
        elif line.startswith("Q "):
            rows.append(("latch", line[2:], {}))
    cases = [(r["stage"], r["init"], r["steps"], r["observed"])
             for r in records]
    starts = 0
    for i in range(1, len(rows)):
        h = 1 + i % 3
        win = rows[i:i + h]
        if len(win) < h or any(k not in ("clause", "link") for k, _, _ in win):
            continue
        if rows[i - 1][0] == "latch":
            continue
        starts += 1
        if starts % 4 != 1 or starts > 48:
            continue
        stage, before = rows[i - 1][1], rows[i - 1][2]
        after = win[-1][2]
        full = {a: after.get(a, 0) for a in set(before) | set(after)}
        first = min(after) if after else min(before)
        off = dict(full)
        off[first] = off.get(first, 0) + 1
        cases += [(stage, before, h, full),
                  (stage, before, h, {first: full.get(first, 0)}),
                  (stage, before, h, off)]
    out = [f"L {name}", side_stages] + [binding_line(b) for b in bindings]
    for stage, before, h, obs in cases:
        like = likelihood(doc, bindings, stage, before, h, obs)
        # 12 significant digits: the filter sums paths in hash order, so
        # the last bits vary with PYTHONHASHSEED (0.9999999999999999 or
        # 1.0); a zero stays exactly 0 (structural)
        out.append(f"P {stage} | {_text(before)} | {h} | {_text(obs)} | "
                   f"{float(like):.12g}")
    return "\n".join(out) + "\n"


def domain_trace(name):
    from metis.kernel.artifact import resolve
    program, packs, _ = resolve(name)
    return trace(name, program, packs, **DOMAINS[name])


def domain_projection(name, golden):
    from metis.kernel.artifact import resolve
    program, packs, _ = resolve(name)
    return projection(name, program, packs, DOMAINS[name]["bindings"], golden,
                      RECORDS.get(name, ()))


def main():
    args = sys.argv[1:]
    if args[:1] == ["--domains"]:
        out = Path(args[1]) if len(args) > 1 else ROOT / "tests/goldens/live"
        out.mkdir(parents=True, exist_ok=True)
        for name in DOMAINS:
            _, golden = domain_trace(name)
            (out / f"{name}.txt").write_text(
                "# metis live golden — tools/gen_live_goldens.py; "
                "METIS_BLESS=1 regenerates\n" + golden)
            (out / f"proj_{name}.txt").write_text(
                "# metis projection golden — tools/gen_live_goldens.py; "
                "METIS_BLESS=1 regenerates\n" + domain_projection(name, golden))
        print(f"wrote {2 * len(DOMAINS)} domain goldens in {out}")
        return
    out = Path(args[0]) if args else Path(".")
    for name, case in PUBLIC.items():
        wire, golden = trace(name, case["program"], [], case["seed"],
                             case["samples"], case["ticks"], case["bindings"],
                             case.get("at"), case.get("force"))
        (out / f"live_{name}.wire").write_text(wire)
        (out / f"live_{name}.txt").write_text(golden)
        (out / f"proj_{name}.txt").write_text(
            projection(name, case["program"], [], case["bindings"], golden))
    print(f"wrote {3 * len(PUBLIC)} live fixtures in {out}")


if __name__ == "__main__":
    main()
