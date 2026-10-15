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
  R <kind> <canon> <stage> | <name> | <a:n …>   one per sample
  E <stage>

    python tools/gen_live_goldens.py DIR             the public cases (metis-lang
                                                     corpus programs): live_<n>.wire
                                                     + live_<n>.txt in DIR
    python tools/gen_live_goldens.py --domains DIR   the catalog domains' goldens
                                                     (<n>.txt; their wires are the
                                                     artifact goldens')

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
    "cascade": dict(program=LANG / "corpus/cascade.llp", seed=11, samples=40,
                    bindings=[("pa", "const", 0.3), ("pb", "if", "a", 0.8, 0.2)],
                    ticks=[["+ta"], [], ["+tb"], ["+ta", "+tb"], ["-za 5"],
                           ["+ta 2"], ["+tb 2"], ["do +ta"], ["+tb 0"], [],
                           ["+ta", "-a"], ["+tb"]],
                    at={0: ["+ta 3"], 3: ["+tb 3"], 6: ["+tb 2", "+ta"]},
                    force={1: "aon", 5: "boff"}),
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
            break
        row()
    out.append(f"E {r.stage}")
    return files["ground.wire"], "\n".join(out) + "\n"


def domain_trace(name):
    from metis.kernel.artifact import resolve
    program, packs, _ = resolve(name)
    return trace(name, program, packs, **DOMAINS[name])


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
        print(f"wrote {len(DOMAINS)} domain goldens in {out}")
        return
    out = Path(args[0]) if args else Path(".")
    for name, case in PUBLIC.items():
        wire, golden = trace(name, case["program"], [], case["seed"],
                             case["samples"], case["ticks"], case["bindings"],
                             case.get("at"), case.get("force"))
        (out / f"live_{name}.wire").write_text(wire)
        (out / f"live_{name}.txt").write_text(golden)
    print(f"wrote {2 * len(PUBLIC)} live fixtures in {out}")


if __name__ == "__main__":
    main()
