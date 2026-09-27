"""Generator for the IR product-artifact smoke (ir_eval_smoke.cpp):
python FactoredProgram is semantic truth. Emits the roverir artifact
(ir_artifact_lines: declared U slots, shipped O/Y schedules, F
mapping, NO expectations) plus an expectations file computed across
per-tick evidence assignments — marginals via fp.marginal(q, init=),
likelihoods via fp.likelihood(obs, init=), and the slot-candidate
decision. The C++ IrEval must reproduce every line from ONE loaded
artifact: the load-once, evidence-per-tick runtime story.

Run from the repo root:  python3 cpp/parity/gen_ir_artifact.py [dir]"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metis.kernel.factorize import FactoredProgram             # noqa: E402
from metis.kernel.ir import ir_artifact_lines                  # noqa: E402
from metis.lang import compile_catalog, factor_hooks           # noqa: E402
from metis.lang.parser import parse                            # noqa: E402

ROVER_IR = """\
% IR product-artifact smoke fixture: the bounded-slot scene story.
catalog roverir 1.
provenance hand "cpp ir artifact smoke" "2026-07-25".
layers (dyn).
stages (act).

type cell {c0 c1 c2}.

namespace bot produce (dyn).

pred tick : bot.
pred at(cell) : bot.
pred reach(cell) : bot.

bwd adj(cell,cell).

adj(c0,c1).
adj(c1,c2).

input scene.
output decision.

weight move(C) reads {tick}.
weight stay reads {tick}.

stage act {
  %% advance one cell along the track.
  go [dyn] : tick * at(C) * adj(C,C2) -o reach(C2)  @w move(C2).
  %% burn the tick doing nothing — slot-independent, so no scene
  %% assignment can stall the tier (the emitted-class condition).
  hold [dyn] : tick -o one  @w stay.
}
"""

REGISTRY = {"move": lambda a, c: 3.0 if a[0] == "c2" else 1.0,
            "stay": lambda a, c: 1.0}

UNKNOWNS = ("at(c0)", "at(c1)")        # the scene slots
QUERY = "reach(c2)"
STEPS = 3
ASSIGNMENTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
CANDIDATES = {"near": {"at(c0)": 0, "at(c1)": 1},
              "far": {"at(c0)": 1, "at(c1)": 0}}


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).parent
    doc = compile_catalog(parse(ROVER_IR))
    prog = doc.catalog.ground({"at(c0)": 1, "at(c1)": 1, "tick": 1})
    hooks = factor_hooks(prog, doc.ports, REGISTRY,
                         doc.catalog.horn_db())
    fp = FactoredProgram(prog, STEPS, hooks, unknowns=UNKNOWNS)

    art = ir_artifact_lines("roverir", fp, [QUERY],
                            observables=[QUERY], ports=doc.ports)
    (out_dir / "ir_rover_artifact.txt").write_text(
        "\n".join(art) + "\n")

    exp = []
    for a0, a1 in ASSIGNMENTS:
        init = {"at(c0)": a0, "at(c1)": a1}
        m = fp.marginal(QUERY, init=init)
        exp.append(f"EM {a0} {a1} {m:.12g}")
        for obs in (0, 1):
            lik = fp.likelihood({QUERY: obs}, init=init)
            exp.append(f"EL {a0} {a1} {obs} {lik:.12g}")
    decision = max(sorted(CANDIDATES),
                   key=lambda c: fp.marginal(QUERY,
                                             init=CANDIDATES[c]))
    exp.append(f"EX {decision}")
    (out_dir / "ir_rover_expect.txt").write_text(
        "\n".join(exp) + "\n")
    print(f"decision={decision}")
    for line in exp[:6]:
        print(line)
    print(f"wrote ir_rover_artifact.txt + ir_rover_expect.txt "
          f"in {out_dir}")


if __name__ == "__main__":
    main()
