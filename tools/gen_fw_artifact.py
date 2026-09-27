"""Generator for the C++ Evaluator smoke (evaluator_smoke.cpp):
python fw.Evaluator is semantic truth. Emits the rover product
artifacts (v1, and v2 = v1 + one rule that FLIPS the decision) via
kernel/wire.py, plus the expectations file the smoke checks against
— decision + scores per version from the ACTUAL python framework
(Evaluator + catalog_domain + Registry with the same overlay chain
the C++ side registers).

Run from the repo root:  python3 cpp/parity/gen_fw_artifact.py [dir]"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metis.fw import Blackboard, Evaluator, Registry           # noqa: E402
from metis.fw.evaluator import catalog_domain                  # noqa: E402
from metis.kernel.wire import emit_artifact                    # noqa: E402
from metis.lang import compile_catalog                         # noqa: E402
from metis.lang.parser import parse                            # noqa: E402

ROVER_V1 = """\
% fw-evaluator smoke fixture — v1.
catalog rover 1.
provenance hand "cpp evaluator smoke" "2026-07-25".
layers (dyn).
stages (act).

type cell {c0 c1 c2}.

namespace bot produce (dyn).

pred tick : bot.
pred at(cell) : bot.

bwd adj(cell,cell).

adj(c0,c1).
adj(c1,c2).

input scene.
output decision.

weight move(cell).
weight stay.

stage act {
  %% advance one cell along the track.
  go [dyn] : tick * at(C) * adj(C,C2) -o at(C2)  @w move(C2).
  %% burn the tick in place.
  hold [dyn] : tick * at(C) -o at(C)  @w stay.
}
"""

# v2 = v1 + ONE rule: the author's add-a-rule/recompile cycle. The
# leap makes the far start overtake the near one — the DECISION flips
# with bindings unchanged.
ROVER_V2 = ROVER_V1.replace("catalog rover 1.", "catalog rover 2.") \
    .replace("""  %% burn the tick in place.
  hold [dyn] : tick * at(C) -o at(C)  @w stay.
""", """  %% burn the tick in place.
  hold [dyn] : tick * at(C) -o at(C)  @w stay.
  %% NEW in v2: leap from the depot straight to the end cell.
  leap [dyn] : tick * at(c0) -o at(c2)  @w 8.0.
""")

STEPS = 3
CANDIDATES = {"near": {"at(c1)": 1, "tick": 1},
              "far": {"at(c0)": 1, "tick": 1}}


def registry() -> Registry:
    reg = Registry()
    reg.weight("move", 1.0)
    reg.overlay("move",
                lambda args, counts: 3.0 if args[0] == "c2" else None)
    reg.weight("stay", 1.0)
    reg.input("scene", lambda v: v)
    reg.output("decision", lambda res, sit: res.action)
    return reg


def p_goal(pr) -> float:
    return pr.final_distribution("at(c2)")[1]


def run_python_truth(doc):
    """Decision + per-candidate P(at c2) through the real framework."""
    reg = registry()
    ev = Evaluator(doc, reg)
    ev.domain = catalog_domain(
        ev,
        candidates_fn=lambda sit: sorted(CANDIDATES),
        init_fn=lambda sit, a: dict(CANDIDATES[a]),
        steps_fn=lambda sit: STEPS,
        score_fn=lambda pr, sit: [1.0 - p_goal(pr)],
        doomed="near")
    bb = Blackboard()
    bb["scene"] = "smoke"
    res = ev.tick(bb)
    assert bb["decision"] == res.action     # output port wrote back
    pg = {}
    for a, init in CANDIDATES.items():
        prog = doc.catalog.ground(dict(init))
        from metis.lang import weight_policy
        pol = weight_policy(prog, doc.ports,
                            reg.weight_bindings())
        from metis.kernel import ForwardProjection
        pg[a] = p_goal(ForwardProjection(prog, STEPS, policy=pol))
    return res, pg


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).parent
    exp = []
    for version, text in ((1, ROVER_V1), (2, ROVER_V2)):
        doc = compile_catalog(parse(text))
        prog = doc.catalog.ground(dict(CANDIDATES["far"]))
        path = out_dir / f"fw_rover_v{version}.txt"
        path.write_text(emit_artifact(doc, prog, STEPS))
        res, pg = run_python_truth(doc)
        scores = res.scores[res.action]
        exp.append(f"X {version} {res.action} "
                   f"{1 if res.doomed else 0} {res.dof} "
                   f"{len(scores)} "
                   + " ".join(f"{s:.12g}" for s in scores))
        for a in sorted(pg):
            exp.append(f"P {version} {a} {pg[a]:.12g}")
        print(f"v{version}: decision={res.action} "
              f"scores={scores} p_goal={pg}")
    (out_dir / "fw_rover_expect.txt").write_text(
        "\n".join(exp) + "\n")
    print(f"wrote fw_rover_v{{1,2}}.txt + fw_rover_expect.txt "
          f"in {out_dir}")


if __name__ == "__main__":
    main()
