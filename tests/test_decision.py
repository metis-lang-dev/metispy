"""The generic decision loop, domain-free mechanics: do() as an
initial-multiset edit, lexicographic hard-safety-first scoring,
doomed fallback, first-candidate tie-break."""

from pytest import approx

from metis.kernel import ForwardProjection, plan
from metis.domains.road import base_catalog, scenario


def committed(cat, action):
    """do(a) for the mini corpus: ego's move is committed by editing the
    initial multiset (no eturn, ego at the outcome), horizon starts at
    the others stage."""
    outcome = {"keep": ("l0", "c2", "v1"),
               "accel": ("l0", "c3", "v2"),
               "brake": ("l0", "c1", "v0")}[action]
    init = [("eturn", ()), ("st", ("ego",) + outcome),
            ("ot", ("alfa",)), ("st", ("alfa", "l0", "c2", "v0"))]
    init = [p for p in init if p[0] != "eturn"]
    return cat.ground(init, init_stage="others")


def crash_mass(pr):
    d = pr.final_distribution("crash")
    return sum(d[1:])


def ego_cell(pr):
    """E[ego cell index] over the final distribution."""
    ev = 0.0
    for (_, kappa), p in pr.final.items():
        for atom, n in kappa:
            if atom.startswith("st(ego,") and n > 0:
                ev += p * int(atom.split(",")[2].strip(")")[1:])
    return ev


def test_plan_picks_safe_then_progress():
    cat = base_catalog()
    res = plan(["keep", "accel", "brake"],
               do=lambda a: committed(cat, a),
               project=lambda prog: ForwardProjection(prog, 5),
               score=lambda pr: (crash_mass(pr), -ego_cell(pr)))
    assert not res.doomed and res.dof == 3
    # keep crashes: collide consumes the st atoms, so the wrecked ego
    # has no position in the final state — progress term 0, and the
    # crash mass dominates lexicographically anyway
    assert res.scores["keep"] == approx((1.0, 0.0))
    assert res.scores["accel"] == approx((0.0, -3.0))
    assert res.scores["brake"] == approx((0.0, -1.0))
    assert res.action == "accel"      # safe first, then progress


def test_plan_doomed_fallback():
    res = plan([], do=None, project=None, score=None,
               doomed_action="brake")
    assert res.doomed and res.action == "brake" and res.dof == 0
    assert res.scores == {}


def test_plan_tie_breaks_to_first_candidate():
    res = plan(["a", "b"],
               do=lambda a: a, project=lambda s: s,
               score=lambda pr: (0.0, 1.0))
    assert res.action == "a"


def test_clamp_domain_decides_over_one_shared_structure():
    """Decision-over-IR, python truth: candidates are evidence
    assignments over declared unknowns on ONE FactoredProgram
    (do() = clamp, no re-grounding); the seam feeds the same plan()
    loop. Mirrors the C++ IrEval mechanism."""
    from metis.kernel.decision import ClampView, clamp_domain
    from metis.kernel.factorize import FactoredProgram
    from metis.lang import compile_catalog, factor_hooks
    from metis.lang.parser import parse

    LLP = """\
catalog roverdec 1.
provenance hand "clamp-domain gate" "2026-07-25".
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
weight move(C) reads {tick}.
weight stay reads {tick}.
stage act {
  %% advance one cell along the track.
  go [dyn] : tick * at(C) * adj(C,C2) -o reach(C2)  @w move(C2).
  %% burn the tick doing nothing.
  hold [dyn] : tick -o one  @w stay.
}
"""
    doc = compile_catalog(parse(LLP))
    prog = doc.catalog.ground({"at(c0)": 1, "at(c1)": 1, "tick": 1})
    hooks = factor_hooks(
        prog, doc.ports,
        {"move": lambda a, c: 3.0 if a[0] == "c2" else 1.0,
         "stay": lambda a, c: 1.0},
        doc.catalog.horn_db())
    fp = FactoredProgram(prog, 3, hooks,
                         unknowns=("at(c0)", "at(c1)"))
    cands = {"near": {"at(c0)": 0, "at(c1)": 1},
             "far": {"at(c0)": 1, "at(c1)": 0}}
    factory = clamp_domain(
        fp,
        candidates_fn=lambda sit: sorted(cands),
        evidence_fn=lambda sit, a: cands[a],
        score_fn=lambda vw, sit: (1.0 - vw.marginal("reach(c2)"),))
    c, do, project, score, doomed = factory({})
    res = plan(c, do, project, score, doomed_action=doomed)
    assert res.action == "near"
    assert res.scores["near"] == approx((0.25,))
    assert res.scores["far"] == approx((1.0,))
    # one shared structure: every projection wraps the SAME fp
    assert all(isinstance(pr, ClampView) and pr.fp is fp
               for pr in res.projections.values())
