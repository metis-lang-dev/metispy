"""Semantic-handles road gates (docs/road-encoding.md): the compiler
extracts the semantic graph from geometry; rules mention only handles
and relations; continuous quantities reach the measure only through
ports; the motion handoff looks geometry up by handle. Analytic
crash arithmetic at 1e-9; the maneuver flip; the factored evaluator
ADMITS the catalog (the reads contracts hold)."""

from pytest import approx

from metis.kernel import program_key
from metis.lang import parse, pretty
from metis.domains.road.semantic import Lane, RoadModel, Scene, \
    SemanticCompiler, rss_gap

EXACT = dict(abs=1e-9)


def lanes3():
    return [Lane("l1", 500.0, 22.0, left=None, right="l2"),
            Lane("l2", 500.0, 27.8, left="l1", right="l3"),
            Lane("l3", 500.0, 27.8, left="l2", right=None)]


def close_scene():
    # ego closing on a slower lead in the same lane, inside RSS
    return Scene({"ego": ("l2", 100.0, 25.0),
                  "alfa": ("l2", 110.0, 20.0)})


def test_compiler_extracts_the_semantic_graph():
    c = SemanticCompiler(lanes3())
    facts = set(c.facts())
    assert "leftof(l1,l2)." in facts and "rightof(l3,l2)." in facts
    assert "leftof(l2,l3)." in facts and "islane(l1)." in facts
    a = parse(c.instance_llp())
    assert parse(pretty(a)) == a
    # geometry never appears in the instance text
    assert "27.8" not in c.instance_llp()
    assert "100.0" not in c.instance_llp()


def test_analytic_crash_arithmetic():
    """One tick, keep: P(crash) = P(gap stays close) x rss =
    (1 - brk) x rss — hand-checkable through the ports."""
    model = RoadModel(lanes3())
    scene = close_scene()
    assert model.compiler.gap_class(scene) == "gclose"
    b = model.compiler.bindings(scene)
    brk = b["brk"]((), {})
    init = model.compiler.init(scene, "keep")
    rss = b["rss"]((), init)
    assert 0 < brk < 1 and 0 < rss < 1
    pr = model.project(scene, "keep")
    assert model.p_crash(pr) == approx((1 - brk) * rss, **EXACT)
    # changing away removes the conflict entirely
    assert model.p_crash(model.project(scene, "left")) == 0.0


def test_planner_flips_to_the_free_lane():
    model = RoadModel(lanes3())
    res = model.plan(close_scene())
    assert res.action in ("left", "right")
    assert res.scores[res.action][0] == 0.0
    assert res.scores["keep"][0] > 0.2
    # edge lane: fewer candidates, shield before measure
    edge = Scene({"ego": ("l1", 100.0, 25.0),
                  "alfa": ("l3", 300.0, 20.0)})
    assert model.candidates(edge) == ["keep", "right"]
    assert model.plan(edge).action == "keep"     # no hazard anywhere


def test_handoff_looks_geometry_up_by_handle():
    model = RoadModel(lanes3())
    scene = close_scene()
    pk = model.handoff(scene, "left")
    assert pk["lane"] == "l1" and pk["v_cap"] == 22.0
    assert pk["s_from"] == 100.0 and pk["s_to"] == 160.0
    pk2 = model.handoff(scene, "keep")
    assert pk2["lane"] == "l2" and pk2["v_cap"] == 27.8


def test_factored_evaluator_admits_the_catalog():
    """The reads contracts hold end to end: the emission admits the
    ground program and reproduces the filter at 1e-9 — the handle
    catalog is factorization-ready by construction."""
    from metis.kernel.factorize import FactoredProgram
    from metis.lang import factor_hooks
    model = RoadModel(lanes3())
    scene = close_scene()
    prog = model.doc.catalog.ground(
        model.compiler.init(scene, "keep"))
    hooks = factor_hooks(prog, model.doc.ports,
                         model.compiler.bindings(scene),
                         model.doc.catalog.horn_db())
    fp = FactoredProgram(prog, 6, hooks)
    pr = model.project(scene, "keep")
    for atom in ("crash", "gclose", "gsafe"):
        want = sum(p for (_, k), p in pr.final.items()
                   if (atom, 1) in k)
        assert fp.marginal(atom) == approx(want, **EXACT), atom


def test_same_scene_same_program_key():
    """Determinism of the compile: scene -> identical artifact."""
    model = RoadModel(lanes3())
    scene = close_scene()
    k1 = program_key(model.doc.catalog.ground(
        model.compiler.init(scene, "keep")))
    k2 = program_key(RoadModel(lanes3()).doc.catalog.ground(
        SemanticCompiler(lanes3()).init(scene, "keep")))
    assert k1 == k2


def test_rss_gap_shape():
    assert rss_gap(25.0, 20.0) > rss_gap(20.0, 25.0)
    assert rss_gap(0.0, 30.0) == 0.0
