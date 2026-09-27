"""Navigation gates: the generated catalog round-trips; exact policy
iteration through the filter RECOVERS THE VI OPTIMUM (value equal,
policy equal); the MAP derivation is the safest route (the explanation
object); framework tick agrees with the converged policy; (slow) real
env: empirical calibration + benchmark vs baselines."""

import importlib.util

import pytest
from pytest import approx

from metis.domains.nav.model import Topology, first_move_scores, \
    gen_catalog, improve_policy, policy_value, vi_optimal
from metis.fw import Blackboard
from metis.kernel import map_derivation
from metis.lang import parse, pretty

EXACT = dict(abs=1e-9)


def instance1_topo() -> Topology:
    xs, ys = ["x6", "x9", "x14", "x21"], ["y12", "y15", "y20"]
    cells = [f"{x}_{y}" for x in xs for y in ys]
    adj = {}
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            c = f"{x}_{y}"
            if yi + 1 < len(ys):
                adj[(c, "north")] = f"{x}_{ys[yi + 1]}"
            if yi - 1 >= 0:
                adj[(c, "south")] = f"{x}_{ys[yi - 1]}"
            if xi + 1 < len(xs):
                adj[(c, "east")] = f"{xs[xi + 1]}_{y}"
            if xi - 1 >= 0:
                adj[(c, "west")] = f"{xs[xi - 1]}_{y}"
    p = {c: 0.0 for c in cells}
    p.update({"x6_y15": .049, "x9_y15": .345, "x14_y15": .637,
              "x21_y15": .928})
    return Topology(cells, adj, p, goal="x21_y20", horizon=40)


START = "x21_y12"


def test_generated_catalog_roundtrips():
    a = parse(gen_catalog(instance1_topo()))
    assert parse(pretty(a)) == a


def test_lifted_include_equals_monolithic():
    """The module-inclusion/lifting gate: the generated INSTANCE (types
    + facts + include of the static lifted domain) grounds to the
    program_key-identical program as the monolithic catalog."""
    from metis.kernel import program_key
    from metis.lang import compile_catalog
    from metis.domains.nav.model import NavModel, gen_instance
    topo = instance1_topo()
    mono = compile_catalog(parse(gen_catalog(topo)))
    lifted = NavModel(topo)          # compiles instance via include
    inst_ast = parse(gen_instance(topo))
    assert parse(pretty(inst_ast)) == inst_ast   # include round-trips
    init = {f"at({START})": 1}
    assert program_key(mono.catalog.ground(init)) == \
        program_key(lifted.doc.catalog.ground(init))


def test_policy_iteration_recovers_the_vi_optimum():
    topo = instance1_topo()
    model = improve_policy(topo)
    V, vipol = vi_optimal(topo)
    assert policy_value(model, START) == approx(V[START], **EXACT)
    for c in topo.cells:
        if c != topo.goal:
            assert topo.policy[c] == vipol[c], c
    # and it found the classic solution: detour west, cross at 0.049
    assert topo.policy["x21_y12"] == "west"
    assert topo.policy["x6_y12"] == "north"


def test_map_derivation_is_the_safest_route():
    """The explanation object: under the converged policy, the MAP
    derivation from start to goal is the west-detour route through the
    0.049 cell, with probability = its survival."""
    topo = instance1_topo()
    model = improve_policy(topo)
    pr = model.project(START, 12)
    goal_atom = f"at({topo.goal})"
    m = map_derivation(
        pr, {12: (lambda s: (goal_atom, 1) in s[1])})
    names = [pr.program.events()[e][2].name for e in m.events
             if e is not None]
    assert any("C2=x6_y15" in n and n.startswith("mv_north") for n in
               names), names
    assert not any("y15" in n.split("C2=")[-1] and "x6" not in n
                   for n in names if n.startswith("mv_")), names
    assert m.prob == approx(1.0 - 0.049, **EXACT)


def test_framework_tick_matches_converged_policy():
    from metis.domains.nav.fw import nav_evaluator
    topo = instance1_topo()
    improve_policy(topo)
    ev = nav_evaluator(topo)
    for pos in (START, "x6_y12", "x6_y15"):
        bb = Blackboard()
        bb["pos"] = pos
        res = ev.tick(bb)
        assert res.action == topo.policy[pos], pos
        assert bb["move"] == topo.policy[pos]


def test_first_move_scores_is_the_improvement_operator():
    topo = instance1_topo()
    model = improve_policy(topo)
    res = first_move_scores(model, START, topo.horizon)
    assert res.action == topo.policy[START]


requires_rddl = pytest.mark.skipif(
    importlib.util.find_spec("pyRDDLGym") is None
    or importlib.util.find_spec("rddlrepository") is None,
    reason="pyRDDLGym/rddlrepository not installed")


@pytest.mark.slow
@requires_rddl
def test_real_instance_benchmark_and_calibration():
    import statistics
    from metis.domains.nav.rddl_adapter import agent_noop, \
        agent_policy, agent_random, load, run_episode
    spec = load("1")
    topo = spec["topo"]
    model = improve_policy(topo)
    V, _ = vi_optimal(topo)
    import re
    start = None
    obs, _ = spec["env"].reset(seed=0)
    from metis.domains.nav.rddl_adapter import position_of
    start = position_of(obs)
    vstar = V[start]
    means = {}
    for name, agent, n_ep in (("metis", agent_policy(topo), 40),
                              ("random", agent_random(spec), 30),
                              ("noop", agent_noop(spec), 5)):
        rs = [run_episode(spec, agent, seed=400 + i)
              for i in range(n_ep)]
        means[name] = (statistics.mean(rs), statistics.pstdev(rs))
    print(f"\nNavigation inst 1: V*={vstar:.3f}  metis "
          f"{means['metis'][0]:.2f}±{means['metis'][1]:.2f}  random "
          f"{means['random'][0]:.2f}  noop {means['noop'][0]:.2f}")
    m, sd = means["metis"]
    # calibration: empirical mean within 3 standard errors of the
    # EXACT policy value (which equals V*)
    assert abs(m - vstar) <= 3 * sd / (40 ** 0.5) + 1e-9, (m, vstar)
    assert m > means["random"][0]
    assert m > means["noop"][0]
    spec["env"].close()
