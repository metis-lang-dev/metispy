"""CrossingTraffic + sampling-evaluator gates. Fast: catalog
round-trip, tick discipline (paid bounded by ticks), exact/sampled
evaluator agreement on one catalog, framework path. Slow: joint-state
policy iteration recovers the VI optimum; real-env preservation and
benchmark."""

import importlib.util

import pytest
from pytest import approx

from metis.domains.xtraffic.model import Topology, XTrafficModel, \
    gen_catalog, improve_policy, vi_optimal
from metis.fw import Blackboard
from metis.lang import parse, pretty

EXACT = dict(abs=1e-9)


def instance1_topo() -> Topology:
    xs, ys = ["x1", "x2", "x3"], ["y1", "y2", "y3"]
    cells = [f"{x}_{y}" for x in xs for y in ys]
    adj = {}
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            c = f"{x}_{y}"
            if yi + 1 < 3:
                adj[(c, "north")] = f"{x}_{ys[yi + 1]}"
            if yi - 1 >= 0:
                adj[(c, "south")] = f"{x}_{ys[yi - 1]}"
            if xi + 1 < 3:
                adj[(c, "east")] = f"{xs[xi + 1]}_{y}"
            if xi - 1 >= 0:
                adj[(c, "west")] = f"{xs[xi - 1]}_{y}"
    return Topology(cells, adj, ["x1_y2", "x2_y2", "x3_y2"],
                    {"x1_y2": "x2_y2", "x2_y2": "x3_y2"},
                    goal="x3_y3", input_rate=0.3, horizon=40)


START, CFG0 = "x3_y1", frozenset({"x1_y2", "x3_y2"})


def test_generated_catalog_roundtrips():
    a = parse(gen_catalog(instance1_topo()))
    assert parse(pretty(a)) == a


def test_tick_discipline_and_cost_bounds():
    model = XTrafficModel(instance1_topo())
    pr = model.project(START, set(CFG0), ticks=10)
    # exactly one move event per tick: paid can never exceed ticks
    assert max(dict(k).get("paid", 0)
               for (_, k), p in pr.final.items()) <= 10
    assert 0.0 < model.expected_cost(pr) <= 10.0
    # dead-on-arrival state pays every epoch
    pr = model.project(None, set(), ticks=7)
    assert model.expected_cost(pr) == approx(7.0, **EXACT)
    # at goal: free forever
    pr = model.project(model.topo.goal, set(), ticks=7)
    assert model.expected_cost(pr) == approx(0.0, **EXACT)


def test_exact_and_sampled_evaluators_agree_on_one_catalog():
    model = XTrafficModel(instance1_topo())
    ex = model.choose(START, set(CFG0), ticks=12)
    sp = model.choose(START, set(CFG0), ticks=12, sampled=(600, 7))
    assert sp.action == ex.action
    for d in ex.scores:
        assert sp.scores[d][0] == approx(ex.scores[d][0], abs=0.6)


def test_framework_path():
    from metis.domains.xtraffic.fw import xtraffic_evaluator
    model = XTrafficModel(instance1_topo())
    ev = xtraffic_evaluator(model, ticks=12)
    bb = Blackboard()
    bb["scene"] = (START, set(CFG0))
    res = ev.tick(bb)
    ref = model.choose(START, set(CFG0), ticks=12)
    assert res.action == ref.action
    assert bb["move"] == ref.action
    for d in ref.scores:
        assert res.scores[d] == approx(ref.scores[d], **EXACT)


@pytest.mark.slow
def test_policy_iteration_recovers_vi_optimum():
    topo = instance1_topo()
    model = XTrafficModel(topo)
    improve_policy(model)
    V, vipol = vi_optimal(topo)
    pr = model.project(START, set(CFG0), ticks=topo.horizon)
    vpi = -model.expected_cost(pr)
    vstar = V[(START, (1, 0, 1))]
    assert vpi == approx(vstar, abs=1e-6), (vpi, vstar)
    assert model.policy[(START, CFG0)] == vipol[(START, (1, 0, 1))]


@pytest.mark.slow
def test_thts_with_informed_rollouts_finds_the_dodge():
    """THTS over sampled 1-tick catalog programs. With UNINFORMED base
    rollouts it is unreliable on the trap (the reckless rollout policy
    under-values the safe detour — the honest limitation, see the
    stage log); with PI-informed rollouts (the live policy table the
    ports already read) it finds west on every seed."""
    from metis.domains.xtraffic.model import thts_choose
    topo = instance1_topo()
    model = XTrafficModel(topo)
    improve_policy(model)
    for seed in (3, 4, 5):
        best, stats = thts_choose(model, START, set(CFG0),
                                  iters=2000, depth=40, seed=seed)
        assert best == "west", (seed, stats)


requires_rddl = pytest.mark.skipif(
    importlib.util.find_spec("pyRDDLGym") is None
    or importlib.util.find_spec("rddlrepository") is None,
    reason="pyRDDLGym/rddlrepository not installed")


@pytest.mark.slow
@requires_rddl
def test_real_env_preservation_and_benchmark():
    import statistics
    from metis.domains.xtraffic.rddl_adapter import agent_noop, \
        agent_policy, agent_random, load, run_episode, scene_of
    spec = load("1")
    topo = spec["topo"]
    model = XTrafficModel(topo)

    # preservation: 1-tick obstacle marginals vs the simulator (noop)
    env = spec["env"]
    obs, _ = env.reset(seed=0)
    pos, obstacles = scene_of(obs)
    pr = model.project(pos, obstacles, ticks=1)
    n = 300
    counts = {c: 0 for c in topo.ocells}
    for k in range(n):
        env.reset(seed=k)
        nxt, _, _, _, _ = env.step({})
        for c in scene_of(nxt)[1]:
            counts[c] += 1
    for c in topo.ocells:
        want = sum(p for (_, kap), p in pr.final.items()
                   if (f"o({c})", 1) in kap)
        assert counts[c] / n == approx(want, abs=0.08), (c,)

    improve_policy(model)
    V, _ = vi_optimal(topo)
    obs, _ = env.reset(seed=0)
    pos, obstacles = scene_of(obs)
    key = tuple(1 if c in obstacles else 0 for c in topo.ocells)
    vstar = V[(pos, key)]
    means = {}
    for name, agent, n_ep in (
            ("metis", agent_policy(model), 40),
            ("random", agent_random(spec), 30),
            ("noop", agent_noop(spec), 5)):
        rs = [run_episode(spec, agent, seed=600 + i)
              for i in range(n_ep)]
        means[name] = (statistics.mean(rs), statistics.pstdev(rs))
    print(f"\nCrossingTraffic inst 1: V*={vstar:.3f}  metis "
          f"{means['metis'][0]:.2f}±{means['metis'][1]:.2f}  random "
          f"{means['random'][0]:.2f}  noop {means['noop'][0]:.2f}")
    m, sd = means["metis"]
    assert abs(m - vstar) <= 3 * sd / (40 ** 0.5) + 1e-9, (m, vstar)
    assert m > means["random"][0] > means["noop"][0] - 40
    spec["env"].close()


def test_lifted_include_equals_monolithic():
    """The RDDL-style split: generated instance (types via range
    sugar + facts + goal/lastidx) + include of the static lifted
    domain grounds to the program_key-identical program (bwd guards
    on LINKS carry the chain end; goal(C) carries the goal cell)."""
    from metis.kernel import program_key
    from metis.lang import compile_catalog, pretty, resolve_includes
    from metis.domains.xtraffic.model import XT_DOMAIN, gen_instance
    topo = instance1_topo()
    inst_ast = parse(gen_instance(topo))
    assert parse(pretty(inst_ast)) == inst_ast
    lifted = compile_catalog(
        resolve_includes(inst_ast, XT_DOMAIN.parent))
    mono = compile_catalog(parse(gen_catalog(topo)))
    init = {"rob(x3_y1)": 1, "mvt": 1, "e(x1_y2)": 1, "o(x2_y2)": 1,
            "e(x3_y2)": 1}
    assert program_key(lifted.catalog.ground(init)) == \
        program_key(mono.catalog.ground(init))
    assert lifted.ports.weight_refs.keys() == \
        mono.ports.weight_refs.keys()
