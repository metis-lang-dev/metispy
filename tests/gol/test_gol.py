"""Game of Life gates: the generic sync compilation is exact on a
hand-computable topology (fast); the framework path agrees with the
direct greedy; preservation vs the real IPPC env and the benchmark
(slow)."""

import importlib.util

import pytest
from pytest import approx

from metis.domains.gol.model import gol_project, greedy_set
from metis.fw import Blackboard
from metis.kernel import marginal_on

EXACT = dict(abs=1e-12)

CELLS = ["a", "b", "c"]
NBRS = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
NOISE = {c: 0.1 for c in CELLS}


def test_conway_bernoullis_exact():
    pr = gol_project(CELLS, NBRS, NOISE, {"a", "b", "c"})
    # a: 1 live neighbor -> dies (noise floor); b: 2 -> lives
    assert marginal_on(pr, "a", "alive") == approx(0.1, **EXACT)
    assert marginal_on(pr, "b", "alive") == approx(0.9, **EXACT)
    assert marginal_on(pr, "c", "alive") == approx(0.1, **EXACT)
    # independence given the snapshot
    joint = sum(p for (_, k), p in pr.final.items()
                if ("alive(a)", 1) in k and ("alive(b)", 1) in k)
    assert joint == approx(0.1 * 0.9, **EXACT)


def test_set_is_bernoulli_not_krondelta():
    # the set action ORs into the CONDITION: forced cell gets 1-noise
    pr = gol_project(CELLS, NBRS, NOISE, set(), force="a")
    assert marginal_on(pr, "a", "alive") == approx(0.9, **EXACT)


def test_dead_cell_birth_rule():
    # b dead with exactly 2 live neighbors (a,c): no birth -> noise
    pr = gol_project(CELLS, NBRS, NOISE, {"a", "c"})
    assert marginal_on(pr, "b", "alive") == approx(0.1, **EXACT)


def test_framework_path_matches_direct():
    from metis.domains.gol.fw import gol_evaluator
    ev = gol_evaluator(CELLS, NBRS, NOISE)
    bb = Blackboard()
    bb["grid"] = {"a", "b"}
    res = ev.tick(bb)
    ref = greedy_set(CELLS, NBRS, NOISE, {"a", "b"})
    assert res.action == ref.action
    assert bb["action"] == ref.action
    for a, s in ref.scores.items():
        assert res.scores[a] == approx(s, **EXACT)


requires_rddl = pytest.mark.skipif(
    importlib.util.find_spec("pyRDDLGym") is None
    or importlib.util.find_spec("rddlrepository") is None,
    reason="pyRDDLGym/rddlrepository not installed")


@pytest.mark.slow
@requires_rddl
def test_preservation_against_rddl_simulator():
    from metis.domains.gol.rddl_adapter import alive_of, load
    spec = load("1")
    env = spec["env"]
    obs, _ = env.reset(seed=0)
    s0 = alive_of(obs)
    pr = gol_project(spec["cells"], spec["neighbors"], spec["noise"],
                     s0)
    want = {c: marginal_on(pr, c, "alive") for c in spec["cells"]}
    n = 400
    counts = {c: 0 for c in spec["cells"]}
    for k in range(n):
        env.reset(seed=k)
        nxt, _, _, _, _ = env.step({})
        for c in alive_of(nxt):
            counts[c] += 1
    for c in spec["cells"]:
        assert counts[c] / n == approx(want[c], abs=0.06), \
            (c, counts[c] / n, want[c])
    env.close()


@pytest.mark.slow
@requires_rddl
def test_benchmark_instance1_vs_baselines():
    from metis.domains.gol.rddl_adapter import agent_metis, \
        agent_noop, agent_random, load, run_episode
    spec = load("1")
    means = {}
    for name, make, n_ep in (("metis", agent_metis, 6),
                             ("random", agent_random, 12),
                             ("noop", agent_noop, 12)):
        agent = make(spec)
        rs = [run_episode(spec, agent, seed=300 + i)
              for i in range(n_ep)]
        means[name] = sum(rs) / n_ep
    print(f"\nGoL IPPC2011 instance 1 mean reward: {means}")
    assert means["metis"] > means["random"], means
    assert means["metis"] > means["noop"], means
    spec["env"].close()


# -- stage 42: the text recode vs the sync engine ------------------------

def test_text_recode_matches_the_sync_path():
    """gen_catalog + life port == kernel.sync + gol_p_on at 1e-9,
    across init patterns, force, and two ticks — the last code-side
    program builder retired to text, pinned against its origin."""
    from pytest import approx
    from metis.domains.gol.model import catalog_init, gen_catalog, \
        gol_project, life_bindings
    from metis.kernel import ForwardProjection, marginal_on, \
        steps_per_tick
    from metis.lang import compile_catalog, parse, weight_policy, \
        factor_hooks
    from metis.kernel.factorize import FactoredProgram

    cells = ["g00", "g01", "g10", "g11"]
    nbrs = {c: [v for v in cells if v != c] for c in cells}
    noise = {c: 0.1 for c in cells}
    doc = compile_catalog(parse(gen_catalog(cells, nbrs)))
    for alive, force, ticks in ((["g00", "g01"], None, 1),
                                (["g00", "g01"], "g10", 1),
                                (["g00"], None, 2),
                                ([], "g11", 2)):
        prog = doc.catalog.ground(catalog_init(cells, set(alive),
                                               force))
        pol = weight_policy(prog, doc.ports,
                            life_bindings(nbrs, noise))
        steps = ticks * steps_per_tick(len(cells))
        pr = ForwardProjection(prog, steps, policy=pol)
        ref = gol_project(cells, nbrs, noise, alive, ticks=ticks,
                          force=force)
        for c in cells:
            got = sum(p for (_, k), p in pr.final.items()
                      if (f"a({c})", 1) in k)
            assert got == approx(marginal_on(ref, c, on="alive"),
                                 abs=1e-9), (c, alive, force, ticks)
    # and the recode is factorization-ready (class F)
    prog = doc.catalog.ground(catalog_init(cells, {"g00", "g01"},
                                           "g10"))
    hooks = factor_hooks(prog, doc.ports,
                         life_bindings(nbrs, noise),
                         doc.catalog.horn_db())
    fp = FactoredProgram(prog, steps_per_tick(len(cells)), hooks)
    ref = gol_project(cells, nbrs, noise, ["g00", "g01"], ticks=1,
                      force="g10")
    for c in cells:
        assert fp.marginal(f"a({c})") == approx(
            marginal_on(ref, c, on="alive"), abs=1e-9), c
