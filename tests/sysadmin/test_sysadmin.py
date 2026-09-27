"""Stage 7 gates. Fast half: the synchronous->interleaved compilation
is EXACT (closed-form Bernoulli marginals and joints to 1e-12,
multi-tick, reboot KronDelta, independence). Slow half (real IPPC 2011
env): preservation against the authoritative RDDL simulator and the
benchmark — metis exact-measure greedy vs pyRDDLGym's Random/NoOp
agents on competition instance 1."""

import importlib.util

import pytest
from pytest import approx

from metis.domains.sysadmin import expected_running, greedy_plan, project
from metis.domains.sysadmin.model import running_marginal

EXACT = dict(abs=1e-12)

ADJ2 = {"A": ["B"], "B": ["A"]}


def joint(pr, atoms):
    return sum(p for (_, k), p in pr.final.items()
               if all((a, 1) in k for a in atoms))


def test_bernoulli_marginals_exact():
    pr = project(["A", "B"], ADJ2, {"A", "B"}, ticks=1)
    assert running_marginal(pr, "A") == approx(0.95, **EXACT)
    assert running_marginal(pr, "B") == approx(0.95, **EXACT)
    # synchronous product: outcomes independent given the snapshot
    assert joint(pr, ["run(A)", "run(B)"]) == approx(0.95 * 0.95,
                                                     **EXACT)
    pr = project(["A", "B"], ADJ2, {"A"}, ticks=1)
    assert running_marginal(pr, "A") == approx(0.70, **EXACT)
    assert running_marginal(pr, "B") == approx(0.04, **EXACT)
    assert joint(pr, ["run(A)", "run(B)"]) == approx(0.70 * 0.04,
                                                     **EXACT)


def test_multi_tick_chain_exact():
    # isolated computer: p_up = .95; two ticks: .95^2 + .05*.04
    pr = project(["A"], {}, {"A"}, ticks=2)
    assert running_marginal(pr, "A") == approx(0.9045, **EXACT)
    assert pr.p_done() == 0.0            # the tick loop never absorbs


def test_reboot_is_krondelta():
    pr = project(["A", "B"], ADJ2, {"A"}, ticks=1, reboot="B")
    assert running_marginal(pr, "B") == approx(1.0, **EXACT)
    assert running_marginal(pr, "A") == approx(0.70, **EXACT)


def test_greedy_planner_prefers_the_bridge_computer():
    adj = {"A": ["B"], "B": ["A", "C"], "C": ["B"]}
    res = greedy_plan(["A", "B", "C"], adj, {"A", "C"})
    assert res.action == "B"
    assert res.scores["B"][0] < res.scores[None][0]
    # and expected_running is a genuine expectation
    pr = project(["A", "B", "C"], adj, {"A", "C"}, ticks=1, reboot="B")
    assert expected_running(pr) == approx(
        sum(running_marginal(pr, c) for c in "ABC"), **EXACT)


# ---- the real environment --------------------------------------------

requires_rddl = pytest.mark.skipif(
    importlib.util.find_spec("pyRDDLGym") is None
    or importlib.util.find_spec("rddlrepository") is None,
    reason="pyRDDLGym/rddlrepository not installed")


@pytest.mark.slow
@requires_rddl
def test_preservation_against_rddl_simulator():
    """Our filter's next-state marginals must match the authoritative
    RDDL simulator empirically (noop from the instance init state)."""
    from metis.domains.sysadmin.rddl_adapter import load, running_of
    spec = load("1")
    env = spec["env"]
    obs, _ = env.reset(seed=0)
    s0 = running_of(obs)
    pr = project(spec["computers"], spec["in_neighbors"], s0, ticks=1,
                 reboot_prob=spec["reboot_prob"])
    want = {c: running_marginal(pr, c) for c in spec["computers"]}
    n = 400
    counts = {c: 0 for c in spec["computers"]}
    for k in range(n):
        env.reset(seed=k)
        nxt, _, _, _, _ = env.step({})
        for c in running_of(nxt):
            counts[c] += 1
    for c in spec["computers"]:
        assert counts[c] / n == approx(want[c], abs=0.06), \
            (c, counts[c] / n, want[c])
    env.close()


@pytest.mark.slow
@requires_rddl
def test_benchmark_instance1_vs_baselines():
    from metis.domains.sysadmin.rddl_adapter import agent_metis, \
        agent_noop, agent_random, load, run_episode
    spec = load("1")
    n_ep = 10
    means = {}
    for name, make in (("metis", agent_metis), ("random", agent_random),
                       ("noop", agent_noop)):
        agent = make(spec)
        rs = [run_episode(spec, agent, seed=100 + i)
              for i in range(n_ep)]
        means[name] = sum(rs) / n_ep
    print("\nSysAdmin IPPC2011 instance 1, mean reward over "
          f"{n_ep} episodes: {means}")
    assert means["metis"] > means["random"], means
    assert means["metis"] > means["noop"], means
    spec["env"].close()
