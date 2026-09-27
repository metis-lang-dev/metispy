"""Step-1 domain gates for the factored evaluator: real IPPC
topologies (GoL 4x4 = instance 5, past the 2^16 support cliff —
2-tick, which the dict-factor v1 could not finish: numpy factors +
min-fill (the profile-ranked items) run it in ~0.04s/marginal;
SysAdmin instance 2) evaluated exactly, cross-checked against the
sampler, with profile numbers printed for the engine-TODO ranking."""

import importlib.util
import time

import pytest
from pytest import approx

from metis.kernel import SampledProjection, marginal_on, sync_policy
from metis.kernel.factored import FactoredSync

requires_rddl = pytest.mark.skipif(
    importlib.util.find_spec("pyRDDLGym") is None,
    reason="pyRDDLGym not installed")


@pytest.mark.slow
@requires_rddl
def test_gol_instance5_factored_exact_past_the_cliff():
    from metis.domains.gol.rddl_adapter import alive_of, load
    from metis.domains.gol.model import gol_p_on, gol_program
    spec = load("5")                      # 16 cells: 2^16 support
    cells, nbrs, noise = spec["cells"], spec["neighbors"], spec["noise"]
    obs, _ = spec["env"].reset(seed=0)
    alive = alive_of(obs)

    def p(u, snap, was_on):
        live = sum(snap[v] for v in nbrs[u])
        cond = (2 <= live <= 3) if was_on else (live == 3)
        return 1.0 - noise[u] if cond else noise[u]

    t0 = time.time()
    ev = FactoredSync(cells, nbrs, p, alive).tick(2)
    marg = {c: ev.marginal(c) for c in cells}
    dt = time.time() - t0
    print(f"\nGoL inst5 (n=16) factored 2-tick + 16 marginals: "
          f"{dt:.1f}s")
    prog = gol_program(cells, alive)
    pol = sync_policy(prog, gol_p_on(nbrs, noise))
    sp = SampledProjection(prog, 2 * (3 * 16 + 3), n=1200, seed=7,
                           policy=pol)
    for c in cells:
        assert marg[c] == approx(marginal_on(sp, c, "alive"),
                                 abs=0.06), c
    assert dt < 120
    spec["env"].close()


@pytest.mark.slow
@requires_rddl
def test_sysadmin_instance2_factored_exact():
    from metis.domains.sysadmin.rddl_adapter import load, running_of
    spec = load("2")
    comps = spec["computers"]
    ins = spec["in_neighbors"]
    obs, _ = spec["env"].reset(seed=0)
    running = running_of(obs)
    rate = spec["reboot_prob"]

    def p(u, snap, was_on):
        if not was_on:
            return rate
        k = ins.get(u, [])
        r = sum(snap[v] for v in k if v in snap)
        return 0.45 + 0.5 * (1 + r) / (1 + len(k))

    parents = {u: ins.get(u, []) for u in comps}
    t0 = time.time()
    ev = FactoredSync(comps, parents, p, running).tick(1)
    marg = {c: ev.marginal(c) for c in comps}
    dt = time.time() - t0
    print(f"SysAdmin inst2 (n={len(comps)}) factored 1-tick + "
          f"marginals: {dt:.1f}s")
    # referee: 400 empirical env steps
    n, counts = 400, {c: 0 for c in comps}
    for k in range(n):
        spec["env"].reset(seed=k)
        nxt, _, _, _, _ = spec["env"].step({})
        for c in running_of(nxt):
            counts[c] += 1
    for c in comps:
        assert marg[c] == approx(counts[c] / n, abs=0.07), c
    assert dt < 60
    spec["env"].close()


@pytest.mark.slow
@requires_rddl
def test_gol_instance5_live_factored_play():
    """LIVE play on instance 5 (16 cells — the support filter cannot
    even project one tick): the factored one-step-exact agent beats
    random and noop on cumulative alive reward in the real IPPC
    environment."""
    from metis.domains.gol.rddl_adapter import agent_metis_factored, \
        agent_noop, agent_random, load
    spec = load("5")
    env = spec["env"]

    def run(agent, episodes=2, steps=10):
        total = 0.0
        for ep in range(episodes):
            obs, _ = env.reset(seed=31 + ep)
            for _ in range(steps):
                obs, r, term, trunc, _ = env.step(agent(obs))
                total += r
                if term or trunc:
                    break
        return total

    t0 = time.time()
    ours = run(agent_metis_factored(spec))
    dt = time.time() - t0
    rand = run(agent_random(spec))
    noop = run(agent_noop(spec))
    print(f"\nGoL inst5 live: metis={ours:.0f} random={rand:.0f} "
          f"noop={noop:.0f} ({dt:.1f}s for ours)")
    assert ours > rand
    assert ours > noop
    assert dt < 120, dt
    env.close()
