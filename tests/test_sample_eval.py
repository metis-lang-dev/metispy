"""Sampling-evaluator gates: the sampler's policy hook preserves the
policy-weighted measure; SampledProjection matches the exact filter
within sampling error at the cliff edge and crosses past it."""

import time
from pathlib import Path

from pytest import approx

from metis.kernel import ForwardProjection, SampledProjection, \
    expected_on, sample_final, sync_policy
from metis.lang import load_llp, weight_policy
from metis.domains.gol.model import gol_p_on, gol_program

ADJ = {"a": ["b"], "b": ["a"]}
SYS = Path("metis/domains/sysadmin/catalogs/sysadmin2.llp")


def _sys_policy():
    doc = load_llp(SYS)
    prog = doc.catalog.ground({"m(i0)": 1, "run(a)": 1, "down(b)": 1})

    def cpt(args, counts):
        c = args[0]
        r = sum(1 for y in ADJ[c] if f"prev({y})" in counts)
        return 0.45 + 0.5 * (1 + r) / (1 + len(ADJ[c]))

    pol = weight_policy(prog, doc.ports,
                        {"cpt": cpt, "rp": lambda a, c: 0.04})
    return prog, pol


def test_sampler_policy_hook_preserves_the_weighted_measure():
    prog, pol = _sys_policy()
    ex = ForwardProjection(prog, 9, policy=pol)
    emp = sample_final(prog, 9, 4000, seed=11, policy=pol)
    for s in set(ex.final) | set(emp):
        assert emp.get(s, 0.0) == approx(ex.final.get(s, 0.0),
                                         abs=0.03), s


def _ring(n, noise=0.05):
    cells = [f"c{i}" for i in range(n)]
    nbrs = {c: [cells[(i - 1) % n], cells[(i + 1) % n]]
            for i, c in enumerate(cells)}
    prog = gol_program(cells, set(cells))
    pol = sync_policy(prog, gol_p_on(nbrs, {c: noise for c in cells}))
    return prog, pol, 3 * n + 3


def test_sampled_projection_matches_exact_at_the_cliff_edge():
    prog, pol, steps = _ring(12)
    ex = ForwardProjection(prog, steps, policy=pol)
    sp = SampledProjection(prog, steps, n=1500, seed=7, policy=pol)
    assert expected_on(sp, "alive") == approx(
        expected_on(ex, "alive"), abs=0.35)      # 3 sigma-ish at n=1500
    assert sp.p_done() == approx(ex.p_done(), abs=0.05)


def test_sampled_projection_crosses_the_cliff():
    # n=18: exact support would be 2^18 = 262k states; sampling is O(n)
    prog, pol, steps = _ring(18)
    t0 = time.time()
    sp = SampledProjection(prog, steps, n=800, seed=7, policy=pol)
    dt = time.time() - t0
    e = expected_on(sp, "alive")
    assert 12.0 < e < 18.0        # all-alive ring holds most cells
    assert dt < 30.0, dt          # tractable where exact is not


def test_history_option_slices_are_distributions():
    prog, pol, steps = _ring(6)
    sp = SampledProjection(prog, steps, n=500, seed=7, policy=pol,
                           history=True)
    assert len(sp.history) == steps + 1
    for d in sp.history:
        assert sum(d.values()) == approx(1.0, abs=1e-9)
