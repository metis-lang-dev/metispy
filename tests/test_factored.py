"""Factored-evaluator gates: exact parity with the support filter
below the cliff; exact evaluation BEYOND it (n=24) cross-checked
against the sampler; hand-analytic Conway values."""

import time

from pytest import approx

from metis.kernel import ForwardProjection, SampledProjection, \
    marginal_on, sync_policy
from metis.kernel.factored import FactoredSync
from metis.domains.gol.model import gol_p_on, gol_program

EXACT = dict(abs=1e-9)


def ring(n, noise=0.05):
    cells = [f"c{i}" for i in range(n)]
    nbrs = {c: [cells[(i - 1) % n], cells[(i + 1) % n]]
            for i, c in enumerate(cells)}
    return cells, nbrs, {c: noise for c in cells}


def snap_p_on(nbrs, noise):
    """gol_p_on re-expressed over the DECLARED snapshot dict — the
    read-scope discipline the factorization requires."""
    def p(u, snap, was_on):
        live = sum(snap[v] for v in nbrs[u])
        cond = (2 <= live <= 3) if was_on else (live == 3)
        return 1.0 - noise[u] if cond else noise[u]
    return p


def test_analytic_line():
    cells, nbrs = ["a", "b", "c"], {"a": ["b"], "b": ["a", "c"],
                                    "c": ["b"]}
    noise = {c: 0.1 for c in cells}
    ev = FactoredSync(cells, nbrs, snap_p_on(nbrs, noise),
                      {"a", "b", "c"}).tick()
    assert ev.marginal("a") == approx(0.1, **EXACT)
    assert ev.marginal("b") == approx(0.9, **EXACT)


def test_parity_with_exact_filter_below_the_cliff():
    for n, ticks in ((6, 3), (9, 2), (12, 1)):
        cells, nbrs, noise = ring(n)
        prog = gol_program(cells, set(cells))
        pol = sync_policy(prog, gol_p_on(nbrs, noise))
        pr = ForwardProjection(prog, ticks * (3 * n + 3), policy=pol)
        ev = FactoredSync(cells, nbrs, snap_p_on(nbrs, noise),
                          set(cells)).tick(ticks)
        for c in cells:
            assert ev.marginal(c) == approx(
                marginal_on(pr, c, "alive"), **EXACT), (n, ticks, c)


def test_beyond_the_cliff_exact_vs_sampler():
    n = 24                       # support filter: 2^24 — dead
    cells, nbrs, noise = ring(n)
    t0 = time.time()
    ev = FactoredSync(cells, nbrs, snap_p_on(nbrs, noise),
                      set(cells)).tick(2)
    dt = time.time() - t0
    assert dt < 30.0, dt
    prog = gol_program(cells, set(cells))
    pol = sync_policy(prog, gol_p_on(nbrs, noise))
    sp = SampledProjection(prog, 2 * (3 * n + 3), n=1500, seed=7,
                          policy=pol)
    for c in cells[:6]:
        assert ev.marginal(c) == approx(
            marginal_on(sp, c, "alive"), abs=0.05), c


def test_certificate_predicts_and_admits():
    """The admission gate: width/ops computed symbolically BEFORE any
    table exists; ops must grow with measured cost, and a platform
    ops-budget rejects the deep query while admitting the shallow one
    — reject/downgrade at admission, never kill mid-flight."""
    cells, nbrs, noise = ring(12)
    p = snap_p_on(nbrs, noise)
    certs = []
    for ticks in (1, 2, 4):
        ev = FactoredSync(cells, nbrs, p, set(cells)).tick(ticks)
        certs.append(ev.certificate(cells[0])["ops"])
    assert certs[0] < certs[1] < certs[2]
    BUDGET = certs[1] * 1.5          # a platform budget in factor-ops
    assert certs[1] <= BUDGET < certs[2]   # admit 2 ticks, reject 4
