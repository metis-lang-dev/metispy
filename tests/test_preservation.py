"""Preservation: the sampler (operational semantics) draws from the
filter's measure — the MORPHISM preservation theorem as a permanent,
in-repo test (no external interpreter binary)."""

from pytest import approx

from metis.kernel import ForwardProjection, sample_final
from fixtures import ALL

N = 4000
TOL = 0.03


def test_sampler_matches_filter_on_all_fixtures():
    for name, (make, steps) in ALL.items():
        pr = ForwardProjection(make(), steps)
        emp = sample_final(make(), steps, N, seed=7)
        support = set(pr.final) | set(emp)
        for s in support:
            assert emp.get(s, 0.0) == approx(pr.final.get(s, 0.0),
                                             abs=TOL), (name, s)


def test_sampler_matches_filter_midrun():
    # partial horizons too — every prefix distribution is the measure
    for name, (make, steps) in ALL.items():
        for k in range(1, steps):
            pr = ForwardProjection(make(), k)
            emp = sample_final(make(), k, N, seed=11)
            for s in set(pr.final) | set(emp):
                assert emp.get(s, 0.0) == approx(
                    pr.final.get(s, 0.0), abs=TOL), (name, k, s)
