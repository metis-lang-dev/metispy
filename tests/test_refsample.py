"""Reference-sampler gates (docs/rewrite/06-sampler-wire.md), fast
half: the spec'd RNG vectors, presentation-invariant traces, and
the sampler-vs-filter agreement on FULL-CLASS programs (the love
triangle: multiplicities, quiescence — everything class F excludes
still verifies here)."""

from pytest import approx

from metis.kernel import ForwardProjection
from metis.kernel.refsample import SplitMix64, run_trace, \
    sample_final


def test_splitmix64_vectors():
    r = SplitMix64(42)
    assert [f"{r.unit():.17g}" for _ in range(3)] == [
        "0.74156487877182331", "0.1599103928769201",
        "0.27860113025513866"]


def test_traces_are_presentation_invariant():
    """Canonical event order: a shuffled catalog samples the SAME
    traces — the run identity quotient, operational."""
    from metis.kernel import Program
    from lovetriangle import INIT, love_triangle
    prog = love_triangle().catalog.ground(INIT)
    shuffled = Program(
        stages={s: list(reversed(cs))
                for s, cs in prog.stages.items()},
        links=list(reversed(prog.links)),
        init_stage=prog.init_stage, init=dict(prog.init))
    for seed in range(1, 8):
        assert run_trace(prog, 4, seed) == \
            run_trace(shuffled, 4, seed), seed


def test_full_class_empirical_matches_exact():
    """The measure agreement on the ORIGIN program — multiplicity-3
    heartbreak and all: empirical marginals track the exact filter
    (Thm 1's distribution, drawn through the portable RNG)."""
    from lovetriangle import love_triangle
    init = {"eros(helena,hermia)": 1, "anger(helena,hermia)": 3,
            "eros(hermia,helena)": 1}
    prog = love_triangle().catalog.ground(init)
    pr = ForwardProjection(prog, 2)
    want = sum(p for (_, k), p in pr.final.items()
               if ("eros(helena,hermia)", 1) in k)
    d = sample_final(prog, 2, n=4000)
    got = sum(p for (_, k), p in d.items()
              if ("eros(helena,hermia)", 1) in k)
    assert got == approx(want, abs=0.03)
