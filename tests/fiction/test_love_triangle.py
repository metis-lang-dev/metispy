"""The lineage gates: the Ceptre tutorial's love triangle as a metis
domain (docs/rewrite/05-ceptre-lineage.md). What the origin gives us
verbatim (the operational core), and what metis adds on top of the
SAME program: an exact measure (Ceptre samples; we also sum), the
falling-factorial pricing of the multiplicity-3 heartbreak (Thm 1 —
this IS Ceptre's token-individuated choice, made a number), the
drama manager as a policy hook (a reweighting of the same measure,
inference stays exact), MAP derivation as the plot summary, and
canonical identity."""

from pytest import approx

from metis.kernel import ForwardProjection, program_key, \
    sample_final
from metis.kernel.posterior import map_derivation
from metis.lang import parse, pretty
from metis.domains.fiction import INIT, love_triangle
from metis.domains.fiction.model import CATALOG, drama_policy

EXACT = dict(abs=1e-9)


def _p(pr, atom, n=1):
    return sum(q for (_, k), q in pr.final.items()
               if (atom, n) in k)


def test_the_port_roundtrips_and_typechecks():
    a = parse(CATALOG.read_text())
    assert parse(pretty(a)) == a
    doc = love_triangle()                  # compile = typecheck
    prog = doc.catalog.ground(INIT)
    assert program_key(prog) == program_key(
        doc.catalog.ground(dict(INIT)))    # canonical identity


def test_act_one_exact_choice_distribution():
    """Five jealousy instances are enabled in the tutorial context
    (hermia's love of helena gives A=hermia TWO rivals); at parity
    weights each fires 1/5. anger(hermia,helena) arises from two of
    them: P = 2/5 — a number Ceptre's interpreter samples but never
    states."""
    prog = love_triangle().catalog.ground(INIT)
    pr = ForwardProjection(prog, 1)
    assert _p(pr, "anger(hermia,helena)") == approx(0.4, **EXACT)
    assert _p(pr, "anger(hermia,hermia)") == approx(0.2, **EXACT)


def test_heartbreak_prices_as_the_falling_factorial():
    """move_on consumes eros + anger^3: at anger count 3 the
    interpreter has 3! = 6 token-individuated instances (Thm 1), so
    against two enabled jealousy instances P(move_on) = 6/8. The
    multiplicity is priced EXACTLY — the same semantics the origin
    executes, measurable."""
    init = {"eros(helena,hermia)": 1, "anger(helena,hermia)": 3,
            "eros(hermia,helena)": 1}
    prog = love_triangle().catalog.ground(init)
    pr = ForwardProjection(prog, 1)
    assert _p(pr, "eros(helena,hermia)", 1) == approx(
        2 / 8, **EXACT)                    # love survives iff not move_on
    assert 1 - _p(pr, "eros(helena,hermia)", 1) == approx(
        6 / 8, **EXACT)


def test_drama_manager_is_a_reweighting_of_the_same_measure():
    """The BENCHMARKS §7 claim, executable: the drama manager is the
    policy hook scaling move_on. factor 1/2 -> P(heartbreak) =
    3/(3+2) = 3/5; the measure renormalizes, exactness holds."""
    init = {"eros(helena,hermia)": 1, "anger(helena,hermia)": 3,
            "eros(hermia,helena)": 1}
    prog = love_triangle().catalog.ground(init)
    pr = ForwardProjection(prog, 1,
                           policy=drama_policy(prog, 0.5))
    assert 1 - _p(pr, "eros(helena,hermia)", 1) == approx(
        3 / 5, **EXACT)


def test_sampler_agrees_with_the_filter():
    """Ceptre's readout (sampling) and metis's (exact filtering) are
    readouts of ONE object."""
    prog = love_triangle().catalog.ground(INIT)
    pr = ForwardProjection(prog, 3)
    want = _p(pr, "anger(hermia,helena)")
    got = sample_final(prog, 3, n=4000, seed=11)
    p = sum(q for (_, k), q in got.items()
            if ("anger(hermia,helena)", 1) in k)
    assert p == approx(want, abs=0.03)


def test_map_derivation_is_the_plot_summary():
    """argmax over derivations = the most probable plot. With the
    3-anger context the modal first event is the heartbreak."""
    init = {"eros(helena,hermia)": 1, "anger(helena,hermia)": 3,
            "eros(hermia,helena)": 1}
    prog = love_triangle().catalog.ground(init)
    res = map_derivation(ForwardProjection(prog, 1))
    names = [prog.events()[e][2].name for e in res.events]
    assert names and names[0].startswith("move_on")
