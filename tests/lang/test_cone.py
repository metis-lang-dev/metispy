"""Spec 11 (branch spec11-cone): the influence cone — the P1 sweep
closes (no out-of-cone event writes a cone atom; cone events read only
cone atoms), and the quotient filter agrees EXACTLY with an independent
full-program enumerator under the same readout (§5 gate 1)."""
import io
from dataclasses import replace
from pathlib import Path

import pytest

from metis.kernel.filter import StateBudgetExceeded
from metis.lang.cone import (check_closed, cone_readout, full_readout,
                             qkey, quotient, sweep)
from metis.lang.repl import Repl

DOMAINS = Path(__file__).resolve().parents[3] / \
    "metis-catalog/domains/ceptre"
pytestmark = pytest.mark.skipif(not DOMAINS.exists(),
                                reason="metis-catalog not checked out")


def shell(name, seed, k):
    r = Repl(DOMAINS / name, seed=seed, stdout=io.StringIO())
    r.onecmd(f"run {k}")
    return r


def names(r, c):
    return frozenset(r.prog.events()[i][2].name for i in c.events)


def agree(r, outcomes, readouts, budget=200_000):
    full = replace(r.prog, init_stage=r.stage, init=dict(r.counts))
    n = 0
    for y in outcomes:
        c = sweep(r.prog, [y], r.stage)
        assert check_closed(r.prog, c, r.stage) == []
        q = quotient(r.prog, c, r.stage, r.counts)
        for ro in readouts:
            try:
                pc, _ = cone_readout(q, y, ro, budget, frame_cap=64)
                pf = full_readout(full, names(r, c), y, ro, budget,
                                  max_steps=200)
            except (StateBudgetExceeded, RuntimeError):
                continue
            assert pc == pytest.approx(pf, abs=1e-12), (y, ro)
            n += 1
    return n


@pytest.mark.parametrize("seed,k", [(1, 0), (1, 3), (3, 2), (7, 1)])
def test_3lp_cone_equals_full_exactly(seed, k):
    r = shell("three_pigs/three_pigs.llp", seed, k)
    atoms = sorted(r.atoms)
    n = agree(r, atoms + ["!" + a for a in atoms],
              [("cone", 1), ("cone", 3), ("frame",)])
    assert n >= 3 * len(atoms)


def test_love_triangle_cone_equals_full_on_answerable_cases():
    r = shell("love_triangle/love_triangle.llp", 1, 2)
    assert agree(r, ["anger(lysander,helena)", "!eros(helena,hermia)"],
                 [("cone", 1), ("cone", 3)]) == 4


def test_the_sweep_closes_and_the_quotient_has_its_own_key():
    r = shell("three_pigs/three_pigs.llp", 1, 0)
    c = sweep(r.prog, ["brickhouse"], r.stage)
    names = {r.prog.events()[i][2].name for i in c.events}
    # brickhouse is written by build_brick only (blow_brick only READS
    # it); build_brick consumes a pig, so every builder competing for
    # pigs joins; the blows write no cone atom and stay out
    assert names == {"build_brick", "build_stick", "build_straw"}
    assert c.atoms == {"brickhouse", "pig", "bricks", "sticks", "straw"}
    q = quotient(r.prog, c, r.stage, r.counts)
    assert qkey(q) != r.key and len(qkey(q)) == 64
    assert r.key == Repl(DOMAINS / "three_pigs/three_pigs.llp",
                         stdout=io.StringIO()).key     # keys never move
    # a hand-broken cone is caught (the construction-bug detector)
    broken = c._replace(events=frozenset())
    assert any("writes" in b for b in check_closed(r.prog, broken,
                                                   r.stage))
    # masks prune first: blocking the only writer leaves an empty cone
    c2 = sweep(r.prog, ["brickhouse"], r.stage,
               masked=lambda n: n == "build_brick")
    assert not c2.events and c2.atoms == {"brickhouse"}
    assert check_closed(r.prog, c2, r.stage,
                        masked=lambda n: n == "build_brick") == []


def test_tragedy_prospective_cone_is_the_whole_world():
    """The §5 finding, pinned: on tragedy every outcome's ground cone is
    (nearly) the whole program — `at(C,L)` is persisted by nearly every
    rule and written by travelTo; despair, murder, anger and love feed
    each other — so P1/P2 alone cannot reduce the filter there."""
    r = shell("tragedy/tragedy.llp", 132, 74)
    c = sweep(r.prog, ["suicidal(mercutio)"], r.stage)
    assert check_closed(r.prog, c, r.stage) == []
    assert len(c.events) == c.n_events and len(c.atoms) >= c.n_atoms - 1


# -- the instruments on the cone (spec 11 P3-P5) ------------------------

FIXTURE = Path(__file__).resolve().parent / "fixtures/disjoint_chains.llp"


def disjoint():
    out = io.StringIO()
    r = Repl(FIXTURE, seed=1, stdout=out)
    for line in ("step e1", "step b_one", "step e2", "step b_two",
                 "step e3", "step e4"):
        r.onecmd(line)

    def say(line):
        out.truncate(0)
        out.seek(0)
        r.onecmd(line)
        return out.getvalue()
    return r, say


def test_gate3_cone_steps_read_pn_zero_where_global_n_leaks():
    """Resource-disjoint pieces: blocking the noise event e1 cannot
    cause b2. A global 4-event horizon leaks (the noise competes for
    steps); @N cone steps reads PN = 0 exactly — and the full validation
    path agrees with both."""
    r, say = disjoint()
    leak = say("necessity 1 b2 4")
    assert "PN ≈ 0.5208" in leak and "scheduling-sensitive" in leak
    assert "PN ≈ 0.5208" in say("necessity 1 b2 4 full")
    cone = say("necessity 1 b2")
    assert "@2 cone steps (or when the cone settles, if sooner) after #0" in cone
    assert "PN ≈ 0: #1 e1 was not necessary for b2" in cone
    assert "tracking 3 of 8 atoms · 2 of 6 events" in cone
    assert "PN ≈ 0:" in say("necessity 1 b2 @2 full")
    assert "PN ≈ 0:" in say("necessity 1 b2 @frame")


def test_quotient_key_cache_never_serves_a_whole_world_answer():
    """Regression: both arms share the B quotient (same key) but differ
    outside it; a global answer must not be served from the cone cache."""
    r, say = disjoint()
    out = say("necessity 1 b2 4")
    assert "without #1   P = 0.4792" in out and "by chance    P = 0.2867" \
        in out


def test_sampled_tier_ci_covers_the_exact_value():
    """§5 gate 1, the sampled tier's own check: across exactly
    answerable questions the 95% CI covers the exact value at (about)
    its stated rate."""
    from metis.lang.measure import Arm, measure
    cases = []
    for name, seed, k in (("three_pigs/three_pigs.llp", 1, 2), ("three_pigs/three_pigs.llp", 3, 1),
                          ("love_triangle/love_triangle.llp", 1, 2)):
        r = shell(name, seed, k)
        for a in sorted(r.atoms)[:6]:
            cases.append((r, a))
    covered = 0
    for r, a in cases:
        arm = [Arm(r.stage, dict(r.counts), frozenset())]
        ex = measure(r.prog, arm, a, ("cone", 3), "exact").ps[0]
        sm = measure(r.prog, arm, a, ("cone", 3), "sampled")
        covered += abs(sm.ps[0] - ex) <= sm.cis[0] + 1e-12
        assert sm.tier.startswith("sampled n=")
    assert covered >= 0.85 * len(cases), (covered, len(cases))


@pytest.mark.slow
def test_gate2_tragedy_pn_answers_in_seconds():
    """§5 gate 2: spec 10 §8's tragedy PN (seed 132, murder -> suicidal)
    answers at the interactive budget in seconds, sampled ± ci, and the
    exact 0.9846 (offline, 6M states) lies inside the interval."""
    import re
    import time
    r = shell("tragedy/tragedy.llp", 132, 80)
    out = io.StringIO()
    r.stdout = out
    t = time.time()
    r.onecmd("necessity 75 suicidal(mercutio)")
    took = time.time() - t
    text = out.getvalue()
    assert took < 20, took
    m = re.search(r"PN ≈ ([0-9.]+) ± ([0-9.]+)", text)
    assert m and abs(float(m[1]) - 0.9846) <= float(m[2]) + 1e-9, text
    assert "tier         sampled n=" in text and "exact refused" in text


def test_love_triangle_coupling_demo_names_the_inhibitor():
    """The coupling demo (spec 11 §5 gate 3, amended): #1 feeds Hermia's
    anger at Helena, move_on then consumes eros(hermia,helena) — the eros
    the jealousy making Lysander angry at Helena reads. #1 works AGAINST
    that outcome: blocking it RAISES P, so it is not necessary, and the
    report says the PN is the outcome's own uncertainty."""
    r = shell("love_triangle/love_triangle.llp", 1, 6)
    out = io.StringIO()
    r.stdout = out
    r.onecmd("necessity 1 anger(lysander,helena)")
    text = out.getvalue()
    assert "was not necessary for anger(lysander,helena)" in text
    assert "Blocking #1 RAISES P(anger(lysander,helena)) from 0.676 to " \
        "0.8828: #1 worked AGAINST" in text
    assert "@6 cone steps" in text and "tier         exact (cone)" in text
