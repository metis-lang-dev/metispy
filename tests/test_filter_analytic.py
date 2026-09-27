"""Hand-computed measures, exact to 1e-12 — the analytic half of the
Stage 1 gate."""

from pytest import approx

from metis.kernel import ForwardProjection
from fixtures import race, ff, persist, staged

EXACT = dict(abs=1e-12)


def test_race_step1():
    pr = ForwardProjection(race(), 1)
    assert pr.final_distribution("wa") == approx([1 / 3, 2 / 3], **EXACT)


def test_race_converges_order_free():
    pr = ForwardProjection(race(), 4)
    assert pr.final_distribution("wa") == approx([0, 0, 1.0], **EXACT)
    assert pr.final_distribution("wb") == approx([0, 1.0], **EXACT)
    assert pr.p_done() == approx(1.0, **EXACT)


def test_falling_factorial_choice():
    pr = ForwardProjection(ff(), 1)
    assert pr.final_distribution("p") == approx([0.25, 0.75], **EXACT)


def test_persistence():
    pr = ForwardProjection(persist(), 1)
    assert pr.final_distribution("done") == approx([0.1, 0.9], **EXACT)
    # !k read, never decremented, on every branch
    assert pr.final_distribution("k") == approx([0, 0, 0, 1.0], **EXACT)


def test_stage_link_quiescence_absorption():
    pr3 = ForwardProjection(staged(), 3)
    assert pr3.stage_distribution() == approx([0, 1.0, 0], **EXACT)
    pr4 = ForwardProjection(staged(), 4)
    assert pr4.stage_distribution() == approx([0, 0, 1.0], **EXACT)
    assert pr4.final_distribution("flag") == approx([0, 1.0], **EXACT)
    assert pr4.final_distribution("d") == approx([0, 0, 1.0], **EXACT)


def test_policy_prunes_and_renormalizes():
    # event 1 is rb (clauses enumerate in stage order)
    pol = lambda stage, counts, ws: [(i, w) for i, w in ws if i != 1]
    pr = ForwardProjection(race(), 1, policy=pol)
    assert pr.final_distribution("wa") == approx([0, 1.0], **EXACT)


def test_clause_prior_reweights_choice():
    p = race()
    p.stages["run"][1].weight = 2.0     # rb: count 1 x prior 2 = 2 = ra
    pr = ForwardProjection(p, 1)
    assert pr.final_distribution("wa") == approx([0.5, 0.5], **EXACT)


def test_event_distribution_and_reachability():
    pr = ForwardProjection(race(), 1)
    assert pr.event_distribution(0) == approx([2 / 3, 1 / 3, 0], **EXACT)
    assert pr.n_reachable() == 2
