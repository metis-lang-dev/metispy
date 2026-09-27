"""The road corpus end to end: guard-pruned grounding matches hand
enumeration; the one-round scenario's measure is exact by hand; the
sampler agrees; the admitted pack composes into the same measure."""

from pytest import approx

from metis.kernel import ForwardProjection, sample_final
from metis.domains.road import base_catalog, scenario, wheelchair_pack

EXACT = dict(abs=1e-12)


def counts_by_schema(program):
    out = {}
    for body in program.stages.values():
        for c in body:
            out.setdefault(c.name.split("[")[0], 0)
            out[c.name.split("[")[0]] += 1
    return out


def test_guard_pruning_matches_hand_enumeration():
    prog = base_catalog().ground(scenario())
    n = counts_by_schema(prog)
    # adv triples: v0:5 + v1:4 + v2:3 = 12, x2 lanes
    assert n["keep"] == 24
    # faster: V2=v1 (4 cells) + V2=v2 (3) = 7, x2 lanes
    assert n["accel"] == 14
    # slower: V2=v0 (5) + V2=v1 (4) = 9, x2 lanes
    assert n["brake"] == 18
    # others: same kinematics x2 cars
    assert n["okeep"] == 48 and n["obrake"] == 36
    # collide: 2 ordered pairs x 2 lanes x 5 cells x 9 vel pairs
    assert n["collide"] == 180


def test_scenario_measure_exact():
    """ego(l0,c1,v1) behind stationary alfa(l0,c2,v0), one round.
    Behavior priors .55/.2/.15 -> P(crash) = keep's share = 11/18
    (accel jumps to c3, brake stays at c1)."""
    prog = base_catalog().ground(scenario())
    pr = ForwardProjection(prog, 6)
    assert pr.p_done() == approx(1.0, **EXACT)
    assert pr.final_distribution("crash") == approx(
        [7 / 18, 11 / 18], **EXACT)
    # wrecks accompany the crash on exactly that mass
    assert pr.final_distribution("wreck(ego)") == approx(
        [7 / 18, 11 / 18], **EXACT)


def test_scenario_sampler_agrees():
    prog = base_catalog().ground(scenario())
    pr = ForwardProjection(prog, 6)
    emp = sample_final(prog, 6, 4000, seed=13)
    for s in set(pr.final) | set(emp):
        assert emp.get(s, 0.0) == approx(pr.final.get(s, 0.0), abs=0.03)


def test_pack_composes_into_the_measure():
    """Admitted pack rule competes in check-stage CHOICE: caution(c2) is
    certain on non-crash branches, 1/3 on the crash branch (collide
    consumes the st its persist reads) -> P(caution)=16/27; crash mass
    untouched."""
    merged = base_catalog().admit(wheelchair_pack())
    prog = merged.ground(scenario(wc=True))
    pr = ForwardProjection(prog, 8)
    assert pr.p_done() == approx(1.0, **EXACT)
    assert pr.final_distribution("crash") == approx(
        [7 / 18, 11 / 18], **EXACT)
    assert pr.final_distribution("caution(c2)") == approx(
        [11 / 27, 16 / 27], **EXACT)
