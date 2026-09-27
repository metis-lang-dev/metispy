"""Grounding parity vs ll2pg.fol — names, multisets, and the measure of
a fully-FO program identical to the oracle. Pins the substitution
collapse (X=Y -> multiplicity 2 -> falling-factorial pricing) and
distinct-pruning semantics."""

import pytest
from pytest import approx

from metis.kernel import ClauseSchema, ForwardProjection, Program, \
    Signature, ground_schema, pat
from oracle import oracle_modules, oracle_projection, requires_oracle

pytestmark = requires_oracle

SIG = Signature({"tok": ["s", "t"], "obj": ["m", "n", "o"]})


def fixtures():
    """(metis schema, equivalent fol.FORule) pairs."""
    fol = oracle_modules("fol")[0]
    fsig = fol.Signature(dict(SIG.types))
    pairs = [
        # two typed vars, mixed patterns
        (ClauseSchema("mk", stage="s",
                      vars={"X": "tok", "Y": "obj"},
                      consume=[pat("p", "X"), pat("q", "X", "Y")],
                      produce=[pat("r", "Y")],
                      persist=[pat("k", "X")]),
         fol.FORule("mk", vars={"X": "tok", "Y": "obj"},
                    consume=[("p", ("X",)), ("q", ("X", "Y"))],
                    produce=[("r", ("Y",))],
                    persist=[("k", ("X",))])),
        # substitution collapse: X=Y merges premises to multiplicity 2
        (ClauseSchema("pair", stage="s",
                      vars={"X": "obj", "Y": "obj"},
                      consume=[pat("p", "X"), pat("p", "Y")],
                      produce=[pat("d")]),
         fol.FORule("pair", vars={"X": "obj", "Y": "obj"},
                    consume=[("p", ("X",)), ("p", ("Y",))],
                    produce=[("d", ())])),
        # distinct side-condition pruning
        (ClauseSchema("swap", stage="s",
                      vars={"X": "obj", "Y": "obj"},
                      distinct=[("X", "Y")],
                      consume=[pat("at", "X", "Y")],
                      produce=[pat("at", "Y", "X")]),
         fol.FORule("swap", vars={"X": "obj", "Y": "obj"},
                    distinct=[("X", "Y")],
                    consume=[("at", ("X", "Y"))],
                    produce=[("at", ("Y", "X"))])),
    ]
    return fol, fsig, pairs


def test_ground_instances_identical():
    fol, fsig, pairs = fixtures()
    for schema, forule in pairs:
        ours = ground_schema(schema, SIG)
        ref = fol.ground_rule(forule, fsig)
        assert len(ours) == len(ref)
        for a, b in zip(ours, ref):
            assert a.name == b.name
            assert a.consume == b.consume
            assert a.produce == b.produce
            assert a.persist == b.persist


def test_collapse_prices_falling_factorial():
    _, _, pairs = fixtures()
    schema = pairs[1][0]
    ground = ground_schema(schema, SIG)
    collapsed = [c for c in ground if c.name == "pair[X=m,Y=m]"]
    assert collapsed and collapsed[0].consume == {"p(m)": 2}


def test_fo_program_measure_parity():
    """Ground the collapse program and compare the full measure with the
    oracle filter: W_r = sum over substitutions, x^(2) on collapse."""
    _, _, pairs = fixtures()
    schema = pairs[1][0]
    prog = Program(stages={"s": ground_schema(schema, SIG)},
                   init_stage="s", init={"p(m)": 2, "p(n)": 1})
    for steps in (1, 2, 3):
        ours = ForwardProjection(prog, steps)
        ref = oracle_projection(prog, steps)
        assert ours.final_distribution("d") == approx(
            ref.final_distribution("d"), abs=1e-12)
        assert ours.p_done() == approx(ref.p_done(), abs=1e-12)
        assert ours.n_reachable() == ref.n_reachable()
