"""Proposition 6 gates: every emitted IR net is a Bayesian proof-net
in the EFD sense (docs/LL-PGM.pdf, Def 3.1 + 3.2, Lemma 3.1) — box
rule, SSA single-writer conclusions, no dangling premise, polarized
acyclicity, row-stochastic CPTs. Positive corpus mirrors the IR
goldens (gen_ir_goldens.py) plus the unknowns lift and a guard-box
program; adversarial mutations prove the checker rejects each
condition, so the gate is not vacuous."""

from types import SimpleNamespace

import numpy as np
from pytest import raises

from metis.kernel.bpn import BpnError, BpnReport, bpn_boxes, bpn_check
from metis.kernel.factored import Factor
from metis.kernel.factorize import FactoredProgram, _Site, _Write
from metis.lang import compile_catalog, factor_hooks, parse
from test_factorize import CASCADE, TRIWAY, cascade_build, ring_build, \
    sys_build

# ---- the positive corpus (the IR-golden programs) ----------------------


def test_sysadmin_corpus_is_bpn():
    for running, steps, reboot in (({"a", "b"}, 9, None),
                                   ({"a", "b"}, 5, None),
                                   ({"a"}, 18, "b")):
        prog, hooks, _ = sys_build(running, reboot)
        fp = FactoredProgram(prog, steps, hooks)
        bpn_check(fp, tables=True)


def test_cascade_shape():
    prog, hooks, _ = cascade_build()
    r = bpn_check(FactoredProgram(prog, 3, hooks), tables=True)
    # coin site + committed-read site + four occurrence writes; the
    # a#0 wire feeds X2's scope (contraction), za/b/zb are Δ
    assert r == BpnReport(boxes=6, wires=6, contractions=2,
                          pending=("b#2", "za#0", "zb#2"),
                          stall_rows=0)


def test_triway_shape():
    doc = compile_catalog(parse(TRIWAY))
    prog = doc.catalog.ground({"tk": 1})
    hooks = factor_hooks(prog, doc.ports,
                         {"w1": lambda a, c: 0.5,
                          "w2": lambda a, c: 0.3},
                         doc.catalog.horn_db())
    r = bpn_check(FactoredProgram(prog, 1, hooks), tables=True)
    assert r == BpnReport(boxes=4, wires=4, contractions=2,
                          pending=("f#0", "r#0", "s#0"),
                          stall_rows=0)


def test_ring_two_ticks_is_bpn():
    n = 6
    prog, hooks, _ = ring_build(n, set(range(0, n, 2)))
    fp = FactoredProgram(prog, 2 * (3 * n + 3), hooks)
    bpn_check(fp, tables=True)


def test_unknowns_lift_prior_boxes():
    prog, hooks, _ = sys_build({"a", "b"})
    unknowns = ("run(a)", "down(a)", "run(b)", "down(b)")
    fp = FactoredProgram(prog, 9, hooks, unknowns)
    bpn_check(fp, tables=True)
    priors = [b for b in bpn_boxes(fp) if b[0].startswith("prior")]
    assert len(priors) == 4 and all(p == () for _, _, p in priors)


GUARDED = """
catalog guarded 1.
provenance hand "bpn gate: the guard constraint box" "2026-07-21".
layers (dyn).
stages (ua ub).
namespace st produce (dyn).
pred a : st.
pred za : st.
pred ta : st.
pred tb : st.
weight pa reads {}.
stage ua {
  %% coin for a.
  aon [dyn] : ta -o a  @w pa.
  %% complement branch.
  aoff [dyn] : ta -o za  @w ~pa.
}
stage ub {
  %% produces a onto a maybe-present a — emitted as a guard box.
  mk [dyn] : tb -o a.
}
%% commit -> gate.
qui ab [dyn] : ua -o ub * tb.
"""


def test_guard_box_is_typed():
    """Double production onto a variable occurrence emits an observed
    constraint box — still a well-typed bpn; the (unreachable-or-Z)
    semantics is the runtime's job, not the typing's."""
    doc = compile_catalog(parse(GUARDED))
    prog = doc.catalog.ground({"ta": 1})
    hooks = factor_hooks(prog, doc.ports, {"pa": lambda a, c: 0.6},
                         doc.catalog.horn_db())
    fp = FactoredProgram(prog, 3, hooks)
    assert fp._guards                      # fixture exercises guards
    bpn_check(fp, tables=True)
    guards = [b for b in bpn_boxes(fp) if b[0].startswith("guard")]
    assert len(guards) == 1


# ---- the adversarial mutations (the checker is not vacuous) ------------


def fake(sites=(), writes=(), guards=None, unknowns=None, cards=None):
    return SimpleNamespace(_sites=list(sites), _writes=list(writes),
                           _guards=guards or {},
                           _unknown_vars=unknowns or {},
                           _cards=cards or {})


def test_rejects_self_parent():
    f = fake(sites=[_Site("X0", ("X0",), [])], cards={"X0": 2})
    with raises(BpnError, match="its own premise"):
        bpn_check(f)


def test_rejects_double_writer():
    f = fake(sites=[_Site("X0", (), [])],
             writes=[_Write("a#0", "X0", 2, None, [0, 1]),
                     _Write("a#0", "X0", 2, None, [1, 0])],
             cards={"X0": 2, "a#0": 2})
    with raises(BpnError, match="concluded twice"):
        bpn_check(f)


def test_rejects_dangling_premise():
    f = fake(sites=[_Site("X0", ("ghost#1",), [])], cards={"X0": 2})
    with raises(BpnError, match="conclusion of no box"):
        bpn_check(f)


def test_rejects_polarized_cycle():
    f = fake(sites=[_Site("X0", (), [])],
             writes=[_Write("a#0", "X0", 2, "b#0", [None, 1]),
                     _Write("b#0", "X0", 2, "a#0", [None, 1])],
             cards={"X0": 2, "a#0": 2, "b#0": 2})
    with raises(BpnError, match="not a DAG"):
        bpn_check(f)


def test_rejects_orphan_variable():
    f = fake(sites=[_Site("X0", (), [])],
             cards={"X0": 2, "phantom#3": 2})
    with raises(BpnError, match="no producing box"):
        bpn_check(f)


def test_rejects_subnormalized_row():
    f = fake(sites=[_Site("X0", (), [])], cards={"X0": 2})
    f._build = lambda: [Factor(("X0",), np.array([0.3, 0.2]))]
    with raises(BpnError, match="neither a distribution nor a stall"):
        bpn_check(f, tables=True)


def test_ir_wire_refuses_ill_typed_artifact():
    """The serialization seam runs the typing gate: a corrupted
    emission never reaches the golden wire the C++ side trusts."""
    from metis.kernel.ir import ir_case_lines
    prog, hooks, _ = cascade_build()
    fp = FactoredProgram(prog, 3, hooks)
    ir_case_lines("ok", fp, ["b"])         # well-typed: serializes
    fp._writes.append(fp._writes[0])       # break SSA single-writer
    with raises(BpnError, match="concluded twice"):
        ir_case_lines("bad", fp, ["b"])
