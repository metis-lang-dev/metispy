"""IR item 1 gates: factor-graph emission from ARBITRARY catalog
programs. Parity with the exact filter on the catalog-authored
sysadmin (whole ticks AND mid-cycle); beyond-sync fixtures the sync
template cannot express (a sequential within-tick cascade; a 3-way
CHOICE site); a beyond-cliff ring cross-checked against the sampler;
the admission certificate (symbolic — zero binding calls); and the
defense rejections (opaque ports, multiplicities, double production,
scope violation, reachable stall)."""

import time
from pathlib import Path

from pytest import approx, raises

from metis.kernel import ForwardProjection, SampledProjection
from metis.kernel.factorize import FactoredProgram, FactorizeError
from metis.lang import LangError, compile_catalog, factor_hooks, \
    load_llp, parse, weight_policy

SYS = Path("metis/domains/sysadmin/catalogs/sysadmin2.llp")
EXACT = dict(abs=1e-9)

ADJ = {"a": ["b"], "b": ["a"]}


def sys_registry():
    def cpt(args, counts):
        c = args[0]
        if f"rb({c})" in counts:
            return 1.0
        ins = ADJ[c]
        r = sum(1 for y in ins if f"prev({y})" in counts)
        return 0.45 + 0.5 * (1 + r) / (1 + len(ins))

    def rp(args, counts):
        return 1.0 if f"rb({args[0]})" in counts else 0.04

    return {"cpt": cpt, "rp": rp}


def sys_build(running, reboot=None, registry=None):
    doc = load_llp(SYS)
    init = {"m(i0)": 1}
    for c in ("a", "b"):
        init[f"run({c})" if c in running else f"down({c})"] = 1
    if reboot:
        init[f"rb({reboot})"] = 1
    prog = doc.catalog.ground(init)
    reg = registry or sys_registry()
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    pol = weight_policy(prog, doc.ports, reg)
    return prog, hooks, pol


def exact_marginal(pr, atom):
    return sum(p for (_, k), p in pr.final.items() if (atom, 1) in k)


# ---- parity with the exact filter (catalog-derived, no hand p_on) ------

def test_sysadmin_catalog_parity_whole_ticks():
    for running, ticks, reboot in (({"a", "b"}, 1, None),
                                   ({"a"}, 1, None),
                                   ({"a"}, 1, "b"),
                                   ({"a", "b"}, 2, None),
                                   (set(), 3, None)):
        prog, hooks, pol = sys_build(running, reboot)
        steps = ticks * 9
        pr = ForwardProjection(prog, steps, policy=pol)
        fp = FactoredProgram(prog, steps, hooks)
        for atom in ("run(a)", "run(b)", "down(a)", "down(b)"):
            assert fp.marginal(atom) == approx(
                exact_marginal(pr, atom), **EXACT), \
                (running, ticks, reboot, atom)


def test_sysadmin_catalog_parity_mid_cycle():
    """The unroll unit is the MICRO-step, not the tick — any horizon
    lands on the same slice the filter lands on."""
    prog, hooks, pol = sys_build({"a"})
    for steps in range(1, 10):
        pr = ForwardProjection(prog, steps, policy=pol)
        fp = FactoredProgram(prog, steps, hooks)
        for atom in ("prev(a)", "tok(b)", "nrun(a)", "ndown(b)",
                     "run(a)", "m(i1)", "u(i1)", "s(i2)"):
            assert fp.marginal(atom) == approx(
                exact_marginal(pr, atom), **EXACT), (steps, atom)


# ---- beyond sync: within-tick cascade ----------------------------------

CASCADE = """
catalog cascade 1.
provenance hand "beyond-sync gate: sequential cascade" "2026-07-19".
layers (dyn).
stages (ua ub).
namespace st produce (dyn).
pred a : st.
pred za : st.
pred b : st.
pred zb : st.
pred ta : st.
pred tb : st.
weight pa reads {}.
weight pb reads {a}.
stage ua {
  %% a commits ON with prob pa.
  aon [dyn] : ta -o a  @w pa.
  %% complement branch.
  aoff [dyn] : ta -o za  @w ~pa.
}
stage ub {
  %% b reads a's COMMITTED value — the within-tick sequential read
  %% the snapshot (sync) discipline cannot express.
  bon [dyn] : tb -o b  @w pb.
  %% complement branch.
  boff [dyn] : tb -o zb  @w ~pb.
}
%% a committed -> b samples.
qui ab [dyn] : ua -o ub * tb.
"""


def cascade_build(pb=None):
    doc = compile_catalog(parse(CASCADE))
    prog = doc.catalog.ground({"ta": 1})
    reg = {"pa": lambda a, c: 0.6,
           "pb": pb or (lambda a, c: 0.9 if "a" in c else 0.2)}
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    pol = weight_policy(prog, doc.ports, reg)
    return prog, hooks, pol


def test_cascade_analytic_and_filter_parity():
    prog, hooks, pol = cascade_build()
    fp = FactoredProgram(prog, 3, hooks)
    # hand: P(b) = 0.6*0.9 + 0.4*0.2
    assert fp.marginal("b") == approx(0.62, **EXACT)
    assert fp.marginal("a") == approx(0.6, **EXACT)
    pr = ForwardProjection(prog, 3, policy=pol)
    for atom in ("a", "b", "za", "zb"):
        assert fp.marginal(atom) == approx(
            exact_marginal(pr, atom), **EXACT), atom


# ---- beyond sync: 3-way CHOICE site ------------------------------------

TRIWAY = """
catalog triway 1.
provenance hand "beyond-sync gate: 3-way choice site" "2026-07-19".
layers (dyn).
stages (day).
namespace st produce (dyn).
pred s : st.
pred r : st.
pred f : st.
pred tk : st.
weight w1 reads {}.
weight w2 reads {}.
stage day {
  %% sunny branch.
  gs [dyn] : tk -o s  @w w1.
  %% rainy branch.
  gr [dyn] : tk -o r  @w w2.
  %% fog: the unweighted remainder — CHOICE renormalizes the site.
  gf [dyn] : tk -o f.
}
"""


def test_triway_site_analytic_and_filter_parity():
    doc = compile_catalog(parse(TRIWAY))
    prog = doc.catalog.ground({"tk": 1})
    reg = {"w1": lambda a, c: 0.5, "w2": lambda a, c: 0.3}
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    fp = FactoredProgram(prog, 1, hooks)
    assert fp.marginal("s") == approx(0.5 / 1.8, **EXACT)
    assert fp.marginal("r") == approx(0.3 / 1.8, **EXACT)
    assert fp.marginal("f") == approx(1.0 / 1.8, **EXACT)
    pr = ForwardProjection(prog, 1,
                           policy=weight_policy(prog, doc.ports, reg))
    for atom in ("s", "r", "f"):
        assert fp.marginal(atom) == approx(
            exact_marginal(pr, atom), **EXACT), atom


# ---- beyond the cliff: generated ring, sampler as referee --------------

def ring_llp(n):
    """A ring SysAdmin instance in rulescript — sysadmin2's RULES
    verbatim (they are variable-generic); only the type tables and
    the Horn topology grow with n."""
    comps = " ".join(f"c{i}" for i in range(n))
    idxs = " ".join(f"i{i}" for i in range(n + 1))
    cidx = " ".join(f"cidx(c{i},i{i})." for i in range(n))
    succ = " ".join(f"succ(i{i},i{i + 1})." for i in range(n))
    adj = " ".join(f"adj(c{i},c{(i - 1) % n}). "
                   f"adj(c{i},c{(i + 1) % n})." for i in range(n))
    rules = Path(SYS).read_text().split("stage mark", 1)[1]
    return f"""
catalog ring 1.
provenance hand "generated ring instance" "2026-07-19".
layers (dyn).
stages (mark update sweep).
type computer {{{comps}}}.
type idx {{{idxs}}}.
namespace state produce (dyn).
namespace chain produce (dyn).
pred run(computer) : state.
pred down(computer) : state.
pred prev(computer) : state.
pred pdown(computer) : state.
pred tok(computer) : state.
pred nrun(computer) : state.
pred ndown(computer) : state.
pred rb(computer) : state.
pred m(idx) : chain.
pred u(idx) : chain.
pred s(idx) : chain.
bwd cidx(computer,idx).
bwd succ(idx,idx).
bwd adj(computer,computer).
{cidx}
{succ}
{adj}
weight cpt(C) reads {{rb(C), prev(D) if adj(C,D)}}.
weight rp(C) reads {{rb(C)}}.
stage mark{rules}""".replace("m(i2)", f"m(i{n})") \
        .replace("u(i2)", f"u(i{n})").replace("s(i2)", f"s(i{n})")


def ring_registry(n):
    def cpt(args, counts):
        if f"rb({args[0]})" in counts:
            return 1.0
        i = int(args[0][1:])
        ins = [f"c{(i - 1) % n}", f"c{(i + 1) % n}"]
        r = sum(1 for y in ins if f"prev({y})" in counts)
        return 0.45 + 0.5 * (1 + r) / (1 + len(ins))

    def rp(args, counts):
        return 1.0 if f"rb({args[0]})" in counts else 0.04

    return {"cpt": cpt, "rp": rp}


def ring_build(n, running):
    doc = compile_catalog(parse(ring_llp(n)))
    init = {"m(i0)": 1}
    for i in range(n):
        init[f"run(c{i})" if i in running else f"down(c{i})"] = 1
    prog = doc.catalog.ground(init)
    reg = ring_registry(n)
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    pol = weight_policy(prog, doc.ports, reg)
    return prog, hooks, pol


def test_ring_beyond_the_cliff_vs_sampler():
    n = 16                     # exact support: 2^16 states — dead
    prog, hooks, pol = ring_build(n, set(range(0, n, 2)))
    steps = 2 * (3 * n + 3)
    t0 = time.time()
    fp = FactoredProgram(prog, steps, hooks)
    marg = {f"run(c{i})": fp.marginal(f"run(c{i})") for i in range(6)}
    dt = time.time() - t0
    assert dt < 60.0, dt
    sp = SampledProjection(prog, steps, n=1500, seed=7, policy=pol)
    for atom, m in marg.items():
        assert m == approx(exact_marginal(sp, atom), abs=0.05), atom


def test_small_ring_parity_with_exact_filter():
    prog, hooks, pol = ring_build(4, {0, 1})
    steps = 2 * 15
    pr = ForwardProjection(prog, steps, policy=pol)
    fp = FactoredProgram(prog, steps, hooks)
    for i in range(4):
        assert fp.marginal(f"run(c{i})") == approx(
            exact_marginal(pr, f"run(c{i})"), **EXACT), i


# ---- the admission certificate -----------------------------------------

def counting_registry(reg):
    calls = [0]

    def wrap(fn):
        def g(args, counts):
            calls[0] += 1
            return fn(args, counts)
        return g

    return {k: wrap(f) for k, f in reg.items()}, calls


def test_certificate_is_symbolic_and_admits():
    """certificate() prices the query BEFORE any table exists — zero
    binding calls — and a factor-ops budget admits the short horizon
    while rejecting the deep one."""
    n = 8
    doc = compile_catalog(parse(ring_llp(n)))
    init = {"m(i0)": 1, **{f"run(c{i})": 1 for i in range(n)}}
    prog = doc.catalog.ground(init)
    reg, calls = counting_registry(ring_registry(n))
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    certs = []
    for ticks in (1, 2, 4):
        fp = FactoredProgram(prog, ticks * (3 * n + 3), hooks)
        certs.append(fp.certificate("run(c0)")["ops"])
    assert calls[0] == 0                   # admission never evaluates
    assert certs[0] < certs[1] < certs[2]
    budget = certs[1] * 1.5
    assert certs[1] <= budget < certs[2]   # admit 2 ticks, reject 4
    fp = FactoredProgram(prog, 3 * n + 3, hooks)
    fp.marginal("run(c0)")
    assert calls[0] > 0                    # evaluation does


def test_elim_order_is_process_deterministic():
    """The shipped schedule (and hence the certified ops and the
    golden wire) must not depend on PYTHONHASHSEED: min-fill ties
    break lexicographically. Two fresh interpreters with different
    hash seeds must produce byte-identical orders."""
    import os
    import subprocess
    import sys
    snippet = (
        "from metis.kernel.factorize import _elim_cost\n"
        "scopes = [{'b#1','a#1'}, {'c#1','b#1'}, {'a#1','c#1'},\n"
        "          {'d#1','a#1'}, {'e#1','b#1'}, {'e#1','d#1'}]\n"
        "print(_elim_cost(scopes, {}, {'a#1'})['order'])\n")
    outs = set()
    for seed in ("1", "2"):
        r = subprocess.run(
            [sys.executable, "-c", snippet],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout)
    assert len(outs) == 1, outs


# ---- the defense rejections --------------------------------------------

OPAQUE = """
catalog opaque 1.
provenance hand "reject gate: undeclared read scope" "2026-07-19".
layers (dyn).
stages (s1).
namespace st produce (dyn).
pred a : st.
pred t : st.
weight q.
stage s1 {
  %% port without a reads contract.
  r1 [dyn] : t -o a  @w q.
}
"""


def test_opaque_port_rejected_at_admission():
    doc = compile_catalog(parse(OPAQUE))
    prog = doc.catalog.ground({"t": 1})
    with raises(LangError, match="declares no read scope"):
        factor_hooks(prog, doc.ports, {"q": lambda a, c: 0.5},
                     doc.catalog.horn_db())


def test_multiplicity_rejected_at_admission():
    doc = compile_catalog(parse(OPAQUE))
    prog = doc.catalog.ground({"t": 2})
    with raises(FactorizeError, match="multiplicity"):
        FactoredProgram(prog, 1)


def test_double_production_rejected():
    doc = compile_catalog(parse(OPAQUE))
    prog = doc.catalog.ground({"t": 1, "a": 1})
    with raises(FactorizeError, match="already be present"):
        FactoredProgram(prog, 1)


def test_scope_violation_raises_at_evaluation():
    """A binding reading outside its declared scope is a contract
    violation with teeth — ScopedCounts raises."""
    prog, hooks, _ = cascade_build(
        pb=lambda a, c: 0.9 if "zb" in c else 0.2)   # zb not declared
    fp = FactoredProgram(prog, 3, hooks)
    with raises(FactorizeError, match="outside its declared scope"):
        fp.marginal("b")


STALL = """
catalog stall 1.
provenance hand "reject gate: reachable stalled tier" "2026-07-19".
layers (dyn).
stages (ua ub).
namespace st produce (dyn).
pred a : st.
pred za : st.
pred b : st.
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
  %% enabled only when a landed — the za branch stalls this tier.
  bon [dyn] : tb * a -o b * a.
}
%% commit -> gate.
qui ab [dyn] : ua -o ub * tb.
"""


def test_reachable_stall_rejected_at_evaluation():
    doc = compile_catalog(parse(STALL))
    prog = doc.catalog.ground({"ta": 1})
    hooks = factor_hooks(prog, doc.ports, {"pa": lambda a, c: 0.6},
                         doc.catalog.horn_db())
    fp = FactoredProgram(prog, 3, hooks)
    with raises(FactorizeError, match="mass"):
        fp.marginal("b")


# ---- IR item 2: the junction tree --------------------------------------

def test_jtree_analytic_chain():
    import numpy as np
    from metis.kernel.jtree import JunctionTree
    from metis.kernel.factored import Factor
    fa = Factor(("a",), np.array([0.4, 0.6]))
    fba = Factor(("a", "b"), np.array([[0.9, 0.1], [0.2, 0.8]]))
    jt = JunctionTree([{"a"}, {"a", "b"}], {"a": 2, "b": 2})
    cal = jt.calibrate([fa, fba])
    assert cal.z == approx(1.0, **EXACT)
    assert cal.marginal("a")[1] == approx(0.6, **EXACT)
    assert cal.marginal("b")[1] == approx(
        0.4 * 0.1 + 0.6 * 0.8, **EXACT)


def test_all_marginals_from_one_calibration():
    """The junction-tree payoff: every unit's marginal from ONE
    two-pass calibration, matching the exact filter."""
    prog, hooks, pol = ring_build(4, {0, 1})
    steps = 2 * 15
    pr = ForwardProjection(prog, steps, policy=pol)
    fp = FactoredProgram(prog, steps, hooks)
    marg = fp.marginals([f"run(c{i})" for i in range(4)]
                        + [f"down(c{i})" for i in range(4)])
    for atom, m in marg.items():
        assert m == approx(exact_marginal(pr, atom), **EXACT), atom


def test_unknowns_one_structure_serves_every_tick():
    """The receding-horizon seam: init atoms lifted to variables —
    ONE compiled structure, per-tick evidence via point factors."""
    from metis.kernel.factorize import factored_for
    doc = load_llp(SYS)
    template = doc.catalog.ground(
        {"m(i0)": 1, "run(a)": 1, "run(b)": 1})
    hooks = factor_hooks(template, doc.ports, sys_registry(),
                         doc.catalog.horn_db())
    unknowns = ("run(a)", "down(a)", "run(b)", "down(b)")
    cache = {}
    fp = factored_for(template, 9, hooks, unknowns, cache=cache)
    assert factored_for(template, 9, hooks, unknowns,
                        cache=cache) is fp
    from metis.kernel import Program
    for running in ({"a", "b"}, {"a"}, set()):
        init = {"m(i0)": 1}
        over = {}
        for c in ("a", "b"):
            up = c in running
            init[f"run({c})" if up else f"down({c})"] = 1
            over[f"run({c})"], over[f"down({c})"] = int(up), int(not up)
        concrete = Program(stages=template.stages,
                           links=template.links,
                           init_stage=template.init_stage, init=init)
        pr = ForwardProjection(
            concrete, 9,
            policy=weight_policy(concrete, doc.ports, sys_registry()))
        for atom in ("run(a)", "run(b)", "down(b)"):
            assert fp.marginal(atom, init=over) == approx(
                exact_marginal(pr, atom), **EXACT), (running, atom)
    with raises(FactorizeError, match="non-unknown"):
        fp.marginal("run(a)", init={"m(i0)": 0})


def test_likelihood_evidence_clamp_is_the_surprise_op():
    """P(observed) by point-factor clamping; 0 IS the unmodeled
    verdict (structural — weights multiply, support cannot appear)."""
    from metis.kernel.posterior import Conditioned
    prog, hooks, pol = cascade_build()
    fp = FactoredProgram(prog, 3, hooks)
    assert fp.likelihood({"b": 1}) == approx(0.62, **EXACT)
    assert fp.likelihood({"b": 1, "zb": 1}) == 0.0   # unmodeled
    c = Conditioned(ForwardProjection(prog, 3, policy=pol),
                    {3: lambda s: ("b", 1) in s[1]})
    assert fp.likelihood({"b": 1}) == approx(c.likelihood, **EXACT)
    prog, hooks, _ = sys_build({"a", "b"})
    fp = FactoredProgram(prog, 9, hooks)
    assert fp.likelihood({"run(a)": 1}) == approx(0.95, **EXACT)
    assert fp.likelihood({"run(a)": 1, "run(b)": 0}) == approx(
        0.95 * 0.05, **EXACT)
