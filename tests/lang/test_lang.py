"""Stage 9 gates: round-trip contract; the recode proofs (road mini
program_key-identical to the python catalog; sysadmin measures exact
through text + weight ports); containment violations rejected FROM
TEXT; JSON pack reload identity; unbound-port fail-fast."""

import json
from pathlib import Path

import pytest
from pytest import approx

from metis.kernel import CatalogError, ForwardProjection, program_key
from metis.lang import LangError, compile_catalog, from_pack, \
    ground_reads, load_llp, parse, pretty, to_pack, weight_policy
from metis.domains.road.mini import base_catalog, scenario, \
    wheelchair_pack

ROAD = Path("metis/domains/road/catalogs")
SYS = Path("metis/domains/sysadmin/catalogs")
EXACT = dict(abs=1e-12)


# ---- round-trip --------------------------------------------------------

def test_roundtrip_contract_on_shipped_catalogs():
    for path in (ROAD / "mini.llp", ROAD / "wheelchair.llp",
                 SYS / "sysadmin2.llp"):
        a1 = parse(path.read_text())
        assert parse(pretty(a1)) == a1, path


# ---- road recode -------------------------------------------------------

def test_road_recode_program_key_identical():
    doc = load_llp(ROAD / "mini.llp")
    text_prog = doc.catalog.ground(scenario())
    py_prog = base_catalog().ground(scenario())
    assert program_key(text_prog) == program_key(py_prog)
    pr = ForwardProjection(text_prog, 6)
    assert pr.final_distribution("crash") == approx(
        [7 / 18, 11 / 18], **EXACT)


def test_road_pack_admission_from_text():
    base = load_llp(ROAD / "mini.llp")
    pack = load_llp(ROAD / "wheelchair.llp", base=base)
    merged = base.catalog.admit(pack.catalog)
    py_merged = base_catalog().admit(wheelchair_pack())
    prog = merged.ground(scenario(wc=True))
    assert program_key(prog) == program_key(
        py_merged.ground(scenario(wc=True)))
    pr = ForwardProjection(prog, 8)
    assert pr.final_distribution("caution(c2)") == approx(
        [11 / 27, 16 / 27], **EXACT)


# ---- sysadmin recode (weight ports) ------------------------------------

ADJ = {"a": ["b"], "b": ["a"]}


def sys_registry(reboot_prob=0.04):
    def cpt(args, counts):
        c = args[0]
        if f"rb({c})" in counts:
            return 1.0
        ins = ADJ[c]
        r = sum(1 for y in ins if f"prev({y})" in counts)
        return 0.45 + 0.5 * (1 + r) / (1 + len(ins))

    def rp(args, counts):
        return 1.0 if f"rb({args[0]})" in counts else reboot_prob

    return {"cpt": cpt, "rp": rp}


def sys_project(running, ticks=1, reboot=None):
    doc = load_llp(SYS / "sysadmin2.llp")
    init = {"m(i0)": 1}
    for c in ("a", "b"):
        init[f"run({c})" if c in running else f"down({c})"] = 1
    if reboot:
        init[f"rb({reboot})"] = 1
    prog = doc.catalog.ground(init)
    pol = weight_policy(prog, doc.ports, sys_registry())
    return ForwardProjection(prog, ticks * 9, policy=pol)


def marginal(pr, c):
    return sum(p for (_, k), p in pr.final.items()
               if (f"run({c})", 1) in k)


def test_sysadmin_recode_measures_exact():
    pr = sys_project({"a", "b"})
    assert marginal(pr, "a") == approx(0.95, **EXACT)
    assert marginal(pr, "b") == approx(0.95, **EXACT)
    both = sum(p for (_, k), p in pr.final.items()
               if ("run(a)", 1) in k and ("run(b)", 1) in k)
    assert both == approx(0.9025, **EXACT)
    pr = sys_project({"a"})
    assert marginal(pr, "a") == approx(0.70, **EXACT)
    assert marginal(pr, "b") == approx(0.04, **EXACT)
    pr = sys_project({"a"}, reboot="b")
    assert marginal(pr, "b") == approx(1.0, **EXACT)


def test_sysadmin_recode_two_ticks_total_probability():
    # hand: after t1 (uu .9025 / ud .0475 / du .0475 / dd .0025);
    # t2 P(a up) = .9025*.95 + .0475*.70 + .0475*.04 + .0025*.04
    pr = sys_project({"a", "b"}, ticks=2)
    assert marginal(pr, "a") == approx(0.892625, **EXACT)
    assert pr.p_done() == 0.0        # the tick cycle never absorbs


def test_unbound_weight_port_fails_fast():
    doc = load_llp(SYS / "sysadmin2.llp")
    prog = doc.catalog.ground({"m(i0)": 1, "run(a)": 1, "run(b)": 1})
    with pytest.raises(LangError, match="unbound weight port 'cpt'"):
        weight_policy(prog, doc.ports, {"rp": lambda a, c: 0.04})


# ---- containment from text ---------------------------------------------

VIOLATIONS = {
    "produce-kinematic": ("""
catalog bad1 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  %% crafted violation
  boost [pack] : tokp -o st(ego,l0,c4,v2).
}
""", CatalogError, "may not produce 'st'"),
    "consume-kinematic": ("""
catalog bad2 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  %% crafted violation
  erase [pack] : st(alfa,l0,c2,v0) -o tokp.
}
""", CatalogError, "may not consume 'st'"),
    "bwd-as-resource": ("""
catalog bad3 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  %% crafted violation
  steal [pack] : tokp -o adv(c0,v0,c0).
}
""", CatalogError, "used as a resource"),
    "undeclared-pred": ("""
catalog bad4 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  %% crafted violation
  ghost [pack] : tokp -o mystery.
}
""", CatalogError, "undeclared predicate"),
    "redeclare-pred": ("""
catalog bad5 1.
extends road.
namespace pk produce (pack).
pred st(car,lane,cell,vel) : pk.
""", CatalogError, "redeclares predicate"),
    "missing-doc": ("""
catalog bad6 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  mute [pack] : tokp -o tokp.
}
""", LangError, "missing %% doc"),
}


def test_violations_rejected_from_text():
    base = load_llp(ROAD / "mini.llp")
    for name, (text, exc, match) in VIOLATIONS.items():
        with pytest.raises(exc, match=match):
            pack = compile_catalog(parse(text), base=base)
            base.catalog.admit(pack.catalog)


# ---- the pack artifact -------------------------------------------------

def test_json_pack_reload_identity():
    for path, init in (
            (ROAD / "mini.llp", dict(scenario and {})),
            (SYS / "sysadmin2.llp", None)):
        doc = load_llp(path)
        blob = json.dumps(to_pack(doc))
        doc2 = from_pack(json.loads(blob))
        if path.name == "mini.llp":
            init_atoms = scenario()
        else:
            init_atoms = {"m(i0)": 1, "run(a)": 1, "down(b)": 1}
        assert program_key(doc.catalog.ground(init_atoms)) == \
            program_key(doc2.catalog.ground(init_atoms))
        assert doc2.ports.weight_refs == doc.ports.weight_refs
    # and the reloaded sysadmin pack still projects correctly
    doc = from_pack(json.loads(json.dumps(
        to_pack(load_llp(SYS / "sysadmin2.llp")))))
    init = {"m(i0)": 1, "run(a)": 1, "run(b)": 1}
    prog = doc.catalog.ground(init)
    pr = ForwardProjection(prog, 9, policy=weight_policy(
        prog, doc.ports, sys_registry()))
    assert marginal(pr, "a") == approx(0.95, **EXACT)


# ---- read-scope declarations -------------------------------------------

def test_ground_reads_resolves_the_scope():
    """The declared scope of a port INSTANCE is symbolic: formals
    substituted, guards enumerated against the Horn world."""
    doc = load_llp(SYS / "sysadmin2.llp")
    horn = doc.catalog.horn_db()
    assert ground_reads(doc.ports, "cpt", ("a",), horn) == \
        ["rb(a)", "prev(b)"]
    assert ground_reads(doc.ports, "cpt", ("b",), horn) == \
        ["rb(b)", "prev(a)"]
    assert ground_reads(doc.ports, "rp", ("b",), horn) == ["rb(b)"]


def test_reads_survive_the_pack_roundtrip():
    doc = load_llp(SYS / "sysadmin2.llp")
    doc2 = from_pack(json.loads(json.dumps(to_pack(doc))))
    assert doc2.ports.reads == doc.ports.reads


READS_HEADER = """
catalog r 1.
layers (dyn).
stages (s1).
type c {a b}.
namespace st produce (dyn).
pred on(c) : st.
bwd adj(c,c).
adj(a,b).
"""

READS_VIOLATIONS = {
    "undeclared-pred": ("weight w(C) reads {off(C)}.",
                        "not a declared resource predicate"),
    "guard-not-bwd": ("weight w(C) reads {on(D) if on(C,D)}.",
                      "not declared bwd"),
    "unbound-var": ("weight w(C) reads {on(D)}.",
                    "reads var D unbound"),
    "const-formals": ("weight w(c) reads {on(a)}.",
                      "reads needs var formals"),
    "ref-arity": ("""weight w(C) reads {on(C)}.
stage s1 {
  %% arity mismatch on the ref
  r1 [dyn] : on(X) -o on(X) @w w(X,X).
}""", "takes 1 args, got 2"),
}


def test_reads_declarations_validated():
    for name, (decl, match) in READS_VIOLATIONS.items():
        with pytest.raises(LangError, match=match):
            compile_catalog(parse(READS_HEADER + decl)), name


def test_reads_on_non_weight_port_rejected():
    from metis.lang.parser import ParseError
    with pytest.raises(ParseError, match="weight ports only"):
        parse(READS_HEADER + "guard g(C) reads {on(C)}.")


# ---- range sugar -------------------------------------------------------

def test_type_range_sugar_expands():
    a = parse("catalog r 1.\nlayers (dyn).\nstages (s1).\n"
              "type cell {c0 .. c3 x}.\ntype idx {0..2}.\n")
    tys = {d.name: d.consts for d in a.decls
           if type(d).__name__ == "TypeDecl"}
    assert tys["cell"] == ("c0", "c1", "c2", "c3", "x")
    assert tys["idx"] == ("0", "1", "2")
    # sugar only: round-trip sees plain constants
    assert parse(pretty(a)) == a


def test_type_range_rejects_mismatched():
    from metis.lang.parser import ParseError
    for rng in ("c0 .. d4", "c4 .. c0", "a .. c3"):
        with pytest.raises(ParseError, match="bad range|bad type"):
            parse(f"catalog r 1.\ntype t {{{rng}}}.\n")
