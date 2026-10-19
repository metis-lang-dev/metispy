"""Formalization gates (docs/rewrite/07-formalization.md): the
tutorial prose round-trip against the stage-42 hand port, plus one
representative per designed-against failure mode. The 'formalizer
outputs' here are fixtures — the LLM seam is pluggable; what these
pin is the VERIFICATION loop."""

import os
from pathlib import Path

from pytest import approx

from metis.kernel import program_key
from metis.lang import load_llp
from metis.lang.formalize import admit_text_pack

ROOT = Path(__file__).resolve().parents[2]
LANG = Path(os.environ.get(
    "METIS_LANG", ROOT.parent / "metis-lang")).resolve()

# Attribution: PROSE and REDUX re-express the Ceptre tutorial's love triangle
# (interactive-lp, Chris Martens, ACSL v1.4); the port is metis-catalog's
# domains/ceptre/love_triangle/.
PROSE = ("Characters can become jealous: if A loves B while C "
         "(someone else) loves A, C becomes angry at A and at B. "
         "Three angers at a person you love, and the love dies.")

# a faithful reformalization — different variable names, different
# rule order: the shape a formalizer plausibly produces
REDUX = """catalog triangle-redux 1.
provenance text "ceptre-tutorial love-triangle prose" "2026-07-23".
layers (story).
stages (go).
type character {helena hermia lysander}.
namespace heart produce (story).
pred eros(character,character) : heart.
pred anger(character,character) : heart.
stage go {
  %% Three angers at a person you love, and the love dies.
  move_on [story] : eros(X,Y) * anger(X,Y) * anger(X,Y) * anger(X,Y)
    -o one.
  %% X loves Y while Z loves X: the rival Z seethes at both.
  jealousy [story] : $eros(X,Y) * $eros(Z,X) * Z <> X
    -o anger(Z,X) * anger(Z,Y).
}
"""

# plausible-but-WRONG: jealousy forgets the second anger product —
# well-typed, passes every static gate, and only the MEASURE can
# tell (the point of the semantic instruments)
WRONG = REDUX.replace("-o anger(Z,X) * anger(Z,Y).",
                      "-o anger(Z,X).")

# vocabulary drift: the formalizer invents `unlock`
DRIFT = REDUX.replace("$eros(Z,X) * Z <> X", "$unlock(Z,X) * Z <> X")

ACT_ONE = {"eros(helena,hermia)": 1, "eros(hermia,helena)": 1,
           "eros(helena,lysander)": 1, "eros(lysander,hermia)": 1}


def test_faithful_reformalization_is_identical():
    """Canonical identity quotients the formalizer's naming freedom:
    the redux grounds program_key-IDENTICAL to the hand port."""
    r = admit_text_pack(PROSE, REDUX, scenarios=(ACT_ONE,), steps=1)
    assert r.ok and not r.findings and not r.warnings
    assert [n for n, _ in r.rules] == ["move_on", "jealousy"]
    assert all(doc.strip() for _, doc in r.rules)
    from lovetriangle import love_triangle
    hand = love_triangle()                 # the catalog's Ceptre port, or skip
    assert r.scenarios[0].key == program_key(
        hand.catalog.ground(dict(ACT_ONE)))
    assert "counterexample" in r.classification
    assert "multiplicity" in r.classification


def test_wrong_formalization_caught_by_the_measure():
    """The forgotten anger product passes every static gate — the
    exact measure is the instrument that catches it."""
    r = admit_text_pack(PROSE, WRONG, scenarios=(ACT_ONE,), steps=1)
    assert r.ok                            # statically impeccable
    good = admit_text_pack(PROSE, REDUX, scenarios=(ACT_ONE,),
                           steps=1)
    assert r.scenarios[0].key != good.scenarios[0].key
    marg = _marginal_of(r, "anger(hermia,hermia)")
    assert marg == approx(0.0)             # the tell: 0.2 in truth
    assert _marginal_of(good, "anger(hermia,hermia)") == approx(
        0.2, abs=1e-9)


def _marginal_of(report, atom):
    from metis.lang.formalize import _final_marginals
    prog = report.merged.ground(dict(report.scenarios[0].init))
    marg, _, _ = _final_marginals(prog, 1)
    return marg.get(atom, 0.0)


def test_vocabulary_drift_rejected_with_named_finding():
    r = admit_text_pack(PROSE, DRIFT, scenarios=(ACT_ONE,))
    assert not r.ok
    assert any("undeclared predicate 'unlock'" in f
               for f in r.findings)


def test_pack_admission_and_measure_diff():
    """A text-derived PACK against a base: containment admission +
    the exact diff of what the new rules changed."""
    base = load_llp(LANG / "corpus/packbase.llp")
    pack_llp = (LANG / "corpus/packmark.llp").read_text() \
        .replace('provenance hand', 'provenance text')
    init = {"coin": 1, "mkt": 1}
    r = admit_text_pack(
        "The marker watches the coin: heads it marks seen, tails "
        "unseen.", pack_llp, base=base, scenarios=(init,), steps=4)
    assert r.ok and not r.findings, r.findings
    diffs = dict((a, (b, m)) for a, b, m in r.scenarios[0].diffs)
    assert "seen" in diffs and "unseen" in diffs
    assert diffs["seen"][0] == approx(0.0)          # base: absent
    # PRIOR measure (no bindings — structure from text, numbers
    # from data): the coin is uniform until a registry binds ph
    assert diffs["seen"][1] == approx(0.5, abs=1e-9)
    assert diffs["unseen"][1] == approx(0.5, abs=1e-9)
    assert r.scenarios[0].map_events                # the read-back


def test_containment_violation_rejected():
    """The pack tries to produce a BASE atom — containment says no."""
    base = load_llp(LANG / "corpus/packbase.llp")
    bad = """catalog grab 1.
extends packbase.
provenance text "adversarial" "2026-07-23".
namespace mk2 produce (pack).
pred tok2 : mk2.
stage note {
  %% tries to write base state.
  cheat [pack] : tok2 -o heads.
}
"""
    r = admit_text_pack("cheat", bad, base=base)
    assert not r.ok
    assert any("may not produce 'heads'" in f for f in r.findings)


def test_empty_text_refused():
    r = admit_text_pack("", REDUX)
    assert not r.ok and "approved unit" in r.findings[0]
