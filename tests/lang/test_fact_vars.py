"""A var in a Horn fact is universally quantified over the type its
position declares — `plus(n0,N,N).` = Π N:nat — the Ceptre/Twelf
reading of a bodiless clause (metisc agrees: metis-lang compile.ml)."""

import pytest

from metis.kernel import ForwardProjection, program_key
from metis.lang import LangError, compile_catalog, parse

HEAD = """
catalog factvars 1.
layers (story).
stages (go).
type nat {n0 n1 n2}.
type color {red blue}.
namespace ns produce (story).
pred tok(nat) : ns.
pred hit(nat) : ns.
"""

RULE = """
stage go {
  %% fires when the guard holds for the token's value.
  r [story] : tok(N) * eq(N,N) -o hit(N).
}
"""


def _prog(facts):
    cat = compile_catalog(parse(HEAD + facts + RULE)).catalog
    return cat, cat.ground({"tok(n1)": 1})


def test_fact_var_is_quantified():
    cat, prog = _prog("bwd eq(nat,nat).\neq(X,X).\n")
    db = cat.horn_db()
    assert db.derivable(("eq", ("n2", "n2")))
    assert not db.derivable(("eq", ("n1", "n2")))
    pr = ForwardProjection(prog, 2)
    assert sum(p for (_, k), p in pr.final.items()
               if dict(k).get("hit(n1)", 0) > 0) == 1.0


def test_quantified_fact_grounds_like_its_enumeration():
    _, quantified = _prog("bwd eq(nat,nat).\neq(X,X).\n")
    _, enumerated = _prog("bwd eq(nat,nat).\n"
                          "eq(n0,n0).\neq(n1,n1).\neq(n2,n2).\n")
    assert program_key(quantified) == program_key(enumerated)


def test_fact_var_typed_like_a_rule_var():
    with pytest.raises(LangError, match="typed both"):
        _prog("bwd eq(nat,nat).\nbwd paint(nat,color).\n"
              "eq(X,X).\npaint(X,X).\n")


def test_unpinned_fact_var_in_solutions_is_loud():
    cat, _ = _prog("bwd eq(nat,nat).\neq(X,X).\n")
    with pytest.raises(ValueError, match="non-ground Horn solution"):
        list(cat.horn_db().solutions([("eq", ("?A", "?B"))]))
