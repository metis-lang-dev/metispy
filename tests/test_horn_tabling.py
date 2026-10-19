"""Variant tabling in the Horn solver is an optimization of the SAME
answers: derivable and solutions (set AND first-wins order) agree with
plain SLD, and the unary-arithmetic grounding that cost ~n^6 is fast."""

import time
from pathlib import Path

from metis.kernel.horn import HornDB
from metis.lang import compile_catalog, parse

PRIMES = (Path(__file__).resolve().parents[2]
          / "metis-catalog/domains/ceptre/primes/primes.llp")


class PlainSLD(HornDB):
    """The pre-tabling solver: one SLD branch per matching clause."""

    def _solve(self, goals, subst, depth):
        if not goals:
            yield subst
            return
        if depth <= 0:
            return
        g0 = goals[0]
        g = (g0[0], tuple(self._walk(a, subst) for a in g0[1]))
        yield from self._resolve(g, tuple(goals[1:]), subst, depth)


def _dbs():
    cat = compile_catalog(parse(PRIMES.read_text())).catalog
    clauses = [e.payload for e in cat.entries if e.kind == "horn"]
    return HornDB(clauses), PlainSLD(clauses)


def test_tabled_solutions_match_plain_sld_in_order():
    tabled, plain = _dbs()
    for goals in ([("plus", ("?A", "?B", "n5"))],
                  [("mult", ("?A", "?B", "n12"))],
                  [("mult", ("n3", "?B", "?C"))],
                  [("plus", ("?A", "n2", "?C")),
                   ("mult", ("n2", "?A", "?D"))]):
        assert list(tabled.solutions(goals)) == \
            list(plain.solutions(goals)), goals


def test_tabled_derivable_matches_plain_sld():
    tabled, plain = _dbs()
    nats = [f"n{i}" for i in range(13)]
    for a in nats[:5]:
        for b in nats[:5]:
            for c in nats:
                g = ("mult", (a, b, c))
                assert tabled.derivable(g) == plain.derivable(g), g


def test_unary_arithmetic_grounding_is_fast():
    cat = compile_catalog(parse(PRIMES.read_text())).catalog
    t = time.time()
    prog = cat.ground({"limit(n12)": 1})
    assert time.time() - t < 2.0          # was ~5 s untabled
    names = {ev.name for _, _, ev in prog.events()}
    assert "collide[N=n2,M=n4,X=n2]" in names
