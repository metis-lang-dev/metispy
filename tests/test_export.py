"""IR item 4 gates: the UAI export is validated by an INDEPENDENT
evaluator — this file's brute-force reader parses the exported text
per the UAI spec and enumerates the full joint; its marginals must
match the kernel's at 1e-9. If pgmpy is installed, it referees too
(skipped otherwise)."""

import itertools

import pytest
from pytest import approx

from metis.kernel.export import to_dot, to_uai
from metis.kernel.factorize import FactoredProgram
from metis.lang import factor_hooks, load_llp
from test_factorize import CASCADE, TRIWAY, SYS, cascade_build, \
    sys_registry
from metis.lang import compile_catalog, parse

EXACT = dict(abs=1e-9)


def parse_uai(text):
    toks = text.split()
    assert toks[0] == "MARKOV"
    i = 1
    n = int(toks[i]); i += 1
    cards = [int(toks[i + k]) for k in range(n)]; i += n
    m = int(toks[i]); i += 1
    scopes = []
    for _ in range(m):
        k = int(toks[i]); i += 1
        scopes.append([int(toks[i + j]) for j in range(k)]); i += k
    tables = []
    for _ in range(m):
        sz = int(toks[i]); i += 1
        tables.append([float(toks[i + j]) for j in range(sz)]); i += sz
    return cards, scopes, tables


def brute_marginal(text, qidx):
    """Full-joint enumeration straight off the exported file — no
    kernel code involved."""
    cards, scopes, tables = parse_uai(text)
    z = on = 0.0
    for asn in itertools.product(*[range(c) for c in cards]):
        w = 1.0
        for scope, tab in zip(scopes, tables):
            f = 0
            for v in scope:              # last scope var fastest
                f = f * cards[v] + asn[v]
            w *= tab[f]
            if w == 0.0:
                break
        z += w
        if asn[qidx] == 1:
            on += w
    return on / z


def cases():
    prog, hooks, _ = cascade_build()
    yield FactoredProgram(prog, 3, hooks), ["a", "b", "zb"], None
    doc = compile_catalog(parse(TRIWAY))
    prog = doc.catalog.ground({"tk": 1})
    hooks = factor_hooks(prog, doc.ports,
                         {"w1": lambda a, c: 0.5,
                          "w2": lambda a, c: 0.3},
                         doc.catalog.horn_db())
    yield FactoredProgram(prog, 1, hooks), ["s", "r", "f"], None
    doc = load_llp(SYS)
    prog = doc.catalog.ground({"m(i0)": 1, "run(a)": 1, "run(b)": 1})
    hooks = factor_hooks(prog, doc.ports, sys_registry(),
                         doc.catalog.horn_db())
    fp = FactoredProgram(prog, 5, hooks,
                         unknowns=("run(a)", "down(a)"))
    yield fp, ["nrun(a)", "prev(a)"], {"run(a)": 0, "down(a)": 1}


def test_uai_export_matches_kernel_via_independent_reader():
    for fp, queries, init in cases():
        text, names = to_uai(fp, init)
        for q in queries:
            v = fp._env[q]
            got = brute_marginal(text, names.index(v))
            assert got == approx(fp.marginal(q, init), **EXACT), q


def test_dot_export_renders_the_net():
    prog, hooks, _ = cascade_build()
    fp = FactoredProgram(prog, 3, hooks)
    dot = to_dot(fp)
    assert dot.startswith("graph metis_net")
    assert '"b#3"' in dot or '"b#2"' in dot or "b#" in dot
    assert "shape=box" in dot


def test_pgmpy_referees_when_available():
    pgmpy = pytest.importorskip("pgmpy")               # noqa: F841
    try:                                    # pgmpy >= 1.x renamed it
        from pgmpy.models import DiscreteMarkovNetwork \
            as MarkovNetwork
    except ImportError:
        from pgmpy.models import MarkovNetwork
    from pgmpy.factors.discrete import DiscreteFactor
    import numpy as np
    prog, hooks, _ = cascade_build()
    fp = FactoredProgram(prog, 3, hooks)
    text, names = to_uai(fp)
    cards, scopes, tables = parse_uai(text)
    g = MarkovNetwork()
    g.add_nodes_from(names)
    for scope, tab in zip(scopes, tables):
        vs = [names[v] for v in scope]
        g.add_edges_from([(a, b) for i, a in enumerate(vs)
                          for b in vs[i + 1:]])
        g.add_factors(DiscreteFactor(
            vs, [cards[v] for v in scope], np.array(tab)))
    from pgmpy.inference import BeliefPropagation
    bp = BeliefPropagation(g)
    q = bp.query([fp._env["b"]], show_progress=False)
    assert float(q.values[1]) == approx(fp.marginal("b"), abs=1e-6)
