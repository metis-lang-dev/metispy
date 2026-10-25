"""guard-cyclic (spec 08 code, eval-perf P11, spec 14 §1): a rule or link whose guard hypergraph is cyclic after
GYO reduction is listed as a warning — its grounding is AGM-bounded, not output-linear; it still compiles."""
from metis.lang.compiler import _guard_cycle, diagnose, load_llp

V = {"A", "B", "C", "D", "X", "Y", "Z"}


def g(*edges):
    return [("p", tuple(e)) for e in edges]


def test_the_reduction():
    assert _guard_cycle(g("XY", "YZ", "ZX"), V) == ["X", "Y", "Z"]          # the triangle
    assert _guard_cycle(g("XY", "YZ"), V) == []                             # a path
    assert _guard_cycle(g("XYZ", "XY", "YZ", "ZX"), V) == []                # a triangle under one guard over all three
    assert _guard_cycle(g("AB", "BC", "CD", "DA"), V) == ["A", "B", "C", "D"]   # a 4-cycle
    assert _guard_cycle([("p", ("X", "n0")), ("p", ("n0", "Y"))], V) == []  # constants are no vertices
    assert _guard_cycle(g("XY", "XY"), V) == []                             # two equal edges: one stays, then goes


SRC = """catalog tri 1.
provenance text "guard-cyclic test" "2026-10-25".
layers (dyn).
stages (go).
type n {a b c}.
namespace ns produce (dyn).
pred tok(n) : ns.
pred hit(n) : ns.
bwd edge(n,n).
edge(a,b). edge(b,c). edge(c,a).
stage go {
  %% A triangle of guards.
  t [dyn] : tok(X) * $edge(X,Y) * $edge(Y,Z) * $edge(Z,X) -o hit(X).
  %% A path of guards.
  p [dyn] : tok(X) * $edge(X,Y) * $edge(Y,Z) -o hit(Z).
}
%% The same triangle on a link, its guards unmarked.
qui back [dyn] : go * tok(X) * edge(X,Y) * edge(Y,Z) * edge(Z,X) -o go * hit(Y).
"""


def test_a_rule_and_a_link_are_listed_and_still_compile(tmp_path):
    f = tmp_path / "tri.llp"
    f.write_text(SRC)
    ds = [d for d in diagnose(str(f)) if d.code == "guard-cyclic"]
    assert [(d.severity, d.loc.decl, dict(d.data)["vars"]) for d in ds] == \
        [("warning", "go/t", "X Y Z"), ("warning", "qui/back", "X Y Z")]
    assert ds[0].text == "t: its guards form a cycle over X, Y, Z: grounding it is AGM-bounded, not output-linear"
    prog = load_llp(f).catalog.ground({})                                   # a warning: it grounds
    assert sorted(c.name for c in prog.stages["go"] if c.name.startswith("t[")) == \
        ["t[X=a,Y=b,Z=c]", "t[X=b,Y=c,Z=a]", "t[X=c,Y=a,Z=b]"]
