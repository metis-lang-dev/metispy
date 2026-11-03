"""Loop and runtime diagnostics (spec 08 step 3): each code is emitted
through the surface that speaks it — the gate cascade, the exact
instruments, the Horn solver, the factorizer, the llm endpoint — and
the REPL lists them with `findings`. metispy-only surfaces (metisc has
no loop and no runtime instruments), so these are the fixtures."""

import os
from pathlib import Path

import pytest

import metis.kernel
from metis.kernel.diagnostics import REGISTRY, render_text

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "metis-lang/corpus"
CATALOG = ROOT / "metis-catalog"

TWO_STAGES = """catalog two 1.
provenance text "one name in two stages, compiled with no resolve" "2026-11-02".
layers (dyn).
stages (a b).
namespace n produce (dyn).
pred p : n.
pred q : n.
stage a {
  %% One.
  h [dyn] : p -o q.
}
stage b {
  %% Two.
  h [dyn] : q -o p.
}
"""

# every loop/runtime code and the test below that emits it
COVERED = {
    "provenance-shape": "test_loop_codes_on_the_gate_report",
    "state-budget": "test_state_budget_query_and_map",
    "exact-unreachable": "test_exact_unreachable_on_trace",
    "horn-nonground": "test_horn_nonground_rides_the_exception",
    "reads-undeclared": "test_reads_undeclared_rides_the_exception",
    "llm-endpoint": "test_llm_endpoint_note",
    "invariant-init": "test_invariant_init_on_the_shell_lint",
}


def test_every_loop_and_runtime_code_has_a_fixture():
    assert set(COVERED) == {c for c, (_, ph) in REGISTRY.items()
                            if ph in ("loop", "runtime")}


def _repl(path, monkeypatch=None):
    from metis.lang.repl import Repl
    return Repl(str(path))


def test_loop_codes_on_the_gate_report():
    """A pack without text provenance whose rule reuses a base rule's short name: a discipline
    warning, each a diagnostic, and the name is refused (dup-rule-names, spec/18 §4)."""
    from metis.lang.compiler import load_llp
    from metis.lang.formalize import admit_text_pack
    here = os.getcwd()
    os.chdir(CORPUS)
    try:
        base = load_llp("packbase.llp")
        src = (CORPUS / "packmark.llp").read_text().replace(
            "obs [pack]", "h [pack]")
        r = admit_text_pack("a market note observes the coin.", src,
                            base=base)
    finally:
        os.chdir(here)
    codes = [d.code for d in r.diagnostics]
    assert "provenance-shape" in codes and "pack-redeclares" in codes


def test_a_catalog_compiled_without_resolve_is_held_to_one_space_of_names():
    """compile_catalog(parse(text)), which never goes through resolve_includes, refuses a name in two stages
    as load_llp does (rule-duplicate): so no base reaches the gate with one, and the loop's old short-name
    warning (rule-name-collision) had nothing left to see; it is gone from the registry."""
    from metis.lang import LangError, compile_catalog, parse
    with pytest.raises(LangError) as e:
        compile_catalog(parse(TWO_STAGES))
    (d,) = e.value.diagnostics
    assert d.code == "rule-duplicate" and d.text == "rule h is named twice in stages a and b: a rule is final, and a second one is not added"
    assert "rule-name-collision" not in REGISTRY


def test_state_budget_query_and_map(monkeypatch):
    monkeypatch.setattr(metis.kernel, "STATE_BUDGET", 10)
    import metis.lang.repl as repl_mod
    monkeypatch.setattr(repl_mod, "STATE_BUDGET", 10)
    r = _repl(CATALOG / "domains/ceptre/love_triangle/love_triangle.llp")
    r.onecmd("query anger(hermia,helena) 6")
    r.onecmd("map 6")
    notes = [d for d in r.notes if d.code == "state-budget"]
    assert [d.loc.decl for d in notes] == ["query", "map"]
    q = dict(notes[0].data)
    assert q["budget"] == "10" and q["retry"].startswith("query ")
    assert render_text(notes[0]).startswith("note state-budget (query)")


def test_exact_unreachable_on_trace():
    """tragedy grounds to more events than the exact cap: trace says so
    on the Q lines AND as a note."""
    r = _repl(CATALOG / "domains/ceptre/tragedy/tragedy.llp")
    r.onecmd("#steps 2")
    r.onecmd("trace")
    (d,) = [d for d in r.notes if d.code == "exact-unreachable"]
    assert "exceeds the exact cap" in dict(d.data)["reason"]
    assert d.loc.decl == "trace"


def test_horn_nonground_rides_the_exception():
    from metis.kernel.horn import HornDB, HornNonground, fact
    from metis.kernel.horn import HornClause
    db = HornDB([HornClause("eq", ("eq", ("X", "X")), vars=("X",))])
    with pytest.raises(HornNonground) as e:
        list(db.solutions([("eq", ("?A", "?B"))]))
    d = e.value.diagnostic
    assert d.code == "horn-nonground" and d.severity == "error"
    assert dict(d.data)["goals"] == "eq(?A,?B)"
    assert isinstance(e.value, ValueError)      # callers' contract


def test_reads_undeclared_rides_the_exception():
    from metis.kernel.factorize import (FactorizeError, ScopedCounts,
                                        ScopeViolation)
    view = ScopedCounts({"tok(n0)"}, {"tok(n0)": 1})
    with pytest.raises(ScopeViolation) as e:
        view["hit(n0)"]
    d = e.value.diagnostic
    assert d.code == "reads-undeclared" and d.severity == "error"
    assert dict(d.data) == {"atom": "hit(n0)"}
    assert isinstance(e.value, FactorizeError)  # callers' contract


def test_llm_endpoint_note(monkeypatch):
    """A command endpoint whose program is not on PATH, and an unset
    endpoint, each leave the note (the removed Ollama probe's case was a
    server unreachable)."""
    r = _repl(CATALOG / "domains/ceptre/three_pigs/three_pigs.llp")
    r.onecmd("llm command:no-such-metis-model --x")
    (d,) = [d for d in r.notes if d.code == "llm-endpoint"]
    assert dict(d.data)["problem"] == "command no-such-metis-model not found"
    assert d.loc.decl == "llm"
    assert "[llm] endpoint" in render_text(d)
    r = _repl(CATALOG / "domains/ceptre/three_pigs/three_pigs.llp")
    r.onecmd("llm")
    (d,) = [d for d in r.notes if d.code == "llm-endpoint"]
    assert dict(d.data)["problem"] == "no LLM endpoint set"


def test_findings_verb_lists_and_shows(capsys, monkeypatch):
    """`findings` lists catalog warnings + the gate report + session
    notes; `findings N` renders one in full; the wire returns JSON."""
    from metis.loop.wire import dispatch
    r = _repl(CATALOG / "domains/ceptre/primes/primes.llp")   # one warning
    r.onecmd("llm command:no-such-metis-model")
    capsys.readouterr()
    r.onecmd("findings")
    listing = capsys.readouterr().out.splitlines()
    assert listing[0].startswith(
        "1. warning fact-var-singleton @ primes.llp:")
    assert listing[1] == "2. note llm-endpoint (llm)"
    r.onecmd("findings 1")
    full = capsys.readouterr().out
    assert "mult(n0,N,n0)." in full and "-> if intended" in full
    wire = dispatch(r, "findings", "")
    assert wire["ok"] and [d["code"] for d in wire["diagnostics"]] == \
        ["fact-var-singleton", "llm-endpoint"]


def test_notes_dedupe_and_retire_with_the_world(monkeypatch):
    """A repeated instrument adds no row; reload (a new world) retires
    the old world's notes."""
    monkeypatch.setattr(metis.kernel, "STATE_BUDGET", 10)
    import metis.lang.repl as repl_mod
    monkeypatch.setattr(repl_mod, "STATE_BUDGET", 10)
    r = _repl(CATALOG / "domains/ceptre/love_triangle/love_triangle.llp")
    for _ in range(3):
        r.onecmd("query anger(hermia,helena) 6")
        r.onecmd("trace")
    assert sorted(d.code for d in r.notes) == ["exact-unreachable",
                                               "state-budget"]
    r.onecmd("reload")
    assert r.notes == []


def test_invariant_init_on_the_shell_lint(tmp_path, capsys):
    """The init half of an `invariant` (spec/00 §5a): the shell's `lint`
    speaks invariant-init against the CURRENT state, so an `assert` that
    breaks the sum is found after the fact and an init that holds it is
    quiet."""
    f = tmp_path / "inv.llp"
    f.write_text("""catalog inv 1.
provenance text "idle or busy, one of them" "2026-11-02".
layers (dyn).
stages (act).
namespace n produce (dyn).
pred idle : n.
pred busy : n.
pred go : n.
invariant mode (idle busy) : 1.
stage act {
  %% Go: idle becomes busy.
  start [dyn] : $go * idle -o busy.
  %% Stop: busy becomes idle.
  stop [dyn] : busy -o idle.
}
#init idle 1
#init go 1
""")
    sh = _repl(f)
    sh.onecmd("lint")
    assert "no lint finding" in capsys.readouterr().out
    sh.onecmd("assert busy")
    sh.onecmd("lint")
    out = capsys.readouterr().out
    assert "error invariant-init (invariant/mode)" in out
    assert "it holds 2" in out
