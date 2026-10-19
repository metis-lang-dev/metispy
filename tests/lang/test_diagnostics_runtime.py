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

# every loop/runtime code and the test below that emits it
COVERED = {
    "provenance-shape": "test_loop_codes_on_the_gate_report",
    "rule-name-collision": "test_loop_codes_on_the_gate_report",
    "state-budget": "test_state_budget_query_and_map",
    "exact-unreachable": "test_exact_unreachable_on_trace",
    "horn-nonground": "test_horn_nonground_rides_the_exception",
    "reads-undeclared": "test_reads_undeclared_rides_the_exception",
    "llm-endpoint": "test_llm_endpoint_note",
}


def test_every_loop_and_runtime_code_has_a_fixture():
    assert set(COVERED) == {c for c, (_, ph) in REGISTRY.items()
                            if ph in ("loop", "runtime")}


def _repl(path, monkeypatch=None):
    from metis.lang.repl import Repl
    return Repl(str(path))


def test_loop_codes_on_the_gate_report():
    """A pack without text provenance whose rule reuses a base rule's
    short name: both discipline warnings, each a diagnostic."""
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
    assert "provenance-shape" in codes and "rule-name-collision" in codes
    (rc,) = [d for d in r.diagnostics if d.code == "rule-name-collision"]
    assert dict(rc.data) == {"rule": "h", "count": "2"}
    assert rc.loc.decl == "note/h"          # the newcomer owns the slice
    assert rc.text in r.warnings


def test_state_budget_query_and_map(monkeypatch):
    monkeypatch.setattr(metis.kernel, "STATE_BUDGET", 10)
    import metis.lang.repl as repl_mod
    monkeypatch.setattr(repl_mod, "STATE_BUDGET", 10)
    r = _repl(CATALOG / "domains/love_triangle/love_triangle.llp")
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
    r = _repl(CATALOG / "domains/tragedy/tragedy.llp")
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
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")   # refused
    r = _repl(CATALOG / "domains/three_pigs/three_pigs.llp")
    r.onecmd("llm ollama:no-such-model")
    (d,) = [d for d in r.notes if d.code == "llm-endpoint"]
    assert "unreachable" in dict(d.data)["problem"]
    assert d.loc.decl == "llm"


def test_findings_verb_lists_and_shows(capsys, monkeypatch):
    """`findings` lists catalog warnings + the gate report + session
    notes; `findings N` renders one in full; the wire returns JSON."""
    from metis.loop.wire import dispatch
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    r = _repl(CATALOG / "domains/primes/primes.llp")   # one warning
    r.onecmd("llm ollama:no-such-model")
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
    r = _repl(CATALOG / "domains/love_triangle/love_triangle.llp")
    for _ in range(3):
        r.onecmd("query anger(hermia,helena) 6")
        r.onecmd("trace")
    assert sorted(d.code for d in r.notes) == ["exact-unreachable",
                                               "state-budget"]
    r.onecmd("reload")
    assert r.notes == []
