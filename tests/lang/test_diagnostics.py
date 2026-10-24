"""Structured diagnostics (spec 08): metispy emits byte-for-byte what
metisc emits — the same goldens metis-lang gate 11 diffs."""

import os
import re
from pathlib import Path

import pytest

from metis.kernel import Catalog, Entry
from metis.kernel.catalog import Provenance
from metis.kernel.diagnostics import (CONTEXT_CAP, GATED_PHASES,
                                      REGISTRY, DiagnosticDefect, Loc,
                                      Source, make, render_text, to_json)
from metis.kernel.schema import Signature
from metis.lang.compiler import diagnose

CORPUS = Path(__file__).resolve().parents[3] / "metis-lang/corpus"
# one run per phase, in gate-11 order: (file, base)
RUNS = (("diag_compile.llp", None), ("diag_kernel.llp", None),
        ("diag_pack.llp", "diag_pack_base.llp"),
        ("diag_cycle_a.llp", None), ("diag_extends.llp", None),
        ("diag_abstract.llp", None))


def _diags():
    here = os.getcwd()
    os.chdir(CORPUS / "diag")          # relative paths, as the gate runs
    try:
        return [d for f, b in RUNS for d in diagnose(f, base=b)]
    finally:
        os.chdir(here)


def test_text_rendering_matches_metisc_golden():
    got = "".join(render_text(d) for d in _diags())
    assert got == (CORPUS / "goldens/diag_goldens.txt").read_text()


def test_json_rendering_matches_metisc_golden():
    norm = re.compile(r'"file":"([^"]*/)?([^"/]*)"')
    got = "".join(norm.sub(r'"file":"\2"', to_json(d)) + "\n"
                  for d in _diags())
    assert got == (CORPUS / "goldens/diag_goldens.jsonl").read_text()


def test_registry_is_metiscs_registry():
    rows = "".join(f"{c} {s} {ph}\n" for c, (s, ph) in REGISTRY.items())
    assert rows == (CORPUS / "goldens/diag_registry.txt").read_text()


def test_every_gated_code_fires_exactly_once():
    fired = [d.code for d in _diags()]
    gated = {c for c, (_, ph) in REGISTRY.items() if ph in GATED_PHASES}
    assert gated <= set(fired)
    assert len(fired) == len(set(fired))      # one row per code


def test_included_subject_names_its_own_file():
    (d,) = [d for d in _diags() if d.code == "containment-produce"]
    assert os.path.basename(d.loc.file) == "diag_dom.llp"
    assert d.loc.decl == "note/obs" and d.loc.line == 14


def test_internal_codes_fire_on_hand_built_catalogs():
    """`internal` codes are unreachable from source (metisc parity by
    construction only); a hand-built catalog still gets diagnostics."""
    from metis.kernel.horn import HornClause
    from metis.kernel.typing import check_entry
    src = Source(Loc("x.llp", 3, 3, "go/r"), ("r [dyn] : a -o a.",), ())
    prov = Provenance("hand", "t", "2026-10-04")
    cat = Catalog("t", 1, ("dyn",), ("go",), Signature({}), {}, {},
                  entries=[Entry("dyn/r", "dyn", "weird", "c", prov,
                                 None, src=src)])
    diags = []
    assert cat.typecheck(diags)
    assert [d.code for d in diags] == ["kind-unknown"]
    assert diags[0].loc.decl == "go/r"
    diags = []
    horn = Entry("horn/h", "dyn", "horn", "", prov,
                 HornClause("h", ("h", ())), src=src)
    check_entry(horn, ("dyn",), ("go",), {}, {}, frozenset({"h"}), {},
                diags)
    assert [d.code for d in diags] == ["comment-missing"]


def test_unknown_code_and_context_overflow_are_loud():
    src = Source(Loc("x.llp", 1, 1, "s/r"), ("r.",),
                 tuple(f"line {i}" for i in range(CONTEXT_CAP + 1)))
    with pytest.raises(DiagnosticDefect, match="unregistered"):
        make("no-such-code", "t", src, ())
    with pytest.raises(DiagnosticDefect, match="exceeds the cap"):
        make("doc-missing", "t", src, (("rule", "r"),))


def test_gate_report_carries_diagnostics():
    """The loop gate (formalize) reports the slices beside the findings
    — the proposal's own example: a pack producing a base atom."""
    from metis.lang.compiler import load_llp
    from metis.lang.formalize import admit_text_pack
    here = os.getcwd()
    os.chdir(CORPUS)
    try:
        base = load_llp("packbase.llp")
        bad = (CORPUS / "packmark.llp").read_text().replace(
            "-o seen.", "-o seen * heads.")
        r = admit_text_pack("a market note observes the coin outcome.",
                            bad, base=base)
    finally:
        os.chdir(here)
    assert not r.ok
    (cp,) = [d for d in r.diagnostics if d.code == "containment-produce"]
    assert cp.text == r.findings[0] and cp.loc.decl == "note/obs"


def test_failing_base_raises_not_listed_as_the_packs():
    from metis.lang.compiler import LangError
    here = os.getcwd()
    os.chdir(CORPUS / "diag")
    try:
        with pytest.raises(LangError):
            diagnose("diag_pack.llp", base="diag_compile.llp")
    finally:
        os.chdir(here)


def test_diamond_include_is_not_a_cycle(tmp_path):
    """a includes b and c; both include d: d is reached twice along
    DIFFERENT paths — not a cycle (metisc resolve_in agrees)."""
    from metis.lang.compiler import load_llp
    (tmp_path / "d.llp").write_text("catalog d 1.\ntype t {x}.\n")
    for n in ("b", "c"):
        (tmp_path / f"{n}.llp").write_text(
            f'catalog {n} 1.\ninclude "d.llp".\n')
    (tmp_path / "a.llp").write_text(
        'catalog a 1.\nlayers (s).\nstages (go).\n'
        'include "b.llp".\ninclude "c.llp".\n')
    assert not any(d.code == "include-cycle"
                   for d in diagnose(tmp_path / "a.llp"))
