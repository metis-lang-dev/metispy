"""Structured diagnostics (spec 08): metispy emits byte-for-byte what
metisc emits — the same goldens metis-lang gate 11 diffs."""

import os
import re
from pathlib import Path

import pytest

from metis.kernel.diagnostics import (CONTEXT_CAP, REGISTRY,
                                      DiagnosticDefect, Loc, Source,
                                      make, render_text, to_json)
from metis.lang.compiler import diagnose

CORPUS = Path(__file__).resolve().parents[3] / "metis-lang/corpus"
FIXTURES = ("diag_compile.llp", "diag_kernel.llp")


def _diags():
    here = os.getcwd()
    os.chdir(CORPUS / "diag")          # relative paths, as the gate runs
    try:
        return [d for f in FIXTURES for d in diagnose(f)]
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
    rows = "".join(f"{c} {s} {'migrated' if m else 'pending'}\n"
                   for c, (s, m) in REGISTRY.items())
    assert rows == (CORPUS / "goldens/diag_registry.txt").read_text()


def test_every_migrated_code_has_a_fixture_row():
    fired = {d.code for d in _diags()}
    assert {c for c, (_, m) in REGISTRY.items() if m} <= fired


def test_included_subject_names_its_own_file():
    (d,) = [d for d in _diags() if d.code == "containment-produce"]
    assert os.path.basename(d.loc.file) == "diag_dom.llp"
    assert d.loc.decl == "note/obs" and d.loc.line == 14


def test_unknown_code_and_context_overflow_are_loud():
    src = Source(Loc("x.llp", 1, 1, "s/r"), ("r.",),
                 tuple(f"line {i}" for i in range(CONTEXT_CAP + 1)))
    with pytest.raises(DiagnosticDefect, match="unregistered"):
        make("no-such-code", "t", src, ())
    with pytest.raises(DiagnosticDefect, match="exceeds the cap"):
        make("doc-missing", "t", src, (("rule", "r"),))
