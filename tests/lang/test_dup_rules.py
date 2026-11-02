"""A rule named twice in a stage (dup-rule-names s-dfb61f45, spec/18 §4): this toolchain against metis-lang's
corpus/dup and metisc: one error, the same words, across blocks and includes; a pack refused at admission."""

import os
import subprocess

import pytest

from langpath import LANG
from metis.kernel.artifact import metisc
from metis.kernel.diagnostics import render_text
from metis.lang.compiler import diagnose

DUP = LANG / "corpus/dup"
REFUSED = {"blocks": "aa", "include": "aa", "nested": "aa", "sibling": "aa", "diamond": "aa", "head": "turn-1"}
LINKS = ("link", "link_include", "link_nested")
MIXED = {"mixed": "back", "mixed_rule": "back", "mixed_include": "back", "mixed_head": "back"}


def _diag(*args):
    here = os.getcwd()
    os.chdir(DUP)
    try:
        return diagnose(*args[:1], base=args[1] if len(args) > 1 else None)
    finally:
        os.chdir(here)


def _metisc(*args):
    mc = metisc()
    if not mc:
        pytest.skip("no metisc: the python half only")
    r = subprocess.run([mc, "--diagnostics", *args], cwd=DUP, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=60)
    return r.stdout + r.stderr


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_one_error_in_metiscs_words(name):
    ds = _diag(f"dup_{name}.llp")
    assert [d.code for d in ds] == ["rule-duplicate"]
    assert f"rule {REFUSED[name]} is named twice in stage act" in ds[0].text
    assert "".join(render_text(d) for d in ds) == _metisc(f"dup_{name}.llp")


@pytest.mark.parametrize("name", LINKS)
def test_a_link_named_twice_is_the_same_error_in_metiscs_words(name):
    ds = _diag(f"dup_{name}.llp")
    assert [d.code for d in ds] == ["rule-duplicate"]
    assert "rule back is named twice among the stage links" in ds[0].text
    assert "".join(render_text(d) for d in ds) == _metisc(f"dup_{name}.llp")


@pytest.mark.parametrize("name", sorted(MIXED))
def test_a_rule_and_a_link_share_no_name_in_metiscs_words(name):
    ds = _diag(f"dup_{name}.llp")
    assert [d.code for d in ds] == ["rule-duplicate"]
    assert f"rule {MIXED[name]} is named twice, as a rule and a stage link" in ds[0].text
    assert "".join(render_text(d) for d in ds) == _metisc(f"dup_{name}.llp")


@pytest.mark.parametrize("pack,base", [("dup_pack", "dup_unit"), ("dup_link_pack", "dup_link_unit"),
                                       ("dup_mixed_pack", "dup_link_unit"), ("dup_mixed_pack_link", "dup_link_unit")])
def test_a_pack_naming_its_bases_rule_or_link_is_refused_as_metisc_refuses_it(pack, base):
    ds = _diag(f"{pack}.llp", f"{base}.llp")
    assert [d.code for d in ds] == ["pack-redeclares"]
    assert "".join(render_text(d) for d in ds) == _metisc(f"{pack}.llp", "--base", f"{base}.llp")


def test_two_names_or_one_name_in_two_stages_stay_legal():
    assert _diag("dup_ok.llp") == []
