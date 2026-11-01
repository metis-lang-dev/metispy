"""`include "unit" hiding (name ...)` (spec/18 §3, hiding-include s-bd3b639d):
this toolchain against metis-lang's corpus/hiding and metisc, and the scanners
that read a program's text instead of parsing it."""

import os
import subprocess
from pathlib import Path

import pytest

from corpus import build_program, parse_manifest
from metis.kernel import program_key
from metis.kernel.artifact import metisc
from metis.kernel.diagnostics import render_text
from metis.lang import ast, compile_catalog, load_llp, parse
from metis.lang.compiler import diagnose, resolve_includes
from metis.lang.directives import embedded_case, render_run

CORPUS = Path(__file__).resolve().parents[3] / "metis-lang/corpus"
HIDING = CORPUS / "hiding"

REFUSED = {"link": "hiding-unknown", "head": "hiding-unknown", "inner": "hiding-unknown",
           "own": "hiding-shadow", "sibling": "hiding-shadow", "nested": "hiding-shadow"}


def _run(name):
    p = HIDING / f"{name}.llp"
    case = embedded_case(p.read_text())
    return render_run(name, case, load_llp(str(p)).catalog.ground(case["init"])) + "\n"


@pytest.mark.parametrize("name", ["hiding_use", "hiding_empty", "hiding_nested"])
def test_the_rows_are_the_goldens(name):
    """hiding_empty hides a whole stage: an empty stage is legal."""
    assert _run(name) == (CORPUS / "goldens/hiding" / f"{name}.txt").read_text()


def test_the_rows_are_the_hand_copys():
    assert _run("hiding_use").split("\n")[1:] == _run("hiding_copy").split("\n")[1:]


def test_the_keys_are_metiscs_and_the_hand_copys():
    gold = dict(l.split()[1:3] for l in (CORPUS / "goldens/hiding/keys.txt").read_text().splitlines()
                if l.startswith("K "))
    got = {c.name: program_key(build_program(c)) for c in parse_manifest(HIDING / "manifest.txt")}
    assert got == gold
    assert got["hiding_use"] == got["hiding_copy"]


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_refusal_is_metiscs_byte_for_byte(name):
    here = os.getcwd()
    os.chdir(HIDING)
    try:
        ds = diagnose(f"hiding_bad_{name}.llp")
        assert [d.code for d in ds] == [REFUSED[name]]
        mc = metisc()
        if not mc:
            pytest.skip("no metisc: the python half only")
        r = subprocess.run([mc, "--diagnostics", f"hiding_bad_{name}.llp"], capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=60)
        assert "".join(render_text(d) for d in ds) == r.stdout + r.stderr
    finally:
        os.chdir(here)


def test_the_clause_prints_sorted_each_once_and_only_when_there_is_one():
    src = 'catalog c 1.\ninclude "u.llp" hiding (b a b).\ninclude "v.llp".\n'
    inc, plain = [d for d in parse(src).decls if isinstance(d, ast.IncludeDecl)]
    assert inc.hiding == ("a", "b") and plain.hiding == ()
    assert ast.decl_line(inc) == 'include "u.llp" hiding (a b).'
    assert ast.decl_line(plain) == 'include "v.llp".'
    assert parse(ast.pretty(parse(src))) == parse(src)


def test_a_rule_name_takes_all_its_heads():
    assert (ast.rule_base("turn-1"), ast.rule_base("turn"), ast.rule_base("a-b"), ast.rule_base("-1")) \
        == ("turn", "turn", "a-b", "-1")
    cat = resolve_includes(parse(f'catalog c 1.\ninclude "{HIDING}/hiding_unit.llp" hiding (turn).\n'), HIDING)
    names = [r.name for d in cat.decls if isinstance(d, ast.Stage) for r in d.rules]
    assert names and not [n for n in names if ast.rule_base(n) == "turn"]
    compile_catalog(cat)


def test_the_bundle_is_metiscs(tmp_path):
    """The linked member's bundle: the sidecar walks the splice, in both toolchains."""
    from metis.kernel import artifact
    mc = metisc()
    if not mc:
        pytest.skip("no metisc: the python half only")
    subprocess.run([mc, "--link", "hiding/composition_h.yaml", "-o", str(tmp_path)], cwd=CORPUS / "iface", check=True,
                   capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    linked = next(tmp_path.glob("hiding_h.image.v-*")) / "members/h"
    files = artifact.build(CORPUS / "iface/hiding/h/h.llp", [])
    assert files["ground.wire"] == (linked / "ground.wire").read_text()
    assert files["sidecar.json"] == (linked / "sidecar.json").read_text()
    assert '"rule": "look"' not in files["sidecar.json"].replace('"rule":"', '"rule": "')


UNIT = """pred lit : u.
pred cmd_on : u.
pred cmd_off : u.
stage act {
  %% It lights.
  on_rule [dyn] : cmd_on -o lit.
  %% It goes out,
  %% and it says so.
  off_rule [dyn] : lit * cmd_off
    -o cmd_on.
}
"""


def _unit(tmp_path, clause):
    (tmp_path / "unit.llp").write_text(UNIT)
    (tmp_path / "mid.llp").write_text('include "unit.llp" hiding (off_rule).\n')
    (tmp_path / "p.llp").write_text(f'catalog p 1.\n{clause}\n')
    return tmp_path / "p.llp"


def test_the_intent_scan_reads_no_hidden_rule(tmp_path):
    from metis.intents import program_rules, program_text
    text = program_text(_unit(tmp_path, 'include "unit.llp" hiding (off_rule).'))
    assert "include" not in text and "off_rule" not in text and "goes out" not in text and "says so" not in text
    assert "pred cmd_off" in text and [r[:2] for r in program_rules(text)] == [("It lights.", "on_rule")]
    nested = program_text(_unit(tmp_path, 'include "mid.llp".'))
    assert "off_rule" not in nested and "on_rule" in nested
    assert "off_rule" in program_text(_unit(tmp_path, 'include "unit.llp".'))


def test_take_and_the_cards_follow_a_hiding_include(tmp_path):
    from metis import catalog
    from metis.loop.cards import _scan
    p = _unit(tmp_path, 'include "unit.llp" hiding (off_rule).')
    assert catalog.includes(tmp_path, [p]) == {"unit.llp": UNIT}
    assert _scan(p)["includes"] == [str(tmp_path / "unit.llp")]
