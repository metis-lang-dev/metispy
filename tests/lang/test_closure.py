"""The closure declarations (spec/00 §5a, patterns CHECKS.md §4):
`family` (the host's exclusive propositions: read, never written) and
`invariant` (a linear P-invariant the rules provably keep), and the
linter (spec/08 lint phase) over a program that compiles."""

import os
import subprocess
import textwrap

import pytest

from langpath import LANG
from metis.kernel.artifact import metisc
from metis.kernel.diagnostics import render_text
from metis.lang import ast, parser
from metis.lang.compiler import LangError, compile_catalog, diagnose, load_llp
from metis.lint import init_counts, invariant_init, lint

CLOSURE = LANG / "corpus/diag/diag_closure.llp"

HEAD = """\
catalog t 1.
layers (dyn).
stages (act).
namespace m produce (dyn).
pred idle : m.
pred busy : m.
pred away : m.
pred go : m.
pred halt : m.
"""


def _compile(body, decls=""):
    return compile_catalog(parser.parse(HEAD + decls + "stage act {\n"
                                        + textwrap.dedent(body) + "}\n"))


def test_round_trip_prints_the_declarations_in_place():
    cat = parser.parse(HEAD + "family way (go halt).\n"
                       "invariant mode (idle busy away) : 1.\n"
                       "stage act {\n  %% d\n  s [dyn] : $go * idle -o busy.\n}\n")
    kinds = [type(d).__name__ for d in cat.decls]
    assert "FamilyDecl" in kinds and "InvariantDecl" in kinds
    assert parser.parse(ast.pretty(cat)) == cat
    assert "family way (go halt)." in ast.pretty(cat)
    assert "invariant mode (idle busy away) : 1." in ast.pretty(cat)


def test_family_and_invariant_stay_identifiers_elsewhere():
    # a pred named `family` and a rule reading it: no keyword collision
    cat = parser.parse(HEAD + "pred family : m.\nstage act {\n  %% d\n"
                       "  s [dyn] : $family * idle -o busy.\n}\n")
    assert any(isinstance(d, ast.PredDecl) and d.name == "family"
               for d in cat.decls)


def test_invariant_holds_on_the_responder_shape():
    doc = _compile("""\
        %% go
        start [dyn] : $go * idle -o busy.
        %% away
        leave [dyn] : $halt * idle -o away.
        %% back
        stop [dyn] : busy -o idle.
        %% back
        home [dyn] : away -o idle.
        """, "family way (go halt).\ninvariant mode (idle busy away) : 1.\n")
    assert doc.families == [("way", ("go", "halt"))]
    assert doc.invariants == [("mode", ("idle", "busy", "away"), 1)]
    assert doc.warnings == []


def test_invariant_broken_names_the_rule_and_the_delta():
    with pytest.raises(LangError) as e:
        _compile("""\
            %% two
            split [dyn] : idle -o busy * away.
            """, "invariant mode (idle busy away) : 1.\n")
    (d,) = e.value.diagnostics
    assert d.code == "invariant-broken"
    assert dict(d.data) == {"rule": "split", "invariant": "mode",
                            "delta": "+1"}
    assert d.loc.decl == "act/split"


def test_family_written_is_refused():
    with pytest.raises(LangError) as e:
        _compile("""\
            %% wish
            wish [dyn] : $halt * idle -o go.
            """, "family way (go halt).\n")
    codes = [d.code for d in e.value.diagnostics]
    assert codes == ["family-produced"]


def test_pack_rules_are_held_to_the_base_invariant():
    base = _compile("""\
        %% go
        start [dyn] : $go * idle -o busy.
        """, "invariant mode (idle busy away) : 1.\n")
    pack = parser.parse("catalog p 1.\nextends t.\nstage act {\n"
                        "  %% twice\n  dup [dyn] : $go * busy -o busy * away.\n}\n")
    with pytest.raises(LangError) as e:
        compile_catalog(pack, base=base)
    assert [d.code for d in e.value.diagnostics] == ["invariant-broken"]


def test_pack_is_not_blamed_for_the_base_readers():
    base = _compile("""\
        %% go
        start [dyn] : $go * idle -o busy.
        """, "family way (go halt).\n")
    pack = parser.parse("catalog p 1.\nextends t.\nstage act {\n"
                        "  %% back\n  back [dyn] : busy -o idle.\n}\n")
    doc = compile_catalog(pack, base=base)
    assert doc.warnings == [] and doc.families == [("way", ("go", "halt"))]


def test_init_counts_squash_inner_spaces():
    assert init_counts("#init at(e0, f0) 2\n#init idle\n") == {"at(e0,f0)": 2,
                                                              "idle": 1}


def test_fixture_rows_one_per_code():
    here = os.getcwd()
    os.chdir(CLOSURE.parent)
    try:
        codes = [d.code for d in diagnose(CLOSURE.name)]
    finally:
        os.chdir(here)
    assert codes == ["family-member-unknown", "invariant-member-unknown",
                     "family-produced", "invariant-broken", "family-unread"]


def test_fixture_in_metiscs_words():
    mc = metisc()
    if not mc:
        pytest.skip("no metisc: the python half only")
    r = subprocess.run([mc, "--diagnostics", CLOSURE.name], cwd=CLOSURE.parent,
                       capture_output=True, text=True,
                       stdin=subprocess.DEVNULL, timeout=60)
    here = os.getcwd()
    os.chdir(CLOSURE.parent)
    try:
        mine = "".join(render_text(d) for d in diagnose(CLOSURE.name))
    finally:
        os.chdir(here)
    assert (r.stdout + r.stderr) == mine


LINT = """\
catalog l 1.
layers (dyn).
stages (act).
namespace m produce (dyn).
pred idle : m.
pred busy : m.
pred go : m.
pred stuck : m.
pred orphan : m.
pred rep(tick__gen) : m.
pred tick__gen(tick__gen) : m.
type tick__gen {g0 g1}.
invariant mode (idle busy) : 1.
stage act {
  %% start
  start [dyn] : $go * idle -o busy.
  %% the same premises, another head
  start2 [dyn] : $go * idle -o busy * stuck.
  %% stop: stuck is left with no consumer
  stop [dyn] : busy -o idle * stuck.
  %% orphan: nothing makes it
  back [dyn] : $orphan * busy -o idle.
  %% echo: a report consumed without its generation
  echo [dyn] : rep(G) -o ().
}
#init idle
#init busy
#init go
"""


def test_lint_findings(tmp_path):
    f = tmp_path / "l.llp"
    f.write_text(LINT)
    doc = load_llp(f)
    found = lint(doc, LINT, None, f)
    assert [(d.code, dict(d.data).get("rule") or dict(d.data).get("pred"))
            for d in found] == [
        ("state-dead-end", "stuck"), ("state-unproduced", "orphan"),
        ("state-unproduced", "rep"), ("rules-twin-body", "start2"),
        ("report-ungated", "echo")]
    assert all(d.severity == "warning" for d in found)
    # the init half: idle and busy both held, declared 1
    (d,) = invariant_init(doc, init_counts(LINT))
    assert d.code == "invariant-init" and dict(d.data)["held"] == "2"


def test_lint_manifest_exempts_the_interface(tmp_path):
    f = tmp_path / "l.llp"
    twin = "  %% the same premises, another head\n  start2 [dyn] : $go * idle -o busy * stuck.\n"
    assert twin in LINT
    f.write_text(LINT.replace(twin, ""))
    (tmp_path / "domain.yaml").write_text(
        "shared: {orphan: in, rep: in, stuck: out}\n"
        "status: {success: [stuck], failure: []}\n")
    doc = load_llp(f)
    found = lint(doc, LINT, tmp_path / "domain.yaml", f)
    assert [d.code for d in found] == ["report-ungated"]


def test_lint_cli_is_quiet_on_the_road_responder():
    road = LANG.parent / "metis-catalog/domains/road/responder"
    if not (road / "responder.llp").is_file():
        pytest.skip("no metis-catalog beside")
    r = subprocess.run(["python3", "-m", "metis.lint", "responder.llp",
                        "--manifest", "domain.yaml", "--strict"],
                       cwd=road, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(LANG.parent / "metispy")})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no lint finding" in r.stdout


def _lint_both(where, args):
    py = subprocess.run(["python3", "-m", "metis.lint", *args], cwd=where, capture_output=True, text=True, timeout=120,
                        env={**os.environ, "PYTHONPATH": str(LANG.parent / "metispy")})
    return py


def test_lint_checks_fixture_fires_l7_l8_l9():
    """checks-open s-17e557f0: the lights' first shape held a stop on the class it entered on; with its manifest a safety
    rule reads a belief and two acts no command begins (stop is none, stop_lat is). metisc's bytes are the corpus golden."""
    where = LANG / "corpus/lint"
    if not (where / "lint_checks.llp").is_file():
        pytest.skip("no metis-lang beside")
    bare = _lint_both(where, ["lint_checks.llp"]).stdout
    full = _lint_both(where, ["lint_checks.llp", "--manifest", "lint_checks.yaml"]).stdout
    codes = lambda out: [ln.split()[1] for ln in out.splitlines() if ln.startswith("warning ")]
    assert codes(bare) == ["state-unproduced", "state-unproduced", "enter-equals-hold"]
    assert codes(full) == ["belief-in-safety", "enter-equals-hold", "name-not-a-command", "name-not-a-command"]
    assert "(act/stop_here)" in full and "(act/slow_down)" in full
    golden = (LANG / "corpus/goldens/lint_goldens.txt").read_text()
    assert ("== metisc --lint lint_checks.llp --manifest lint_checks.yaml\n" + full) in golden


@pytest.mark.parametrize("member", ["responder", "planner", "signals"])
def test_lint_is_quiet_on_the_road_members_with_their_manifest_lines(member):
    road = LANG.parent / "metis-catalog/domains/road" / member
    if not (road / ("%s.llp" % member)).is_file():
        pytest.skip("no metis-catalog beside")
    if "commands:" not in (road / "domain.yaml").read_text():
        pytest.skip("the catalog before its lint lines (checks-open s-17e557f0)")
    r = _lint_both(road, ["%s.llp" % member, "--manifest", "domain.yaml", "--strict"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no lint finding" in r.stdout

