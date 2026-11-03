"""metis.gate (loop-parity s-72d9d36b; moved from metis_loop_sdk.gate by metis-packages s-65778b98): the loop's policy over the catalog —
propose shows the plan, confirm writes exactly it, refuse remembers; the kinds new, edit, take and ask
(workspace-files s-9ea4cfa4: no copy of a text over a package's file), the seed-first rule, the refused-first
rule, the take, the journal row of a workspace, and the CLI keeping its memory in the loop's state (gate.json)
across calls."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from catalogs import installed_catalog
from metis import catalog, gate

METISC = catalog.metisc()
needs_metisc = pytest.mark.skipif(not METISC, reason="no metisc (METISC or PATH): a new package's program is checked")
CAT = installed_catalog()
if CAT is None or not (CAT / "domains" / "plant").is_dir():
    pytest.skip(f"no metis-catalog plant at {CAT} (METIS_CATALOG)", allow_module_level=True)
LAMP = "% lamp — a lamp\ncatalog lamp 1.\nlayers (dyn).\nstages (main).\nnamespace lamp produce (dyn).\n" \
       "pred off : lamp.\npred on : lamp.\npred cmd_on : lamp.\nstage main {\n  %% I switch it on.\n" \
       "  on [dyn] : cmd_on * off -o on.\n}\n#init off 1\n"


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "cat"
    shutil.copytree(CAT / "domains" / "plant", r / "domains" / "plant",
                    ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    return r


def test_the_kinds_are_new_edit_take_ask():
    assert gate.KINDS == ("new", "edit", "take", "ask") and gate.JUDGED == ("new", "edit")


@needs_metisc
def test_propose_then_confirm_writes_exactly_the_plan(root):
    g = gate.Gate()
    pl = g.propose("new", "lamp", root, text=LAMP, display="the lamp", line="a lamp")
    assert [f["path"] for f in pl["files"]] == ["domains/lamp/lamp.llp", "domains/lamp/domain.yaml"]
    assert all(f["new"] for f in pl["files"]) and pl["reopen"] and len(pl["token"]) == 16
    assert not (root / "domains" / "lamp").exists()                           # a plan writes nothing
    assert all(not x["refuses"] for x in pl["findings"])
    mf = next(f for f in pl["files"] if f["path"].endswith("domain.yaml"))["text"]
    assert "program: lamp.llp" in mf and "display: the lamp" in mf and "host:" not in mf
    out = g.confirm(pl["token"])
    assert sorted(out["written"]) == ["domains/lamp/domain.yaml", "domains/lamp/lamp.llp"]
    assert out["serve"] == {"program": str((root / "domains/lamp/lamp.llp").resolve()), "host": None}
    with pytest.raises(gate.Refused, match="no such plan"):
        g.confirm(pl["token"])                                                # a plan is written once
    with pytest.raises(gate.Refused, match="lamp: the catalog has it: change it with edit"):
        g.propose("new", "lamp", root, text=LAMP)


@needs_metisc
def test_a_new_package_is_one_plan_with_its_extensions_intents_and_readme(root):
    g = gate.Gate()
    ext = "catalog bright 1.\nextends lamp.\nstage main {\n  %% I stay on.\n  stay [dyn] : on -o on.\n}\n"
    pl = g.propose("new", "lamp", root, text=LAMP, extensions={"bright": ext}, intents=True,
                   readme={"text": "A lamp you switch on.", "title": "a lamp"})
    paths = [f["path"] for f in pl["files"]]
    assert {"domains/lamp/lamp.llp", "domains/lamp/extensions/bright.llp", "domains/lamp/intent.yaml",
            "domains/lamp/README.md", "domains/lamp/domain.yaml"} <= set(paths)
    rd = next(f for f in pl["files"] if f["path"] == "domains/lamp/README.md")["text"]
    assert rd.startswith("# lamp — a lamp\n\nA lamp you switch on.") and "`extensions/bright.llp`" in rd
    g.confirm(pl["token"])
    assert catalog.get("lamp", root).intent and (root / "domains/lamp/extensions/bright.llp").read_text() == ext
    with pytest.raises(gate.Refused, match="lowercase"):
        g.propose("new", "lamp2", root, text=LAMP, extensions={"Tired Plant": "x"})
    with pytest.raises(gate.Refused, match="readme: text is the summary paragraph"):
        g.propose("new", "lamp3", root, text=LAMP, readme={"text": "  "})


@needs_metisc
def test_a_new_package_that_does_not_compile_is_refused_before_a_plan(root):
    g = gate.Gate()
    with pytest.raises(gate.Refused, match="(?s)adds 1 diagnostic.*lamp.llp: error"):
        g.propose("new", "lamp", root, text=LAMP.replace("-o on.", "-o onn."))
    assert g.store["plans"] == {}


@needs_metisc
def test_a_new_domain_names_the_host_beside_it(root):
    (root / "domains" / "lamp").mkdir(parents=True)
    (root / "domains" / "lamp" / "lamp_host.py").write_text("# the world\n")
    g = gate.Gate()
    pl = g.propose("new", "lamp", root, text=LAMP)
    mf = next(f for f in pl["files"] if f["path"].endswith("domain.yaml"))["text"]
    assert "host: lamp_host.py:make" in mf


@needs_metisc
def test_a_world_with_no_commands_gets_no_intents_and_a_note(root):
    still = "catalog still 1.\nlayers (d).\nstages (s).\nnamespace w produce (d).\npred a : w.\npred b : w.\n" \
            "stage s {\n  %% a becomes b.\n  ab [d] : a -o b.\n}\n#init a 1\n"
    pl = gate.Gate().propose("new", "still", root, text=still, intents=True)
    assert "domains/still/intent.yaml" not in [f["path"] for f in pl["files"]] and "no intent.yaml" in pl["say"]


def test_a_source_changed_since_the_plan_refuses(root):
    g = gate.Gate()
    rd = root / "domains/plant/README.md"
    rd.write_text("# plant\n\nA plant that dries out.\n")             # an edit of a non-.llp: no metisc needed
    pl = g.propose("edit", "plant", root, path="README.md", old="dries out", new="dries up")
    assert pl["findings"] == [] and not pl["reopen"]
    rd.write_text(rd.read_text().replace("dries out", "dries out slowly"))
    with pytest.raises(gate.Refused, match="README.md changed after the person approved the edit: nothing written"):
        g.confirm(pl["token"])


def test_refuse_remembers_and_a_write_forgets(root):
    g = gate.Gate()
    rd = root / "domains/plant/README.md"
    rd.write_text("# plant\n\nA plant that dries out.\n")
    pl = g.propose("edit", "plant", root, path="README.md", old="dries out", new="dries up")
    g.refuse(pl["token"])
    assert "domains/plant/README.md" in g.refused(root)
    again = g.propose("edit", "plant", root, path="README.md", old="dries out", new="dries out, watered")
    f = next(f for f in again["files"] if f["path"] == "domains/plant/README.md")
    assert "since_diff" in f and "+A plant that dries out, watered." in f["since_diff"]
    g.confirm(again["token"])
    assert "domains/plant/README.md" not in g.refused(root)                    # written: forgotten
    third = g.propose("edit", "plant", root, path="README.md", old="watered", new="watered daily")
    assert all("since_diff" not in f for f in third["files"])


def test_the_refused_first_rule(root):
    """A file that was never written because its proposal was refused is made again (old empty), its gate
    showing what changed since; an edit of its text is refused in words."""
    (root / "domains/plant/README.md").unlink(missing_ok=True)
    g = gate.Gate()
    g.refuse(g.propose("edit", "plant", root, path="README.md", old="", new="# plant\n\nA plant.\n")["token"])
    with pytest.raises(gate.Refused, match="README.md was never written: the person refused its proposal. "
                                           "Propose it again with old_string empty"):
        g.propose("edit", "plant", root, path="README.md", old="A plant.", new="A plant, watered.")
    again = g.propose("edit", "plant", root, path="README.md", old="", new="# plant\n\nA plant, watered.\n")
    assert "+A plant, watered." in again["files"][0]["since_diff"]


def test_the_refused_first_rule_for_a_package_outside_domains(tmp_path):
    """cwd-root rule 9: a package anywhere under the root (DIR/plant/, no domains/) — the
    refused file is remembered by its path from the root, and the rule says its words."""
    r = tmp_path / "flat"
    shutil.copytree(CAT / "domains" / "plant", r / "plant", ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    (r / "plant/README.md").unlink(missing_ok=True)
    g = gate.Gate()
    pl = g.propose("edit", "plant", r, path="README.md", old="", new="# plant\n\nA plant.\n")
    assert [f["path"] for f in pl["files"]] == ["plant/README.md"]
    g.refuse(pl["token"])
    with pytest.raises(gate.Refused, match="README.md was never written: the person refused its proposal"):
        g.propose("edit", "plant", r, path="./README.md", old="A plant.", new="A plant, watered.")


def test_the_journal_row_of_a_workspace(root):
    """A root with a .metis/ journals the write: who (the caller's), kind, the domain, the files."""
    (root / ".metis").mkdir()
    rd = root / "domains/plant/README.md"
    rd.write_text("# plant\n\nA plant that dries out.\n")
    g = gate.Gate()
    g.confirm(g.propose("edit", "plant", root, path="README.md", old="dries out", new="dries up", who="pi")["token"])
    rows = [json.loads(x) for x in (root / ".metis/loop/journal.jsonl").read_text().splitlines()]
    assert [(r["who"], r["kind"], r["domain"], r["files"]) for r in rows] == \
        [("pi", "edit", "plant", ["domains/plant/README.md"])]
    with pytest.raises(gate.Refused, match="who: backend | pi"):
        g.propose("edit", "plant", root, path="README.md", old="up", new="down", who="judge")


def test_the_cli_memory_is_locked_atomic_and_its_metis_ignored(root, tmp_path):
    """gate.json is written whole (a temp file replaced), under gate.lock; a new .metis/
    is made through catalog.state(), so a checkout ignores it."""
    (root / ".git").mkdir()
    st = root / ".metis"
    rc, pl = _cli(st, "propose", "edit", "plant", "--root", str(root),
                  stdin=json.dumps({"path": "README.md", "old": "plant", "new": "Plant", "replace_all": True}))
    assert rc == 0 and (st / "gate.json").is_file() and (st / "gate.lock").is_file()
    assert not (st / "gate.json.tmp").exists()
    assert "/.metis/" in (root / ".gitignore").read_text()


@needs_metisc
def test_seed_first(root, tmp_path):
    """A seeded run makes the seed first, under its own name; then the DIR is a catalog like any."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "seed.json").write_text(json.dumps({"seed": "x.llp", "name": "lamp"}))
    g = gate.Gate()
    with pytest.raises(gate.Refused, match=r"a seeded run makes the seed first \(catalog_new domain=lamp program=served\)"):
        g.propose("new", "other", root, run=run, text=LAMP.replace("lamp", "other"))
    with pytest.raises(gate.Refused, match=r"a seeded run makes the seed first \(catalog_new domain=lamp program=served\)"):
        g.propose("new", "lamp", root, run=run, text=LAMP)              # the seed's own name, but not the seed
    with pytest.raises(gate.Refused, match="the REPL serves a blank world"):
        g.propose("new", "lamp", root, run=run, served=True)
    g.confirm(g.propose("new", "lamp", root, run=run, text=LAMP, served=True, display="the lamp")["token"])
    assert (root / "domains/lamp/lamp.llp").read_text() == LAMP
    assert g.propose("new", "other", root, run=run, text=LAMP.replace("lamp", "other"))["files"]


def test_a_package_seed_comes_whole_and_says_its_display_is_its_own(root, tmp_path):
    run = tmp_path / "run"
    pkg = run / "seed" / "domains" / "pigs"
    pkg.mkdir(parents=True)
    (pkg / "domain.yaml").write_text("display: the pigs\nline: three pigs\nprogram: pigs.llp\n")
    (pkg / "pigs.llp").write_text("% the pigs\ncatalog pigs 1.\n")
    (run / "seed.json").write_text(json.dumps({"seed": str(pkg), "name": "pigs", "package": True}))
    g = gate.Gate()
    with pytest.raises(gate.Refused, match=r"a seeded run makes the seed first \(catalog_new domain=pigs program=served\)"):
        g.propose("new", "other", root, run=run, served=True)
    pl = g.propose("new", "pigs", root, run=run, served=True, display="the dog")
    assert "was not" not in pl.get("say", "") and 'display "the pigs" and line "three pigs"' in pl["say"]
    out = g.confirm(pl["token"])
    assert sorted(out["written"]) == ["domains/pigs/domain.yaml", "domains/pigs/pigs.llp"] and out["serve"]["program"]
    # a package that brings no README gets one drafted around the summary, its own files as they are
    (root / "domains/pigs").rename(root / "pigs-taken")
    pl = g.propose("new", "pigs", root, run=run, served=True, readme={"text": "Three pigs.", "title": "the pigs"})
    rd = next(f for f in pl["files"] if f["path"] == "domains/pigs/README.md")["text"]
    assert rd.startswith("# pigs — the pigs\n\nThree pigs.\n") and "seeded from" in rd


def test_take_into_another_catalog_with_the_runs_readme(root, tmp_path):
    into = tmp_path / "into"
    into.mkdir()
    g = gate.Gate()
    (root / "domains/plant/README.md").unlink(missing_ok=True)
    run = tmp_path / "run"
    run.mkdir()
    pl = g.propose("take", "plant", into, run=run, src=str(root))
    assert any(f["path"] == "domains/plant/README.md" for f in pl["files"])     # the run's README
    out = g.confirm(pl["token"])
    assert (into / "domains/plant/plant.llp").is_file() and "bundle" in out
    with pytest.raises(ValueError, match="has domains/plant already"):
        g.propose("take", "plant", into, run=run, src=str(root))
    with pytest.raises(gate.Refused, match="take: src"):
        g.propose("take", "plant", into)


def test_the_rules_words_for_bad_input(root):
    g = gate.Gate()
    with pytest.raises(gate.Refused, match="kind: new | edit | take | ask"):
        g.propose("program", "lamp", root)
    with pytest.raises(gate.Refused, match='domain "Lamp": lowercase letters, digits, _'):
        g.propose("new", "Lamp", root, text=LAMP)
    with pytest.raises(gate.Refused, match="new: text, the program's text"):
        g.propose("new", "lamp", root)


def _cli(state, *args, stdin=""):
    r = subprocess.run([sys.executable, "-m", "metis.gate", "--state", str(state), *args, "--json"],
                       input=stdin, capture_output=True, text=True, timeout=120)
    return r.returncode, json.loads(r.stdout)


def test_the_cli_keeps_the_memory_in_the_state(root, tmp_path):
    st = tmp_path / ".metis"
    (root / "domains/plant/README.md").write_text("# plant\n\nA plant.\n")
    edit = lambda new: json.dumps({"path": "README.md", "old": "A plant.", "new": new})   # noqa: E731
    rc, pl = _cli(st, "propose", "edit", "plant", "--root", str(root), stdin=edit("A plant, dry."))
    assert rc == 0 and pl["token"] and "write" not in pl
    assert _cli(st, "refuse", pl["token"]) == (0, {"refused": [f["path"] for f in pl["files"]]})
    rc, again = _cli(st, "propose", "edit", "plant", "--root", str(root), stdin=edit("A plant, watered."))
    assert any("since_diff" in f for f in again["files"])                     # remembered across processes
    rc, out = _cli(st, "confirm", again["token"])
    assert rc == 0 and "domains/plant/README.md" in out["written"]
    assert _cli(st, "confirm", again["token"]) == (1, {"error": "not approved: no such plan (propose it again)"})
    rc, err = _cli(st, "propose", "edit", "lamp", "--root", str(root), stdin=edit("x"))
    assert rc == 1 and "lamp" in err["error"]
    assert json.loads((st / "gate.json").read_text())["plans"] == {}
