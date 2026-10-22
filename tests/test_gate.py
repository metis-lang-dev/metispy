"""metis.gate (loop-parity s-72d9d36b; moved from metis_loop_sdk.gate by metis-packages s-65778b98): the loop's policy over the catalog —
propose shows the plan, confirm writes exactly it, refuse remembers; the seed-first rule,
the README-refused rule, a note needs no confirmation, the take, and the CLI keeping its
memory in the loop's state (gate.json) across calls."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from catalogs import installed_catalog
from metis import catalog, gate

CAT = installed_catalog()
LAMP = "% lamp — a lamp\ncatalog lamp 1.\nlayers (dyn).\nstages (main).\nnamespace lamp produce (dyn).\n" \
       "pred off : lamp.\npred on : lamp.\npred cmd_on : lamp.\nstage main {\n  %% I switch it on.\n" \
       "  on [dyn] : cmd_on * off -o on.\n}\n#init off 1\n"


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "cat"
    shutil.copytree(CAT / "domains" / "plant", r / "domains" / "plant",
                    ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    return r


def test_propose_then_confirm_writes_exactly_the_plan(root):
    g = gate.Gate()
    pl = g.propose("program", "lamp", root, text=LAMP, display="the lamp", line="a lamp")
    assert [f["path"] for f in pl["files"]] == ["domains/lamp/lamp.llp", "domains/lamp/domain.yaml"]
    assert all(f["new"] for f in pl["files"]) and pl["reopen"] and len(pl["token"]) == 16
    assert not (root / "domains" / "lamp").exists()                           # a plan writes nothing
    mf = next(f for f in pl["files"] if f["path"].endswith("domain.yaml"))["text"]
    assert "program: lamp.llp" in mf and "display: the lamp" in mf and "host:" not in mf
    out = g.confirm(pl["token"])
    assert sorted(out["written"]) == ["domains/lamp/domain.yaml", "domains/lamp/lamp.llp"]
    assert out["serve"] == {"program": str((root / "domains/lamp/lamp.llp").resolve()), "host": None}
    with pytest.raises(gate.Refused, match="no such plan"):
        g.confirm(pl["token"])                                                # a plan is written once


def test_a_program_plan_carries_its_extensions(root):
    """loop-parity step 2: the console writes a program with its extensions in one plan."""
    g = gate.Gate()
    plant = (root / "domains/plant/plant.llp").read_text()
    pl = g.propose("program", "plant", root, text=plant, extensions={"tired": "% an extension\n"})
    assert [f["path"] for f in pl["files"]] == ["domains/plant/extensions/tired.llp"]   # the program is unchanged
    g.confirm(pl["token"])
    assert (root / "domains/plant/extensions/tired.llp").read_text() == "% an extension\n"
    with pytest.raises(gate.Refused, match="lowercase"):
        g.propose("program", "plant", root, text=plant, extensions={"Tired Plant": "x"})


def test_a_new_domain_names_the_host_beside_it(root):
    (root / "domains" / "lamp").mkdir(parents=True)
    (root / "domains" / "lamp" / "lamp_host.py").write_text("# the world\n")
    g = gate.Gate()
    pl = g.propose("program", "lamp", root, text=LAMP)
    mf = next(f for f in pl["files"] if f["path"].endswith("domain.yaml"))["text"]
    assert "host: lamp_host.py:make" in mf


def test_a_source_changed_since_the_plan_refuses(root):
    g = gate.Gate()
    rd = root / "domains/plant/README.md"
    rd.write_text("# plant\n\nA plant that dries out.\n")             # an edit of a non-.llp: no metisc needed
    pl = g.propose("edit", "plant", root, path="README.md", old="dries out", new="dries up")
    rd.write_text(rd.read_text().replace("dries out", "dries out slowly"))
    with pytest.raises(gate.Refused, match="README.md changed after the person approved the edit: nothing written"):
        g.confirm(pl["token"])
    prog = root / "domains/plant/plant.llp"
    pl = g.propose("program", "plant", root, text=LAMP.replace("lamp", "plant"))
    prog.write_text(prog.read_text() + "% moved\n")
    pl2 = g.propose("intents", "plant", root)                 # the plans of other writes stay apart
    assert pl2["token"] != pl["token"]


def test_refuse_remembers_and_a_write_forgets(root):
    g = gate.Gate()
    pl = g.propose("readme", "plant", root, text="A plant that dries out.", title="a plant")
    g.refuse(pl["token"])
    assert "domains/plant/README.md" in g.refused(root)
    again = g.propose("readme", "plant", root, text="A plant that dries out, watered.", title="a plant")
    rd = next(f for f in again["files"] if f["path"] == "domains/plant/README.md")
    assert "since_diff" in rd and "+A plant that dries out, watered." in rd["since_diff"]
    g.confirm(again["token"])
    assert "domains/plant/README.md" not in g.refused(root)                    # written: forgotten
    third = g.propose("readme", "plant", root, text="A plant.", title="a plant")
    assert all("since_diff" not in f for f in third["files"])


def test_the_readme_refused_rule(root):
    """An edit of a file that was never written because its proposal was refused
    goes through the proposal again (its gate shows what changed since)."""
    (root / "domains/plant/README.md").unlink(missing_ok=True)
    g = gate.Gate()
    g.refuse(g.propose("readme", "plant", root, text="A plant.", title="a plant")["token"])
    with pytest.raises(gate.Refused, match="README.md was never written: the person refused its proposal. "
                                           "Send it again through catalog_apply with only the fix"):
        g.propose("edit", "plant", root, path="README.md", old="A plant.", new="A plant, watered.")


def test_the_readme_refused_rule_for_a_package_outside_domains(tmp_path):
    """cwd-root rule 9: a package anywhere under the root (DIR/plant/, no domains/) — the
    refused file is remembered by its path from the root, and the rule says its words."""
    r = tmp_path / "flat"
    shutil.copytree(CAT / "domains" / "plant", r / "plant", ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    (r / "plant/README.md").unlink(missing_ok=True)
    g = gate.Gate()
    pl = g.propose("readme", "plant", r, text="A plant.", title="a plant")
    assert [f["path"] for f in pl["files"]] == ["plant/README.md"]
    g.refuse(pl["token"])
    with pytest.raises(gate.Refused, match="README.md was never written: the person refused its proposal"):
        g.propose("edit", "plant", r, path="./README.md", old="A plant.", new="A plant, watered.")


def test_the_cli_memory_is_locked_atomic_and_its_metis_ignored(root, tmp_path):
    """gate.json is written whole (a temp file replaced), under gate.lock; a new .metis/
    is made through catalog.state(), so a checkout ignores it."""
    (root / ".git").mkdir()
    st = root / ".metis"
    rc, pl = _cli(st, "propose", "readme", "plant", "--root", str(root), stdin=json.dumps({"text": "A.", "title": "a"}))
    assert rc == 0 and (st / "gate.json").is_file() and (st / "gate.lock").is_file()
    assert not (st / "gate.json.tmp").exists()
    assert "/.metis/" in (root / ".gitignore").read_text()


def test_readme_refusals(root):
    g = gate.Gate()
    with pytest.raises(gate.Refused, match="no domain lamp in the loop's catalog yet: its README follows its program"):
        g.propose("readme", "lamp", root, text="x")
    with pytest.raises(gate.Refused, match="what=readme: text is the summary paragraph"):
        g.propose("readme", "plant", root, text="  ")


def test_seed_first(root, tmp_path):
    """A seeded run applies the seed first, under its own name; then the DIR is a catalog like any."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "seed.json").write_text(json.dumps({"seed": "x.llp", "name": "lamp"}))
    g = gate.Gate()
    for kind, kw in (("program", {}), ("extension", {"name": "x"})):
        for d in ("lamp", "other"):
            with pytest.raises(gate.Refused, match=rf"a seeded run applies the seed first \(catalog_apply domain={d} what=served\)"):
                g.propose(kind, d, root, run=run, text=LAMP, **kw)
    with pytest.raises(gate.Refused, match="the seed is lamp: catalog_apply domain=lamp what=served"):
        g.propose("served", "other", root, run=run, text=LAMP)
    g.confirm(g.propose("served", "lamp", root, run=run, text=LAMP, display="the lamp")["token"])
    assert (root / "domains/lamp/lamp.llp").read_text() == LAMP
    assert g.propose("program", "other", root, run=run, text=LAMP.replace("lamp", "other"))["files"]


def test_a_package_seed_comes_whole_and_says_its_display_is_its_own(root, tmp_path):
    run = tmp_path / "run"
    pkg = run / "seed" / "domains" / "pigs"
    pkg.mkdir(parents=True)
    (pkg / "domain.yaml").write_text("display: the pigs\nline: three pigs\nprogram: pigs.llp\n")
    (pkg / "pigs.llp").write_text("% the pigs\ncatalog pigs 1.\n")
    (run / "seed.json").write_text(json.dumps({"seed": str(pkg), "name": "pigs", "package": True}))
    g = gate.Gate()
    with pytest.raises(gate.Refused, match="the seed is the package pigs: catalog_apply domain=pigs what=served"):
        g.propose("served", "other", root, run=run)
    pl = g.propose("served", "pigs", root, run=run, display="the dog")
    assert "was not" not in pl.get("say", "") and 'display "the pigs" and line "three pigs"' in pl["say"]
    out = g.confirm(pl["token"])
    assert sorted(out["written"]) == ["domains/pigs/domain.yaml", "domains/pigs/pigs.llp"] and out["serve"]["program"]


def test_intents_without_commands_is_a_note(root):
    g = gate.Gate()
    (root / "domains/still").mkdir(parents=True)
    (root / "domains/still/still.llp").write_text("catalog still 1.\nlayers (d).\nstages (s).\nnamespace w produce (d).\n"
                                                  "pred a : w.\nstage s {\n}\n#init a 1\n")
    (root / "domains/still/domain.yaml").write_text("program: still.llp\n")
    pl = g.propose("intents", "still", root)
    assert pl["files"] == [] and "no intent.yaml" in pl["note"]
    assert g.confirm(pl["token"]) == {"written": [], "fingerprint": None, "note": pl["note"]}
    sk = g.propose("intents", "plant", root)                                   # the skeleton: no model
    assert any(f["path"] == "domains/plant/intent.yaml" for f in sk["files"])


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
    with pytest.raises(gate.Refused, match="kind: program"):
        g.propose("nope", "lamp", root)
    with pytest.raises(gate.Refused, match='domain "Lamp": lowercase letters, digits, _'):
        g.propose("program", "Lamp", root, text=LAMP)
    with pytest.raises(gate.Refused, match="an extension needs a name"):
        g.propose("extension", "plant", root, text=LAMP, name="Bad")


def _cli(state, *args, stdin=""):
    r = subprocess.run([sys.executable, "-m", "metis.gate", "--state", str(state), *args, "--json"],
                       input=stdin, capture_output=True, text=True, timeout=120)
    return r.returncode, json.loads(r.stdout)


def test_the_cli_keeps_the_memory_in_the_state(root, tmp_path):
    st = tmp_path / ".metis"
    rc, pl = _cli(st, "propose", "readme", "plant", "--root", str(root), stdin=json.dumps({"text": "A plant.", "title": "a plant"}))
    assert rc == 0 and pl["token"] and "write" not in pl
    assert _cli(st, "refuse", pl["token"]) == (0, {"refused": [f["path"] for f in pl["files"]]})
    rc, again = _cli(st, "propose", "readme", "plant", "--root", str(root), stdin=json.dumps({"text": "A plant, watered.", "title": "a plant"}))
    assert any("since_diff" in f for f in again["files"])                     # remembered across processes
    rc, out = _cli(st, "confirm", again["token"])
    assert rc == 0 and "domains/plant/README.md" in out["written"]
    assert _cli(st, "confirm", again["token"]) == (1, {"error": "not approved: no such plan (propose it again)"})
    rc, err = _cli(st, "propose", "readme", "lamp", "--root", str(root), stdin=json.dumps({"text": "x"}))
    assert rc == 1 and "its README follows its program" in err["error"]
    assert json.loads((st / "gate.json").read_text())["plans"] == {}
