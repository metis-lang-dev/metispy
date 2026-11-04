"""The terminal REPL in a workspace (two-terminals s-065d46d5; metis-loop docs/WORKSPACE.md §10): what `metis ARG`
serves from where the person stands (a member's unit → its program, a package name, the package here), the cwd
joining the catalog's places, the workspace found up the tree, and the session traced into .metis/traces/ with
drop / mark / surprises writing the pi loop's queue."""
import io
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from metis import catalog
from metis import trace as T
from metis.catalog import workspace as W
from metis.lang.repl import Repl

GOSSIP = Path(__file__).resolve().parents[1] / "loop" / "fixtures" / "gossip3.llp"
UNIT = "% {name}'s rules\ncatalog {name}_domain 1.\n"


def member(hier, name, extra=()):
    """domains/hier/NAME/NAME.llp including ../NAME_domain.llp (and EXTRA units), with its manifest."""
    d = hier / name
    d.mkdir(parents=True)
    (d / "domain.yaml").write_text(f"program: {name}.llp\ndisplay: {name}\nline: {name}\n")
    inc = "".join(f'include "../{u}".\n' for u in (f"{name}_domain.llp", *extra))
    (d / f"{name}.llp").write_text(f"catalog {name} 1.\n{inc}")
    (hier / f"{name}_domain.llp").write_text(UNIT.format(name=name))
    return d


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    hier = ws / "domains" / "hier"
    member(hier, "grasp", ("cap.llp",))
    member(hier, "nav", ("cap.llp",))
    (hier / "cap.llp").write_text("% the capacity\n")
    (hier / "composition.yaml").write_text("name: alf\nspecialists: [grasp, nav]\n")
    monkeypatch.delenv("METIS_CATALOG", raising=False)
    monkeypatch.chdir(ws)
    return ws


def test_serve_target_from_where_the_person_stands(bundle, monkeypatch):
    hier = bundle / "domains" / "hier"
    p, how = catalog.serve_target("domains/hier/grasp_domain.llp")      # a member's unit: its member's program
    assert p == hier / "grasp" / "grasp.llp" and "is a unit domains/hier/grasp/grasp.llp includes" in how
    with pytest.raises(ValueError, match="several programs include: name one of domains/hier/grasp/grasp.llp, "
                                         "domains/hier/nav/nav.llp"):
        catalog.serve_target("domains/hier/cap.llp")
    assert catalog.serve_target("domains/hier/nav/nav.llp") == (hier / "nav" / "nav.llp", None)
    p, how = catalog.serve_target("nav")                                 # a package name, the workspace's
    assert p == hier / "nav" / "nav.llp" and "the package nav" in how
    monkeypatch.chdir(hier / "grasp")
    p, how = catalog.serve_target(None)                                  # `metis` alone in a package: it
    assert p == hier / "grasp" / "grasp.llp" and "the package here" in how
    assert catalog.serve_target("../nav")[0] == hier / "nav" / "nav.llp"  # a package directory
    p, _ = catalog.serve_target("nav")                                   # from a member's dir: the workspace's still
    assert p == hier / "nav" / "nav.llp"
    monkeypatch.chdir(bundle.parent)
    assert catalog.serve_target(None) == (None, None)                    # nothing here: the scratch world
    assert catalog.serve_target("no-such") == (Path("no-such"), None)    # the compiler says what is wrong


def test_the_cwd_joins_the_places_when_the_root_is_elsewhere(tmp_path, monkeypatch):
    root, here = tmp_path / "root", tmp_path / "here"
    root.mkdir(), here.mkdir()
    (here / "unit.llp").write_text("% here\n")
    monkeypatch.setenv("METIS_CATALOG", str(root))
    monkeypatch.chdir(here)
    assert [w for _, w in catalog.places()] == ["the catalog root", "the current directory", "the standard units",
                                                "METIS_HOME"]
    assert catalog.find("unit.llp") == here / "unit.llp"
    assert [w for _, w in catalog.places(root)][1] == "the standard units"   # an explicit root: its own search
    monkeypatch.chdir(root)
    assert "the current directory" not in [w for _, w in catalog.places()]   # the cwd is the root: once


def test_workspace_find(bundle, tmp_path, monkeypatch):
    monkeypatch.chdir(bundle / "domains" / "hier" / "grasp")
    assert W.find().root == bundle                                   # a bundle up the tree
    (tmp_path / "other" / ".metis").mkdir(parents=True)
    (tmp_path / "other" / "deep").mkdir()
    assert W.find(tmp_path / "other" / "deep").root == tmp_path / "other"   # .metis/ up the tree
    assert W.find(tmp_path) is None
    assert W.find(Path(tempfile.gettempdir())) is None and W.find(Path.home()) is None   # never one
    monkeypatch.setenv("METIS_CATALOG", str(tmp_path / "other"))
    assert W.find(tmp_path).root == tmp_path / "other"               # the console's rule first


def _say(r, cmd):
    r.stdout.truncate(0), r.stdout.seek(0)
    r.onecmd(cmd)
    return r.stdout.getvalue()


def test_the_terminal_repl_in_a_workspace_traces_and_drops(tmp_path, monkeypatch):
    ws_dir = tmp_path / "ws"
    (ws_dir / ".metis").mkdir(parents=True)
    prog = ws_dir / "gossip3.llp"
    shutil.copy(GOSSIP, prog)
    monkeypatch.chdir(ws_dir)
    ws = W.Workspace(ws_dir)
    r = Repl(prog, seed=1, terminal=True, workspace=ws, stdout=io.StringIO())
    assert "workspace .: no bundle · 0 open surprise(s) · trace .metis/traces/repl-" in r.intro
    assert "drop: nothing has happened" in _say(r, "drop too early")
    _say(r, "run 3")
    _say(r, "events")                                                # a read: no tick
    out = _say(r, "drop alice heard nothing. She should have.")
    assert "dropped " in out and "(gossip3, k 1)" in out
    (rec,) = ws.surprises("open")
    tr = r._ws_trace
    assert rec["member"] == "gossip3" and rec["trace"] == "traces/" + tr.path.name and rec["k"] == 1
    assert rec["reading"] == "alice heard nothing." and rec["words"] == "alice heard nothing. She should have."
    _say(r, "run 2")
    assert "(gossip3, window [1, 2])" in _say(r, "mark 1..2 the window")
    assert "ticks 1..2" in _say(r, "mark 5 too far")
    assert "mark K WORDS" in _say(r, "mark nothing")
    assert _say(r, "status") == ("no bundle · pi: none · 0.0000 USD spent by pi's estimate (0 calls; the key's bill "
                                 "runs higher)\nsurprises: 2 open, 0 taken, 0 ruled\n")
    listed = _say(r, "surprises")
    assert listed.count("open") == 2 and "the window" in listed
    head = T.read(tr.path)[0].header
    assert head["live"] == T.SESSION_LIVE and head["members"]["gossip3"]["program"] == str(prog)
    _say(r, "quit")
    rows = [json.loads(x) for x in (tr.path / "seg-0.jsonl").read_text().splitlines()[1:]]
    assert [x["k"] for x in rows] == [1, 2] and rows[0]["members"]["gossip3"]["cmd"] == "run 3"


def test_a_reload_with_a_new_program_starts_a_new_segment(tmp_path, monkeypatch):
    ws_dir = tmp_path / "ws"
    (ws_dir / ".metis").mkdir(parents=True)
    prog = ws_dir / "gossip3.llp"
    shutil.copy(GOSSIP, prog)
    monkeypatch.chdir(ws_dir)
    r = Repl(prog, seed=1, terminal=True, workspace=W.Workspace(ws_dir), stdout=io.StringIO())
    _say(r, "run 1")
    prog.write_text(prog.read_text().replace("follows(carol,alice).", "follows(carol,alice).\nfollows(alice,carol)."))
    _say(r, "reload")
    segs = sorted(p.name for p in r._ws_trace.path.glob("seg-*.jsonl"))
    assert len(segs) == 2
    assert json.loads((r._ws_trace.path / segs[-1]).read_text().splitlines()[0])["cut"] == "reload"


def test_not_in_a_workspace_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = Repl(GOSSIP, seed=1, terminal=True, workspace=None, stdout=io.StringIO())
    assert "workspace: none (no .metis/ or bundle here or above)" in r.intro
    assert "not in a workspace" in _say(r, "drop x")
    assert "not in a workspace" in _say(r, "surprises")
    r2 = Repl(GOSSIP, seed=1, stdout=io.StringIO())                  # the wire's / console's shell: no line, no trace
    assert "workspace" not in r2.intro and r2.__dict__.get("_ws_trace") is None
