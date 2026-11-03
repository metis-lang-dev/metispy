"""metis.catalog.workspace (loop-workspace s-5b4be5fa step 1; metis-loop docs/WORKSPACE.md §2): the layout table,
the surprise → take → ruling files, the journal, the bundle's stale test; on an empty DIR, on a composition made
here, and on metis-worlds/road read only (its .metis/road/, the world runner's run dir, untouched)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from metis.catalog import workspace as W

ROOT = Path(__file__).resolve().parents[2]


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_the_layout_is_one_table_of_relative_paths():
    lay = W.layout()
    assert lay["format"] == "metis-workspace/1" and lay["state"] == ".metis"
    assert lay["who"] == ["backend", "pi", "judge", "world"] and lay["locks"] == ["pi", "backend"]
    for name, row in lay["paths"].items():
        p = row["path"]
        assert not p.startswith("/") and ".." not in p.split("/"), name
        assert row["kind"] in ("dir", "json", "jsonl", "text") and (row["kind"] == "dir") == p.endswith("/"), name
    assert re.match(lay["id"], W.new_id()) and W.new_id() != W.new_id()
    ws_paths = {r["path"] for r in lay["paths"].values()}
    assert {"loop/journal.jsonl", "surprises/{id}.json", "surprises/{id}.taken", "rulings/{id}.md", "rulings/{id}.json",
            "packs/{id}.llp", "traces/{run}/", "cost.jsonl", "loop/{kind}.json"} <= ws_paths


def test_an_empty_dir_is_a_workspace_with_nothing_in_it(tmp_path):
    ws = W.workspace(tmp_path)
    assert ws.root == tmp_path.resolve() and ws.state == tmp_path.resolve() / ".metis"
    assert ws.bundle() is None and ws.traces() == [] and ws.surprises() == [] and ws.rulings() == []
    assert ws.events() == [] and ws.cost() == {"total": 0.0, "calls": 0, "by_model": {}, "by_who": {}}
    assert "no bundle" in ws.context()
    assert list(tmp_path.iterdir()) == [], "reads make nothing, not even .metis/"


def test_the_first_write_makes_the_state_ignored_in_a_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    ws = W.workspace(tmp_path)
    ws.bundle(), ws.surprises(), ws.events()
    assert not (tmp_path / ".gitignore").exists() and not ws.state.exists()
    ws.journal("backend", "hello")
    assert ws.state.is_dir() and "/.metis/" in (tmp_path / ".gitignore").read_text()


def test_a_surprise_is_dropped_taken_and_ruled_in_files(tmp_path):
    ws = W.workspace(tmp_path)
    sid = ws.drop_surprise({"member": "responder", "trace": "traces/s1", "k": 41, "window": [30, 41],
                            "atoms": ["brake_lon"], "words": "it should have braked"})
    assert re.match(W.ID, sid)
    rec = json.loads((ws.state / "surprises" / (sid + ".json")).read_text())
    assert rec["status"] == "open" and rec["who"] == "backend" and rec["window"] == [30, 41]
    assert [s["id"] for s in ws.surprises("open")] == [sid] and ws.surprises("taken") == []
    ws.take_surprise(sid, pid=os.getpid())
    assert ws.surprises("taken")[0]["taken"]["pid"] == os.getpid() and ws.surprises("open") == []
    with pytest.raises(ValueError, match="taken by pid %d" % os.getpid()):
        ws.take_surprise(sid, pid=os.getpid())                        # a live taker holds it
    j = ws.rule(sid, "# ruling\n\naccepted: the case holds\n", "accepted", pack="% the pack\n",
                moved=[], lint=[], spent=0.0)
    r = json.loads(j.read_text())
    assert r["verdict"] == "accepted" and r["md"] == "rulings/%s.md" % sid and r["pack"] == "packs/%s.llp" % sid
    assert (ws.state / r["pack"]).read_text() == "% the pack\n"
    assert not (ws.state / "surprises" / (sid + ".taken")).exists(), "the take goes with the ruling"
    assert ws.surprise(sid)["status"] == "ruled" and ws.surprise(sid)["ruling"]["verdict"] == "accepted"
    assert json.loads((ws.state / "surprises" / (sid + ".json")).read_text())["status"] == "open", \
        "a status is a file's existence, never a field edited in place"
    with pytest.raises(ValueError, match="ruled already"):
        ws.take_surprise(sid)
    with pytest.raises(ValueError, match="ruled already"):
        ws.rule(sid, "again", "refused")
    assert [(e["who"], e["kind"]) for e in ws.events()] == [("backend", "surprise"), ("pi", "take"), ("pi", "ruling")]
    assert ws.events(since=2)[0]["kind"] == "ruling" and ws.events(since=3) == []
    assert all("body" not in e and "words" not in e for e in ws.events()), "the journal carries pointers"
    assert not list(ws.state.rglob(".*.tmp")), "every file written whole then renamed"


def test_a_dead_takers_take_is_taken_over(tmp_path):
    ws = W.workspace(tmp_path)
    sid = ws.drop_surprise({"member": "m", "reading": "x(1) expected"})
    gone = dead_pid()
    ws.take_surprise(sid, pid=gone)
    ws.take_surprise(sid, pid=os.getpid())
    assert ws.surprise(sid)["taken"]["pid"] == os.getpid()
    assert ws.events()[-1] == {**ws.events()[-1], "kind": "take", "stale_pid": gone}


@pytest.mark.parametrize("rec, why", [
    ({"reading": "r"}, "names its member"),
    ({"member": "m"}, "a reading or atoms"),
    ({"member": "m", "atoms": ["a"], "window": [5, 2]}, "k_start <= k_end"),
    ({"member": "m", "atoms": ["a"], "trace": "/etc"}, "traces/<run>"),
    ({"member": "m", "atoms": ["a"], "trace": "traces/../x"}, "not a run name"),
    ({"member": "m", "atoms": ["a"], "k": "41"}, "a tick"),
    ({"member": "m", "atoms": ["a"], "who": "me"}, "who"),
])
def test_a_surprise_is_refused_in_its_own_words(tmp_path, rec, why):
    ws = W.workspace(tmp_path)
    with pytest.raises(ValueError, match=re.escape(why)):
        ws.drop_surprise(rec)
    assert ws.surprises() == [] and ws.events() == []


def test_a_trace_under_the_workspace_reads_as_metis_trace_writes_it(tmp_path):
    from metis import trace as T
    ws = W.workspace(tmp_path)
    s = T.Segments(ws.state / "traces" / "s1", every=2)
    for k in range(5):
        if s.due(k):
            s.open_segment(k, {"members": {}}, {"state": {}})
        s.line({"k": k})
    s.close()
    assert ws.traces() == [{"run": "s1", "path": "traces/s1", "segments": 3}]
    t = ws.trace("s1")
    assert (t["k0"], t["k1"], t["checkpoints"]) == (0, 4, [0, 2, 4])
    with pytest.raises(ValueError):
        ws.trace("../s1")


def composition(root):
    d = root / "domains"
    for m in ("boss", "a"):
        (d / m).mkdir(parents=True)
        (d / m / "domain.yaml").write_text("program: %s.llp\nline: the %s member\n" % (m, m))
        (d / m / ("%s.llp" % m)).write_text("% " + m + "\n")
    (d / "composition.yaml").write_text("# composition — two members, for the test\nname: duo\n"
                                        "specialists: [a]\ndirect: boss\n")
    (d / "image.golden").write_text("abc123\n")
    return d


def test_the_bundle_is_stale_until_built_and_when_a_member_moves(tmp_path):
    d = composition(tmp_path)
    ws = W.workspace(tmp_path)
    b = ws.bundle()
    assert (b["name"], b["composition"], b["members"], b["golden"]) == ("duo", "domains/composition.yaml", ["boss", "a"], "abc123")
    assert b["stale"] and b["image"] is None and "no build yet" in b["why"]
    (d / "build").mkdir()
    rec = d / "build" / "duo.build"
    rec.write_text("image duo.image.v-abc123\nmember boss x y\nmember a x y\n")
    t = rec.stat().st_mtime
    for f in d.rglob("*"):
        if f != rec and f.is_file():
            os.utime(f, (t - 10, t - 10))
    b = ws.bundle()
    assert (b["image"], b["build"], b["stale"]) == ("abc123", "domains/build/duo.build", False)
    os.utime(d / "a" / "a.llp", (t + 10, t + 10))
    b = ws.bundle()
    assert b["stale"] and b["why"] == "domains/a/a.llp is newer than the build"
    ctx = ws.context(patterns=str(tmp_path / "patterns"))
    assert "composition — two members" in ctx and "- a: the a member" in ctx and "STALE" in ctx
    assert str(tmp_path / "patterns") in ctx


def test_one_composition_under_domains_is_the_bundle_and_a_name_picks_one(tmp_path):
    d = composition(tmp_path)
    (d / "pkg").mkdir()
    (d / "composition.yaml").rename(d / "pkg" / "composition.yaml")
    ws = W.workspace(tmp_path)
    assert ws.bundle()["composition"] == "domains/pkg/composition.yaml" and ws.bundle()["name"] == "duo"
    assert ws.bundle("pkg")["name"] == "duo"
    with pytest.raises(ValueError, match="not a composition"):
        ws.bundle("a")


def test_cost_is_summed_by_model_and_by_who(tmp_path):
    ws = W.workspace(tmp_path)
    ws.made()
    (ws.state / "cost.jsonl").write_text('{"who":"pi","model":"m1","cost":0.01}\n{"who":"judge","model":"m1",'
                                         '"cost":0.02}\n{"who":"pi","model":"m2","cost":0.5}\nnot json\n')
    c = ws.cost()
    assert (c["total"], c["calls"]) == (0.53, 3)
    assert c["by_model"] == pytest.approx({"m1": 0.03, "m2": 0.5}) and c["by_who"] == pytest.approx({"pi": 0.51, "judge": 0.02})


def cli(root, *args, input=None):
    p = subprocess.run([sys.executable, "-m", "metis.catalog.workspace", "--root", str(root), *args, "--json"],
                       capture_output=True, text=True, input=input, cwd=ROOT)
    return p.returncode, json.loads(p.stdout)


def test_the_cli_answers_what_the_module_does(tmp_path):
    assert cli(tmp_path, "layout") == (0, W.layout())
    rc, out = cli(tmp_path, "drop", "-", input=json.dumps({"member": "m", "atoms": ["a"]}))
    assert rc == 0 and W.workspace(tmp_path).surprises()[0]["id"] == out["id"]
    assert cli(tmp_path, "surprises", "--status", "open")[1][0]["id"] == out["id"]
    assert cli(tmp_path, "events")[1] == W.workspace(tmp_path).events()
    rc, err = cli(tmp_path, "drop", "-", input="{}")
    assert rc == 1 and "member" in err["error"]
    rc, err = cli(tmp_path, "surprise", "nope")
    assert rc == 1 and "not a workspace id" in err["error"]


def worlds():
    w = os.environ.get("METIS_WORLDS")
    for p in ([Path(w)] if w else [ROOT.parent / "metis-worlds"]):
        if (p / "road" / "domains" / "composition.yaml").is_file():
            return p
    return None


def snapshot(d):
    return sorted((str(f.relative_to(d)), f.stat().st_size, f.stat().st_mtime_ns) for f in d.rglob("*")) if d.is_dir() else None


def test_metis_worlds_road_is_a_workspace_read_and_its_run_dir_untouched():
    w = worlds()
    if w is None:
        pytest.skip("no metis-worlds with road/ ($METIS_WORLDS, else beside this checkout)")
    road = w / "road"
    before_state = sorted(p.name for p in (road / ".metis").iterdir()) if (road / ".metis").is_dir() else None
    before_run = snapshot(road / ".metis" / "road")
    ws = W.workspace(road)
    b = ws.bundle()
    assert b["name"] == "road" and b["members"] == ["road_director", "responder", "signals", "planner"]
    assert b["composition"] == "domains/composition.yaml"
    if b["build"]:
        assert re.match(r"^[0-9a-f]{12}$", b["image"])
    assert "the bundle road" in ws.context() and "- responder:" in ws.context()
    ws.traces(), ws.surprises(), ws.rulings(), ws.events(), ws.cost()
    assert snapshot(road / ".metis" / "road") == before_run, "the world runner's run dir untouched"
    if before_state is not None:
        assert sorted(p.name for p in (road / ".metis").iterdir()) == before_state, "reads make nothing"
    assert not (road / ".gitignore").exists() or (road / ".git").exists(), "no .gitignore where road is not a checkout"
