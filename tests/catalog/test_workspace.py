"""metis.catalog.workspace (loop-workspace s-5b4be5fa step 1; metis-loop docs/WORKSPACE.md §2): the layout table,
the surprise → take → ruling files, the journal, the bundle's stale test; on an empty DIR, on a composition made
here, and on metis-worlds/road read only (its .metis/road/, the world runner's run dir, untouched)."""
import json
import os
import re
import shutil
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


def test_the_layout_is_one_table_of_relative_paths(tmp_path):
    lay = W.layout()
    assert lay["format"] == "metis-workspace/1" and lay["state"] == ".metis"
    assert lay["who"] == ["backend", "pi", "judge", "world"] and lay["locks"] == ["pi", "backend"]
    for name, row in lay["paths"].items():
        p = row["path"]
        assert not p.startswith("/") and ".." not in p.split("/"), name
        assert row["kind"] in ("dir", "json", "jsonl", "text") and (row["kind"] == "dir") == p.endswith("/"), name
    assert re.match(lay["id"], W.new_id()) and W.new_id() != W.new_id()
    ws = W.workspace(tmp_path)
    assert [n for n in lay["face"] if not hasattr(ws, n) and not callable(getattr(W, n, None))] == [], "py's names"
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


def test_three_dropped_in_one_second_read_back_in_drop_order(tmp_path, monkeypatch):
    ws = W.workspace(tmp_path)
    hexes = iter(["c00000", "b00000", "a00000"])          # each draw lower than the last: the bump orders them
    real_hex = W.secrets.token_hex
    monkeypatch.setattr(W.secrets, "token_hex", lambda n: next(hexes) if n == 3 else real_hex(n))   # the ids' draws only
    monkeypatch.setattr(W, "_LAST_ID", [""])
    sec = 1793700000.0
    real = W.time.localtime
    monkeypatch.setattr(W.time, "localtime", lambda now=None: real(sec))
    ids = [ws.drop_surprise({"member": "m", "atoms": ["a%d" % i]}) for i in range(3)]
    assert len({i[:15] for i in ids}) == 1, "one second"
    assert [i[16:] for i in ids] == ["c00000", "c00001", "c00002"], "monotonic within the process, the hex bumped"
    assert [s["id"] for s in ws.surprises()] == ids
    ts = [s["ts"] for s in ws.surprises()]
    assert all("." in x for x in ts) and ts == sorted(ts), "sub-second ts"


def test_across_processes_the_ts_orders_one_second(tmp_path):
    """Each CLI call is its own process (the ts face's way): the ids' hex is random across them, the ts is not."""
    got = [cli(tmp_path, "drop", "-", input=json.dumps({"member": "m", "atoms": ["a%d" % i]}))[1]["id"] for i in range(3)]
    ws = W.workspace(tmp_path)
    by_ts = sorted(ws.surprises(), key=lambda s: s["ts"])
    assert [s["id"] for s in ws.surprises()] == [s["id"] for s in by_ts] == got
    for s in ws.surprises():
        (ws.state / "surprises" / (s["id"] + ".json")).write_text(json.dumps({**s, "ts": "2026-11-03T12:00:00+02:00"}))
    assert [s["id"] for s in ws.surprises()] == sorted(got), "an old second-only ts: the id breaks the tie"


def test_a_dead_takers_take_is_taken_over(tmp_path):
    ws = W.workspace(tmp_path)
    sid = ws.drop_surprise({"member": "m", "reading": "x(1) expected"})
    gone = dead_pid()
    ws.take_surprise(sid, pid=gone)
    s = ws.surprise(sid)
    assert s["status"] == "open" and s["stale_take"]["pid"] == gone and "taken" not in s, "a dead pid's take is stale"
    assert [x["id"] for x in ws.surprises("open")] == [sid] and ws.surprises("taken") == []
    ws.take_surprise(sid, pid=os.getpid())
    assert ws.surprise(sid)["status"] == "taken" and ws.surprise(sid)["taken"]["pid"] == os.getpid()
    assert "stale_take" not in ws.surprise(sid)
    assert [(e["kind"], e.get("dead_pid")) for e in ws.events()[-2:]] == [("take", None), ("retake", gone)]


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


def test_a_copied_workspace_reads_its_own_members_not_the_locks_original(tmp_path):
    """qwen-loop F5: image.lock names each member's dir absolute, as built; in a copy (cp -a) that is the ORIGINAL
    tree, and an edit in the copy read not stale. The lock's dir counts only inside the workspace."""
    import json
    orig, copy = tmp_path / "orig", tmp_path / "copy"
    d = composition(orig)
    (d / "build").mkdir()
    rec = d / "build" / "duo.build"
    rec.write_text("image duo.image.v-abc123\nmember boss x y\nmember a x y\n")
    (d / "build" / "duo.image.v-abc123").mkdir()
    (d / "build" / "duo.image.v-abc123" / "image.lock").write_text(json.dumps(
        {"members": {m: {"dir": {"path": str(d / m)}} for m in ("boss", "a")}}))
    t = rec.stat().st_mtime
    for f in orig.rglob("*"):
        if f != rec and f.is_file():
            os.utime(f, (t - 10, t - 10))
    shutil.copytree(orig, copy, copy_function=shutil.copy2)             # cp -a: the mtimes kept
    assert W.workspace(copy).bundle()["stale"] is False
    os.utime(copy / "domains" / "a" / "a.llp", (t + 10, t + 10))         # the copy's own member, edited
    b = W.workspace(copy).bundle()
    assert b["stale"] and b["why"] == "domains/a/a.llp is newer than the build"
    assert W.workspace(orig).bundle()["stale"] is False                  # the original is untouched


def test_one_composition_under_domains_is_the_bundle_and_a_name_picks_one(tmp_path):
    d = composition(tmp_path)
    (d / "pkg").mkdir()
    (d / "composition.yaml").rename(d / "pkg" / "composition.yaml")
    ws = W.workspace(tmp_path)
    assert ws.bundle()["composition"] == "domains/pkg/composition.yaml" and ws.bundle()["name"] == "duo"
    assert ws.bundle("pkg")["name"] == "duo"
    with pytest.raises(ValueError, match="not a composition"):
        ws.bundle("a")


def test_the_bundles_service_config_is_named_or_beside_it(tmp_path):
    d = composition(tmp_path)
    ws = W.workspace(tmp_path)
    assert ws.bundle()["config"] is None
    (d / "service.yaml").write_text("trace: {}\n")
    assert ws.bundle()["config"] == str((d / "service.yaml").resolve())
    (d / "conf").mkdir()
    (d / "composition.yaml").write_text((d / "composition.yaml").read_text() + "service: conf/run.yaml\n")
    assert ws.bundle()["config"] == str((d / "conf" / "run.yaml").resolve()), "the named one wins, absolute"
    assert not (ws.state).exists(), "config is the bundle's, not .metis/'s"


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


# ── the locks (WORKSPACE.md §2.3) ──
def test_one_pi_loop_a_second_is_refused_with_the_holders_line(tmp_path):
    ws = W.workspace(tmp_path)
    assert ws.loops() == {"pi": None, "backend": None}
    ws.lock("pi", pid=os.getpid(), mode="tui", model="m", cwd=str(tmp_path))
    with pytest.raises(ValueError, match=r"the pi loop holds this workspace: pid %d since .*: mode tui, model m" % os.getpid()):
        ws.lock("pi", pid=os.getppid(), mode="rpc")
    ws.lock("backend", pid=os.getppid(), port=8399)                   # the other kind: its own lock
    lp = ws.loops()
    assert (lp["pi"]["pid"], lp["pi"]["mode"], lp["pi"]["alive"]) == (os.getpid(), "tui", True)
    assert (lp["backend"]["port"], lp["backend"]["alive"]) == (8399, True)
    with pytest.raises(ValueError, match="pid %d's, not" % os.getpid()):
        ws.unlock("pi", pid=os.getppid())
    assert ws.unlock("pi", pid=os.getpid()) and ws.loops()["pi"] is None and not ws.unlock("pi")
    assert [(e["who"], e["kind"]) for e in ws.events()] == [("pi", "lock"), ("backend", "lock"), ("pi", "unlock")]


def test_a_dead_pids_lock_is_stale_and_taken_over(tmp_path):
    ws = W.workspace(tmp_path)
    gone = dead_pid()
    ws.lock("pi", pid=gone, mode="rpc")
    assert ws.loops()["pi"]["alive"] is False
    ws.lock("pi", pid=os.getpid(), mode="tui")
    assert ws.loops()["pi"]["pid"] == os.getpid() and ws.events()[-1]["stale_pid"] == gone
    with pytest.raises(ValueError, match="one of pi, backend"):
        ws.lock("judge")


def test_the_cli_lock_is_the_callers(tmp_path):
    rc, out = cli(tmp_path, "lock", "pi", "--fields", json.dumps({"mode": "rpc"}))
    assert rc == 0 and out["pid"] == os.getpid() and out["mode"] == "rpc"      # the CLI's parent: this test
    rc, err = cli(tmp_path, "lock", "pi")
    assert rc == 0, "the same holder again: its own lock"
    rc, err = cli(tmp_path, "lock", "pi", "--pid", str(os.getppid()))
    assert rc == 1 and "the pi loop holds this workspace: pid %d" % os.getpid() in err["error"]
    assert cli(tmp_path, "loops")[1]["pi"]["pid"] == os.getpid()
    assert cli(tmp_path, "unlock", "pi") == (0, {"released": True})


# ── the trace verbs as declared tools (WORKSPACE.md §3.1) ──
def test_the_declaration_table_is_what_the_verbs_argparse_declares():
    assert W.TOOLS_FILE.read_text() == W.tools_json(), "python tools/gen_trace_tools.py"
    from metis import lint, trace
    names = [t["name"] for t in W.tools()["tools"]]
    sub = next(a for a in trace.parser()._actions if a.__class__.__name__ == "_SubParsersAction")
    assert names == ["trace_" + v for v in sub.choices] + ["lint"]


def sample(p, k):
    return {"STRING": "x_" + k, "INTEGER": 3, "BOOLEAN": True, "ARRAY": ["a", "b"]}[p["type"]]


@pytest.mark.parametrize("tool", [t["name"] for t in json.loads(W.TOOLS_FILE.read_text())["tools"]])
def test_a_declared_tools_command_line_parses_back_to_its_arguments(tool):
    from metis import lint, trace
    t = next(x for x in W.tools()["tools"] if x["name"] == tool)
    args = {k: sample(p, k) for k, p in t["parameters"]["properties"].items()}
    argv = W.tool_argv(tool, args)
    assert argv[:3] == [sys.executable, "-m", t["module"]]
    ns = vars((trace if t["module"] == "metis.trace" else lint).parser().parse_args(argv[3:]))
    assert {k: ns[k] for k in args} == args
    req = {k: args[k] for k in t["parameters"]["required"]}
    vars((trace if t["module"] == "metis.trace" else lint).parser().parse_args(W.tool_argv(tool, req)[3:]))
    if req:
        with pytest.raises(ValueError, match="required"):
            W.tool_argv(tool, {})
    with pytest.raises(ValueError, match="no parameter nope"):
        W.tool_argv(tool, {**req, "nope": 1})


def test_the_cli_serves_the_table_and_the_argv(tmp_path):
    assert cli(tmp_path, "tools")[1] == W.tools()
    rc, out = cli(tmp_path, "argv", "trace_count", "--args", json.dumps({"trace": "t", "atom": "a", "by": ["b"]}))
    assert rc == 0 and out["argv"][3:] == ["count", "t", "a", "--by", "b"]
    rc, err = cli(tmp_path, "argv", "trace_no_such_verb")       # (trace_sort was the example until pascal's sort landed)
    assert rc == 1 and "not a declared trace tool" in err["error"]


def ring_trace(ws, run, ticks, every):
    from metis import trace as T
    s = T.Segments(ws.state / "traces" / run, every=every)
    for k in range(ticks):
        if s.due(k):
            s.open_segment(k, {"members": {}}, {"state": {"k": k}})
        s.line({"k": k})
    s.close()
    return ws.state / "traces" / run


def test_a_window_is_kept_past_the_ring_with_its_marks_and_config(tmp_path):
    """loop-workspace step 5 (WORKSPACE.md §6.1, item 4): keep_window copies the segments that hold [k0, k1], each with
    its checkpoint, and the run's marks and config, to traces/<run>-<name>/, which reads as a trace; the original may then
    lose them to its ring. A window the trace no longer holds is refused with the range it does."""
    from metis import trace as T
    ws = W.workspace(tmp_path)
    d = ring_trace(ws, "r1", 10, 3)                      # segments at 0, 3, 6, 9
    (d / "marks.jsonl").write_text('{"mark": 1, "edge": "start", "at_tick": 3, "at_k": 4}\n'
                                   '{"mark": 1, "edge": "stop", "at_tick": 6, "at_k": 7}\n'
                                   '{"mark": 2, "edge": "start", "at_tick": 8, "at_k": 9}\nnot json\n'
                                   '{"mark": 3, "edge": "start", "at_tick": 1}\n')
    (d / "config.yaml").write_text("trace: {}\n")
    assert ws.marks("r1") == [
        {"mark": 1, "k0": 4, "k1": 7, "start": {"mark": 1, "edge": "start", "at_tick": 3, "at_k": 4},
         "stop": {"mark": 1, "edge": "stop", "at_tick": 6, "at_k": 7}},
        {"mark": 2, "k0": 9, "k1": None, "start": {"mark": 2, "edge": "start", "at_tick": 8, "at_k": 9}}]
    assert ws.keep_window("r1", 4, 7, "mark-1") == "traces/r1-mark-1"
    kept = ws.state / "traces" / "r1-mark-1"
    assert sorted(p.name for p in kept.glob("seg-*.jsonl")) == ["seg-3.jsonl", "seg-6.jsonl"]
    assert sorted(p.name for p in (kept / "checkpoints").iterdir()) == ["k3.json", "k6.json"]
    assert [m["mark"] for m in ws.marks("r1-mark-1")] == [1] and (kept / "config.yaml").is_file()   # the marks inside only
    t = ws.trace("r1-mark-1")
    assert (t["k0"], t["k1"], t["checkpoints"]) == (3, 8, [3, 6])
    assert [r["kind"] for r in ws.events()][-1] == "keep"
    with pytest.raises(ValueError, match="there already"):
        ws.keep_window("r1", 4, 7, "mark-1")
    for seg in ("seg-0.jsonl", "seg-3.jsonl"):           # the writer's ring drops the oldest
        (d / seg).unlink()
    with pytest.raises(ValueError, match=r"not in traces/r1 any more: it holds 6\.\.9"):
        ws.keep_window("r1", 4, 7, "again")
    with pytest.raises(ValueError):
        ws.keep_window("r1", 7, 4, "back")
    with pytest.raises(ValueError):
        ws.keep_window("r1", 6, 7, "../x")
    assert len(T.read(kept)) == 2


def test_a_window_the_ring_drops_during_the_copy_leaves_nothing(tmp_path, monkeypatch):
    """metis's review of step 5 (1)-(3): a segment the live writer's ring drops while it is copied is the window's refusal, in
    words, and the temp dir goes; traces() never lists a copy in progress; a rename onto a copy made meanwhile is 'there already'."""
    import shutil
    ws = W.workspace(tmp_path)
    d = ring_trace(ws, "r1", 10, 3)
    real = shutil.copy2

    def ring(a, b, *k, **kw):
        if str(a).endswith("seg-6.jsonl"):
            (d / "seg-6.jsonl").unlink()
        return real(a, b, *k, **kw)
    monkeypatch.setattr(shutil, "copy2", ring)
    with pytest.raises(ValueError, match=r"the ring dropped seg-6.jsonl while it was copied"):
        ws.keep_window("r1", 4, 7, "m")
    monkeypatch.setattr(shutil, "copy2", real)
    assert sorted(p.name for p in (ws.state / "traces").iterdir()) == ["r1"]
    (ws.state / "traces" / ".r1-x.tmp1").mkdir()
    assert [t["run"] for t in ws.traces()] == ["r1"]
    real_rename = os.rename

    def raced(a, b):
        os.makedirs(os.path.join(b, "seg-0.jsonl"))       # another keep made it, not empty
        return real_rename(a, b)
    monkeypatch.setattr(os, "rename", raced)
    with pytest.raises(ValueError, match="there already"):
        ws.keep_window("r1", 0, 1, "y")
    assert not [p for p in (ws.state / "traces").iterdir() if p.name.startswith(".r1-y")]


def test_the_spend_line_is_one_count_from_the_journal(tmp_path):
    """loop-ui I8: the calls are the journal's `usage` rows (pi's own, one per model call, TUI or rpc), the USD
    cost.jsonl's; on a local model the line says so instead of '0.0000 USD … (0 calls …)' after eleven calls."""
    ws = W.workspace(tmp_path)
    assert ws.spend()["line"] == "0.0000 USD spent by pi's estimate (0 calls; the key's bill runs higher)"
    for _ in range(2):
        ws.journal("pi", "usage", model="ollama/qwen3:4b", cost=0, input=5000, output=300)
    ws.journal("backend", "usage", model="x", cost=1)                  # not pi's: not a call of the loop
    sp = ws.spend()
    assert (sp["calls"], sp["local"], sp["line"]) == (2, True, "0 USD, local (2 calls)")
    ws.journal("pi", "usage", model="openrouter/deepseek/deepseek-v4-pro", cost=0.01)
    ws.path("cost").write_text(json.dumps({"who": "pi", "model": "deepseek/deepseek-v4-pro", "cost": 0.01}) + "\n")
    sp = ws.spend()
    assert not sp["local"] and sp["line"] == "0.0100 USD spent by pi's estimate (3 calls; the key's bill runs higher)"


def test_an_accepted_ruling_applies_as_the_judges_merge(tmp_path):
    """loop-ui 2.3 (the user's /apply): the plan of an accepted ruling is its pack merged onto its unit by the judge's
    own code (metis.trace.apply_pack), as an edit of the domain's file; refused when not accepted, and once applied."""
    d = tmp_path / "domains" / "lamp"
    d.mkdir(parents=True)
    (d / "domain.yaml").write_text("program: lamp.llp\n")
    (tmp_path / "domains" / "lamp_rules.llp").write_text("catalog lamp_rules 1.\nstage main {\n  go [dyn] : $on * idle -o busy.\n}\n")
    (d / "lamp.llp").write_text('catalog lamp 1.\ninclude "../lamp_rules.llp".\n')
    ws = W.workspace(tmp_path)
    sid = ws.drop_surprise({"member": "lamp", "reading": "it went while off", "unit": "domains/lamp_rules.llp"})
    with pytest.raises(ValueError, match="not ruled: nothing to apply"):
        ws.apply_plan(sid)
    pack = "drop go.\nstage main {\n  go [dyn] : $on * $ready * idle -o busy.\n}\n"
    ws.rule(sid, "# Ruling: accepted\n", "accepted", pack=pack, member="lamp")
    pl = ws.apply_plan(sid)
    assert (pl["domain"], pl["unit"], pl["path"], pl["added"], pl["dropped"]) == (
        "lamp", "domains/lamp_rules.llp", "../lamp_rules.llp", ["go"], ["go"])
    from metis import trace as T
    assert pl["new"] == T.apply_pack(pl["old"], pack)[0] and "$ready" in pl["new"] and "$on * idle" not in pl["new"]
    (tmp_path / "domains" / "lamp_rules.llp").write_text(pl["new"])                 # written: the same plan again
    with pytest.raises(ValueError, match="applied already"):
        ws.apply_plan(sid)
    other = ws.drop_surprise({"member": "lamp", "reading": "x"})
    ws.rule(other, "# Ruling: refused\n", "refused", pack=pack, member="lamp")
    with pytest.raises(ValueError, match="ruled REFUSED: only an accepted ruling is applied"):
        ws.apply_plan(other)
