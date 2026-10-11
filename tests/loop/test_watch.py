"""runtime-wire step 1 (metis-agent docs/INTEGRATION.md): G-D, the
`watch` signal stream (event/quiescence/surprise pushed unpolled and in
order from a served catalog under a PortPlugin host), G-E, `ctx
--mirror` (one line per domain), and the kitchen host's three injected
surprises — transient, wrong-domain, novel — as kernel surprise
records."""
import io
import json

import pytest

from metis.kernel.record import recover
from metis.lang import load_llp
from metis.lang.repl import Repl, main
from metis.loop.watch import Watch, mirror, mirror_line
from metis.loop.wire import INTENT_VERBS, dispatch, serve_json

from metis.domains.kitchen.model import CATALOG

CAT = CATALOG.parent
SCENE = CAT / "kitchen_host.llp"
HOST = "metis.domains.kitchen.host:"


def _shell(factory="make", seed=1):
    return Repl(SCENE, seed=seed, plugin=HOST + factory, host=True,
                stdout=io.StringIO())


def _run(sh, beats=30):
    w = Watch(sh, beats=beats)
    sigs = []
    while not w.done:
        sigs += w.step()
    return w, sigs


def test_clean_kitchen_streams_events_and_quiescence_in_order():
    sh = _shell()
    w, sigs = _run(sh)
    assert [s["seq"] for s in sigs] == list(range(1, len(sigs) + 1))
    assert {s["signal"] for s in sigs} == {"event", "quiescence"}
    assert all(s["path"] == "kitchen" for s in sigs)
    names = [s["name"] for s in sigs if s.get("kind") == "clause"]
    assert names.count("serve") == 2           # both orders served
    orders = [s for s in sigs if s.get("kind") == "exo"]
    assert [o["ops"] for o in orders] == [["+pot_space 3"]] * 2
    # quiescence once per entry, never per idle beat
    qs = [s for s in sigs if s["signal"] == "quiescence"]
    # before order 1, after each soup, the script's end
    assert len(qs) == 4 and qs[-1]["episode_over"]
    assert not any(q["episode_over"] for q in qs[:-1])
    # quiet beats said nothing: 30 beats, 2 exo lines
    assert w.beat == 30 and len(orders) == 2
    assert sh.counts.get("served") == 2 and not sh.loop.records


@pytest.mark.parametrize("factory,diff,after", [
    ("make_transient", {"pot_onion": 1}, "load"),
    ("make_wrong_domain", {"pot_space": -1, "pot_onion": 1}, "load"),
    ("make_novel", {"held_soup": -1, "hand_free": 1}, "plate"),
])
def test_each_injected_surprise_is_one_unmodeled_record(factory, diff,
                                                        after):
    sh = _shell(factory)
    w, sigs = _run(sh)
    sur = [s for s in sigs if s["signal"] == "surprise"]
    assert len(sur) == 1
    s = sur[0]
    assert s["kind"] == "unmodeled" and s["likelihood"] == 0
    assert s["diff"] == diff and s["observe_only"]
    # the event that fired just before it is the one the env botched
    before = [x for x in sigs if x["seq"] < s["seq"]
              and x.get("kind") == "clause"]
    assert before[-1]["name"] == after
    # the record: a kernel surprise record, the loop's own (record N)
    rec = sh.loop.records[s["record"]]
    assert rec["format"] == "metis-surprise" and rec["kind"] == "unmodeled"
    assert rec["observed"] == s["observed"] and rec["steps"] == 1
    assert rec["meta"]["path"] == "kitchen"
    assert rec["meta"]["since"] == [after]
    assert "hash" in rec["env"] and "blob" in rec["env"]
    json.dumps(rec)
    # observe-only: nothing fired after the surprise but env beats
    later = [x for x in sigs if x["seq"] > s["seq"]]
    assert not any(x.get("kind") in ("clause", "link") for x in later)
    assert later[-1]["signal"] == "quiescence" and later[-1]["observe_only"]
    assert sh.loop.records == [rec]
    assert w.observe_only


def test_transient_reads_consistent_on_the_next_beat():
    sh = _shell("make_transient")
    _, sigs = _run(sh)
    k = next(i for i, s in enumerate(sigs) if s["signal"] == "surprise")
    nxt = sigs[k + 1]
    assert nxt["kind"] == "reading" and nxt["consistent"]
    assert nxt["beat"] == sigs[k]["beat"] + 1
    # it speaks once — repeated identical readings say nothing
    assert sum(1 for s in sigs if s.get("kind") == "reading") == 1


def _replay(rec, catalog, extra):
    prog = catalog.ground(rec["init"], init_stage=rec["meta"]["stage"])
    return recover(rec, prog, steps=rec["steps"] + extra)["explains"]


def test_the_sibling_card_explains_only_the_wrong_domain_record():
    """The step-3 explanation test, on the records step 1 logs: with
    one move allowed beside the cook's own, kitchen_shared (a second
    cook loads the pot) explains the wrong-domain reading and the
    served kitchen does not; nothing explains the novel spill or the
    transient misread."""
    shared = load_llp(CAT / "kitchen_shared.llp").catalog
    got = {}
    for f in ("make_transient", "make_wrong_domain", "make_novel"):
        sh = _shell(f)
        _run(sh)
        rec = sh.loop.records[0]
        got[f] = (_replay(rec, sh.doc.catalog, 0),
                  _replay(rec, sh.doc.catalog, 1),
                  _replay(rec, shared, 1))
    assert got == {"make_transient": (False, False, False),
                   "make_wrong_domain": (False, False, True),
                   "make_novel": (False, False, False)}


def test_mirror_is_one_line_per_domain():
    sh = _shell("make_novel")
    w = Watch(sh, beats=30)
    rows = mirror(sh, w)
    assert len(rows) == 1
    assert set(rows[0]["beliefs"]) == {"served", "cooking", "held_soup"}
    while not any(s["signal"] == "surprise" for s in w.step()):
        pass
    row, = mirror(sh, w)
    assert row["status"] == "observe-only"
    assert row["surprise"] == {"record": 0, "kind": "unmodeled",
                               "beat": w.beat}
    line = mirror_line(row)
    assert line.startswith("kitchen  stage task  observe-only  "
                           "surprise #0 unmodeled @beat ")
    assert "P[served]=" in line and "\n" not in line
    # the wire face
    res = dispatch(sh, "ctx", "--mirror")
    assert res["ok"] and res["lines"] == [line] and res["text"] == line


def test_mirror_beliefs_are_the_query_marginals_from_now():
    sh = _shell()
    w = Watch(sh, beats=30)
    while "cooking" not in sh.counts:
        w.step()
    row, = mirror(sh, w)
    assert row["status"] == "running" and row["surprise"] is None
    # the soup is on: served is near-certain within 12 steps (the
    # 0.05 off-weights keep a stray onion fetch priced)
    assert 0.9 < row["beliefs"]["served"] < 1.0
    assert row["beliefs"]["cooking"] < row["beliefs"]["served"]


def test_serve_json_pushes_signals_unpolled():
    sh = _shell("make_wrong_domain")
    out = io.StringIO()
    serve_json(sh, io.StringIO(
        '{"op": "watch", "arg": "every=0 beats=20"}\n'), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0]["ok"] and lines[0]["watch"] == "on"
    sigs = lines[1:]                       # no request after the first
    assert [s["seq"] for s in sigs] == list(range(1, len(sigs) + 1))
    assert sum(1 for s in sigs if s["signal"] == "surprise") == 1
    assert sigs[-1]["signal"] == "quiescence" and sigs[-1]["episode_over"]


def test_watch_op_usage_off_and_status():
    sh = _shell()
    assert dispatch(sh, "watch", "status") == {"ok": True,
                                                "watch": "none"}
    assert not dispatch(sh, "watch", "beats=x")["ok"]
    assert not dispatch(sh, "watch", "speed=2")["ok"]
    assert dispatch(sh, "watch", "every=0 beats=5")["watch"] == "on"
    assert not dispatch(sh, "watch", "")["ok"]          # one at a time
    off = dispatch(sh, "watch", "off")
    assert off["watch"] == "off" and off["seq"] == 0
    assert dispatch(sh, "watch", "every=0")["ok"]       # a fresh one


def test_host_is_argv_only_and_d4_holds(tmp_path, monkeypatch, capsys):
    # the wire can neither start a page-intent watch nor name a plugin
    assert "watch" not in INTENT_VERBS and "plugin" not in INTENT_VERBS
    # --host belongs to the --json runtime
    assert main(["--host", HOST + "make", str(SCENE)]) == 2
    # a non-host session still refuses the plugin (the console path)
    with pytest.raises(Exception):
        Repl(SCENE, plugin=HOST + "make", stdout=io.StringIO())
    # the runtime process: --json --host serves the kitchen
    monkeypatch.setattr("sys.stdin", io.StringIO(
        '{"op": "watch", "arg": "every=0 beats=3"}\n'))
    assert main(["--json", "--host", HOST + "make", str(SCENE)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert json.loads(lines[0])["host"] == HOST + "make"
    assert json.loads(lines[-1])["episode_over"]
