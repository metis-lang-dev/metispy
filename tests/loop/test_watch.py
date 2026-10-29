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
from metis.loop.watch import Watch, ledger, mirror, mirror_line
from metis.loop.wire import INTENT_VERBS, dispatch, serve_json

from metis.domains.kitchen.model import CATALOG

CAT = CATALOG.parent / "tick"           # the tick form: one action per world turn (the window one action)
SCENE = CAT / "kitchen_host.llp"
HOST = "metis.domains.kitchen.host:tick_"


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
    exo = [s for s in sigs if s.get("kind") == "exo"]
    orders = [o for o in exo if "+pot_space 3" in o["ops"]]
    assert len(orders) == 2
    # the tick form: a turn at each tick the cook used its last one (one action per tick, the window one action)
    assert all(o["ops"] in (["+turn"], ["+pot_space 3", "+turn"], ["+pot_space 3"]) for o in exo)
    # quiescence once per entry, never per idle beat, and declared at the tick whose reading left nothing to do
    # (copy semantics): after each soup and at the script's end. None before order 1: the first tick brought it
    qs = [s for s in sigs if s["signal"] == "quiescence"]
    assert [(q["beat"], q["episode_over"]) for q in qs] == [(12, False), (25, False), (30, True)]
    assert w.beat == 30
    # ... and left no idle row: two soups of 10 (fetch_onion, load ×3, cook, fetch_dish, plate, serve), a stray
    # fetch_onion at the end, one per tick, and the inputs (the orders and the turns)
    soup = ["fetch_onion", "load"] * 3 + ["cook", "fetch_dish", "plate", "serve"]
    assert names == soup * 2 + ["fetch_onion"]
    assert len(sh.fired) == len(sh.fired_rows) == len(names) + len(exo)
    assert len(sh._envs) == len(sh.fired) + 1
    assert sh.counts.get("served") == 2 and not sh.loop.records


@pytest.mark.parametrize("factory,diff,after", [
    ("make_transient", {"pot_onion": 1}, "load"),
    ("make_wrong_domain", {"pot_space": -1, "pot_onion": 1}, "load"),
    # the window's diff: the atoms the member's own events touched (copy semantics). plate touches no hand_free, so
    # the spill's free hand is a reading, not part of the surprise
    ("make_novel", {"held_soup": -1}, "plate"),
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
    assert "hash" in rec["env"] and rec["env"]["format"] == "json"
    assert len(rec["env"]["data"]) == 5     # the host's (beat, kitchen, …) as an array
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
    nxt = next(s for s in sigs[k + 1:] if s.get("kind") != "exo")     # past the tick's input row
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
    while not (sur := [s for s in w.step() if s["signal"] == "surprise"]):
        pass
    row, = mirror(sh, w)
    assert row["status"] == "observe-only"
    # read at the tick, before that tick's beat counts (copy semantics)
    assert row["surprise"] == {"record": 0, "kind": "unmodeled",
                               "beat": sur[0]["beat"]} and sur[0]["beat"] == w.beat - 1
    line = mirror_line(row)
    assert line.startswith("kitchen  stage task  observe-only  "
                           "surprise #0 unmodeled @beat ")
    assert "P[served]=" in line and "\n" not in line
    # the wire face
    res = dispatch(sh, "ctx", "--mirror")
    assert res["ok"] and res["lines"] == [line] and res["text"] == line


def test_mirror_beliefs_are_the_query_marginals_from_now():
    # the chained scene: the projection runs the catalog alone, and the tick form's catalog waits for the world's
    # turns, which no projection supplies
    sh = Repl(CATALOG.parent / "kitchen_host.llp", seed=1, plugin="metis.domains.kitchen.host:make", host=True,
              stdout=io.StringIO())
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


# -- step 3, the metis-line half: cards and intents under the horizon rule

def _triage(factory, intents, per_surprise=None, scope="order"):
    """Run a watch; on each surprise, send the next intent(s) (formatted
    with the record id) and collect their result signals in order."""
    sh = _shell(factory)
    w = Watch(sh, beats=30, scope=scope)
    sigs, pending = [], list(intents)
    while not w.done:
        for s in w.step():
            sigs.append(s)
            if s["signal"] == "surprise":
                for _ in range(per_surprise or 1):
                    if pending:
                        ack = dispatch(sh, "intent",
                                       pending.pop(0).format(r=s["record"]))
                        assert ack["ok"] and ack["intent"] >= 1
                sigs += w.outbox
                w.outbox.clear()
    return sh, w, sigs


def _results(sigs):
    return [s for s in sigs if s["signal"] == "result"]


def test_cards_are_generated_from_provenance_and_include():
    res = dispatch(_shell(), "cards", "")
    by = {c["name"]: c for c in res["cards"]}
    assert list(by) == ["kitchen", "kitchen_shared"]   # served first
    assert by["kitchen"]["relation"] == "served"
    shared = by["kitchen_shared"]
    assert shared["relation"] == "sibling"
    assert shared["text"].startswith("a shared kitchen")
    assert shared["moves"] == ["partner_load"] and shared["slots"] == []
    assert shared["effects"] == [{"name": "partner_load",
                                  "consume": {"pot_space": 1},
                                  "produce": {"pot_onion": 1},
                                  "persist": {}}]
    assert res["text"].splitlines()[1] == (
        'kitchen_shared [sibling] "a shared kitchen: a second cook also '
        'loads onions into the pot" moves partner_load (-pot_space '
        '+pot_onion)')


def test_script_carries_the_catalogs_own_words():
    """metis-agent RAG §11.1: per clause its %% comment, premises and
    effects as atom names, a cross-domain clause's target; per atom its
    doc. The sibling lists only the move it adds and the atoms it
    touches."""
    res = dispatch(_shell(), "script", "")
    by = {s["name"]: s for s in res["script"]}
    assert list(by) == ["kitchen", "kitchen_shared"]
    k = by["kitchen"]
    assert [c["name"] for c in k["clauses"]] == [
        "fetch_onion", "load", "cook", "fetch_dish", "plate", "serve"]
    fetch = k["clauses"][0]
    assert fetch["doc"] == "walk to a dispenser and take an onion."
    turn = {"atom": "turn", "n": 1, "kept": False}            # the tick form: every action takes the world's turn
    assert fetch["premises"] == [turn, {"atom": "hand_free", "n": 1,
                                        "kept": False}]
    assert k["clauses"][2]["premises"] == [
        turn, {"atom": "hand_free", "n": 1, "kept": True},
        {"atom": "pot_onion", "n": 3, "kept": False}]
    assert {a["name"]: a["doc"] for a in k["atoms"]}["pot_onion"] == \
        "onions in the pot"
    assert k["missing"] == {"clause_doc": [], "atom_doc": []}
    pl, = by["kitchen_shared"]["clauses"]
    assert pl["name"] == "partner_load" and pl["target"] == "kitchen"
    assert pl["doc"] == ("the second cook drops an onion into the first "
                         "cook's pot.")          # the tag line stripped
    assert [a["name"] for a in by["kitchen_shared"]["atoms"]] == [
        "pot_space", "pot_onion"]
    assert "partner_load → kitchen:" in res["text"]


def test_transient_is_reset_and_the_kitchen_resumes():
    sh, w, sigs = _triage("make_transient", ["reset {r}"])
    res, = _results(sigs)
    assert res["verb"] == "reset" and res["status"] == "admitted"
    assert sh.loop.records[0]["resolution"]["by"] == "reset"
    assert not w.observe_only and sh.counts.get("served") == 2


def test_wrong_domain_is_explained_by_the_sibling_card_only():
    sh, w, sigs = _triage(
        "make_wrong_domain",
        ["explain {r} kitchen", "explain {r} kitchen_shared load",
         "explain {r} kitchen_shared partner_load"], per_surprise=3)
    served, own, sib = _results(sigs)
    # the served world at exactly k: its own verdict, no invented moves
    assert served["status"] == "refused"
    assert "exactly k=1" in served["reason"]
    # a card names one of ITS moves, not the cook's own
    assert own["status"] == "refused" and "partner_load" in own["reason"]
    assert sib["status"] == "admitted"
    assert sib["witness"] == ["partner_load", "load"]
    assert sib["path"] == "kitchen"       # the signal's domain, never clobbered
    assert sib["horizon"] == {"served": 1, "card": 2}
    assert sib["adopted"] == {"pot_space": -1, "pot_onion": 1}
    # adopted and resumed — one exo row named for the card's move; the
    # served world is NOT switched (activate is case 2, phase 2)
    assert w.path == "kitchen" and not w.observe_only
    assert mirror(sh, w)[0]["path"] == "kitchen"
    assert sh.doc.catalog.name == "kitchen"
    assert "kitchen_shared:partner_load do +pot_onion -pot_space" in sh.fired
    assert sh.loop.records[0]["resolution"] == {
        "by": "explain", "beat": sib["beat"], "card": "kitchen_shared",
        "move": "partner_load"}
    assert sh.counts.get("served", 0) >= 1
    assert not [s for s in sigs if s["signal"] == "surprise"][1:]


def test_novel_is_explained_by_nothing_and_escalated():
    text = "the soup left the dish after plating and nothing was served"
    sh, w, sigs = _triage(
        "make_novel", ["explain {r} kitchen_shared partner_load",
                       "escalate {r}", "escalate {r} " + text],
        per_surprise=3)
    no, empty, esc = _results(sigs)
    assert no["status"] == "refused" and "does not reach" in no["reason"]
    assert no["horizon"] == {"served": 1, "card": 2}
    assert empty["status"] == "refused"
    assert esc["status"] == "admitted"
    rec = sh.loop.records[0]
    assert rec["mined_text"] == text           # the revision loop's input
    assert w.observe_only and "resolution" not in rec   # still frozen


def test_reset_is_bounded_on_the_same_surprise():
    sh, w, sigs = _triage("make_transient_twice",
                          ["reset {r}", "reset {r}"])
    first, second = _results(sigs)
    assert first["status"] == "admitted" and first["record"] == 0
    assert second["status"] == "refused" and second["record"] == 1
    assert "no more resets; escalate" in second["reason"]
    assert second["mined_text"] == (
        "the same misreading (pot_onion +1) recurred after a reset of "
        "record 0 at beat 5: not a one-off")
    assert w.observe_only and w.open == 1
    # the model calls it a one-off; the record keeps the fact ahead
    dispatch(sh, "intent", "escalate 1 the reading was noise, a one-off")
    w.outbox.clear()
    assert sh.loop.records[1]["mined_text"].splitlines() == [
        second["mined_text"], "the reading was noise, a one-off"]
    # a reset of an already-closed record is refused, with why
    dispatch(sh, "intent", "reset 0")
    late, = w.outbox
    assert late["status"] == "refused"
    assert late["reason"].startswith("record 0 is not open (reset at beat")


def test_intents_are_exactly_once_and_refusals_are_signals():
    sh = _shell("make_novel")
    assert not dispatch(sh, "intent", "reset 0")["ok"]   # no watch yet
    w = Watch(sh, beats=30)
    acks = [dispatch(sh, "intent", x)["intent"]
            for x in ("activate kitchen_shared", "reset", "reset 7")]
    assert acks == [1, 2, 3]
    outs = w.outbox
    assert [o["intent"] for o in outs] == [1, 2, 3]
    assert all(o["status"] == "refused" for o in outs)
    assert "unknown intent 'activate'" in outs[0]["reason"]
    assert [o["seq"] for o in outs] == sorted(o["seq"] for o in outs)


def test_the_wire_streams_the_result_right_after_the_ack():
    sh = _shell()
    out = io.StringIO()
    serve_json(sh, io.StringIO(
        '{"op": "watch", "arg": "every=0 beats=2"}\n'
        '{"op": "intent", "arg": "reset 0"}\n'), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[1] == {"ok": True, "intent": 1}
    assert lines[2]["signal"] == "result" and lines[2]["intent"] == 1
    assert lines[2]["reason"] == "no record 0" and lines[2]["seq"] == 1


def test_a_recurring_explained_surprise_is_explained_then_escalated():
    """The partner loads on every order. The first reading is explained
    by kitchen_shared and adopted, but the served world stays the
    one-cook kitchen, so the next order surprises the same way: the
    second explain is refused with a mined_text naming the card, and the
    escalate carries it — explained, then escalated (the case for G-G),
    not a triage miss."""
    sh, w, sigs = _triage("make_partner", [
        "explain {r} kitchen_shared partner_load",
        "explain {r} kitchen_shared partner_load"], scope="session")
    sur = [s for s in sigs if s["signal"] == "surprise"]
    assert len(sur) == 2 and sur[0]["diff"] == sur[1]["diff"]
    first, second = _results(sigs)
    assert first["status"] == "admitted"
    assert second["status"] == "refused" and second["record"] == 1
    assert "activate is phase 2" in second["reason"]
    assert second["card"] == "kitchen_shared"
    assert "kitchen_shared (partner_load)" in second["mined_text"]
    dispatch(sh, "intent", "escalate 1 the pot has one onion too many")
    esc, = w.outbox
    assert esc["status"] == "admitted"
    # the runtime's fact first, the model's description after
    assert sh.loop.records[1]["mined_text"] == (
        second["mined_text"] + "\nthe pot has one onion too many")
    assert w.observe_only and w.open == 1


def test_ledger_is_the_runtime_account_of_the_run():
    sh, w, sigs = _triage("make_partner", [
        "explain {r} kitchen_shared partner_load",
        "explain {r} kitchen_shared partner_load"], scope="session")
    dispatch(sh, "intent", "escalate 1 the partner loads every order")
    led = dispatch(sh, "ledger", "")
    assert led["ok"] and [r["record"] for r in led["records"]] == [0, 1]
    r0, r1 = led["records"]
    assert r0["resolution"]["by"] == "explain" and r0["since"] == ["load"]
    assert r0["diff"] == {"pot_space": -1, "pot_onion": 1}
    assert r1["refusals"] == ["explain"] and r1["resolution"] is None
    assert r1["mined_text"].splitlines()[-1] == \
        "the partner loads every order"
    assert r1["mined_text"].startswith("kitchen_shared (partner_load)")
    assert r1["facts"] == [r1["mined_text"].splitlines()[0]]
    assert [x["status"] for x in led["results"]] == [
        "admitted", "refused", "admitted"]
    assert led["watch"]["surprise"] == 1
    json.dumps(led)


# -- step 4/5 prep: the hour, the operator, arm A (blind) ------------------

def _oracle(sh, w, s):
    """The intended triage per diff (the eval's 'oracle' arm)."""
    r, d = s["record"], s["diff"]
    x = (f"reset {r}" if d == {"pot_onion": 1} else
         f"explain {r} kitchen_shared partner_load"
         if d == {"pot_onion": 1, "pot_space": -1} else f"escalate {r} x")
    dispatch(sh, "intent", x)
    if w.outbox[-1]["status"] == "refused":
        dispatch(sh, "intent", f"escalate {r} y")


def _hour(scope):
    sh = _shell("make_hour")
    w = Watch(sh, beats=7200, operator=120, scope=scope)
    sigs, calls = [], []
    while not w.done:
        for s in w.step():
            sigs.append(s)
            if s["signal"] == "surprise":
                _oracle(sh, w, s)
                calls.append((s["record"], [o["verb"] + ":" + o["status"]
                                            for o in w.outbox]))
                w.outbox.clear()
    return sh, w, sigs, calls, ledger(sh, w)


def test_the_hour_fires_its_whole_schedule_with_an_operator():
    sh, w, sigs, calls, led = _hour("order")
    kinds = [e["kind"] for e in led["host"]["injected"]]
    assert kinds == ["transient", "wrong_domain", "novel", "partner",
                     "partner", "partner", "transient_twice",
                     "transient_twice"]
    assert len(led["host"]["schedule"]["orders"]) == 26
    assert sum(1 for s in sigs if s["signal"] == "surprise") == 8
    assert led["final"]["diff"] == {}
    assert led["final"]["served_model"] == led["final"]["served_host"] >= 20
    assert led["watch"]["scope"] == "order"


def test_same_surprise_is_scoped_to_the_order_in_the_hour():
    """same_surprise_scope=order (the default): the bounds forget a
    resolved surprise once its order is served. 35 min: each partner
    onion lands on a NEW order, so every explain is admitted again (the
    15-min wrong_domain no longer blocks it). 45 min: the twice-misread
    falls inside ONE order, so the 5-min transient no longer bounds the
    first reset, and the recurrence within the order is refused and
    escalates."""
    sh, w, sigs, calls, led = _hour("order")
    by = [r["resolution"]["by"] for r in led["records"]]
    assert by == ["reset", "explain", "operator",          # 5, 15, 25 min
                  "explain", "explain", "explain",          # 35 min
                  "reset", "operator"]                      # 45 min
    # explained, then noticed: the same card+move in consecutive orders
    assert calls[3:6] == [(3, ["explain:admitted"]),
                          (4, ["explain:admitted+notice"]),
                          (5, ["explain:admitted+notice"])]
    # the 15-min wrong_domain is the same card+move, but not consecutive
    assert calls[1] == (1, ["explain:admitted"])
    assert led["records"][5]["notice"] == {
        "card": "kitchen_shared", "move": "partner_load", "orders": 3}
    assert calls[6:] == [(6, ["reset:admitted"]),
                         (7, ["reset:refused", "escalate:admitted"])]
    assert led["records"][7]["facts"][0].startswith(
        "the same misreading (pot_onion +1) recurred after a reset of "
        "record 6")


def test_session_scope_remembers_for_the_whole_watch():
    sh, w, sigs, calls, led = _hour("session")
    by = [r["resolution"]["by"] for r in led["records"]]
    assert by == ["reset", "explain"] + ["operator"] * 6
    assert calls[3] == (3, ["explain:refused", "escalate:admitted"])
    assert calls[6] == (6, ["reset:refused", "escalate:admitted"])


def test_operator_resumes_only_an_escalated_surprise():
    sh = _shell("make_novel")
    w = Watch(sh, beats=30, operator=3)
    sigs = []
    while not w.done:
        sigs += w.step()
    assert w.observe_only                 # never escalated: no operator
    assert not any(s.get("kind") == "operator" for s in sigs)
    sh = _shell("make_novel")
    w = Watch(sh, beats=30, operator=3)
    sigs = []
    while not w.done:
        for s in w.step():
            sigs.append(s)
            if s["signal"] == "surprise":
                dispatch(sh, "intent", f"escalate {s['record']} spilled")
    op, = [s for s in sigs if s.get("kind") == "operator"]
    assert op["adopted"] == {"hand_free": 1, "held_soup": -1}
    assert sh.loop.records[0]["resolution"]["by"] == "operator"
    assert "operator do +hand_free -held_soup" in sh.fired
    assert not w.observe_only


def test_blind_arm_applies_every_intent_and_keeps_the_shadow_verdict():
    sh = _shell("make_novel")
    w = Watch(sh, beats=30, blind=True)
    sigs = []
    while not w.done:
        for s in w.step():
            sigs.append(s)
            if s.get("kind") == "reading" and not s["consistent"] \
                    and not w.results:
                # arm A's mistake: call the spill the partner's onion
                dispatch(sh, "intent",
                         f"explain {s['record']} kitchen_shared partner_load")
    assert not any(s["signal"] == "surprise" for s in sigs)
    res, = w.results
    assert res["status"] == "applied" and res["shadow"] == "refused"
    # applied blind: the wrong reading adopted, the session ran on
    assert sh.loop.records[0]["resolution"]["by"] == "explain"
    assert any(s.get("kind") == "clause" and s["seq"] > res["seq"]
               for s in sigs)
    led = ledger(sh, w)
    assert led["watch"]["blind"] is True


def test_partner_at_order_scope_is_explained_then_noticed():
    """The silent-wrong-context case: each order's partner onion is
    explained and the kitchen keeps working (no freeze), but from the
    2nd consecutive order on the record carries a NOTICE for the
    revision loop — what phase 2's activate will consume."""
    sh, w, sigs = _triage("make_partner", [
        "explain {r} kitchen_shared partner_load"] * 3)
    res = _results(sigs)
    assert [r["status"] for r in res] == [
        "admitted", "admitted+notice", "admitted+notice"]
    assert res[2]["mined_text"] == (
        "kitchen_shared (partner_load) explained this diff in 3 consecutive"
        " orders: the served world kitchen lacks it; activate is phase 2")
    recs = sh.loop.records
    assert "notice" not in recs[0]
    assert recs[1]["notice"]["orders"] == 2
    assert recs[2]["mined_text"] == res[2]["mined_text"]
    assert all(r["resolution"]["by"] == "explain" for r in recs)
    # no freeze; (the 3rd soup can stall on the catalog's held-onion
    # dead end — cook needs a free hand — until an order that never comes)
    assert not w.observe_only and sh.counts.get("served") >= 2
    # the knob: off (0) never notices; 3 notices only the third
    for n, want in ((0, ["admitted"] * 3),
                    (3, ["admitted", "admitted", "admitted+notice"])):
        sh = _shell("make_partner")
        w = Watch(sh, beats=44, explain_repeat_notice=n)
        while not w.done:
            for s in w.step():
                if s["signal"] == "surprise":
                    dispatch(sh, "intent", f"explain {s['record']} "
                             "kitchen_shared partner_load")
        assert [r["status"] for r in w.results] == want


def test_mirror_carries_progress_and_the_last_event():
    """RACK v2 finding 1: an ask sees the mirror line only, so the line
    says the #query atoms' counts now and the last thing the domain did."""
    sh = _shell()
    w = Watch(sh, beats=30)
    while not w.done:
        w.step()
    row, = mirror(sh, w)
    assert row["now"] == {"served": 2, "cooking": 0, "held_soup": 0}
    assert row["last"] == "fetch_onion"          # not the env's tick
    line = mirror_line(row)
    assert line.endswith("now served=2 cooking=0 held_soup=0  last fetch_onion")


# -- repl-tools step 4: the orchestrator's verbs come from the REPL API

def test_the_wire_dispatches_every_declared_verb():
    """INTENT_VERBS = the loop's design verbs ∪ the REPL API's reads
    (served_by repl) and surprise verbs (served_by watch), derived; a
    declared verb the wire cannot dispatch fails here; help carries each
    declared verb's description, the line the model reads."""
    from metis.lang.repl_api import REPL_API
    from metis.loop.wire import _WIRE_OPS, LOOP_VERBS
    declared = {t["name"]: t["description"] for t in REPL_API
                if t["served_by"] in ("repl", "watch")}
    assert INTENT_VERBS == LOOP_VERBS | set(declared)
    sh = _shell()
    for name in declared:
        assert name in _WIRE_OPS or callable(getattr(sh, "cmd_" + name, None)), \
            f"the REPL API declares '{name}' and the wire has no op for it"
    h = dispatch(sh, "help", "")
    assert h["ops"] == sorted(INTENT_VERBS)
    assert {n: h["descriptions"][n] for n in declared} == declared


def test_help_describes_every_verb_from_what_serves_it():
    """agentic-loop s-3ddb3e87 finding 1: help carries words for every
    wire verb, not only the declared ones — the docstring of what serves
    it (the wire's op, the loop verb, the REPL command), one line, so a
    tool host (pi's metis-repl) has a description for each."""
    from metis.loop.wire import _WIRE_OPS
    sh = _shell()
    h = dispatch(sh, "help", "")
    d = h["descriptions"]
    assert sorted(d) == h["ops"]
    bare = [n for n in h["ops"] if not d[n]]
    assert not bare, f"verbs without a description on the wire: {bare}"
    assert all("\n" not in t and "  " not in t for t in d.values())
    assert d["formalize"].startswith("Draft the candidate from the TEXT")   # metis.loop.verbs
    assert d["run"] == sh.do_run.__doc__.strip()                            # metis.lang.repl
    assert d["trace"] == " ".join(_WIRE_OPS["trace"].__doc__.split("\n\n")[0].split())
    assert "CONSOLE §2a" not in " ".join(d.values())     # no placeholder words
    assert dispatch(sh, "help", "ops") == {"ok": True, "ops": h["ops"]}
    assert dispatch(sh, "help", "run gate")["descriptions"] == {"run": d["run"], "gate": d["gate"]}
    assert not dispatch(sh, "help", "nosuch")["ok"]


def test_the_surprise_verbs_are_page_intents_on_the_watch():
    from metis.loop.wire import run_intent
    sh = _shell("make_transient")
    w = Watch(sh, beats=30)
    sh.__dict__["_watch"] = w
    sigs = []
    while not w.done:
        for s in w.step():
            sigs.append(s)
            if s["signal"] == "surprise":
                ack = run_intent(sh, f":reset {s['record']}")
                assert ack == {"ok": True, "intent": 1}
                sigs += w.outbox
                w.outbox.clear()
    res, = _results(sigs)
    assert res["verb"] == "reset" and res["status"] == "admitted"
    assert sh.loop.records[0]["resolution"]["by"] == "reset"
    # explain/escalate go down as the watch's intents
    assert run_intent(sh, ":escalate 0 a one-off")["intent"] == 2
    late, = w.outbox[-1:]
    assert late["verb"] == "escalate" and late["status"] == "refused"
    # the bare reset stays the loop's own: the session back to #init
    r = run_intent(sh, ":reset")
    assert r["ok"] and "intent" not in r and r["counts"] == dict(sorted(sh.counts.items()))
    assert sh.stage == sh.prog.init_stage and sh.counts == dict(sh.case["init"])
    # with no watch, a surprise verb says why
    assert "no watch" in dispatch(_shell(), "explain", "0 kitchen_shared partner_load")["error"]
