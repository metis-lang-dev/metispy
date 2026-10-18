"""CONSOLE §4 (metis-loop docs/CONSOLE.md), step 1: the `ctx`
aggregate (wire op + REPL command), the read-only ops in run_intent's
whitelist, and result postings that carry `json` beside the text."""

import io
from pathlib import Path

from metis.lang.repl import Repl
from metis.loop.wire import (INTENT_VERBS, dispatch, intent_posting,
                             parse_intent)

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ROOT.parent / "metis-catalog/domains"


def _shell(name, *cmds):
    r = Repl(DOMAINS / name, seed=1, stdout=io.StringIO())
    for c in cmds:
        r.onecmd(c)
    r.stdout = io.StringIO()
    return r


def test_ctx_aggregate_on_3lp():
    """3lp seed 1 after run 3: running, beat 3, two enabled events
    with normalized p, and the six rules in source order with list's
    states as data."""
    c = dispatch(_shell("three_pigs/three_pigs.llp", "run 3"), "ctx", "")
    assert c["ok"] and c["catalog"] == "three-pigs"
    assert (c["stage"], c["status"], c["fired"], c["beat"],
            c["episode"]) == ("go", "running", 3, 3, 0)
    assert c["counts"] == {"bricks": 1, "pig": 1, "stickhouse": 1,
                           "wolf": 1}
    assert [e["name"] for e in c["enabled"]] == ["build_brick",
                                                 "blow_stick"]
    assert abs(sum(e["p"] for e in c["enabled"]) - 1) < 1e-12
    assert [(r["name"], r["state"]) for r in c["rules"]] == [
        ("build_straw", "rests_on"), ("build_stick", "rests_on"),
        ("build_brick", "fireable"), ("blow_straw", "fired_pruned"),
        ("blow_stick", "fireable"), ("blow_brick", "waiting")]
    waiting = c["rules"][-1]
    assert waiting["detail"] == "? brickhouse ← build_brick*"
    assert waiting["line"] == 43
    assert c["rules"][1]["rests_on"] == [1]
    assert c["last_event"] == "blow_straw" and c["rests_on_stages"] == []


def test_ctx_rule_states_are_lists_marks_one_to_one():
    """Every rule's state is the mark `list` prints on its head line."""
    for name, cmds in (("three_pigs/three_pigs.llp", ["run 3"]), ("three_pigs/three_pigs.llp", ["run 6"]),
                       ("herstory/herstory.llp", ["run 12"]),
                       ("love_triangle/love_triangle.llp", ["run 4"])):
        r = _shell(name, *cmds)
        c = r.ctx()
        listing = r.capture("list").splitlines()
        sym = {v: k for k, v in Repl._CTX_STATE.items()}
        for rule in c["rules"]:
            head = listing[next(k for k, l in enumerate(listing)
                                if l.lstrip().startswith(
                                    tuple("*◂·?– "))
                                and f" {rule['line']} " in l[:12])]
            assert head.lstrip()[0] == sym[rule["state"]] or \
                (rule["state"] == "idle" and head[2] == " "), head


def test_ctx_status_running_quiescent_done():
    """herstory: running mid-act; after florence_dies only the act2
    link can fire (quiescent: the stage has settled); done at the
    end. Episodes count the links fired, rests_on_stages fill up."""
    assert _shell("herstory/herstory.llp", "run 2").ctx()["status"] == "running"
    q = _shell("herstory/herstory.llp", "run 3").ctx()
    assert q["status"] == "quiescent"
    assert [e["name"] for e in q["enabled"]] == ["act2"]
    d = _shell("herstory/herstory.llp", "run 40").ctx()
    assert d["status"] == "done" and d["episode"] == 3
    assert [s["stage"] for s in d["rests_on_stages"]][:2] == \
        ["beginning", "courtship"]


def test_ctx_text_band_and_sections():
    r = _shell("three_pigs/three_pigs.llp", "run 3")
    full = r.capture("ctx")
    assert full.startswith("CATALOG three-pigs  KEY 1e3be62725e7  STAGE "
                           "go  BEAT 3  EPISODE 0  RUNNING  3 fired")
    assert "TOKENS  bricks ●  pig ●  stickhouse ●  wolf ●" in full
    assert all(f"── {s}" in full for s in ("tokens", "rules", "trace"))
    rules = r.capture("ctx", "rules")
    assert "── rules" in rules and "── tokens" not in rules
    assert "? brickhouse ← build_brick*" in rules
    assert "usage" in r.capture("ctx", "bogus")
    assert dispatch(r, "ctx", "trace")["text"].endswith("frontier: 3")
    assert dispatch(r, "ctx", "bogus")["ok"] is False


def test_whitelist_admits_read_only_ops_and_still_refuses_the_rest():
    for line in (":ctx", ":ctx rules", ":graph", ":why", ":causes a b",
                 ":findings", ":findings 1"):
        assert parse_intent(line) is not None, line
    for line in (":accept", ":load x.llp", ":export d", ":sink f",
                 ":llm command:sh", ":formalize command:x", "ctx"):
        assert parse_intent(line) is None, line


def test_intent_posting_carries_json_beside_text():
    r = _shell("three_pigs/three_pigs.llp", "run 3")
    p = intent_posting(r, ":ctx rules", session="w1")
    assert p["status"] == "done" and p["session"] == "w1"
    assert p["result"].startswith("CATALOG three-pigs")
    assert "text" not in p["json"] and p["json"]["status"] == "running"
    why = intent_posting(r, ":why")
    assert why["result"] == r.capture("why")       # the REPL's own text
    assert why["json"]["cone"] == [0, 1, 2]
    assert intent_posting(r, ":accept")["status"] == "refused"
    bad = intent_posting(r, ":ctx bogus")
    assert bad["status"] == "error" and "usage" in bad["result"]
    assert "session" not in intent_posting(r, ":graph")


def test_both_help_surfaces_name_ctx():
    r = _shell("three_pigs/three_pigs.llp")
    assert "ctx [tokens|rules|trace]" in r.capture("help")
    ops = dispatch(r, "help", "")["ops"]
    assert {"ctx", "findings", "graph", "why", "causes"} <= set(ops)
    assert "ctx" in INTENT_VERBS and "accept" not in INTENT_VERBS


# -- console step 2: the execution ops behind the page's chips --------

def test_execution_ops_return_text_and_beats():
    """run/step carry the REPL's own text plus each fired event with
    its beat, episode and linear effect (the page's beat blocks);
    reset/undo/state/log/list/events are whitelisted and structured."""
    r = _shell("herstory/herstory.llp")
    out = dispatch(r, "run", "5")
    assert out["text"].startswith("eve_hannah_birth hannah_eve_meet")
    f = out["fired"]
    assert [x["beat"] for x in f] == [1, 2, 3, 4, 5]
    assert f[3]["kind"] == "link" and f[3]["episode"] is None
    assert (f[3]["stage"], f[3]["post"]) == ("beginning", "courtship")
    assert f[0]["consume"] == {"pregnant(mother)": 1}
    assert f[4]["episode"] == 1
    ev = dispatch(r, "events", "")
    assert ev["enabled"][0]["name"] == "breakup_carl"
    assert dispatch(r, "step", "")["fired"][0]["i"] == 6
    assert dispatch(r, "undo", "")["fired"] == []
    assert len(r.fired) == 5
    assert dispatch(r, "reset", "")["stage"] == "beginning"
    assert dispatch(r, "log", "")["fired"] == 5
    for verb in ("run", "step", "events", "reset", "undo", "state",
                 "log", "list"):
        assert parse_intent(":" + verb) is not None, verb


def test_ctx_carries_rule_source_and_run_key():
    r = _shell("three_pigs/three_pigs.llp", "run 3")
    c = r.ctx()
    brick = next(x for x in c["rules"] if x["name"] == "build_brick")
    assert brick["src"] == ("  %% a pig spends the bricks on a brick "
                            "house.\n  build_brick [story] : pig * bricks"
                            " -o brickhouse.")
    assert c["run_key"] and len(c["run_key"]) == 64
    r.onecmd("reset")
    assert r.ctx()["run_key"] != c["run_key"]


def test_wire_narrate_tells_the_session_run_bare_and_a_seed_on_request():
    """Console step 4: bare `narrate` narrates what the stream shows —
    the session's run since the base (REPL parity); `narrate N` a fresh
    seeded trace. `command:cat` echoes the certified material."""
    r = _shell("three_pigs/three_pigs.llp", "run 3")
    r.llm = "command:cat"
    s = dispatch(r, "narrate", "")
    assert (s["source"], s["events"]) == ("session", 3)
    beats = s["narration"].rsplit("== the run, in beats ==", 1)[1] \
        .split("== the final state ==")[0]     # this run, not an example
    assert "blow_straw" in beats and "build_brick" not in beats
    t = dispatch(r, "narrate", "2")
    assert (t["source"], t["seed"]) == ("seed", 2)
    r.onecmd("reset")
    assert dispatch(r, "narrate", "")["source"] == "seed"   # nothing since


def test_wire_trace_bare_is_the_repls_case_replay_with_its_notes():
    """Console step 4: bare `trace` carries the REPL's own replay text
    and records its runtime notes (tragedy: exact out of reach), so
    `findings` sees them; `trace SEED` keeps the seeded JSON only."""
    r = _shell("tragedy/tragedy.llp")
    t = dispatch(r, "trace", "")
    assert t["text"].startswith("case tragedy") and "Q " in t["text"]
    codes = [d["code"] for d in dispatch(r, "findings", "")["diagnostics"]]
    assert "exact-unreachable" in codes
    assert "text" not in dispatch(r, "trace", "2")
