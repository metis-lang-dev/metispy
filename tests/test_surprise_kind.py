"""A surprise record's kind (metis-lang spec/20 §4, §8; surprise-kinds s-7a02c2ac step 1): the record's new fields in
the workspace, the one-guard-away listing, and the triage over the trace's three columns.

The six records of spec 20 §9 are fixtures (tests/fixtures/surprise_kinds/: the records as the road dropped them on
2026-11-05, and the trace segments that held their windows on that day). No road trace has a truth column yet (spec 20
§10.2, surprise-kinds step 2), so as recorded every one is evidential. The tests of the procedure's branches add a
truth column written by hand from spec 20 §9's truth column (`TRUTH` below: ids and classes illustrative, the things
and the ticks the section names); each such test says so."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from metis import surprise_kind as SK
from metis import trace as T
from metis.catalog import workspace as W
from metis.lang.repl import Repl as Shell

FIX = Path(__file__).parent / "fixtures" / "surprise_kinds"
IDS = {"463d7d": "20261105T082838-463d7d", "503be4": "20261105T031542-503be4", "40f229": "20261105T031542-40f229",
       "87279b": "20261105T031542-87279b", "668e45": "20261105T051440-668e45", "13b2f9": "20261105T051440-13b2f9"}


@pytest.fixture
def ws(tmp_path):
    """A workspace holding the six records and their trace windows (copied: a test may write truth and kinds)."""
    st = tmp_path / ".metis"
    shutil.copytree(FIX / "surprises", st / "surprises")
    shutil.copytree(FIX / "traces", st / "traces")
    return W.Workspace(tmp_path)


def rec_of(ws, short):
    return ws.surprise(IDS[short])


def trace_of(ws, rec):
    return ws.state / rec["trace"]


def write_truth(path, rows, header=None):
    lines = ([json.dumps(header)] if header else []) + [json.dumps(r) for r in rows]
    (Path(path) / SK.TRUTH).write_text("\n".join(lines) + "\n")


# ── the record (workspace) ──
def test_a_record_carries_its_kind_and_evidence_and_the_journal_names_the_kind(tmp_path):
    w = W.Workspace(tmp_path)
    sid = w.drop_surprise({"member": "planner", "reading": "x", "kind": "structural",
                           "evidence": {"facts": ["a"], "law": {"fired": {}}}, "response": "a pack"})
    rec = w.surprise(sid)
    assert rec["kind"] == "structural" and rec["status"] == "open"
    assert w.events()[-1]["surprise_kind"] == "structural"


@pytest.mark.parametrize("bad, why", [
    ({"kind": "mystery"}, "kind 'mystery'"),
    ({"kind": "structural", "evidence": {"opinion": 1}}, "evidence: an object over"),
    ({"kind": "structural", "verdict": "agreed"}, "belong to a confusion record"),
    ({"kind": "confusion", "mark": {"n": 1, "k0": 3, "k1": 9, "said": "w"}}, "carries the person's verdict"),
    ({"kind": "confusion", "verdict": "maybe"}, "verdict 'maybe'"),
    ({"kind": "confusion", "verdict": "disagreed", "mark": {"k0": "3"}}, "a tick"),
    ({"kind": "confusion", "verdict": "disagreed", "then": "confusion"}, "then 'confusion'"),
    ({"kind": "confusion", "verdict": "disagreed", "narration": "one line"}, "a list of strings"),
])
def test_the_kind_fields_are_refused_in_their_own_words(tmp_path, bad, why):
    with pytest.raises(ValueError, match=why):
        W.Workspace(tmp_path).drop_surprise(dict({"member": "m", "reading": "x"}, **bad))


def test_an_agreed_confusion_is_a_closed_mark_and_a_disagreed_one_is_open(tmp_path):
    w = W.Workspace(tmp_path)
    mark = {"n": 1, "k0": 40, "k1": 60, "said": "it should have braked"}
    agreed = w.drop_surprise({"member": "responder", "reading": "a mark", "kind": "confusion", "mark": mark,
                              "narration": ["the crosser could stop: the duty was its own"], "verdict": "agreed"})
    disagreed = w.drop_surprise({"member": "responder", "reading": "a mark", "kind": "confusion", "mark": mark,
                                 "narration": ["the crosser could stop"], "verdict": "disagreed"})
    assert w.surprise(agreed)["status"] == "closed" and w.surprise(disagreed)["status"] == "open"
    assert [r["id"] for r in w.surprises("closed")] == [agreed]


def test_the_triage_rewrites_the_record_whole_and_keeps_the_words(tmp_path):
    w = W.Workspace(tmp_path)
    sid = w.drop_surprise({"member": "planner", "reading": "the agent's", "words": "the person's"})
    new = w.set_kind(sid, "ontological", {"truth": [], "missing": []}, "new vocabulary", also=["structural"])
    rec = w.surprise(sid)
    assert rec["kind"] == "ontological" and rec["also"] == ["structural"] and rec["words"] == "the person's" and rec["reading"] == "the agent's"
    assert new["kind"] == "ontological" and w.events()[-1]["kind"] == "kind" and w.events()[-1]["surprise_kind"] == "ontological"


# ── the law column: one guard away ──
PROGRAM = """
catalog guard 1.
provenance text "a small program for the one-guard-away listing" "2026-11-05".
layers (dyn).
stages (main).
namespace g produce (dyn).
pred go : g.
pred near : g.
pred clear : g.
pred stopped : g.
stage main {
  %% Something near and the way clear: I stop.
  halt [dyn] : go * $near * $clear -o stopped.
}
#init go 1
#steps 20
#seeds 1
"""


def test_one_guard_away_lists_the_rule_short_of_one_premise_and_when_it_fired(tmp_path):
    p = tmp_path / "guard.llp"
    p.write_text(PROGRAM)
    w = T.Segments(tmp_path / "t", every=0)
    head = {"members": {"m": {"key": T.program_key(p), "seed": 1, "weights": [], "out": [], "in": []}}, "commands": []}
    w.open_segment(0, head, {"sessions": {"m": T.session_state(Shell(str(p), seed=1))}, "copies": {"m": {}}})
    for k, beat, fired in [(0, ["+near"], []), (1, [], []), (2, ["+clear"], ["halt"])]:
        w.line({"k": k, "members": {"m": {"why": "change", "seq": [{"beat": beat, "fired": fired}]}}})
    w.close()
    rows = SK.one_guard_away(tmp_path / "t", "m", 0, 2, p)
    by = {(r["ground"], r["missing"]): r for r in rows}
    assert by[("halt", "clear")]["ticks"] == [0, 1] and by[("halt", "clear")]["stage"] == "main"
    assert by[("halt", "go")]["ticks"] == [2] and by[("halt", "go")]["fired_at"] == [2]       # after it fired, go is spent
    assert SK.one_guard_away(tmp_path / "t", "m", 1, 1, p)[0]["ticks"] == [1]                  # the window bounds the probes


# ── the triage on the six records as recorded ──
def test_as_recorded_every_record_is_evidential_and_names_what_it_lacks(ws):
    for short in IDS:
        rec = rec_of(ws, short)
        out = SK.triage(rec, trace_of(ws, rec))
        assert out["kind"] == "evidential", (short, out["kind"])
        assert "truth" in out["evidence"]["missing"], short
    rec = rec_of(ws, "13b2f9")                    # its run was overwritten by a shorter one: no segment holds its window
    assert "facts" in SK.triage(rec, trace_of(ws, rec))["evidence"]["missing"]


# ── the procedure's branches, with spec 20 §9's truth column written by hand ──
def car(tid, role, cls="car", as_="obj", **attrs):
    return dict({"id": tid, "class": cls, "as": as_, "role": role}, **attrs)


def window(rec):
    return range(rec["window"][0], rec["window"][1] + 1)


def test_the_lane_stay_is_ontological_a_word_no_symbol_holds(ws):
    """§9: 463d7d, the truth has the ego on the bidirectional lane 65_2; no atom of the member's vocabulary holds it.
    (The fixture's trace is the card's rerun after the lane-kind pack, whose facts name only driving and none: the
    vocabulary read here is the facts', so 'bidirectional' is held by no symbol of them.)"""
    rec = rec_of(ws, "463d7d")
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "65_2", "kind": "bidirectional"}, "things": []} for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "ontological" and "bidirectional" in out["evidence"]["why"][0]
    assert out["response"].startswith("new vocabulary")


def test_the_priority_taker_is_a_bound_crossed_and_also_structural(ws):
    """§9: 503be4, the crosser is a1 throughout, the ego in contact at the window's end; the fold asks no longitudinal
    response (resp_lon(none)) and the responder fires no command: probabilistic first, structural also."""
    rec = rec_of(ws, "503be4")
    k1 = rec["window"][1]
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "e_0", **({"contact": ["adversary"]} if k == k1 else {})},
                                     "things": [car("adversary", "a1", lane="c_0")]} for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "probabilistic" and out["also"] == ["structural"]
    assert "no longitudinal response" in out["evidence"]["why"][0]


@pytest.mark.parametrize("short, cut", [("40f229", 63), ("87279b", 83)])
def test_a_cut_in_is_a_bound_crossed_with_only_a_lateral_response(ws, short, cut):
    """§9: 40f229 / 87279b, the car that cut in is the lead from the cut on; at contact the fold asks only a lateral response."""
    rec = rec_of(ws, short)
    k1 = rec["window"][1]
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "e_1", **({"contact": ["cutin"]} if k == k1 else {})},
                                     "things": [car("cutin", "lead" if k >= cut else "leadright")]} for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "probabilistic" and "only a lateral one" in out["evidence"]["why"][0]


def test_the_gap_cards_are_ontological(ws):
    """§9: 668e45, the oncoming lane a pass would borrow; 13b2f9, an emergency vehicle behind: no atom names either.
    13b2f9 has no facts left (its run was overwritten), and the vocabulary decides it all the same."""
    rec = rec_of(ws, "668e45")
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "-1_1"}, "things": [
        car("parked", "lead"), {"id": "-1_0", "class": "lane", "as": None, "role": None, "direction": "oncoming"}]} for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "ontological" and "oncoming" in out["evidence"]["why"][0]
    rec = rec_of(ws, "13b2f9")
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "-67_1"}, "things": [car("ambulance", "follower", cls="emergency", as_=None)]}
                                    for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "ontological" and "emergency" in out["evidence"]["why"][0] and "facts" in out["evidence"]["missing"]


def test_a_car_the_host_never_bound_is_perception(ws):
    """§9's SUMO row of 503be4: the crosser is in the truth, unbound until the window's end (out of the sensor's 60 m),
    and its class projects onto obj: the host did not project it."""
    rec = dict(rec_of(ws, "503be4"), window=[39, 44])
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "e_0"}, "things": [car("adversary", None)]} for k in window(rec)])
    out = SK.triage(rec, trace_of(ws, rec))
    assert out["kind"] == "perception" and out["response"].startswith("a host parameter")


def test_a_thing_the_world_cannot_hold_is_world(ws):
    """§9's walkers: the words name a thing the host's truth header says the world cannot hold."""
    rec = dict(rec_of(ws, "503be4"), words="a walker stepped off the sidewalk")
    write_truth(trace_of(ws, rec), [{"k": k, "things": []} for k in window(rec)], header={"truth": 1, "cannot": ["sidewalk", "walker"]})
    assert SK.triage(rec, trace_of(ws, rec))["kind"] == "world"


def test_a_confusion_mark_closes_on_agreement_and_runs_the_procedure_on_the_words(ws):
    rec = dict(rec_of(ws, "503be4"), mark={"n": 1, "k0": 39, "k1": 64, "said": "that car took my way and I never braked"},
               verdict="agreed", narration=["the crosser could stop"])
    assert SK.triage(rec, trace_of(ws, rec))["then"] is None
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "e_0", **({"contact": ["adversary"]} if k == 64 else {})},
                                     "things": [car("adversary", "a1")]} for k in range(39, 65)])
    out = SK.triage(dict(rec, verdict="disagreed"), trace_of(ws, rec))
    assert out["kind"] == "confusion" and out["verdict"] == "disagreed" and out["then"] == "probabilistic"


def test_the_kind_verb_writes_the_record(ws):
    rec = rec_of(ws, "463d7d")
    write_truth(trace_of(ws, rec), [{"k": k, "ego": {"lane": "65_2", "kind": "bidirectional"}, "things": []} for k in window(rec)])
    p = subprocess.run([sys.executable, "-m", "metis.trace", "kind", rec["id"], "--root", str(ws.root), "--write"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["kind"] == "ontological"
    assert ws.surprise(rec["id"])["kind"] == "ontological" and ws.events()[-1]["kind"] == "kind"


def test_t10_truth_prints_the_truth_column_and_the_unbound_things(ws, capsys):
    rec = rec_of(ws, "503be4")
    write_truth(trace_of(ws, rec), [{"k": 1, "t": 0.0, "ego": {"lane": "e_1", "kind": "driving", "contact": []},
                                     "things": [{"id": "lead", "class": "passenger", "as": "obj", "role": "lead", "lane": "e_1", "s": 60.0, "v": 13.0, "kind": "driving"},
                                                {"id": "x", "class": "passenger", "as": "obj", "role": None, "lane": "e_2", "s": 90.0, "v": 9.0}]}],
                header={"truth": 1, "cannot": ["sidewalk"]})
    t = T.truth_table(trace_of(ws, rec))
    assert t.notes == ["truth: 1 ticks (1..1), 3 rows", "the world cannot hold: sidewalk"]
    assert t.rows[0][:2] == [1, "ego"] and t.rows[1][:5] == [1, "lead", "passenger", "obj", "lead"]
    assert [r[1] for r in T.truth_table(trace_of(ws, rec), unbound=True).rows] == ["x"]
    assert T.main(["truth", str(trace_of(ws, rec)), "--id", "lead"]) == 0 and "lead" in capsys.readouterr().out
    with pytest.raises(T.TraceError, match="no truth column"):
        T.truth_table(trace_of(ws, rec_of(ws, "40f229")))


# ── step 1b: a record keeps its window at the drop; the loops' tools never write a kind ──
def _two_segment_trace(where):
    s = T.Segments(where, every=5)
    for k in range(10):
        if s.due(k):
            s.open_segment(k, {"members": {"m": {"key": "", "seed": 1, "weights": [], "out": [], "in": []}}},
                           {"sessions": {"m": {"stage": "main", "quiet": None, "rng": "1", "counts": {}}}, "copies": {"m": {}}})
        s.line({"k": k, "members": {"m": {"why": "change", "seq": [{"beat": ["+x%d" % k], "fired": []}]}}})
    s.close()


def test_a_dropped_record_keeps_its_window_past_a_rerun(tmp_path):
    w = W.Workspace(tmp_path)
    _two_segment_trace(w.state / "traces" / "card-1")
    sid = w.drop_surprise({"member": "m", "reading": "x", "trace": "traces/card-1", "k": 7, "window": [6, 8]})
    rec = w.surprise(sid)
    assert rec["trace"] == "traces/card-1-%s" % sid and rec["run"] == "card-1" and "keep_error" not in rec
    assert [e["kind"] for e in w.events()] == ["keep", "surprise"]
    shutil.rmtree(w.state / "traces" / "card-1")                 # a rerun under the run's name empties the original
    _two_segment_trace(w.state / "traces" / "card-1")
    ticks, _ = T.timeline(w.state / rec["trace"], "m")
    assert {6, 7, 8} <= {t.k for t in ticks} and "x7" in ticks[[t.k for t in ticks].index(7)].counts


def test_a_window_the_trace_lost_is_the_records_own_error_and_the_drop_goes_on(tmp_path):
    w = W.Workspace(tmp_path)
    _two_segment_trace(w.state / "traces" / "card-1")
    sid = w.drop_surprise({"member": "m", "reading": "x", "trace": "traces/card-1", "window": [40, 50]})
    rec = w.surprise(sid)
    assert rec["trace"] == "traces/card-1" and "not in traces/card-1 any more" in rec["keep_error"]
    sid = w.drop_surprise({"member": "m", "reading": "x", "trace": "traces/nowhere", "window": [2, 3]})
    assert "keep_error" in w.surprise(sid) and w.surprise(sid)["trace"] == "traces/nowhere"
    sid = w.drop_surprise({"member": "m", "reading": "x", "trace": "traces/card-1", "k": 7})     # no window: as it is
    assert w.surprise(sid)["trace"] == "traces/card-1" and "keep_error" not in w.surprise(sid)


def test_the_loops_tools_do_not_get_kind_write():
    tools = {t["name"]: t for t in W.declarations()["tools"]}
    props = tools["trace_kind"]["parameters"]["properties"]
    assert "write" not in props and "record" in props and "program" in props


def test_a_drop_pointing_at_a_kept_copy_keeps_it_as_it_is(tmp_path):
    w = W.Workspace(tmp_path)
    _two_segment_trace(w.state / "traces" / "card-1")
    kept = w.keep_window("card-1", 6, 8, "w6-8")                   # the console's and the chat's drops keep first
    sid = w.drop_surprise({"member": "m", "reading": "x", "trace": kept, "window": [6, 8]})
    rec = w.surprise(sid)
    assert rec["trace"] == kept and rec["run"] == "card-1" and "keep_error" not in rec
    assert sorted(p.name for p in (w.state / "traces").iterdir()) == ["card-1", "card-1-w6-8"]
