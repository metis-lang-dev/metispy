"""A member whose trace header carries weight ports (alfworld's task, `weight look`): the trace does not carry the ports'
values, so every verb opens the member with each port stood in at STAND_IN (strictly inside (0, 1)), the replay draws the
record's choices where the recorded rule is enabled, and each verb says so in one line (loop-prompt s-3591d53b step 3)."""
import json

from metis import trace as T
from metis.fw.registry import Registry
from metis.lang.repl import Repl as Shell

WEIGHTED = """
catalog picky 1.
provenance text "a weight port for the trace verbs" "2026-11-04".
layers (dyn).
stages (main).
type place {door sofa}.
namespace g produce (dyn).
pred want(place) : g.
pred at(place) : g.
pred ready : g.
weight pick(P) reads {want(P)}.
stage main {
  %% I go to {P}, or I stay.
  go [dyn] : want(P) -o (at(P) @w pick(P) | ready @w ~pick(P)).
}
#steps 20
#seeds 1
"""
NOTE = "pick stood in at 0.5: this member's choices are the record's, not evidence"


def session(tmp_path, weights=("pick",), live="metis-loop-console", second=True):
    """A one-member trace of PICKY: a take of want(door) at 1, then the record's choice at 2 (the second clause, ready,
    when SECOND: the one a stand-in of 1.0 would never enable)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "picky.llp"
    p.write_text(WEIGHTED)
    sh = Shell(str(p), seed=1, registry=Registry().weight("pick", T.STAND_IN))
    names = sorted({sh.events_tbl[i][2].name for i in range(len(sh.events_tbl)) if "door" in sh.events_tbl[i][2].name})
    chose = names[-1] if second else names[0]
    d = tmp_path / "t"
    w = T.Segments(d, every=0)
    head = {"members": {"m": {"key": T.program_key(p), "program": str(p), "seed": 1, "weights": list(weights), "out": []}},
            "commands": [], **({"live": live, "clock": "command"} if live else {})}
    w.open_segment(0, head, {"sessions": {"m": T.session_state(sh)}, "copies": {"m": {}}})
    w.line({"k": 1, "members": {"m": {"why": "command", "seq": [{"take": ["+want(door)"], "by": "person"}]}}})
    w.line({"k": 2, "members": {"m": {"why": "command", "seq": [{"step": [chose], "by": "tool"}]}}})
    w.close()
    return d, p, chose


def test_the_stand_in_is_strictly_inside_zero_one():
    assert 0 < T.STAND_IN < 1
    assert T.stood_in_note(["pick"]) == NOTE and T.stood_in_note([]) is None


def test_replay_stands_the_port_in_and_draws_the_records_choice(tmp_path):
    for second in (True, False):
        d, p, chose = session(tmp_path / str(second), second=second)
        r = T.replay(d, "m", p)
        assert r.ok, (chose, r.divergence)
        assert r.fired == {2: [chose]} and r.note == NOTE


def test_replay_says_where_the_record_left_the_program(tmp_path):
    """A recorded rule the program does not enable is the member's own draw, and the replay names the divergence."""
    d, p, chose = session(tmp_path)
    q = tmp_path / "cand.llp"
    q.write_text(WEIGHTED.replace("go [dyn] : want(P) -o", "go [dyn] : want(P) * $ready -o"))
    r = T.replay(d, "m", q, check_key=False)
    assert r.divergence["k"] == 2 and r.divergence["what"] == "step" and r.divergence["want"] == [chose], r.divergence


def test_the_cli_replay_prints_the_note(tmp_path, capsys):
    d, p, _ = session(tmp_path)
    assert T.main(["replay", str(d), "m", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["note"] == NOTE


def test_blame_opens_the_member_and_says_so(tmp_path):
    d, p, chose = session(tmp_path)
    r = T.blame(d, "m", 2, ["ready"])
    assert r["note"] == NOTE
    row = r["atoms"][0]
    assert row["have"] == 1 and row["last"] == {"k": 2, "by": "rule", "what": chose}
    t = T.blame_rows(d, "m", 2, ["ready"])
    assert NOTE in t.notes


def test_the_timeline_reads_the_rules_effects(tmp_path):
    """/replay's verb: the session's program from its header, opened stood in, so the rule's made atom is seen."""
    d, p, chose = session(tmp_path)
    ticks, exact = T.timeline(d, "m")
    assert exact and [t.made for t in ticks if t.k == 2] == [{"ready"}]
    assert T.trace_note(d, "m") == [NOTE]


def test_count_holds_outcomes_families_and_diff_carry_the_note(tmp_path):
    d, p, _ = session(tmp_path)
    assert NOTE in T.count(d, "ready").notes
    assert NOTE in T.holds_windows(d, "ready", 0).notes
    assert NOTE in T.outcomes(d, "want", "ready", "at").notes
    assert NOTE in T.families(d, [["ready", "want"]])[0].notes
    assert "m: " + NOTE in T.diff(d, d)[0].notes


def test_fired_searches_the_seeds_stood_in(tmp_path):
    d, p, chose = session(tmp_path)
    t = T.fired_on_seeds(d, "go", 2, p, "m")
    assert [r[0] for r in t.rows] == [1, 2] and NOTE in t.notes


def test_a_member_without_weight_ports_says_nothing(tmp_path):
    d, p, _ = session(tmp_path, weights=(), live=None)        # a composition's trace without --program opens no session
    assert T.trace_note(d, "m") == [] and "note" not in T.blame(d, "m", 2, ["ready"])
    assert all("stood in" not in n for n in T.count(d, "ready").notes)
