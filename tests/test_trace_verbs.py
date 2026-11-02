"""metis.trace's instruments (metis-loop patterns/CHECKS.md §5): count, families (and the flicker), holds, diff, outcomes, fired, and blame's CLI,
each on a small synthetic trace: ten ticks of one member, its state moved by do-lines (a tick row carries no counts: the reader reconstructs them)."""
import json

import pytest

from metis import trace as T
from metis.lang.repl import Repl as Shell

PROGRAM = """
catalog small 1.
provenance text "a small program for the instruments' tests" "2026-11-02".
layers (dyn).
stages (main).
namespace small produce (dyn).
pred go : small.
pred l : small.
pred r : small.
stage main {
  %% The sampler picks one.
  left [dyn] : go -o l.
  %% Or the other.
  right [dyn] : go -o r.
}
#init go 1
#steps 20
#seeds 1
"""


@pytest.fixture
def program(tmp_path):
    p = tmp_path / "small.llp"
    p.write_text(PROGRAM)
    return p


def write(where, ticks, program=None, counts=None, members=("m",)):
    """A one-segment trace at WHERE: its checkpoint holds COUNTS (or the program's session's), then TICKS."""
    w = T.Segments(where, every=0)
    state = T.session_state(Shell(str(program), seed=1)) if program else {"stage": "main", "quiet": None, "rng": "1", "counts": counts or {}}
    head = {"members": {m: {"key": T.program_key(program) if program else "", "seed": 1, "weights": [], "out": [], "in": []} for m in members}, "commands": []}
    w.open_segment(0, head, {"sessions": {m: state for m in members}, "copies": {m: {} for m in members}})
    for t in ticks:
        w.line(t)
    w.close()
    return where


def tick(k, beat, fired, member="m"):
    return {"k": k, "members": {member: {"why": "change", "seq": [{"beat": list(beat), "fired": list(fired)}]}}}


# busy: held 1-3 (done appears at 4), 6-7 (cancel at 8), 9 to the end. speed(slow) 0-4, speed(fast) 5-9.
# red/green: red 0-2, green 3-4, both at 5, none at 6, green 7-9. p/q: p,q,p,q,... from 2 on. flip on even ticks, flop on odd.
TICKS = [tick(0, ["+red", "+speed(slow)"], ["flip"]),
         tick(1, ["+busy"], ["flop"]),
         tick(2, ["+p"], ["flip"]),
         tick(3, ["-red", "+green", "-p", "+q"], ["flop"]),
         tick(4, ["-busy", "+done", "-q", "+p"], ["flip", "finish"]),
         tick(5, ["+red", "-speed(slow)", "+speed(fast)", "-p", "+q", "-done"], ["flop"]),
         tick(6, ["-red", "-green", "+busy", "-q", "+p"], ["flip"]),
         tick(7, ["+green", "-p", "+q"], ["flop"]),
         tick(8, ["-busy", "+cancel", "-q", "+p"], ["flip"]),
         tick(9, ["+busy", "-p", "+q", "-cancel"], ["flop"])]


@pytest.fixture
def trace(tmp_path):
    return write(tmp_path / "t", TICKS)


def test_the_timeline_reconstructs_the_state_from_the_checkpoint_through_the_do_lines(trace):
    ticks, exact = T.timeline(trace, "m")
    assert not exact and [t.k for t in ticks] == list(range(10))
    assert ticks[0].counts == {"red": 1, "speed(slow)": 1} and ticks[4].counts == {"green": 1, "done": 1, "p": 1, "speed(slow)": 1}
    assert ticks[4].fired == ["flip", "finish"] and ticks[4].made == {"done", "p"}
    assert T.held("speed", ticks[5].counts) == ["speed(fast)"] and T.held("speed(slow)", ticks[5].counts) == []
    with pytest.raises(T.TraceError, match="no member"):
        T.timeline(trace, "nobody")


def test_count_is_the_cross_tab_of_an_atom_by_the_others(trace):
    t = T.count(trace, "busy", by=["speed"])
    assert t.notes[:2] == ["ticks: 10", "ticks with busy: 6"]
    assert t.header == ["speed", "with_busy", "without"]
    assert t.rows == [["speed(fast)", 3, 2], ["speed(slow)", 3, 2]]
    # an exact atom tabulates as holds / absent; no --by: one row
    assert T.count(trace, "busy", by=["speed(fast)"]).rows == [["absent", 3, 2], ["holds", 3, 2]]
    assert T.count(trace, "done", member="m").rows == [[1, 9]]


def test_families_say_exactly_one_per_tick_and_the_flicker(trace):
    summary, bad, flicker = T.families(trace, [["red", "green"], ["p", "q"]])
    assert summary.rows == [["red+green", ["red", "green"], 10, 8, 1, 1], ["p+q", ["p", "q"], 10, 8, 2, 0]]
    assert bad.rows == [["red+green", 5, ["red", "green"]], ["red+green", 6, "none"], ["p+q", 0, "none"], ["p+q", 1, "none"]]
    # p and q alternate from tick 2 to the end: a family toggle; flip and flop alternate over the whole trace: a rule pair, told once
    assert flicker.rows == [["family", "p+q", 2, 9, 8, "p <-> q"], ["rule", "flip", 0, 9, 10, "flip <-> flop"]]


def test_a_rule_toggling_alone_is_paired_with_idle(tmp_path):
    d = write(tmp_path / "t", [tick(k, [], ["r"] if k % 2 == 0 else []) for k in range(6)])
    assert T.families(d, [])[2].rows == [["rule", "r", 0, 5, 6, "r <-> idle"]]
    assert T._toggles(["a", "b", "a", "b"]) == [] and T._toggles(["a", "b", "c", "b", "c", "b", "c"]) == [(1, 6)]


def test_families_read_the_programs_declared_ones_when_it_has_them(trace, program, monkeypatch):
    real = T.open_session

    def with_families(p, seed=1):
        sh = real(p, seed)
        sh.doc.families = [("lights", ["red", "green"])]
        return sh
    monkeypatch.setattr(T, "open_session", with_families)
    summary = T.families(trace, [["p", "q"]], program=program)[0]
    assert [r[0] for r in summary.rows] == ["p+q", "lights"] and summary.rows[1][3:] == [8, 1, 1]


def test_holds_names_the_windows_past_the_bound_and_the_rules_within(trace):
    t = T.holds_windows(trace, "busy", 2)
    assert t.rows == [[1, 3, 3, ["flip", "flop"]]] and t.notes[1] == "windows of busy past 2 ticks: 1"
    assert [r[:3] for r in T.holds_windows(trace, "busy", 0).rows] == [[1, 3, 3], [6, 7, 2], [9, 9, 1]]


def test_outcomes_count_how_each_commitment_window_ended(trace):
    t = T.outcomes(trace, "busy", "done", "cancel")
    assert t.notes[:4] == ["windows of busy: 3", "done: 1", "cancel: 1", "neither: 1"]
    assert t.rows == [[1, 3, 3, "done"], [6, 7, 2, "cancel"], [9, 9, 1, "neither: the trace ended"]]
    # done appearing and consumed within the release tick still counts: it was made
    d = write(trace.parent / "t2", [tick(0, ["+busy"], []), tick(1, ["-busy", "+done", "-done"], ["x"]), tick(2, [], [])])
    assert T.outcomes(d, "busy", "done", "cancel").rows == [[0, 0, 1, "done"]]


def test_diff_names_the_first_tick_the_fired_rows_and_the_state_differ(tmp_path, trace):
    other = [json.loads(json.dumps(t)) for t in TICKS]
    other[5]["members"]["m"]["seq"][0]["fired"].append("extra")
    other[7]["members"]["m"]["seq"][0]["beat"].append("+z")
    b = write(tmp_path / "b", other)
    fired, state = T.diff(trace, b)
    assert fired.notes == ["fired differ first at tick 5 (m)"] and fired.rows == [["A", 5, "m", ["flop"]], ["B", 5, "m", ["flop", "extra"]]]
    assert state.notes == ["state differ first at tick 7 (m)"] and state.rows == [["A", 7, "m", []], ["B", 7, "m", ["z 1"]]]
    same = write(tmp_path / "c", TICKS[:8])
    fired, state = T.diff(trace, same, member="m")
    assert fired.notes == ["fired identical over 8 ticks (A has 10 ticks, B has 8)"] and state.rows == []


def test_fired_searches_the_seeds(tmp_path, program):
    d = write(tmp_path / "t", [tick(0, [], ["left"])], program=program)
    left, right = T.fired_on_seeds(d, "left", 12, program, "m"), T.fired_on_seeds(d, "right", 12, program, "m")
    assert [r[0] for r in left.rows] == list(range(1, 13))
    assert all((l[1], r[1]) in ((True, False), (False, True)) for l, r in zip(left.rows, right.rows))
    assert all(r[2] == (0 if r[1] else None) for r in left.rows)
    n = sum(1 for r in left.rows if r[1])
    assert 0 < n < 12 and left.notes == ["left fired on %d of 12 seeds" % n]
    with pytest.raises(T.TraceError, match="no member"):
        T.fired_on_seeds(d, "left", 2, program, "nobody")


def test_the_cli_prints_notes_a_header_and_rows(trace, tmp_path, capsys):
    assert T.main(["blame", str(trace), "--member", "m", "--k", "3", "--atoms", "red, green 0"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "verdict: not this package's" and lines[1].split() == ["whose", "atom", "want", "have", "last_k", "last_by", "last_what"]
    assert lines[2].split("  ")[0] == "the world's reading" and lines[2].split()[-6:] == ["red", "1", "0", "3", "world", "-red"]
    assert T.main(["count", str(trace), "busy", "--by", "speed", "--member", "m"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:2] == ["ticks: 10", "ticks with busy: 6"] and out[3].split() == ["speed", "with_busy", "without"] and out[4].split() == ["speed(fast)", "3", "2"]
    assert T.main(["families", str(trace), "--family", "red,green"]) == 0
    rows = [x.split() for x in capsys.readouterr().out.splitlines()]
    assert ["red+green", "red,green", "10", "8", "1", "1"] in rows and ["red+green", "5", "red,green"] in rows
    assert ["rule", "flip", "0", "9", "10", "flip", "<->", "flop"] in rows
    assert T.main(["holds", str(trace), "busy", "--max", "2"]) == 0
    assert ["1", "3", "3", "flip,flop"] in [x.split() for x in capsys.readouterr().out.splitlines()]
    assert T.main(["outcomes", str(trace), "--commit", "busy", "--done", "done", "--cancel", "cancel"]) == 0
    assert ["9", "9", "1", "neither:", "the", "trace", "ended"] in [x.split() for x in capsys.readouterr().out.splitlines()]
    assert T.main(["diff", str(trace), str(trace)]) == 0
    assert "fired identical over 10 ticks" in capsys.readouterr().out


def test_a_trace_of_several_members_needs_the_member_named(tmp_path):
    d = write(tmp_path / "t", [tick(0, ["+a"], [])], members=("m", "n"))
    with pytest.raises(T.TraceError, match="--member: the trace has m, n"):
        T.count(d, "a")
    assert T.count(d, "a", member="m").rows == [[1, 0]] and T.count(d, "a", member="n").rows == [[0, 1]]
