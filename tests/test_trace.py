"""metis.trace (trace-replay s-c111ff41): the ring on disk, the session's checkpoint, and the replay of one member over a
trace: its rules to quiescence, the checkpoints held, the first divergence named. The worlds' own traces are tested where
they run: metis-agent (the Live reader), metis-wheelchair (the goldens' fired column), metis-worlds (the road)."""
import json

import pytest

from metis import trace as T
from metis.kernel.ground import GroundBudget
from metis.lang.repl import Repl as Shell

PROGRAM = """
catalog small 1.
provenance text "a small program for the trace's tests" "2026-11-01".
layers (dyn).
stages (main).
namespace small produce (dyn).
pred a : small.
pred b : small.
pred c : small.
pred out_b : small.
stage main {
  %% A becomes b, and says so.
  a_to_b [dyn] : a -o b * out_b.
  %% B becomes c.
  b_to_c [dyn] : b -o c.
}
#init a 1
#steps 20
#seeds 1
"""


BIG = """
catalog big 1.
provenance text "a rule that ranges over a whole type" "2026-11-01".
layers (dyn).
stages (main).
namespace big produce (dyn).
type t {a b c d e f g h}.
pred on(t, t) : big.
pred go : big.
stage main {
  %% It reads nothing that binds its variables.
  spread [dyn] : go * on(X, Y) -o go.
}
#init go 1
#steps 2
#seeds 1
"""


@pytest.fixture
def program(tmp_path):
    p = tmp_path / "small.llp"
    p.write_text(PROGRAM)
    return p


def write(tmp_path, program, ticks, every=0):
    w = T.Segments(tmp_path / "t", every=every)
    sh = Shell(str(program), seed=1)
    head = {"members": {"m": {"key": T.program_key(program), "seed": 1, "weights": [], "out": ["out_b"]}}, "commands": []}
    w.open_segment(0, head, {"sessions": {"m": T.session_state(sh)}, "world": {"step": 0}})
    for t in ticks:
        w.line(t)
    w.close()
    return tmp_path / "t"


def tick(k, *seq, out=None):
    e = {"why": "change", "seq": list(seq)}
    if out:
        e["out"] = out
    return {"k": k, "members": {"m": e}}


GOOD = [tick(0, {"beat": [], "fired": ["a_to_b", "b_to_c"]}, out=["out_b"]),
        tick(1, {"take": ["+a"]}),                           # a take fires nothing until the next beat
        tick(2, {"beat": [], "fired": ["a_to_b", "b_to_c"]}, out=["out_b"])]


def test_a_member_replays_to_quiescence_and_a_take_waits_for_its_beat(tmp_path, program):
    r = T.replay(write(tmp_path, program, GOOD), "m", program)
    assert r.ok, r.divergence
    assert r.fired == {0: ["a_to_b", "b_to_c"], 2: ["a_to_b", "b_to_c"]} and r.rules == 4 and r.beats == 2
    assert r.out == {0: ["out_b"], 2: ["out_b"]}


def test_a_beat_that_fired_nothing_must_fire_nothing(tmp_path, program):
    r = T.replay(write(tmp_path, program, [{"k": 0, "members": {"m": {"why": "clock"}}}]), "m", program)
    assert not r.ok and r.divergence["k"] == 0 and r.divergence["what"] == "fired" and r.divergence["got"] == ["a_to_b", "b_to_c"]


def test_the_first_divergence_is_named(tmp_path, program):
    bad = [tick(0, {"beat": [], "fired": ["a_to_b"]})]
    r = T.replay(write(tmp_path, program, bad), "m", program)
    assert r.divergence == {"k": 0, "what": "fired", "want": ["a_to_b"], "got": ["a_to_b", "b_to_c"]}


def test_a_beat_left_with_rules_enabled_replays_only_what_it_fired(tmp_path, program):
    left = [tick(0, {"beat": [], "fired": ["a_to_b"], "left": True}, out=["out_b"])]
    assert T.replay(write(tmp_path, program, left), "m", program).ok


def test_an_output_that_differs_is_a_divergence(tmp_path, program):
    r = T.replay(write(tmp_path, program, [tick(0, {"beat": [], "fired": ["a_to_b", "b_to_c"]})]), "m", program)
    assert r.divergence["what"] == "out" and r.divergence["got"] == ["out_b"]


def test_another_program_is_refused_unless_asked(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    other = tmp_path / "other.llp"
    other.write_text(PROGRAM + "\n% another\n")
    with pytest.raises(T.TraceError, match="not the program"):
        T.replay(d, "m", other)
    assert T.replay(d, "m", other, check_key=False).ok


def test_a_member_with_weight_ports_is_refused_in_words(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    f = d / "seg-0.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["members"]["m"]["weights"] = ["look"]
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(T.TraceError, match="weight ports"):
        T.replay(d, "m", program)


def test_the_session_round_trips_and_the_rng_is_a_string(program):
    a = Shell(str(program), seed=7)
    a._sample()
    st = T.session_state(a)
    assert isinstance(st["rng"], str) and int(st["rng"]) == a.rng.state
    json.dumps(st)
    b = Shell(str(program), seed=1)
    T.restore_session(b, st)
    assert T.session_state(b) == st and b._sample() == a._sample()


def test_the_next_checkpoint_must_hold(tmp_path, program):
    w = T.Segments(tmp_path / "t", every=1)
    sh = Shell(str(program), seed=1)
    head = {"members": {"m": {"seed": 1, "weights": [], "out": ["out_b"]}}, "commands": []}
    w.open_segment(0, head, {"sessions": {"m": T.session_state(sh)}})
    w.line(tick(0, {"take": ["+c"]}))
    wrong = T.session_state(sh)                        # the checkpoint without the +c the take applied
    w.open_segment(1, head, {"sessions": {"m": wrong}})
    w.close()
    r = T.replay(tmp_path / "t", "m", program)
    assert r.divergence["what"] == "checkpoint counts" and r.divergence["k"] == 1


def test_the_ring_keeps_at_least_its_ticks_and_each_segment_its_checkpoint(tmp_path):
    w = T.Segments(tmp_path / "t", every=10, ring=25)
    for k in range(100):
        if w.due(k):
            w.open_segment(k, {"members": {}}, {"sessions": {}})
        w.line({"k": k, "members": {}})
    w.close()
    segs = T.read(tmp_path / "t")
    ks = [t["k"] for s in segs for t in s.ticks]
    assert ks == list(range(ks[0], 100)) and len(ks) >= 25 and len(ks) < 45
    assert sorted(p.name for p in (tmp_path / "t" / "checkpoints").iterdir()) == sorted("k%d.json" % s.k0 for s in segs)


def test_ring_zero_keeps_it_all(tmp_path):
    w = T.Segments(tmp_path / "t", every=10, ring=0)
    for k in range(35):
        if w.due(k):
            w.open_segment(k, {"members": {}}, {"sessions": {}})
        w.line({"k": k, "members": {}})
    w.close()
    assert [s.k0 for s in T.read(tmp_path / "t")] == [0, 10, 20, 30]


def test_another_protocol_does_not_replay(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    f = d / "seg-0.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["protocol"] = "not-this-one"
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    with pytest.raises(T.TraceError, match="across a protocol change"):
        T.replay(d, "m", program)
    rows[0]["protocol"] = T.protocol_key()
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert T.replay(d, "m", program).ok


def test_a_session_past_the_budget_is_refused_in_the_rules_words(tmp_path, monkeypatch):
    p = tmp_path / "big.llp"
    p.write_text(BIG)
    monkeypatch.delenv("METIS_GROUND_BUDGET", raising=False)
    monkeypatch.setattr(T, "GROUND_BUDGET", 10)
    with pytest.raises(GroundBudget, match="ground-budget: spread"):
        T.open_session(p)
    assert "METIS_GROUND_BUDGET" not in __import__("os").environ
    monkeypatch.setattr(T, "GROUND_BUDGET", 1000)
    assert T.open_session(p)


def test_a_kept_trace_goes_on_after_what_is_there(tmp_path):
    w = T.Segments(tmp_path / "t", every=0)
    w.open_segment(0, {"members": {}}, {"sessions": {}})
    w.line({"k": 0})
    w.line({"k": 1})
    w.close()
    w = T.Segments(tmp_path / "t", every=0, keep=True)
    assert w.last_k() == 1
    w.open_segment(2, {"members": {}}, {"sessions": {}})
    w.line({"k": 2})
    w.close()
    assert [t["k"] for s in T.read(tmp_path / "t") for t in s.ticks] == [0, 1, 2]
    assert T.Segments(tmp_path / "t").k0s == [] and not list((tmp_path / "t").glob("seg-*"))


def test_an_acting_episode_replays_by_its_seed_and_lines(program):
    start = {"stage": "main", "counts": {"a": 1}}
    import random
    sh = Shell(str(program), seed=1)                   # the loop's own choosing: which of two enabled it takes is the rng's
    sh.stage, sh.counts, rng, names = "main", {"a": 1}, random.Random(5), []
    from metis.fw.plugin import exo_event
    lines = [["+a"], [], [], []]
    while sh._candidates():
        ws = sh._candidates()
        names.append(sh._fire(rng.choices([i for i, _ in ws], weights=[x for _, x in ws])[0], ws=ws))
        if lines[len(names) - 1]:
            _, sh.counts = exo_event(lines[len(names) - 1], sh.counts)
    firings = [{"name": n, "lines": lines[i]} for i, n in enumerate(names)]
    r = T.replay_firings(program, start, 5, firings)
    assert r.ok and r.rules == len(names) >= 3
    bad = [dict(firings[0], name="b_to_c")] + firings[1:]
    assert T.replay_firings(program, start, 5, bad).divergence["k"] == 0
    assert T.replay_firings(program, {"stage": "main", "counts": {}}, 5, firings).divergence["got"] == []


def test_no_segment_is_said(tmp_path):
    (tmp_path / "t").mkdir()
    with pytest.raises(T.TraceError, match="no segment"):
        T.read(tmp_path / "t")
