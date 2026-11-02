"""metis.trace (trace-replay s-c111ff41): the ring on disk, the session's checkpoint, and the replay of one member over a
trace: its rules to quiescence, the checkpoints held, the first divergence named. The worlds' own traces are tested where
they run: metis-agent (the Live reader), metis-wheelchair (the goldens' fired column), metis-worlds (the road)."""
import json
from pathlib import Path

import pytest
import yaml

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


def test_a_command_says_its_kind_and_args_the_hosts_words_over_the_rules(tmp_path, program):
    sh = Shell(str(program), seed=1)
    ev = next(ev for _, _, ev in sh.events_tbl if ev.name == "a_to_b")
    assert T.outputs(ev, ["a_to_b"], {"out_b"}) == [{"rule": "a_to_b", "kind": "a_to_b", "args": []}, "out_b"]
    assert T.outputs(ev, ["a_to_b"], (), lambda name: ("brake", "brake_min")) == [{"rule": "a_to_b", "kind": "brake", "args": ["brake_min"]}]
    assert T.outputs(ev, ["a_to_b"], (), lambda name: ("keep", None))[0]["args"] == []
    assert T.command_entry("reach[D=d0,E=e1]") == {"rule": "reach[D=d0,E=e1]", "kind": "reach", "args": ["d0", "e1"]}
    # the replay compares the rule: the host's words are not the model's
    d = write(tmp_path, program, [{"k": 0, "members": {"m": {"why": "change", "seq": [{"beat": [], "fired": ["a_to_b", "b_to_c"]}], "out": ["out_b"]}}}])
    f = d / "seg-0.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["commands"] = ["a_to_b"]
    rows[1]["members"]["m"]["out"] = [{"rule": "a_to_b", "kind": "brake", "args": []}, "out_b"]
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    assert T.replay(d, "m", program).ok


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


MANIFEST = {"shared": {"a": "world-private", "b": "in", "c": "world-private", "out_b": "out"}, "status": {"success": [], "failure": []}, "gating": "program"}


def as_case(tmp_path, text):
    f = tmp_path / "surprise.yaml"
    f.write_text(text)
    return f


def run_it(program, f, **kw):
    from metis.catalog import stub
    return stub.run_case((program, MANIFEST), f, **kw)


def test_a_surprise_is_a_case_that_fails_until_a_rule_passes_it(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    # the world left two b's: the program's b_to_c takes them on to c
    f = as_case(tmp_path, T.case(d, "m", 2, after=["b 2"], note="the world kept b"))
    r = run_it(program, f)
    assert not r.passed and "after: b is 0, the world made it 2" in r.boundaries[-1].failures
    better = tmp_path / "better.llp"
    better.write_text(PROGRAM.replace("b_to_c [dyn] : b -o c.", "b_to_c [dyn] : c -o b."))
    assert run_it(better, f).passed


def test_a_case_names_no_rule_and_carries_its_window(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    c = yaml.safe_load(T.case(d, "m", 2, after=["c 2"]))
    assert "fired" not in json.dumps(c) and c["rng"] and c["stage"] == "main" and c["init"] == ["a"]
    assert [b.get("run", True) for b in c["boundaries"]] == [True, False, True]
    assert c["boundaries"][-1]["expect"] == {"after": ["c 2"]} and c["boundaries"][1]["lines"] == ["+a"]
    with pytest.raises(T.TraceError, match="asserts something"):
        T.case(d, "m", 2)


def test_a_case_may_assert_the_command_kind_a_norm_requires(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    ok = as_case(tmp_path, T.case(d, "m", 2, out=[{"kind": "a_to_*"}, {"not": "keep*"}]))
    assert run_it(program, ok).passed
    bad = as_case(tmp_path, T.case(d, "m", 2, out=[{"kind": "yield_*"}]))
    assert "no command of kind yield_*" in " ".join(x for b in run_it(program, bad).boundaries for x in b.failures)
    no = as_case(tmp_path, T.case(d, "m", 2, out=[{"not": "a_to*"}]))
    assert "the norm forbids" in " ".join(x for b in run_it(program, no).boundaries for x in b.failures)


def test_a_cases_start_holds_the_rng_and_the_stage(tmp_path, program):
    from metis.catalog import stub
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump({"rng": "12345", "stage": "main", "init": ["a"], "boundaries": [{"expect": {"after": ["c"]}}]}))
    assert stub.load_case(f)["rng"] == "12345" and run_it(program, f).passed
    assert not stub.load_case(f).get("nope")


def test_blame_says_whose_is_what_is_missing(tmp_path, program):
    ticks = [tick(0, {"beat": [], "fired": ["a_to_b", "b_to_c"]}, out=["out_b"]),
             tick(1, {"take": ["+b"], "by": "relay"}),
             tick(2, {"beat": ["+a"], "fired": ["a_to_b", "b_to_c", "b_to_c"]}, out=["out_b"]),
             tick(3, {"take": ["-c 3"], "by": "relay"})]
    d = write(tmp_path, program, ticks)
    f = d / "seg-0.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["members"]["m"]["in"] = ["b"]
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    # gone because a take took it: not this package's
    r = T.blame(d, "m", 3, ["c"], program)
    assert r["verdict"] == "not this package's" and r["atoms"][0]["last"] == {"k": 3, "by": "relay", "what": "-c 3"}
    # a rule of the member made it, the world did not want it: this package's
    r = T.blame(d, "m", 2, ["c 0"], program)
    assert r["verdict"] == "this package's" and r["atoms"][0]["last"]["by"] == "rule" and r["atoms"][0]["have"] == 3
    # holding as it should: no verdict against anyone
    assert T.blame(d, "m", 2, ["c 3"], program)["verdict"] == "this package's"
    # on an in-port and never there at the start: never delivered
    r = T.blame(d, "m", 0, ["b 5"], program)
    assert r["verdict"] == "this package's" and r["atoms"][0]["last"]["by"] == "rule"


def test_an_in_port_never_delivered_is_not_this_packages(tmp_path, program):
    d = write(tmp_path, program, [{"k": 0, "members": {"m": {"why": "clock"}}}])
    f = d / "seg-0.jsonl"
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    rows[0]["members"]["m"]["in"] = ["tick__grant"]
    f.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    r = T.blame(d, "m", 0, ["tick__grant(m,r)"], program)
    assert r["verdict"] == "not this package's" and r["atoms"][0]["last"] is None


def test_a_stale_world_reading_is_not_this_packages(tmp_path, program):
    """A projection's lie: nobody's rule left it so. Unmoved in the whole trace, or moved by the world's own line."""
    w = T.Segments(tmp_path / "t", every=0)
    sh = Shell(str(program), seed=1)
    w.open_segment(0, {"members": {"m": {"seed": 1, "weights": [], "out": ["out_b"], "in": []}}}, {"sessions": {"m": T.session_state(sh)}, "copies": {"m": {"held(x)": 1}}})
    w.line(tick(0, {"beat": [], "fired": ["a_to_b", "b_to_c"]}, out=["out_b"]))
    w.line({"k": 1, "members": {"m": {"why": "change", "seq": [{"beat": ["+seen(y)"], "fired": []}], "copy": {"+": ["seen(y)"], "-": []}}}})
    w.close()
    r = T.blame(tmp_path / "t", "m", 1, ["held(z)"], program)
    assert r["verdict"] == "not this package's" and r["atoms"][0]["last"] is None and r["atoms"][0]["whose"] == "the world's reading"
    r = T.blame(tmp_path / "t", "m", 1, ["seen(y) 0"], program)
    assert r["verdict"] == "not this package's" and r["atoms"][0]["last"]["by"] == "world" and r["atoms"][0]["whose"] == "the world's reading"
    # the member's own token, nothing ever moved it: its own
    r = T.blame(tmp_path / "t", "m", 1, ["never_made"], program)
    assert r["verdict"] == "this package's" and r["atoms"][0]["whose"] == "the member's: nothing ever moved it"
    # a rule left it so: its own, whatever else is wrong
    r = T.blame(tmp_path / "t", "m", 1, ["c 0"], program)
    assert r["verdict"] == "this package's" and r["atoms"][0]["whose"] == "the member's rule"
    # what holds as it should: nobody's
    r = T.blame(tmp_path / "t", "m", 1, ["c"], program)
    assert r["verdict"] == "this package's" and r["atoms"][0]["whose"] is None


def test_blame_reads_the_whole_trace_not_only_the_last_segment(tmp_path, program):
    w = T.Segments(tmp_path / "t", every=2)
    sh = Shell(str(program), seed=1)
    head = {"members": {"m": {"seed": 1, "weights": [], "out": [], "in": []}}}
    w.open_segment(0, head, {"sessions": {"m": T.session_state(sh)}, "copies": {"m": {}}})
    w.line({"k": 0, "members": {"m": {"why": "change", "seq": [{"beat": ["-a"], "fired": []}]}}})
    w.line({"k": 1, "members": {}})
    w.open_segment(2, head, {"sessions": {"m": dict(T.session_state(sh), counts={})}, "copies": {"m": {}}})
    w.line({"k": 2, "members": {}})
    w.close()
    r = T.blame(tmp_path / "t", "m", 2, ["a"], program)
    assert r["atoms"][0]["last"] == {"k": 0, "by": "world", "what": "-a"} and r["verdict"] == "not this package's"


def test_goldens_name_the_case_that_refuses(tmp_path, program):
    from metis.catalog import manifests
    pkg = tmp_path / "root" / "domains" / "small"
    (pkg / "cases").mkdir(parents=True)
    (pkg / "small.llp").write_text(PROGRAM)
    (pkg / "domain.yaml").write_text("program: small.llp\ndisplay: small\nline: a small program\n")
    (pkg / "cases" / "good.yaml").write_text(yaml.safe_dump({"case": "good", "boundaries": [{"expect": {"after": ["c"]}}]}))
    d = next(x for x in manifests.domains(tmp_path / "root") if x.name == "small")
    assert T.goldens(d) == (True, [])
    (pkg / "cases" / "bad.yaml").write_text(yaml.safe_dump({"case": "the golden that says no", "boundaries": [{"expect": {"after": ["a"]}}]}))
    ok, failed = T.goldens(d, seeds=(1, 2))
    assert not ok and [(x["case"], x["file"], x["seed"]) for x in failed] == [("the golden that says no", "bad.yaml", 1), ("the golden that says no", "bad.yaml", 2)]
    assert "after: a is 0" in failed[0]["failures"][0]


def test_a_consumer_runs_its_familys_cases_too(tmp_path, monkeypatch):
    from metis.catalog import manifests
    pkg = tmp_path / "root" / "domains" / "small"
    pkg.mkdir(parents=True)
    (pkg / "small.llp").write_text(PROGRAM)
    (pkg / "domain.yaml").write_text("program: small.llp\ndisplay: small\nline: a small program\n")
    fam = tmp_path / "family" / "small" / "cases"
    fam.mkdir(parents=True)
    (fam / "from_the_family.yaml").write_text(yaml.safe_dump({"case": "the family's", "boundaries": [{"expect": {"after": ["a"]}}]}))
    d = next(x for x in manifests.domains(tmp_path / "root") if x.name == "small")
    monkeypatch.delenv("METIS_HOME", raising=False)
    assert T.golden_files(d) == [] and T.goldens(d) == (True, [])
    monkeypatch.setenv("METIS_HOME", str(tmp_path / "family"))
    assert [f.name for f in T.golden_files(d)] == ["from_the_family.yaml"]
    assert [x["file"] for x in T.goldens(d, seeds=(1,))[1]] == ["from_the_family.yaml"]


def family_setup(tmp_path, monkeypatch, cases):
    from metis.catalog import manifests
    pkg = tmp_path / "root" / "domains" / "small"
    pkg.mkdir(parents=True)
    (pkg / "small.llp").write_text(PROGRAM)
    (pkg / "domain.yaml").write_text("program: small.llp\ndisplay: small\nline: a small program\n")
    fam = tmp_path / "family" / "small" / "cases"
    fam.mkdir(parents=True)
    for name, c in cases.items():
        (fam / name).write_text(yaml.safe_dump(c))
    monkeypatch.setenv("METIS_HOME", str(tmp_path / "family"))
    return next(x for x in manifests.domains(tmp_path / "root") if x.name == "small")


def test_a_family_case_is_keyed_to_the_rule_it_expects(tmp_path, monkeypatch):
    d = family_setup(tmp_path, monkeypatch, {
        "of_a_rule_it_lacks.yaml": {"boundaries": [{"expect": {"fired": ["change"]}}]},
        "of_a_kind_it_lacks.yaml": {"boundaries": [{"expect": {"out": [{"kind": "yield_*"}]}}]},
        "of_a_unit_it_lacks.yaml": {"units": ["change.llp"], "boundaries": [{"expect": {"after": ["nothing_here"]}}]},
        "its_own_but_wrong.yaml": {"boundaries": [{"expect": {"fired": ["a_to_b"]}}]},
        "its_own_and_right.yaml": {"boundaries": [{"expect": {"after": ["c"]}}]}})
    skips = T.family_skips(d)
    assert [x["file"] for x in skips] == ["of_a_kind_it_lacks.yaml", "of_a_rule_it_lacks.yaml", "of_a_unit_it_lacks.yaml"]
    assert "change, which this program does not have" in skips[1]["why"] and "change.llp, which this program does not include" in skips[2]["why"]
    ok, failed = T.goldens(d, seeds=(1,))
    assert [x["file"] for x in failed] == ["its_own_but_wrong.yaml"], "skipped, not failing; and a rule it has that does not fire so is a failure"
    assert {f.name for f in T.golden_files(d, [x["file"] for x in skips])} == {"its_own_but_wrong.yaml", "its_own_and_right.yaml"}


def test_a_unit_it_includes_is_its_case(tmp_path, monkeypatch):
    d = family_setup(tmp_path, monkeypatch, {"c.yaml": {"units": ["tick_types.llp"], "boundaries": [{"expect": {"after": ["c"]}}]}})
    (tmp_path / "family" / "tick_types.llp").write_text('catalog tick_types 1.\nprovenance text "nothing here" "2026-11-01".\n')
    Path(d.program).write_text(PROGRAM.replace("layers (dyn).", 'include "tick_types.llp".\nlayers (dyn).'))
    assert T.family_skips(d) == [] and T.goldens(d, seeds=(1,)) == (True, [])


def test_a_packages_own_case_naming_a_rule_it_lacks_is_a_failure_not_a_skip(tmp_path, monkeypatch):
    d = family_setup(tmp_path, monkeypatch, {})
    (Path(d.dir) / "cases").mkdir()
    (Path(d.dir) / "cases" / "mine.yaml").write_text(yaml.safe_dump({"boundaries": [{"expect": {"fired": ["gone"]}}]}))
    assert T.family_skips(d) == [] and [x["file"] for x in T.goldens(d, seeds=(1,))[1]] == ["mine.yaml"]


def test_a_candidate_that_drops_a_rule_is_not_excused_by_the_skip(tmp_path, monkeypatch):
    """The skip is read on the program as it is, and handed to the candidate's run: a rule the pack takes out still has its case."""
    d = family_setup(tmp_path, monkeypatch, {"wants_b_to_c.yaml": {"boundaries": [{"expect": {"fired": ["a_to_b", "b_to_c"]}}]}})
    skip = [x["file"] for x in T.family_skips(d)]
    assert skip == [] and T.goldens(d, seeds=(1,), skip=skip) == (True, [])
    Path(d.program).write_text(T.apply_pack(PROGRAM, "drop b_to_c\n")[0])
    assert [x["file"] for x in T.goldens(d, seeds=(1,), skip=skip)[1]] == ["wants_b_to_c.yaml"]
    assert [x["file"] for x in T.family_skips(d)] == ["wants_b_to_c.yaml"], "read now it would be excused: which is why it is read before"


def test_a_case_carries_its_units(tmp_path, program):
    d = write(tmp_path, program, GOOD)
    assert yaml.safe_load(T.case(d, "m", 2, after=["a"], units=["signals.llp"]))["units"] == ["signals.llp"]
    assert "units" not in yaml.safe_load(T.case(d, "m", 2, after=["a"]))


def test_new_rules_are_those_added_or_changed():
    other = PROGRAM.replace("b_to_c [dyn] : b -o c.", "b_to_c [dyn] : b -o c * a.\n  %% New.\n  c_to_a [dyn] : c -o a.")
    assert T.new_rules(other, PROGRAM) == ["b_to_c", "c_to_a"] and T.new_rules(PROGRAM, PROGRAM) == []


def test_the_cli_writes_the_case_and_replays(tmp_path, program, capsys):
    d = write(tmp_path, program, GOOD)
    assert T.main(["replay", str(d), "m", str(program)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert T.main(["case", str(d), "m", "--at", "2", "--after", "a", "--out", "kind:yield_*", "--into", str(tmp_path / "cases")]) == 0
    f = tmp_path / "cases" / "surprise-m-2.yaml"
    c = yaml.safe_load(f.read_text())
    assert f.is_file() and c["boundaries"][-1]["expect"] == {"after": ["a"], "out": [{"kind": "yield_*"}]}


def test_a_pack_is_appended_and_its_drops_taken_out():
    pack = "drop b_to_c\npred d : small.\nstage main {\n  %% C becomes d.\n  c_to_d [dyn] : c -o d.\n}\n"
    text, info = T.apply_pack(PROGRAM, pack)
    assert info == {"whole": False, "added": ["c_to_d"], "dropped": ["b_to_c"]} and "B becomes c" not in text and "c_to_d" in text and "a_to_b" in text
    assert sorted(T.rules_of(text)) == ["a_to_b", "c_to_d"]
    whole, info = T.apply_pack(PROGRAM, PROGRAM.replace("b_to_c", "b_on"))
    assert info["whole"] and info["dropped"] == ["b_to_c"] and "b_on" in whole
    with pytest.raises(T.TraceError, match="drops nope"):
        T.apply_pack(PROGRAM, "drop nope\n")


def test_a_pack_that_repeats_a_rule_is_found_by_name_or_by_its_body():
    same = "stage main {\n  %% Again.\n  b_to_c [dyn] : b -o c.\n}\n"
    assert T.duplicates(same, {"small.llp": PROGRAM}) == [{"rule": "b_to_c", "where": "small.llp", "other": "b_to_c", "how": "the same rule"}]
    renamed = "stage main {\n  %% Again.\n  b_again [dyn] : b -o c.\n}\n"
    assert T.duplicates(renamed, {"small.llp": PROGRAM})[0] == {"rule": "b_again", "where": "small.llp", "other": "b_to_c", "how": "the same premises and effects"}
    assert T.duplicates("stage main {\n  %% New.\n  c_to_a [dyn] : c -o a.\n}\n", {"small.llp": PROGRAM}) == []


def test_a_pack_that_names_a_rule_again_with_another_body_is_the_compilers_refusal():
    """What duplicates leaves to the compiler, the compiler refuses; a pack that drops the rule first may take its name."""
    from metis.lang import parse
    from metis.lang.compiler import LangError, resolve_includes
    again = "stage main {\n  %% Another body.\n  b_to_c [dyn] : c -o a.\n}\n"
    assert T.duplicates(again, {"small.llp": PROGRAM}) == []
    text, _ = T.apply_pack(PROGRAM, again)
    with pytest.raises(LangError, match="rule b_to_c is named twice in stage main"):
        resolve_includes(parse(text), ".")
    text, _ = T.apply_pack(PROGRAM, "drop b_to_c\n" + again)
    resolve_includes(parse(text), ".")


def test_an_expectation_holds_on_a_replays_session():
    assert T.holds({"after": ["a", "b 0"]}, {"a": 1}, []) == []
    assert "after: b is 1" in " ".join(T.holds({"after": ["b 0"]}, {"b": 1}, []))
    assert T.holds({"out": [{"kind": "yield_*"}]}, {}, ["yield_red[Z=z1]"]) == []
    assert "no command of kind accel*" in " ".join(T.holds({"out": [{"kind": "accel*"}]}, {}, ["yield_red[Z=z1]"]))


def test_a_deadline_fires_in_the_body_and_keeps_the_outer_one():
    import time
    with pytest.raises(T.OutOfTime, match="spin: past its budget"):
        with T.deadline(0.05, "spin"):
            time.sleep(2)
    with T.deadline(5, "outer"):
        with pytest.raises(T.OutOfTime):
            with T.deadline(0.05, "inner"):
                time.sleep(2)
        import signal
        left = signal.getitimer(signal.ITIMER_REAL)[0]
        assert 0 < left <= 5, "the outer goes on counting"
    assert __import__("signal").getitimer(__import__("signal").ITIMER_REAL)[0] == 0
    with T.deadline(None):
        pass


def test_a_golden_past_its_budget_is_not_judged_not_failed(tmp_path, monkeypatch):
    import time
    from metis.catalog import stub
    d = family_setup(tmp_path, monkeypatch, {"slow.yaml": {"case": "the slow one", "boundaries": [{"expect": {"after": ["c"]}}]}})
    monkeypatch.setattr(stub, "run_case", lambda *a, **k: time.sleep(5))
    ok, failed = T.goldens(d, seeds=(1, 2), budget=0.05)
    assert not ok and [(x["file"], x["seed"], x.get("not_judged")) for x in failed] == [("slow.yaml", 1, True), ("slow.yaml", 2, True)]
    assert "past its budget of 0.05" in failed[0]["failures"][0]


def test_one_session_for_the_goldens_says_what_a_fresh_one_says(tmp_path, monkeypatch):
    from metis.catalog import stub
    d = family_setup(tmp_path, monkeypatch, {
        "good.yaml": {"boundaries": [{"expect": {"after": ["c"]}}]},
        "bad.yaml": {"boundaries": [{"expect": {"after": ["a"]}}]},
        "order.yaml": {"init": ["a"], "boundaries": [{"expect": {"fired": ["a_to_b", "b_to_c"]}}, {"lines": ["+a"], "run": False}, {"expect": {"fired": ["a_to_b", "b_to_c"]}}]},
        "good_after.yaml": {"boundaries": [{"expect": {"after": ["c", "a 0"]}}]}})
    shared = T.goldens(d, seeds=(1, 2))
    fresh = [(f.name, s) for f in T.golden_files(d) for s in (1, 2) if not stub.run_case(d, f, seed=s).passed]
    assert sorted((x["file"], x["seed"]) for x in shared[1]) == sorted(fresh) == [("bad.yaml", 1), ("bad.yaml", 2)]


def test_no_segment_is_said(tmp_path):
    (tmp_path / "t").mkdir()
    with pytest.raises(T.TraceError, match="no segment"):
        T.read(tmp_path / "t")
