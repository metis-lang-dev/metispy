"""metis.domains.rddl_recorder — the RDDL experiment's recorder tool:
the mechanical fluent<->atom mapping, the surprise sink (readouts,
negatives, tick token) on synthetic transitions, and — where pyRDDLGym
+ rddlrepository are installed (metispy's `sysadmin` extra) — a REAL
episodic batch: raw-transition JSONL, pyRDDLGym's own logs, the sink."""
import json

import pytest

from metis.domains.rddl_recorder import (atom_of, main, neg_of,
                                         read_transitions, surprise_sink)

VOCAB = ["door-closed___e0", "at___e0__f0", "at___e0__f1",
         "waiting___f1"]

# a toy catalog over the mechanically mapped atoms
CAT = """catalog toy 1.
provenance text "recorder test" "2026-10-09".
layers (env).
stages (go).
type elevator {e0}.
type floor {f0 f1}.
namespace w produce (env).
pred door_closed(elevator) : w.
pred not_door_closed(elevator) : w.
pred at(elevator, floor) : w.
pred not_at(elevator, floor) : w.
pred waiting(floor) : w.
pred not_waiting(floor) : w.
pred close_door(elevator) : w.
pred tick : w.
stage go {
  %% "closing"
  close [env] : close_door(E) * not_door_closed(E) -o door_closed(E).
  %% "a waiter may appear, once per tick"
  arrive [env] : tick * not_waiting(F) -o (waiting(F) @w 0.5 | not_waiting(F) @w 0.5).
}
"""


def tr(state, action, nxt, ep=0, t=0):
    return {"domain": "Toy", "instance": "1", "episode": ep, "seed": 1,
            "t": t, "state": state, "action": action, "reward": 0.0,
            "next": nxt}


def test_the_mechanical_mapping():
    assert atom_of("elevator-at-floor___e0__f2") == "elevator_at_floor(e0,f2)"
    assert atom_of("elevator-closed___e1") == "elevator_closed(e1)"
    assert atom_of("noop") == "noop"
    assert neg_of("person-waiting-up___f1") == "not_person_waiting_up(f1)"


def test_sink_counts_surprises_per_episode(tmp_path):
    cat = tmp_path / "toy.llp"
    cat.write_text(CAT)
    trs = [  # explained: close fires (1 step)
        tr(["at___e0__f0"], ["close-door___e0"],
           ["at___e0__f0", "door-closed___e0"], 0, 0),
        # unexplained: the elevator moved, no rule moves it
        tr(["at___e0__f0"], [], ["at___e0__f1"], 0, 1),
        tr(["at___e0__f0"], [], ["at___e0__f0"], 1, 0)]
    out = tmp_path / "sink.jsonl"
    res = surprise_sink(cat, trs, VOCAB, out, steps=1, negatives=True)
    assert res["transitions"] == 3 and res["per_episode"] == {0: 1, 1: 0}
    assert res["readout"] == "1 step(s)"
    (rec,) = [json.loads(l) for l in out.read_text().splitlines()]
    assert rec["kind"] == "unmodeled" and rec["meta"]["t"] == 1
    # negatives: the false fluents are observed as not_ atoms
    assert rec["observed"]["not_door_closed(e0)"] == 1
    assert rec["init"]["not_at(e0,f1)"] == 1


def test_frame_readout_and_tick_token(tmp_path):
    """A tick-structured catalog is only right under the frame readout:
    with `tick` the arrival draw happens once; without it never."""
    cat = tmp_path / "toy.llp"
    cat.write_text(CAT)
    arrived = tr([], [], ["waiting___f1"])
    no_tick = surprise_sink(cat, [arrived], VOCAB, tmp_path / "a.jsonl",
                            steps="frame", negatives=True)
    with_tick = surprise_sink(cat, [arrived], VOCAB, tmp_path / "b.jsonl",
                              steps="frame", negatives=True, tick_token=True)
    assert no_tick["surprises"] == 1 and with_tick["surprises"] == 0
    assert with_tick["readout"] == "frame" and with_tick["unsettled"] == 0
    # the draw was one of two branches at the frame: likelihood 0.5 * ...
    assert (tmp_path / "b.jsonl").read_text() == ""



def test_sampled_tier_agrees_and_auto_falls_back(tmp_path):
    """The sampled tier prices like the exact filter (within its ci),
    and `auto` falls back to it when the filter refuses its budget —
    each record says which tier priced it."""
    cat = tmp_path / "toy.llp"
    cat.write_text(CAT)
    stay = tr([], [], [])                   # nobody arrived: P = 0.5
    rare = 0.6                              # make P = 0.5 a "rare" record
    ex = surprise_sink(cat, [stay], VOCAB, tmp_path / "e.jsonl",
                       steps="frame", negatives=True, tick_token=True,
                       rare=rare)
    sa = surprise_sink(cat, [stay], VOCAB, tmp_path / "s.jsonl",
                       steps="frame", negatives=True, tick_token=True,
                       rare=rare, tier="sampled", samples=4000)
    re_ = json.loads((tmp_path / "e.jsonl").read_text())
    rs = json.loads((tmp_path / "s.jsonl").read_text())
    assert ex["tiers"] == {"exact": 1} and sa["tiers"] == {"sampled": 1}
    assert re_["tier"] == "exact" and re_["likelihood"] == pytest.approx(0.5)
    assert rs["tier"] == "sampled" and rs["n"] == 4000
    assert abs(rs["likelihood"] - 0.5) <= rs["ci"] + 1e-9
    au = surprise_sink(cat, [stay], VOCAB, tmp_path / "a.jsonl",
                       steps="frame", negatives=True, tick_token=True,
                       rare=rare, tier="auto", max_states=1)
    assert au["tiers"] == {"sampled": 1}
    with pytest.raises(Exception):          # exact never silently degrades
        surprise_sink(cat, [stay], VOCAB, tmp_path / "x.jsonl",
                      steps="frame", negatives=True, tick_token=True,
                      max_states=1)


PORTED = CAT.replace(
    "pred tick : w.",
    "pred tick : w.\nweight parr(F) reads {}.").replace(
    "-o (waiting(F) @w 0.5 | not_waiting(F) @w 0.5).",
    "-o (waiting(F) @w parr(F) | not_waiting(F) @w ~parr(F)).")

PORTS_PY = """
class P:
    def __init__(self, case): pass
    def bind(self, reg): reg.weight("parr", lambda a, c: 0.2)
    def tick(self, counts): return None
    def on_event(self, name, args): pass
    def snapshot(self): return None
    def restore(self, snap): pass

def make(case):
    return P(case)
"""


def test_sink_prices_weight_ports_with_plugin_and_overlay(tmp_path):
    """A ported catalog: unbound -> refused; --plugin prices both tiers
    with the base (P[nobody arrives] = 0.8); a notch overlay rides on
    top (0.6 -> P = 0.4)."""
    cat = tmp_path / "toy.llp"
    cat.write_text(PORTED)
    (tmp_path / "ports_py.py").write_text(PORTS_PY)
    stay = tr([], [], [])
    kw = dict(steps="frame", negatives=True, tick_token=True, rare=0.9)
    with pytest.raises(Exception):
        surprise_sink(cat, [stay], VOCAB, tmp_path / "u.jsonl", **kw)
    for tier in ("exact", "sampled"):
        surprise_sink(cat, [stay], VOCAB, tmp_path / f"{tier}.jsonl",
                      tier=tier, samples=4000, plugin="ports_py:make", **kw)
        r = json.loads((tmp_path / f"{tier}.jsonl").read_text())
        assert abs(r["likelihood"] - 0.8) <= (r.get("ci") or 1e-9), tier
    ov = tmp_path / "notch.json"
    ov.write_text(json.dumps({"rows": [{"port": "parr", "args": ["f1"],
                                         "row": [], "notched": 0.6}]}))
    surprise_sink(cat, [stay], VOCAB, tmp_path / "o.jsonl",
                  plugin="ports_py:make", overlay=ov, **kw)
    r = json.loads((tmp_path / "o.jsonl").read_text())
    assert r["likelihood"] == pytest.approx(0.4)


def _have_rddl():
    try:
        import pyRDDLGym  # noqa: F401
        import rddlrepository  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _have_rddl(),
                    reason="pyRDDLGym + rddlrepository (sysadmin extra)")
def test_a_real_episodic_batch(tmp_path, capsys):
    """One REAL Elevators episode: the raw JSONL (horizon rows, stamped),
    pyRDDLGym's own log + debug log, and the sink against a catalog."""
    cat = tmp_path / "toy.llp"
    cat.write_text(CAT)
    rc = main(["batch", "--problem", "Elevators_MDP_ippc2011",
               "--instance", "1", "--episodes", "1", "--seed", "3",
               "--dir", str(tmp_path / "exp"), "--catalog", str(cat),
               "--negatives"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert rc == 0
    rows = list(read_transitions(out["record"]["path"]))
    assert len(rows) == 40 == out["record"]["steps"]
    assert {r["seed"] for r in rows} == {3} and rows[0]["t"] == 0
    assert all("action" in r and "reward" in r for r in rows)
    logs = sorted(p.name for p in (tmp_path / "exp" / "gym-logs").iterdir())
    assert any(n.endswith("_log.csv") for n in logs)
    assert any(n.endswith("_debug.log") for n in logs)
    sink = out["sink"]
    assert sink["transitions"] == 40 and "surprises_per_episode" in sink
    # batches never overwrite recorded episodes
    rc2 = main(["batch", "--problem", "Elevators_MDP_ippc2011",
                "--instance", "1", "--episodes", "1", "--seed", "3",
                "--dir", str(tmp_path / "exp")])
    assert rc2 == 1 and "exists" in capsys.readouterr().out
