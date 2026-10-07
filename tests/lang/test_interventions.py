"""Spec 10 step 1 — fork/back and `do` as first-class marks: forced
choices excluded from the trace likelihood, blocks that renormalize the
tier, deterministic prefix replay, and the honesty marks in log /
graph / narrate. Session/driver code only: keys never move."""
import io
from pathlib import Path

import pytest

from metis.lang.causality import as_obj, beats_text, dot
from metis.lang.repl import Repl
from metis.loop.narrate import material

CAT = Path(__file__).resolve().parents[3] / "metis-catalog"
THREE = CAT / "domains/fiction/3lp.llp"

pytestmark = pytest.mark.skipif(not THREE.exists(),
                                reason="metis-catalog not checked out")


def shell(seed=3):
    out = io.StringIO()
    r = Repl(THREE, seed=seed, stdout=out)

    def say(line):
        out.truncate(0)
        out.seek(0)
        r.onecmd(line)
        return out.getvalue()
    return r, say


def tier(r):
    ws = r._candidates()
    t = sum(w for _, w in ws)
    return {r.events_tbl[i][2].name: w / t for i, w in ws}


def test_likelihood_counts_sampled_steps_only():
    r, say = shell()
    p1 = tier(r)
    say("run 1")
    lik1 = p1[r.fired[-1]]
    p2 = tier(r)
    say("step build_stick" if "build_stick" in p2 else "step 1")
    p3 = tier(r)
    say("do " + sorted(p3)[0])
    assert [h["how"] for h in r.fired_how] == ["sampled", "forced", "do"]
    lik, ns, nf = r.likelihood()
    assert (ns, nf) == (1, 2) and lik == pytest.approx(lik1)
    # the forced rows still record their tier probability (for whatif)
    assert r.fired_how[1]["p"] == pytest.approx(p2[r.fired[1]])
    log = say("log")
    assert "[forced]" in log and "[do]" in log
    assert "over 1 sampled step (2 forced/do excluded" in log


def test_block_renormalizes_and_reset_undo_clear_it():
    r, say = shell()
    before = tier(r)
    assert "blocked (1 ground event)" in say("do ~build_brick")
    after = tier(r)
    assert "build_brick" not in after
    assert sum(after.values()) == pytest.approx(1.0)
    rest = {k: v for k, v in before.items() if k != "build_brick"}
    z = sum(rest.values())
    assert after == pytest.approx({k: v / z for k, v in rest.items()})
    assert "no event or rule" in say("do ~nosuch")
    say("undo")
    assert r.masks == set() and tier(r) == pytest.approx(before)
    say("do ~build_brick")
    say("reset")
    assert r.masks == set()


def test_force_must_be_enabled_and_respects_masks():
    r, say = shell()
    assert "no enabled candidate" in say("do blow_brick")
    say("do ~build_stick")
    assert "no enabled candidate" in say("do build_stick")
    assert not r.fired


def test_fork_replays_the_prefix_and_back_restores_exactly():
    r, say = shell(seed=5)
    states = [dict(r.counts)]
    for _ in range(4):
        say("run 1")
        states.append(dict(r.counts))
    actual = (list(r.fired), dict(r.counts), r.stage, r.rng.state,
              [dict(h) for h in r.fired_how])
    key = r.key
    for k in (0, 2, 4):
        assert "forked at" in say(f"fork {k}")
        assert r.counts == states[k] and len(r.fired) == k
        assert "already forked" in say("fork 1")      # D1: one world
        say("run 2")                                 # the alt diverges
        assert "back in the actual world" in say("back")
        assert (list(r.fired), dict(r.counts), r.stage, r.rng.state,
                r.fired_how) == actual
    assert "not forked" in say("back")
    assert "fork K needs" in say("fork 9")
    assert r.key == key                              # keys never move


def test_fork_before_a_state_edit_is_refused():
    r, say = shell()
    say("run 2")
    say("do +wolf")
    assert "before #2 lies an edit" in say("fork 1")
    assert "forked at #2" in say("fork")


def test_graph_marks_interventions_and_the_fork_point():
    r, say = shell()
    say("step build_stick")
    say("run 1")
    say("fork 2")
    say("do blow_stick" if "blow_stick" in tier(r) else "do 1")
    g, _ = r.causal()
    mk = r.marks()
    assert mk[1] == "forced" and mk[2] == "fork" and mk[3] == "do"
    text = "\n".join(beats_text(g, "", mk))
    assert "[forced]" in text and "[fork]" in text and "[do]" in text
    nodes = {n["i"]: n.get("mark") for n in as_obj(g, r.stage, r.key,
                                                   mk)["nodes"]}
    assert nodes[1] == "forced" and nodes[2] == "fork" and nodes[3] == "do"
    d = dot(g, "t", mk)
    assert 'label="forced: 1 build_stick"' in d
    assert "do: 3 " in d and "⑂ fork" in d and "orange" in d
    assert "orange" not in dot(g, "t")               # no marks, no legend


def test_state_do_is_the_base_and_named_in_graph_and_log():
    r, say = shell()
    say("run 1")
    say("do -wolf")
    say("do +wolf 2")
    assert r._base_do == ["-wolf", "+wolf 2"]
    g, _ = r.causal()
    assert "[do -wolf, +wolf 2]" in "\n".join(
        beats_text(g, "", r.marks(), r._base_do))
    assert '0 do(-wolf, +wolf 2)' in dot(g, "", r.marks(), r._base_do)
    log = say("log")
    assert "do -wolf" in log and "do +wolf 2" in log
    say("run 1")
    say("assert straw")                       # a plain edit: not a do
    assert r._base_do == []
    c = r.ctx()
    assert c["base_do"] == [] and c["likelihood"]["sampled"] == 2


def test_narrate_material_never_presents_an_intervention_as_sampled():
    r, say = shell()
    say("step build_stick")
    say("run 1")
    rows = list(r.fired_rows)
    hows = [h["how"] for h in r.fired_how]
    m = material(rows, r.stage, sorted(r.counts.items()), r.text, 0,
                 hows, ())
    beats = m["beats"]
    assert "build_stick" in beats and "[forced]" in beats
    assert "== interventions (NOT chance) ==" in beats
    assert "never as luck" in beats
    # an all-sampled run: the material is exactly what it was before
    plain = material(rows, r.stage, sorted(r.counts.items()), r.text, 0)
    same = material(rows, r.stage, sorted(r.counts.items()), r.text, 0,
                    ["sampled"] * len(rows), ())
    assert same == plain and "interventions" not in plain["beats"]
    # `narrate user` carries it on the session's own run
    assert "[forced]" in say("narrate user")


def test_ctx_lists_masks_shelf_and_likelihood():
    r, say = shell()
    say("run 2")
    say("do ~blow_stick")
    say("fork 1")
    c = r.ctx()
    assert c["masks"] == ["blow_stick"]
    assert c["fork"] == {"at": 1, "actual": 2}
    assert c["likelihood"]["sampled"] == 1
    band = say("ctx tokens")
    assert "DO      blocked blow_stick  FORKED at #1" in band
