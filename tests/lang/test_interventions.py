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


# -- step 2: whatif ----------------------------------------------------

def world(r):
    return (list(r.fired), dict(r.counts), r.stage, r.rng.state,
            [dict(h) for h in r.fired_how], set(r.masks), list(r.ledger),
            len(r.undo_stack), r.shelf, r._base)


def exact(r, k, dos, atom, horizon):
    """The report's numbers the long way: fork, query, do, query."""
    r._quiet(f"fork {k}")
    try:
        base = r._presence(atom, horizon)
        for d in dos:
            r._quiet("do " + d)
        forced = sum(1 for d in dos if d[0] not in "+-~")
        return base, r._presence(atom, horizon - forced)
    finally:
        r._quiet("back")


def test_whatif_query_reports_fact_counterfactual_and_chance():
    r, say = shell()
    say("run 6")
    before = world(r)
    out = say("whatif 1 do(~blow_stick) do(~blow_straw) query wolf 4")
    assert world(r) == before                         # session untouched
    lines = out.strip().splitlines()
    assert lines[0].startswith("WHATIF at #1  do(~blow_stick) "
                               "do(~blow_straw)  ·  query wolf  ·  4 ")
    assert lines[1].startswith("  in fact     wolf ")
    assert lines[2].startswith("  with do     P = ")
    assert lines[3].startswith("  by chance   P = ")
    assert lines[4] == "  interventions: do ~blow_stick; do ~blow_straw"
    assert lines[5].startswith("  ⇒ had do(~blow_stick) do(~blow_straw) "
                               "at #1, P(wolf) = ")
    base, cf = exact(r, 1, ["~blow_stick", "~blow_straw"], "wolf", 4)
    assert f"P = {cf:.4g}" in lines[2] and f"P = {base:.4g}" in lines[3]
    assert world(r) == before


def test_whatif_default_horizon_is_the_actual_worlds_and_counts_forced():
    r, say = shell()
    say("run 5")
    out = say("whatif 0 do(build_straw) query brickhouse")
    assert "5 events after #0 (the actual world's horizon)" in out
    assert "(1 forced, 4 by chance)" in out
    base, cf = exact(r, 0, ["build_straw"], "brickhouse", 5)
    assert f"P = {cf:.4g}" in out and f"P = {base:.4g}" in out


def test_whatif_run_is_side_by_side_and_labelled_one_draw():
    r, say = shell()
    say("run 4")
    before = world(r)
    out = say("whatif 2 do(+bricks) run 3")
    assert world(r) == before
    assert "ONE sampled draw, not a measure" in out
    assert "in fact (the actual world)" in out and "with do" in out
    assert f"3. {r.fired[2]}" in out and "4. " in out
    assert "interventions: do +bricks" in out


def test_whatif_refusals_leave_the_session_alone():
    r, say = shell()
    say("run 3")
    before = world(r)
    assert "did not apply at #0" in say("whatif 0 do(nosuch) query wolf")
    assert "usage: whatif" in say("whatif 1 query wolf")
    assert "fork K needs" in say("whatif 9 do(~blow_stick) query wolf")
    assert world(r) == before
    say("fork 1")
    assert "already forked" in say("whatif 1 do(~blow_stick) query wolf")


def test_whatif_budget_says_so_and_suggests_a_lower_horizon(monkeypatch):
    import metis.lang.repl as repl
    monkeypatch.setattr(repl, "STATE_BUDGET", 3)
    r, say = shell()
    say("run 4")
    before = world(r)
    out = say("whatif 0 do(~build_brick) query wolf 6")
    assert "budget" in out and "(try: whatif 0 do(~build_brick) " \
        "query wolf " in out
    assert world(r) == before


def test_query_honours_masks_and_says_so():
    r, say = shell()
    plain = say("query brickhouse 3")
    say("do ~build_brick")
    masked = say("query brickhouse 3")
    assert "| do ~build_brick] = 0" in masked
    assert "do ~" not in plain


# -- step 3: necessity + the PS proxy ----------------------------------

def arms(r, k, op, atom, horizon):
    """Both arms the long way: fork k-1, query, do, query, back."""
    r._quiet(f"fork {k - 1}")
    try:
        base = r._presence(atom, horizon)
        r._quiet("do " + op)
        return base, r._presence(atom, horizon - (op[0] != "~"))
    finally:
        r._quiet("back")


def three_brick():
    r, say = shell(seed=1)     # build_brick fires at #4; brickhouse holds
    say("run 8")
    assert r.fired[3] == "build_brick" and r.counts.get("brickhouse")
    return r, say


def test_necessity_blocks_k_and_reports_pn_beside_the_fact():
    r, say = three_brick()
    before = world(r)
    out = say("necessity 4 brickhouse")
    assert world(r) == before
    lines = out.strip().splitlines()
    assert lines[0] == ("NECESSITY of #4 build_brick for brickhouse  ·  "
                        "3 events after #3 (the actual world's horizon)")
    assert lines[1].startswith("  in fact      brickhouse PRESENT")
    assert lines[2].startswith("  without #4   P = 0 ")
    assert "do ~build_brick (blocked to the horizon)" in lines[2]
    assert lines[3].startswith("  by chance    P = ")
    assert lines[4].startswith("  ⇒ PN ≈ 1: #4 build_brick was necessary "
                               "for brickhouse")
    assert "prefix-fixed" in lines[5]
    base, cf = arms(r, 4, "~build_brick", "brickhouse", 3)
    assert cf == 0 and f"P = {base:.4g}" in lines[3]


def test_necessity_default_atom_from_the_forward_cone():
    r, say = three_brick()
    assert r._forward_cone_atoms(4) == ["brickhouse"]
    assert "NECESSITY of #4 build_brick for brickhouse" in say("necessity 4")
    r2, say2 = shell(seed=10)                # several candidates: ask
    say2("run 8")
    out = say2("necessity 1")
    assert "needs an ATOM — present atoms in #1's forward cone:" in out


def test_necessity_fact_is_read_at_the_horizon():
    r, say = three_brick()
    out = say("necessity 4 brickhouse 1")
    assert "1 event after #3" in out and "the actual world, #4" in out
    assert "(the actual world's horizon)" not in out
    # an outcome absent at that point is refused, naming the point
    assert "absent in fact at #6" in say("necessity 1 stickhouse")


def test_necessity_refusals():
    r, say = three_brick()
    assert "usage: necessity K [ATOM]" in say("necessity")
    assert "needs an event of the graph: #1..#6" in say("necessity 9 x")
    say("fork 2")
    assert "already forked" in say("necessity 1 brickhouse")


def test_sufficiency_is_labelled_a_proxy_and_says_what_ps_needs():
    r, say = three_brick()
    before = world(r)
    out = say("sufficiency 4 brickhouse")
    assert world(r) == before
    assert out.startswith("SUFFICIENCY (interventional PROXY, not "
                          "Pearl's PS) of #4 build_brick for brickhouse")
    assert "PROXY: true PS" in out and "posterior over those ¬K worlds" \
        in out and "deferred (spec 10 D3)" in out
    base, cf = arms(r, 4, "build_brick", "brickhouse", 3)
    assert f"forced #4   P = {cf:.4g}" in out
    assert f"by chance   P = {base:.4g}" in out
    assert "usage: sufficiency K ATOM" in say("sufficiency 4")


def test_causation_budget_says_so_and_suggests_a_lower_horizon(monkeypatch):
    import metis.lang.repl as repl
    monkeypatch.setattr(repl, "STATE_BUDGET", 2)
    r, say = three_brick()
    before = world(r)
    for verb in ("necessity", "sufficiency"):
        out = say(f"{verb} 4 brickhouse")
        assert "budget" in out and f"(try: {verb} 4 brickhouse " in out
        assert world(r) == before


def test_counts_at_replays_the_actual_world():
    r, say = three_brick()
    states = []
    r2, say2 = shell(seed=1)
    for _ in range(6):
        say2("run 1")
        states.append(dict(r2.counts))
    for j, st in enumerate(states, 1):
        assert r._counts_at(j) == st


def test_necessity_budget_never_suggests_a_retry_that_would_be_refused(
        monkeypatch):
    import metis.lang.repl as repl
    r, say = three_brick()                    # brickhouse holds from #4
    monkeypatch.setattr(repl, "STATE_BUDGET", 3)    # blows at step 2/3
    assert "(try: necessity 4 brickhouse 1 " in say("necessity 4 brickhouse")
    # had the outcome held only later, the retry would be refused: then
    # the message says so instead of suggesting it
    real = r._counts_at
    monkeypatch.setattr(r, "_counts_at",
                        lambda j: real(j) if j >= 6 else {})
    out = say("necessity 4 brickhouse")
    assert "(try:" not in out
    assert "within the 1 event it can answer, brickhouse does not yet " \
        "hold in fact (it holds from #6)" in out
