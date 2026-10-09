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
    say("fork 1")      # instruments work INSIDE a user fork (own shelf)
    before_fork = world(r)
    assert "WHATIF at #1" in say("whatif 1 do(~blow_stick) query wolf")
    assert world(r) == before_fork and r.fork_at == 1


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


def test_necessity_default_atom_from_the_forward_cone():
    r, say = three_brick()
    assert r._forward_cone_atoms(4) == ["brickhouse"]
    assert "NECESSITY of #4 build_brick for brickhouse" in say("necessity 4")
    r2, say2 = shell(seed=10)                # several candidates: ask
    say2("run 8")
    out = say2("necessity 1")
    assert "needs an ATOM — present atoms in #1's forward cone:" in out


def test_necessity_refusals():
    r, say = three_brick()
    assert "usage: necessity K [ATOM]" in say("necessity")
    assert "needs an event of the graph: #1..#6" in say("necessity 9 x")
    say("fork 2")      # inside a user fork: answers, fork intact
    before_fork = world(r)
    out = say("necessity 1 brickhouse")
    assert "NECESSITY of #1" in out or "needs" in out or "does not" in out
    assert world(r) == before_fork and r.fork_at == 2


def test_counts_at_replays_the_actual_world():
    r, say = three_brick()
    states = []
    r2, say2 = shell(seed=1)
    for _ in range(6):
        say2("run 1")
        states.append(dict(r2.counts))
    for j, st in enumerate(states, 1):
        assert r._counts_at(j) == st


# -- step 4: the wire (spec 10 P5) -------------------------------------

def test_intervention_verbs_ride_the_intent_whitelist():
    from metis.loop.wire import INTENT_VERBS, intent_posting
    assert {"do", "fork", "back", "whatif", "necessity",
            "sufficiency"} <= INTENT_VERBS
    r, say = three_brick()
    before = world(r)
    p = intent_posting(r, ":necessity 4 brickhouse", session="s1")
    assert p["status"] == "done" and p["session"] == "s1"
    assert p["result"].startswith("NECESSITY of #4 build_brick")
    assert p["json"]["masks"] == [] and p["json"]["fork"] is None
    assert world(r) == before
    p = intent_posting(r, ":whatif 3 do(~build_brick) query brickhouse")
    assert p["status"] == "done" and "WHATIF at #3" in p["result"]
    p = intent_posting(r, ":sufficiency 4 brickhouse")
    assert "interventional PROXY" in p["result"]
    # the mutating ones post what fired and the state after
    p = intent_posting(r, ":fork 3")
    assert p["status"] == "done" and p["json"]["counts"] == r.counts
    p = intent_posting(r, ":do build_brick")
    assert [f["name"] for f in p["json"]["fired"]] == ["build_brick"]
    assert r.fired_how[-1]["how"] == "do"
    p = intent_posting(r, ":back")
    assert "back in the actual world" in p["result"]
    assert world(r) == before
    # the graph op carries the marks to the console
    from metis.loop.wire import dispatch
    intent_posting(r, ":do ~blow_brick")
    assert dispatch(r, "ctx")  # ctx still serves
    assert r.ctx()["masks"] == ["blow_brick"]


def test_absence_outcomes_bang_atom():
    """`!wolf` — the outcome "the wolf is gone" (spec 10 §7's 3lp
    question): P[!a] = 1 - P[a]; facts and sentences read ¬a."""
    r, say = three_brick()                     # seed 1: wolf spent at #6
    assert not r.counts.get("wolf")
    out = say("necessity 4 !wolf")
    assert out.startswith("NECESSITY of #4 build_brick for ¬wolf")
    assert "in fact      wolf absent" in out
    base, cf = arms(r, 4, "~build_brick", "wolf", 3)
    pn = 1 - (1 - cf)                          # P(¬wolf fails) = P(wolf)
    assert f"PN ≈ {pn:.4g}" in out
    assert "PN ≈ 1: #4 build_brick was necessary for ¬wolf" in out
    w = say("whatif 3 do(~build_brick) query !wolf")
    assert "P(¬wolf) = 0" in w and "in fact ¬wolf holds" in w
    assert "does not hold in fact" in say("necessity 4 !brickhouse")
    assert r._presence("!wolf", 0) == 1.0 and r._y("!wolf") == "¬wolf"


# -- spec 11: readouts and tiers on the spec 10 instruments -------------

def test_whatif_global_readout_reports_and_matches_the_long_way():
    """A plain N is the GLOBAL readout (validation, flagged): the report
    shape and its numbers re-derived through the full filter."""
    r, say = shell()
    say("run 6")
    before = world(r)
    out = say("whatif 1 do(~blow_stick) do(~blow_straw) query wolf 4")
    assert world(r) == before                         # session untouched
    lines = out.strip().splitlines()
    assert lines[0] == ("WHATIF at #1  do(~blow_stick) do(~blow_straw)  ·"
                        "  query wolf  ·  4 global events (scheduling-"
                        "sensitive — validation readout) after #1")
    assert lines[1].startswith("  in fact      wolf ")
    assert lines[2].startswith("  with do      P = ")
    assert lines[3].startswith("  by chance    P = ")
    assert lines[4] == "  interventions: do ~blow_stick; do ~blow_straw"
    assert lines[5].startswith("  ⇒ had do(~blow_stick) do(~blow_straw) "
                               "at #1, P(wolf) = ")
    assert lines[6] == "  tier         exact (full world)"
    base, cf = exact(r, 1, ["~blow_stick", "~blow_straw"], "wolf", 4)
    assert f"P = {cf:.4g}" in lines[2] and f"P = {base:.4g}" in lines[3]
    assert world(r) == before


def test_whatif_default_reads_the_remainders_cone_steps_on_the_cone():
    from metis.lang.cone import sweep
    r, say = shell()
    say("run 5")
    out = say("whatif 0 do(build_straw) query brickhouse")
    c = sweep(r.prog, ["brickhouse"], r.prog.init_stage)
    names = {r.prog.events()[i][2].name for i in c.events}
    n = sum(1 for x in r.fired if x in names)
    assert f"query brickhouse  ·  @{n} cone step" in out
    assert "(1 forced)" in out and "tier         exact (cone)" in out
    assert "cone with do" in out and "cone chance" in out and "key " in out


def test_auto_tier_samples_over_budget_and_exact_forced_refuses(
        monkeypatch):
    import metis.lang.repl as repl
    monkeypatch.setattr(repl, "STATE_BUDGET", 3)
    r, say = shell()
    say("run 4")
    before = world(r)
    out = say("whatif 0 do(~build_brick) query !wolf 6")
    assert "tier         sampled n=" in out and "exact refused:" in out
    assert "common random numbers — variance reduction only" in out
    assert " ± " in out
    assert world(r) == before
    ref = say("whatif 0 do(~build_brick) query !wolf 6 exact")
    assert "budget" in ref and "`exact` was forced" in ref
    assert world(r) == before


def test_necessity_default_cone_readout_and_pn():
    r, say = three_brick()
    before = world(r)
    out = say("necessity 4 brickhouse")
    assert world(r) == before
    lines = out.strip().splitlines()
    assert lines[0] == ("NECESSITY of #4 build_brick for brickhouse  ·  "
                        "@1 cone step (or when the cone settles, if sooner) "
                        "after #3")
    assert lines[1].startswith("  in fact      brickhouse PRESENT")
    assert lines[2].startswith("  without #4   P = 0 ")
    assert "do ~build_brick (blocked through the readout)" in lines[2]
    assert lines[4].startswith("  ⇒ PN ≈ 1: #4 build_brick was necessary "
                               "for brickhouse")
    assert "tier         exact (cone)" in out


def test_necessity_global_readout_reads_the_fact_there():
    r, say = three_brick()
    out = say("necessity 4 brickhouse 1")
    assert "1 global event (scheduling-sensitive" in out
    assert "the actual world, #4" in out
    base, cf = arms(r, 4, "~build_brick", "brickhouse", 1)
    assert f"P = {cf:.4g}" in out and f"P = {base:.4g}" in out
    refused = say("necessity 1 stickhouse")
    assert "does not hold in fact at #6" in refused


def test_sufficiency_is_labelled_a_proxy_and_says_what_ps_needs():
    r, say = three_brick()
    before = world(r)
    out = say("sufficiency 4 brickhouse 3")
    assert world(r) == before
    assert out.startswith("SUFFICIENCY (interventional PROXY, not "
                          "Pearl's PS) of #4 build_brick for brickhouse")
    assert "PROXY: true PS" in out and "posterior over those ¬K worlds" \
        in out and "deferred (spec 10 D3)" in out
    base, cf = arms(r, 4, "build_brick", "brickhouse", 3)
    assert f"P = {cf:.4g}" in out and f"P = {base:.4g}" in out
    assert "usage: sufficiency K ATOM" in say("sufficiency 4")


def test_causation_over_budget_samples_and_forced_exact_refuses(
        monkeypatch):
    import metis.lang.repl as repl
    monkeypatch.setattr(repl, "STATE_BUDGET", 2)
    r, say = three_brick()
    before = world(r)
    for verb in ("necessity", "sufficiency"):
        out = say(f"{verb} 4 brickhouse 3")
        assert "tier         sampled n=" in out, out
        ref = say(f"{verb} 4 brickhouse 3 exact")
        assert "`exact` was forced" in ref
        assert world(r) == before


# -- the architect's REPL friction (2026-10-09) -------------------------

def test_instruments_have_their_own_shelf_inside_a_user_fork():
    """mask-and-wander, THEN ask: necessity/whatif fork onto a private
    shelf; the user's fork (shelf, cut, wandered world) is untouched."""
    r, say = three_brick()
    actual = list(r.fired)
    say("fork 3")
    say("do ~blow_stick")
    say("run 2")                                   # wander
    wandered = world(r)
    shelf_fired = list(r.shelf["fired"])
    out = say("whatif 3 do(~build_brick) query brickhouse")
    assert out.startswith("WHATIF at #3")
    k = len(r.fired)
    nec = say(f"necessity {k} " + (sorted(r._forward_cone_atoms(k))
                                   or ["brickhouse"])[0])
    assert "already forked" not in nec
    assert world(r) == wandered and r.fork_at == 3
    assert list(r.shelf["fired"]) == shelf_fired
    say("back")
    assert list(r.fired) == actual
    assert "already forked" in (say("fork 1"), say("fork 1"))[1]


def test_likelihood_footer_names_the_masks_it_was_drawn_under():
    r, say = shell()
    say("do ~build_brick")
    say("run 2")
    log = say("log")
    assert "under do ~build_brick" in log
    assert r.ctx()["likelihood"]["under_masks"] == ["build_brick"]
    r2, say2 = shell()
    say2("run 2")
    assert "under do" not in say2("log")


def test_a_zero_event_cone_reads_as_impossible_not_as_p_zero():
    r, say = shell()
    say("do ~build_brick")                 # the only brickhouse producer
    q = say("query brickhouse")
    assert "= 0  (impossible — no event can produce brickhouse under " \
        "the active masks)" in q
    assert "tracking 1 of" in q and "0 of 6 events" in q
    r2, say2 = three_brick()
    out = say2("necessity 4 brickhouse")
    assert "(blocked through the readout) — impossible — no event can " \
        "produce brickhouse under the active masks" in out

