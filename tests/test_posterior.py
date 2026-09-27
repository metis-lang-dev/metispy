"""Stage 5 gate: analytic conditioning to 1e-12, smoothing vs
brute-force run enumeration, MAP vs brute-force argmax, planted-row
identifiability with recovery rate, likelihood-ratio surprise with a
constructed ROC."""

import random

from pytest import approx, raises

from metis.kernel import Conditioned, ForwardProjection, apply_event, \
    behavior_posterior, map_derivation, run_trace, surprise
from metis.domains.road import base_catalog, scenario
from fixtures import ff, race

EXACT = dict(abs=1e-12)


def has(atom, n):
    return lambda s: dict(s[1]).get(atom, 0) == n


def at_least(atom, n):
    return lambda s: dict(s[1]).get(atom, 0) >= n


# -- brute force reference ----------------------------------------------

def enum_runs(prog, K):
    """All (prob, states[0..K]) via exhaustive choice enumeration,
    using the projection's own transition machinery."""
    pr = ForwardProjection(prog, 0)
    runs = []

    def rec(t, s, p, states):
        if t == K:
            runs.append((p, states))
            return
        stage, kappa = s
        counts = dict(kappa)
        for e, pe in pr._choice(stage, counts):
            s2 = apply_event(pr.events, stage, counts, e)
            rec(t + 1, s2, p * pe, states + (s2,))

    s0 = prog.initial_state()
    rec(0, s0, 1.0, (s0,))
    return runs


def brute_gamma(runs, evidence, t):
    tot, marg = 0.0, {}
    for p, states in runs:
        if all(pred(states[k]) for k, pred in evidence.items()):
            tot += p
            marg[states[t]] = marg.get(states[t], 0.0) + p
    return tot, {s: q / tot for s, q in marg.items()}


# -- analytic conditioning ----------------------------------------------

def test_conditioning_analytic():
    # race, 2 steps, evidence: both a consumed -> only the ra,ra path
    c = Conditioned(ForwardProjection(race(), 2),
                    {2: has("wa", 2)})
    assert c.likelihood == approx(1 / 3, **EXACT)
    assert c.event_posterior(0) == approx({0: 1.0}, **EXACT)
    # ff, 1 step, evidence: p produced
    c = Conditioned(ForwardProjection(ff(), 1), {1: has("p", 1)})
    assert c.likelihood == approx(0.75, **EXACT)
    # impossible evidence raises
    with raises(ValueError, match="zero likelihood"):
        Conditioned(ForwardProjection(race(), 1), {1: has("wa", 5)})


def test_unconditioned_gamma_equals_filter():
    pr = ForwardProjection(race(), 3)
    c = Conditioned(pr)
    assert c.likelihood == approx(1.0, **EXACT)
    for t in range(4):
        assert c.gamma(t) == approx(pr.history[t], **EXACT)


def test_smoothing_vs_brute_force():
    cases = [
        (race(), 3, {1: has("wa", 1)}),
        (base_catalog().ground(scenario()), 6, {6: at_least("crash", 1)}),
    ]
    for prog, K, evidence in cases:
        runs = enum_runs(prog, K)
        c = Conditioned(ForwardProjection(prog, K), evidence)
        for t in range(K + 1):
            L, g = brute_gamma(runs, evidence, t)
            assert c.likelihood == approx(L, **EXACT)
            assert c.gamma(t) == approx(g, **EXACT)
        # event posteriors are distributions
        for t in range(K):
            assert sum(c.event_posterior(t).values()) == approx(
                1.0, **EXACT)


def test_crash_posterior_explains_away_ego_choice():
    """Observing the crash at the end pins ego's slice-0 choice to keep
    — diagnostic reasoning through the collider."""
    prog = base_catalog().ground(scenario())
    c = Conditioned(ForwardProjection(prog, 6), {6: at_least("crash", 1)})
    assert c.likelihood == approx(11 / 18, **EXACT)
    xi0 = c.event_posterior(0)
    names = {prog.events()[e][2].name.split("[")[0]: p
             for e, p in xi0.items()}
    assert names == approx({"keep": 1.0}, **EXACT)


# -- MAP derivation ------------------------------------------------------

def test_map_vs_brute_force():
    cases = [(race(), 3, None),
             (ff(), 2, None),
             (race(), 2, {2: has("wa", 2)}),
             (base_catalog().ground(scenario()), 6, None)]
    for prog, K, evidence in cases:
        runs = enum_runs(prog, K)
        if evidence:
            runs = [(p, st) for p, st in runs
                    if all(pred(st[k]) for k, pred in evidence.items())]
        best = max(p for p, _ in runs)
        m = map_derivation(ForwardProjection(prog, K), evidence)
        assert m.prob == approx(best, **EXACT)
        # the returned path replays to its final state
        s = prog.initial_state()
        pr0 = ForwardProjection(prog, 0)
        for e in m.events:
            s = apply_event(pr0.events, s[0], dict(s[1]), e)
        assert s == m.final


def test_map_road_is_the_crash_story():
    """Unconditioned MAP of the demo scenario is the keep->collide
    derivation (11/36) — the most probable proof doubles as the
    explanation."""
    prog = base_catalog().ground(scenario())
    m = map_derivation(ForwardProjection(prog, 6))
    assert m.prob == approx(11 / 36, **EXACT)
    names = [prog.events()[e][2].name.split("[")[0]
             for e in m.events if e is not None]   # None = quiesce/done
    assert names[0] == "keep" and "collide" in names


# -- identifiability -----------------------------------------------------

def row_prog(cat, keep_w, brake_w):
    """The mini corpus with alfa's behavior row planted as clause
    weights (alfa on the other lane, moving: both moves enabled)."""
    prog = cat.ground(scenario(alfa=("l1", "c3", "v1")))
    for body in prog.stages.values():
        for c in body:
            if c.name.startswith("okeep["):
                c.weight = keep_w
            elif c.name.startswith("obrake["):
                c.weight = brake_w
    return prog


def test_identifiability_planted_row_recovered():
    cat = base_catalog()
    rows = {"calm": (0.8, 0.2), "nervous": (0.3, 0.7)}
    progs = {h: row_prog(cat, *w) for h, w in rows.items()}
    K, episodes, reps = 6, 40, 10
    rng = random.Random(23)
    recovered = 0
    finals = []
    for _ in range(reps):
        post = {h: 0.5 for h in rows}
        for _ in range(episodes):
            _, obs = run_trace(progs["nervous"], K, rng)   # planted
            evidence = {K: (lambda s, o=obs: s == o)}
            post, _ = behavior_posterior(
                {h: (lambda p=progs[h]: ForwardProjection(p, K))
                 for h in rows},
                evidence, prior=post)
        finals.append(post["nervous"])
        recovered += post["nervous"] > 0.9
    rate = recovered / reps
    assert rate >= 0.9, (rate, finals)
    assert min(finals) > 0.5      # never actively wrong


# -- surprise ------------------------------------------------------------

def test_surprise_kinds_and_roc():
    prog = base_catalog().ground(scenario())
    for c in prog.stages["ego"]:
        if c.name.startswith("accel["):
            c.weight = 0.01               # plant a rare behavior
    pr = ForwardProjection(prog, 6)
    by_kind = {}
    for s in pr.final:
        atoms = dict(s[1])
        if atoms.get("crash"):
            by_kind["crash"] = s
        elif "st(ego,l0,c3,v2)" in atoms:
            by_kind["accel"] = s
        else:
            by_kind["brake"] = s
    assert set(by_kind) == {"crash", "accel", "brake"}

    crash = surprise(pr, 6, by_kind["crash"])
    acc = surprise(pr, 6, by_kind["accel"])
    brk = surprise(pr, 6, by_kind["brake"])
    ghost = surprise(pr, 6, ("check", frozenset({("ghost", 1)})))

    assert crash["kind"] is None and crash["p"] == approx(55 / 71, **EXACT)
    assert acc["kind"] == "rare" and acc["p"] == approx(1 / 71, **EXACT)
    assert acc["ratio"] == approx(1 / 55, **EXACT)
    assert brk["kind"] is None and brk["p"] == approx(15 / 71, **EXACT)
    assert ghost["kind"] == "unmodeled" and ghost["p"] == 0.0

    # ROC on the constructed set: rank by surprise score (-p); the
    # planted-rare state must rank above every common state
    rare, common = [acc], [crash, brk]
    pairs = [(r, c) for r in rare for c in common]
    auc = sum(r["p"] < c["p"] for r, c in pairs) / len(pairs)
    assert auc == 1.0


def test_soft_evidence_reweights_the_measure():
    """Likelihood evidence: a float-valued predicate scales branch
    mass. race step 1 with soft weight 0.5 on the ra branch:
    posterior P(ra) = (2/3*0.5)/(2/3*0.5 + 1/3) = 0.5."""
    soft = {1: (lambda s: 0.5 if dict(s[1]).get("wa") else 1.0)}
    c = Conditioned(ForwardProjection(race(), 1), soft)
    assert c.likelihood == approx(2 / 3, **EXACT)
    assert c.event_posterior(0)[0] == approx(0.5, **EXACT)
