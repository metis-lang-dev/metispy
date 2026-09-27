"""Learning leg-2 gates: planted-parameter recovery by Dirichlet
notching from observed runs; categorical sites recovered relative to
their reference candidate; the exact E-step checked ANALYTICALLY;
MAP-EM on partially observed sysadmin episodes (monotone LL, planted
rows approached); the overlay plugs into the same projection
machinery; out-of-class sites are refused."""

import random

from pytest import approx

from metis.kernel import ForwardProjection, run_trace
from metis.kernel.factorize import FactoredProgram
from metis.kernel.posterior import Conditioned
from metis.kernel.notch import em_notch, expected_site_counts, \
    notch, site_counts
from metis.lang import compile_catalog, factor_hooks, load_llp, \
    parse, weight_policy
from test_factorize import CASCADE, TRIWAY, SYS, exact_marginal, \
    sys_registry

EXACT = dict(abs=1e-9)


def true_reg():
    return {"pa": lambda a, c: 0.55,
            "pb": lambda a, c: 0.7 if "a" in c else 0.3}


def flat_reg():
    return {"pa": lambda a, c: 0.5, "pb": lambda a, c: 0.5}


def build(reg, text=CASCADE, init=None):
    doc = compile_catalog(parse(text))
    prog = doc.catalog.ground(init or {"ta": 1})
    hooks = factor_hooks(prog, doc.ports, reg, doc.catalog.horn_db())
    pol = weight_policy(prog, doc.ports, reg)
    return doc, prog, hooks, pol


def sample_traces(prog, pol, n, steps=3, seed=5):
    rng = random.Random(seed)
    return [run_trace(prog, steps, rng, policy=pol)[0]
            for _ in range(n)]


# ---- observed-run notching (paired sites) ------------------------------

def test_planted_parameters_recovered_and_sheet_readable():
    _, prog, _, true_pol = build(true_reg())
    traces = sample_traces(prog, true_pol, 3000)
    _, _, base_hooks, _ = build(flat_reg())
    fp = FactoredProgram(prog, 3, base_hooks)
    overlay, learned, sheet = notch(fp, traces, flat_reg(), alpha=2.0)
    got = {r: post for (f, _a, r), (_p0, post, _n) in learned.items()
           if f == "pb"}
    assert got[(("a", 1),)] == approx(0.7, abs=0.05)
    assert got[(("a", 0),)] == approx(0.3, abs=0.05)
    pa = [post for (f, _a, _r), (_p0, post, _n) in learned.items()
          if f == "pa"]
    assert pa and pa[0] == approx(0.55, abs=0.05)
    # the row sheet is the approval artifact: prior -> posterior + n
    assert "pb | a=1 : 0.500 ->" in sheet and "n=" in sheet

    def joint_ab(pr):        # P(a AND b) — where the rows matter
        return sum(p for (_, k), p in pr.final.items()
                   if ("a", 1) in k and ("b", 1) in k)

    doc, prog2, _, _ = build(true_reg())
    true_pr = ForwardProjection(prog2, 3, policy=weight_policy(
        prog2, doc.ports, true_reg()))
    over_pr = ForwardProjection(prog2, 3, policy=weight_policy(
        prog2, doc.ports, overlay))
    assert joint_ab(over_pr) == approx(joint_ab(true_pr), abs=0.03)
    flat_pr = ForwardProjection(prog2, 3, policy=weight_policy(
        prog2, doc.ports, flat_reg()))
    assert abs(joint_ab(flat_pr) - joint_ab(true_pr)) > 0.05


# ---- categorical sites (reference-anchored) ----------------------------

def test_categorical_site_recovered_via_reference():
    """Triway: gs @w w1, gr @w w2, gf unweighted — the unweighted
    branch pins the scale CHOICE normalization absorbs."""
    true = {"w1": lambda a, c: 0.55, "w2": lambda a, c: 0.15}
    flat = {"w1": lambda a, c: 0.4, "w2": lambda a, c: 0.4}
    _, prog, _, true_pol = build(true, TRIWAY, {"tk": 1})
    traces = sample_traces(prog, true_pol, 4000, steps=1)
    _, _, base_hooks, _ = build(flat, TRIWAY, {"tk": 1})
    fp = FactoredProgram(prog, 1, base_hooks)
    _, learned, sheet = notch(fp, traces, flat, alpha=2.0)
    vals = {f: post for (f, _a, _r), (_p0, post, _n)
            in learned.items()}
    assert vals["w1"] == approx(0.55, abs=0.06)
    assert vals["w2"] == approx(0.15, abs=0.06)
    assert "w1 | " in sheet


def test_out_of_class_sites_refused():
    """A complemented port in a site with an unweighted candidate
    fits neither the paired nor the categorical rule."""
    bad = TRIWAY.replace("gr [dyn] : tk -o r  @w w2.",
                         "gr [dyn] : tk -o r  @w ~w1.")
    reg = {"w1": lambda a, c: 0.5, "w2": lambda a, c: 0.3}
    _, prog, hooks, pol = build(reg, bad, {"tk": 1})
    fp = FactoredProgram(prog, 1, hooks)
    traces = sample_traces(prog, pol, 100, steps=1)
    assert site_counts(fp, traces) == {}


# ---- the exact E-step, analytically ------------------------------------

def test_expected_counts_match_hand_posterior():
    """Cascade, observe ONLY b at the horizon: the E-step's expected
    row counts are hand-computable Bayes."""
    _, prog, hooks, pol = build(true_reg())
    fp = FactoredProgram(prog, 3, hooks)
    c = Conditioned(ForwardProjection(prog, 3, policy=pol),
                    {3: lambda s: ("b", 1) in s[1]})
    counts = expected_site_counts(fp, [c])
    pb1 = 0.55 * 0.7 / (0.55 * 0.7 + 0.45 * 0.3)   # P(a=1 | b=1)
    n1, s1 = counts[("b", "pb", (), (("a", 1),))]
    n0, s0 = counts[("b", "pb", (), (("a", 0),))]
    assert n1 == approx(pb1, **EXACT) and s1 == approx(pb1, **EXACT)
    assert n0 == approx(1 - pb1, **EXACT)
    na, sa = counts[("b", "pa", (), ())]
    assert na == approx(1.0, **EXACT)
    assert sa == approx(pb1, **EXACT)      # up-branch = a produced


# ---- MAP-EM on partially observed sysadmin -----------------------------

def test_em_recovers_planted_rows_partially_observed():
    """Two ticks; run/down observed at each TICK BOUNDARY (steps 9
    and 18), every mid-tick outcome hidden — the telemetry shape.
    (Observing only step 18 is a genuine identifiability RIDGE:
    high-cpt/low-rp and low-cpt/high-revive explain the same finals;
    EM then climbs LL onto the ridge, correctly but uselessly. Tick
    boundaries restore identifiability.) LL is monotone and the
    learned cpt/rp rows move from the perturbed base to the planted
    truth."""
    from metis.kernel import apply_event
    doc = load_llp(SYS)
    init = {"m(i0)": 1, "run(a)": 1, "run(b)": 1}
    prog = doc.catalog.ground(init)
    horn = doc.catalog.horn_db()
    events = prog.events()
    true = sys_registry()                 # cpt rows .95/.70, rp .04
    # NOT exactly flat: 0.5 everywhere makes the observations
    # independent of hidden outcomes — a stationary point of the
    # likelihood (the classic EM symmetry trap). Any asymmetry
    # breaks it.
    flat = {"cpt": lambda a, c: 0.6, "rp": lambda a, c: 0.3}
    true_pol = weight_policy(prog, doc.ports, true)
    rng = random.Random(11)

    def updown(state):
        return frozenset(x for x in state[1]
                         if x[0].startswith(("run(", "down(")))

    episodes = []
    for _ in range(400):
        trace, final = run_trace(prog, 18, rng, policy=true_pol)
        s = prog.initial_state()
        for e in trace[:9]:
            s = apply_event(events, s[0], dict(s[1]), e)
        mid = updown(s)
        episodes.append({9: (lambda st, o=mid: updown(st) == o),
                         18: (lambda st, o=updown(final):
                              updown(st) == o)})
    overlay, learned, sheet, lls = em_notch(
        prog, 18, flat, episodes,
        hooks_factory=lambda r: factor_hooks(prog, doc.ports, r,
                                             horn),
        policy_factory=lambda r: weight_policy(prog, doc.ports, r),
        alpha=1.0, iters=10)
    assert all(b >= a - 1e-9 for a, b in zip(lls, lls[1:])), lls
    rows = {(f, r): post for (f, _a, r), (_p0, post, _n)
            in learned.items()}
    # the data-rich row: was-up computer with its neighbor's snapshot
    # up — truth 0.95, flat start 0.5
    up_row = rows[("cpt", (("prev(b)", 1), ("rb(a)", 0)))]
    assert abs(up_row - 0.95) < abs(0.5 - 0.95)
    assert up_row == approx(0.95, abs=0.12)
    # revive probability — truth 0.04, start 0.5
    rp_rows = [v for (f, _r), v in rows.items() if f == "rp"]
    assert rp_rows and min(rp_rows) < 0.25
    # the learned measure predicts the observable better than flat
    def pboth(reg):
        pr = ForwardProjection(prog, 18, policy=weight_policy(
            prog, doc.ports, reg))
        return sum(p for (_, k), p in pr.final.items()
                   if ("run(a)", 1) in k and ("run(b)", 1) in k)
    assert abs(pboth(overlay) - pboth(true)) < \
        abs(pboth(flat) - pboth(true))
