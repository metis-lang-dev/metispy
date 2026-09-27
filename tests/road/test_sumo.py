"""Stage 6 SUMO gates: planted-row recovery + held-out learning win,
surprise as a model-mismatch detector, and the closed-loop safety
invariant with SUMO's own safety layer disabled for ego."""

import shutil

import pytest

from metis.domains.road import sumo_harness as sh

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("sumo") is None
                       or shutil.which("netconvert") is None,
                       reason="sumo/netconvert not installed"),
]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return str(tmp_path_factory.mktemp("sumo"))


def test_fit_recovers_planted_rows_and_wins_held_out(work):
    train = []
    for seed in (1, 2, 3):
        train += sh.observe(work, seed)
    fitted = sh.fit_by_type(train)

    # planted ground truth: the aggressive vType churns speed, the calm
    # one holds it — the fit must recover the ordering decisively
    assert fitted["aggr"]["accel"] > 3 * fitted["calm"]["accel"]
    assert fitted["aggr"]["brake"] > 3 * fitted["calm"]["brake"]
    assert fitted["calm"]["keep"] > fitted["aggr"]["keep"]

    held = []
    for seed in (4, 5):
        held += sh.observe(work, seed)
    ll_f, _, n = sh.evaluate(held, fitted)
    ll_d, _, _ = sh.evaluate(held, {})
    assert n > 1000, "held-out sample too thin"
    assert ll_f > ll_d + 0.3, (ll_f, ll_d)   # decisive LL win

    # mismatch detector = branch-aware LL gap (threshold-free
    # likelihood-ratio test on the whole behavior). Rate-based rare
    # flags proved to be a bucket-boundary artifact, and "fitted flags
    # less than default" rewards flat models — see the stage log.
    branches = sh.fit_branches(train)
    aggr = [r for r in held if r["type"] == "aggr"]
    calm = [r for r in held if r["type"] == "calm"]
    gap_a = sh.branch_ll(aggr, branches["aggr"]) \
        - sh.branch_ll(aggr, branches["calm"])
    gap_c = sh.branch_ll(calm, branches["calm"]) \
        - sh.branch_ll(calm, branches["aggr"])
    assert gap_a > 0.3, gap_a       # measured 0.52 on held-out
    assert gap_c > 0.05, gap_c      # measured 0.11 on held-out


def test_closed_loop_safety_invariant(work):
    total = {"ego_collisions": 0, "ticks": 0, "doomed": 0}
    braked = 0
    for seed in (11, 12, 13, 14):
        stats = sh.closed_loop(work, seed=seed, steps=280)
        for k in total:
            total[k] += stats[k]
        braked += stats["actions"].get("brake", 0)
    assert total["ticks"] > 800, total
    assert braked > 20, "episodes were vacuous — ego never negotiated"
    assert total["ego_collisions"] == 0, total


def test_online_posteriors_in_the_closed_loop(work):
    """Stage-5 machinery LIVE: per-vehicle style posteriors updated
    from observed transitions feed mixture rows into the planner every
    tick. Safety holds and planted styles are recovered online
    (misses are honest: a blocked aggressive driver BEHAVES calm)."""
    train = []
    for seed in (1, 2):
        train += sh.observe(work, seed)
    models = sh.fit_branches(train)
    stats = sh.closed_loop(work, seed=11, steps=240,
                           adapt_models=models)
    assert stats["ego_collisions"] == 0
    assert stats["ticks"] > 100
    posts = stats["posteriors"]
    scored = [(vid, h) for vid, (h, conf, n) in posts.items()
              if n >= 25]
    assert len(scored) >= 15
    acc = sum(vid.startswith(h) for vid, h in scored) / len(scored)
    assert acc >= 0.65, (acc, sorted(posts.items())[:8])
