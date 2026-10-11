"""runtime-wire step 5 scaffolding: seeded episodes, the episode host,
the in-process baselines, and the scorer (metis.domains.kitchen.eval)."""
from metis.domains.kitchen.eval import (EXPECTED, always_escalate,
                                        episodes, render, run_episode,
                                        score)

EPS = episodes(3, 12)
BY = {e["id"]: e for e in EPS}


def test_episodes_are_seeded_and_balanced():
    assert episodes(3, 12) == EPS and episodes(4, 12) != EPS
    kinds = [e["kind"] for e in EPS]
    assert sorted(set(kinds)) == sorted(EXPECTED)
    assert all(kinds.count(k) == 2 for k in EXPECTED)
    for e in EPS:
        assert (e["plan"] == []) == (e["kind"] == "clean")
        assert e["beats"] > e["orders"][-1] and e["operator"] > 0


def test_oracle_is_perfect_and_always_escalate_is_the_baseline():
    oracle = [run_episode(e, "oracle", arm="C") for e in EPS]
    esc = [run_episode(e, "escalate", arm="E") for e in EPS]
    res = score(oracle + esc, BY)
    c, e = res["arms"]["C"], res["arms"]["E"]
    assert c["triage_all"] == 1.0 and c["false_resumes"] == 0
    assert c["illegal"] == 0 and c["diverged"] == 0
    # the bounds refused the repeats (partner, transient_twice: 2 each)
    assert c["refused"] == 4
    # the in-process always-escalate run IS the analytic baseline
    assert e["triage"] == always_escalate([x["kind"] for x in EPS])
    assert res["always_escalate"]["triage_all"] == e["triage_all"] == 0.2
    assert "C triage <= always-escalate (narrator only): False" in \
        res["stop_conditions"]
    assert "| always-escalate |" in render(res)


def test_blind_arm_counts_illegal_actions_and_false_resumes():
    a = [run_episode(e, "oracle", blind=True, arm="A") for e in EPS]
    c = [run_episode(e, "oracle", arm="B") for e in EPS]
    res = score(a + c, BY)
    A = res["arms"]["A"]
    # without the runtime the repeats are resumed again: illegal, false
    assert A["illegal"] == A["false_resumes"] == 4
    assert A["triage"]["partner"] == A["triage"]["transient_twice"] == 0
    assert "B<=A (premise fails): False" in res["stop_conditions"]


def test_harness_wakes_and_tokens_override_the_runtime_count():
    line = run_episode(EPS[0], "oracle", arm="C")
    res = score([line], BY, {(EPS[0]["id"], "C"): {"wakes": 3,
                                                   "tokens": 900}})
    c = res["arms"]["C"]
    assert c["tokens_per_hour"] > 0 and c["wakes_per_hour"] > 0
