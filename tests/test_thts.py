"""Kernel THTS unit gates on a hand-made two-armed MDP: the safe arm
must win, terminal absorption charges correctly, and the reward-scaled
default exploration keeps both arms sampled."""

from metis.kernel import mcts_plan


def toy(a_risky_death=0.4):
    """States: 's' start; 'g' goal (terminal). Arm 'safe': 3 epochs of
    -1 then goal. Arm 'risky': 1 epoch, but death w.p. p charges the
    remaining depth."""
    def actions(s):
        return ["safe", "risky"] if s == "s" else []

    def step(s, a, rng, d):
        if a == "safe":
            return "g", -3.0, True
        if rng.random() < a_risky_death:
            return "dead", -float(d), True
        return "g", -1.0, True

    def rollout(s, rng, d):
        return 0.0

    return actions, step, rollout


def test_picks_the_better_arm_by_expected_value():
    # depth 20: risky = .6*(-1) + .4*(-20) = -8.6 < safe -3
    actions, step, rollout = toy()
    best, stats = mcts_plan(actions, step, rollout, "s",
                            iters=1500, depth=20, seed=1)
    assert best == "safe"
    assert stats["risky"][0] > 50      # scaled c keeps exploring
    # depth 4: risky = .6*(-1)+.4*(-4) = -2.2 > safe -3: flips
    best, _ = mcts_plan(actions, step, rollout, "s",
                        iters=1500, depth=4, seed=1)
    assert best == "risky"


def test_terminal_root_returns_none():
    actions, step, rollout = toy()
    best, stats = mcts_plan(actions, step, rollout, "g",
                            iters=50, depth=10, seed=1)
    assert best is None and stats == {}


def trap():
    """Corridor trap (the stage-14 finding's shape): arm 'a' pays
    +50 only through THREE consecutive coherent moves (m0,m0,m0 in a
    3-action corridor; any wrong move dead-ends at 0); arm 'b' pays
    a flat +15. An INFORMED rollout (the base policy walks the
    corridor — the PI-informed shape) values corridor states at
    their true 50; a uniform rollout worths them (1/3)^remaining and
    walks into 'b'. Coherent multi-step payoff is exactly what
    uninformed rollouts systematically undervalue."""
    L, K, FLAT = 3, 3, 15.0

    def actions(s):
        if s == "s":
            return ["a", "b"]
        if isinstance(s, int):
            return [f"m{i}" for i in range(K)]
        return []

    def step(s, a, rng, d):
        if s == "s":
            return (0, 0.0, False) if a == "a" \
                else ("g", FLAT, True)
        if a == "m0":
            nxt = s + 1
            return ("g", 50.0, True) if nxt == L \
                else (nxt, 0.0, False)
        return ("g", 0.0, True)

    def informed(s, rng, d):
        return 50.0 if isinstance(s, int) else 0.0

    def uniform(s, rng, d):
        if not isinstance(s, int):
            return 0.0
        for _ in range(L - s):
            if rng.randrange(K) != 0:
                return 0.0
        return 50.0

    return actions, step, informed, uniform


def test_informed_rollouts_beat_the_trap_at_small_budgets():
    """Both runs OBEY the c_uct doctrine (scale 50, the return
    range) — the only difference is the rollout, which is the
    point."""
    actions, step, informed, uniform = trap()
    for seed in (0, 1, 2):
        best_i, _ = mcts_plan(actions, step, informed, "s",
                              iters=60, depth=4, seed=seed, c_uct=50)
        best_u, _ = mcts_plan(actions, step, uniform, "s",
                              iters=60, depth=4, seed=seed, c_uct=50)
        assert best_i == "a", seed    # the doctrine
        assert best_u == "b", seed    # the trap, pinned


def test_wall_clock_budget_is_anytime():
    import time as _t
    actions, step, informed, _ = trap()

    def slow_rollout(s, rng, d):
        _t.sleep(0.001)
        return informed(s, rng, d)

    t0 = _t.time()
    best, stats = mcts_plan(actions, step, slow_rollout, "s",
                            iters=10 ** 6, depth=2, seed=3,
                            budget_s=0.15)
    dt = _t.time() - t0
    assert dt < 1.0               # stopped by the clock, not iters
    assert best is not None and stats
