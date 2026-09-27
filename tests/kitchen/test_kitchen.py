"""Overcooked strategic-layer gates. Fast: the authored catalog
round-trips and the measure orders every recipe situation correctly.
Slow: real cramped_room episodes — the metis strategist with their
GreedyHumanModel partner must match the greedy expert pair and crush
an unhelpful partner baseline."""

import importlib.util

import pytest

from metis.lang import parse, pretty

requires_oc = pytest.mark.skipif(
    importlib.util.find_spec("overcooked_ai_py") is None,
    reason="overcooked-ai not installed")


def test_catalog_roundtrips():
    from metis.domains.kitchen.model import CATALOG
    a = parse(CATALOG.read_text())
    assert parse(pretty(a)) == a


def test_intent_ordering_covers_the_recipe():
    from metis.domains.kitchen.model import KitchenModel, abstract
    from metis.domains.kitchen.agent import default_durations
    m = KitchenModel(default_durations())
    cases = [((0, False, False, "none"), "fetch_onion"),
             ((1, False, False, "none"), "fetch_onion"),
             ((2, False, False, "onion"), "load"),
             ((3, False, False, "none"), "cook"),
             ((0, True, False, "none"), "fetch_dish"),
             ((0, True, False, "dish"), "plate"),
             ((0, False, False, "soup"), "serve")]
    for (po, ck, rd, held), want in cases:
        res = m.choose(abstract(po, ck, rd, held))
        assert res.action == want, (po, ck, held, res.action, want)


@pytest.mark.slow
@requires_oc
def test_cramped_room_parity_with_expert_pair():
    import random as pyrandom
    import statistics
    import numpy as np
    np.Inf = np.inf
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.planning.planners import \
        MediumLevelActionManager, NO_COUNTERS_PARAMS
    from overcooked_ai_py.agents.agent import Agent, AgentPair, \
        GreedyHumanModel
    from overcooked_ai_py.mdp.actions import Action
    from metis.domains.kitchen.agent import MetisStrategist

    class LocalRandom(Agent):
        # their RandomAgent trips numpy 2 (np.random.choice on tuples)
        def action(self, state):
            return pyrandom.choice(Action.ALL_ACTIONS), {}

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    mlam = MediumLevelActionManager.from_pickle_or_compute(
        mdp, NO_COUNTERS_PARAMS, force_compute=False)

    def episodes(mk, n=4):
        out = []
        for i in range(n):
            np.random.seed(1000 + i)
            pyrandom.seed(1000 + i)
            env = OvercookedEnv.from_mdp(mdp, horizon=400)
            # run_agents -> (traj, time, SPARSE reward, shaped); [1]
            # is TIME (always == horizon — the saturated-looking 400s
            # that fooled the first protocol)
            out.append(env.run_agents(mk(),
                                      include_final_state=False)[2])
        return out

    from overcooked_ai_py.agents.agent import StayAgent
    means = {}
    for name, me, partner in (
            ("metis+greedy", lambda: MetisStrategist(mlam),
             lambda: GreedyHumanModel(mlam)),
            ("greedy+greedy", lambda: GreedyHumanModel(mlam),
             lambda: GreedyHumanModel(mlam)),
            ("random+greedy", LocalRandom,
             lambda: GreedyHumanModel(mlam)),
            ("metis+stay", lambda: MetisStrategist(mlam), StayAgent),
            ("greedy+stay", lambda: GreedyHumanModel(mlam), StayAgent),
            ("random+stay", LocalRandom, StayAgent)):
        means[name] = statistics.mean(
            episodes(lambda: AgentPair(me(), partner())))
    print(f"\ncramped_room mean sparse reward: {means}")
    # parity with THEIR expert, both cooperative and solo
    assert means["metis+greedy"] >= 0.95 * means["greedy+greedy"]
    assert means["metis+stay"] >= 0.95 * means["greedy+stay"]
    assert means["metis+greedy"] > means["random+greedy"] + 50
    assert means["metis+stay"] > means["random+stay"] + 100
