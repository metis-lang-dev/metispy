"""Bridge to the real IPPC 2011 Game of Life environment."""

from typing import Set

from .model import greedy_set, greedy_set_factored


def load(instance: str = "1"):
    import pyRDDLGym
    from rddlrepository.core.manager import RDDLRepoManager
    prob = RDDLRepoManager().get_problem("GameOfLife_MDP_ippc2011")
    env = pyRDDLGym.make(prob.get_domain(), prob.get_instance(instance))
    mdl = env.model
    ground = mdl.ground_vars_with_values(mdl.non_fluents)
    xs = list(mdl.type_to_objects["x_pos"])
    ys = list(mdl.type_to_objects["y_pos"])
    cells = [f"{x}__{y}" for x in xs for y in ys]
    neighbors = {c: [] for c in cells}
    for k, v in ground.items():
        if k.startswith("NEIGHBOR___") and v:
            x1, y1, x2, y2 = k.split("___")[1].split("__")
            neighbors[f"{x1}__{y1}"].append(f"{x2}__{y2}")
    noise = {}
    for c in cells:
        x, y = c.split("__")
        noise[c] = float(ground.get(f"NOISE-PROB___{x}__{y}",
                                    ground.get("NOISE-PROB", 0.1)))
    return {"env": env, "cells": cells, "neighbors": neighbors,
            "noise": noise, "horizon": env.horizon}


def alive_of(obs) -> Set[str]:
    return {k.split("___")[1] for k, v in obs.items()
            if k.startswith("alive___") and v}


def agent_metis(spec):
    def act(obs):
        res = greedy_set(spec["cells"], spec["neighbors"],
                         spec["noise"], alive_of(obs))
        if res.action is None:
            return {}
        x, y = res.action.split("__")
        return {f"set___{x}__{y}": True}

    return act


def agent_metis_factored(spec):
    """The beyond-cliff agent: exact one-step lookahead through the
    factored evaluator — playable on instances the support filter
    cannot even project."""
    def act(obs):
        a = greedy_set_factored(spec["cells"], spec["neighbors"],
                                spec["noise"], alive_of(obs))
        if a is None:
            return {}
        x, y = a.split("__")
        return {f"set___{x}__{y}": True}

    return act


def agent_noop(spec):
    return lambda obs: {}


def agent_random(spec):
    from pyRDDLGym.core.policy import RandomAgent
    env = spec["env"]
    ra = RandomAgent(action_space=env.action_space,
                     num_actions=env.max_allowed_actions, seed=0)
    return lambda obs: ra.sample_action(obs)


def run_episode(spec, agent, seed: int) -> float:
    env = spec["env"]
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(spec["horizon"]):
        obs, r, term, trunc, _ = env.step(agent(obs))
        total += r
        if term or trunc:
            break
    return total
