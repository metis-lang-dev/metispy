"""Bridge to the real IPPC 2011 SysAdmin environment (pyRDDLGym +
rddlrepository): read the instance's topology and constants, drive
episodes with the metis exact-measure greedy agent, and compare against
pyRDDLGym's own RandomAgent and NoOpAgent baselines."""

from typing import Dict, List, Set

from .model import greedy_plan


def load(instance: str = "1"):
    import pyRDDLGym
    from rddlrepository.core.manager import RDDLRepoManager
    prob = RDDLRepoManager().get_problem("SysAdmin_MDP_ippc2011")
    env = pyRDDLGym.make(prob.get_domain(), prob.get_instance(instance))
    mdl = env.model
    ground = mdl.ground_vars_with_values(mdl.non_fluents)
    computers = list(mdl.type_to_objects["computer"])
    in_neighbors: Dict[str, List[str]] = {c: [] for c in computers}
    for k, v in ground.items():
        if k.startswith("CONNECTED___") and v:
            y, x = k.split("___")[1].split("__")
            in_neighbors[x].append(y)
    return {"env": env, "computers": computers,
            "in_neighbors": in_neighbors,
            "reboot_prob": float(ground["REBOOT-PROB"]),
            "penalty": float(ground["REBOOT-PENALTY"]),
            "horizon": env.horizon}


def running_of(obs) -> Set[str]:
    return {k.split("___")[1] for k, v in obs.items()
            if k.startswith("running___") and v}


def metis_action(spec, obs) -> dict:
    res = greedy_plan(spec["computers"], spec["in_neighbors"],
                      running_of(obs), reboot_prob=spec["reboot_prob"],
                      penalty=spec["penalty"])
    if res.action is None:
        return {}
    return {f"reboot___{res.action}": True}


def run_episode(spec, agent, seed: int) -> float:
    env = spec["env"]
    obs, _ = env.reset(seed=seed)
    total = 0.0
    for _ in range(spec["horizon"]):
        obs, r, terminated, truncated, _ = env.step(agent(obs))
        total += r
        if terminated or truncated:
            break
    return total


def agent_metis(spec):
    return lambda obs: metis_action(spec, obs)


def agent_noop(spec):
    return lambda obs: {}


def agent_random(spec):
    from pyRDDLGym.core.policy import RandomAgent
    env = spec["env"]
    ra = RandomAgent(action_space=env.action_space,
                     num_actions=env.max_allowed_actions, seed=0)
    return lambda obs: ra.sample_action(obs)
