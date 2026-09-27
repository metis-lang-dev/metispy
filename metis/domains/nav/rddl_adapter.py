"""Bridge to the real IPPC 2011 Navigation environment."""

from typing import Optional

from .model import Topology


def load(instance: str = "1"):
    import pyRDDLGym
    from rddlrepository.core.manager import RDDLRepoManager
    prob = RDDLRepoManager().get_problem("Navigation_MDP_ippc2011")
    env = pyRDDLGym.make(prob.get_domain(), prob.get_instance(instance))
    mdl = env.model
    g = mdl.ground_vars_with_values(mdl.non_fluents)
    xs = list(mdl.type_to_objects["xpos"])
    ys = list(mdl.type_to_objects["ypos"])

    def chain(pairs):                     # order an axis from its pairs
        nxt = dict(pairs)
        first = (set(nxt) - set(nxt.values())).pop()
        out = [first]
        while out[-1] in nxt:
            out.append(nxt[out[-1]])
        return out

    xs = chain([tuple(k.split("___")[1].split("__"))
                for k, v in g.items() if k.startswith("EAST___") and v]) \
        if any(k.startswith("EAST___") and v for k, v in g.items()) else xs
    ys = chain([tuple(k.split("___")[1].split("__"))
                for k, v in g.items() if k.startswith("NORTH___") and v])
    cells = [f"{x}_{y}" for x in xs for y in ys]
    adj = {}
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            c = f"{x}_{y}"
            if yi + 1 < len(ys):
                adj[(c, "north")] = f"{x}_{ys[yi + 1]}"
            if yi - 1 >= 0:
                adj[(c, "south")] = f"{x}_{ys[yi - 1]}"
            if xi + 1 < len(xs):
                adj[(c, "east")] = f"{xs[xi + 1]}_{y}"
            if xi - 1 >= 0:
                adj[(c, "west")] = f"{xs[xi - 1]}_{y}"
    p_die = {f"{x}_{y}": float(g.get(f"P___{x}__{y}", 0.0))
             for x in xs for y in ys}
    goal = next(k.split("___")[1].split("__")
                for k, v in g.items() if k.startswith("GOAL___") and v)
    topo = Topology(cells, adj, p_die, goal="_".join(goal),
                    horizon=env.horizon)
    return {"env": env, "topo": topo}


def position_of(obs) -> Optional[str]:
    for k, v in obs.items():
        if k.startswith("robot-at___") and v:
            return "_".join(k.split("___")[1].split("__"))
    return None                            # vanished


def agent_policy(topo: Topology):
    """Follow the converged policy table (improve_policy first)."""
    def act(obs):
        pos = position_of(obs)
        d = topo.policy.get(pos) if pos else None
        return {f"move-{d}": True} if d else {}
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
    for _ in range(spec["topo"].horizon):
        obs, r, term, trunc, _ = env.step(agent(obs))
        total += r
        if term or trunc:
            break
    return total
