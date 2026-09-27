"""Bridge to the real IPPC 2011 CrossingTraffic environment."""

from typing import Optional, Set, Tuple

from .model import Topology


def load(instance: str = "1"):
    import pyRDDLGym
    from rddlrepository.core.manager import RDDLRepoManager
    prob = RDDLRepoManager().get_problem(
        "CrossingTraffic_MDP_ippc2011")
    env = pyRDDLGym.make(prob.get_domain(), prob.get_instance(instance))
    mdl = env.model
    g = mdl.ground_vars_with_values(mdl.non_fluents)

    def chain(pairs):
        nxt = dict(pairs)
        first = (set(nxt) - set(nxt.values())).pop()
        out = [first]
        while out[-1] in nxt:
            out.append(nxt[out[-1]])
        return out

    xs = chain([tuple(k.split("___")[1].split("__"))
                for k, v in g.items() if k.startswith("EAST___") and v])
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
    interior = ys[1:-1]
    ocells, east_of = [], {}
    for y in interior:
        row = [f"{x}_{y}" for x in xs]
        ocells += row
        for a, b in zip(row, row[1:]):
            east_of[a] = b
    goal = "_".join(next(k.split("___")[1].split("__")
                         for k, v in g.items()
                         if k.startswith("GOAL___") and v))
    topo = Topology(cells, adj, ocells, east_of, goal=goal,
                    input_rate=float(g["INPUT-RATE"]),
                    horizon=env.horizon)
    return {"env": env, "topo": topo}


def scene_of(obs) -> Tuple[Optional[str], Set[str]]:
    pos, obstacles = None, set()
    for k, v in obs.items():
        if not v:
            continue
        if k.startswith("robot-at___"):
            pos = "_".join(k.split("___")[1].split("__"))
        elif k.startswith("obstacle-at___"):
            obstacles.add("_".join(k.split("___")[1].split("__")))
    return pos, obstacles


def agent_policy(model):
    """Follow the converged joint-state table; fall back to a live
    choose for states outside it."""
    def act(obs):
        pos, obstacles = scene_of(obs)
        if pos is None or pos == model.topo.goal:
            return {}
        d = model.policy.get((pos, frozenset(obstacles)))
        if d is None:
            d = model.choose(pos, obstacles, ticks=12).action
        return {f"move-{d}": True}
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
