"""THTS/UCT over sampled transitions — the depth half of the scale
bridge (SampledProjection is the breadth half).

Generic MCTS for receding-horizon MDP planning where the model is a
metis program: the domain supplies
    actions(state)          -> candidate list
    step(state, a, rng, d)  -> (state', reward, terminal); d is the
                               REMAINING depth, so absorbing-cost
                               states (dead pays every remaining
                               epoch) can charge their full toll
                               (typically: sample ONE model tick via
                               run_trace on the catalog program)
    rollout(state, rng, d)  -> return-to-go estimate (typically one
                               sampled d-tick trajectory under the
                               base policy — the same program, no cmd)

States must be hashable. UCB1 selection, unvisited-first expansion,
mean-value backup. Returns (best action, stats)."""

import math
import random
import time
from typing import Callable, Dict, Optional, Tuple


def mcts_plan(actions: Callable, step: Callable, rollout: Callable,
              root, iters: int = 500, depth: int = 12,
              c_uct: float = None, seed: int = 7,
              budget_s: Optional[float] = None):
    """Doctrine (stage 14, hardened stage 29):
    - c_uct must match the RETURN SCALE (UCB1 assumes [0,1] rewards);
      default = depth, the return range of a -1-per-epoch domain —
      found when 1.4 starved arms after one bad rollout.
    - rollouts must be INFORMED (the base-policy measure, PI-improved
      where available): at real-time budgets an uninformed rollout
      cannot see needle rewards behind an action fan-out, and the
      tree alone will not save it — pinned by the trap gate.
    - budget_s bounds wall clock the ANYTIME way: the tree simply
      stops growing and answers from what it has — never a mid-flight
      kill (the admission doctrine's sibling; calibrate iters from
      the measured per-simulation cost when determinism matters).
      iters still caps total simulations when both are given."""
    if c_uct is None:
        c_uct = float(depth)
    rng = random.Random(seed)
    tree: Dict[object, Dict[object, Tuple[int, float]]] = {}

    def simulate(s, d):
        if d <= 0:
            return 0.0
        node = tree.get(s)
        if node is None:
            acts = actions(s)
            tree[s] = {a: (0, 0.0) for a in acts}
            if not acts:
                return 0.0
            return rollout(s, rng, d)
        if not node:
            return 0.0
        untried = [a for a, (n, _) in node.items() if n == 0]
        if untried:
            a = untried[rng.randrange(len(untried))]
        else:
            total = sum(n for n, _ in node.values())
            logt = math.log(total)
            a = max(node, key=lambda x: node[x][1] / node[x][0]
                    + c_uct * math.sqrt(logt / node[x][0]))
        s2, r, terminal = step(s, a, rng, d)
        v = r if terminal else r + simulate(s2, d - 1)
        n, w = node[a]
        node[a] = (n + 1, w + v)
        return v

    t0 = time.time()
    for _ in range(iters):
        if budget_s is not None and time.time() - t0 >= budget_s:
            break
        simulate(root, depth)
    node = tree.get(root, {})
    visited = {a: (n, w) for a, (n, w) in node.items() if n > 0}
    if not visited:
        return None, {}
    best = max(visited, key=lambda a: visited[a][1] / visited[a][0])
    stats = {a: (n, w / n) for a, (n, w) in visited.items()}
    return best, stats
