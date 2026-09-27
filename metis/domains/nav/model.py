"""Navigation (IPPC 2011 MDP) — the exact-depth showcase, and the
first domain whose CATALOG IS THE DOMAIN: the .llp text (generated per
instance: topology as Horn facts, survival and policy as weight ports)
is the entire model; planning is EXACT POLICY ITERATION where policy
evaluation is the metis filter and improvement is kernel.plan over
first moves.

Dynamics (competition file): the robot occupies one cell; a move to an
adjacent cell succeeds with Bernoulli(1-P(target)) and otherwise the
robot vanishes (absorbing dead, paying -1 per remaining step); moves
off the grid stay put deterministically; the goal cell is absorbing.
Reward each step: -1 unless the goal is occupied.

The single-token state is why exact DEPTH is trivial here: a K-tick
projection's support is at most |cells|+2 states — precisely the
regime where sampling planners are weakest and exactness is free."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from metis.kernel import ForwardProjection, Program, plan
from metis.lang import compile_catalog, parse, weight_policy

DIRS = ("north", "south", "east", "west")


@dataclass
class Topology:
    cells: List[str]                       # lowercase composite names
    adj: Dict[Tuple[str, str], str]        # (cell, dir) -> target cell
    p_die: Dict[str, float]
    goal: str
    horizon: int = 40
    # filled by improve_policy: cell -> the dir to take
    policy: Dict[str, str] = field(default_factory=dict)


# ---- the generated catalog --------------------------------------------

def gen_catalog(topo: Topology) -> str:
    lines = [
        "% GENERATED per instance — Navigation: the catalog IS the",
        "% domain (topology as Horn facts, survival/policy as ports).",
        "catalog nav 1.",
        'provenance hand "generated from IPPC 2011 Navigation" '
        '"2026-07-18".',
        "layers (dyn).",
        "stages (nav).",
        f"type cell {{{' '.join(topo.cells)}}}.",
        "namespace loc produce (dyn).",
        "pred at(cell) : loc.",
        "pred dead : loc.",
    ]
    for d in DIRS:
        lines.append(f"bwd {d}(cell,cell).")
    for (c, d), c2 in sorted(topo.adj.items()):
        lines.append(f"{d}({c},{c2}).")
    lines += ["input pos.", "output move.",
              # empty reads: the bindings are pure functions of the
              # port args — declared, so the factored emission admits
              "weight go(C1,C2) reads {}.",
              "weight die(C1,C2) reads {}."]
    lines.append("stage nav {")
    for d in DIRS:
        lines.append(f"  %% move {d}: arrive with survival probability.")
        lines.append(f"  mv_{d} [dyn] : at(C) * {d}(C,C2) -o at(C2)"
                     f"  @w go(C,C2).")
        lines.append(f"  %% move {d}: the cell claims the robot.")
        lines.append(f"  die_{d} [dyn] : at(C) * {d}(C,C2) -o dead"
                     f"  @w die(C,C2).")
    lines.append("}")
    return "\n".join(lines) + "\n"


NAV_DOMAIN = "nav_domain.llp"


def gen_domain() -> str:
    """The LIFTED domain — instance-independent: rules Π-quantified
    over the cell type the INSTANCE declares (the RDDL
    domain/instance split, via literal include)."""
    lines = [
        "% Navigation LIFTED domain — included by generated instances.",
        "catalog nav-domain 1.",
        'provenance hand "IPPC 2011 Navigation, lifted domain" '
        '"2026-07-18".',
        "layers (dyn).",
        "stages (nav).",
        "namespace loc produce (dyn).",
        "pred at(cell) : loc.",
        "pred dead : loc.",
    ]
    for d in DIRS:
        lines.append(f"bwd {d}(cell,cell).")
    lines += ["input pos.", "output move.",
              # empty reads: the bindings are pure functions of the
              # port args — declared, so the factored emission admits
              "weight go(C1,C2) reads {}.",
              "weight die(C1,C2) reads {}."]
    lines.append("stage nav {")
    for d in DIRS:
        lines.append(f"  %% move {d}: arrive with survival probability.")
        lines.append(f"  mv_{d} [dyn] : at(C) * {d}(C,C2) -o at(C2)"
                     f"  @w go(C,C2).")
        lines.append(f"  %% move {d}: the cell claims the robot.")
        lines.append(f"  die_{d} [dyn] : at(C) * {d}(C,C2) -o dead"
                     f"  @w die(C,C2).")
    lines.append("}")
    return "\n".join(lines) + "\n"


def gen_instance(topo: Topology) -> str:
    """The INSTANCE: objects (the cell type) + non-fluents (adjacency
    facts) + include of the lifted domain."""
    lines = ["% GENERATED instance — Navigation.",
             "catalog nav 1.",
             'provenance hand "generated from IPPC 2011 Navigation" '
             '"2026-07-18".',
             f'include "{NAV_DOMAIN}".',
             f"type cell {{{' '.join(topo.cells)}}}."]
    for (c, d), c2 in sorted(topo.adj.items()):
        lines.append(f"{d}({c},{c2}).")
    return "\n".join(lines) + "\n"


def nav_registry_weights(topo: Topology):
    """go/die bindings: base-policy weight of the connecting direction
    times survival/death of the target. The policy table is read LIVE
    from topo.policy (uniform over all dirs when unset), so policy
    iteration just mutates the table."""
    dir_of = {(c, c2): d for (c, d), c2 in topo.adj.items()}

    def w_dir(c, c2):
        d = dir_of[(c, c2)]
        if c == topo.goal:
            return 0.0                      # goal absorbs
        pol = topo.policy.get(c)
        if pol is None:
            return 1.0                      # uniform base
        return 1.0 if d == pol else 0.0

    def go(args, counts):
        c, c2 = args
        return w_dir(c, c2) * (1.0 - topo.p_die[c2])

    def die(args, counts):
        c, c2 = args
        return w_dir(c, c2) * topo.p_die[c2]

    return {"go": go, "die": die}


def nav_registry(topo: Topology):
    from metis.fw import Registry
    reg = Registry()
    reg.input("pos", lambda v: v)
    reg.output("move", lambda res, sit: res.action)
    for name, fn in nav_registry_weights(topo).items():
        reg.weight(name, fn)
    return reg


# ---- projection machinery (ground once, re-init per query) ------------

class NavModel:
    def __init__(self, topo: Topology):
        from pathlib import Path
        from metis.lang import resolve_includes
        self.topo = topo
        cat_dir = Path(__file__).parent / "catalogs"
        cat_dir.mkdir(exist_ok=True)
        dom_path = cat_dir / NAV_DOMAIN
        if not dom_path.exists() or dom_path.read_text() != gen_domain():
            dom_path.write_text(gen_domain())
        ast_cat = resolve_includes(parse(gen_instance(topo)), cat_dir)
        doc = compile_catalog(ast_cat)
        self.doc = doc
        probe = doc.catalog.ground({"at(%s)" % topo.cells[0]: 1})
        self.stages, self.links = probe.stages, probe.links
        self._bindings = nav_registry_weights(topo)

    def project(self, cell: str, steps: int) -> ForwardProjection:
        prog = Program(stages=self.stages, links=self.links,
                       init_stage="nav", init={f"at({cell})": 1})
        pol = weight_policy(prog, self.doc.ports, self._bindings)
        return ForwardProjection(prog, steps, policy=pol)

    def expected_cost(self, cell: str, steps: int) -> float:
        """E[# of the next `steps` decision epochs whose PRE-move state
        has the goal unoccupied] — the RDDL convention: reward is
        evaluated on (s_t, a_t), so the arrival step still pays -1
        (found when V^pi came out ABOVE the VI optimum: the filter was
        counting post-move slices, one epoch optimistic)."""
        if cell == self.topo.goal:
            return 0.0
        pr = self.project(cell, steps - 1) if steps > 1 else None
        goal_atom = (f"at({self.topo.goal})", 1)
        cost = 0.0
        for t in range(steps):
            if t == 0:
                cost += 1.0        # cell != goal
                continue
            p_goal = sum(p for (_, k), p in pr.history[t].items()
                         if goal_atom in k)
            cost += 1.0 - p_goal
        return cost


def first_move_scores(model: NavModel, cell: str, steps: int):
    """kernel.plan over the four first moves; the committed move is
    closed-form (deterministic target + Bernoulli survival), the rest
    of the horizon is the exact measure under the current policy."""
    topo = model.topo

    def outcome(d):
        c2 = topo.adj.get((cell, d), cell)      # off-grid: stay
        p = topo.p_die[c2] if c2 != cell else 0.0
        return c2, p

    def score_of(d):
        c2, p = outcome(d)
        rest = steps - 1
        cost_now = 1.0            # pre-move state pays (cell != goal)
        cost_dead = float(rest)
        cost_alive = model.expected_cost(c2, rest)
        return (cost_now + p * cost_dead + (1.0 - p) * cost_alive,)

    return plan(list(DIRS), do=lambda d: d, project=lambda d: d,
                score=score_of, doomed_action=None)


# ---- exact policy iteration -------------------------------------------

def improve_policy(topo: Topology, rounds: int = 8,
                   steps: Optional[int] = None) -> NavModel:
    """Policy iteration with EXACT evaluation: evaluation is the metis
    filter over the generated catalog; improvement is kernel.plan over
    first moves. Converges on these instances in a few rounds."""
    steps = steps or topo.horizon
    model = NavModel(topo)
    for _ in range(rounds):
        changed = False
        for c in topo.cells:
            if c == topo.goal:
                continue
            res = first_move_scores(model, c, steps)
            if topo.policy.get(c) != res.action:
                topo.policy[c] = res.action
                changed = True
        if not changed:
            break
    return model


def policy_value(model: NavModel, start: str) -> float:
    """Exact E[total reward] of the current policy from `start`
    (negative: -1 per goal-unoccupied step over the horizon)."""
    return -model.expected_cost(start, model.topo.horizon)


# ---- the value-iteration ceiling (the honesty baseline) ---------------

def vi_optimal(topo: Topology) -> Tuple[float, Dict[str, str]]:
    """Finite-horizon VI on the tiny MDP — the true optimum our policy
    is measured against. Reward: -1 per step with the goal unoccupied
    (dead keeps paying)."""
    H = topo.horizon
    cells = topo.cells
    V = {c: 0.0 for c in cells}
    V["<dead>"] = 0.0
    pol: Dict[str, str] = {}
    for _ in range(H):
        V2: Dict[str, float] = {}
        pol = {}
        for c in cells:
            if c == topo.goal:
                V2[c] = V[c]                    # absorbing, cost 0
                continue
            best, barg = None, None
            for d in DIRS:
                c2 = topo.adj.get((c, d), c)
                p = topo.p_die[c2] if c2 != c else 0.0
                v = -1.0 + p * V["<dead>"] + (1.0 - p) * V[c2]
                if best is None or v > best:
                    best, barg = v, d
            V2[c] = best
            pol[c] = barg
        V2["<dead>"] = -1.0 + V["<dead>"]
        V = V2
    return V, pol
