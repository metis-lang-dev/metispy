"""CrossingTraffic (IPPC 2011 MDP) — Frogger: obstacles stream west
along interior rows (Bernoulli(INPUT-RATE) arrivals at the east edge,
deterministic shift elsewhere, border rows safe); the robot dies iff
an obstacle occupies its cell at decision time; reward -1 per epoch
with the goal unoccupied.

Catalog-first, per the directive — NO python program builders:
the generated .llp is the whole model. Design notes:
- obstacles are a sync bank where p(next) reads the EAST neighbor's
  snapshot (a degenerate Bernoulli: 0/1 inland, INPUT-RATE at the
  edge) — the generic sync shape, authored in text with chain guards;
- the robot is a nav-style token in a `move` stage that runs BEFORE
  the shift (death checks PRE-state obstacles, per the RDDL);
- death, goal-absorption, first-move forcing (cmd tokens) and the base
  policy all live in WEIGHT-PORT BINDINGS reading the counts — no
  negative conditions, no python evaluation workarounds;
- cost is a `paid` token: exactly one move-stage event fires per tick
  and produces `paid` iff the epoch's pre-move state has the goal
  unoccupied (dead robots keep paying) — E[cost] is one final-count
  query, no slice bookkeeping.

Both evaluators run the SAME catalog: exact filtering while the
support fits, SampledProjection past it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from metis.kernel import ForwardProjection, Program, \
    SampledProjection, plan
from metis.lang import compile_catalog, parse, resolve_includes, \
    weight_policy

DIRS = ("north", "south", "east", "west")
XT_DOMAIN = Path(__file__).parent / "catalogs" / "xtraffic_domain.llp"


@dataclass
class Topology:
    cells: List[str]
    adj: Dict[Tuple[str, str], str]
    ocells: List[str]              # interior-row cells, WEST to EAST
    east_of: Dict[str, str]        # ocell -> its east neighbor ocell
    goal: str
    input_rate: float
    horizon: int = 40


def gen_catalog(topo: Topology) -> str:
    n = len(topo.ocells)
    idx = [f"i{k}" for k in range(n + 1)]
    lines = [
        "% GENERATED — CrossingTraffic: the catalog IS the domain.",
        "catalog xtraffic 1.",
        'provenance hand "generated from IPPC 2011 CrossingTraffic" '
        '"2026-07-18".',
        "layers (dyn).",
        "stages (move mark update sweep).",
        f"type cell {{{' '.join(topo.cells)}}}.",
        f"type idx {{{' '.join(idx)}}}.",
        "namespace state produce (dyn).",
    ]
    for pr, ty in (("rob", "cell"), ("o", "cell"), ("e", "cell"),
                   ("po", "cell"), ("pe", "cell"), ("tok", "cell"),
                   ("no2", "cell"), ("ne2", "cell")):
        lines.append(f"pred {pr}({ty}) : state.")
    for pr in ("dead", "paid", "mvt", "cmdnorth", "cmdsouth",
               "cmdeast", "cmdwest"):
        lines.append(f"pred {pr} : state.")
    for pr in ("m", "u", "s"):
        lines.append(f"pred {pr}(idx) : state.")
    for d in DIRS:
        lines.append(f"bwd {d}(cell,cell).")
        lines.append(f"bwd noadj_{d}(cell).")
    lines.append("bwd cidx(cell,idx).")
    lines.append("bwd succ(idx,idx).")
    lines.append("bwd ocell(cell).")
    for (c, d), c2 in sorted(topo.adj.items()):
        lines.append(f"{d}({c},{c2}).")
    for d in DIRS:
        for c in topo.cells:
            if (c, d) not in topo.adj:
                lines.append(f"noadj_{d}({c}).")
    for k, c in enumerate(topo.ocells):
        lines.append(f"cidx({c},i{k}).")
    for k in range(n):
        lines.append(f"succ(i{k},i{k + 1}).")
    for c in topo.ocells:
        lines.append(f"ocell({c}).")
    # declared read scopes — derived from bindings() below, so the
    # factored emission admits the catalog: obs reads the east
    # neighbor's snapshot; go/stayw read the cmd tokens, the own-cell
    # obstacle and the full configuration (the policy table's key);
    # the forced moves read only the own-cell obstacle
    cfg = ("cmdnorth, cmdsouth, cmdeast, cmdwest, o(C1), "
           "o(D) if ocell(D)")
    lines += ["input scene.", "output move.",
              "weight obs(C1) reads {po(D) if east(C1,D)}.",
              f"weight go(C1,C2) reads {{{cfg}}}.",
              f"weight stayw(C1) reads {{{cfg}}}.",
              "weight fgo(C1,C2) reads {o(C1)}.",
              "weight fstayw(C1) reads {o(C1)}."]

    g = topo.goal
    lines.append("stage move {")
    for d in DIRS:
        lines.append(f"  %% forced first move {d} (cmd token).")
        lines.append(f"  fmv_{d} [dyn] : mvt * rob(C) * cmd{d} * "
                     f"{d}(C,C2) -o rob(C2) * paid  @w fgo(C,C2).")
        lines.append(f"  %% forced first move {d} against the edge:"
                     f" stay.")
        lines.append(f"  fst_{d} [dyn] : mvt * rob(C) * cmd{d} * "
                     f"noadj_{d}(C) -o rob(C) * paid  @w fstayw(C).")
        lines.append(f"  %% base-policy move {d}.")
        lines.append(f"  mv_{d} [dyn] : mvt * rob(C) * {d}(C,C2)"
                     f" -o rob(C2) * paid  @w go(C,C2).")
    lines.append("  %% base-policy stay (edge mass).")
    lines.append("  str [dyn] : mvt * rob(C) -o rob(C) * paid"
                 "  @w stayw(C).")
    lines.append("  %% an obstacle on the robot's cell kills it "
                 "(pre-shift state, per the RDDL).")
    lines.append("  dier [dyn] : mvt * rob(C) * $o(C)"
                 " -o dead * paid.")
    lines.append(f"  %% goal is absorbing and free.")
    lines.append(f"  atgoal [dyn] : mvt * rob({g}) -o rob({g}).")
    lines.append("  %% a vanished robot keeps paying every epoch.")
    lines.append("  deadtick [dyn] : mvt * dead -o dead * paid.")
    lines.append("}")

    lines.append("stage mark {")
    lines.append("  %% snapshot an obstacle cell.")
    lines.append("  mrO [dyn] : m(I) * o(C) * cidx(C,I) * succ(I,I2)"
                 " -o po(C) * tok(C) * m(I2).")
    lines.append("  %% snapshot an empty obstacle cell.")
    lines.append("  mrE [dyn] : m(I) * e(C) * cidx(C,I) * succ(I,I2)"
                 " -o pe(C) * tok(C) * m(I2).")
    lines.append("}")
    lines.append("stage update {")
    lines.append("  %% next-state obstacle: east neighbor's snapshot "
                 "(edge: arrival Bernoulli).")
    lines.append("  upO [dyn] : u(I) * tok(C) * cidx(C,I) * succ(I,I2)"
                 " -o no2(C) * u(I2)  @w obs(C).")
    lines.append("  %% next-state empty.")
    lines.append("  upE [dyn] : u(I) * tok(C) * cidx(C,I) * succ(I,I2)"
                 " -o ne2(C) * u(I2)  @w ~obs(C).")
    lines.append("}")
    lines.append("stage sweep {")
    for snap, out, dest in (("po", "no2", "o"), ("po", "ne2", "e"),
                            ("pe", "no2", "o"), ("pe", "ne2", "e")):
        lines.append(f"  %% commit {snap}/{out}.")
        lines.append(f"  sw_{snap}_{out} [dyn] : s(I) * {snap}(C) * "
                     f"{out}(C) * cidx(C,I) * succ(I,I2)"
                     f" -o {dest}(C) * s(I2).")
    lines.append("}")
    lines.append("%% robot resolved -> shift obstacles.")
    lines.append(f"qui mm [dyn] : move -o mark * m(i0).")
    lines.append("%% snapshot done -> update pass.")
    lines.append(f"qui mu [dyn] : mark * m(i{n}) -o update * u(i0).")
    lines.append("%% update done -> commit.")
    lines.append(f"qui us [dyn] : update * u(i{n}) -o sweep * s(i0).")
    lines.append("%% committed -> next epoch.")
    lines.append(f"qui sm [dyn] : sweep * s(i{n}) -o move * mvt.")
    return "\n".join(lines) + "\n"


def gen_instance(topo: Topology) -> str:
    """The INSTANCE: objects (cell/idx types — idx via range sugar) +
    non-fluents (adjacency, chain, goal, lastidx) + include of the
    lifted domain. The RDDL-style domain/instance split; nav is the
    exemplar, this is the second occupant."""
    n = len(topo.ocells)
    lines = ["% GENERATED instance — CrossingTraffic.",
             "catalog xtraffic 1.",
             'provenance hand "generated from IPPC 2011 '
             'CrossingTraffic" "2026-07-18".',
             f'include "{XT_DOMAIN}".',
             f"type cell {{{' '.join(topo.cells)}}}.",
             f"type idx {{i0 .. i{n}}}."]
    for (c, d), c2 in sorted(topo.adj.items()):
        lines.append(f"{d}({c},{c2}).")
    for d in DIRS:
        for c in topo.cells:
            if (c, d) not in topo.adj:
                lines.append(f"noadj_{d}({c}).")
    for k, c in enumerate(topo.ocells):
        lines.append(f"cidx({c},i{k}).")
    for k in range(n):
        lines.append(f"succ(i{k},i{k + 1}).")
    lines.append(f"goal({topo.goal}).")
    lines.append(f"lastidx(i{n}).")
    for c in topo.ocells:
        lines.append(f"ocell({c}).")
    return "\n".join(lines) + "\n"


STEPS_PER_TICK_BASE = 6      # move + 4 links + ... computed below


def steps_for(topo: Topology, ticks: int) -> int:
    # per tick: 1 (move) + 1 (mm) + n (mark) + 1 (mu) + n (update)
    #           + 1 (us) + n (sweep) + 1 (sm)
    n = len(topo.ocells)
    return ticks * (3 * n + 5)


class XTrafficModel:
    def __init__(self, topo: Topology):
        self.topo = topo
        self.doc = compile_catalog(resolve_includes(
            parse(gen_instance(topo)), XT_DOMAIN.parent))
        probe = self.doc.catalog.ground({"rob(%s)" % topo.cells[0]: 1})
        self.stages, self.links = probe.stages, probe.links
        # (pos, frozenset(obstructed ocells)) -> dir; read LIVE by the
        # go/stayw bindings — policy iteration just mutates this
        self.policy: Dict[Tuple[str, frozenset], str] = {}
        self._bindings = self.bindings()

    # -- the weight-port bindings (ALL domain logic lives here) ---------

    def bindings(self):
        topo = self.topo

        def obs(args, counts):
            c = args[0]
            east = topo.east_of.get(c)
            if east is None:                      # east edge: arrivals
                return topo.input_rate
            return 1.0 if f"po({east})" in counts else 0.0

        def goal_ward(c, d):
            c2 = topo.adj.get((c, d))
            if c2 is None:
                return False
            return self._dist(c2) < self._dist(c)

        def any_cmd(counts):
            return any(f"cmd{d}" in counts for d in DIRS)

        def cfg_of(counts):
            return frozenset(c for c in topo.ocells
                             if f"o({c})" in counts)

        def go(args, counts):
            c, c2 = args
            if any_cmd(counts) or f"o({c})" in counts \
                    or c == topo.goal:
                return 0.0
            d = self._dir(c, c2)
            chosen = self.policy.get((c, cfg_of(counts)))
            if chosen is not None:
                return 1.0 if d == chosen else 0.0
            wards = [d2 for d2 in DIRS if goal_ward(c, d2)]
            return (1.0 / len(wards)) if d in wards else 0.0

        def stayw(args, counts):
            c = args[0]
            if any_cmd(counts) or f"o({c})" in counts \
                    or c == topo.goal:
                return 0.0
            chosen = self.policy.get((c, cfg_of(counts)))
            if chosen is not None:
                # chosen dir has no adjacency -> the policy says stay
                return 0.0 if (c, chosen) in topo.adj else 1.0
            wards = [d2 for d2 in DIRS if goal_ward(c, d2)]
            return 0.0 if wards else 1.0

        def fgo(args, counts):
            c, c2 = args
            return 0.0 if f"o({c})" in counts else 1.0

        def fstayw(args, counts):
            return 0.0 if f"o({args[0]})" in counts else 1.0

        return {"obs": obs, "go": go, "stayw": stayw, "fgo": fgo,
                "fstayw": fstayw}

    def _dir(self, c, c2):
        for d in DIRS:
            if self.topo.adj.get((c, d)) == c2:
                return d
        return None

    def _dist(self, c):
        cx, cy = c.split("_")
        gx, gy = self.topo.goal.split("_")
        xs = sorted({cc.split("_")[0] for cc in self.topo.cells})
        ys = sorted({cc.split("_")[1] for cc in self.topo.cells})
        return abs(xs.index(cx) - xs.index(gx)) + \
            abs(ys.index(cy) - ys.index(gy))

    # -- projection ------------------------------------------------------

    def init_atoms(self, pos: Optional[str], obstacles: Set[str],
                   cmd: Optional[str] = None) -> Dict[str, int]:
        init = {"mvt": 1}
        if pos is not None:
            init[f"rob({pos})"] = 1
        else:
            init["dead"] = 1
        for c in self.topo.ocells:
            init[f"o({c})" if c in obstacles else f"e({c})"] = 1
        if cmd:
            init[f"cmd{cmd}"] = 1
        return init

    def project(self, pos, obstacles, ticks, cmd=None,
                sampled: Optional[Tuple[int, int]] = None):
        prog = Program(stages=self.stages, links=self.links,
                       init_stage="move",
                       init=self.init_atoms(pos, obstacles, cmd))
        pol = weight_policy(prog, self.doc.ports, self._bindings)
        steps = steps_for(self.topo, ticks)
        if sampled:
            n, seed = sampled
            return SampledProjection(prog, steps, n=n, seed=seed,
                                     policy=pol)
        return ForwardProjection(prog, steps, policy=pol)

    def expected_cost(self, pr) -> float:
        return sum(p * dict(k).get("paid", 0)
                   for (_, k), p in pr.final.items())

    # -- the receding-horizon agent -------------------------------------

    def choose(self, pos, obstacles, ticks,
               sampled: Optional[Tuple[int, int]] = None):
        return plan(
            list(DIRS),
            do=lambda d: self.project(pos, obstacles, ticks, cmd=d,
                                      sampled=sampled),
            project=lambda pr: pr,
            score=lambda pr: (self.expected_cost(pr),),
            doomed_action=None)


# ---- the honesty ceiling ----------------------------------------------

def vi_optimal(topo: Topology):
    """Finite-horizon VI on (pos|dead) x obstacle-config. Exact
    transition: death iff obstacle at pos pre-state; shift west;
    Bernoulli arrival at the east ocell."""
    import itertools
    n = len(topo.ocells)
    configs = list(itertools.product((0, 1), repeat=n))
    rate = topo.input_rate

    def shift_dists(cfg):
        """[(new_cfg, prob)] — deterministic shift + edge arrival."""
        base = list(cfg[1:]) + [0]           # west to east; edge last
        out = []
        for arr in (0, 1):
            base2 = tuple(base[:-1] + [arr])
            p = rate if arr else 1.0 - rate
            out.append((base2, p))
        return out

    ocell_index = {c: i for i, c in enumerate(topo.ocells)}
    states = [(p, c) for p in topo.cells + ["<dead>"] for c in configs]
    V = {s: 0.0 for s in states}
    pol = {}
    for _ in range(topo.horizon):
        V2, pol = {}, {}
        for (p, cfg) in states:
            if p == topo.goal:
                V2[(p, cfg)] = 0.0 + _exp_shift(V, p, cfg, shift_dists)
                continue
            if p == "<dead>":
                V2[(p, cfg)] = -1.0 + _exp_shift(V, p, cfg, shift_dists)
                continue
            if p in ocell_index and cfg[ocell_index[p]]:
                V2[(p, cfg)] = -1.0 + _exp_shift(V, "<dead>", cfg,
                                                 shift_dists)
                continue
            best, barg = None, None
            for d in DIRS:
                p2 = topo.adj.get((p, d), p)
                v = -1.0 + _exp_shift(V, p2, cfg, shift_dists)
                if best is None or v > best:
                    best, barg = v, d
            V2[(p, cfg)] = best
            pol[(p, cfg)] = barg
        V = V2
    return V, pol


def _exp_shift(V, pos, cfg, shift_dists):
    return sum(pr * V[(pos, c2)] for c2, pr in shift_dists(cfg))


def improve_policy(model: XTrafficModel, rounds: int = 8,
                   ticks: int = 16):
    """Policy iteration on the JOINT (pos, obstacle-config) state —
    the nav pattern lifted: evaluation is the exact filter over the
    generated catalog (the policy table read live by the weight
    ports), improvement is kernel.plan over forced first moves."""
    import itertools
    topo = model.topo
    configs = [frozenset(c for c, b in zip(topo.ocells, bits) if b)
               for bits in itertools.product((0, 1),
                                             repeat=len(topo.ocells))]
    for _ in range(rounds):
        changed = False
        for pos in topo.cells:
            if pos == topo.goal:
                continue
            for cfg in configs:
                if pos in cfg:
                    continue                  # dead on arrival
                res = model.choose(pos, set(cfg), ticks)
                key = (pos, cfg)
                if model.policy.get(key) != res.action:
                    model.policy[key] = res.action
                    changed = True
        if not changed:
            break
    return model


# ---- THTS adapter: sampled 1-tick steps over the same catalog ---------

def _state_of_final(model, kappa):
    counts = dict(kappa)
    pos = None
    for c in model.topo.cells:
        if counts.get(f"rob({c})"):
            pos = c
            break
    cfg = frozenset(c for c in model.topo.ocells
                    if counts.get(f"o({c})"))
    return pos, cfg, counts.get("paid", 0)


def thts_choose(model: XTrafficModel, pos, obstacles,
                iters: int = 400, depth: int = 12, seed: int = 7):
    """UCT where step = ONE sampled model tick of the catalog program
    (cmd-forced) and rollout = one sampled base-policy trajectory —
    all measure machinery, no domain simulator."""
    from metis.kernel import mcts_plan, run_trace
    from metis.lang import weight_policy
    import random as _random

    def make_prog(p_, cfg, cmd):
        prog = Program(stages=model.stages, links=model.links,
                       init_stage="move",
                       init=model.init_atoms(p_, set(cfg), cmd))
        return prog, weight_policy(prog, model.doc.ports,
                                   model._bindings)

    tick = steps_for(model.topo, 1)

    def actions(state):
        p_, cfg = state
        if p_ == model.topo.goal:
            return []
        return list(DIRS)

    def step(state, a, rng, d):
        p_, cfg = state
        prog, pol = make_prog(p_, cfg, a)
        _, final = run_trace(prog, tick, rng, policy=pol)
        p2, cfg2, paid = _state_of_final(model, final[1])
        terminal = p2 == model.topo.goal
        if p2 is None:
            # dead: this epoch's paid PLUS every remaining epoch
            return (None, cfg2), -float(paid) - float(d - 1), True
        return (p2, cfg2), -float(paid), terminal

    def rollout(state, rng, d):
        p_, cfg = state
        if p_ is None:
            return -float(d)
        prog, pol = make_prog(p_, cfg, None)
        _, final = run_trace(prog, steps_for(model.topo, d), rng,
                             policy=pol)
        _, _, paid = _state_of_final(model, final[1])
        return -float(paid)

    best, stats = mcts_plan(actions, step, rollout,
                            (pos, frozenset(obstacles)),
                            iters=iters, depth=depth, seed=seed)
    return best, stats
