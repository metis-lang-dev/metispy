"""gridcat — the semantic compiler for the LIFTED lolli recode
(catalogs/grid_domain.llp): road geometry becomes Horn tables
(ring advance, speed lattice, lane HANDLE adjacency, per-actor
windows, crash pairs), the scene becomes the init multiset, and the
whole thing grounds program_key-IDENTICAL to lldsl.build_program —
the recode proof, same pattern as mini.llp.

The merge with the handle encoding (docs/road-encoding.md): the
lateral vocabulary IS the handle relations (leftof/rightof, shared
with handles_domain.llp and producible from Lane objects), while the
longitudinal grid classes are compiler outputs of geometry — the
grid is a choice of semantic classes inside the handle discipline,
now literally."""

from pathlib import Path
from typing import Dict, List, Tuple

from metis.lang import compile_catalog, parse, resolve_includes

from .sim import Car, RoadState

GRID_DOMAIN = Path(__file__).parent / "catalogs" / "grid_domain.llp"

GHOST_SPEEDS = (0, 1)


# -- state/window helpers (absorbed from the retired lldsl oracle;
#    stage 42 — the catalog is the only road encoding) ------------------

def decode_state(road, counts: Dict[str, int]) -> RoadState:
    """Rebuild a RoadState from a projection state's atom multiset."""
    cars = []
    for a in counts:
        if a.startswith("st("):
            name, lane, cell, v = a[3:-1].split(",")
            cars.append(Car(name, int(lane), int(cell), int(v)))
    return RoadState(road, tuple(cars))


def windows(state: RoadState, ticks: int,
            peds: Dict[str, int] = {},
            walkers: Dict[str, int] = {}) -> Dict[str, List[int]]:
    """Cells each car could occupy within the horizon (ring window).
    Vehicles reach forward only; holonomic actors (peds: name ->
    disc radius) reach symmetrically; directed walkers (walkers:
    name -> +-1) reach along their sign."""
    n = state.road.length
    reach = ticks * state.road.v_max + state.road.v_max
    out = {}
    for c in state.cars:
        if c.name in peds:
            r = peds[c.name] * (ticks + 1)
            out[c.name] = [(c.cell + k) % n
                           for k in range(-r, r + 1)]
        elif walkers.get(c.name, 1) < 0:
            out[c.name] = [(c.cell - k) % n for k in range(reach + 1)]
        else:
            out[c.name] = [(c.cell + k) % n for k in range(reach + 1)]
    return out


def slices_for(ticks: int, n_others: int) -> int:
    """An upper bound on forward-filter slices for `ticks` full
    ticks: per tick 1 ego move + n others + up to 1 crash + 3
    links, plus slack."""
    return ticks * (1 + n_others + 1 + 3) + 3


# -- objective readouts (absorbed with the retirement: scoring is
#    production machinery, not oracle rules) ----------------------------

def crash_probability(pr) -> float:
    d = pr.final_distribution("crash")
    return sum(d[1:])


def ego_turn_states(pr, road):
    """(probability, decoded state) at every future ego turn."""
    for dist in pr.history:
        for (stage, kappa), p in dist.items():
            if stage != "ego":
                continue
            counts = dict(kappa)
            if "eturn" not in counts:
                continue
            yield p, decode_state(road, counts)


def expected_dof(pr, road):
    from .semantics import rss_actions
    return sum(p * len(rss_actions(s, "ego"))
               for p, s in ego_turn_states(pr, road))


def expected_danger_time(pr, road):
    """E[# future ego turns spent longitudinally dangerous] — the
    paper's 'get back to a non-dangerous situation quickly', made a
    number."""
    from .semantics import dangerous_ahead
    return sum(p for p, s in ego_turn_states(pr, road)
               if dangerous_ahead(s, s.car("ego")))


def expected_progress(pr, road):
    """E[sum of ego speed over future ego turns] (cells traveled)."""
    return sum(p * s.car("ego").speed
               for p, s in ego_turn_states(pr, road))


def reachable_set_size(pr, road):
    """How many distinct ego configurations (lane, cell, speed)
    remain reachable at the last ego turn of the horizon — freedom
    as volume of futures rather than count of immediate moves."""
    last = None
    for dist in pr.history:
        egos = set()
        for (stage, kappa), p in dist.items():
            counts = dict(kappa)
            if stage == "ego" and "eturn" in counts and p > 0:
                c = decode_state(road, counts).car("ego")
                egos.add((c.lane, c.cell, c.speed))
        if egos:
            last = egos
    return len(last or ())


# Objective variants: score tuple AFTER P(crash); all lexicographic,
# larger-is-better entries negated.
OBJECTIVES = {
    # the original axiom, pure: maximize expected future freedom
    "dof": lambda pr, road: (-expected_dof(pr, road),),
    # paper p.15: first get out of danger fast, then freedom
    "escape": lambda pr, road: (expected_danger_time(pr, road),
                                -expected_dof(pr, road)),
    # anti-crawl: freedom as reachable volume (speed widens it)
    "reach": lambda pr, road: (-reachable_set_size(pr, road),
                               -expected_dof(pr, road)),
    # anti-crawl: freedom plus an explicit progress term
    "dof+progress": lambda pr, road: (-(expected_dof(pr, road)
                                        + expected_progress(pr,
                                                            road)),),
    # everything: escape danger, then volume, then progress
    "escape+reach": lambda pr, road: (expected_danger_time(pr, road),
                                      -reachable_set_size(pr, road),
                                      -expected_progress(pr, road)),
}


def tick_support(pr, road):
    """The measure's support one full tick ahead: {car-config: prob}
    at the first future ego turn (configs are frozensets of
    (name,lane,cell,v))."""
    for dist in pr.history:
        found = {}
        for (stage, kappa), p in dist.items():
            counts = dict(kappa)
            if stage == "ego" and "eturn" in counts and p > 0:
                st = decode_state(road, counts)
                key = frozenset((c.name, c.lane, c.cell, c.speed)
                                for c in st.cars)
                found[key] = found.get(key, 0.0) + p
        if found:
            return found
    return {}


def ghost_cells(state: RoadState, ticks: int, frontiers) -> List[int]:
    n, reach = state.road.length, ticks * state.road.v_max + \
        state.road.v_max
    return sorted({(c + k) % n for _, c in frontiers
                   for k in range(reach + 1)})


def gen_instance(state: RoadState, ticks: int,
                 walkers: Dict[str, int] = {},
                 peds: Dict[str, int] = {},
                 frontiers=None, domain: Path = GRID_DOMAIN) -> str:
    road = state.road
    win = windows(state, ticks, peds, walkers)
    names = [c.name for c in state.cars]
    others = [n for n in names if n != "ego"]
    ghosts = bool(frontiers)
    # ghost is ALWAYS car vocabulary: the domain's occlusion rules
    # (gapp/gabsent/ghit) name it unconditionally, and a constant must
    # be a member of its declared type (spec §5). Without frontiers
    # those rules stay dead (no maybe/occ, no pos(ghost)); only the
    # ghost's facts and turn token below remain frontier-gated.
    car_consts = list(names) + ["ghost"]
    lines = [
        "% GENERATED instance — grid tables from geometry.",
        "catalog road-scene-grid 1.",
        'provenance hand "gridcat compiler output" "2026-07-20".',
        f'include "{domain}".',
        f"type car {{{' '.join(car_consts)}}}.",
        f"type lane {{0 .. {road.lanes - 1}}}.",
        f"type cell {{0 .. {road.length - 1}}}.",
        f"type speed {{0 .. {road.v_max}}}.",
    ]
    for x in range(road.length):
        for v in range(road.v_max + 1):
            lines.append(f"advf({x},{v},{(x + v) % road.length}).")
            lines.append(f"advb({x},{v},{(x - v) % road.length}).")
    for v in range(road.v_max + 1):
        lines.append(f"vup({v},{min(road.v_max, v + 1)}).")
        lines.append(f"vdn({v},{max(0, v - 1)}).")
        lines.append(f"vhb({v},{max(0, v - 2)}).")
        if v > 0:
            lines.append(f"vpos({v}).")
    for lane in range(road.lanes):
        if lane > 0:
            lines.append(f"leftof({lane - 1},{lane}).")
        if lane < road.lanes - 1:
            lines.append(f"rightof({lane + 1},{lane}).")
    for n in names:
        for x in win[n]:
            lines.append(f"win({n},{x}).")
    # real-real crash pairs, EXCEPT ped-ped (pedestrians pass each
    # other); ego-ghost is the separate ghit rule, so no ghost pair.
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a in peds and b in peds:
                continue
            lines.append(f"pair({a},{b}).")
    for n in others:
        if n in peds:
            lines.append(f"pd({n}).")
        elif n in walkers:
            lines.append(f"{'wplus' if walkers[n] > 0 else 'wminus'}"
                         f"({n}).")
        else:
            lines.append(f"veh({n}).")
    # ped disc tables (single-tick manifest reach + lateral handles)
    for n, radius in peds.items():
        for x in win[n]:
            for dc in range(-radius, radius + 1):
                lines.append(f"pdisc({n},{x},{(x + dc) % road.length}).")
    if peds:
        for lane in range(road.lanes):
            lines.append(f"plat({lane},{lane}).")
            if lane > 0:
                lines.append(f"plat({lane},{lane - 1}).")
            if lane < road.lanes - 1:
                lines.append(f"plat({lane},{lane + 1}).")
    turn_others = list(others)
    if ghosts:
        gcells = ghost_cells(state, ticks, frontiers)
        for x in gcells:
            lines.append(f"win(ghost,{x}).")
        lines.append("veh(ghost).")
        lines.append("occluded.")
        for lane, cell in frontiers:
            lines.append(f"frontier({lane},{cell}).")
        for v in GHOST_SPEEDS:
            lines.append(f"gspeed({v}).")
        turn_others.append("ghost")
    turns = " * ".join(["eturn"] + [f"ot({n})" for n in turn_others])
    lines.append("%% collision sweep done -> next tick's turn tokens.")
    lines.append(f"qui ce [dyn] : check * $run -o ego * {turns}.")
    return "\n".join(lines) + "\n"


def init_atoms(state: RoadState, start_stage: str = "ego",
               frontiers=None) -> Dict[str, int]:
    init = {"run": 1}
    others = [c.name for c in state.cars if c.name != "ego"]
    for c in state.cars:
        if not c.wrecked:
            init[f"st({c.name},{c.lane},{c.cell},{c.speed})"] = 1
            init[f"pos({c.name},{c.lane},{c.cell})"] = 1
    turn_others = list(others)
    if frontiers:
        init["maybe"] = 1
        for lane, cell in frontiers:
            init[f"occ({lane},{cell})"] = 1
        turn_others.append("ghost")
    for n in turn_others:
        init[f"ot({n})"] = 1
    if start_stage == "ego":
        init["eturn"] = 1
    return init


def build(state: RoadState, ticks: int,
          walkers: Dict[str, int] = {}, peds: Dict[str, int] = {},
          frontiers=None, start_stage: str = "ego",
          domain: Path = GRID_DOMAIN):
    """(CatalogDoc, ground Program) for the scene — the catalog road
    to the same object lldsl.build_program hand-assembles. `domain`
    selects the catalog file (default: the in-repo kernel fixture;
    chariot passes its canonical copy — docs/catalog-reconciliation.md)."""
    doc = compile_catalog(resolve_includes(
        parse(gen_instance(state, ticks, walkers, peds, frontiers,
                           domain=domain)),
        domain.parent))
    prog = doc.catalog.ground(
        init_atoms(state, start_stage, frontiers),
        init_stage=start_stage)
    return doc, prog


# ---- the policy as a REGISTRY (RSS/behavior/manifest/ghost) --------

DEFAULT_BEHAVIOR = {"keep": 0.55, "accel": 0.2, "brake": 0.15,
                    "hbrake": 0.1}
DEFAULT_WALKER = {"keep": 0.72, "accel": 0.08, "brake": 0.1,
                  "left": 0.05, "right": 0.05}
PED_STAY_W, GHOST_P = 0.5, 0.3


def road_bindings(state: RoadState, ticks: int,
                  behaviors: Dict[str, Dict[str, float]] = {},
                  walkers: Dict[str, int] = {},
                  peds: Dict[str, int] = {}, frontiers=None,
                  rogues=(), ego_vcap=None, dmin_pad: int = 0):
    """The make_policy logic as weight-port bindings — the RSS shield,
    behavior rows, danger response, holonomic manifest prior and the
    ghost/occlusion priors, all reading the decoded state from counts.
    Bindings are OPAQUE (RSS is global): consumed via weight_policy,
    not the factored evaluator (docs/road-encoding.md)."""
    from .semantics import dangerous_ahead, rss_actions
    behaviors = behaviors or {}
    peds = peds or {}
    road = state.road
    n_appear = max(1, len(frontiers or ()) * len(GHOST_SPEEDS))
    lanes_reach = min(3, road.lanes)

    def decode(counts):
        return decode_state(road, {a: c for a, c in counts.items()
                                   if c > 0})

    def emove(args, counts):
        act = args[0]
        st = decode(counts)
        allowed = rss_actions(st, "ego", pad=dmin_pad)
        if ego_vcap is not None and st.car("ego").speed >= ego_vcap \
                and "accel" in allowed:
            allowed.remove("accel")
        if not allowed:                    # doomed: don't filter
            return 1.0
        return 1.0 if act in allowed else 0.0

    def omove(args, counts):
        c, act = args
        if c == "ghost":                   # worst-case hypothesis
            return 1.0 if act in ("keep", "left", "right") else 0.0
        st = decode(counts)
        if c not in rogues and dangerous_ahead(st, st.car(c)):
            return {"brake": 0.7, "hbrake": 0.3}.get(act, 0.0)
        return behaviors.get(c, behaviors.get(
            "default", DEFAULT_BEHAVIOR)).get(act, 0.0)

    def wmove(args, counts):
        c, act = args
        return behaviors.get(c, behaviors.get(
            "default-walker", DEFAULT_WALKER)).get(act, 0.0)

    def pmove(args, counts):
        c, l0, x0, l2, x2 = args
        dl = int(l2) - int(l0)
        dc = (int(x2) - int(x0) + road.length // 2) % road.length \
            - road.length // 2
        radius = peds.get(c, 1)
        if dl == 0 and dc == 0:
            return PED_STAY_W
        n_cells = lanes_reach * (2 * radius + 1) - 1
        return (1.0 - PED_STAY_W) / max(1, n_cells)

    return {"emove": emove, "omove": omove, "wmove": wmove,
            "pmove": pmove,
            "gappear": lambda a, c: GHOST_P / n_appear,
            "gdecline": lambda a, c: 1.0 - GHOST_P}


# ---- the canonical Python binding: DOF planning over the catalog ---

def plan_seam(state: RoadState, ticks: int = 3,
              behaviors: Dict[str, Dict[str, float]] = {},
              objective: str = "dof", walkers=(),
              peds: Dict[str, int] = {}, occlusion=None,
              rogues=(), ego_vcap=None, dmin_pad: int = 0,
              bulky=None):
    """The kernel.plan 5-tuple (candidates, do, project, score,
    doomed) over the LIFTED catalog — the metis.fw.Evaluator seam.
    The program is the .llp catalog (gridcat.build), the policy is
    road_bindings; candidate preamble and objective READOUTS are the
    shared domain machinery, so the measure equals the oracle's."""
    from metis.kernel import ForwardProjection
    from metis.lang import weight_policy
    from .semantics import (occlusion_frontiers, rss_actions,
                            visible_state)
    from .sim import kinematics
    behaviors = behaviors or {}
    peds = peds or {}
    if not hasattr(walkers, "get"):
        walkers = {w: 1 for w in walkers}
    frontiers = None
    if occlusion in ("blind", "aware"):
        if occlusion == "aware":
            frontiers = occlusion_frontiers(state, "ego", bulky)
        state = visible_state(state, "ego", bulky)
    candidates = rss_actions(state, "ego", pad=dmin_pad)
    if ego_vcap is not None and state.car("ego").speed >= ego_vcap \
            and "accel" in candidates and len(candidates) > 1:
        candidates.remove("accel")
    score_fn = OBJECTIVES[objective]
    # +2 turns for a ghost (its appear + move cycle), mirroring
    # project_horizon's slice budget
    steps = slices_for(ticks, len(state.cars) - 1
                       + (2 if frontiers else 0))

    def do(a):
        ego2 = kinematics(state.road, state.car("ego"), a)
        return RoadState(state.road, tuple(
            ego2 if c.name == "ego" else c for c in state.cars))

    def project(mid):
        doc, prog = build(mid, ticks, walkers=walkers, peds=peds,
                          frontiers=frontiers, start_stage="others")
        reg = road_bindings(mid, ticks, behaviors=behaviors,
                           walkers=walkers, peds=peds,
                           frontiers=frontiers, rogues=rogues,
                           ego_vcap=ego_vcap, dmin_pad=dmin_pad)
        return ForwardProjection(
            prog, steps, policy=weight_policy(prog, doc.ports, reg))

    def score(pr):
        return (crash_probability(pr), *score_fn(pr, state.road))

    return (candidates, do, project, score, "brake"), frontiers


def plan(state: RoadState, ticks: int = 3, **kw):
    """Choose ego's command over the LIFTED catalog — the production
    Python binding (the analog of the C++ plan_metis). Reproduces the
    oracle planner.plan decision by construction; gated at the corpus
    level (tests/test_gridcat.py)."""
    from metis.kernel import plan as kernel_plan
    seam, frontiers = plan_seam(state, ticks, **kw)
    res = kernel_plan(*seam)
    if res.doomed:
        return {"action": "brake", "doomed": True, "dof": 0,
                "scores": {}, "frontiers": frontiers or []}
    return {"action": res.action, "doomed": False, "dof": res.dof,
            "scores": res.scores, "frontiers": frontiers or []}
