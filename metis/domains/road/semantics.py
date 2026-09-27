"""Road semantics: geometry -> actor relations -> RSS-discretized guards.

Implements the discrete d_min of docs/RSS-NOTES.md (exact worst-case series,
not the continuous formula), the pairwise star-shape action filter, and the
proper-response hysteresis (dangerous ahead => brake only; stopped in danger
must not start). rss_actions() is the single source of truth for "what may
ego do here" — the planner's per-state DOF measure, the LL policy filter and
the simulator-side check all call it."""

from typing import Dict, List, Optional

from .sim import ACTIONS, Car, Road, RoadState, gap_ahead, kinematics

# discretized RSS parameters (cells / tick^2); a_min_brake < a_max_brake
RHO = 1
A_MAX_ACCEL = 1
A_MIN_BRAKE = 1
A_MAX_BRAKE = 2


def travel_response_then_min_brake(v: int, v_max: int) -> int:
    """Worst-case travel of the rear car: accelerate for RHO ticks, then
    brake at A_MIN_BRAKE to a stop (accelerate-then-move per tick)."""
    d = 0
    for _ in range(RHO):
        v = min(v_max, v + A_MAX_ACCEL)
        d += v
    while v > 0:
        v = max(0, v - A_MIN_BRAKE)
        d += v
    # final tick at speed v' before braking is already counted; braking
    # series is v'-1 + v'-2 + ... + 0
    return d


def travel_max_brake(v: int) -> int:
    """Best-case escape of the front car braking as hard as A_MAX_BRAKE."""
    d = 0
    while v > 0:
        v = max(0, v - A_MAX_BRAKE)
        d += v
    return d


def d_min(v_rear: int, v_front: int, v_max: int) -> int:
    """Discrete RSS safe longitudinal distance (Lemma 2, exact series)."""
    return max(0, travel_response_then_min_brake(v_rear, v_max)
               - travel_max_brake(v_front))


def front_car(state: RoadState, lane: int, cell: int,
              exclude: str) -> Optional[Car]:
    best, best_gap = None, None
    for c in state.cars:
        if c.name == exclude or c.lane != lane:
            continue
        g = gap_ahead(state.road, cell, c.cell)
        if best_gap is None or g < best_gap:
            best, best_gap = c, g
    return best


def rear_car(state: RoadState, lane: int, cell: int,
             exclude: str) -> Optional[Car]:
    best, best_gap = None, None
    for c in state.cars:
        if c.name == exclude or c.lane != lane:
            continue
        g = gap_ahead(state.road, c.cell, cell)
        if best_gap is None or g < best_gap:
            best, best_gap = c, g
    return best


def long_safe(state: RoadState, rear_cell: int, v_rear: int,
              front: Optional[Car], pad: int = 0) -> bool:
    """pad: extra guard cells on top of d_min — the profile-supplied
    discretization pad. The bucket/cell rounding of the SI boundary is
    anti-conservative by up to 2 cells against the exact continuous
    formula (proven by lattice sweep, test_profile); live planners
    pass profile.dmin_pad_cells, the kernel default stays 0 so the
    discrete semantics (and goldens) are unchanged."""
    if front is None:
        return True
    g = gap_ahead(state.road, rear_cell, front.cell)
    return g >= d_min(v_rear, front.speed, state.road.v_max) + pad


def dangerous_ahead(state: RoadState, car: Car, pad: int = 0) -> bool:
    """Current longitudinal danger for `car` in its own lane."""
    return not long_safe(state, car.cell,
                         car.speed, front_car(state, car.lane, car.cell,
                                              car.name), pad)


def action_safe(state: RoadState, car: Car, action: str,
                pad: int = 0) -> bool:
    """Star-shape RSS check of one command (one-step verifiable, per the
    paper's inductive principle): the post-action state must be
    longitudinally safe toward the (possibly new) front car, and a lane
    change must additionally be safe for the target-lane rear car assuming
    it accelerates during the response time (rule 2: no reckless cut-ins).
    pad: profile discretization pad (see long_safe)."""
    nxt = kinematics(state.road, car, action)
    if nxt is None:
        return False
    front = front_car(state, nxt.lane, car.cell, car.name)
    if front is not None:
        # front car worst case: it brakes hard this tick while we act
        g_after = gap_ahead(state.road, nxt.cell,
                            (front.cell + max(0, front.speed - A_MAX_BRAKE))
                            % state.road.length)
        swept = (nxt.cell - car.cell) % state.road.length
        g_now = gap_ahead(state.road, car.cell, front.cell)
        if swept > g_now:                     # would pass through it
            return False
        if g_after < d_min(nxt.speed, max(0, front.speed - A_MAX_BRAKE),
                           state.road.v_max) + pad:
            return False
    if action in ("left", "right"):
        # a car directly beside us (same cell, target lane) is neither
        # "ahead" nor "behind" on the ring: forbid explicitly (mu-margin)
        if any(c.name != car.name and not c.wrecked
               and c.lane == nxt.lane and c.cell == car.cell
               for c in state.cars):
            return False
        rear = rear_car(state, nxt.lane, car.cell, car.name)
        if rear is not None:
            g = gap_ahead(state.road, rear.cell, car.cell)
            v_rear_worst = min(state.road.v_max, rear.speed + A_MAX_ACCEL)
            if g < d_min(v_rear_worst, nxt.speed,
                         state.road.v_max) + pad:
                return False
    return True


def rss_actions(state: RoadState, name: str,
                pad: int = 0) -> List[str]:
    """The RSS-allowed command set for a car — proper response included:
    if the current lane is already dangerous ahead, the response is to
    brake (a stopped car in danger must not start driving). This set's
    size is the state's degrees of freedom. pad: profile discretization
    pad — widens the guard comparisons; the doomed (collision-
    unavoidable) check stays unpadded, it is physics not margin."""
    car = state.car(name)
    if car.wrecked:
        return []
    if dangerous_ahead(state, car, pad):
        # longitudinal proper response: brake; at v=0 that means remain
        # stopped (a stopped car in danger must not start driving). DOF 0
        # ("doomed", the axiom's zero-freedom state) only when even the
        # response sweeps into the front car this tick.
        front = front_car(state, car.lane, car.cell, car.name)
        v_brake = max(0, car.speed - A_MIN_BRAKE)
        if front is not None:
            g_now = gap_ahead(state.road, car.cell, front.cell)
            front_worst = max(0, front.speed - A_MAX_BRAKE)
            if v_brake > g_now + front_worst:
                return []          # collision unavoidable: doomed
        return ["brake"]
    return [a for a in ACTIONS if action_safe(state, car, a, pad)]


def dof(state: RoadState, name: str) -> int:
    return len(rss_actions(state, name))


# ---------- occlusion (RSS Defs. 23-26, discretized) ----------

SIGHT = 10          # cells of forward sight per lane
SHADOW = 3          # a bulky vehicle's diagonal shadow into adjacent lanes
BULKY = ("truck",)  # vehicles that cast an adjacent-lane shadow
GHOST_SPEEDS = (0, 1)  # "reasonable" speeds of a hidden vehicle (Def. 24)


def hidden_cells(state: RoadState, observer: str, bulky=None):
    """Cells the observer cannot see, per lane, within SIGHT: every car
    blocks its own lane strictly beyond itself; BULKY vehicles
    additionally shadow the adjacent lanes alongside/shortly beyond
    themselves. bulky: the set of bulky actor NAMES — the default
    keeps the historical name-based constant; live planners pass the
    detection layer's bulky class members (bridge names are vNNN)."""
    if bulky is None:
        bulky = BULKY
    ego = state.car(observer)
    n = state.road.length
    hidden = set()
    for c in state.cars:
        if c.name == observer:
            continue
        d = (c.cell - ego.cell) % n
        if d == 0 or d > SIGHT:
            continue
        # own-lane blocking: cells strictly beyond c
        for k in range(d + 1, SIGHT + 1):
            hidden.add((c.lane, (ego.cell + k) % n))
        if c.name in bulky:
            for lane in (c.lane - 1, c.lane + 1):
                if 0 <= lane < state.road.lanes:
                    for k in range(d, min(d + SHADOW, SIGHT) + 1):
                        hidden.add((lane, (ego.cell + k) % n))
    return hidden


def visible_state(state: RoadState, observer: str = "ego",
                  bulky=None) -> RoadState:
    """What the observer actually sees: cars in hidden cells are absent.
    (The planner plans on this; the simulator keeps the whole truth.)"""
    hidden = hidden_cells(state, observer, bulky)
    cars = tuple(c for c in state.cars
                 if c.name == observer or (c.lane, c.cell) not in hidden)
    return RoadState(state.road, cars)


def occlusion_frontiers(state: RoadState, observer: str = "ego",
                        bulky=None):
    """The nearest hidden cell per lane — the worst-case position of a
    hidden vehicle (RSS: assume an occluded object anywhere it could
    reasonably be; the nearest dominates for safety)."""
    ego = state.car(observer)
    n = state.road.length
    hidden = hidden_cells(state, observer, bulky)
    frontiers = []
    for lane in range(state.road.lanes):
        for k in range(1, SIGHT + 1):
            cell = (ego.cell + k) % n
            if (lane, cell) in hidden:
                frontiers.append((lane, cell))
                break
    return frontiers
