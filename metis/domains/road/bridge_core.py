"""CARLA bridge, the carla-independent core (unit-testable anywhere).

The corridor abstraction: a route-anchored window of cells along ego's
path plus one adjacent lane. Everything CARLA-specific reduces to plain
data before it reaches this module:

    CorridorMap   per lane, a list of cell anchor points (x, y, yaw) every
                  CELL_M meters along the route, ego sitting at EGO_CELL
    snapshot      [(name, lane, cell, speed_mps)] for actors mapped onto
                  the corridor (nearest anchor within capture radius)

From those we build the planner's RoadState (the window is a ring far
larger than any d_min, so ring semantics are harmless), run plan(), and
translate the decision back into a *setpoint*: a corridor anchor + target
speed for the low-level pilot (MPC/PID). This is the behavior/motion
split: the LL layer decides lane, gap and speed bound; the pilot tracks.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .profile import CAR, EnvProfile
from .sim import Car, Road, RoadState

# module constants = the CAR profile (import-compat; new code takes an
# EnvProfile — rss.profile — and the wheelchair pack selects its own)
CELL_M = CAR.cell_m
TICK_S = CAR.tick_s
EGO_CELL = CAR.ego_cell
WINDOW = CAR.window
CAPTURE_M = CAR.capture_m
V_MAX = CAR.v_max

LOOKAHEAD = {     # setpoint distance (cells ahead) per decided action
    "keep": 3, "accel": 4, "brake": 2, "left": 3, "right": 3}


@dataclass
class Anchor:
    x: float
    y: float
    yaw: float = 0.0


@dataclass
class CorridorMap:
    """lane index -> WINDOW anchors along the route (lane 0 = ego's)."""
    lanes: Dict[int, List[Optional[Anchor]]]

    def anchor(self, lane: int, cell: int) -> Optional[Anchor]:
        cells = self.lanes.get(lane)
        if cells is None or not (0 <= cell < len(cells)):
            return None
        return cells[cell]


def speed_to_cells(speed_mps: float, profile: EnvProfile = CAR) -> int:
    return profile.speed_to_cells(speed_mps)


def cells_to_speed(v_cells: int, profile: EnvProfile = CAR) -> float:
    return profile.cells_to_speed(v_cells)


def locate(corridor: CorridorMap, x: float, y: float,
           profile: EnvProfile = CAR, capture_m: float = None
           ) -> Optional[Tuple[int, int, float]]:
    """Nearest corridor anchor to a world point: (lane, cell, distance);
    None when outside the capture radius of every anchor."""
    if capture_m is None:
        capture_m = profile.capture_m
    best = None
    for lane, anchors in corridor.lanes.items():
        for cell, a in enumerate(anchors):
            if a is None:
                continue
            d = math.hypot(x - a.x, y - a.y)
            if best is None or d < best[2]:
                best = (lane, cell, d)
    if best is None or best[2] > capture_m:
        return None
    return best


def snapshot_to_state(actors: List[Tuple[str, float, float, float]],
                      corridor: CorridorMap,
                      ego_speed_mps: float, ego_lane: int = 0,
                      ego_cell: Optional[int] = None,
                      profile: EnvProfile = CAR,
                      wide=()) -> RoadState:
    """actors: [(name, x, y, speed_mps)] for non-ego vehicles/walkers.
    Actors that don't land on the corridor are ignored (they are not in
    our window; the occlusion machinery covers what we can't see).
    wide: names captured from walker_capture_m instead of capture_m
    (pedestrian-class actors: track the approach, don't discover the
    crossing one tick before the path)."""
    if ego_cell is None:
        ego_cell = profile.ego_cell
    road = Road(lanes=max(corridor.lanes) + 1, length=profile.window,
                v_max=profile.v_max)
    cars = [Car("ego", ego_lane, ego_cell,
                profile.speed_to_cells(ego_speed_mps))]
    taken = {(ego_lane, ego_cell)}
    for name, x, y, v in actors:
        cap = profile.walker_capture_m if name in wide else None
        hit = locate(corridor, x, y, profile, cap)
        if hit is None:
            continue
        lane, cell, _ = hit
        if (lane, cell) in taken:      # two actors in one cell: keep first
            continue
        taken.add((lane, cell))
        cars.append(Car(name, lane, cell, profile.speed_to_cells(v)))
    return RoadState(road, tuple(cars))


MAX_SPEED_MPS = CAR.max_speed_mps   # v0 city cap (import-compat)
CURVE_YAW_DEG = 20.0  # yaw swing along the window that triggers slowdown
CURVE_SPEED_MPS = CAR.curve_speed_mps


def curvature_cap(corridor: CorridorMap, lane: int, from_cell: int,
                  to_cell: int, profile: EnvProfile = CAR) -> float:
    """Speed cap from corridor geometry: if the route bends more than
    CURVE_YAW_DEG within the lookahead, slow to the profile's curve
    speed."""
    yaws = [a.yaw
            for c in range(from_cell, min(to_cell + 3, profile.window))
            for a in [corridor.anchor(lane, c)] if a is not None]
    if len(yaws) < 2:
        return profile.max_speed_mps
    swing = max(abs((y - yaws[0] + 180) % 360 - 180) for y in yaws)
    return profile.curve_speed_mps if swing > CURVE_YAW_DEG \
        else profile.max_speed_mps


@dataclass
class Setpoint:
    anchor: Anchor
    lane: int
    cell: int
    target_speed: float   # m/s bound for the pilot
    action: str


def decision_to_setpoint(state: RoadState, action: str,
                         corridor: CorridorMap,
                         profile: EnvProfile = CAR
                         ) -> Optional[Setpoint]:
    """Materialize the behavior decision as a pilot setpoint."""
    ego = state.car("ego")
    lane = ego.lane + (1 if action == "right" else
                       -1 if action == "left" else 0)
    v = {"accel": min(profile.v_max, ego.speed + 1),
         "brake": max(0, ego.speed - 1)}.get(action, ego.speed)
    cell = ego.cell + max(1, LOOKAHEAD.get(action, 3))
    cap = min(profile.max_speed_mps,
              curvature_cap(corridor, lane, ego.cell, cell, profile))
    # walk back toward ego until an anchor exists
    while cell > ego.cell:
        a = corridor.anchor(lane, cell)
        if a is not None:
            return Setpoint(a, lane, cell,
                            min(profile.cells_to_speed(v), cap),
                            action)
        cell -= 1
    a = corridor.anchor(lane, ego.cell)
    return Setpoint(a, lane, ego.cell,
                    min(profile.cells_to_speed(v), cap),
                    action) if a else None


def state_in_corridor(corridor: CorridorMap,
                      ego_xyv: Tuple[float, float, float],
                      actors: List[Tuple[str, float, float, float]],
                      profile: EnvProfile = CAR,
                      wide=()) -> Optional[RoadState]:
    """Express a *later* world observation in an existing corridor frame
    (for the surprise check: the measure was computed in that frame).
    None when ego has left the corridor window."""
    ex, ey, ev = ego_xyv
    hit = locate(corridor, ex, ey, profile)
    if hit is None:
        return None
    lane, cell, _ = hit
    return snapshot_to_state(actors, corridor, ev, ego_lane=lane,
                             ego_cell=cell, profile=profile,
                             wide=wide)


def straight_corridor(n_lanes=2, lane_w=3.5, cell0_x=0.0) -> CorridorMap:
    """Synthetic straight corridor for tests: lane l at y = l*lane_w."""
    return CorridorMap({
        lane: [Anchor(cell0_x + c * CELL_M, lane * lane_w)
               for c in range(WINDOW)]
        for lane in range(n_lanes)})


CATALOG_CLASSES = ("vehicle", "bulky", "twowheeler", "pedestrian")


def expand_class_behaviors(behaviors: dict, classes: dict) -> dict:
    """Class-keyed CHOICE weights -> the per-name dict make_policy
    expects. behaviors: {"vehicle": w, ..., "default": w} (fit_behaviors
    --by-class output); classes: {actor_name: catalog class or None}
    (detection layer over live type_ids). Names whose class has no
    fitted entry fall through to "default" inside make_policy. A
    name-keyed or default-only behaviors dict passes through unchanged."""
    if not any(k in CATALOG_CLASSES for k in behaviors):
        return behaviors
    out = {name: behaviors[cls] for name, cls in classes.items()
           if cls in behaviors}
    if "default" in behaviors:
        out["default"] = behaviors["default"]
    return out


WALKER_ACTS = ("keep", "accel", "brake", "left", "right")


def classify_step(dv_bucket: int, along_m: float, lat_m: float,
                  walker: bool) -> str:
    """Action label for one observed transition (the harvester's
    labeler, pure for tests). Vehicles: speed-bucket delta, hbrake at
    <= -2. Walkers: a crossing step is a DIRECTION change — moving
    (>0.15 m this tick) at more than ~27 degrees off the previous
    heading (|lat| > 0.5*|along|); scale-free, unlike any absolute
    lateral threshold (a walker's whole step is under a meter, so
    half a car cell could never trigger). Otherwise the bucket delta,
    with no hbrake — a walker's full stop is one brake bucket."""
    if walker:
        moving = (along_m ** 2 + lat_m ** 2) > 0.15 ** 2
        if moving and abs(lat_m) > 0.5 * abs(along_m):
            return "left" if lat_m > 0 else "right"
    if dv_bucket >= 1:
        return "accel"
    if dv_bucket == 0:
        return "keep"
    if dv_bucket == -1 or walker:
        return "brake"
    return "hbrake"


def corridor_from_polyline(points, profile: EnvProfile = CAR,
                           ego_xy=None,
                           shoulders_m: float = 0.0) -> CorridorMap:
    """Corridor for unstructured space (park paths, field tracks —
    wheelchair item f): resample a routed polyline [(x, y), ...] into
    anchors every profile.cell_m, yaw from the segment direction, and
    window the result so ego sits at profile.ego_cell. No lane graph
    required. shoulders_m > 0 adds SIDE STRIPS at +-shoulders_m as
    lanes 0 (left) and 2 (right) around the path (lane 1): a crossing
    pedestrian is tracked on the shoulder cells and the walker thread
    predicts the lateral hop — without them the profile's small
    capture radius leaves the approach invisible until under one tick
    before the path (found live: CPNA graze at 1 m/s). Returns the
    map; ego's lane is 1 with shoulders, else 0."""
    import math as _math
    if len(points) < 2:
        raise ValueError("polyline needs at least two points")
    # arclength resample
    anchors = []
    px, py = points[0]
    carry = 0.0
    heading = 0.0
    for qx, qy in points[1:]:
        seg = _math.hypot(qx - px, qy - py)
        if seg <= 1e-9:
            continue
        heading = _math.degrees(_math.atan2(qy - py, qx - px))
        d = carry
        while d < seg:
            t = d / seg
            anchors.append(Anchor(px + (qx - px) * t,
                                  py + (qy - py) * t, heading))
            d += profile.cell_m
        carry = d - seg
        px, py = qx, qy
    anchors.append(Anchor(px, py, heading))
    # window around ego
    start = 0
    if ego_xy is not None:
        ex, ey = ego_xy
        start = min(range(len(anchors)),
                    key=lambda i: _math.hypot(anchors[i].x - ex,
                                              anchors[i].y - ey))
    start = max(0, start - profile.ego_cell)
    win = anchors[start:start + profile.window]
    win += [None] * (profile.window - len(win))
    if shoulders_m <= 0:
        return CorridorMap({0: win})

    def strip(side):
        out = []
        for a in win:
            if a is None:
                out.append(None)
                continue
            yaw = _math.radians(a.yaw)
            nx, ny = -_math.sin(yaw), _math.cos(yaw)
            out.append(Anchor(a.x + nx * shoulders_m * side,
                              a.y + ny * shoulders_m * side, a.yaw))
        return out

    return CorridorMap({0: strip(-1), 1: win, 2: strip(+1)})
