"""A minimal ring-road simulator: lanes x cells, integer speeds and ticks.

The road is a ring (length >> any safe distance) so episodes run
indefinitely and cars keep interacting. Per tick every car picks an action;
speeds update first, then every car advances by its *new* speed (worst-case
ordering consistent with RSS's response-time reasoning: an acceleration
command takes effect within the same tick).

Collision detection is exact sweep-based (same target lane, rear's travel
closes the gap to <= 0, i.e. same final cell or a pass-through), which is
stricter than the LL horizon model's same-cell rule — the simulator is the
ground truth, the LL program the planning abstraction."""

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

ACTIONS = ("keep", "accel", "brake", "left", "right")


@dataclass(frozen=True)
class Car:
    name: str
    lane: int
    cell: int
    speed: int
    wrecked: bool = False


@dataclass(frozen=True)
class Road:
    lanes: int = 2
    length: int = 24
    v_max: int = 3


@dataclass(frozen=True)
class RoadState:
    road: Road
    cars: Tuple[Car, ...]

    def car(self, name) -> Car:
        return next(c for c in self.cars if c.name == name)

    def others(self, name):
        return [c for c in self.cars if c.name != name]


def kinematics(road: Road, car: Car, action: str) -> Optional[Car]:
    """The car after one tick of `action`; None if kinematically invalid
    (lane off road, braking at standstill is 'keep', lane change needs
    motion)."""
    lane, v = car.lane, car.speed
    if action == "accel":
        v = min(road.v_max, v + 1)
    elif action == "brake":
        v = max(0, v - 1)
    elif action == "hbrake":
        v = max(0, v - 2)      # others' hard brake (= A_MAX_BRAKE)
    elif action == "left":
        lane -= 1
    elif action == "right":
        lane += 1
    if not (0 <= lane < road.lanes):
        return None
    if action in ("left", "right") and car.speed == 0:
        return None
    return replace(car, lane=lane, speed=v,
                   cell=(car.cell + v) % road.length)


def gap_ahead(road: Road, rear_cell: int, front_cell: int) -> int:
    """Cells strictly between rear and front along the ring."""
    return (front_cell - rear_cell - 1) % road.length


def sweep_collisions(road: Road, before: Dict[str, Car],
                     after: Dict[str, Car]) -> List[Tuple[str, str]]:
    """Pairs that end in the same lane with the rear having closed the gap
    to <= 0 during the tick (same cell or pass-through)."""
    crashes = []
    names = list(after)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ca, cb = after[a], after[b]
            if ca.lane != cb.lane:
                continue
            # identify rear at start of tick, on the ring
            d_ab = (before[b].cell - before[a].cell) % road.length
            if d_ab == 0:
                crashes.append((a, b))
                continue
            if d_ab <= road.length // 2:
                rear, front = a, b
                d = d_ab
            else:
                rear, front = b, a
                d = (before[a].cell - before[b].cell) % road.length
            final_gap = d + after[front].speed - after[rear].speed
            if final_gap <= 0:
                crashes.append((rear, front))
    return crashes


def step(state: RoadState, actions: Dict[str, str],
         ) -> Tuple[RoadState, List[Tuple[str, str]]]:
    before = {c.name: c for c in state.cars}
    after = {}
    for c in state.cars:
        if c.wrecked:
            after[c.name] = c
            continue
        nxt = kinematics(state.road, c, actions.get(c.name, "keep"))
        after[c.name] = nxt if nxt is not None else \
            kinematics(state.road, c, "keep")
    crashes = sweep_collisions(
        state.road,
        {n: c for n, c in before.items() if not c.wrecked},
        {n: c for n, c in after.items() if not c.wrecked})
    wrecked = {n for pair in crashes for n in pair}
    cars = tuple(replace(c, speed=0, wrecked=True) if c.name in wrecked
                 else c for c in after.values())
    return RoadState(state.road, cars), crashes


def render(state: RoadState) -> str:
    """ASCII road: one row per lane; cars shown by initial + speed."""
    rows = []
    for lane in range(state.road.lanes):
        row = ["."] * state.road.length
        for c in state.cars:
            if c.lane == lane:
                mark = "#" if c.wrecked else \
                    (c.name[0].upper() if c.name == "ego"
                     else c.name[0]) + str(c.speed)
                row[c.cell] = mark if c.wrecked else mark[0]
                # two-char label: place speed in next cell if free
                if not c.wrecked:
                    row[c.cell] = mark[0]
        rows.append("|" + "".join(f"{x:>2}" for x in row) + "|")
    legend = "  ".join(f"{c.name}@L{c.lane}C{c.cell} v{c.speed}"
                       + ("[WRECK]" if c.wrecked else "")
                       for c in state.cars)
    return "\n".join(rows) + "\n " + legend


def sample_other_action(rng: random.Random, weights: Dict[str, float]) -> str:
    acts, ws = zip(*weights.items())
    return rng.choices(acts, ws)[0]
