"""Euro NCAP AEB VRU car-to-pedestrian scenarios, encoded from the
official test protocol v4.5.1 (docs/euro_ncap_aeb_lss_vru_test_
protocol_v451.pdf, section 7.2 + figures 7-1..7-5) as machine-readable
specs. Three consumers:

  1. the CARLA park/pedestrian scenario driver (wheelchair item e):
     spawn geometry + walker trajectory per spec, HABIT motions for
     realistic articulation;
  2. corridor/ring-sim test vectors: a crossing pedestrian is a disc
     actor entering ego's lane at a known TTC — once holonomic threads
     land (item a), the planner must brake per RSS on every spec;
  3. the wheelchair profile: the same geometry scaled by the
     environment profile (item c) — a walking adult crossing a 2 m/s
     wheelchair's path is CPNA at 1:6 speed scale.

Pass criterion (ours, stricter than NCAP's mitigate-above-threshold):
zero contact at every VUT test speed, stop at >= MARGIN before the
conflict point, and no FPBR-style phantom braking on the non-crossing
variants (HABIT's not-crossing labels supply those).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

KMH = 1 / 3.6


@dataclass(frozen=True)
class Obstruction:
    """CPNCO parked-vehicle screen (protocol Annex A + fig 7-3)."""
    dummy_to_front_m: float = 1.0     # I: dummy H-point to obstruction
    lane_gap_m: float = 1.0           # J: VUT lane to larger vehicle


@dataclass(frozen=True)
class PedSpec:
    speed_kmh: float                  # steady target speed
    direction: str                    # nearside | farside |
    #                                   longitudinal-same | standing
    accel_dist_m: float               # G: ramp-up before steady state
    start_to_impact_m: float          # D/E/S: H-point start -> impact
    child: bool = False


@dataclass(frozen=True)
class VutMotion:
    kind: str                         # forward | turn | reverse
    turn_side: Optional[str] = None   # farside | nearside (CPTA)
    ped_from: Optional[str] = None    # same | opposite (CPTA variants)


@dataclass(frozen=True)
class Scenario:
    id: str
    vut_speeds_kmh: Tuple[int, ...]
    vut: VutMotion
    ped: PedSpec
    impact_fraction: float            # of vehicle width, no-brake case
    obstruction: Optional[Obstruction] = None
    notes: str = ""


def _fwd():
    return VutMotion("forward")


SCENARIOS = [
    Scenario("CPFA-50", tuple(range(10, 65, 5)), _fwd(),
             PedSpec(8.0, "farside", 1.5, 6.0), 0.50,
             notes="adult running from farside"),
    Scenario("CPNA-25", tuple(range(10, 65, 5)), _fwd(),
             PedSpec(5.0, "nearside", 1.0, 4.0), 0.25,
             notes="walking adult from nearside, 25% impact"),
    Scenario("CPNA-75", tuple(range(10, 65, 5)), _fwd(),
             PedSpec(5.0, "nearside", 1.0, 4.0), 0.75,
             notes="walking adult from nearside, 75% impact; "
                   "prerequisite variant: VUT 20 km/h, EPTa 3 km/h"),
    Scenario("CPNCO-50", tuple(range(10, 65, 5)), _fwd(),
             PedSpec(5.0, "nearside", 1.0, 4.0, child=True), 0.50,
             obstruction=Obstruction(),
             notes="running child from behind two parked vehicles — "
                   "the occlusion-machinery scenario"),
    Scenario("CPLA-50", tuple(range(20, 65, 5)), _fwd(),
             PedSpec(5.0, "longitudinal-same", 1.0, 10.0), 0.50,
             notes="walking adult ahead, same direction (AEB)"),
    Scenario("CPLA-25", tuple(range(50, 85, 5)), _fwd(),
             PedSpec(5.0, "longitudinal-same", 1.0, 10.0), 0.25,
             notes="FCW/ESS variant, high speed"),
    Scenario("CPTA-50-f", (10, 15, 20),
             VutMotion("turn", turn_side="farside"),
             PedSpec(5.0, "farside", 1.0, 9.5), 0.50,
             notes="VUT turns across crossing adult, farside turn"),
    Scenario("CPTA-50-n", (10,),
             VutMotion("turn", turn_side="nearside"),
             PedSpec(5.0, "nearside", 1.0, 9.5), 0.50,
             notes="nearside turn variant"),
    Scenario("CPRA-Cs", (4, 8), VutMotion("reverse"),
             PedSpec(0.0, "standing", 0.0, 0.0), 0.50,
             notes="reverse toward standing adult/child; impact "
                   "25/50/75 selected by lab"),
    Scenario("CPRA-Cm-50", (4, 8), VutMotion("reverse"),
             PedSpec(5.0, "nearside", 1.0, 4.0), 0.50,
             notes="reverse toward crossing pedestrian"),
]


def by_id(sid: str) -> Scenario:
    return next(s for s in SCENARIOS if s.id == sid)


def crossing_ttc_s(sc: Scenario) -> float:
    """Time from pedestrian steady-state start to the conflict point —
    the window the planner has to react once the pedestrian is
    visible (for CPNCO, visibility starts later: at the obstruction's
    edge, which is the point of the scenario)."""
    if sc.ped.speed_kmh <= 0:
        return float("inf")
    return sc.ped.start_to_impact_m / (sc.ped.speed_kmh * KMH)


def scale(sc: Scenario, factor: float) -> Scenario:
    """Geometric scale-down for the wheelchair profile: distances and
    speeds multiplied by `factor` (e.g. 1/6 maps a 30 km/h urban
    approach onto a 1.4 m/s wheelchair approach), impact fraction and
    topology unchanged."""
    return Scenario(
        f"{sc.id}@{factor:g}",
        tuple(max(1, round(v * factor)) for v in sc.vut_speeds_kmh),
        sc.vut,
        PedSpec(sc.ped.speed_kmh * factor, sc.ped.direction,
                sc.ped.accel_dist_m * factor,
                sc.ped.start_to_impact_m * factor, sc.ped.child),
        sc.impact_fraction, sc.obstruction, sc.notes)
