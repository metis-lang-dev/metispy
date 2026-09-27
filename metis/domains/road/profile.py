"""Environment scale profiles — wheelchair inventory item (c).

The discrete kernel (rules, d_min series, forward filter) is
scale-free: it speaks cells and ticks. Everything with SI units is the
boundary, and it used to be hardcoded car-scale constants scattered
over bridge_core/continuous/carla_run. A profile gathers that boundary
in one object the pack/environment selects:

    bucket   = cell_m / tick_s          (m/s per speed bucket)
    a_*      = buckets/tick^2 -> m/s^2  (continuous cross-check)

CAR reproduces the historical constants exactly (goldens/behavior
unchanged). WHEELCHAIR: 0.5 m cells at the same 0.5 s tick give a
1 m/s bucket — v_max 2 tops out at 2 m/s (a 6 km/h powered chair is
1.67), braking one bucket/tick = 2 m/s^2 comfortable and hbrake
4 m/s^2 emergency, d_min(2,0) = 6 cells = 3 m, the 24-cell window is
12 m of park path, and the pack's MU_PERSONAL of 1 cell means 0.5 m of
personal space. The discretization contract scales with it: the pilot
must settle inside a 1 m/s bucket, margin 0.15 m/s.
"""

from dataclasses import dataclass

from .semantics import A_MAX_ACCEL, A_MAX_BRAKE, A_MIN_BRAKE, RHO


@dataclass(frozen=True)
class EnvProfile:
    name: str
    cell_m: float
    tick_s: float
    v_max: int                 # speed buckets
    max_speed_mps: float       # policy cap (feeds ego_vcap)
    curve_speed_mps: float     # slowdown in bends
    bucket_margin_mps: float   # pilot bucket-contract margin
    capture_m: float           # actor->anchor lateral capture radius
    window: int = 24           # corridor cells
    ego_cell: int = 3
    # pedestrian-class actors map onto the corridor from farther out:
    # a crossing walker must be tracked on approach, not discovered a
    # tick before the path (the walker thread's lateral-hop model
    # makes the wide mapping conservative)
    walker_capture_m: float = 3.0
    # discretization pad: extra guard cells over d_min so the bucketed
    # verdict is conservative against the exact continuous formula —
    # 2 cells is lattice-proven clean for both profiles within the
    # operating envelope (test_profile); the kernel default stays 0,
    # live planners apply this
    dmin_pad_cells: int = 2

    @property
    def bucket_mps(self) -> float:
        return self.cell_m / self.tick_s

    def speed_to_cells(self, speed_mps: float) -> int:
        return max(0, min(self.v_max,
                          round(speed_mps * self.tick_s / self.cell_m)))

    def cells_to_speed(self, v_cells: int) -> float:
        return v_cells * self.cell_m / self.tick_s

    # SI forms of the discrete RSS parameters (continuous cross-check)
    @property
    def acc_si(self) -> float:
        return A_MAX_ACCEL * self.cell_m / self.tick_s ** 2

    @property
    def bmin_si(self) -> float:
        return A_MIN_BRAKE * self.cell_m / self.tick_s ** 2

    @property
    def bmax_si(self) -> float:
        return A_MAX_BRAKE * self.cell_m / self.tick_s ** 2

    @property
    def rho_si(self) -> float:
        return RHO * self.tick_s


# class top speeds (SI). Adult pedestrians are DIRECTED agents (walker
# threads — motion vector, measured bucket); only dogs/kids get the
# holonomic manifest disc, radius derived from these per profile
# (DESIGN actor models).
CLASS_SPEED_MPS = {"pedestrian": 2.0, "dog": 4.0, "kid": 3.0}


def disc_radius(profile: "EnvProfile", cls: str) -> int:
    """Manifest disc radius in cells: how far the class can displace
    in one tick, ceil'd (conservative — the disc must contain every
    reachable cell)."""
    import math
    v = CLASS_SPEED_MPS.get(cls, 2.0)
    return max(1, math.ceil(v * profile.tick_s / profile.cell_m))


CAR = EnvProfile("car", cell_m=2.5, tick_s=0.5, v_max=3,
                 max_speed_mps=6.0, curve_speed_mps=3.0,
                 bucket_margin_mps=0.4, capture_m=1.8)

WHEELCHAIR = EnvProfile("wheelchair", cell_m=0.5, tick_s=0.5, v_max=2,
                        max_speed_mps=1.8, curve_speed_mps=1.0,
                        bucket_margin_mps=0.15, capture_m=0.6)

PROFILES = {p.name: p for p in (CAR, WHEELCHAIR)}
