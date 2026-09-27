"""Environment scale profiles (wheelchair item c): CAR reproduces the
historical constants exactly; WHEELCHAIR rescales the SI boundary while
the discrete kernel stays scale-free."""

from metis.domains.road import bridge_core
from metis.domains.road.bridge_core import (Anchor, CorridorMap, decision_to_setpoint,
                             snapshot_to_state)
from metis.domains.road.continuous import d_min_continuous
from metis.domains.road.gridcat import plan
from metis.domains.road.profile import CAR, PROFILES, WHEELCHAIR
from metis.domains.road.scenarios.euroncap_vru import by_id, crossing_ttc_s, scale
from metis.domains.road.semantics import d_min


def test_car_profile_is_the_historical_constants():
    assert (CAR.cell_m, CAR.tick_s, CAR.v_max) == (2.5, 0.5, 3)
    assert bridge_core.CELL_M == CAR.cell_m
    assert bridge_core.MAX_SPEED_MPS == CAR.max_speed_mps
    assert CAR.speed_to_cells(11.0) == 2      # legacy behavior
    assert CAR.cells_to_speed(3) == 15.0
    # continuous SI mapping unchanged
    assert d_min_continuous(10.4, 6.0) == d_min_continuous(
        10.4, 6.0, CAR)


def test_wheelchair_scale():
    w = WHEELCHAIR
    assert w.bucket_mps == 1.0                # 0.5 m / 0.5 s
    assert w.cells_to_speed(w.v_max) == 2.0   # 2 m/s top speed
    assert w.speed_to_cells(1.4) == 1         # strolling pace
    # discrete d_min(2,0) at v_max 2: accel is capped, so 3 cells=1.5 m
    assert d_min(2, 0, w.v_max) * w.cell_m == 1.5
    # braking one bucket per tick = 2 m/s^2; emergency 4
    assert w.bmin_si == 2.0 and w.bmax_si == 4.0
    # continuous (v_rho capped at the 2 m/s platform top): 2.25 m —
    # ABOVE the discretized 1.5 m: at small scales the round()-bucket
    # anti-conservatism (DESIGN open question) is material, and the
    # cross-check exists to flag exactly this band
    assert d_min_continuous(2.0, 0.0, w) == 2.25
    assert w.window * w.cell_m == 12.0


def straight_corridor(profile, lanes=1):
    return CorridorMap({
        lane: [Anchor(c * profile.cell_m, lane * 1.2)
               for c in range(profile.window)]
        for lane in range(lanes)})


def test_wheelchair_end_to_end_plan_and_setpoint():
    # a strolling pedestrian-speed actor 3 m ahead of a 2 m/s chair:
    # snapshot at wheelchair scale, plan on the scale-free kernel,
    # setpoint back in SI
    w = WHEELCHAIR
    corr = straight_corridor(w)
    actors = [("ped", 6 * w.cell_m + w.ego_cell * w.cell_m, 0.0, 0.0)]
    st = snapshot_to_state(actors, corr, ego_speed_mps=2.0, profile=w)
    assert st.road.v_max == w.v_max
    ego = st.car("ego")
    ped = st.car("ped")
    assert (ped.cell - ego.cell) == 6
    assert ego.speed == 2
    info = plan(st, ticks=2, objective="dof+progress")
    # gap 5 < d_min(2,0)=6: proper response, brake only
    assert info["action"] == "brake"
    sp = decision_to_setpoint(st, info["action"], corr, w)
    assert sp is not None
    assert sp.target_speed <= w.cells_to_speed(1)   # slowed, in SI


def test_euroncap_scaled_fits_wheelchair_profile():
    # CPNA at 1:6 lands in the wheelchair envelope: approach speeds
    # within the profile's buckets, and the pedestrian crossing window
    # exceeds the chair's worst-case stopping time at top speed
    w = WHEELCHAIR
    sc = scale(by_id("CPNA-25"), 1 / 6)
    assert max(sc.vut_speeds_kmh) / 3.6 <= w.cells_to_speed(w.v_max) + 1
    stop_time_s = w.rho_si + w.cells_to_speed(w.v_max) / w.bmin_si
    assert crossing_ttc_s(sc) > stop_time_s


def test_profiles_registry():
    assert set(PROFILES) == {"car", "wheelchair"}


def test_dmin_pad_is_lattice_clean():
    # the discretization-pad proof: with the profile pad, the bucketed
    # verdict is conservative against the exact continuous formula
    # over the whole (v_r, v_f, gap) operating lattice — cont-only
    # violations are impossible, closing the DESIGN open question on
    # bucket rounding direction. Sweep coarsened for test speed; the
    # fine sweep (n=60, step cell/5) was run to choose pad=2.
    for prof in (CAR, WHEELCHAIR):
        pad = prof.dmin_pad_cells
        top = prof.cells_to_speed(prof.v_max)
        n = 24
        for i in range(n + 1):
            v_r = prof.max_speed_mps * i / n
            for j in range(n + 1):
                v_f = top * j / n
                dc = d_min_continuous(v_r, v_f, prof)
                gap = 0.0
                while gap <= dc + 2 * prof.cell_m:
                    if gap < dc:   # continuous says dangerous
                        gap_b = int(gap // prof.cell_m)
                        disc = gap_b < d_min(
                            prof.speed_to_cells(v_r),
                            prof.speed_to_cells(v_f),
                            prof.v_max) + pad
                        assert disc, (prof.name, v_r, v_f, gap)
                    gap += prof.cell_m / 3
