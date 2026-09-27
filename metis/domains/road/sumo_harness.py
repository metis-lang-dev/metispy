"""SUMO learning loop — the Stage-6 harness.

Three jobs, all against a generated 1 km 2-lane straight (CAR profile,
0.5 s model tick = SUMO step length):

  observe()      per-vehicle discrete traces: speed BUCKET per tick
                 (profile.speed_to_cells), action classified from
                 consecutive buckets (keep/accel/brake/hbrake), the
                 `dangerous` flag from the leader gap vs the discrete
                 RSS d_min — rows in the exact fit_behaviors shape.
  fit/evaluate   imitation-fit CHOICE rows per PLANTED vType (calm vs
                 aggressive SUMO parameters are the ground truth the
                 fit must recover); held-out evaluation by (a) mean
                 log-likelihood of observed safe-flag actions, (b)
                 CHOICE-level surprise rate (P(observed action) < eps).
                 Rows enter the policy only on the non-dangerous branch
                 (make_policy), so fitting/evaluation condition on it.
  closed_loop()  ego driven by the metis planner: corridor RoadState
                 from traci (PLAN_WINDOW cells, K_ACTORS nearest — the
                 relevance cap; plan() cost is exponential in actors),
                 plan() per tick memoized by discrete state, action
                 applied via setSpeed (speedMode 0: SUMO's own safety
                 layer OFF — the RSS shield is under test); safety =
                 no collision involving ego.
"""

import os
import random
import subprocess
import sys
from typing import Dict, List

from .fit_behaviors import ACTS, fit
from .gridcat import DEFAULT_BEHAVIOR, plan
from .profile import CAR
from .semantics import d_min
from .sim import Car, Road, RoadState

EDGE_M = 1000.0
LANES = 2
EPS_RARE = 0.05

VTYPES = {
    "calm": dict(accel=1.0, decel=2.0, sigma=0.2, maxSpeed=8.0,
                 tau=1.4, color="0,0,1"),
    "aggr": dict(accel=3.0, decel=4.5, sigma=0.8, maxSpeed=15.0,
                 tau=0.9, color="1,0,0"),
}


def _traci():
    try:
        import traci
    except ImportError:
        sys.path.append(os.path.join(
            os.environ.get("SUMO_HOME", "/usr/share/sumo"), "tools"))
        import traci
    return traci


def ensure_net(work: str) -> str:
    net = os.path.join(work, "road.net.xml")
    if os.path.exists(net):
        return net
    nod = os.path.join(work, "road.nod.xml")
    edg = os.path.join(work, "road.edg.xml")
    with open(nod, "w") as f:
        f.write('<nodes><node id="a" x="0" y="0"/>'
                f'<node id="b" x="{EDGE_M}" y="0"/></nodes>')
    with open(edg, "w") as f:
        f.write(f'<edges><edge id="E" from="a" to="b" numLanes="{LANES}"'
                ' speed="15"/></edges>')
    subprocess.run(["netconvert", "-n", nod, "-e", edg, "-o", net],
                   check=True, capture_output=True)
    return net


def write_routes(work: str, seed: int, n: int = 40,
                 ego: bool = False) -> str:
    rng = random.Random(seed)
    path = os.path.join(work, f"routes_{seed}{'_ego' if ego else ''}.xml")
    lines = ["<routes>"]
    for name, p in VTYPES.items():
        lines.append(
            f'<vType id="{name}" accel="{p["accel"]}" '
            f'decel="{p["decel"]}" sigma="{p["sigma"]}" '
            f'maxSpeed="{p["maxSpeed"]}" tau="{p["tau"]}" '
            f'color="{p["color"]}"/>')
    lines.append('<route id="r" edges="E"/>')
    # SUMO requires vehicles SORTED BY DEPART TIME — an out-of-order
    # entry silently wrecks the schedule (ego listed first at t=8
    # emptied the first 8 s of traffic; found by the empty-corridor
    # probe). Build (depart, xml) pairs and sort.
    vehicles = []
    if ego:
        # ego starts BEHIND the traffic and faster than the calm cap:
        # it must catch up and negotiate (brake / lane change) — a
        # front-runner ego would make the safety invariant vacuous
        vehicles.append((8.0, '<vehicle id="ego" type="aggr" route="r" '
                              'depart="8.0" departLane="0" '
                              'departPos="0" departSpeed="6"/>'))
    for i in range(n):
        vt = "calm" if rng.random() < 0.5 else "aggr"
        vehicles.append((i * 0.9,
                         f'<vehicle id="{vt}_{i}" type="{vt}" route="r" '
                         f'depart="{i * 0.9:.1f}" departLane="random" '
                         f'departSpeed="random"/>'))
    lines += [xml for _, xml in sorted(vehicles)]
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _start(traci, net, routes, seed):
    traci.start(["sumo", "-n", net, "-r", routes,
                 "--step-length", str(CAR.tick_s), "--seed", str(seed),
                 "--no-warnings", "--no-step-log",
                 "--collision.action", "warn",
                 "--default.speeddev", "0.2"])


def _bucket(speed_mps: float) -> int:
    return CAR.speed_to_cells(speed_mps)


def _dangerous(traci, vid: str, bucket: int) -> bool:
    lead = traci.vehicle.getLeader(vid)
    if lead is None:
        return False
    lead_id, dist = lead
    gap_cells = max(0, int(dist / CAR.cell_m))
    lead_b = _bucket(traci.vehicle.getSpeed(lead_id))
    return gap_cells <= d_min(bucket, lead_b, CAR.v_max)


def classify(prev_bucket: int, bucket: int) -> str:
    dv = bucket - prev_bucket
    if dv > 0:
        return "accel"
    if dv == 0:
        return "keep"
    if dv == -1:
        return "brake"
    return "hbrake"


def observe(work: str, seed: int, steps: int = 240,
            n: int = 20) -> List[dict]:
    """One traffic episode -> fit_behaviors-shaped rows. Observation
    runs FREE-FLOW density (n=20): congestion pushes everyone into the
    dangerous state where the safe-branch rows never fire, starving the
    fit (found when the dense closed-loop setting thinned the aggr
    signal to 1%)."""
    traci = _traci()
    net = ensure_net(work)
    routes = write_routes(work, seed, n=n)
    _start(traci, net, routes, seed)
    prev: Dict[str, int] = {}
    rows: List[dict] = []
    try:
        for _ in range(steps):
            traci.simulationStep()
            for vid in traci.vehicle.getIDList():
                b = _bucket(traci.vehicle.getSpeed(vid))
                if vid in prev:
                    rows.append({
                        "type": traci.vehicle.getTypeID(vid),
                        "action": classify(prev[vid], b),
                        "dangerous": _dangerous(traci, vid, b)})
                prev[vid] = b
    finally:
        traci.close()
    return rows


def fit_by_type(rows: List[dict], alpha: float = 1.0):
    """One safe-branch CHOICE row per planted vType (rows enter the
    policy on the non-dangerous branch only)."""
    out = {}
    for vt in VTYPES:
        sub = [r for r in rows if r["type"] == vt]
        out[vt] = fit(sub, alpha)["default"]
    return out


def fit_branches(rows: List[dict], alpha: float = 1.0):
    """Both flag-conditional rows per vType — the model of a driver
    style, for mismatch detection (not for the policy)."""
    out = {}
    for vt in VTYPES:
        sub = [r for r in rows if r["type"] == vt]
        f = fit(sub, alpha)
        out[vt] = {"default": f["default"], "dangerous": f["dangerous"]}
    return out


def branch_ll(rows: List[dict], model: Dict[str, dict]) -> float:
    """Mean log-likelihood of ALL rows under a two-branch model —
    the threshold-free mismatch detector (LL gap own-vs-foreign is a
    likelihood-ratio test on the whole behavior, immune to the
    bucket-boundary churn artifact that broke rate-based flags)."""
    import math
    ll, n = 0.0, 0
    for r in rows:
        if r["action"] not in ACTS:
            continue
        row = model["dangerous" if r["dangerous"] else "default"]
        z = sum(row.get(a, 0.0) for a in ACTS)
        p = max(row.get(r["action"], 0.0) / z, 1e-12)
        ll += math.log(p)
        n += 1
    return ll / max(1, n)


def evaluate(rows: List[dict], rows_by_type: Dict[str, dict]):
    """(mean log-likelihood, surprise rate) of observed SAFE-branch
    actions under per-type rows (missing types fall back to the
    planner default row)."""
    import math
    ll, n, rare = 0.0, 0, 0
    for r in rows:
        if r["dangerous"] or r["action"] not in ACTS:
            continue
        row = rows_by_type.get(r["type"], DEFAULT_BEHAVIOR)
        z = sum(row.get(a, 0.0) for a in ACTS)
        p = max(row.get(r["action"], 0.0) / z, 1e-12)
        ll += math.log(p)
        rare += p < EPS_RARE
        n += 1
    return ll / max(1, n), rare / max(1, n), n


# closed-loop planning window: SMALLER than the corpus window and
# capped to the nearest actors — plan() cost is exponential in actor
# count (a 4-car 24-cell block costs ~10 s in pure python), and the
# relevance cap is what the production corridor does anyway.
PLAN_WINDOW = 12
PLAN_EGO_CELL = 3
K_ACTORS = 3


def _corridor_state(traci, ego_id: str) -> RoadState:
    road = Road(LANES, PLAN_WINDOW, CAR.v_max)
    epos = traci.vehicle.getLanePosition(ego_id)
    cars = [Car("ego", traci.vehicle.getLaneIndex(ego_id), PLAN_EGO_CELL,
                _bucket(traci.vehicle.getSpeed(ego_id)))]
    others = []
    for vid in traci.vehicle.getIDList():
        if vid == ego_id or traci.vehicle.getRoadID(vid) != "E":
            continue
        cell = PLAN_EGO_CELL + round(
            (traci.vehicle.getLanePosition(vid) - epos) / CAR.cell_m)
        if 0 <= cell < PLAN_WINDOW:
            others.append(Car(vid, traci.vehicle.getLaneIndex(vid), cell,
                              _bucket(traci.vehicle.getSpeed(vid))))
    others.sort(key=lambda c: abs(c.cell - PLAN_EGO_CELL))
    return RoadState(road, tuple(cars + others[:K_ACTORS]))


def closed_loop(work: str, seed: int, steps: int = 200,
                behaviors=None, evaluator=None,
                adapt_models=None) -> dict:
    """adapt_models: {style: {"default": row, "dangerous": row}} from
    fit_branches — enables ONLINE per-vehicle posteriors (stage-5
    machinery live): each nearby vehicle's observed bucket transitions
    update P(style | history) via the branch-conditional categorical
    likelihood; the planner receives per-vehicle MIXTURE rows. The
    plan memo is bypassed while adapting (beliefs change the measure
    every tick)."""
    """Ego under the metis planner; SUMO's own safety OFF for ego.
    Returns collision/tick/action accounting."""
    traci = _traci()
    net = ensure_net(work)
    routes = write_routes(work, seed, ego=True)
    _start(traci, net, routes, seed)
    stats = {"ticks": 0, "ego_collisions": 0, "actions": {}, "doomed": 0,
             "plan_cache_hits": 0}
    plan_cache = {}
    seen_ego = False
    posteriors: Dict[str, Dict[str, float]] = {}
    prev_bucket: Dict[str, int] = {}
    n_obs: Dict[str, int] = {}
    try:
        for _ in range(steps):
            traci.simulationStep()
            for c in traci.simulation.getCollisions():
                if "ego" in (c.collider, c.victim):
                    stats["ego_collisions"] += 1
            if "ego" not in traci.vehicle.getIDList():
                if seen_ego:
                    break          # ego finished the edge
                continue           # not inserted yet
            seen_ego = True
            if adapt_models is not None:
                # stage-5 live: update per-vehicle style posteriors
                # from this tick's observed transitions
                for vid in traci.vehicle.getIDList():
                    if vid == "ego":
                        continue
                    b = _bucket(traci.vehicle.getSpeed(vid))
                    if vid in prev_bucket:
                        act = classify(prev_bucket[vid], b)
                        branch = "dangerous" if _dangerous(
                            traci, vid, b) else "default"
                        post = posteriors.setdefault(
                            vid, {h: 1.0 / len(adapt_models)
                                  for h in adapt_models})
                        z = 0.0
                        for h, m in adapt_models.items():
                            row = m[branch]
                            tot = sum(row.get(a, 0.0) for a in ACTS)
                            lik = max(row.get(act, 0.0) / tot, 1e-6)
                            post[h] *= lik
                            z += post[h]
                        for h in post:
                            post[h] /= z
                        n_obs[vid] = n_obs.get(vid, 0) + 1
                    prev_bucket[vid] = b
                behaviors = {
                    vid: {a: sum(post[h]
                                 * adapt_models[h]["default"].get(a, 0.0)
                                 for h in post)
                          for a in ACTS}
                    for vid, post in posteriors.items()}
            traci.vehicle.setSpeedMode("ego", 0)   # metis owns safety
            traci.vehicle.setLaneChangeMode("ego", 0)
            state = _corridor_state(traci, "ego")
            # steady-state configs repeat heavily; the plan is a pure
            # function of the discrete state — memoize
            key = tuple((c.name != "ego" and "o" or "ego", c.lane,
                         c.cell, c.speed) for c in state.cars)
            hit = plan_cache.get(key) if adapt_models is None else None
            if hit is not None:
                act, doomed = hit
                stats["plan_cache_hits"] += 1
            else:
                info = plan(state, ticks=2, objective="dof",
                            behaviors=behaviors, ego_vcap=CAR.v_max,
                            dmin_pad=CAR.dmin_pad_cells)
                act, doomed = info["action"], info["doomed"]
                if adapt_models is None:
                    plan_cache[key] = (act, doomed)
            if evaluator is not None:
                # the stage-10 gate: the framework path (corridor port
                # in -> setpoint port out) must agree with the direct
                # path tick-for-tick
                from metis.fw import Blackboard
                bb = Blackboard()
                bb["corridor"] = state
                fres = evaluator.tick(bb)
                assert (fres.action, fres.doomed) == (act, doomed), (
                    "framework/direct divergence", fres.action, act)
                assert bb["setpoint"]["action"] == act
                stats["fw_ticks"] = stats.get("fw_ticks", 0) + 1
            stats["actions"][act] = stats["actions"].get(act, 0) + 1
            stats["doomed"] += doomed
            from .sim import kinematics
            ego2 = kinematics(state.road, state.car("ego"), act)
            v2 = ego2.speed if ego2 else 0
            traci.vehicle.setSpeed("ego", CAR.cells_to_speed(v2))
            if ego2 and ego2.lane != state.car("ego").lane \
                    and 0 <= ego2.lane < LANES:
                traci.vehicle.changeLane("ego", ego2.lane, CAR.tick_s)
            stats["ticks"] += 1
    finally:
        traci.close()
    if adapt_models is not None:
        stats["posteriors"] = {
            vid: (max(p, key=p.get), round(max(p.values()), 3),
                  n_obs.get(vid, 0))
            for vid, p in posteriors.items()}
    return stats
