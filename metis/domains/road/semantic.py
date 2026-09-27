"""The semantic compiler — road geometry is the source program, road
semantics is the IR, linear logic is the execution semantics over it
(docs/road-encoding.md). Geometry is NEVER translated into logic:
the compiler extracts the semantic graph (adjacency Horn facts) and
binds continuous quantities behind weight ports; every logical
object keeps a HANDLE back to the exact geometric entity, and the
motion planner receives (LaneHandle, s-interval, v-cap) — geometry
looked up lazily, no conversion back from semantics.

v1 scope, honest: ego + one lead vehicle (alfa); the longitudinal
calculus is the two-class RSS gap {gclose, gsafe} (the measure lives
on classes; exact s,v ride as metadata); one-tick maneuver planning
(the myopia is the planner's horizon choice, not the encoding's)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metis.kernel import ForwardProjection, plan
from metis.lang import compile_catalog, parse, resolve_includes, \
    weight_policy

DOMAIN = Path(__file__).parent / "catalogs" / "handles_domain.llp"

# RSS-style parameters (response time, comfortable/max braking)
RHO, B_MAX, B_MIN = 1.0, 6.0, 4.0


@dataclass
class Lane:
    """A geometric entity. The logic only ever carries `id`; the
    numbers (and, in production, the centerline spline) stay here."""
    id: str
    length: float
    speed_limit: float
    left: Optional[str] = None
    right: Optional[str] = None
    centerline: object = None          # clothoid/spline — opaque


@dataclass
class Scene:
    """Perception after the one-time Frenet projection: everything
    downstream is (LaneHandle, s, v)."""
    vehicles: Dict[str, Tuple[str, float, float]]  # id -> (lane, s, v)


def rss_gap(v_rear: float, v_front: float) -> float:
    """Minimal safe longitudinal gap (RSS shape)."""
    d = v_rear * RHO + v_rear * v_rear / (2 * B_MIN) \
        - v_front * v_front / (2 * B_MAX)
    return max(d, 0.0)


class SemanticCompiler:
    def __init__(self, lanes: List[Lane]):
        self.lanes = {ln.id: ln for ln in lanes}

    def facts(self) -> List[str]:
        out = [f"islane({i})." for i in self.lanes]
        for ln in self.lanes.values():
            if ln.left:
                out.append(f"leftof({ln.left},{ln.id}).")
            if ln.right:
                out.append(f"rightof({ln.right},{ln.id}).")
        return out

    def instance_llp(self) -> str:
        return "\n".join(
            ["% GENERATED instance — semantic graph from geometry.",
             "catalog road-scene 1.",
             'provenance hand "semantic compiler output" '
             '"2026-07-20".',
             f'include "{DOMAIN}".',
             f"type vehicle {{ego alfa}}.",
             f"type lane {{{' '.join(self.lanes)}}}.",
             "type maneuver {keep left right}."]
            + self.facts()) + "\n"

    # -- the abstraction boundary (continuous -> classes) --------------

    def gap_class(self, scene: Scene) -> str:
        (le, se, ve) = scene.vehicles["ego"]
        (la, sa, va) = scene.vehicles["alfa"]
        if le != la or sa <= se:
            return "gsafe"
        return "gclose" if sa - se < rss_gap(ve, va) else "gsafe"

    def init(self, scene: Scene, cmd: str) -> Dict[str, int]:
        out = {"emvt": 1, f"cmd({cmd})": 1, self.gap_class(scene): 1}
        for v, (ln, _s, _v) in scene.vehicles.items():
            out[f"on({v},{ln})"] = 1
        return out

    # -- geometry reaches the measure only through ports ---------------

    def bindings(self, scene: Scene):
        (le, se, ve) = scene.vehicles["ego"]
        (la, sa, va) = scene.vehicles["alfa"]
        limit = self.lanes[la].speed_limit

        def brk(args, counts):
            # a lead over the limit sheds speed more eagerly
            return min(0.9, 0.2 + 0.5 * max(0.0, va / limit - 1.0))

        def cls(args, counts):
            # closing probability from relative speed (continuous!)
            return min(0.9, max(0.05, (ve - va) / max(ve, 1.0)))

        def rss(args, counts):
            if "gclose" not in counts:
                return 0.0
            same = any(f"on(ego,{i})" in counts
                       and f"on(alfa,{i})" in counts
                       for i in self.lanes)
            if not same:
                return 0.0
            margin = (sa - se) / max(rss_gap(ve, va), 1e-6)
            return min(0.95, max(0.05, 1.0 - margin))

        return {"brk": brk, "cls": cls, "rss": rss}


class RoadModel:
    def __init__(self, lanes: List[Lane]):
        self.compiler = SemanticCompiler(lanes)
        self.doc = compile_catalog(resolve_includes(
            parse(self.compiler.instance_llp()), DOMAIN.parent))

    def candidates(self, scene: Scene) -> List[str]:
        """The legal maneuver set — the hard shield runs BEFORE the
        measure (adjacency is legality; RSS prices the rest)."""
        lane = self.compiler.lanes[scene.vehicles["ego"][0]]
        out = ["keep"]
        if lane.left:
            out.append("left")
        if lane.right:
            out.append("right")
        return out

    def project(self, scene: Scene, cmd: str,
                ticks: int = 1) -> ForwardProjection:
        prog = self.doc.catalog.ground(
            self.compiler.init(scene, cmd))
        pol = weight_policy(prog, self.doc.ports,
                            self.compiler.bindings(scene))
        return ForwardProjection(prog, ticks * 6, policy=pol)

    def p_crash(self, pr: ForwardProjection) -> float:
        return sum(p for (_, k), p in pr.final.items()
                   if ("crash", 1) in k)

    def plan(self, scene: Scene):
        return plan(self.candidates(scene),
                    do=lambda m: m,
                    project=lambda m: self.project(scene, m),
                    score=lambda pr: (self.p_crash(pr),),
                    doomed_action="keep")

    def handoff(self, scene: Scene, maneuver: str,
                lookahead: float = 60.0) -> dict:
        """The motion-planner packet: handles + geometry looked up
        lazily — no conversion back from semantics."""
        (le, se, _ve) = scene.vehicles["ego"]
        lane = self.compiler.lanes[le]
        if maneuver == "left" and lane.left:
            lane = self.compiler.lanes[lane.left]
        elif maneuver == "right" and lane.right:
            lane = self.compiler.lanes[lane.right]
        return {"lane": lane.id,
                "s_from": se,
                "s_to": min(se + lookahead, lane.length),
                "v_cap": lane.speed_limit,
                "geometry": lane.centerline}
