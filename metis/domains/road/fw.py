"""Road domain framework bindings — the whole domain-side cost of
joining the framework: an input decoder, a setpoint encoder, and the
planner seam. The seam now projects through the LIFTED CATALOG
(gridcat: grid_domain.llp + road_bindings) — the recoded production
path, not the hand-built oracle. Everything else is the catalog + the
engine."""

from pathlib import Path

from metis.fw import Evaluator, Registry
from metis.lang import load_llp
from . import gridcat

# the grid instance is generated per-tick from the scene; this static
# catalog is only the framework's load anchor (its ports/manifest).
CATALOG = Path(__file__).parent / "catalogs" / "mini.llp"


def road_registry() -> Registry:
    reg = Registry()
    # the corridor input arrives as an already-decoded RoadState (the
    # SUMO/percept side owns sensor decoding); identity binding
    reg.input("corridor", lambda v: v)
    reg.output("setpoint", lambda res, sit: {
        "action": res.action, "doomed": res.doomed, "dof": res.dof,
        "scores": res.scores})
    return reg


def road_evaluator(**plan_kw) -> Evaluator:
    reg = road_registry()
    return Evaluator(
        load_llp(CATALOG), reg,
        domain=lambda sit: gridcat.plan_seam(
            sit["corridor"], **plan_kw)[0])
