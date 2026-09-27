"""CrossingTraffic framework bindings — fifth domain, same Evaluator;
the seam exposes the evaluator choice (exact / sampled) so one catalog
serves both sides of the cliff."""

from typing import Optional, Tuple

from metis.fw import Evaluator, Registry
from .model import DIRS, XTrafficModel


def xtraffic_evaluator(model: XTrafficModel,
                       ticks: int = 12,
                       sampled: Optional[Tuple[int, int]] = None
                       ) -> Evaluator:
    reg = Registry()
    reg.input("scene", lambda v: v)          # (pos, obstacles)
    reg.output("move", lambda res, sit: res.action)
    for name, fn in model.bindings().items():
        reg.weight(name, fn)

    def seam(sit):
        pos, obstacles = sit["scene"]

        def do(d):
            return model.project(pos, obstacles, ticks, cmd=d,
                                 sampled=sampled)

        return (list(DIRS), do, lambda pr: pr,
                lambda pr: (model.expected_cost(pr),), None)

    return Evaluator(model.doc, reg, domain=seam)
