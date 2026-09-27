"""Game of Life framework bindings — the third domain through the SAME
Evaluator: input decoder, action encoder, the greedy seam."""

from pathlib import Path

from metis.fw import Evaluator, Registry
from metis.kernel import expected_on
from metis.lang import load_llp
from .model import gol_project

CATALOG = Path(__file__).parent / "catalogs" / "gol.llp"


def gol_evaluator(cells, neighbors, noise) -> Evaluator:
    reg = Registry()
    reg.input("grid", lambda alive: set(alive))
    reg.output("action", lambda res, sit: res.action)

    def seam(sit):
        alive = sit["grid"]

        def do(a):
            return gol_project(cells, neighbors, noise, alive,
                               ticks=1, force=a)

        return ([None] + list(cells), do, lambda pr: pr,
                lambda pr: (-expected_on(pr, on="alive"),), None)

    return Evaluator(load_llp(CATALOG), reg, domain=seam)
