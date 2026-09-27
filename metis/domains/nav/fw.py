"""Navigation framework bindings — fourth domain through the same
Evaluator, and the first whose catalog is fully GENERATED text."""

from metis.fw import Evaluator
from metis.lang import compile_catalog, parse
from .model import DIRS, NavModel, Topology, gen_catalog, nav_registry


def nav_evaluator(topo: Topology) -> Evaluator:
    model = NavModel(topo)
    reg = nav_registry(topo)

    def seam(sit):
        pos = sit["pos"]
        rest = topo.horizon - 1

        def score_of(d):
            c2 = topo.adj.get((pos, d), pos)
            p = topo.p_die[c2] if c2 != pos else 0.0
            return (1.0 + p * rest
                    + (1.0 - p) * model.expected_cost(c2, rest),)

        return (list(DIRS), lambda d: d, lambda d: d, score_of, None)

    return Evaluator(compile_catalog(parse(gen_catalog(topo))), reg,
                     domain=seam)
