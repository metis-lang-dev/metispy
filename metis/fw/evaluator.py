"""The Evaluator: serve a catalog, bind its ports, tick.

    ev = Evaluator(load_llp("road.llp"), registry, domain=seam_factory)
    ev.tick(blackboard)     # inputs -> kernel.plan -> outputs

The DOMAIN SEAM is a factory `(situation) -> (candidates, do, project,
score, doomed_action)` — exactly kernel.plan's contract; the catalog
stays declarative, the seam is the registered implementation (the
BT.cpp node registration). For catalog-ground domains, catalog_domain()
builds the seam generically from the serving catalog + weight ports.

Hot loading: load_pack() compiles against the serving doc, ADMITS
(containment), revalidates ports, and only then swaps. Any failure —
parse, compile, admission, unbound new port — leaves the previous
catalog serving (the mis-deployed-pack doctrine: degrade to the last
good catalog, never to no-action)."""

from typing import Callable, Optional

from metis.kernel import ForwardProjection, plan as kernel_plan
from metis.lang import compile_catalog, weight_policy
from metis.lang.compiler import CatalogDoc, Ports
from metis.lang.parser import parse_file
from .blackboard import Blackboard
from .registry import PortError, Registry


def _merge_ports(a: Ports, b: Ports) -> Ports:
    return Ports(weights={**a.weights, **b.weights},
                 guards={**a.guards, **b.guards},
                 inputs=a.inputs + tuple(
                     p for p in b.inputs if p not in a.inputs),
                 outputs=a.outputs + tuple(
                     p for p in b.outputs if p not in a.outputs),
                 weight_refs={**a.weight_refs, **b.weight_refs})


class Evaluator:
    def __init__(self, doc: CatalogDoc, registry: Registry,
                 domain: Optional[Callable] = None):
        self.registry = registry
        self.domain = domain
        self._serve(doc)

    # -- port validation (fail fast) -------------------------------------

    def _validate(self, doc: CatalogDoc):
        missing = []
        missing += [f"weight:{w}" for w in doc.ports.weights
                    if not self.registry.has_weight(w)]
        missing += [f"guard:{g}" for g in doc.ports.guards
                    if g not in self.registry.guards]
        missing += [f"input:{i}" for i in doc.ports.inputs
                    if i not in self.registry.inputs]
        missing += [f"output:{o}" for o in doc.ports.outputs
                    if o not in self.registry.outputs]
        if missing:
            raise PortError(missing)

    def _serve(self, doc: CatalogDoc):
        self._validate(doc)
        self.doc = doc

    # -- hot pack loading ------------------------------------------------

    def load_pack(self, source) -> CatalogDoc:
        """Compile+admit+revalidate, then swap. Raises (LangError /
        CatalogError / PortError) WITHOUT touching the serving catalog
        on any failure."""
        ast = parse_file(source) if not hasattr(source, "decls") \
            else source
        pack = compile_catalog(ast, base=self.doc)
        merged_cat = self.doc.catalog.admit(pack.catalog)
        merged = CatalogDoc(merged_cat,
                            _merge_ports(self.doc.ports, pack.ports),
                            source=self.doc.source)
        self._serve(merged)         # PortError here -> old doc serving
        return merged

    # -- the decision cycle ----------------------------------------------

    def tick(self, bb: Blackboard):
        if self.domain is None:
            raise RuntimeError("no domain seam registered")
        bb.require(self.doc.ports.inputs)
        sit = {p: self.registry.inputs[p](bb[p])
               for p in self.doc.ports.inputs}
        candidates, do, project, score, doomed = self.domain(sit)
        res = kernel_plan(candidates, do, project, score,
                          doomed_action=doomed)
        for p in self.doc.ports.outputs:
            bb[p] = self.registry.outputs[p](res, sit)
        return res


def catalog_domain(evaluator: Evaluator, candidates_fn, init_fn,
                   steps_fn, score_fn, doomed=None):
    """Generic seam for catalog-ground domains: do(a) edits the initial
    multiset, project grounds the SERVING catalog (hot swaps take
    effect on the next tick) with weight ports resolved through the
    registry (overlays included)."""

    def factory(sit):
        def do(a):
            return init_fn(sit, a)

        def project(init_atoms):
            doc = evaluator.doc
            prog = doc.catalog.ground(init_atoms)
            pol = None
            if doc.ports.weight_refs:
                pol = weight_policy(prog, doc.ports,
                                    evaluator.registry.weight_bindings())
            return ForwardProjection(prog, steps_fn(sit), policy=pol)

        return (candidates_fn(sit), do, project,
                lambda pr: score_fn(pr, sit), doomed)

    return factory
