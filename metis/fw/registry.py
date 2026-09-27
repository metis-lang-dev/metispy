"""Port bindings — the BT.cpp registerNodeType seam.

A catalog DECLARES ports (input/output/weight/guard); the registry
BINDS them to code. The evaluator fails fast at load when a declared
port has no binding.

Weight resolution implements the learning doctrine — experience is an
OVERLAY over the base distribution: overlays are consulted newest
first and may return None (miss -> fall through); the base binding is
total. Fitted tables and online posteriors are overlays; the catalog's
authored numbers/formulas are the base."""

from typing import Callable, Dict, List

Binding = Callable[[tuple, dict], float]     # (args, counts) -> float


class PortError(ValueError):
    def __init__(self, missing):
        super().__init__("unbound ports: " + ", ".join(sorted(missing)))
        self.missing = sorted(missing)


def _as_binding(b) -> Binding:
    if callable(b):
        return b
    if isinstance(b, (int, float)):
        return lambda args, counts, v=float(b): v
    if isinstance(b, dict):                  # table: args-tuple -> float
        return lambda args, counts, t=b: t.get(args)
    raise TypeError(f"cannot bind weight to {type(b).__name__}")


class Registry:
    def __init__(self):
        self.inputs: Dict[str, Callable] = {}
        self.outputs: Dict[str, Callable] = {}
        self.guards: Dict[str, Callable] = {}
        self._weights: Dict[str, Binding] = {}
        self._overlays: Dict[str, List[Binding]] = {}

    # -- registration ----------------------------------------------------

    def input(self, name, fn):
        self.inputs[name] = fn
        return self

    def output(self, name, fn):
        self.outputs[name] = fn
        return self

    def guard(self, name, fn):
        self.guards[name] = fn
        return self

    def weight(self, name, binding):
        """The BASE distribution for a factor (total: must yield a
        number for every reachable query)."""
        self._weights[name] = _as_binding(binding)
        return self

    def overlay(self, name, binding):
        """Experience over the base: consulted newest-first; None means
        miss (fall through). Tables miss on absent keys naturally."""
        self._overlays.setdefault(name, []).append(_as_binding(binding))
        return self

    def pop_overlay(self, name):
        if self._overlays.get(name):
            self._overlays[name].pop()
        return self

    # -- resolution ------------------------------------------------------

    def has_weight(self, name) -> bool:
        return name in self._weights

    def weight_bindings(self) -> Dict[str, Binding]:
        """Resolved factor table for lang.weight_policy: overlay chain
        first, base as the total fallback."""
        out = {}
        for name, base in self._weights.items():
            chain = list(self._overlays.get(name, ()))

            def resolved(args, counts, _chain=chain, _base=base):
                for ov in reversed(_chain):
                    v = ov(args, counts)
                    if v is not None:
                        return v
                v = _base(args, counts)
                if v is None:
                    raise PortError([f"weight base returned None for "
                                     f"args {args}"])
                return v

            out[name] = resolved
        return out
