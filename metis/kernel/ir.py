"""The IR wire — IR item 3 (docs/ir-plan.md): serialize a compiled
FactoredProgram (sites + SSA writes + guards + evidence points) with
its AOT elimination schedules into the line-based golden format the
C++ interpreter consumes (cpp/include/metis/ir.hpp). Python is the
compiler and the referee: every `M` expectation is computed here; the
C++ side must reproduce it by TABLE OPS ALONE — it ships no min-fill,
no grounding, no typechecker, exactly the item-3 decision.

Wire records (whitespace-separated; atoms/vars carry no spaces):
    C <name>                          case start
    V <var> <card>                    variable
    S <xvar> <nparents> <parents...> <ncands>
    K <weight> <ndeps> <deps...> <hook?>
        [<factor> <compl> <nargs> <args...> <nbase> <base...>
         <nvat> (<atom> <parentvar>)...]   one K per candidate
    W <xvar> <card> <newvar> <oldvar|-> <vals...>   val '-' = keep
    G <xvar> <card> <oldvar> <n> <outcomes...>
    P <var> <0|1>                     evidence point factor
    O <qvar> <n> <vars...>            elimination schedule
    M <atom> <qvar> <expected>        marginal expectation
    Y <n> <vars...>                   full-elimination schedule
    Z <n> (<var> <val>)... <expected> evidence-clamp mass (the
                                      surprise op: 0 = unmodeled)
    .                                 case end

PRODUCT-ARTIFACT records (ir_artifact_lines — the load-once,
evidence-per-tick runtime story; no expectations ship, certification
happened on the authoring side; cpp ir.hpp IrEval consumes):
    U <atom> <var> <0|1>              declared UNKNOWN slot: runtime-
                                      assignable evidence variable +
                                      its compile-time default (the
                                      evaluator clamps EVERY unknown:
                                      assignment else default —
                                      mirroring _calibration)
    A <atom> <qvar>                   answerable marginal query; its O
                                      schedule follows
    F <atom> <var>                    final-occurrence mapping for
                                      runtime likelihood clamps (the
                                      surprise op over observations)"""

from typing import Dict, List, Sequence

from .bpn import bpn_check
from .factorize import FactoredProgram, FactorizeError, _elim_cost


def _net_lines(fp: FactoredProgram) -> List[str]:
    """The structural records (V, S/K, W, G) shared by the golden
    case and the product artifact."""
    out = []
    for v in sorted(fp._cards):
        out.append(f"V {v} {fp._cards[v]}")
    for s in fp._sites:
        out.append(f"S {s.var} {len(s.parents)} "
                   + " ".join(s.parents)
                   + (" " if s.parents else "") + f"{len(s.recs)}")
        for ev, deps, hook in s.recs:
            line = f"K {ev.weight:.17g} {len(deps)}"
            for d in deps:
                line += f" {d}"
            if hook is None:
                line += " 0"
            else:
                _fn, _scope, base, varats, compl, meta = hook
                if meta is None:
                    raise FactorizeError(
                        f"{ev.name}: hook lacks serialization "
                        f"metadata (factor name/args)")
                factor, args = meta
                line += f" 1 {factor} {1 if compl else 0} {len(args)}"
                for a in args:
                    line += f" {a}"
                line += f" {len(base)}"
                for a in sorted(base):
                    line += f" {a}"
                line += f" {len(varats)}"
                for a, pv in varats:
                    line += f" {a} {pv}"
            out.append(line)
    for w in fp._writes:
        vals = " ".join("-" if v is None else str(v) for v in w.vals)
        out.append(f"W {w.x} {w.card} {w.new} {w.old or '-'} {vals}")
    for (x, card, old), outcomes in fp._guards.items():
        out.append(f"G {x} {card} {old} {len(outcomes)} "
                   + " ".join(str(j) for j in sorted(outcomes)))
    return out


def ir_case_lines(name: str, fp: FactoredProgram,
                  queries: List[str],
                  likelihoods: Sequence[Dict[str, int]] = ()
                  ) -> List[str]:
    bpn_check(fp)      # no ill-typed net ships (paper Prop 6; the
    #                    C++ side trusts the wire, it never re-types)
    out = [f"C {name}"]
    out += _net_lines(fp)
    for a, v in fp._unknown_vars.items():
        out.append(f"P {v} {1 if fp.program.init.get(a, 0) else 0}")
    scopes = fp._structure()
    for q in queries:
        v = fp._env.get(q, 0)
        if isinstance(v, int):
            raise FactorizeError(f"query '{q}' folded to the constant "
                                 f"{v} — corpus queries must be "
                                 f"variables")
        order = _elim_cost(scopes, fp._cards, {v})["order"]
        out.append(f"O {v} {len(order)} " + " ".join(order))
        out.append(f"M {q} {v} {fp.marginal(q):.12g}")
    for obs in likelihoods:
        clamps = []
        for a, val in sorted(obs.items()):
            v = fp._env.get(a, 0)
            if isinstance(v, int):
                raise FactorizeError(f"likelihood atom '{a}' folded "
                                     f"to a constant — corpus clamps "
                                     f"must be variables")
            clamps.append((v, 1 if val else 0))
        zorder = _elim_cost(scopes, fp._cards, set())["order"]
        out.append(f"Y {len(zorder)} " + " ".join(zorder))
        line = f"Z {len(clamps)}"
        for v, val in clamps:
            line += f" {v} {val}"
        out.append(line + f" {fp.likelihood(obs):.12g}")
    out.append(".")
    return out


def ir_artifact_lines(name: str, fp: FactoredProgram,
                      queries: List[str],
                      observables: Sequence[str] = (),
                      ports=None) -> List[str]:
    """The PRODUCT artifact: structure + declared unknown slots +
    per-query schedules + the full-elimination schedule + final-
    occurrence mapping — NO expectations (certification happened
    here, on the authoring side). The C++ IrEval loads it once and
    serves every tick: evidence = an assignment over the U slots,
    marginals along the shipped O orders, observation likelihood
    (the surprise op) along Y against the F mapping.

    Schedules stay exact under arbitrary slot clamps: every unknown
    is a singleton scope in the elimination graph already, and a
    point factor adds no edge."""
    bpn_check(fp)
    out = [f"C {name}"]
    if ports is not None:            # the B manifest (see wire.py)
        for name_ in sorted(ports.inputs):
            out.append(f"B in {name_}")
        for name_ in sorted(ports.outputs):
            out.append(f"B out {name_}")
        for name_ in sorted(ports.guards):
            out.append(f"B guard {name_}")
        for name_ in sorted(ports.weights):
            out.append(f"B weight {name_}")
    out += _net_lines(fp)
    for a, v in sorted(fp._unknown_vars.items()):
        out.append(f"U {a} {v} "
                   f"{1 if fp.program.init.get(a, 0) else 0}")
    scopes = fp._structure()
    for q in queries:
        v = fp._env.get(q, 0)
        if isinstance(v, int):
            raise FactorizeError(f"query '{q}' folded to the constant "
                                 f"{v} — artifact queries must be "
                                 f"variables")
        order = _elim_cost(scopes, fp._cards, {v})["order"]
        out.append(f"A {q} {v}")
        out.append(f"O {v} {len(order)} " + " ".join(order))
    zorder = _elim_cost(scopes, fp._cards, set())["order"]
    out.append(f"Y {len(zorder)} " + " ".join(zorder))
    for a in observables:
        v = fp._env.get(a, 0)
        if isinstance(v, int):
            raise FactorizeError(f"observable '{a}' folded to a "
                                 f"constant — cannot clamp")
        out.append(f"F {a} {v}")
    out.append(".")
    return out


def ir_debug_lines(name: str, fp: FactoredProgram) -> List[str]:
    """The DETACHED DEBUG INFO — the DWARF analogue for the wire.

    The artifact itself is name-free by design; 01-ir-spec's
    determinism contract (site-candidate order = canonical event
    order) makes rule identity DERIVABLE, so it ships as a sidecar
    (`D` lines), keyed to its artifact by name + program identity,
    never as authority: an interpreter joins artifact + sidecar to
    answer WHICH RULE a site's chosen candidate is (and, through the
    catalog's %% docs, WHY — the narrated-read-back seam of
    07-formalization). A missing sidecar degrades to indices, never
    to wrong names."""
    out = [f"C {name}"]
    for s in fp._sites:
        out.append(f"D {s.var} {len(s.recs)} "
                   + " ".join(ev.name for ev, _, _ in s.recs))
    return out
