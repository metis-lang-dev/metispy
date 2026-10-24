"""Structured diagnostics — findings as certified slices (spec 08 D0).

One shape on every surface, byte-identical with metisc (src/diag.ml):

    code      stable kebab ident (REGISTRY)
    severity  error | warning | note (from the registry, never ad hoc)
    loc       file, line, col, decl (the owning path: stage/rule, qui/link,
              fact/pred, horn/pred)
    subject   the offending construct ALONE: its %% doc lines + one
              canonical line (rule_line / decl_line)
    context   the minimal closure that makes the subject readable without
              the catalog (spec 08 D0 amendment): the subject's preds in
              source order, first occurrence only -> their namespaces ->
              their types as name + cardinality -> `layers (...)` when the
              code concerns layers. Capped at CONTEXT_CAP; overflow raises
              (a diagnostic defect to fix, never silently truncated).
    data      code-specific (key, value) pairs, fixed order per code
    hint      one imperative repair sentence
    text      the legacy finding string — the text rendering every
              existing surface prints, unchanged

`make` is the ONE constructor; no call site assembles a Diagnostic
ad hoc. The compiler precomputes each declaration's `Source` (loc +
subject + context) and threads it through Entry, so kernel checks that
fire far from the AST (containment, at admission) copy it verbatim."""

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

CONTEXT_CAP = 12

# code -> (severity, phase). phase says WHERE a code fires:
#   compile | kernel | admission  — from a .llp; the D2 gate requires a
#                                   fixture row for every one of these
#   ground                        — grounding a unit that compiles
#                                   (spec 12: an abstract type left
#                                   abstract); gated alike — a listing
#                                   runs the ground check after a clean
#                                   compile
#   loop | runtime                — the gate cascade / instruments
#                                   (fixtures with the REPL surface)
#   internal                      — defensive kernel invariants
#                                   unreachable from source (hand-built
#                                   catalogs only)
#   retired                       — kept so the code is never reused;
#                                   no emitter
REGISTRY: Dict[str, Tuple[str, str]] = {
    # compile
    "doc-missing": ("error", "compile"),
    # fact-var-conflict / var-type-conflict: SIBLINGS — the same
    # inexactness on facts vs rules/links. Kept apart on purpose (codes
    # are API; step 1 froze fact-var-conflict in goldens): don't unify.
    "fact-var-conflict": ("error", "compile"),
    "var-type-conflict": ("error", "compile"),
    "var-untypable": ("error", "compile"),
    "rhs-var-unbound": ("error", "compile"),
    "arity-mismatch": ("error", "compile"),
    "fact-not-bwd": ("error", "compile"),
    "link-post-bwd": ("error", "compile"),
    "const-not-in-type": ("error", "compile"),
    "fact-var-singleton": ("warning", "compile"),
    "weight-port-undeclared": ("error", "compile"),
    "weight-arg-not-var": ("error", "compile"),
    "reads-formal-not-var": ("error", "compile"),
    "reads-persist-marker": ("error", "compile"),
    "reads-not-resource": ("error", "compile"),
    "reads-guard-not-bwd": ("error", "compile"),
    "reads-var-unbound": ("error", "compile"),
    "include-cycle": ("error", "compile"),
    "extends-no-base": ("error", "compile"),
    "decl-duplicate": ("error", "compile"),
    # spec 12: a signature naming a type nobody declared (it replaces
    # var-type-unknown, which reported it per rule variable, later);
    # an abstract type the unit leaves abstract (advisory)
    "type-unknown": ("error", "compile"),
    "type-abstract": ("warning", "compile"),
    # kernel
    "containment-produce": ("error", "kernel"),
    "containment-consume": ("error", "kernel"),
    "pred-undeclared": ("error", "kernel"),
    "bwd-as-resource": ("error", "kernel"),
    "layer-unknown": ("error", "kernel"),
    "stage-unknown": ("error", "kernel"),
    "link-stage-unknown": ("error", "kernel"),
    "horn-head-not-bwd": ("error", "kernel"),
    "horn-body-not-bwd": ("error", "kernel"),
    "pred-namespace-unknown": ("error", "kernel"),
    "pred-resource-and-bwd": ("error", "kernel"),
    "namespace-layer-unknown": ("error", "kernel"),
    # admission
    "pack-redeclares": ("error", "admission"),
    "pack-claims-base-namespace": ("error", "admission"),
    "pack-adds-stage": ("error", "admission"),
    # ground (spec 12)
    "type-incomplete": ("error", "ground"),
    # loop / runtime (step 3)
    "rule-name-collision": ("warning", "loop"),
    "provenance-shape": ("warning", "loop"),
    # a read outside the declared scope RAISES (factorize): an error
    "reads-undeclared": ("error", "runtime"),
    "state-budget": ("note", "runtime"),
    "horn-nonground": ("error", "runtime"),
    "exact-unreachable": ("note", "runtime"),
    "llm-endpoint": ("note", "runtime"),
    # internal (defensive invariants, unreachable from source)
    "guard-not-bwd": ("error", "internal"),
    "guard-var-undeclared": ("error", "internal"),
    "comment-missing": ("error", "internal"),
    "var-untyped": ("error", "internal"),
    "kind-unknown": ("error", "internal"),
    # retired (spec 12): every variable's type comes from a signature,
    # so type-unknown at the signature reports it, once
    "var-type-unknown": ("error", "retired"),
}

GATED_PHASES = ("compile", "kernel", "admission", "ground")

# code -> hint template over the code's data keys (data key order is
# fixed per code at the emission site)
HINTS: Dict[str, str] = {
    "doc-missing":
        "add a %% doc line above {rule}: the doc is the approved text "
        "unit, mandatory on every rule and link",
    "containment-produce":
        "layer '{layer}' may not produce {pred} (namespace "
        "'{namespace}'): read it with ${pred}, or declare the atom in a "
        "namespace your layer owns",
    "containment-consume":
        "layer '{layer}' may not consume {pred} (namespace "
        "'{namespace}'): read it with ${pred} instead of spending it",
    "fact-var-conflict":
        "{var} sits at a {type1} and a {type2} position: rename one "
        "occurrence, or fix the predicate signature",
    "var-type-conflict":
        "{var} sits at a {type1} and a {type2} position: rename one "
        "occurrence, or fix the predicate signature",
    "var-untypable":
        "{var} occurs in no typed position: use it in an atom whose "
        "signature types it, or drop it",
    "rhs-var-unbound":
        "a rule produces only what it read or consumed: bind {var} in an "
        "atom on the left; a fresh value needs a bwd table or a constant",
    "arity-mismatch":
        "{pred} takes {expected} argument(s), got {got}: match the "
        "declared signature",
    "fact-not-bwd":
        "a fact is Horn knowledge: declare {pred} with `bwd`, or make it "
        "an #init resource",
    "link-post-bwd":
        "a link may not produce the Horn atom {pred}: bwd predicates are "
        "knowledge, not state",
    "const-not-in-type":
        "'{const}' is not a constant of {type}: add it to the type, or "
        "fix argument {arg} of {pred}",
    "fact-var-singleton":
        "if intended, name it Any{var} (a leading Any marks a "
        "don't-care); otherwise repeat the variable where it belongs",
    "weight-port-undeclared":
        "declare the port (`weight {port}(...) reads {{...}}.`), or drop "
        "the @w reference",
    "weight-arg-not-var":
        "pass rule variables to {port}: {arg} is not a variable of this "
        "rule",
    "reads-formal-not-var":
        "the formals of weight {port} must be variables (got {formals})",
    "reads-persist-marker":
        "drop the $ in {atom}: a reads scope is already read-only",
    "reads-not-resource":
        "a reads scope names resource predicates only: declare {pred} "
        "with `pred`, or drop it from the scope of {port}",
    "reads-guard-not-bwd":
        "a reads guard must be a bwd predicate: declare {pred} with "
        "`bwd`",
    "reads-var-unbound":
        "bind {var}: make it a formal of {port}, or enumerate it with a "
        "guard",
    "include-cycle":
        "break the include cycle at {path}",
    "extends-no-base":
        "compile this pack against its base catalog {base}",
    "decl-duplicate":
        "a name is declared once per kind: remove this {kind} "
        "declaration of {name}, or rename it (first at {first})",
    "type-unknown":
        "declare {type}, or fix the signature of {pred} that names it",
    "type-abstract":
        "the instance that includes this file completes {type} with "
        "`type {type} {{…}}.`: ground that instance",
    "pred-undeclared":
        "declare {pred}: `pred {pred}(...) : <namespace>.`",
    "bwd-as-resource":
        "{pred} is a bwd predicate: use it as a guard premise only, "
        "never as a resource ({mode})",
    "layer-unknown":
        "add {layer} to the catalog's layers (...), or tag the rule with "
        "a declared layer",
    "stage-unknown":
        "add {stage} to stages (...)",
    "link-stage-unknown":
        "declare both link stages ({pre}, {post}) in stages (...)",
    "horn-head-not-bwd":
        "a Horn rule may only derive bwd predicates: declare {pred} with "
        "`bwd`",
    "horn-body-not-bwd":
        "the Horn world is closed: {pred} must be a bwd predicate",
    "pred-namespace-unknown":
        "declare namespace {namespace}, or move {pred} into a declared "
        "one",
    "pred-resource-and-bwd":
        "{pred} is declared both `pred` and `bwd`: keep one",
    "namespace-layer-unknown":
        "add {layer} to layers (...), or drop it from namespace "
        "{namespace}",
    "pack-redeclares":
        "a pack may only add: rename the pack's {kind} {name}",
    "pack-claims-base-namespace":
        "a pack owns only its own namespaces: put {pred} in a namespace "
        "the pack declares",
    "pack-adds-stage":
        "a pack may not add stage {stage}: the stage set is the base's",
    "type-incomplete":
        "ground the instance that includes this file, or complete {type} "
        "here with `type {type} {{…}}.`",
    # loop (the gate cascade) — metispy only: metisc has no loop
    "rule-name-collision":
        "rename one of the {count} rules named {rule}: trace events and "
        "the debug sidecar read SHORT names",
    "provenance-shape":
        "declare `provenance text \"<ref>\" \"<date>\".`: the "
        "formalization contract records its source",
    # runtime (instruments speaking)
    "reads-undeclared":
        "add {atom} to the port's reads scope, or stop reading it in the "
        "binding: the declared scope is the contract",
    "state-budget":
        "lower the horizon (try: {retry}), or this is clique-tree / "
        "factored territory",
    "horn-nonground":
        "guard the unbound variable with a predicate that enumerates it",
    "exact-unreachable":
        "the sampled counts stand; for exact numbers lower #steps or "
        "shrink the scene ({reason})",
    "llm-endpoint":
        "{problem}: start the server or `ollama pull {model}`, or pick "
        "another endpoint with `llm`",
}


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int
    decl: str


@dataclass(frozen=True)
class Source:
    """A declaration's precomputed slice (built by the compiler)."""
    loc: Loc
    subject: Tuple[str, ...]
    context: Tuple[str, ...]        # preds -> namespaces -> types
    layers: str = ""                # the `layers (...)` line


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: str
    loc: Loc
    subject: Tuple[str, ...]
    context: Tuple[str, ...]
    data: Tuple[Tuple[str, str], ...]
    hint: str
    text: str


class DiagnosticDefect(ValueError):
    """A diagnostic that cannot be built as specified (unknown code,
    context over the cap) — a bug in the emitter, never silent."""


_LAYER_CODES = {"containment-produce", "containment-consume",
                "layer-unknown", "namespace-layer-unknown"}


def make(code: str, text: str, src: Optional[Source],
         data: Tuple[Tuple[str, str], ...]) -> Diagnostic:
    if code not in REGISTRY:
        raise DiagnosticDefect(f"unregistered diagnostic code '{code}'")
    severity = REGISTRY[code][0]
    if src is None:
        src = Source(Loc("", 0, 0, ""), (), ())
    context = src.context
    if code in _LAYER_CODES and src.layers:
        context = context + (src.layers,)
    if len(context) > CONTEXT_CAP:
        raise DiagnosticDefect(
            f"{code} @ {src.loc.decl}: context of {len(context)} lines "
            f"exceeds the cap of {CONTEXT_CAP}")
    hint = HINTS.get(code, "").format(**dict(data))
    return Diagnostic(code, severity, src.loc, src.subject, context,
                      tuple(data), hint, text)


def note(code: str, text: str, decl: str, subject=(), data=()) \
        -> Diagnostic:
    """A runtime / loop diagnostic: no declaration owns it, so its slice
    is named by the INSTRUMENT that spoke (decl = query | map | trace |
    scenario | llm | factorize | horn | gate) and its subject is the
    invocation itself."""
    return make(code, text,
                Source(Loc("", 0, 0, decl), tuple(subject), ()),
                tuple(data))


def budget_note(e, instrument: str, invocation: str,
                retry: str) -> Diagnostic:
    """state-budget from a StateBudgetExceeded: the exact filter
    retained more states (cumulative, Σ support) than the budget — a
    structural answer, not a crash."""
    return note("state-budget", str(e), instrument, (invocation,),
                (("step", str(e.step)), ("steps", str(e.steps)),
                 ("retained", str(e.retained)),
                 ("budget", str(e.budget)), ("retry", retry)))


def render_text(d: Diagnostic) -> str:
    """The human rendering (REPL, metisc stderr) — byte-identical with
    metisc Diag.render_text."""
    if d.loc.line > 0:
        f = os.path.basename(d.loc.file) if d.loc.file else "<input>"
        out = [f"{d.severity} {d.code} @ {f}:{d.loc.line}:{d.loc.col} "
               f"({d.loc.decl})"]
    else:       # no source position (runtime notes, header-level)
        out = [f"{d.severity} {d.code} ({d.loc.decl})"]
    out += [f"  {s}" for s in d.subject]
    out += [f"  | {c}" for c in d.context]
    if d.hint:
        out.append(f"  -> {d.hint}")
    return "\n".join(out) + "\n"


def to_json(d: Diagnostic) -> str:
    """The wire rendering: one compact JSON object, keys in this order,
    non-ASCII raw — byte-identical with metisc Diag.to_json."""
    obj = {
        "code": d.code, "severity": d.severity,
        "loc": {"file": d.loc.file, "line": d.loc.line,
                "col": d.loc.col, "decl": d.loc.decl},
        "subject": list(d.subject), "context": list(d.context),
        "data": {k: v for k, v in d.data},
        "hint": d.hint, "text": d.text,
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
