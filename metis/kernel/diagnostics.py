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

# code -> (severity, migrated). `migrated` marks codes emitted as
# diagnostics end to end; the parity gate requires a fixture row for
# every migrated code. (Step-1 scaffolding: removed when step 2 lands.)
REGISTRY: Dict[str, Tuple[str, bool]] = {
    "doc-missing": ("error", True),
    "containment-produce": ("error", True),
    "containment-consume": ("error", True),
    "fact-var-conflict": ("error", True),
    "fact-not-bwd": ("error", False),
    "var-untypable": ("error", False),
    "fact-var-singleton": ("warning", False),
    "const-not-in-type": ("error", False),
    "rule-name-collision": ("warning", False),
    "provenance-shape": ("warning", False),
    "reads-undeclared": ("warning", False),
    "state-budget": ("note", False),
    "horn-nonground": ("error", False),
    "exact-unreachable": ("note", False),
    "llm-endpoint": ("note", False),
}

# code -> hint template over the code's data keys
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


_LAYER_CODES = {"containment-produce", "containment-consume"}


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


def render_text(d: Diagnostic) -> str:
    """The human rendering (REPL, metisc stderr) — byte-identical with
    metisc Diag.render_text."""
    f = os.path.basename(d.loc.file) if d.loc.file else "<input>"
    out = [f"{d.severity} {d.code} @ {f}:{d.loc.line}:{d.loc.col} "
           f"({d.loc.decl})"]
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
