"""The formalization gate harness (docs/rewrite/07-formalization.md):
text -> LLP candidate -> the SAME admission gates as hand-authored
code, plus semantic instruments (classification, exact measure
diffs, MAP read-back). The formalizer that produced the .llp is
UNTRUSTED and pluggable; this module is the verification loop."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from metis.kernel import (STATE_BUDGET, ForwardProjection,
                          StateBudgetExceeded, program_key)
from metis.kernel.catalog import Catalog, CatalogError
from metis.kernel.factorize import FactoredProgram, FactorizeError
from metis.kernel.posterior import map_derivation

from .compiler import CatalogDoc, LangError, compile_catalog
from .parser import ParseError, parse


@dataclass
class Scenario:
    init: Dict[str, int]
    key: str = ""                       # ground program_key (merged)
    support: int = 0                    # final-slice support size
    diffs: List[Tuple[str, float, float]] = field(
        default_factory=list)           # atom, base P, merged P
    map_events: List[str] = field(default_factory=list)


@dataclass
class Formalization:
    text: str
    ok: bool = False
    findings: List[str] = field(default_factory=list)   # gate fails
    warnings: List[str] = field(default_factory=list)   # discipline
    rules: List[Tuple[str, str]] = field(
        default_factory=list)           # (name, doc) — approval units
    classification: str = "unclassified"
    reads_declared: Optional[bool] = None
    doc: Optional[CatalogDoc] = None
    merged: Optional[Catalog] = None
    scenarios: List[Scenario] = field(default_factory=list)


def _final_marginals(prog, steps) -> Tuple[Dict[str, float], int, list]:
    pr = ForwardProjection(prog, steps, max_states=STATE_BUDGET,
                           keep_history=False)
    marg: Dict[str, float] = {}
    for (_, kappa), p in pr.final.items():
        for atom, n in kappa:
            if n > 0:
                marg[atom] = marg.get(atom, 0.0) + p
    res = map_derivation(pr)
    events = prog.events()
    names = [events[e][2].name for e in res.events
             if e is not None]
    return marg, len(pr.final), names


def admit_text_pack(text: str, llp: str,
                    base: Optional[CatalogDoc] = None,
                    scenarios: Tuple[Dict[str, int], ...] = (),
                    steps: int = 6) -> Formalization:
    """Run a formalizer's candidate through the full gate chain.

    Returns a report: `ok` + findings on gate failure; on success
    the compiled doc, the merged catalog (when `base`), the (rule,
    doc) approval pairs, the class-F classification, and per
    scenario the ground key, exact measure diff vs the base (what
    the new rules CHANGED), and the MAP event read-back."""
    r = Formalization(text=text)
    if not text.strip():
        r.findings.append("empty source text — the text is the "
                          "approved unit (07-formalization §1.2)")
        return r

    # gate 1: parse + roundtrip
    try:
        cat = parse(llp)
    except (ParseError, Exception) as e:
        r.findings.append(f"parse: {e}")
        return r
    from . import ast as A
    try:
        if parse(A.pretty(cat)) != cat:
            r.findings.append("roundtrip: parse(pretty(a)) != a")
            return r
    except Exception as e:
        r.findings.append(f"roundtrip: {e}")
        return r

    # discipline warnings (not gates): provenance + doc carriage
    if cat.provenance is None or cat.provenance[0] != "text":
        r.warnings.append(
            'provenance is not `text "<ref>" "<date>"` — the '
            "formalization contract records its source")

    # gate 2: compile (against the base when given)
    try:
        r.doc = compile_catalog(cat, base=base)
    except (LangError, CatalogError) as e:
        r.findings.extend(getattr(e, "findings", [str(e)]))
        return r

    # gate 3: containment admission
    merged_cat = r.doc.catalog
    if base is not None:
        try:
            merged_cat = base.catalog.admit(r.doc.catalog)
        except CatalogError as e:
            r.findings.extend(getattr(e, "findings", [str(e)]))
            return r
    r.merged = merged_cat

    # discipline warning: traces, MAP read-back and the debug
    # sidecar identify events by the SHORT rule name — a collision
    # (across layers, or within one catalog) makes the narration
    # ambiguous even though entry names stay layer-qualified.
    short_counts: Dict[str, int] = {}
    for ent in merged_cat.entries:
        if type(ent.payload).__name__ in ("ClauseSchema", "LinkSchema"):
            short = ent.name.rsplit("/", 1)[-1]
            short_counts[short] = short_counts.get(short, 0) + 1
    for short, n in sorted(short_counts.items()):
        if n > 1:
            r.warnings.append(
                f"rule name '{short}' appears {n}x across layers — "
                "trace events and the debug sidecar read short "
                "names; rename to keep the read-back unambiguous")

    # the approval units
    for d in cat.decls:
        if isinstance(d, A.Stage):
            r.rules += [(rl.name, rl.doc) for rl in d.rules]
        elif isinstance(d, A.LinkDecl):
            r.rules.append((d.name, d.doc))

    # reads discipline for referenced weight ports
    refs = {ref.factor for ref in r.doc.ports.weight_refs.values()}
    if refs:
        r.reads_declared = refs <= set(r.doc.ports.reads)

    # classification + per-scenario semantics
    for init in scenarios:
        sc = Scenario(init=dict(init))
        prog = merged_cat.ground(dict(init))
        sc.key = program_key(prog)
        if r.classification == "unclassified":
            try:
                FactoredProgram(prog, steps)
                r.classification = ("class-F candidate (statics "
                                    "admit; Z-check applies with "
                                    "bindings)")
            except FactorizeError as e:
                r.classification = f"counterexample: {e}"
        try:
            marg, sc.support, sc.map_events = _final_marginals(prog,
                                                               steps)
            if base is not None:
                base_prog = base.catalog.ground(dict(init))
                base_marg, _, _ = _final_marginals(base_prog, steps)
                atoms = sorted(set(marg) | set(base_marg))
                sc.diffs = [
                    (a, base_marg.get(a, 0.0), marg.get(a, 0.0))
                    for a in atoms
                    if abs(base_marg.get(a, 0.0)
                           - marg.get(a, 0.0)) > 1e-12]
        except StateBudgetExceeded as e:
            # semantics out of exact reach is a WARNING on the
            # report, never a dead gate: the statics verdicts above
            # stand, the scenario ships without measure numbers.
            r.warnings.append(f"scenario {sc.key[:12]}…: {e}")
        r.scenarios.append(sc)
    if not scenarios:
        r.classification = "unclassified (no scenario provided)"

    r.ok = True
    return r
