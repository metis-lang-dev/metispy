"""The containment type system — "packs constrain, never enable",
mechanized as typechecking (docs/typing.md gives the rules in prose).

Vocabulary is partitioned into NAMESPACES; each namespace grants
PRODUCE and CONSUME rights to an explicit set of layers. Reading
(persist premises) is always allowed — observation cannot violate
containment. Backward predicates form a separate closed Horn world:
never resources, never produced, bodies reference only backward
predicates.

The containment theorem (by construction, admit() in catalog.py): a
pack can add new namespaces and rules, but cannot touch the rights of
existing namespaces — so no pack rule can ever produce or consume a
kernel atom the kernel didn't already grant. Ill-typed, not just
lint-flagged."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List


@dataclass(frozen=True)
class Namespace:
    name: str
    produce: FrozenSet[str]      # layers that may produce these atoms
    consume: FrozenSet[str]      # layers that may consume (linear premise)


def ns(name: str, produce, consume=None) -> Namespace:
    return Namespace(name, frozenset(produce),
                     frozenset(consume if consume is not None
                               else produce))


def _check_patterns(entry, schema, patterns, mode, preds, namespaces,
                    bwd, findings, diags=None):
    for pred, args in patterns:
        if pred in bwd:
            findings.append(f"{entry.name}: bwd predicate '{pred}' used "
                            f"as a resource ({mode})")
            continue
        if pred not in preds:
            findings.append(f"{entry.name}: undeclared predicate "
                            f"'{pred}' ({mode})")
            continue
        space = namespaces[preds[pred]]
        for m in ("produce", "consume"):
            allowed = space.produce if m == "produce" else space.consume
            if mode == m and entry.layer not in allowed:
                text = (f"{entry.name}: layer '{entry.layer}' may not "
                        f"{m} '{pred}' (namespace '{space.name}')")
                findings.append(text)
                if diags is not None:
                    from .diagnostics import make
                    diags.append(make(
                        f"containment-{m}", text, entry.src,
                        (("layer", entry.layer), ("pred", pred),
                         ("namespace", space.name))))
        for a in args:
            if a in schema.vars and schema.vars[a] is None:
                findings.append(f"{entry.name}: untyped var '{a}'")


def check_entry(entry, layers, stages, namespaces: Dict[str, Namespace],
                preds: Dict[str, str], bwd: FrozenSet[str],
                types: Dict[str, list], diags=None) -> List[str]:
    """Findings for one catalog entry (empty = well-typed)."""
    f: List[str] = []
    if not entry.comment.strip():
        f.append(f"{entry.name}: missing comment (the unit a human "
                 f"approves is mandatory)")
    if entry.layer not in layers:
        f.append(f"{entry.name}: unknown layer '{entry.layer}'")
        return f

    if entry.kind == "horn":
        hc = entry.payload
        if hc.head[0] not in bwd:
            f.append(f"{entry.name}: horn head '{hc.head[0]}' not a "
                     f"declared bwd predicate")
        for b in hc.body:
            if b[0] not in bwd:
                f.append(f"{entry.name}: horn body '{b[0]}' not bwd — "
                         f"the Horn world is closed")
        return f
    if entry.kind in ("prior", "binding"):
        return f

    sc = entry.payload   # ClauseSchema or LinkSchema
    for v, t in sc.vars.items():
        if t not in types:
            f.append(f"{entry.name}: var '{v}' has unknown type '{t}'")
    if entry.kind == "schema" and sc.stage not in stages:
        f.append(f"{entry.name}: unknown stage '{sc.stage}'")
    if entry.kind == "link" and (sc.pre not in stages
                                 or sc.post not in stages):
        f.append(f"{entry.name}: link stage(s) '{sc.pre}'->'{sc.post}' "
                 f"unknown")
    _check_patterns(entry, sc, sc.consume, "consume", preds, namespaces,
                    bwd, f, diags)
    _check_patterns(entry, sc, sc.produce, "produce", preds, namespaces,
                    bwd, f, diags)
    _check_patterns(entry, sc, sc.persist, "read", preds, namespaces,
                    bwd, f)
    for pred, args in sc.guards:
        if pred not in bwd:
            f.append(f"{entry.name}: guard on non-bwd predicate "
                     f"'{pred}' — guards are Horn premises only")
        for a in args:
            if a[:1].isupper() and a not in sc.vars:
                f.append(f"{entry.name}: guard uses undeclared var "
                         f"'{a}'")
    return f
