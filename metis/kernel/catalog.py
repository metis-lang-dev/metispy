"""The catalog: the authored unit — typed vocabulary, layered rights,
entries with provenance — and its two operations: typecheck and admit.

A catalog owns: a Signature (finite types), namespaces with layer
rights (typing.py), a predicate->namespace map, the bwd predicate set,
stage order, and ENTRIES (schemas, links, horn clauses, priors,
bindings), each carrying a mandatory human-approvable comment and a
provenance record (hand | surprise | template) — inherited verbatim
from the wayplan catalog doctrine.

admit(base, pack) is containment by construction: a pack may only ADD —
new namespaces, new predicates in ITS OWN new namespaces, new bwd
predicates, new types, new entries. It may never redeclare or extend
anything existing; therefore no pack entry can acquire produce/consume
rights on a base namespace, and the typechecker rejects any attempt.

ground() instantiates the whole catalog against an initial situation
into a kernel Program."""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

from .clause import Clause, Link                     # noqa: F401
from .diagnostics import Diagnostic, Source
from .ground import ground_program
from .horn import HornClause, HornDB
from .program import Program
from .schema import ClauseSchema, LinkSchema, Signature
from .typing import Namespace, check_entry

KINDS = ("schema", "link", "horn", "prior", "binding")


@dataclass(frozen=True)
class Provenance:
    source: str          # hand | surprise | template
    ref: str             # human-readable origin
    date: str            # YYYY-MM-DD


@dataclass
class Entry:
    name: str            # layer-qualified, e.g. "rss/dmin-floor"
    layer: str
    kind: str            # one of KINDS
    comment: str         # NL intent/trigger/effect — mandatory
    provenance: Provenance
    payload: Union[ClauseSchema, LinkSchema, HornClause, dict, None] \
        = None
    # the declaration's precomputed diagnostic slice (loc + subject +
    # context, spec 08): kernel checks far from the AST copy it
    src: Optional[Source] = field(default=None, compare=False,
                                  repr=False)


class CatalogError(ValueError):
    def __init__(self, findings: List[str],
                 diagnostics: Optional[List[Diagnostic]] = None):
        super().__init__("catalog failed checks:\n  "
                         + "\n  ".join(findings))
        self.findings = findings
        # the structured findings that are migrated (spec 08); the
        # strings above stay the text rendering
        self.diagnostics = list(diagnostics or [])


@dataclass
class Catalog:
    name: str
    version: int
    layers: Tuple[str, ...]                  # authority order
    stages: Tuple[str, ...]                  # program stage order
    sig: Signature
    namespaces: Dict[str, Namespace]
    preds: Dict[str, str]                    # predicate -> namespace
    bwd: FrozenSet[str] = frozenset()
    entries: List[Entry] = field(default_factory=list)

    # -- typecheck -------------------------------------------------------

    def typecheck(self, diags: Optional[List[Diagnostic]] = None) \
            -> List[str]:
        f: List[str] = []
        for p, space in self.preds.items():
            if space not in self.namespaces:
                f.append(f"predicate '{p}': unknown namespace '{space}'")
            if p in self.bwd:
                f.append(f"predicate '{p}' declared both resource and "
                         f"bwd")
        for space in self.namespaces.values():
            for layer in space.produce | space.consume:
                if layer not in self.layers:
                    f.append(f"namespace '{space.name}': unknown layer "
                             f"'{layer}'")
        for e in self.entries:
            if e.kind not in KINDS:
                f.append(f"{e.name}: unknown kind '{e.kind}'")
                continue
            f.extend(check_entry(e, self.layers, self.stages,
                                 self.namespaces, self.preds, self.bwd,
                                 self.sig.types, diags))
        return f

    def checked(self) -> "Catalog":
        diags: List[Diagnostic] = []
        f = self.typecheck(diags)
        if f:
            raise CatalogError(f, diags)
        return self

    # -- admission (containment by construction) -------------------------

    def admit(self, pack: "Catalog") -> "Catalog":
        f: List[str] = []
        for t in pack.sig.types:
            if t in self.sig.types:
                f.append(f"pack redeclares type '{t}'")
        for n in pack.namespaces:
            if n in self.namespaces:
                f.append(f"pack redeclares namespace '{n}'")
        for p, space in pack.preds.items():
            if p in self.preds or p in self.bwd:
                f.append(f"pack redeclares predicate '{p}'")
            elif space in self.namespaces:
                f.append(f"pack claims base namespace '{space}' for "
                         f"'{p}' — packs own only their namespaces")
        for p in pack.bwd:
            if p in self.bwd or p in self.preds:
                f.append(f"pack redeclares bwd predicate '{p}'")
        for s in pack.stages:
            if s not in self.stages:
                f.append(f"pack adds stage '{s}' — stage set is the "
                         f"kernel's")
        if f:
            raise CatalogError(f)
        merged = Catalog(
            name=f"{self.name}+{pack.name}",
            version=self.version,
            layers=self.layers,
            stages=self.stages,
            sig=Signature({**self.sig.types, **pack.sig.types}),
            namespaces={**self.namespaces, **pack.namespaces},
            preds={**self.preds, **pack.preds},
            bwd=self.bwd | pack.bwd,
            entries=self.entries + pack.entries)
        return merged.checked()

    # -- grounding -------------------------------------------------------

    # A catalog is never mutated after construction (packs and
    # revisions build NEW catalogs), so its Horn world is cached on the
    # instance and keeps its variant tables across groundings. The
    # GROUND PROGRAM is not cached here: callers may treat it as a
    # fresh mutable value (e.g. planting clause weights) — read-only
    # callers cache it themselves (metis.loop.wire._scene_prog).

    def horn_db(self) -> HornDB:
        db = self.__dict__.get("_horn_db")
        if db is None:
            db = HornDB([e.payload for e in self.entries
                         if e.kind == "horn"])
            self.__dict__["_horn_db"] = db
        return db

    def ground(self, init_atoms, init_stage: Optional[str] = None) \
            -> Program:
        self.checked()
        schemas = [e.payload for e in self.entries if e.kind == "schema"]
        links = [e.payload for e in self.entries if e.kind == "link"]
        return ground_program(self.stages, schemas, links, self.sig,
                              self.horn_db(),
                              init_stage or self.stages[0], init_atoms)
