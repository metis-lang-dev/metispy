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
from .diagnostics import Diagnostic, Loc, Source
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


def _layer_order(space) -> List[str]:
    """A namespace's layers in a deterministic order (sorted — the
    frozensets keep no declaration order; metisc sorts identically)."""
    return sorted(space.produce | space.consume)


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
    # abstract types (spec 12): declared by `types (…)`, completed by no
    # `type t {…}` — in the signature with no constants; grounding
    # refuses while one is left (an instance, or a pack, completes it)
    abstract: Tuple[str, ...] = ()
    # decl slices for catalog-level / admission diagnostics (spec 08),
    # keyed "pred/x", "bwd/x", "namespace/y", "type/t", "types/t"; filled by the
    # compiler, never compared
    srcs: Dict[str, Source] = field(default_factory=dict, compare=False,
                                    repr=False)

    # -- typecheck -------------------------------------------------------

    def typecheck(self, diags: Optional[List[Diagnostic]] = None) \
            -> List[str]:
        from .typing import _emit
        f: List[str] = []
        for p, space in self.preds.items():
            if space not in self.namespaces:
                _emit(f, diags, "pred-namespace-unknown",
                      f"predicate '{p}': unknown namespace '{space}'",
                      self.srcs.get(f"pred/{p}"),
                      (("pred", p), ("namespace", space)))
            if p in self.bwd:
                _emit(f, diags, "pred-resource-and-bwd",
                      f"predicate '{p}' declared both resource and bwd",
                      self.srcs.get(f"pred/{p}"), (("pred", p),))
        for space in self.namespaces.values():
            for layer in _layer_order(space):
                if layer not in self.layers:
                    _emit(f, diags, "namespace-layer-unknown",
                          f"namespace '{space.name}': unknown layer "
                          f"'{layer}'",
                          self.srcs.get(f"namespace/{space.name}"),
                          (("namespace", space.name), ("layer", layer)))
        for e in self.entries:
            if e.kind not in KINDS:
                _emit(f, diags, "kind-unknown",
                      f"{e.name}: unknown kind '{e.kind}'", e.src,
                      (("kind", e.kind),))
                continue
            f.extend(check_entry(e, self.layers, self.stages,
                                 self.namespaces, self.preds, self.bwd,
                                 self.sig.types, diags))
        return f

    def incomplete(self, diags: Optional[List[Diagnostic]] = None) \
            -> List[str]:
        """type-incomplete (spec 12 P3): one finding per abstract type
        still abstract, in declaration order, at its `types (…)` line —
        what grounding refuses with."""
        from .typing import _emit
        f: List[str] = []
        for t in self.abstract:
            src = self.srcs.get(f"types/{t}")
            decl = src.subject[0][:-1] if src is not None and src.subject \
                else f"types ({t})"
            _emit(f, diags, "type-incomplete",
                  f"type '{t}' has no members: declared by `{decl}`, "
                  f"completed by no `type {t} {{…}}` — the instance that "
                  f"includes this file supplies it", src, (("type", t),))
        return f

    def checked(self) -> "Catalog":
        diags: List[Diagnostic] = []
        f = self.typecheck(diags)
        if f:
            raise CatalogError(f, diags)
        return self

    # -- admission (containment by construction) -------------------------

    def admit(self, pack: "Catalog") -> "Catalog":
        from .typing import _emit
        f: List[str] = []
        d: List[Diagnostic] = []
        src = pack.srcs.get
        for t in pack.sig.types:
            # a pack may complete the base's abstract type (spec 12 P5)
            if t in self.sig.types and t not in self.abstract \
                    and t not in pack.abstract:
                _emit(f, d, "pack-redeclares",
                      f"pack redeclares type '{t}'", src(f"type/{t}"),
                      (("kind", "type"), ("name", t)))
        for n in pack.namespaces:
            if n in self.namespaces:
                _emit(f, d, "pack-redeclares",
                      f"pack redeclares namespace '{n}'",
                      src(f"namespace/{n}"),
                      (("kind", "namespace"), ("name", n)))
        for p, space in pack.preds.items():
            if p in self.preds or p in self.bwd:
                _emit(f, d, "pack-redeclares",
                      f"pack redeclares predicate '{p}'", src(f"pred/{p}"),
                      (("kind", "predicate"), ("name", p)))
            elif space in self.namespaces:
                _emit(f, d, "pack-claims-base-namespace",
                      f"pack claims base namespace '{space}' for '{p}' — "
                      f"packs own only their namespaces", src(f"pred/{p}"),
                      (("pred", p), ("namespace", space)))
        for p in pack.bwd:
            if p in self.bwd or p in self.preds:
                _emit(f, d, "pack-redeclares",
                      f"pack redeclares bwd predicate '{p}'",
                      src(f"bwd/{p}"),
                      (("kind", "bwd predicate"), ("name", p)))
        for s in pack.stages:
            if s not in self.stages:
                _emit(f, d, "pack-adds-stage",
                      f"pack adds stage '{s}' — stage set is the kernel's",
                      Source(Loc("", 0, 0, "stages"),
                             (f"stages ({' '.join(pack.stages)}).",), ()),
                      (("stage", s),))
        if f:
            raise CatalogError(f, d)
        completes = {t for t in pack.sig.types if t not in pack.abstract}
        merged = Catalog(
            name=f"{self.name}+{pack.name}",
            version=self.version,
            layers=self.layers,
            stages=self.stages,
            sig=Signature({**self.sig.types,
                           **{t: cs for t, cs in pack.sig.types.items()
                              if t in completes or t not in self.sig.types}}),
            namespaces={**self.namespaces, **pack.namespaces},
            preds={**self.preds, **pack.preds},
            bwd=self.bwd | pack.bwd,
            entries=self.entries + pack.entries,
            abstract=tuple(t for t in self.abstract if t not in completes)
            + tuple(t for t in pack.abstract if t not in self.sig.types),
            srcs={**self.srcs, **pack.srcs})
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
        diags: List[Diagnostic] = []
        f = self.incomplete(diags)
        if f:
            raise CatalogError(f, diags)
        schemas = [e.payload for e in self.entries if e.kind == "schema"]
        links = [e.payload for e in self.entries if e.kind == "link"]
        return ground_program(self.stages, schemas, links, self.sig,
                              self.horn_db(),
                              init_stage or self.stages[0], init_atoms)
