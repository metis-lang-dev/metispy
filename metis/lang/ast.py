"""The lossless AST and its canonical printer.

The interpretability contract (inherited from rulescript R1): the
printed catalog is the reviewable artifact, and parse(pretty(ast)) ==
ast for every AST the parser produces. Docs (%% lines) are part of the
AST — a rule's doc is the unit a human approves."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class Term:
    kind: str            # "var" (Uppercase-first) | "const"
    value: str

    def __str__(self):
        return self.value


@dataclass(frozen=True)
class Atom:
    pred: str
    terms: Tuple[Term, ...] = ()
    persist: bool = False        # $-premise

    def __str__(self):
        base = self.pred if not self.terms else \
            f"{self.pred}({','.join(map(str, self.terms))})"
        return f"${base}" if self.persist else base


@dataclass(frozen=True)
class Distinct:
    a: str
    b: str

    def __str__(self):
        return f"{self.a} <> {self.b}"


@dataclass(frozen=True)
class Weight:
    value: Optional[float] = None        # static @w 0.55
    factor: Optional[str] = None         # port  @w beh(ego,keep)
    args: Tuple[Term, ...] = ()
    complement: bool = False             # @w ~cpt(C)

    def __str__(self):
        if self.factor is None:
            return f"@w {self.value:g}"
        a = "" if not self.args else \
            f"({','.join(map(str, self.args))})"
        return f"@w {'~' if self.complement else ''}{self.factor}{a}"


@dataclass(frozen=True)
class Rule:
    name: str
    layer: str
    doc: str
    body: Tuple[Union[Atom, Distinct], ...]
    head: Tuple[Atom, ...]
    weight: Optional[Weight] = None
    # source line (diagnostics only: excluded from equality/hash, so
    # the round-trip contract parse(pretty(ast)) == ast is untouched)
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class Stage:
    name: str
    rules: Tuple[Rule, ...]


@dataclass(frozen=True)
class LinkDecl:
    name: str
    layer: str
    doc: str
    pre: str
    pre_atoms: Tuple[Atom, ...]
    post: str
    post_atoms: Tuple[Atom, ...]
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class TypeDecl:
    name: str
    consts: Tuple[str, ...]
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class NamespaceDecl:
    name: str
    produce: Tuple[str, ...]
    consume: Optional[Tuple[str, ...]] = None    # None -> same as produce


@dataclass(frozen=True)
class PredDecl:
    name: str
    args: Tuple[str, ...]        # argument TYPES
    namespace: str
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class BwdDecl:
    name: str
    args: Tuple[str, ...]
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class ReadPattern:
    """One element of a weight port's declared read-scope: a state
    atom over the port's formal args, optionally guarded by bwd atoms
    that enumerate its free vars (`prev(D) if adj(C,D)`)."""
    atom: Atom
    guards: Tuple[Atom, ...] = ()

    def __str__(self):
        if not self.guards:
            return str(self.atom)
        return f"{self.atom} if {' * '.join(map(str, self.guards))}"


@dataclass(frozen=True)
class PortDecl:
    kind: str                    # weight | guard | input | output
    name: str
    args: Tuple[str, ...] = ()
    # weight ports only. None = undeclared (binding opaque); a tuple —
    # possibly empty — is the CONTRACT: the binding reads exactly these
    # atoms. The factored evaluator admits only declared ports.
    reads: Optional[Tuple[ReadPattern, ...]] = None


@dataclass(frozen=True)
class IncludeDecl:
    path: str


@dataclass(frozen=True)
class Fact:
    atom: Atom
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class HornRule:
    head: Atom
    body: Tuple[Atom, ...]
    line: int = field(default=0, compare=False, repr=False)
    col: int = field(default=0, compare=False, repr=False)
    file: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class Catalog:
    name: str
    version: int
    provenance: Optional[Tuple[str, str, str]]   # source, ref, date
    layers: Tuple[str, ...]
    stages: Tuple[str, ...]
    extends: Optional[str] = None                # pack mode: base name
    decls: Tuple[object, ...] = field(default_factory=tuple)


# ---- canonical printer -------------------------------------------------

def _doc(txt: str) -> str:
    return "".join(f"%% {line}\n" for line in txt.splitlines()) if txt \
        else ""


def _atoms(atoms) -> str:
    return " * ".join(str(a) for a in atoms)


def _head(atoms) -> str:
    """A sole LL-unit head prints as '()' (spec 07 P4, the canonical
    spelling; the parser reads both spellings onto the atom 'one')."""
    if len(atoms) == 1 and atoms[0].pred == "one" \
            and not atoms[0].terms and not atoms[0].persist:
        return "()"
    return _atoms(atoms)


def rule_line(r: Rule) -> str:
    """One clause, canonical, no doc and no indent (the pretty() body
    of a stage line; the diagnostic SUBJECT of a rule)."""
    w = f"  {r.weight}" if r.weight else ""
    return (f"{r.name} [{r.layer}] : {_atoms(r.body)} -o "
            f"{_head(r.head)}{w}.")


def decl_line(d) -> str:
    """One top-level declaration on one canonical line, no doc (stages
    are multi-line: use rule_line per clause)."""
    if isinstance(d, IncludeDecl):
        return f'include "{d.path}".'
    if isinstance(d, TypeDecl):
        return f"type {d.name} {{{' '.join(d.consts)}}}."
    if isinstance(d, NamespaceDecl):
        line = f"namespace {d.name} produce ({' '.join(d.produce)})"
        if d.consume is not None:
            line += f" consume ({' '.join(d.consume)})"
        return line + "."
    if isinstance(d, PredDecl):
        a = "" if not d.args else f"({','.join(d.args)})"
        return f"pred {d.name}{a} : {d.namespace}."
    if isinstance(d, BwdDecl):
        a = "" if not d.args else f"({','.join(d.args)})"
        return f"bwd {d.name}{a}."
    if isinstance(d, PortDecl):
        a = "" if not d.args else f"({','.join(d.args)})"
        r = "" if d.reads is None else \
            f" reads {{{', '.join(map(str, d.reads))}}}"
        return f"{d.kind} {d.name}{a}{r}."
    if isinstance(d, Fact):
        return f"{d.atom}."
    if isinstance(d, HornRule):
        return f"{d.head} :- {', '.join(str(b) for b in d.body)}."
    if isinstance(d, LinkDecl):
        pre = d.pre if not d.pre_atoms else \
            f"{d.pre} * {_atoms(d.pre_atoms)}"
        post = d.post if not d.post_atoms else \
            f"{d.post} * {_atoms(d.post_atoms)}"
        return f"qui {d.name} [{d.layer}] : {pre} -o {post}."
    raise TypeError(f"decl_line: {type(d).__name__}")


def pretty(cat: Catalog) -> str:
    out = [f"catalog {cat.name} {cat.version}."]
    if cat.extends:
        out.append(f"extends {cat.extends}.")
    if cat.provenance:
        s, r, d = cat.provenance
        out.append(f'provenance {s} "{r}" "{d}".')
    if cat.layers:
        out.append(f"layers ({' '.join(cat.layers)}).")
    if cat.stages:
        out.append(f"stages ({' '.join(cat.stages)}).")
    for d in cat.decls:
        if isinstance(d, Stage):
            out.append(f"stage {d.name} {{")
            for r in d.rules:
                out.append(f"{_doc(r.doc)}  {rule_line(r)}")
            out.append("}")
        elif isinstance(d, LinkDecl):
            out.append(f"{_doc(d.doc)}{decl_line(d)}")
        else:
            out.append(decl_line(d))
    return "\n".join(out) + "\n"
