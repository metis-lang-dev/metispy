"""AST -> kernel Catalog (+ ports). The compiler is where the language
meets the stage-2 machinery: everything semantic (namespaces, rights,
guards-vs-resources, Horn closure, admission) is the KERNEL's job — the
compiler only translates and infers.

Var typing is inferred from predicate signatures: a variable's type is
the declared argument type at every position it occupies; conflicts and
untypable vars are errors. Guards need no marker — a body atom whose
predicate is declared `bwd` IS a guard (matching kernel typing).

Weights: `@w 0.55` -> the schema prior; `@w [~]factor(args)` -> prior 1
plus a PORT REFERENCE resolved at projection time by weight_policy()
against a registry {factor: callable(args_tuple, counts) -> float}
(complement = 1 - f). Registry resolution order for learning overlays
(experience over base distribution) is the stage-10 Registry's job —
here a plain dict suffices."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from metis.kernel import Catalog, CatalogError, ClauseSchema, Entry, \
    HornClause, LinkSchema, Provenance, Signature, ns
from . import ast
from .parser import parse_file


class LangError(ValueError):
    def __init__(self, findings, diagnostics=None):
        super().__init__("catalog compilation failed:\n  "
                         + "\n  ".join(findings))
        self.findings = findings
        # the migrated structured findings (spec 08); the strings above
        # stay the text rendering
        self.diagnostics = list(diagnostics or [])


@dataclass(frozen=True)
class WeightRef:
    factor: str
    args: Tuple[str, ...]        # constants or schema var names
    complement: bool


@dataclass
class Ports:
    weights: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    guards: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    inputs: Tuple[str, ...] = ()
    outputs: Tuple[str, ...] = ()
    # schema name -> its weight-port reference
    weight_refs: Dict[str, WeightRef] = field(default_factory=dict)
    # factor -> declared read-scope; PRESENCE is the contract (a port
    # absent here has an opaque binding — not admissible for factoring)
    reads: Dict[str, Tuple[ast.ReadPattern, ...]] = \
        field(default_factory=dict)


@dataclass
class CatalogDoc:
    catalog: Catalog
    ports: Ports
    source: Optional[ast.Catalog] = None
    # advisory, never gating (e.g. a singleton var in a Horn fact)
    warnings: List[str] = field(default_factory=list)
    # structured findings of a SUCCESSFUL compile (spec 08; warnings and
    # notes once migrated — errors travel on LangError/CatalogError)
    diagnostics: List = field(default_factory=list)


def _pattern(a: ast.Atom):
    return (a.pred, tuple(t.value for t in a.terms))


def _infer_vars(name, atoms, sigs, findings, emit=None,
                conflict_code="var-type-conflict") -> Dict[str, str]:
    vars_: Dict[str, str] = {}
    for a in atoms:
        sig = sigs.get(a.pred)
        if sig is None:
            continue          # kernel typecheck reports undeclared preds
        if len(sig) != len(a.terms):
            text = (f"{name}: {a.pred} arity "
                    f"{len(a.terms)} != declared {len(sig)}")
            if emit is not None:
                emit("arity-mismatch", text,
                     (("pred", a.pred), ("expected", str(len(sig))),
                      ("got", str(len(a.terms)))))
            else:
                findings.append(text)
            continue
        for t, ty in zip(a.terms, sig):
            if t.kind != "var":
                continue
            if vars_.setdefault(t.value, ty) != ty:
                text = (f"{name}: var {t.value} typed both "
                        f"{vars_[t.value]} and {ty}")
                if emit is not None:
                    emit(conflict_code, text,
                         (("var", t.value), ("type1", vars_[t.value]),
                          ("type2", ty)))
                else:
                    findings.append(text)
    return vars_


def _check_consts(where, atoms, sigs, types, findings,
                  emit=None) -> None:
    """A CONSTANT in an argument position must be a member of the type
    that position declares (`eq(red,n7)` with eq(nat,nat): finding).
    Positions whose type is not declared here are skipped — the kernel
    typecheck reports undeclared types."""
    for a in atoms:
        sig = sigs.get(a.pred)
        if sig is None or len(sig) != len(a.terms):
            continue          # undeclared pred / arity: reported elsewhere
        for k, (t, ty) in enumerate(zip(a.terms, sig), 1):
            if t.kind == "var" or ty not in types:
                continue
            if t.value not in types[ty]:
                text = (f"{where}: constant '{t.value}' is not "
                        f"a {ty} ({a.pred} argument {k})")
                if emit is not None:
                    emit("const-not-in-type", text,
                         (("const", t.value), ("type", ty),
                          ("pred", a.pred), ("arg", str(k))))
                else:
                    findings.append(text)


class _Vocab:
    """The declarations a diagnostic's CONTEXT may cite: own decls and,
    for a pack, the base source's (spec 08 D0)."""

    def __init__(self, decls, layers):
        self.preds, self.spaces, self.types = {}, {}, {}
        for d in decls:
            if isinstance(d, (ast.PredDecl, ast.BwdDecl)):
                self.preds.setdefault(d.name, d)
            elif isinstance(d, ast.NamespaceDecl):
                self.spaces.setdefault(d.name, d)
            elif isinstance(d, ast.TypeDecl):
                self.types.setdefault(d.name, d)
        self.layers = layers

    def source(self, node, decl: str, doc: str, line_txt: str, atoms):
        """The precomputed slice of one declaration: loc + subject +
        context (preds in source order, first occurrence -> their
        namespaces -> their types as name + cardinality) + layers."""
        from metis.kernel.diagnostics import Loc, Source
        from .ast import decl_line
        subject = tuple(f"%% {ln}" for ln in doc.split("\n")) \
            if doc else ()
        subject += (line_txt,)
        pdecls = []
        for a in atoms:
            d = self.preds.get(a.pred)
            if d is not None and d not in pdecls:
                pdecls.append(d)
        ctx = [decl_line(d) for d in pdecls]
        seen_ns = []
        for d in pdecls:
            if isinstance(d, ast.PredDecl) and d.namespace not in seen_ns \
                    and d.namespace in self.spaces:
                seen_ns.append(d.namespace)
                ctx.append(decl_line(self.spaces[d.namespace]))
        seen_ty = []
        for d in pdecls:
            for ty in d.args:
                if ty not in seen_ty and ty in self.types:
                    seen_ty.append(ty)
                    ctx.append(f"type {ty}: "
                               f"{len(self.types[ty].consts)} constants")
        return self._mk(node, decl, subject, ctx)

    def decl_source(self, d, decl: str):
        """The slice of a vocabulary declaration (catalog-level and
        admission diagnostics): subject = the decl itself; context =
        what it references (a pred's namespace, its arg types)."""
        ctx = []
        if isinstance(d, ast.PredDecl) and d.namespace in self.spaces:
            ctx.append(ast.decl_line(self.spaces[d.namespace]))
        if isinstance(d, (ast.PredDecl, ast.BwdDecl)):
            seen = []
            for ty in d.args:
                if ty not in seen and ty in self.types:
                    seen.append(ty)
                    ctx.append(f"type {ty}: "
                               f"{len(self.types[ty].consts)} constants")
        return self._mk(d, decl, (ast.decl_line(d),), ctx)

    def _mk(self, node, decl, subject, ctx):
        from metis.kernel.diagnostics import Loc, Source
        lay = f"layers ({' '.join(self.layers)})." if self.layers else ""
        return Source(Loc(getattr(node, "file", ""),
                          getattr(node, "line", 0),
                          getattr(node, "col", 0), decl),
                      subject, tuple(ctx), lay)


def _singleton_fact_vars(atom, sigs) -> List[Tuple[str, str, str]]:
    """A var occurring ONCE in a fact is almost always a typo
    (`plus(n0,N,M).` meant N twice): it silently means 'every value'.
    A deliberate don't-care is named Any or Any<Name> (the Ceptre
    corpus's own spelling, talespin's `solves_problem Any thirst
    water`) and stays silent. Advisory — never a finding."""
    sig = sigs.get(atom.pred) or ()
    names = [t.value for t in atom.terms if t.kind == "var"]
    out = []
    for i, t in enumerate(atom.terms):
        v = t.value
        if t.kind != "var" or names.count(v) != 1 or v.startswith("Any"):
            continue
        ty = sig[i] if i < len(sig) else "value"
        out.append((f"fact {atom}: var {v} occurs once, so the fact "
                    f"holds for every {ty} — if intended, name it "
                    f"Any{v} (a leading Any marks a don't-care)", v, ty))
    return out


def resolve_includes(cat: ast.Catalog, base_dir) -> ast.Catalog:
    """Splice included catalogs' DECLS in place of each include (the
    literal-inclusion semantics; nested includes resolve recursively,
    cycles rejected). The includer keeps its own name/version/
    provenance; layers/stages are ADOPTED from an included file when
    the includer omits them — that is the RDDL-style lifting split: a
    static DOMAIN file (rules, preds, ports, stages) included by a
    generated INSTANCE file (types, facts)."""
    from pathlib import Path

    def resolve(c, base, seen):
        layers, stages = c.layers, c.stages
        decls = []
        for d in c.decls:
            if not isinstance(d, ast.IncludeDecl):
                decls.append(d)
                continue
            path = (Path(base) / d.path).resolve()
            if str(path) in seen:     # seen = THIS include path only
                from metis.kernel.diagnostics import Loc, Source, make
                text = f"include cycle at {d.path}"
                raise LangError([text], [make(
                    "include-cycle", text,
                    Source(Loc(d.file, d.line, d.col,
                               f"include/{d.path}"),
                           (ast.decl_line(d),), ()),
                    (("path", d.path),))])
            # extend `seen` along the recursion path only (metisc
            # resolve_in): a diamond — one file included by two
            # siblings — is not a cycle
            inc = parse_file(path)
            inc = resolve(inc, path.parent, seen | {str(path)})
            decls.extend(inc.decls)
            layers = layers or inc.layers
            stages = stages or inc.stages
        return ast.Catalog(c.name, c.version, c.provenance, layers,
                           stages, extends=c.extends,
                           decls=tuple(decls))

    return resolve(cat, base_dir, frozenset())


def compile_catalog(cat: ast.Catalog,
                    base: Optional[CatalogDoc] = None) -> CatalogDoc:
    """base: required when the AST declares `extends` — supplies pred
    signatures and the bwd set for inference/classification, inherits
    layers/stages when the pack omits them, and DEFERS typechecking to
    admission (a pack referencing base vocabulary cannot typecheck
    standalone)."""
    findings: List[str] = []
    warnings: List[str] = []
    diags: List = []
    if any(isinstance(d, ast.IncludeDecl) for d in cat.decls):
        raise LangError(["unresolved include — call resolve_includes "
                         "(or load_llp, which does)"])
    if cat.extends and base is None:
        from metis.kernel.diagnostics import Loc, Source, make
        text = (f"catalog extends '{cat.extends}' but no base catalog "
                f"was provided")
        raise LangError([text], [make(
            "extends-no-base", text,
            Source(Loc("", 0, 0, f"catalog/{cat.name}"),
                   (f"extends {cat.extends}.",), ()),
            (("base", cat.extends),))])
    prov = Provenance(*cat.provenance) if cat.provenance else \
        Provenance("hand", f"catalog {cat.name}", "unknown")

    types: Dict[str, List[str]] = {}
    namespaces = {}
    preds: Dict[str, str] = {}
    sigs: Dict[str, Tuple[str, ...]] = {}
    bwd: Dict[str, Tuple[str, ...]] = {}
    inherited_bwd = frozenset()
    layers, stages = cat.layers, cat.stages
    if base is not None:
        b = base.catalog
        # signatures for inference come from the base source decls
        if base.source is not None:
            for d in base.source.decls:
                if isinstance(d, (ast.PredDecl, ast.BwdDecl)):
                    sigs[d.name] = d.args
        inherited_bwd = b.bwd
        layers = layers or b.layers
        stages = stages or b.stages
    base_types = dict(base.catalog.sig.types) if base is not None else {}
    vocab = _Vocab(list(cat.decls) + (list(base.source.decls)
                                      if base is not None
                                      and base.source is not None
                                      else []), layers)
    from metis.kernel.diagnostics import make as _make

    def emitter(src, prefix=""):
        """emit(code, text, data): the legacy finding (with metispy's
        location prefix) + the structured diagnostic (location-free
        text), appended at the same point so the two stay in step."""
        def emit(code, text, data):
            d = _make(code, text, src, data)
            diags.append(d)
            if d.severity == "error":
                findings.append(prefix + text)
            else:
                warnings.append(text)
        return emit
    ports = Ports()
    entries: List[Entry] = []
    horn_entries: List[Entry] = []

    # decl-duplicate (a pre-pass, before sweep 1): one name, one
    # declaration per kind — a second pred/bwd/type/namespace/port of a
    # name is a finding at the SECOND occurrence (the two toolchains
    # used to keep different ones: last-wins here, first-wins in metisc)
    import os as _os
    first_seen: Dict = {}
    _dkind = {ast.PredDecl: "pred", ast.BwdDecl: "bwd",
              ast.TypeDecl: "type", ast.NamespaceDecl: "namespace",
              ast.PortDecl: "port"}
    for d in cat.decls:
        kind = _dkind.get(type(d))
        if kind is None:
            continue
        p0 = first_seen.setdefault((kind, d.name), d)
        if p0 is not d:
            where = (f"{_os.path.basename(p0.file) or '<input>'}:"
                     f"{p0.line}")
            emitter(vocab.decl_source(d, f"{kind}/{d.name}"))(
                "decl-duplicate",
                f"{kind} '{d.name}' declared twice (first at {where})",
                (("kind", kind), ("name", d.name), ("first", where)))
    for d in cat.decls:
        if isinstance(d, ast.TypeDecl):
            types[d.name] = list(d.consts)
        elif isinstance(d, ast.NamespaceDecl):
            namespaces[d.name] = ns(d.name, set(d.produce),
                                    set(d.consume)
                                    if d.consume is not None else None)
        elif isinstance(d, ast.PredDecl):
            preds[d.name] = d.namespace
            sigs[d.name] = d.args
        elif isinstance(d, ast.BwdDecl):
            bwd[d.name] = d.args
            sigs[d.name] = d.args
        elif isinstance(d, ast.PortDecl):
            if d.kind == "weight":
                ports.weights[d.name] = d.args
                if d.reads is not None:
                    ports.reads[d.name] = d.reads
            elif d.kind == "guard":
                ports.guards[d.name] = d.args
            elif d.kind == "input":
                ports.inputs += (d.name,)
            else:
                ports.outputs += (d.name,)

    # read-scope validation: formals are vars; scope atoms are declared
    # resource preds; guards are bwd; every scope-atom var is bound by a
    # formal or enumerated by a guard.
    known_preds = dict(preds)
    if base is not None:
        known_preds.update(base.catalog.preds)
    all_bwd = set(bwd) | set(inherited_bwd)
    port_decls = {d.name: d for d in cat.decls
                  if isinstance(d, ast.PortDecl)}
    for factor, pats in ports.reads.items():
        pd = port_decls[factor]
        emit = emitter(vocab.source(
            pd, f"weight/{factor}", "", ast.decl_line(pd),
            [a for pat in pats for a in (pat.atom,) + pat.guards]))
        formals = ports.weights[factor]
        bad = [a for a in formals if not a[0].isupper()]
        if bad:
            emit("reads-formal-not-var",
                 f"weight {factor}: reads needs var formals, got "
                 f"{','.join(bad)}",
                 (("port", factor), ("formals", ",".join(bad))))
        for pat in pats:
            for a in (pat.atom,) + pat.guards:
                if a.persist:
                    emit("reads-persist-marker",
                         f"weight {factor}: $ marker in reads {a}",
                         (("port", factor), ("atom", str(a))))
                sig = sigs.get(a.pred)
                if sig is not None and len(sig) != len(a.terms):
                    emit("arity-mismatch",
                         f"weight {factor}: {a.pred} arity "
                         f"{len(a.terms)} != declared {len(sig)}",
                         (("pred", a.pred), ("expected", str(len(sig))),
                          ("got", str(len(a.terms)))))
            if pat.atom.pred not in known_preds:
                emit("reads-not-resource",
                     f"weight {factor}: reads atom '{pat.atom.pred}' is "
                     f"not a declared resource predicate",
                     (("port", factor), ("pred", pat.atom.pred)))
            for g in pat.guards:
                if g.pred not in all_bwd:
                    emit("reads-guard-not-bwd",
                         f"weight {factor}: reads guard '{g.pred}' is "
                         f"not declared bwd",
                         (("port", factor), ("pred", g.pred)))
            bound = set(formals) | {t.value for g in pat.guards
                                    for t in g.terms if t.kind == "var"}
            for t in pat.atom.terms:
                if t.kind == "var" and t.value not in bound:
                    emit("reads-var-unbound",
                         f"weight {factor}: reads var {t.value} unbound "
                         f"(not a formal, not guard-enumerated)",
                         (("port", factor), ("var", t.value)))

    def horn_entry(payload, doc, src=None):
        horn_entries.append(Entry(
            name=f"horn/{payload.name}", layer=cat.layers[0]
            if cat.layers else "horn", kind="horn",
            comment=doc or "Horn clause.", provenance=prov,
            payload=payload, src=src))

    for d in cat.decls:
        if isinstance(d, ast.Fact):
            fsrc = vocab.source(d, f"fact/{d.atom.pred}", "",
                                ast.decl_line(d), [d.atom])
            emit = emitter(fsrc)
            if d.atom.pred not in bwd:
                emit("fact-not-bwd",
                     f"fact {d.atom}: predicate not declared bwd",
                     (("pred", d.atom.pred),))
                continue
            # a var in a fact is universally quantified over the type
            # its position declares (`plus(n0,N,N).` = Π N:nat) — the
            # Ceptre/Twelf reading of a bodiless clause; typed like a
            # rule var, so one var at two differently-typed positions
            # is a finding
            _infer_vars(f"fact {d.atom}", [d.atom], sigs, findings,
                        emit, "fact-var-conflict")
            _check_consts(f"fact {d.atom}", [d.atom], sigs,
                          {**base_types, **types}, findings, emit)
            for text, v, ty in _singleton_fact_vars(d.atom, sigs):
                emit("fact-var-singleton", text,
                     (("var", v), ("type", ty)))
            vs = tuple(sorted({t.value for t in d.atom.terms
                               if t.kind == "var"}))
            horn_entry(HornClause(str(d.atom), _pattern(d.atom),
                                  vars=vs), "", fsrc)
        elif isinstance(d, ast.HornRule):
            hsrc = vocab.source(d, f"horn/{d.head.pred}", "",
                                ast.decl_line(d), (d.head,) + d.body)
            _check_consts(f"horn {d.head}", (d.head,) + d.body, sigs,
                          {**base_types, **types}, findings,
                          emitter(hsrc))
            vs = tuple(sorted({t.value for a in (d.head,) + d.body
                               for t in a.terms if t.kind == "var"}))
            horn_entry(HornClause(str(d.head), _pattern(d.head),
                                  tuple(_pattern(b) for b in d.body),
                                  vars=vs), "", hsrc)
        elif isinstance(d, ast.Stage):
            for r in d.rules:
                rloc = f"line {r.line}: " if r.line else ""
                atoms = [a for a in r.body
                         if isinstance(a, ast.Atom)] + list(r.head)
                rsrc = vocab.source(r, f"{d.name}/{r.name}", r.doc,
                                    ast.rule_line(r),
                                    [a for a in atoms if a.pred != "one"])
                emit = emitter(rsrc, rloc)
                if not r.doc.strip():
                    emit("doc-missing",
                         f"{r.name}: missing %% doc (mandatory)",
                         (("rule", r.name),))
                vars_ = _infer_vars(r.name, atoms, sigs, findings, emit)
                _check_consts(r.name, atoms, sigs,
                              {**base_types, **types}, findings, emit)
                for a in atoms:
                    for t in a.terms:
                        if t.kind == "var" and t.value not in vars_:
                            emit("var-untypable",
                                 f"{r.name}: var {t.value} untypable",
                                 (("var", t.value),))
                # range restriction: a right-side variable no left-side atom
                # binds would be drawn over its whole type by the grounder
                # (policy-tuning step 1: a brain's rule OOM-killed at 6G)
                bound = {t.value for a in r.body if isinstance(a, ast.Atom)
                         for t in a.terms if t.kind == "var"}
                free = []
                for a in r.head:
                    for t in a.terms:
                        if (t.kind == "var" and t.value not in bound
                                and t.value not in free):
                            free.append(t.value)
                for v in free:
                    emit("rhs-var-unbound",
                         f"{r.name}: variable {v} on the right side is "
                         f"bound nowhere on the left",
                         (("var", v),))
                consume, persist, guards = [], [], []
                for a in r.body:
                    if isinstance(a, ast.Distinct):
                        continue
                    if a.pred in bwd or a.pred in inherited_bwd:
                        guards.append(_pattern(a))
                    elif a.persist:
                        persist.append(_pattern(a))
                    else:
                        consume.append(_pattern(a))
                weight = 1.0
                if r.weight is not None:
                    if r.weight.factor is None:
                        weight = r.weight.value
                    else:
                        wf = r.weight.factor
                        if wf not in ports.weights:
                            emit("weight-port-undeclared",
                                 f"{r.name}: weight port '{wf}' not "
                                 f"declared", (("port", wf),))
                        for t in r.weight.args:
                            if t.kind == "var" and t.value not in vars_:
                                emit("weight-arg-not-var",
                                     f"{r.name}: weight arg {t.value} is "
                                     f"not a rule var",
                                     (("port", wf), ("arg", t.value)))
                        if wf in ports.reads and len(r.weight.args) != \
                                len(ports.weights[wf]):
                            n = len(ports.weights[wf])
                            emit("arity-mismatch",
                                 f"{r.name}: weight {wf} takes {n} args, "
                                 f"got {len(r.weight.args)}",
                                 (("pred", wf), ("expected", str(n)),
                                  ("got", str(len(r.weight.args)))))
                        ports.weight_refs[r.name] = WeightRef(
                            r.weight.factor,
                            tuple(t.value for t in r.weight.args),
                            r.weight.complement)
                entries.append(Entry(
                    name=f"{r.layer}/{r.name}", layer=r.layer,
                    kind="schema", comment=r.doc, provenance=prov,
                    src=rsrc,
                    payload=ClauseSchema(
                        r.name, stage=d.name, vars=vars_,
                        consume=consume,
                        produce=[_pattern(a) for a in r.head
                                 if a.pred != "one"],   # LL unit: drops
                        persist=persist, guards=guards,
                        distinct=[(x.a, x.b) for x in r.body
                                  if isinstance(x, ast.Distinct)],
                        weight=weight)))
        elif isinstance(d, ast.LinkDecl):
            dloc = f"line {d.line}: " if d.line else ""
            atoms = list(d.pre_atoms) + list(d.post_atoms)
            lsrc = vocab.source(d, f"qui/{d.name}", d.doc,
                                ast.decl_line(d), atoms)
            emit = emitter(lsrc, dloc)
            if not d.doc.strip():
                emit("doc-missing", f"{d.name}: missing %% doc (mandatory)",
                     (("rule", d.name),))
            vars_ = _infer_vars(d.name, atoms, sigs, findings, emit)
            _check_consts(d.name, atoms, sigs, {**base_types, **types},
                          findings, emit)
            # bwd atoms in the pre are Horn GUARDS, exactly as on
            # clauses (the grounding machinery already supports them)
            # — what makes chain-end links liftable (lastidx(I));
            # $-atoms are persists (read, never decremented), the
            # run-token pattern of the road cycle.
            lconsume, lguards, lpersist = [], [], []
            for a in d.pre_atoms:
                if a.pred in bwd or a.pred in inherited_bwd:
                    lguards.append(_pattern(a))
                elif a.persist:
                    lpersist.append(_pattern(a))
                else:
                    lconsume.append(_pattern(a))
            for a in d.post_atoms:
                if a.pred in bwd or a.pred in inherited_bwd:
                    emit("link-post-bwd",
                         f"{d.name}: bwd '{a.pred}' in link post",
                         (("pred", a.pred),))
            entries.append(Entry(
                name=f"{d.layer}/{d.name}", layer=d.layer, kind="link",
                comment=d.doc, provenance=prov, src=lsrc,
                payload=LinkSchema(
                    d.name, pre=d.pre, post=d.post, vars=vars_,
                    consume=lconsume, guards=lguards, persist=lpersist,
                    produce=[_pattern(a) for a in d.post_atoms])))

    if findings:
        raise LangError(findings, diags)

    kcat = Catalog(name=cat.name, version=cat.version,
                   layers=layers, stages=stages,
                   sig=Signature(types), namespaces=namespaces,
                   preds=preds, bwd=frozenset(bwd),
                   entries=entries + horn_entries)
    for d in cat.decls:                 # decl slices (spec 08)
        key = {ast.PredDecl: "pred", ast.BwdDecl: "bwd",
               ast.NamespaceDecl: "namespace",
               ast.TypeDecl: "type"}.get(type(d))
        if key is not None:
            kcat.srcs.setdefault(f"{key}/{d.name}",
                                 vocab.decl_source(d, f"{key}/{d.name}"))
    if cat.extends is None:
        kdiags: List = []
        f = kcat.typecheck(kdiags)
        if f:
            raise CatalogError(f, kdiags)
    return CatalogDoc(kcat, ports, source=cat, warnings=warnings,
                      diagnostics=diags)


def diagnose(path, base=None) -> List:
    """Every structured diagnostic (spec 08) of compiling one file —
    the mirror of `metisc --diagnostics FILE [--base BASE]`: a failing
    compile's errors (LangError / CatalogError) or a clean compile's
    diagnostics; with a base (a path or a CatalogDoc) the file is
    compiled as a pack AND admitted, so admission diagnostics join."""
    if base is not None and not isinstance(base, CatalogDoc):
        base = load_llp(base)
    try:
        doc = load_llp(path, base=base)
    except (LangError, CatalogError) as e:
        return list(e.diagnostics)
    out = list(doc.diagnostics)
    if base is not None:
        try:
            base.catalog.admit(doc.catalog)
        except CatalogError as e:
            out += e.diagnostics
    return out


def load_llp(path, base: Optional[CatalogDoc] = None) -> CatalogDoc:
    from pathlib import Path
    ast_cat = resolve_includes(parse_file(path), Path(path).parent)
    return compile_catalog(ast_cat, base=base)


# ---- read-scope grounding ----------------------------------------------

def ground_reads(ports: Ports, factor: str, args: Tuple[str, ...],
                 horn) -> List[str]:
    """The GROUND read-scope of one port instance: substitute the
    port's formal args, enumerate each pattern's guards against the
    Horn world, render ground atoms in the multiset key format. This
    is what makes an opaque binding symbolically scoped — the
    factored evaluator's admission precondition."""
    pats = ports.reads.get(factor)
    if pats is None:
        raise LangError([f"weight port '{factor}' declares no read "
                         f"scope"])
    subst = dict(zip(ports.weights[factor], args))

    def resolve(t):
        if t.kind != "var":
            return t.value
        return subst.get(t.value, f"?{t.value}")

    def render(pred, terms):
        return f"{pred}({','.join(terms)})" if terms else pred

    out: List[str] = []
    for pat in pats:
        head = tuple(resolve(t) for t in pat.atom.terms)
        if not pat.guards:
            out.append(render(pat.atom.pred, head))
            continue
        goals = [(g.pred, tuple(resolve(t) for t in g.terms))
                 for g in pat.guards]
        for sol in horn.solutions(goals):
            out.append(render(pat.atom.pred,
                              tuple(sol.get(a[1:], a)
                                    if a.startswith("?") else a
                                    for a in head)))
    return list(dict.fromkeys(out))


# ---- weight ports at projection time -----------------------------------

def _tag_subst(clause_name: str) -> Dict[str, str]:
    if "[" not in clause_name:
        return {}
    tag = clause_name.split("[", 1)[1].rstrip("]")
    return dict(kv.split("=") for kv in tag.split(",")) if tag else {}


def factor_hooks(program, ports: Ports, registry: Dict[str, object],
                 horn):
    """Per-event weight hooks for the factored evaluator
    (kernel.factorize.FactoredProgram): binding closure, GROUND
    read-scope, complement flag. Fail-fast like weight_policy, plus
    the factorization admission rule: every referenced port must
    declare `reads`."""
    missing = {r.factor for r in ports.weight_refs.values()
               if r.factor not in registry}
    if missing:
        raise LangError([f"unbound weight port '{m}'"
                         for m in sorted(missing)])
    opaque = {r.factor for r in ports.weight_refs.values()
              if r.factor not in ports.reads}
    if opaque:
        raise LangError([f"weight port '{m}' declares no read scope — "
                         f"required for factoring"
                         for m in sorted(opaque)])
    table = {}
    for kind, _stage, ev in program.events():
        if kind != "clause":
            continue
        base = ev.name.split("[", 1)[0]
        ref = ports.weight_refs.get(base)
        if ref is None:
            continue
        subst = _tag_subst(ev.name)
        args = tuple(subst.get(a, a) for a in ref.args)
        fn = registry[ref.factor]
        scope = ground_reads(ports, ref.factor, args, horn)
        table[ev.name] = (
            (lambda f, a: lambda counts: f(a, counts))(fn, args),
            scope, ref.complement, (ref.factor, args))
    return table.get


def merge_ports(a: Ports, b: Ports) -> Ports:
    """A pack's ports over its base's (the merged world's port
    surface — weight refs and read scopes included)."""
    return Ports(weights={**a.weights, **b.weights},
                 guards={**a.guards, **b.guards},
                 inputs=a.inputs + tuple(p for p in b.inputs
                                         if p not in a.inputs),
                 outputs=a.outputs + tuple(p for p in b.outputs
                                           if p not in a.outputs),
                 weight_refs={**a.weight_refs, **b.weight_refs},
                 reads={**a.reads, **b.reads})


class PortPricer:
    """Name-keyed port pricing — the ONE resolution of `@w f(args)`
    every interactive tier shares (the REPL sampler, the measure's
    exact and sampled tiers, gate scenarios). Keyed by ground event
    NAME, not by an event index, so masked/pruned/quotient programs
    price identically.

    G5: with `scoped`, a binding sees its counts through ScopedCounts
    over the GROUND declared read-scope — an out-of-scope read raises
    ScopeViolation here exactly as factor_hooks makes it raise in the
    factored tier. A port with no `reads` declaration is opaque: it
    sees the whole multiset (and is listed in `.opaque`; it cannot be
    factored). The interactive tier passes the scope atoms' true
    counts (the factored tier synthesizes presence bits): a binding
    that reads presence (`counts.get(a, 0) > 0`) agrees in both."""

    def __init__(self, ports: Ports, registry: Dict[str, object],
                 horn=None, scoped: bool = True):
        missing = {ref.factor for ref in ports.weight_refs.values()
                   if ref.factor not in registry}
        if missing:
            raise LangError([f"unbound weight port '{m}'"
                             for m in sorted(missing)])
        self.ports, self.registry, self.horn = ports, registry, horn
        self.scoped = scoped
        self.opaque = sorted({r.factor for r in ports.weight_refs.values()
                              if r.factor not in ports.reads})
        self._sites: Dict[str, Optional[tuple]] = {}

    def site(self, name: str) -> Optional[tuple]:
        """(factor, args, complement, scope|None) for a ground clause
        name, or None when its rule references no port."""
        if name in self._sites:
            return self._sites[name]
        ref = self.ports.weight_refs.get(name.split("[", 1)[0])
        out = None
        if ref is not None:
            subst = _tag_subst(name)
            args = tuple(subst.get(a, a) for a in ref.args)
            scope = None
            if self.scoped and ref.factor in self.ports.reads:
                guarded = any(p.guards for p in
                              self.ports.reads[ref.factor])
                if self.horn is not None or not guarded:
                    scope = frozenset(ground_reads(
                        self.ports, ref.factor, args, self.horn))
            out = (ref.factor, args, ref.complement, scope)
        self._sites[name] = out
        return out

    def raw(self, name: str, counts) -> Optional[float]:
        """The port's value f (before the complement) for an event in
        state `counts`, or None for an unpriced event."""
        s = self.site(name)
        if s is None:
            return None
        factor, args, _compl, scope = s
        if scope is not None:
            from metis.kernel.factorize import ScopedCounts
            counts = ScopedCounts(scope, {a: counts[a] for a in scope
                                          if counts.get(a, 0) > 0})
        return float(self.registry[factor](args, counts))

    def factor(self, name: str, counts) -> Optional[float]:
        """The CHOICE multiplier (complement applied), or None."""
        f = self.raw(name, counts)
        if f is None:
            return None
        return 1.0 - f if self.site(name)[2] else f

    def policy(self, program, canonical: bool = False):
        """An index-keyed CHOICE policy for one concrete program. The
        index space differs by consumer: the filter (ForwardProjection,
        sample_eval) indexes `program.events()`; the reference sampler
        (refsample.run_trace) indexes `canonical_events(program)` — pass
        canonical=True for it."""
        if canonical:
            from metis.kernel.refsample import canonical_events
            names = [ev.name for _, _, ev in canonical_events(program)]
        else:
            names = [ev.name for _, _, ev in program.events()]

        def pol(stage, counts, ws):
            out = []
            for i, w in ws:
                f = self.factor(names[i], counts)
                if f is None:
                    out.append((i, w))
                elif f > 0:
                    out.append((i, w * f))
            return out
        return pol


def weight_policy(program, ports: Ports, registry: Dict[str, object]):
    """CHOICE policy resolving @w factor references against a registry
    {factor: callable(args_tuple, counts) -> float}. Unbound factors
    fail HERE, at build time — the BT.cpp fail-fast rule."""
    missing = {ref.factor for ref in ports.weight_refs.values()
               if ref.factor not in registry}
    if missing:
        raise LangError([f"unbound weight port '{m}'"
                         for m in sorted(missing)])
    plan = {}      # event index -> (callable, args, complement, subst)
    for i, (kind, _stage, ev) in enumerate(program.events()):
        if kind != "clause":
            continue
        base = ev.name.split("[", 1)[0]
        ref = ports.weight_refs.get(base)
        if ref is None:
            continue
        subst = _tag_subst(ev.name)
        args = tuple(subst.get(a, a) for a in ref.args)
        plan[i] = (registry[ref.factor], args, ref.complement)

    def policy(stage, counts, ws):
        out = []
        for i, w in ws:
            hook = plan.get(i)
            if hook is None:
                out.append((i, w))
                continue
            fn, args, compl = hook
            f = float(fn(args, counts))
            f = 1.0 - f if compl else f
            if f > 0:
                out.append((i, w * f))
        return out

    return policy
