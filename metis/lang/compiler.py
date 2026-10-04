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


def _infer_vars(name, atoms, sigs, findings, conflicts=None) \
        -> Dict[str, str]:
    vars_: Dict[str, str] = {}
    for a in atoms:
        sig = sigs.get(a.pred)
        if sig is None:
            continue          # kernel typecheck reports undeclared preds
        if len(sig) != len(a.terms):
            findings.append(f"{name}: {a.pred} arity "
                            f"{len(a.terms)} != declared {len(sig)}")
            continue
        for t, ty in zip(a.terms, sig):
            if t.kind != "var":
                continue
            if vars_.setdefault(t.value, ty) != ty:
                findings.append(
                    f"{name}: var {t.value} typed both "
                    f"{vars_[t.value]} and {ty}")
                if conflicts is not None:
                    conflicts.append((findings[-1], t.value,
                                      vars_[t.value], ty))
    return vars_


def _check_consts(where, atoms, sigs, types, findings) -> None:
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
                findings.append(f"{where}: constant '{t.value}' is not "
                                f"a {ty} ({a.pred} argument {k})")


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
        lay = f"layers ({' '.join(self.layers)})." if self.layers else ""
        return Source(Loc(getattr(node, "file", ""),
                          getattr(node, "line", 0),
                          getattr(node, "col", 0), decl),
                      subject, tuple(ctx), lay)


def _singleton_fact_vars(atom, sigs) -> List[str]:
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
        out.append(f"fact {atom}: var {v} occurs once, so the fact holds "
                   f"for every {ty} — if intended, name it Any{v} (a "
                   f"leading Any marks a don't-care)")
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
            if str(path) in seen:
                raise LangError([f"include cycle at {d.path}"])
            seen = seen | {str(path)}
            inc = parse_file(path)
            inc = resolve(inc, path.parent, seen)
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
        raise LangError([f"catalog extends '{cat.extends}' but no base "
                         f"catalog was provided"])
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
    ports = Ports()
    entries: List[Entry] = []
    horn_entries: List[Entry] = []

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
    for factor, pats in ports.reads.items():
        formals = ports.weights[factor]
        bad = [a for a in formals if not a[0].isupper()]
        if bad:
            findings.append(f"weight {factor}: reads needs var formals,"
                            f" got {','.join(bad)}")
        for pat in pats:
            for a in (pat.atom,) + pat.guards:
                if a.persist:
                    findings.append(f"weight {factor}: $ marker in "
                                    f"reads {a}")
                sig = sigs.get(a.pred)
                if sig is not None and len(sig) != len(a.terms):
                    findings.append(f"weight {factor}: {a.pred} arity "
                                    f"{len(a.terms)} != declared "
                                    f"{len(sig)}")
            if pat.atom.pred not in known_preds:
                findings.append(f"weight {factor}: reads atom "
                                f"'{pat.atom.pred}' is not a declared "
                                f"resource predicate")
            for g in pat.guards:
                if g.pred not in all_bwd:
                    findings.append(f"weight {factor}: reads guard "
                                    f"'{g.pred}' is not declared bwd")
            bound = set(formals) | {t.value for g in pat.guards
                                    for t in g.terms if t.kind == "var"}
            for t in pat.atom.terms:
                if t.kind == "var" and t.value not in bound:
                    findings.append(f"weight {factor}: reads var "
                                    f"{t.value} unbound (not a formal, "
                                    f"not guard-enumerated)")

    def horn_entry(payload, doc):
        horn_entries.append(Entry(
            name=f"horn/{payload.name}", layer=cat.layers[0]
            if cat.layers else "horn", kind="horn",
            comment=doc or "Horn clause.", provenance=prov,
            payload=payload))

    for d in cat.decls:
        if isinstance(d, ast.Fact):
            if d.atom.pred not in bwd:
                findings.append(f"fact {d.atom}: predicate not "
                                f"declared bwd")
                continue
            # a var in a fact is universally quantified over the type
            # its position declares (`plus(n0,N,N).` = Π N:nat) — the
            # Ceptre/Twelf reading of a bodiless clause; typed like a
            # rule var, so one var at two differently-typed positions
            # is a finding
            conflicts: List = []
            _infer_vars(f"fact {d.atom}", [d.atom], sigs, findings,
                        conflicts)
            if conflicts:
                from metis.kernel.diagnostics import make
                src = vocab.source(d, f"fact/{d.atom.pred}", "",
                                   ast.decl_line(d), [d.atom])
                for text, v, t1, t2 in conflicts:
                    diags.append(make("fact-var-conflict", text, src,
                                      (("var", v), ("type1", t1),
                                       ("type2", t2))))
            _check_consts(f"fact {d.atom}", [d.atom], sigs,
                          {**base_types, **types}, findings)
            warnings.extend(_singleton_fact_vars(d.atom, sigs))
            vs = tuple(sorted({t.value for t in d.atom.terms
                               if t.kind == "var"}))
            horn_entry(HornClause(str(d.atom), _pattern(d.atom),
                                  vars=vs), "")
        elif isinstance(d, ast.HornRule):
            _check_consts(f"horn {d.head}", (d.head,) + d.body, sigs,
                          {**base_types, **types}, findings)
            vs = tuple(sorted({t.value for a in (d.head,) + d.body
                               for t in a.terms if t.kind == "var"}))
            horn_entry(HornClause(str(d.head), _pattern(d.head),
                                  tuple(_pattern(b) for b in d.body),
                                  vars=vs), "")
        elif isinstance(d, ast.Stage):
            for r in d.rules:
                rloc = f"line {r.line}: " if r.line else ""
                atoms = [a for a in r.body
                         if isinstance(a, ast.Atom)] + list(r.head)
                rsrc = vocab.source(r, f"{d.name}/{r.name}", r.doc,
                                    ast.rule_line(r),
                                    [a for a in atoms if a.pred != "one"])
                if not r.doc.strip():
                    from metis.kernel.diagnostics import make
                    text = f"{r.name}: missing %% doc (mandatory)"
                    findings.append(f"{rloc}{text}")
                    diags.append(make("doc-missing", text, rsrc,
                                      (("rule", r.name),)))
                vars_ = _infer_vars(f"{rloc}{r.name}", atoms, sigs,
                                    findings)
                _check_consts(f"{rloc}{r.name}", atoms, sigs,
                              {**base_types, **types}, findings)
                for a in atoms:
                    for t in a.terms:
                        if t.kind == "var" and t.value not in vars_:
                            findings.append(f"{rloc}{r.name}: var "
                                            f"{t.value} untypable")
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
                        if r.weight.factor not in ports.weights:
                            findings.append(
                                f"{rloc}{r.name}: weight port "
                                f"'{r.weight.factor}' not declared")
                        for t in r.weight.args:
                            if t.kind == "var" and t.value not in vars_:
                                findings.append(
                                    f"{rloc}{r.name}: weight arg "
                                    f"{t.value} is not a rule var")
                        if r.weight.factor in ports.reads and \
                                len(r.weight.args) != \
                                len(ports.weights[r.weight.factor]):
                            findings.append(
                                f"{rloc}{r.name}: weight "
                                f"{r.weight.factor} takes "
                                f"{len(ports.weights[r.weight.factor])}"
                                f" args, got {len(r.weight.args)}")
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
            if not d.doc.strip():
                from metis.kernel.diagnostics import make
                text = f"{d.name}: missing %% doc (mandatory)"
                findings.append(f"{dloc}{text}")
                diags.append(make("doc-missing", text, lsrc,
                                  (("rule", d.name),)))
            vars_ = _infer_vars(f"{dloc}{d.name}", atoms, sigs,
                                findings)
            _check_consts(f"{dloc}{d.name}", atoms, sigs,
                          {**base_types, **types}, findings)
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
                    findings.append(f"{dloc}{d.name}: bwd '{a.pred}' "
                                    f"in link post")
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
    if cat.extends is None:
        kdiags: List = []
        f = kcat.typecheck(kdiags)
        if f:
            raise CatalogError(f, kdiags)
    return CatalogDoc(kcat, ports, source=cat, warnings=warnings,
                      diagnostics=diags)


def diagnose(path, base: Optional[CatalogDoc] = None) -> List:
    """Every structured diagnostic (spec 08) of compiling one file —
    the mirror of `metisc --diagnostics`: a failing compile's errors
    (LangError / CatalogError) or a clean compile's diagnostics."""
    try:
        return list(load_llp(path, base=base).diagnostics)
    except (LangError, CatalogError) as e:
        return list(e.diagnostics)


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
