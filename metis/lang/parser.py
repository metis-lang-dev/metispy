"""Recursive-descent parser for rulescript v2. Dependency-free.

Line comments: `% ...` dropped; `%% ...` are DOC lines, attached to
the next rule or link (mandatory there — the compiler enforces it) or
pred (optional: the atom's words, RAG §11.1).
`# ...` lines are interpreter directives (embedded case lines,
metis/lang/directives.py) — a comment class here: the compiler never
sees them. Idents may contain hyphens (object-behavior); the arrow
`-o` must be whitespace-separated."""

import re
from typing import List, Optional, Tuple

from . import ast

_TOKEN = re.compile(r"""
    (?P<doc>%%[^\n]*) |
    (?P<comment>%[^%\n][^\n]*|%$) |
    (?P<directive>\#[^\n]*) |
    (?P<string>"[^"\n]*") |
    (?P<arrow>-o(?![A-Za-z0-9_-])) |
    (?P<horn>:-) |
    (?P<ne><>) |
    (?P<weight>@w) |
    (?P<range>\.\.) |
    (?P<num>\d+\.\d+|\.\d+|\d+) |
    (?P<ident>[A-Za-z][A-Za-z0-9_-]*) |
    (?P<punct>[{}().,*:\[\]~$|]) |
    (?P<ws>[ \t\n]+) |
    (?P<bad>.)
""", re.VERBOSE | re.MULTILINE)


class ParseError(ValueError):
    pass


_RANGE_END = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*?)?(\d+)$")


def _expand_range(a: str, b: str):
    """`c0 .. c4` -> c0 c1 c2 c3 c4 (shared prefix, inclusive int
    suffixes; pure numbers allowed). Sugar only: the AST carries the
    expansion, so round-trip and hashing see plain constants."""
    ma, mb = _RANGE_END.match(a), _RANGE_END.match(b)
    if not ma or not mb or ma.group(1) != mb.group(1) or \
            int(ma.group(2)) > int(mb.group(2)):
        raise ParseError(f"bad range {a}..{b}")
    pre = ma.group(1) or ""
    return [f"{pre}{i}" for i in range(int(ma.group(2)),
                                      int(mb.group(2)) + 1)]


def _line_col(text: str, offset: int) -> Tuple[int, int]:
    """1-based line:column of a character offset."""
    line = text.count("\n", 0, offset) + 1
    bol = text.rfind("\n", 0, offset) + 1
    return line, offset - bol + 1


def _tokens(text: str):
    """-> (tokens, positions): parallel lists of (kind, val) and the
    1-based (line, col) each token starts at."""
    out, pos = [], []
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind in ("comment", "ws", "directive"):
            continue
        if kind == "bad":
            ln, col = _line_col(text, m.start())
            raise ParseError(f"{ln}:{col}: bad character "
                             f"{m.group()!r}")
        val = m.group()
        if kind == "doc":
            val = val.strip()[2:].strip()
        if kind == "string":
            val = val[1:-1]
        out.append((kind, val))
        pos.append(_line_col(text, m.start()))
    return out, pos


class _P:
    def __init__(self, toks, pos):
        self.toks = toks
        self.pos = pos
        self.i = 0
        self.docs: List[str] = []

    def where(self, k: Optional[int] = None) -> str:
        """line:col of token index k (default: the token just
        consumed), for error messages."""
        j = self.i - 1 if k is None else k
        if not self.pos:
            return "1:1"
        j = max(0, min(j, len(self.pos) - 1))
        ln, col = self.pos[j]
        return f"{ln}:{col}"

    def err(self, msg: str):
        raise ParseError(f"{self.where()}: {msg}")

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("eof", "")

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def expect(self, kind, val=None):
        k, v = self.next()
        if k != kind or (val is not None and v != val):
            self.err(f"expected {val or kind}, got {v!r}")
        return v

    def punct(self, ch):
        self.expect("punct", ch)

    def take_docs(self) -> str:
        d, self.docs = self.docs, []
        return "\n".join(d)

    def drain_docs(self):
        while self.peek()[0] == "doc":
            self.docs.append(self.next()[1])

    # -- terms / atoms ---------------------------------------------------

    def term(self) -> ast.Term:
        k, v = self.next()
        if k == "num":
            return ast.Term("const", v)
        if k != "ident":
            self.err(f"expected term, got {v!r}")
        return ast.Term("var" if v[0].isupper() else "const", v)

    def atom(self) -> ast.Atom:
        persist = False
        if self.peek() == ("punct", "$"):
            self.next()
            persist = True
        name = self.expect("ident")
        terms: List[ast.Term] = []
        if self.peek() == ("punct", "("):
            self.next()
            while True:
                terms.append(self.term())
                k, v = self.next()
                if (k, v) == ("punct", ")"):
                    break
                if (k, v) != ("punct", ","):
                    self.err(f"expected , or ) got {v!r}")
        return ast.Atom(name, tuple(terms), persist=persist)

    def body_elem(self):
        if self.peek()[0] == "ident" and self.peek()[1][0].isupper() \
                and self.peek(1) == ("ne", "<>"):
            a = self.expect("ident")
            self.next()
            b = self.expect("ident")
            return ast.Distinct(a, b)
        return self.atom()

    def atoms_star(self, elem_fn):
        out = [elem_fn()]
        while self.peek() == ("punct", "*"):
            self.next()
            out.append(elem_fn())
        return tuple(out)

    # -- declarations ----------------------------------------------------

    def name_list_paren(self) -> Tuple[str, ...]:
        self.punct("(")
        out = []
        while self.peek() != ("punct", ")"):
            out.append(self.expect("ident"))
        self.next()
        return tuple(out)

    def sig_args(self) -> Tuple[str, ...]:
        if self.peek() != ("punct", "("):
            return ()
        self.next()
        out = []
        while True:
            out.append(self.expect("ident"))
            k, v = self.next()
            if (k, v) == ("punct", ")"):
                return tuple(out)
            if (k, v) != ("punct", ","):
                self.err(f"expected , or ) got {v!r}")

    def weight(self) -> ast.Weight:
        k, v = self.peek()
        if k == "num":
            self.next()
            return ast.Weight(value=float(v))
        complement = False
        if (k, v) == ("punct", "~"):
            self.next()
            complement = True
        name = self.expect("ident")
        args: Tuple[ast.Term, ...] = ()
        if self.peek() == ("punct", "("):
            self.next()
            lst = []
            while self.peek() != ("punct", ")"):
                lst.append(self.term())
                if self.peek() == ("punct", ","):
                    self.next()
            self.next()
            args = tuple(lst)
        return ast.Weight(factor=name, args=args,
                          complement=complement)

    def read_pattern(self) -> ast.ReadPattern:
        a = self.atom()
        guards: Tuple[ast.Atom, ...] = ()
        if self.peek() == ("ident", "if"):
            self.next()
            guards = self.atoms_star(self.atom)
        return ast.ReadPattern(a, guards)

    def reads_block(self) -> Tuple[ast.ReadPattern, ...]:
        self.punct("{")
        out = []
        while self.peek() != ("punct", "}"):
            out.append(self.read_pattern())
            if self.peek() == ("punct", ","):
                self.next()
        self.next()
        return tuple(out)

    def head_atoms(self) -> Tuple[ast.Atom, ...]:
        """A head position: '()' is the LL unit (alias of 'one'),
        else a '*'-list of atoms."""
        if self.peek() == ("punct", "(") and \
                self.peek(1) == ("punct", ")"):
            self.next()
            self.next()
            return (ast.Atom("one"),)
        return self.atoms_star(self.atom)

    def _here(self) -> Tuple[int, int]:
        """(line, col) of the current token; (0, 0) at end of input."""
        return self.pos[self.i] if self.i < len(self.pos) else (0, 0)

    def rule(self) -> List[ast.Rule]:
        """One or more clauses: the alternative-head sugar
        '-o ( head @w w | head @w w ... )' expands here (spec 07 P1),
        exactly as in metisc — one clause per branch, named
        <name>-1 .. <name>-k, sharing the body (one CHOICE site)."""
        doc = self.take_docs()
        line, col = self._here()
        name = self.expect("ident")
        self.punct("[")
        layer = self.expect("ident")
        self.punct("]")
        self.punct(":")
        body = self.atoms_star(self.body_elem)
        self.expect("arrow")
        if self.peek() == ("punct", "(") and \
                self.peek(1) != ("punct", ")"):
            self.next()
            branches = []
            while True:
                h = self.head_atoms()
                if self.peek()[0] != "weight":
                    self.err(f"alternative head: every branch needs "
                             f"'@w' (rule {name})")
                self.next()
                branches.append((h, self.weight()))
                k, v = self.next()
                if (k, v) == ("punct", ")"):
                    break
                if (k, v) != ("punct", "|"):
                    self.err(f"expected '|' or ')' in alternative "
                             f"head (rule {name})")
            if len(branches) < 2:
                self.err(f"alternative head needs >= 2 branches "
                         f"(rule {name})")
            self.punct(".")
            return [ast.Rule(f"{name}-{k + 1}", layer, doc, body,
                             h, w, line=line, col=col)
                    for k, (h, w) in enumerate(branches)]
        head = self.head_atoms()
        w = None
        if self.peek()[0] == "weight":
            self.next()
            w = self.weight()
        self.punct(".")
        return [ast.Rule(name, layer, doc, body, head, w, line=line,
                         col=col)]

    def stage(self) -> ast.Stage:
        name = self.expect("ident")
        self.punct("{")
        rules: List[ast.Rule] = []
        while True:
            self.drain_docs()
            if self.peek() == ("punct", "}"):
                self.next()
                break
            rules.extend(self.rule())
        seen = set()
        for r in rules:
            if r.name in seen:
                self.err(f"duplicate rule name {r.name} in stage "
                         f"{name} (note: alternative heads reserve "
                         f"<name>-<k>)")
            seen.add(r.name)
        return ast.Stage(name, tuple(rules))

    def link(self) -> ast.LinkDecl:
        doc = self.take_docs()
        line, col = self._here()
        name = self.expect("ident")
        self.punct("[")
        layer = self.expect("ident")
        self.punct("]")
        self.punct(":")
        pre = self.expect("ident")
        pre_atoms: Tuple[ast.Atom, ...] = ()
        if self.peek() == ("punct", "*"):
            self.next()
            pre_atoms = self.atoms_star(self.atom)
        self.expect("arrow")
        post = self.expect("ident")
        post_atoms: Tuple[ast.Atom, ...] = ()
        if self.peek() == ("punct", "*"):
            self.next()
            post_atoms = self.atoms_star(self.atom)
        self.punct(".")
        return ast.LinkDecl(name, layer, doc, pre, pre_atoms, post,
                            post_atoms, line=line, col=col)


def parse(text: str, file: str = "") -> ast.Catalog:
    cat = _parse(text)
    return _stamp_file(cat, file) if file else cat


def _stamp_file(cat: ast.Catalog, file: str) -> ast.Catalog:
    """Name the source in every positioned decl (diagnostics only)."""
    from dataclasses import replace
    out = []
    for d in cat.decls:
        if isinstance(d, ast.Stage):
            d = replace(d, rules=tuple(replace(r, file=file)
                                       for r in d.rules))
        elif hasattr(d, "file"):
            d = replace(d, file=file)
        out.append(d)
    return replace(cat, decls=tuple(out))


def _parse(text: str) -> ast.Catalog:
    p = _P(*_tokens(text))
    p.drain_docs()
    p.expect("ident", "catalog")
    name = p.expect("ident")
    version = int(p.expect("num"))
    p.punct(".")
    provenance = None
    extends = None
    layers: Tuple[str, ...] = ()
    stages: Tuple[str, ...] = ()
    decls: List[object] = []
    while p.peek()[0] != "eof":
        p.drain_docs()
        k, v = p.peek()
        if k == "eof":
            break
        at = p._here()  # loc = the DECLARED NAME's token (re-taken below
                        # for keyword decls: type/pred/bwd point at the name)
        if k != "ident":
            p.err(f"expected declaration, got {v!r}")
        if v == "include":
            p.next()
            at = p._here()
            path = p.expect("string")
            p.punct(".")
            decls.append(ast.IncludeDecl(path, *at))
        elif v == "extends":
            p.next()
            extends = p.expect("ident")
            p.punct(".")
        elif v == "provenance":
            p.next()
            src = p.expect("ident")
            ref = p.expect("string")
            date = p.expect("string")
            p.punct(".")
            provenance = (src, ref, date)
        elif v == "layers":
            p.next()
            layers = p.name_list_paren()
            p.punct(".")
        elif v == "stages" and p.peek(1) == ("punct", "("):
            p.next()
            stages = p.name_list_paren()
            p.punct(".")
        elif v == "types" and p.peek(1) == ("punct", "("):
            # abstract types (spec 12): the loc is the first declared
            # name's token (A1), the '(' when the list is empty
            p.next()
            at = p.pos[p.i + 1] if p.peek(1)[0] == "ident" else p._here()
            names = p.name_list_paren()
            p.punct(".")
            decls.append(ast.TypesDecl(names, *at))
        elif v == "type":
            p.next()
            at = p._here()
            tname = p.expect("ident")
            p.punct("{")
            consts = []
            while p.peek() != ("punct", "}"):
                k2, v2 = p.next()
                if k2 not in ("ident", "num"):
                    p.err(f"bad type constant {v2!r}: a type is its finite set "
                          f"of constants, `type {tname} {{a b c}}` or a range "
                          f"`{{k0..k9}}`; a constructor with arguments has no "
                          f"Metis form (Ceptre's `succ nat : nat` becomes a "
                          f"range and a bwd table over it)")
                if p.peek()[0] == "range":
                    p.next()
                    k3, v3 = p.next()
                    if k3 not in ("ident", "num"):
                        p.err(f"bad range end {v3!r}: a range is `lo..hi`, "
                              f"one alpha prefix with integer suffixes, "
                              f"`{{k0..k9}}`")
                    consts.extend(_expand_range(v2, v3))
                else:
                    consts.append(v2)
            p.next()
            p.punct(".")
            decls.append(ast.TypeDecl(tname, tuple(consts), *at))
        elif v == "namespace":
            p.next()
            at = p._here()
            nname = p.expect("ident")
            p.expect("ident", "produce")
            produce = p.name_list_paren()
            consume = None
            if p.peek() == ("ident", "consume"):
                p.next()
                consume = p.name_list_paren()
            p.punct(".")
            decls.append(ast.NamespaceDecl(nname, produce, consume, *at))
        elif v == "pred":
            p.next()
            at = p._here()
            pname = p.expect("ident")
            args = p.sig_args()
            p.punct(":")
            space = p.expect("ident")
            p.punct(".")
            decls.append(ast.PredDecl(pname, args, space, p.take_docs(),
                                      *at))
        elif v == "bwd":
            p.next()
            at = p._here()
            bname = p.expect("ident")
            args = p.sig_args()
            p.punct(".")
            decls.append(ast.BwdDecl(bname, args, *at))
        elif v in ("weight", "guard", "input", "output"):
            p.next()
            at = p._here()
            oname = p.expect("ident")
            args = p.sig_args()
            reads = None
            if p.peek() == ("ident", "reads"):
                if v != "weight":
                    p.err(f"{v} port {oname}: reads is for "
                          f"weight ports only")
                p.next()
                reads = p.reads_block()
            p.punct(".")
            decls.append(ast.PortDecl(v, oname, args, reads, *at))
        elif v == "stage":
            p.next()
            decls.append(p.stage())
        elif v == "qui":
            p.next()
            decls.append(p.link())
        else:
            # a bare atom: Horn fact or Horn rule
            head = p.atom()
            if p.peek()[0] == "horn":
                p.next()
                body = [p.atom()]
                while p.peek() == ("punct", ","):
                    p.next()
                    body.append(p.atom())
                p.punct(".")
                decls.append(ast.HornRule(head, tuple(body), *at))
            else:
                p.punct(".")
                decls.append(ast.Fact(head, *at))
    return ast.Catalog(name, version, provenance, layers, stages,
                       extends=extends, decls=tuple(decls))


def parse_file(path) -> ast.Catalog:
    with open(path) as f:
        return parse(f.read(), file=str(path))
