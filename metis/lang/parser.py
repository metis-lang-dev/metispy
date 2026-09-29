"""Recursive-descent parser for rulescript v2. Dependency-free.

Line comments: `% ...` dropped; `%% ...` are DOC lines, attached to
the next rule or link (mandatory there — the compiler enforces it).
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
    (?P<punct>[{}().,*:\[\]~$]) |
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


def _tokens(text: str) -> List[Tuple[str, str]]:
    out = []
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind in ("comment", "ws", "directive"):
            continue
        if kind == "bad":
            raise ParseError(f"bad character {m.group()!r} at "
                             f"offset {m.start()}")
        val = m.group()
        if kind == "doc":
            val = val.strip()[2:].strip()
        if kind == "string":
            val = val[1:-1]
        out.append((kind, val))
    return out


class _P:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0
        self.docs: List[str] = []

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
            raise ParseError(f"expected {val or kind}, got {v!r} "
                             f"(token {self.i - 1})")
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
            raise ParseError(f"expected term, got {v!r}")
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
                    raise ParseError(f"expected , or ) got {v!r}")
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
                raise ParseError(f"expected , or ) got {v!r}")

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

    def rule(self) -> ast.Rule:
        doc = self.take_docs()
        name = self.expect("ident")
        self.punct("[")
        layer = self.expect("ident")
        self.punct("]")
        self.punct(":")
        body = self.atoms_star(self.body_elem)
        self.expect("arrow")
        head = self.atoms_star(self.atom)
        w = None
        if self.peek()[0] == "weight":
            self.next()
            w = self.weight()
        self.punct(".")
        return ast.Rule(name, layer, doc, body, head, w)

    def stage(self) -> ast.Stage:
        name = self.expect("ident")
        self.punct("{")
        rules = []
        while True:
            self.drain_docs()
            if self.peek() == ("punct", "}"):
                self.next()
                break
            rules.append(self.rule())
        return ast.Stage(name, tuple(rules))

    def link(self) -> ast.LinkDecl:
        doc = self.take_docs()
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
                            post_atoms)


def parse(text: str) -> ast.Catalog:
    p = _P(_tokens(text))
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
        if k != "ident":
            raise ParseError(f"expected declaration, got {v!r}")
        if v == "include":
            p.next()
            path = p.expect("string")
            p.punct(".")
            decls.append(ast.IncludeDecl(path))
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
        elif v == "type":
            p.next()
            tname = p.expect("ident")
            p.punct("{")
            consts = []
            while p.peek() != ("punct", "}"):
                k2, v2 = p.next()
                if k2 not in ("ident", "num"):
                    raise ParseError(f"bad type constant {v2!r}")
                if p.peek()[0] == "range":
                    p.next()
                    k3, v3 = p.next()
                    if k3 not in ("ident", "num"):
                        raise ParseError(f"bad range end {v3!r}")
                    consts.extend(_expand_range(v2, v3))
                else:
                    consts.append(v2)
            p.next()
            p.punct(".")
            decls.append(ast.TypeDecl(tname, tuple(consts)))
        elif v == "namespace":
            p.next()
            nname = p.expect("ident")
            p.expect("ident", "produce")
            produce = p.name_list_paren()
            consume = None
            if p.peek() == ("ident", "consume"):
                p.next()
                consume = p.name_list_paren()
            p.punct(".")
            decls.append(ast.NamespaceDecl(nname, produce, consume))
        elif v == "pred":
            p.next()
            pname = p.expect("ident")
            args = p.sig_args()
            p.punct(":")
            space = p.expect("ident")
            p.punct(".")
            decls.append(ast.PredDecl(pname, args, space))
        elif v == "bwd":
            p.next()
            bname = p.expect("ident")
            args = p.sig_args()
            p.punct(".")
            decls.append(ast.BwdDecl(bname, args))
        elif v in ("weight", "guard", "input", "output"):
            p.next()
            oname = p.expect("ident")
            args = p.sig_args()
            reads = None
            if p.peek() == ("ident", "reads"):
                if v != "weight":
                    raise ParseError(f"{v} port {oname}: reads is for "
                                     f"weight ports only")
                p.next()
                reads = p.reads_block()
            p.punct(".")
            decls.append(ast.PortDecl(v, oname, args, reads=reads))
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
                decls.append(ast.HornRule(head, tuple(body)))
            else:
                p.punct(".")
                decls.append(ast.Fact(head))
    return ast.Catalog(name, version, provenance, layers, stages,
                       extends=extends, decls=tuple(decls))


def parse_file(path) -> ast.Catalog:
    with open(path) as f:
        return parse(f.read())
