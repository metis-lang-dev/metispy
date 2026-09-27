#!/usr/bin/env python3
"""LLP language server — the metis toolchain as an LSP (stdio).

Diagnostics ARE the real pipeline: lexer offsets, parser token
positions, compiler/typecheck findings (parse -> resolve_includes ->
compile_catalog), mapped back to source ranges. Also: completion
(keywords + declared vocabulary), hover (pred/port/type/rule decls),
document symbols (catalog outline), go-to-definition (vocabulary).

Editors launch:  <python> /path/to/metis/metis/lang/lsp.py
(stdlib only besides metis itself; sys.path self-locating so any
python with metis's deps works — the theia venv on this host)."""

import json
import re
import sys
from pathlib import Path

# script-mode: python .../metis/lang/lsp.py puts metis/lang on
# sys.path[0], where our ast.py would SHADOW the stdlib ast module
# (inspect imports it) — drop the script dir, add the repo root
_here = Path(__file__).resolve()
sys.path = [p for p in sys.path
            if Path(p or ".").resolve() != _here.parent]
sys.path.insert(0, str(_here.parents[2]))

from metis.lang import LangError, compile_catalog, resolve_includes  # noqa: E402
from metis.lang.parser import _TOKEN, ParseError, parse             # noqa: E402
from metis.lang import ast as A                                     # noqa: E402
from metis.kernel.catalog import CatalogError                       # noqa: E402

KEYWORDS = [
    "catalog", "provenance", "extends", "include", "layers",
    "stages", "type", "namespace", "produce", "consume", "pred",
    "bwd", "weight", "guard", "input", "output", "reads", "stage",
    "qui", "if", "one",
]


# -- text/position machinery ---------------------------------------------

class Doc:
    def __init__(self, text: str):
        self.text = text
        self.line_starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.line_starts.append(i + 1)

    def pos(self, offset: int):
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return {"line": lo, "character": offset - self.line_starts[lo]}

    def rng(self, start: int, end: int):
        return {"start": self.pos(start), "end": self.pos(end)}

    def word_at(self, line: int, char: int):
        if line >= len(self.line_starts):
            return "", 0, 0
        start = self.line_starts[line]
        end = (self.line_starts[line + 1] - 1
               if line + 1 < len(self.line_starts) else len(self.text))
        s = self.text[start:end]
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_-]*", s):
            if m.start() <= char <= m.end():
                return m.group(), start + m.start(), start + m.end()
        return "", 0, 0

    def tokens(self):
        """(kind, value, start, end) — the parser's own lexer regex,
        with positions kept (the parser drops them; we need them to
        map its token-index errors back to ranges)."""
        out = []
        for m in _TOKEN.finditer(self.text):
            kind = m.lastgroup
            if kind in ("comment", "ws"):
                continue
            if kind == "bad":
                break                      # the lex error path reports
            out.append((kind, m.group(), m.start(), m.end()))
        return out

    def find_decl(self, name: str):
        """Offset range of a declaration of `name` (pred/bwd/type/
        namespace/port/stage/rule/link) — textual, good enough for
        finding-mapping and go-to-definition."""
        pats = [
            rf"^\s*(?:pred|bwd|type|namespace|weight|guard|input|"
            rf"output|stage)\s+({re.escape(name)})\b",
            rf"^\s*qui\s+({re.escape(name)})\b",
            rf"^\s*({re.escape(name)})\s*\[",       # rule head
        ]
        for pat in pats:
            m = re.search(pat, self.text, re.MULTILINE)
            if m:
                return m.start(1), m.end(1)
        return None


def diag(rng, msg, severity=1):
    return {"range": rng, "message": msg, "severity": severity,
            "source": "metis"}


# -- analysis ------------------------------------------------------------

class Analysis:
    def __init__(self, uri: str, text: str):
        self.doc = Doc(text)
        self.diags = []
        self.cat = None                    # parsed AST (maybe None)
        path = uri_to_path(uri)
        try:
            self.cat = parse(text)
        except ParseError as e:
            self._parse_diag(str(e))
            return
        except Exception as e:            # lex error carries an offset
            self._offset_diag(str(e))
            return
        try:
            resolved = resolve_includes(
                self.cat, path.parent if path else Path("."))
            compile_catalog(resolved)
        except LangError as e:
            findings = getattr(e, "findings", [str(e)])
            if len(findings) == 1 and "extends" in findings[0] \
                    and "no base" in findings[0]:
                self.diags.append(diag(
                    self._name_range("extends"), findings[0] +
                    " — pack files check fully at admission",
                    severity=3))
                return
            for f in findings:
                self._finding_diag(f)
        except CatalogError as e:
            for f in getattr(e, "findings", [str(e)]):
                self._finding_diag(f)
        except FileNotFoundError as e:
            self.diags.append(diag(
                self._name_range("include"), f"include: {e}", 1))
        except Exception as e:
            self.diags.append(diag(
                self.doc.rng(0, 1), f"internal: {e}", 1))

    def _offset_diag(self, msg):
        m = re.search(r"offset (\d+)", msg)
        off = int(m.group(1)) if m else 0
        self.diags.append(diag(self.doc.rng(off, off + 1), msg))

    def _parse_diag(self, msg):
        m = re.search(r"token (\d+)", msg)
        toks = self.doc.tokens()
        if m and toks:
            i = min(int(m.group(1)), len(toks) - 1)
            _, _, s, e = toks[i]
            self.diags.append(diag(self.doc.rng(s, e), msg))
        else:
            self.diags.append(diag(self.doc.rng(0, 1), msg))

    def _name_range(self, word):
        m = re.search(rf"\b{re.escape(word)}\b", self.doc.text)
        if m:
            return self.doc.rng(m.start(), m.end())
        return self.doc.rng(0, 1)

    def _finding_diag(self, finding: str):
        """Compiler findings name their subject ('rule: ...',
        'weight f: ...', \"predicate 'p' ...\"); anchor the range at
        that subject's declaration."""
        subject = None
        m = re.match(r"([A-Za-z][A-Za-z0-9_/-]*):", finding)
        if m:
            # kernel findings are layer-qualified (dyn/r1): anchor
            # at the bare rule name
            subject = m.group(1).rsplit("/", 1)[-1]
        else:
            m = re.search(r"'([A-Za-z][A-Za-z0-9_-]*)'", finding)
            if m:
                subject = m.group(1)
            else:
                m = re.match(r"(?:weight|fact|guard)\s+"
                             r"([A-Za-z][A-Za-z0-9_-]*)", finding)
                if m:
                    subject = m.group(1)
        loc = self.doc.find_decl(subject) if subject else None
        rng = (self.doc.rng(*loc) if loc
               else self.doc.rng(0, 1))
        self.diags.append(diag(rng, finding))

    # -- vocabulary ------------------------------------------------------

    def decls(self):
        """(kind, name, detail) harvested from the AST."""
        out = []
        if self.cat is None:
            return out
        c = self.cat
        out.append(("catalog", c.name,
                    f"catalog {c.name} {c.version}"))
        for s in c.stages:
            out.append(("stage-name", s, f"stage {s}"))
        for lay in c.layers:
            out.append(("layer", lay, f"layer {lay}"))
        for d in c.decls:
            if isinstance(d, A.TypeDecl):
                out.append(("type", d.name,
                            f"type {d.name} {{{' '.join(d.consts)}}}"))
                for k in d.consts:
                    out.append(("const", k, f"{k} : {d.name}"))
            elif isinstance(d, A.NamespaceDecl):
                out.append(("namespace", d.name,
                            f"namespace {d.name} produce "
                            f"({' '.join(d.produce)})"))
            elif isinstance(d, A.PredDecl):
                a = f"({','.join(d.args)})" if d.args else ""
                out.append(("pred", d.name,
                            f"pred {d.name}{a} : {d.namespace}"))
            elif isinstance(d, A.BwdDecl):
                a = f"({','.join(d.args)})" if d.args else ""
                out.append(("bwd", d.name, f"bwd {d.name}{a}"))
            elif isinstance(d, A.PortDecl):
                a = f"({','.join(d.args)})" if d.args else ""
                r = ""
                if d.reads is not None:
                    r = " reads {" + ", ".join(
                        str(p) for p in d.reads) + "}"
                out.append((d.kind, d.name,
                            f"{d.kind} {d.name}{a}{r}"))
            elif isinstance(d, A.Stage):
                out.append(("stage", d.name, f"stage {d.name}"))
                for r in d.rules:
                    out.append(("rule", r.name,
                                f"{r.name} [{r.layer}] — {r.doc}"))
            elif isinstance(d, A.LinkDecl):
                out.append(("link", d.name,
                            f"qui {d.name}: {d.pre} -o {d.post} — "
                            f"{d.doc}"))
        return out


def uri_to_path(uri: str):
    if uri.startswith("file://"):
        return Path(uri[7:])
    return None


# -- the server loop -----------------------------------------------------

class Server:
    def __init__(self):
        self.docs = {}       # uri -> Analysis

    def analyze(self, uri, text):
        a = Analysis(uri, text)
        self.docs[uri] = a
        self.notify("textDocument/publishDiagnostics",
                    {"uri": uri, "diagnostics": a.diags})

    # transport
    def send(self, payload):
        body = json.dumps(payload).encode()
        sys.stdout.buffer.write(
            b"Content-Length: %d\r\n\r\n" % len(body))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()

    def reply(self, rid, result):
        self.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def notify(self, method, params):
        self.send({"jsonrpc": "2.0", "method": method,
                   "params": params})

    def run(self):
        stdin = sys.stdin.buffer
        while True:
            line = stdin.readline()
            if not line:
                return
            length = 0
            while line and line.strip():
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":")[1])
                line = stdin.readline()
            if not length:
                continue
            msg = json.loads(stdin.read(length))
            try:
                self.dispatch(msg)
            except Exception as e:
                if "id" in msg:
                    self.send({"jsonrpc": "2.0", "id": msg["id"],
                               "error": {"code": -32603,
                                         "message": str(e)}})

    def dispatch(self, msg):
        method = msg.get("method", "")
        params = msg.get("params", {})
        rid = msg.get("id")
        if method == "initialize":
            self.reply(rid, {"capabilities": {
                "textDocumentSync": 1,          # full
                "completionProvider": {
                    "triggerCharacters": [" ", "(", "$"]},
                "hoverProvider": True,
                "definitionProvider": True,
                "documentSymbolProvider": True,
            }, "serverInfo": {"name": "metis-llp",
                              "version": "0.1"}})
        elif method == "shutdown":
            self.reply(rid, None)
        elif method == "exit":
            sys.exit(0)
        elif method == "textDocument/didOpen":
            td = params["textDocument"]
            self.analyze(td["uri"], td["text"])
        elif method == "textDocument/didChange":
            uri = params["textDocument"]["uri"]
            self.analyze(uri, params["contentChanges"][-1]["text"])
        elif method == "textDocument/didClose":
            self.docs.pop(params["textDocument"]["uri"], None)
        elif method == "textDocument/completion":
            self.reply(rid, self.completion(params))
        elif method == "textDocument/hover":
            self.reply(rid, self.hover(params))
        elif method == "textDocument/definition":
            self.reply(rid, self.definition(params))
        elif method == "textDocument/documentSymbol":
            self.reply(rid, self.symbols(params))
        elif rid is not None:
            self.reply(rid, None)

    def _doc(self, params):
        return self.docs.get(params["textDocument"]["uri"])

    def completion(self, params):
        a = self._doc(params)
        items = [{"label": k, "kind": 14} for k in KEYWORDS]
        if a:
            kinds = {"pred": 3, "bwd": 3, "type": 7, "const": 21,
                     "namespace": 9, "stage-name": 13, "layer": 14,
                     "weight": 3, "rule": 15}
            for kind, name, detail in a.decls():
                items.append({"label": name,
                              "kind": kinds.get(kind, 6),
                              "detail": detail})
        return {"isIncomplete": False, "items": items}

    def hover(self, params):
        a = self._doc(params)
        if not a:
            return None
        p = params["position"]
        word, s, e = a.doc.word_at(p["line"], p["character"])
        if not word:
            return None
        for kind, name, detail in a.decls():
            if name == word:
                return {"contents": {
                    "kind": "markdown",
                    "value": f"```llp\n{detail}\n```"},
                    "range": a.doc.rng(s, e)}
        if word in KEYWORDS:
            return {"contents": {"kind": "plaintext",
                                 "value": f"llp keyword: {word}"}}
        return None

    def definition(self, params):
        a = self._doc(params)
        if not a:
            return None
        p = params["position"]
        word, _, _ = a.doc.word_at(p["line"], p["character"])
        loc = a.doc.find_decl(word) if word else None
        if not loc:
            return None
        return {"uri": params["textDocument"]["uri"],
                "range": a.doc.rng(*loc)}

    def symbols(self, params):
        a = self._doc(params)
        if not a or a.cat is None:
            return []
        out = []
        kinds = {"type": 5, "namespace": 3, "pred": 12, "bwd": 12,
                 "weight": 12, "guard": 12, "input": 12,
                 "output": 12, "stage": 2, "link": 24,
                 "rule": 6}
        for kind, name, detail in a.decls():
            if kind in ("catalog", "stage-name", "layer", "const"):
                continue
            loc = a.doc.find_decl(name)
            if not loc:
                continue
            rng = a.doc.rng(*loc)
            out.append({"name": name,
                        "detail": detail,
                        "kind": kinds.get(kind, 13),
                        "range": rng, "selectionRange": rng})
        return out


if __name__ == "__main__":
    Server().run()
