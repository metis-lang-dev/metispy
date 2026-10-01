"""LLP language-server gates: drive metis/lang/lsp.py over real
stdio JSON-RPC exactly as an editor would — initialize, open a good
catalog (zero diagnostics + hover + symbols + completion), open a
broken one (parse error at the right token; compiler finding
anchored at the offending rule)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOOD = (ROOT / "metis/domains/fiction/catalogs/"
        "love_triangle.llp").read_text()
BAD_PARSE = "catalog broken 1.\nstage go {\n  r1 [dyn] : a -o .\n}\n"
BAD_COMPILE = """catalog badc 1.
layers (dyn).
stages (go).
namespace st produce (dyn).
pred a : st.
pred b : st.
stage go {
  %% consumes an undeclared predicate.
  r1 [dyn] : ghost -o b.
}
"""


class Client:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "metis/lang/lsp.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.rid = 0

    def send(self, method, params, request=True):
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if request:
            self.rid += 1
            msg["id"] = self.rid
        body = json.dumps(msg).encode()
        self.proc.stdin.write(
            b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self.proc.stdin.flush()
        return self.rid if request else None

    def recv(self):
        length = 0
        line = self.proc.stdout.readline()
        while line and line.strip():
            if line.lower().startswith(b"content-length:"):
                length = int(line.split(b":")[1])
            line = self.proc.stdout.readline()
        return json.loads(self.proc.stdout.read(length))

    def wait_for(self, pred):
        for _ in range(20):
            msg = self.recv()
            if pred(msg):
                return msg
        raise AssertionError("expected message never arrived")

    def open(self, uri, text):
        self.send("textDocument/didOpen",
                  {"textDocument": {"uri": uri, "languageId": "llp",
                                    "version": 1, "text": text}},
                  request=False)
        return self.wait_for(
            lambda m: m.get("method") ==
            "textDocument/publishDiagnostics"
            and m["params"]["uri"] == uri)["params"]["diagnostics"]

    def request(self, method, params):
        rid = self.send(method, params)
        return self.wait_for(lambda m: m.get("id") == rid)["result"]

    def close(self):
        self.proc.stdin.close()
        self.proc.terminate()


def make_client():
    c = Client()
    r = c.request("initialize", {"capabilities": {}})
    assert r["serverInfo"]["name"] == "metis-llp"
    assert r["capabilities"]["hoverProvider"]
    return c


def test_clean_catalog_and_queries():
    c = make_client()
    try:
        uri = "file:///tmp/lt.llp"
        assert c.open(uri, GOOD) == []          # zero diagnostics
        comp = c.request("textDocument/completion",
                         {"textDocument": {"uri": uri},
                          "position": {"line": 0, "character": 0}})
        labels = {i["label"] for i in comp["items"]}
        assert {"stage", "qui", "eros", "anger",
                "helena"} <= labels
        # hover over `eros` in the jealousy rule body
        lines = GOOD.splitlines()
        ln = next(i for i, l in enumerate(lines)
                  if "$eros(A,B)" in l)
        h = c.request("textDocument/hover",
                      {"textDocument": {"uri": uri},
                       "position": {"line": ln,
                                    "character":
                                        lines[ln].index("eros")}})
        assert "pred eros(character,character)" in \
            h["contents"]["value"]
        syms = c.request("textDocument/documentSymbol",
                         {"textDocument": {"uri": uri}})
        names = {s["name"] for s in syms}
        assert {"go", "jealousy", "move_on", "eros"} <= names
        d = c.request("textDocument/definition",
                      {"textDocument": {"uri": uri},
                       "position": {"line": ln,
                                    "character":
                                        lines[ln].index("eros")}})
        decl_ln = next(i for i, l in enumerate(lines)
                       if l.startswith("pred eros"))
        assert d["range"]["start"]["line"] == decl_ln
    finally:
        c.close()


def test_parse_error_positions():
    c = make_client()
    try:
        diags = c.open("file:///tmp/bad.llp", BAD_PARSE)
        assert len(diags) == 1
        assert "expected" in diags[0]["message"]
        assert diags[0]["range"]["start"]["line"] == 2   # the rule
    finally:
        c.close()


def test_compile_finding_anchored_at_rule():
    c = make_client()
    try:
        diags = c.open("file:///tmp/badc.llp", BAD_COMPILE)
        assert any("undeclared predicate 'ghost'" in d["message"]
                   for d in diags)
        d = next(d for d in diags if "ghost" in d["message"])
        assert d["range"]["start"]["line"] == 8   # the r1 rule line
    finally:
        c.close()


def test_send_to_interpreter():
    """workspace/executeCommand llp.trace: the editor's Send to
    Interpreter — replay the buffer's embedded '#'-directive case."""
    c = make_client()
    try:
        uri = "file:///tmp/lt.llp"
        assert c.open(uri, GOOD) == []
        r = c.request("workspace/executeCommand",
                      {"command": "llp.trace", "arguments": [uri]})
        assert r["ok"]
        lines = r["text"].splitlines()
        assert lines[0] == "case lt steps=6 seeds=3"
        # the editor surface renders the pretty (Foata-beat) form
        assert any(l.startswith("T 1 ") and "beats" in l for l in lines)
        assert any("jealousy" in l for l in lines)
        assert any(l.startswith("Q anger(hermia,helena)")
                   for l in lines)
        # a buffer with no directives says how to add them
        u2 = "file:///tmp/bare.llp"
        c.open(u2, "\n".join(
            l for l in GOOD.splitlines() if not l.startswith("#")))
        r2 = c.request("workspace/executeCommand",
                       {"command": "llp.trace", "arguments": [u2]})
        assert not r2["ok"] and "#init" in r2["text"]
    finally:
        c.close()
