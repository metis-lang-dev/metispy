"""The REPL's model (metis.loop.llm; the user, 2026-11-04: narrate and llm stay with the REPL): the spec forms,
the words `llm` takes, the Ollama and OpenRouter backends against stub servers (the token cap on every request,
the OpenRouter key from the environment only, the cost log), and the setting saved in the dot file."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from metis.loop import config, llm


def test_resolve_forms():
    assert llm.resolve("ollama:phi3") == llm.Endpoint("ollama", "phi3", "http://127.0.0.1:11434")
    assert llm.resolve("ollama:phi3@http://box:11434/").url == "http://box:11434"
    assert llm.resolve("ollama:phi3").spec == "ollama:phi3"
    assert llm.resolve("ollama:phi3@http://box:11434").spec == "ollama:phi3@http://box:11434"
    ep = llm.resolve("openrouter:deepseek/deepseek-v4.1-flash-1017")
    assert (ep.backend, ep.arg) == ("openrouter", "deepseek/deepseek-v4.1-flash-1017")
    assert llm.resolve("anthropic").arg == "" and llm.resolve("command:cat").arg == "cat"
    for bad, words in (("", "no LLM endpoint"), ("sorcery:x", "unknown LLM backend"), ("ollama", "needs a model"),
                       ("openrouter", "needs a model"), ("command", "needs a command line")):
        with pytest.raises(ValueError) as e:
            llm.resolve(bad)
        assert words in str(e.value)
    assert "ollama:MODEL[@URL] | openrouter:MODEL" in llm.UNSET


def test_spec_from_words():
    f = llm.spec_from_words
    assert f("ollama phi3") == "ollama:phi3"
    assert f("ollama phi3 http://box:11434") == "ollama:phi3@http://box:11434"
    assert f("http://box:11434 phi3") == "ollama:phi3@http://box:11434"
    assert f("openrouter deepseek/deepseek-v4.1-flash-1017") == "openrouter:deepseek/deepseek-v4.1-flash-1017"
    assert f("anthropic") == "anthropic" and f("anthropic claude-opus-5-5") == "anthropic:claude-opus-5-5"
    assert f("command cat -n") == "command:cat -n" and f("ollama:phi3") == "ollama:phi3"
    assert f("command:cat -n") == "command:cat -n"
    with pytest.raises(ValueError) as e:
        f("ollama:phi3 extra")
    assert "a spec is one word" in str(e.value)
    for bad, words in (("ollama", "needs a model"), ("http://box:11434", "needs a model after it"),
                       ("ollama phi3 box", "third word is its URL"), ("sorcery x", "unknown LLM backend")):
        with pytest.raises(ValueError) as e:
            f(bad)
        assert words in str(e.value)


class _Stub(BaseHTTPRequestHandler):
    seen = []
    reply = {}
    status = 200

    def do_GET(self):
        self.do_POST()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        _Stub.seen.append((self.path, dict(self.headers), body))
        out = json.dumps(_Stub.reply).encode()
        self.send_response(_Stub.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    _Stub.seen.clear()
    _Stub.status = 200
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_backend_caps_and_reports_the_cut(stub, monkeypatch):
    monkeypatch.setenv("METIS_CONFIG", "")               # no dot file: the default cap
    _Stub.reply = {"response": "The lamp went dark. Then", "done_reason": "length", "eval_count": 1024}
    meta = {}
    text = llm.complete(f"ollama:phi3@{stub}", "tell it", options={"temperature": 0.3}, system="doctrine", meta=meta)
    assert text == "The lamp went dark. Then" and meta == {"cut": True, "tokens": 1024}
    path, _, body = _Stub.seen[-1]
    assert path == "/api/generate" and body["model"] == "phi3" and body["stream"] is False
    assert body["system"] == "doctrine" and body["options"] == {"num_predict": 1024, "temperature": 0.3}
    with pytest.raises(RuntimeError) as e:
        llm.complete("ollama:phi3@http://127.0.0.1:9", "x")
    assert "ollama at http://127.0.0.1:9 not reachable" in str(e.value)


def test_openrouter_backend_key_from_env_only_cap_and_cost_log(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("METIS_CONFIG", "")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_URL", stub)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as e:
        llm.complete("openrouter:deepseek/deepseek-v4.1-flash-1017", "x")
    assert "OPENROUTER_API_KEY is not set" in str(e.value) and not _Stub.seen
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-never-printed")
    _Stub.reply = {"choices": [{"message": {"content": "It waited."}, "finish_reason": "stop"}],
                   "usage": {"completion_tokens": 7, "prompt_tokens": 300, "cost": 0.00042}}
    meta = {}
    assert llm.complete("openrouter:deepseek/deepseek-v4.1-flash-1017", "tell", system="doc", meta=meta) == "It waited."
    path, headers, body = _Stub.seen[-1]
    assert path == "/chat/completions" and headers["Authorization"] == "Bearer sk-test-never-printed"
    assert body["max_tokens"] == 1024 and body["usage"] == {"include": True}
    assert body["reasoning"] == {"enabled": False}           # think off: the cap is the told text's
    assert body["messages"][0] == {"role": "system", "content": "doc"}
    assert meta == {"cut": False, "tokens": 7, "cost": 0.00042}
    rows = [json.loads(l) for l in (tmp_path / "metis" / "cost.jsonl").read_text().splitlines()]
    assert rows[-1]["model"] == "deepseek/deepseek-v4.1-flash-1017" and rows[-1]["cost"] == 0.00042
    assert rows[-1]["who"] == "repl" and rows[-1]["seat"] == "narrate" and "sk-test" not in json.dumps(rows)


def test_config_set_writes_the_dot_file(tmp_path, monkeypatch):
    f = tmp_path / "config.toml"
    f.write_text("# mine\n[narrate]\ntemperature = 0.3\n\n[llm]\nendpoint = \"command:cat\"\n# a comment\n")
    monkeypatch.setenv("METIS_CONFIG", str(f))
    config.load(reload=True)
    assert config.set("llm", "endpoint", "ollama:phi3") == f
    assert f.read_text() == "# mine\n[narrate]\ntemperature = 0.3\n\n[llm]\nendpoint = \"ollama:phi3\"\n# a comment\n"
    assert config.get("llm", "endpoint") == "ollama:phi3"
    config.set("llm", "max_tokens", 512)
    assert config.get("llm", "max_tokens") == 512 and llm.max_tokens() == 512
    config.set("ports", "plugin", "x")                                  # a new section is appended
    assert f.read_text().endswith("[llm]\nendpoint = \"ollama:phi3\"\n# a comment\nmax_tokens = 512\n\n[ports]\nplugin = \"x\"\n")
    g = tmp_path / "new" / "config.toml"                                # no file yet: made
    monkeypatch.setenv("METIS_CONFIG", str(g))
    config.load(reload=True)
    config.set("llm", "endpoint", "openrouter:m")
    assert g.read_text() == "[llm]\nendpoint = \"openrouter:m\"\n"
    monkeypatch.setenv("METIS_CONFIG", "")                              # pinned empty: nothing to write
    config.load(reload=True)
    with pytest.raises(ValueError):
        config.set("llm", "endpoint", "x")


def _repl(capsys=None):
    import io
    from metis.lang.repl import Repl
    from pathlib import Path
    return Repl(Path(__file__).resolve().parent / "fixtures/gossip3.llp", seed=1, stdout=io.StringIO())


def _say(r, cmd):
    r.stdout.truncate(0), r.stdout.seek(0)
    r.onecmd(cmd)
    return r.stdout.getvalue()


def test_repl_llm_saves_and_the_next_repl_reads_it(monkeypatch, tmp_path):
    """`llm ollama phi3` saves [llm] endpoint in the dot file; a new REPL
    starts on it, and its intro says it and whether the OpenRouter key
    is set (never its value)."""
    f = tmp_path / "config.toml"
    f.write_text("[narrate]\ntemperature = 0.2\n")
    monkeypatch.setenv("METIS_CONFIG", str(f))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config.load(reload=True)
    r = _repl()
    assert "llm (unset: llm ollama MODEL | llm openrouter MODEL) · OpenRouter key not set" in r.intro
    out = _say(r, "llm ollama phi3 http://127.0.0.1:9")
    assert f"saved to {f}" in out and "not reachable" in out
    assert 'endpoint = "ollama:phi3@http://127.0.0.1:9"' in f.read_text()
    assert "temperature = 0.2" in f.read_text()               # the rest stays
    out = _say(r, "llm ollama phi3")
    assert 'endpoint = "ollama:phi3"' in f.read_text()
    config.load(reload=True)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-never-shown")
    r2 = _repl()
    assert r2.llm == "ollama:phi3"
    assert "llm ollama:phi3 · OpenRouter key set (OPENROUTER_API_KEY)" in r2.intro
    assert "sk-never-shown" not in r2.intro
    assert "unknown LLM backend" in _say(r2, "llm sorcery x")
    assert 'endpoint = "ollama:phi3"' in f.read_text()        # a refused set saves nothing


def test_repl_llm_openrouter_without_the_key_is_one_line_and_a_note(monkeypatch, tmp_path):
    monkeypatch.setenv("METIS_CONFIG", str(tmp_path / "c.toml"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config.load(reload=True)
    r = _repl()
    out = _say(r, "llm openrouter deepseek/deepseek-v4.1-flash-1017")
    assert out.count("\n") == 1 and "OPENROUTER_API_KEY is not set" in out
    assert 'endpoint = "openrouter:deepseek/deepseek-v4.1-flash-1017"' in (tmp_path / "c.toml").read_text()
    out = _say(r, "llm")
    assert out.count("\n") == 1 and "OPENROUTER_API_KEY is not set" in out
    notes = [d for d in r.notes if d.code == "llm-endpoint"]
    assert notes and dict(notes[-1].data)["problem"] == "OPENROUTER_API_KEY not set"
    r.onecmd("run 3")
    out = _say(r, "narrate")
    assert out.count("\n") == 1 and "narrate failed" in out and "OPENROUTER_API_KEY is not set" in out


def test_repl_llm_status_checks_the_ollama_server(stub, monkeypatch):
    """`llm` alone asks the Ollama server's /api/tags (no generation):
    a model it has is fine, one it lacks is named with what it has; a
    narrate on a model the server does not have says so in one line."""
    r = _repl()
    _Stub.reply = {"models": [{"name": "phi3:latest"}, {"name": "gemma3:4b"}]}
    assert _say(r, f"llm {stub} phi3").startswith(f"llm: ollama:phi3@{stub} — not saved")
    out = _say(r, f"llm ollama mistral {stub}")
    assert "model mistral is not pulled" in out and "phi3:latest" in out
    assert _Stub.seen[-1][0] == "/api/tags"


def test_ollama_passes_the_seat_options_and_the_lower_cap(stub, monkeypatch, tmp_path):
    """[narrate] options ride to Ollama over a temperature-0 default
    (num_ctx, repeat_penalty: the narrate seat's tuning), think off, and
    a seat num_predict below the cap is the request's cap; a 404 is a
    model not pulled, not a dead server."""
    _Stub.reply = {"response": "```\nThe pig built.\n```", "done_reason": "stop", "eval_count": 5}
    text = llm.complete(f"ollama:gemma3:4b@{stub}", "p",
                        options={"temperature": 0.2, "num_ctx": 8192, "num_predict": 700})
    assert text == "The pig built."
    body = _Stub.seen[-1][2]
    assert body["think"] is False and body["model"] == "gemma3:4b"
    assert body["options"] == {"temperature": 0.2, "num_ctx": 8192, "num_predict": 700}
    llm.complete(f"ollama:m@{stub}", "p", options={"num_predict": 99999})
    assert _Stub.seen[-1][2]["options"] == {"temperature": 0, "num_predict": 1024}
    _Stub.status, _Stub.reply = 404, {"error": "model 'm' not found"}
    with pytest.raises(RuntimeError) as e:
        llm.complete(f"ollama:m@{stub}", "p")
    assert str(e.value) == f"ollama: model 'm' is not pulled at {stub} (ollama pull m, or `llm ollama MODEL`)"


def test_openrouter_http_error_is_one_line_without_the_key(stub, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_URL", stub)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-never-printed")
    _Stub.status, _Stub.reply = 402, {"error": {"message": "Insufficient credits", "code": 402}}
    with pytest.raises(RuntimeError) as e:
        llm.complete("openrouter:m", "p")
    assert str(e.value) == "openrouter: HTTP 402 Insufficient credits"
    assert "sk-test" not in str(e.value) and not (tmp_path / "metis" / "cost.jsonl").exists()
