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
    for bad, words in (("ollama", "needs a model"), ("http://box:11434", "needs a model after it"),
                       ("ollama phi3 box", "third word is its URL"), ("sorcery x", "unknown LLM backend")):
        with pytest.raises(ValueError) as e:
            f(bad)
        assert words in str(e.value)


class _Stub(BaseHTTPRequestHandler):
    seen = []
    reply = {}

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        _Stub.seen.append((self.path, dict(self.headers), body))
        out = json.dumps(_Stub.reply).encode()
        self.send_response(200)
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
