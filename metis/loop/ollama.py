"""The Ollama adapter — a local model in the formalizer seat,
stdlib only (urllib), no plugins.

Two shapes, one seam (core.formalize_loop's contract: prompt in,
LLP out):

  - in-process: `backend(model)` returns a callable the REPL's
    :formalize uses directly;
  - command:    `python -m loop.ollama [--model M] [--host H]`
    reads the prompt on stdin and writes LLP on stdout — drop-in
    for `--backend` anywhere an external command is expected.

The candidate is UNTRUSTED either way; only the gate verdict
counts. Generation runs at temperature 0 (reproducible drafts),
and markdown fences are stripped — models add them despite the
prompt saying not to."""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request


def default_host() -> str:
    from . import config
    return os.environ.get("OLLAMA_HOST") \
        or config.get("ollama", "host", "http://localhost:11434")


def default_model() -> str:
    from . import config
    return os.environ.get("OLLAMA_MODEL") \
        or config.get("ollama", "model", "llama3.1")


def default_think():
    """Whether the model thinks before it answers: [ollama] think in the
    dot file (true | false | a level the model knows, e.g. "low"),
    false when unset. A thinking model (Gemma 4, qwen3) left to its
    own default thinks for minutes before a draft; the gate reads only
    the answer, so the seat asks for none."""
    from . import config
    return config.get("ollama", "think", False)


def tags(host: str = None, timeout: int = 5):
    """Installed model names at the server — the validation URI
    (GET /api/tags); raises URLError when the server is down."""
    with urllib.request.urlopen(
            (host or default_host()).rstrip("/") + "/api/tags",
            timeout=timeout) as r:
        out = json.loads(r.read())
    return [m.get("name", "?") for m in out.get("models", [])]


_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def strip_fences(text: str) -> str:
    lines = [l for l in text.splitlines() if not _FENCE.match(l)]
    return "\n".join(lines).strip() + "\n"


def generate(prompt: str, model: str = None, host: str = None,
             timeout: int = 600, options: dict = None,
             system: str = None, meta: dict = None,
             think=None) -> str:
    """One completion from the local Ollama server. `options` merge
    over the temperature-0 default (the narrate seat warms it);
    `think` (default: default_think(), false) is sent as Ollama's
    think field, so a thinking model answers without thinking first;
    `system` goes in the model's system slot, apart from the
    material — small models keep the two straight only that way.
    `meta`, when given, is filled with cut (the server stopped at
    num_predict: done_reason "length") and tokens (eval_count)."""
    req = urllib.request.Request(
        (host or default_host()).rstrip("/") + "/api/generate",
        data=json.dumps({
            "model": model or default_model(),
            "prompt": prompt,
            "stream": False,
            "think": default_think() if think is None else think,
            "options": {"temperature": 0, **(options or {})},
            **({"system": system} if system else {}),
        }).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:              # unknown model, not a dead server
            m, h = model or default_model(), host or default_host()
            try:
                have = ", ".join(tags(h)) or "(none pulled)"
            except OSError:
                have = "(could not list)"
            raise RuntimeError(
                f"ollama: model '{m}' not found at {h} — installed: "
                f"{have}. Pick one with `llm ollama:MODEL`, or set "
                f"[ollama] model in .metis.toml / "
                f"~/.config/metis/config.toml") from e
        raise
    if "response" not in out:
        raise RuntimeError(f"ollama: no response field in {out!r}")
    if meta is not None:
        meta["cut"] = out.get("done_reason") == "length"
        meta["tokens"] = out.get("eval_count")
    return strip_fences(out["response"])


def backend(model: str = None, host: str = None, timeout: int = 600):
    """The in-process formalizer seam: a callable(prompt) -> LLP."""
    return lambda prompt: generate(prompt, model, host, timeout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m loop.ollama",
        description="formalizer command seam over a local Ollama "
                    "server: prompt on stdin, LLP on stdout")
    ap.add_argument("--model", default=None,
                    help=f"model name (default: $OLLAMA_MODEL or "
                         f"{default_model()})")
    ap.add_argument("--host", default=None,
                    help=f"server (default: $OLLAMA_HOST or "
                         f"{default_host()})")
    ap.add_argument("--timeout", type=int, default=600)
    ns = ap.parse_args(argv)
    sys.stdout.write(generate(sys.stdin.read(), ns.model, ns.host,
                              ns.timeout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
