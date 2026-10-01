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
import urllib.request


def default_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def default_model() -> str:
    return os.environ.get("OLLAMA_MODEL", "llama3.1")


_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*$")


def strip_fences(text: str) -> str:
    lines = [l for l in text.splitlines() if not _FENCE.match(l)]
    return "\n".join(lines).strip() + "\n"


def generate(prompt: str, model: str = None, host: str = None,
             timeout: int = 600) -> str:
    """One completion from the local Ollama server."""
    req = urllib.request.Request(
        (host or default_host()).rstrip("/") + "/api/generate",
        data=json.dumps({
            "model": model or default_model(),
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if "response" not in out:
        raise RuntimeError(f"ollama: no response field in {out!r}")
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
