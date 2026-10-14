"""The LLM endpoint — ONE configuration for both text boundaries
(formalize in, narrate out), stdlib only.

An endpoint spec is a string:

    ollama[:MODEL]        the local Ollama server (loop/ollama.py;
                          $OLLAMA_HOST / $OLLAMA_MODEL defaults)
    anthropic[:MODEL]     the Anthropic API ($ANTHROPIC_API_KEY;
                          default model claude-opus-5-5)
    command:ARGV...       an external command, prompt on stdin,
                          completion on stdout (the Formalizer seam's
                          shape) — never reachable from page intents

`resolve` parses a spec, `complete` runs one prompt through it. The
completion is UNTRUSTED text either way: the formalize path gates it,
the narrate path labels it as narration — neither feeds it back into
the semantics."""
import json
import os
import shlex
import subprocess
import urllib.request
from dataclasses import dataclass

DEFAULT_SPEC = "ollama"
ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_VERSION = "2023-06-01"


@dataclass(frozen=True)
class Endpoint:
    backend: str                 # "ollama" | "anthropic" | "command"
    arg: str = ""                # model, or the command line


def resolve(spec: str) -> Endpoint:
    spec = (spec or DEFAULT_SPEC).strip()
    backend, _, rest = spec.partition(":")
    if backend not in ("ollama", "anthropic", "command"):
        raise ValueError(f"unknown LLM backend '{backend}' — "
                         f"ollama[:MODEL] | anthropic[:MODEL] | "
                         f"command:ARGV")
    if backend == "command" and not rest.strip():
        raise ValueError("command backend needs a command line")
    return Endpoint(backend, rest.strip())


def _anthropic(prompt: str, model: str, timeout: int,
               options: dict = None, system: str = None,
               meta: dict = None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    url = os.environ.get("ANTHROPIC_API_URL",
                         "https://api.anthropic.com").rstrip("/")
    req = urllib.request.Request(
        url + "/v1/messages",
        data=json.dumps({
            "model": model or ANTHROPIC_DEFAULT_MODEL,
            "max_tokens": 2048,
            **({"temperature": options["temperature"]}
               if options and "temperature" in options else {}),
            **({"system": system} if system else {}),
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        headers={"Content-Type": "application/json",
                 "x-api-key": key,
                 "anthropic-version": ANTHROPIC_VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    if meta is not None:
        meta["cut"] = out.get("stop_reason") == "max_tokens"
        meta["tokens"] = (out.get("usage") or {}).get("output_tokens")
    return "".join(b.get("text", "")
                   for b in out.get("content", [])).strip()


def complete(spec, prompt: str, timeout: int = 600,
             options: dict = None, system: str = None,
             meta: dict = None) -> str:
    """One completion from the configured endpoint. `options` are
    sampling knobs (Ollama's: temperature, num_ctx, ...); the
    anthropic backend honors temperature only, command ignores them.
    `system` rides the backend's system slot; the command backend,
    which has none, gets it prepended to the prompt. `meta`, when
    given, says whether the endpoint stopped at its token cap (cut)
    and how many tokens it wrote; the command backend cannot tell
    (cut False)."""
    ep = spec if isinstance(spec, Endpoint) else resolve(spec)
    if meta is not None:
        meta.setdefault("cut", False)
    if ep.backend == "ollama":
        from .ollama import generate
        return generate(prompt, model=ep.arg or None, timeout=timeout,
                        options=options, system=system, meta=meta)
    if ep.backend == "anthropic":
        return _anthropic(prompt, ep.arg, timeout, options, system, meta)
    if system:
        prompt = system + "\n\n" + prompt
    proc = subprocess.run(shlex.split(ep.arg), input=prompt,
                          capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command backend failed "
                           f"({proc.returncode}): "
                           f"{proc.stderr.strip()[:400]}")
    return proc.stdout.strip()
