"""The LLM endpoint — ONE configuration for the REPL's text boundary
(narrate out; formalize moved to the pi loop), stdlib only.

An endpoint spec is a string:

    ollama:MODEL[@URL]    an Ollama server (default URL
                          http://127.0.0.1:11434), e.g. ollama:phi3
    openrouter:MODEL      OpenRouter's OpenAI-compatible API, the key
                          from $OPENROUTER_API_KEY ONLY (never a file,
                          never printed), e.g.
                          openrouter:deepseek/deepseek-v4.1-flash-1017
    anthropic[:MODEL]     the Anthropic API ($ANTHROPIC_API_KEY;
                          default model claude-opus-5-5)
    command:ARGV...       an external command, prompt on stdin,
                          completion on stdout (the Formalizer seam's
                          shape) — never reachable from page intents

The REPL's `llm` takes the words a person types and makes the spec
(`spec_from_words`): `llm ollama phi3`, `llm ollama phi3 http://host:11434`,
`llm openrouter deepseek/deepseek-v4.1-flash-1017`, `llm http://host:11434 phi3`
(a URL alone is an Ollama server), or a spec as above; what it sets is
saved in the config dot file ([llm] endpoint) so every REPL on this
host has it (the user, 2026-11-04: narrate and llm stay with the REPL;
people working on a single .llp file keep them).

There is NO default endpoint: an unset spec is an error that says how to
set one, so no model is ever called that the person did not name.

Every paid request has a hard token cap (MAX_TOKENS, [llm] max_tokens)
and the OpenRouter backend appends one row per call to the cost log
(~/.config/metis/cost.jsonl: ts, seat, model, tokens, cost in USD as
the API reports it), the standing rule for paid runs.

ONE configuration for every seat (llm-config s-07a070ad, the user, 2026-11-05: "harmonize config for narration and
formalization. both path can be configured to ollama or openroute"): the dot file's [llm] carries, each in this
grammar, `endpoint` (the REPL's narrate), `loop` (the pi loop's model: loop.sh and the console's loop start read it
when MODEL is not given) and `formalize` (the kernel's formalize seat in the pi loop; default the loop's). `seat`
reads one, `pi_model` maps a spec to the name pi runs (openrouter/M, ollama/M over URL/v1, anthropic/M), and
`python -m metis.loop.llm pi [loop|formalize]` says it to loop.sh as JSON.

`resolve` parses a spec, `complete` runs one prompt through it. The
completion is UNTRUSTED text either way: the narrate path labels it as
narration — it never feeds back into the semantics."""
import json
import os
import shlex
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ANTHROPIC_DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_VERSION = "2023-06-01"
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
MAX_TOKENS = 1024            # the per-request cap, every paid backend
BACKENDS = ("ollama", "openrouter", "anthropic", "command")
FORMS = ("ollama:MODEL[@URL] | openrouter:MODEL | anthropic[:MODEL] | "
         "command:ARGV")


@dataclass(frozen=True)
class Endpoint:
    backend: str                 # one of BACKENDS
    arg: str = ""                # the model, or the command line
    url: str = ""                # the server (ollama), else ""

    @property
    def spec(self) -> str:
        if self.backend == "ollama":
            return f"ollama:{self.arg}" + (
                f"@{self.url}" if self.url and self.url != OLLAMA_DEFAULT_URL
                else "")
        return f"{self.backend}:{self.arg}" if self.arg else self.backend


UNSET = ("no LLM endpoint: `llm ollama MODEL [URL]` or `llm openrouter MODEL` "
         "(saved as [llm] endpoint in the config dot file: $METIS_CONFIG | "
         f".metis.toml | ~/.config/metis/config.toml) — {FORMS}")


def resolve(spec: str) -> Endpoint:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError(UNSET)
    backend, _, rest = spec.partition(":")
    rest = rest.strip()
    if backend not in BACKENDS:
        raise ValueError(f"unknown LLM backend '{backend}' — {FORMS}")
    if backend == "command" and not rest:
        raise ValueError("command backend needs a command line")
    if backend == "ollama":
        model, _, url = rest.partition("@")
        if not model.strip():
            raise ValueError("ollama needs a model: ollama:MODEL[@URL], "
                             "e.g. ollama:phi3")
        return Endpoint("ollama", model.strip(),
                        (url.strip() or OLLAMA_DEFAULT_URL).rstrip("/"))
    if backend == "openrouter" and not rest:
        raise ValueError("openrouter needs a model: openrouter:MODEL, "
                         "e.g. openrouter:deepseek/deepseek-v4.1-flash-1017")
    return Endpoint(backend, rest)


def _is_url(w: str) -> bool:
    return w.startswith("http://") or w.startswith("https://")


def spec_from_words(words) -> str:
    """The spec from what a person types after `llm`: a provider and a
    model (`ollama phi3 [URL]`, `openrouter MODEL`, `anthropic [MODEL]`),
    a URL and a model (an Ollama server), or a spec as it is."""
    ws = [w for w in (words.split() if isinstance(words, str) else words) if w]
    if not ws:
        raise ValueError(UNSET)
    if ws[0].startswith("command:"):           # a command line is words
        return resolve(" ".join(ws)).spec
    if ":" in ws[0] and not _is_url(ws[0]) and len(ws) > 1:
        raise ValueError(f"a spec is one word ({ws[0]}), or say it in words: "
                         "llm ollama MODEL [URL] | llm openrouter MODEL")
    if len(ws) == 1:
        if _is_url(ws[0]):
            raise ValueError("a URL needs a model after it: "
                             "llm http://host:11434 MODEL")
        return resolve(ws[0]).spec if ":" in ws[0] or ws[0] == "anthropic" \
            else _missing(ws[0])
    head = ws[0]
    if _is_url(head):
        return resolve(f"ollama:{ws[1]}@{head}").spec
    if head == "ollama":
        url = ws[2] if len(ws) > 2 else ""
        if url and not _is_url(url):
            raise ValueError(f"ollama's third word is its URL, not {url!r}")
        return resolve(f"ollama:{ws[1]}" + (f"@{url}" if url else "")).spec
    if head in ("openrouter", "anthropic"):
        return resolve(f"{head}:{ws[1]}").spec
    if head == "command":
        return resolve("command:" + " ".join(ws[1:])).spec
    raise ValueError(f"unknown LLM backend '{head}' — {FORMS}")


def _missing(word):
    if word in ("ollama", "openrouter"):
        raise ValueError(f"{word} needs a model: llm {word} MODEL")
    if word == "command":
        raise ValueError("command backend needs a command line")
    raise ValueError(f"unknown LLM backend '{word}' — {FORMS}")


def max_tokens() -> int:
    """The per-request cap: [llm] max_tokens in the dot file, else MAX_TOKENS."""
    try:
        from . import config
        return int(config.get("llm", "max_tokens", MAX_TOKENS) or MAX_TOKENS)
    except Exception:  # noqa: BLE001 — a broken dot file never lifts the cap
        return MAX_TOKENS


def cost_log_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "metis" / "cost.jsonl"


def log_cost(row: dict):
    """One row of the cost log, best effort (a log that cannot be written
    never fails the narration, but is said on stderr once)."""
    try:
        p = cost_log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                **row}) + "\n")
    except OSError as e:
        import sys
        print(f"metis: cost log not written: {e}", file=sys.stderr)


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _anthropic(prompt: str, model: str, timeout: int,
               options: dict = None, system: str = None,
               meta: dict = None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    url = os.environ.get("ANTHROPIC_API_URL",
                         "https://api.anthropic.com").rstrip("/")
    out = _post(url + "/v1/messages", {
        "model": model or ANTHROPIC_DEFAULT_MODEL,
        "max_tokens": _cap(options),
        **({"temperature": options["temperature"]}
           if options and "temperature" in options else {}),
        **({"system": system} if system else {}),
        "messages": [{"role": "user", "content": prompt}],
    }, {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}, timeout)
    if meta is not None:
        meta["cut"] = out.get("stop_reason") == "max_tokens"
        meta["tokens"] = (out.get("usage") or {}).get("output_tokens")
    return "".join(b.get("text", "")
                   for b in out.get("content", [])).strip()


def _cap(options: dict = None) -> int:
    """The tokens one request may write: the cap ([llm] max_tokens), or
    the seat's own num_predict ([narrate] num_predict) when lower."""
    cap = max_tokens()
    try:
        n = int((options or {}).get("num_predict") or 0)
    except (TypeError, ValueError):
        n = 0
    return min(cap, n) if n > 0 else cap


def _fences(text: str) -> str:
    """A completion wrapped whole in a ``` fence, unwrapped (small local
    models fence their answer)."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```") and len(t) > 6:
        t = t[3:-3]
        t = t.split("\n", 1)[1] if "\n" in t and not t.split("\n", 1)[0].strip().count(" ") \
            else t
    return t.strip()


def _ollama(prompt: str, model: str, url: str, timeout: int,
            options: dict = None, system: str = None,
            meta: dict = None) -> str:
    """Ollama's /api/generate, one shot, no stream, think off (a thinking
    model answers without thinking first). The seat's options ([narrate]:
    temperature, num_ctx, repeat_penalty, ...) go over a temperature-0
    default, as before loop-workspace; num_predict is the cap (_cap)."""
    import urllib.error
    opts = {"temperature": 0, **(options or {}), "num_predict": _cap(options)}
    try:
        out = _post(url + "/api/generate", {
            "model": model, "prompt": prompt, "stream": False, "think": False,
            **({"system": system} if system else {}), "options": opts,
        }, {}, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:              # a model not pulled, not a dead server
            raise RuntimeError(f"ollama: model '{model}' is not pulled at {url} "
                               f"(ollama pull {model}, or `llm ollama MODEL`)") from e
        raise RuntimeError(f"ollama at {url}: HTTP {e.code}") from e
    except OSError as e:
        raise RuntimeError(f"ollama at {url} not reachable: "
                           f"{getattr(e, 'reason', e)}") from e
    if meta is not None:
        meta["cut"] = out.get("done_reason") == "length"
        meta["tokens"] = out.get("eval_count")
    return _fences(str(out.get("response", "")))


def _openrouter(prompt: str, model: str, timeout: int,
                options: dict = None, system: str = None,
                meta: dict = None, seat: str = "narrate") -> str:
    """OpenRouter's chat completions: the key from the environment only,
    the token cap on every request, reasoning off (a thinking model spent
    the whole cap thinking: the smoke of 2026-11-04, 700 tokens for two
    told sentences), the API's cost into the cost log."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (the key is read "
                           "from the environment only)")
    url = os.environ.get("OPENROUTER_API_URL", OPENROUTER_URL).rstrip("/")
    messages = ([{"role": "system", "content": system}] if system else []) \
        + [{"role": "user", "content": prompt}]
    import urllib.error
    try:
        out = _post(url + "/chat/completions", {
            "model": model, "messages": messages, "max_tokens": _cap(options),
            **({"temperature": options["temperature"]}
               if options and "temperature" in options else {}),
            "usage": {"include": True},
            "reasoning": {"enabled": False},      # think off, as Ollama's: the cap is the told text's
        }, {"Authorization": "Bearer " + key, "HTTP-Referer": "https://metis-lang.dev",
            "X-Title": "metis REPL"}, timeout)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")
            body = str((json.loads(body).get("error") or {}).get("message") or body)
        except Exception:  # noqa: BLE001 — the status alone then
            body = ""
        raise RuntimeError(f"openrouter: HTTP {e.code} {body[:200]}".rstrip()) from e
    except OSError as e:
        raise RuntimeError(f"openrouter not reachable: {getattr(e, 'reason', e)}") from e
    if out.get("error"):
        raise RuntimeError(f"openrouter: {str(out['error'])[:300]}")
    choice = (out.get("choices") or [{}])[0]
    usage = out.get("usage") or {}
    log_cost({"who": "repl", "seat": seat, "model": model,
              "tokens": usage.get("completion_tokens"),
              "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
              "prompt_tokens": usage.get("prompt_tokens"),
              "cost": usage.get("cost")})
    if meta is not None:
        meta["cut"] = choice.get("finish_reason") == "length"
        meta["tokens"] = usage.get("completion_tokens")
        meta["cost"] = usage.get("cost")
    msg = choice.get("message") or {}
    return str(msg.get("content") or "").strip()


# the seats of the one configuration: seat -> its key under [llm]
SEATS = {"narrate": "endpoint", "loop": "loop", "formalize": "formalize"}


def seat(name):
    """(spec, where) of the seat NAME from the dot file's [llm]: formalize falls back to the loop's ("" when unset)."""
    from . import config
    key = SEATS[name]
    spec = str(config.get("llm", key, "") or "").strip()
    if spec:
        return spec, f"[llm] {key}"
    if name == "formalize":
        spec = str(config.get("llm", "loop", "") or "").strip()
        if spec:
            return spec, "[llm] loop (formalize's default)"
    return "", None


def unset_line(name):
    """What an unset seat says: how to set it (no model is called that the person did not name)."""
    if name == "narrate":
        return UNSET
    return (f"no {name} model: `llm {name} ollama MODEL [URL]` or `llm {name} openrouter MODEL` in the REPL "
            f"(saved as [llm] {SEATS[name]} in the config dot file), or MODEL= for one start")


def pi_model(spec):
    """The model pi runs for SPEC: {spec, backend, model (pi's provider/id), paid, base_url?, id?}. ollama:M@URL is
    pi's provider `ollama` over the server's OpenAI-compatible URL/v1 (no key, cost 0); openrouter:M is pi's
    openrouter/M (the key and the budget, loop.sh's rules); anthropic[:M] is pi's anthropic/M. A command spec has no
    pi model: refused."""
    ep = resolve(spec)
    if ep.backend == "ollama":
        return {"spec": ep.spec, "backend": "ollama", "model": f"ollama/{ep.arg}", "id": ep.arg,
                "base_url": ep.url + "/v1", "paid": False}
    if ep.backend == "openrouter":
        return {"spec": ep.spec, "backend": "openrouter", "model": f"openrouter/{ep.arg}", "id": ep.arg, "paid": True}
    if ep.backend == "anthropic":
        m = ep.arg or ANTHROPIC_DEFAULT_MODEL
        return {"spec": ep.spec, "backend": "anthropic", "model": f"anthropic/{m}", "id": m, "paid": True}
    raise ValueError(f"{spec}: a command has no pi model: the pi loop runs on ollama, openrouter or anthropic")


def ollama_show(ep, timeout=2):
    """Ollama's /api/show for the endpoint's model: its capabilities (completion, tools, thinking, vision ...).
    OSError when the server does not answer; {} when it does not know the model."""
    import urllib.error
    try:
        return _post(ep.url + "/api/show", {"model": ep.arg}, {}, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {}
        raise


def main(argv=None):
    """python -m metis.loop.llm pi [loop|formalize] — the seat's model for loop.sh, as JSON on stdout ({seat, where,
    spec, backend, model, id, paid, base_url?}); unset or a spec pi cannot run: the line on stderr, exit 2."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m metis.loop.llm")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pi", help="the seat's pi model, from the dot file's [llm]")
    p.add_argument("seat", nargs="?", default="loop", choices=("loop", "formalize"))
    o = ap.parse_args(argv)
    spec, where = seat(o.seat)
    if not spec:
        print(unset_line(o.seat), file=sys.stderr)
        return 2
    try:
        out = pi_model(spec)
    except ValueError as e:
        print(f"[llm] {SEATS[o.seat]} = {spec!r}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"seat": o.seat, "where": where, **out}))
    return 0


def complete(spec, prompt: str, timeout: int = 600,
             options: dict = None, system: str = None,
             meta: dict = None) -> str:
    """One completion from the configured endpoint. `options` are
    sampling knobs (temperature, ...); the command backend ignores them.
    `system` rides the backend's system slot; the command backend,
    which has none, gets it prepended to the prompt. `meta`, when
    given, says whether the endpoint stopped at its token cap (cut)
    and how many tokens it wrote; the command backend cannot tell
    (cut False)."""
    ep = spec if isinstance(spec, Endpoint) else resolve(spec)
    if meta is not None:
        meta.setdefault("cut", False)
    if ep.backend == "anthropic":
        return _anthropic(prompt, ep.arg, timeout, options, system, meta)
    if ep.backend == "ollama":
        return _ollama(prompt, ep.arg, ep.url, timeout, options, system, meta)
    if ep.backend == "openrouter":
        return _openrouter(prompt, ep.arg, timeout, options, system, meta)
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


if __name__ == "__main__":
    import sys
    sys.exit(main())
