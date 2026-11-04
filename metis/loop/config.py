"""The metis dot file — one small TOML, stdlib only (tomllib).

Search order (first hit wins, no merging):

  $METIS_CONFIG              an explicit file; set-but-EMPTY pins
                             the search to nothing (hermetic runs:
                             tests, CI, a clean demo)
  ./.metis.toml              the working directory (per-project)
  ~/.config/metis/config.toml   ($XDG_CONFIG_HOME respected)

Recognized today:

  [llm]
  endpoint = "ollama:phi3"     # ollama:MODEL[@URL] | openrouter:MODEL |
                               # anthropic[:MODEL] | command:ARGV;
                               # no default: unset is an error; the
                               # REPL's `llm` writes it here (set)
  max_tokens = 1024            # the per-request cap of every paid call
  [narrate]                    # narrate seat only; formalize stays
  temperature = 0.3            # at temperature 0

Keys never live here: a backend reads its key from the environment
($OPENROUTER_API_KEY, $ANTHROPIC_API_KEY; the command's own variable;
the console's gateway loads the person's private key file into its own
environment). `set` writes one key (the REPL's `llm`). Unknown keys are ignored, a broken file
is reported once and treated as absent — a dot file must never
stop the shell from serving."""
import os
import sys
from pathlib import Path

_cache = None            # (path or None, dict)


def _candidates():
    env = os.environ.get("METIS_CONFIG")
    if env is not None:
        if env:
            yield Path(env)
        return                    # set (even empty) pins the search
    yield Path(".metis.toml")
    xdg = os.environ.get("XDG_CONFIG_HOME",
                         str(Path.home() / ".config"))
    yield Path(xdg) / "metis" / "config.toml"


def load(reload: bool = False):
    """(path or None, dict) — cached after the first read."""
    global _cache
    if _cache is not None and not reload:
        return _cache
    import tomllib
    for p in _candidates():
        try:
            if not p.is_file():
                continue
            with open(p, "rb") as f:
                _cache = (p, tomllib.load(f))
            return _cache
        except (OSError, tomllib.TOMLDecodeError) as e:
            print(f"metis: ignoring config {p}: {e}", file=sys.stderr)
            _cache = (p, {})
            return _cache
    _cache = (None, {})
    return _cache


def get(section: str, key: str, default=None):
    _, conf = load()
    v = conf.get(section)
    return v.get(key, default) if isinstance(v, dict) else default


def path():
    return load()[0]


def set(section: str, key: str, value):
    """Write KEY = VALUE under [SECTION] in the dot file the search finds
    (else the user's ~/.config/metis/config.toml), as text: the line
    replaced where it is, added under the section, or the section
    appended; every other line stays. A pinned-empty $METIS_CONFIG
    refuses (nothing to write to). → the path written."""
    p = path()
    if p is None:
        env = os.environ.get("METIS_CONFIG")
        if env is not None and not env:
            raise ValueError("METIS_CONFIG is set and empty: no config file "
                             "to write to")
        if env:
            p = Path(env)
        else:
            xdg = os.environ.get("XDG_CONFIG_HOME",
                                 str(Path.home() / ".config"))
            p = Path(xdg) / "metis" / "config.toml"
    import json
    val = json.dumps(value) if isinstance(value, str) else \
        ("true" if value is True else "false" if value is False else str(value))
    lines = p.read_text().splitlines() if p.is_file() else []
    out, i, in_sec, done = [], 0, False, False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if in_sec and not done:          # the section ended without the key: add it
                out.append(f"{key} = {val}")
                done = True
            in_sec = stripped[1:-1].strip() == section
        elif in_sec and not done and stripped.split("=", 1)[0].strip() == key \
                and "=" in stripped and not stripped.startswith("#"):
            line = f"{key} = {val}"
            done = True
        out.append(line)
        i += 1
    if in_sec and not done:
        out.append(f"{key} = {val}")
        done = True
    if not done:
        if out and out[-1].strip():
            out.append("")
        out += [f"[{section}]", f"{key} = {val}"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out) + "\n")
    load(reload=True)
    return p
