"""The metis dot file — one small TOML, stdlib only (tomllib).

Search order (first hit wins, no merging):

  $METIS_CONFIG              an explicit file; set-but-EMPTY pins
                             the search to nothing (hermetic runs:
                             tests, CI, a clean demo)
  ./.metis.toml              the working directory (per-project)
  ~/.config/metis/config.toml   ($XDG_CONFIG_HOME respected)

Recognized today:

  [llm]
  endpoint = "ollama:qwen2.5-coder:7b"   # ollama[:MODEL] |
                                         # anthropic[:MODEL] |
                                         # command:ARGV
  [ollama]
  host  = "http://localhost:11434"
  model = "qwen2.5-coder:7b"
  think = false                # Ollama's think field (default false;
                               # true, or a level: "low", ...)
  [narrate]                    # narrate seat only; formalize stays
  temperature = 0.3            # at temperature 0
  num_ctx     = 8192

Environment variables keep the last word ($OLLAMA_HOST /
$OLLAMA_MODEL beat [ollama]); the config supplies defaults where
the environment is silent. Unknown keys are ignored, a broken file
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
