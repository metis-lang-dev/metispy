"""Port plugins — real values and time behind a qualitative catalog
(metis-loop docs/PLUGIN-PORTS.md). The Python prototyping twin of the
theia C++ ports, under the SAME contract, so a prototyped domain
transfers by reimplementing bindings only:

    class PortPlugin(Protocol):
        def bind(self, reg: Registry) -> None
        def tick(self, counts: dict) -> list[str] | None
        def on_event(self, name: str, args: tuple) -> None
        def snapshot(self) -> object
        def restore(self, snap: object) -> None

PURITY: a weight binding is a function of (args, declared-scope
counts, env snapshot) — nothing else; env state mutates ONLY in tick
and on_event; snapshot is total. That pair is the determinism
contract deterministic replay needs.

tick() is the INPUT face: the env advances one world beat and returns
do-lines ("+atom [n]" / "-atom [n]"); [] is a quiet beat (time passed,
nothing arrived); None ends the env's episode. on_event() is the
OUTPUT face: a fired event actuates the env. Optional attributes:
`beat_stage` (tick on entry to that stage — the settle stage of a
tick-stage catalog — besides at quiescence) and `name`.

Attachment is operator code: TERMINAL ONLY (D4 — the web/shared
console never imports it). Generic like the rest of fw: never imports
metis.domains."""

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .registry import Registry


@runtime_checkable
class PortPlugin(Protocol):
    def bind(self, reg: Registry) -> None: ...

    def tick(self, counts: dict) -> Optional[List[str]]: ...

    def on_event(self, name: str, args: tuple) -> None: ...

    def snapshot(self) -> object: ...

    def restore(self, snap: object) -> None: ...


class PluginError(ValueError):
    pass


_REQUIRED = ("bind", "tick", "on_event", "snapshot", "restore")


def check_plugin(obj) -> None:
    """The protocol, checked up front (fail fast, BT.cpp rule): every
    face present and callable — snapshot/restore included, since
    fork/back/undo/whatif LIE without them."""
    missing = [m for m in _REQUIRED if not callable(getattr(obj, m, None))]
    if missing:
        raise PluginError(f"{type(obj).__name__} is not a PortPlugin: "
                          f"missing {', '.join(missing)}")


def load_plugin(spec: str, case: dict, base_dir=None):
    """`package.module:factory` (or `path/to/file.py:factory`, relative
    to `base_dir` — the catalog's folder) -> factory(case), checked.
    The catalog's folder is importable, so a plugin can live beside
    its catalog."""
    mod_name, sep, fac = spec.strip().partition(":")
    if not sep or not mod_name or not fac:
        raise PluginError(f"plugin spec {spec!r}: want "
                          "package.module:factory or file.py:factory")
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    try:
        if mod_name.endswith(".py"):
            path = (base / mod_name).resolve()
            name = f"metis_plugin_{path.stem}_{abs(hash(str(path)))}"
            sp = importlib.util.spec_from_file_location(name, path)
            if sp is None or sp.loader is None:
                raise PluginError(f"no plugin file {path}")
            mod = importlib.util.module_from_spec(sp)
            sp.loader.exec_module(mod)
        else:
            if str(base) not in sys.path:
                sys.path.insert(0, str(base))
            mod = importlib.import_module(mod_name)
            mod = importlib.reload(mod)      # `plugin SPEC` re-reads it
    except PluginError:
        raise
    except Exception as e:
        raise PluginError(f"cannot import plugin {mod_name!r}: "
                          f"{type(e).__name__}: {e}") from e
    factory = getattr(mod, fac, None)
    if not callable(factory):
        raise PluginError(f"plugin module {mod_name!r} has no factory "
                          f"{fac!r}")
    obj = factory(case)
    check_plugin(obj)
    return obj


# -- exogenous input: a do-line stream, as fired rows ---------------------

@dataclass
class ExoEvent:
    """One env beat's input as a fired row of the session (kind "exo"):
    the do-lines' net multiset edit as consume/produce, so prefix
    replay (fork/whatif/counts_at), the causality graph, `why` and
    `log` treat exogenous input exactly like an event — first-class,
    and never moving the graph base the way a hand `do` does."""
    name: str
    consume: Dict[str, int] = field(default_factory=dict)
    produce: Dict[str, int] = field(default_factory=dict)
    persist: Dict[str, int] = field(default_factory=dict)
    weight: float = 1.0
    ops: Tuple[str, ...] = ()


def parse_do_line(line: str) -> Tuple[str, str, int]:
    """'+atom [n]' / '-atom [n]' -> (sign, atom, n)."""
    s = line.strip()
    if s.startswith("do "):
        s = s[3:].strip()
    if not s or s[0] not in "+-":
        raise PluginError(f"tick do-line {line!r}: want +atom [n] or "
                          "-atom [n]")
    parts = s[1:].split()
    if not parts or len(parts) > 2:
        raise PluginError(f"tick do-line {line!r}: want +atom [n]")
    try:
        n = int(parts[1]) if len(parts) == 2 else 1
    except ValueError:
        raise PluginError(f"tick do-line {line!r}: count must be an "
                          "integer") from None
    if n < 0:
        raise PluginError(f"tick do-line {line!r}: negative count")
    return s[0], parts[0], n


def exo_event(lines, counts) -> Tuple[ExoEvent, Dict[str, int]]:
    """Apply a beat's do-lines to `counts` (in order; a retract clamps
    at what is present) -> (the row, the new counts). The row's net
    consume/produce replays the same edit exactly."""
    after = dict(counts)
    ops = []
    for line in lines:
        sign, atom, n = parse_do_line(line)
        have = after.get(atom, 0)
        after[atom] = have + n if sign == "+" else max(0, have - n)
        ops.append(f"{sign}{atom}" + (f" {n}" if n != 1 else ""))
    consume, produce = {}, {}
    for a in set(counts) | set(after):
        d = after.get(a, 0) - counts.get(a, 0)
        if d < 0:
            consume[a] = -d
        elif d > 0:
            produce[a] = d
    name = "do " + " ".join(ops) if ops else "tick"
    return (ExoEvent(name, consume, produce, ops=tuple(ops)),
            {a: n for a, n in after.items() if n > 0})


# -- env snapshots in records: kernel-level (record.py), re-exported
from metis.kernel.record import (env_blob, env_hash,  # noqa: F401,E402
                                 env_record, env_unblob)
