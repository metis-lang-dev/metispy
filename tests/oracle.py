"""Shared access to the ll2pg reference oracle in chariot_ws/wayplan.
Tests that need it apply `requires_oracle` and skip cleanly elsewhere."""

import os
import sys
from pathlib import Path

import pytest

def _default_oracle():
    """First existing wayplan checkout across the known workspace layouts."""
    candidates = [Path.home() / base / "chariot_ws/packages/wayplan/py"
                  for base in ("repo", "repo/perotheia")]
    for p in candidates:
        if (p / "ll2pg").is_dir():
            return p
    return candidates[0]


ORACLE = Path(os.environ.get("METIS_ORACLE_PATH") or _default_oracle())

requires_oracle = pytest.mark.skipif(
    not (ORACLE / "ll2pg").is_dir(),
    reason=f"wayplan ll2pg oracle not found at {ORACLE}")


def oracle_modules(*names):
    """Import ll2pg submodules from the wayplan checkout.

    fol.py needs only Rule from ll.py, but ll.py imports pgmpy (the
    graph evaluators). Where pgmpy is absent, stub ll2pg.ll with the
    core re-export so the GROUNDING oracle stays usable — the graph
    evaluators are not what these tests reference."""
    sys.path.insert(0, str(ORACLE))
    try:
        import importlib
        if "fol" in names and "ll2pg.ll" not in sys.modules:
            try:
                importlib.import_module("pgmpy")
            except ImportError:
                import types
                core = importlib.import_module("ll2pg.core")
                stub = types.ModuleType("ll2pg.ll")
                stub.Rule = core.Rule
                sys.modules["ll2pg.ll"] = stub
        return [importlib.import_module(f"ll2pg.{n}") for n in names]
    finally:
        sys.path.pop(0)


def rss_modules(*names):
    """Import wayplan rss submodules (the road-domain oracle: sim,
    semantics, lldsl, planner). They pull ll2pg.forward/core only —
    no pgmpy."""
    sys.path.insert(0, str(ORACLE))
    try:
        import importlib
        return [importlib.import_module(f"rss.{n}") for n in names]
    finally:
        sys.path.pop(0)


def oracle_projection(program, steps):
    """Run the oracle forward filter on a metis Program (priors must all
    be 1 — parity mode)."""
    core, forward = oracle_modules("core", "forward")
    stages = {s: [core.Rule(c.name, consume=dict(c.consume),
                            produce=dict(c.produce),
                            persist=dict(c.persist))
                  for c in body]
              for s, body in program.stages.items()}
    links = [core.Link(lk.name, pre=lk.pre, post=lk.post,
                       consume=dict(lk.consume),
                       produce=dict(lk.produce),
                       persist=dict(lk.persist))
             for lk in program.links]
    return forward.ForwardProjection(stages, links, program.init_stage,
                                     dict(program.init), steps)
