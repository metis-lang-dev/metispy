"""The live goldens (runtime-binding K1 + K2; tools/gen_live_goldens.py):
metispy's live REPL, seeded, on the catalog domains' served programs with
a scripted host, every row and the counts after it — what metis::Live
(metis-lang runtime/include/metis/live.hpp) must reproduce byte for byte.

  fast: the REPL still writes the committed goldens (METIS_BLESS=1
        regenerates); the golden's kitchen bindings are the host's
  slow: metis-lang's live_run replays each golden on the artifact
        golden's ground.wire (the C++ loader loads every emitted wire,
        the architect's step-2 ask) — LIVE PARITY n/n

Reads metis-catalog (the domains; CI: CATALOG_TESTS)."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gen_live_goldens as gen  # noqa: E402
from langpath import LANG  # noqa: E402

GOLDENS = Path(__file__).parent / "goldens"
CXX = shutil.which("g++") or shutil.which("clang++")
HEAD = "# metis live golden — tools/gen_live_goldens.py; METIS_BLESS=1 regenerates\n"


@pytest.mark.parametrize("name", sorted(gen.DOMAINS))
def test_the_repl_writes_the_golden(name):
    _, golden = gen.domain_trace(name)
    p = GOLDENS / "live" / f"{name}.txt"
    if os.environ.get("METIS_BLESS"):
        p.parent.mkdir(exist_ok=True)
        p.write_text(HEAD + golden)
    assert HEAD + golden == p.read_text(), \
        f"live/{name}.txt drifted (METIS_BLESS=1 to regenerate, then review)"


def test_the_goldens_kitchen_bindings_are_the_hosts():
    """gen.KITCHEN mirrors metis/domains/kitchen/host.py _weights."""
    from metis.domains.kitchen.host import _weights
    from metis.fw.registry import Registry
    reg = Registry()
    _weights(reg)
    host = reg.weight_bindings()
    assert sorted(host) == sorted(b[0] for b in gen.KITCHEN)
    for spec in gen.KITCHEN:
        mine = gen.py_binding(spec)
        for counts in ({}, {"cooking": 1}, {"cooking": 2, "hand_free": 1}):
            want = host[spec[0]]((), counts)
            got = mine((), counts) if callable(mine) else mine
            assert got == want, (spec, counts)


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    exe = tmp_path_factory.mktemp("live") / "live_run"
    subprocess.run([CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
                    str(LANG / "runtime/tools/live_run.cpp"), "-o", str(exe)],
                   check=True)
    return exe


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
@pytest.mark.parametrize("name", sorted(gen.DOMAINS))
def test_metis_live_replays_the_golden(name, live_run):
    import json
    art = GOLDENS / "artifact" / f"{name}.metis"
    r = subprocess.run([str(live_run), str(art / "ground.wire"),
                        str(GOLDENS / "live" / f"{name}.txt")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"LIVE PARITY (\d+)/(\d+) \(\S+ seed \d+, (\d+) events\)",
                  r.stdout)
    assert m and m.group(1) == m.group(2) and int(m.group(2)) > 0, r.stdout
    # the C++ loader loads the emitted wire whole: every sidecar event
    side = json.loads((art / "sidecar.json").read_text())
    assert int(m.group(3)) == len(side["events"]), r.stdout
