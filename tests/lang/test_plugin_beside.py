"""A catalog whose weight ports no plugin binds: the error names the
plugin beside it by convention (<stem>_plugin.py, <stem>/<stem>_plugin.py)
and a fitted overlay <stem>.notch.json beside it; `notch load FILE` ships
that overlay as `notch accept` ships a fit (the user's wildfire finding)."""
import io
import os
from pathlib import Path

import pytest

from metis.fw.registry import PortError
from metis.lang.repl import Repl

RDDL = Path(__file__).resolve().parents[3] / "metis-catalog/domains/rddl"
WILD = RDDL / "wildfire.llp"

pytestmark = pytest.mark.skipif(not (WILD.exists() and (RDDL / "wildfire.notch.json").exists()),
                                reason="metis-catalog's rddl wildfire not checked out")


def bare_copy(tmp_path):
    """wildfire.llp without its #plugin line, its plugin folder and overlay beside it."""
    d = tmp_path / "rddl"
    d.mkdir()
    (d / "wildfire.llp").write_text("".join(
        ln for ln in WILD.read_text().splitlines(keepends=True) if not ln.startswith("#plugin")))
    os.symlink(RDDL / "wildfire", d / "wildfire")
    os.symlink(RDDL / "wildfire.notch.json", d / "wildfire.notch.json")
    return d / "wildfire.llp"


def test_unbound_port_names_the_plugin_beside(tmp_path):
    with pytest.raises(PortError) as e:
        Repl(bare_copy(tmp_path), seed=1, terminal=True, stdout=io.StringIO())
    msg = " ".join(e.value.findings)
    assert "metis --plugin wildfire/wildfire_plugin.py:make wildfire.llp" in msg
    assert "'#plugin wildfire/wildfire_plugin.py:make'" in msg
    assert "notch load wildfire.notch.json" in msg
    assert "pkg.mod:factory" not in msg


def test_unbound_port_without_a_plugin_keeps_the_generic_hint(tmp_path):
    p = bare_copy(tmp_path)
    (p.parent / "wildfire").unlink()
    with pytest.raises(PortError) as e:
        Repl(p, seed=1, terminal=True, stdout=io.StringIO())
    assert "pkg.mod:factory" in " ".join(e.value.findings)


def test_notch_load_ships_the_overlay_and_pop_drops_it():
    out = io.StringIO()
    r = Repl(WILD, seed=1, terminal=True, stdout=out)

    def say(line):
        out.truncate(0)
        out.seek(0)
        r.onecmd(line)
        return out.getvalue()
    assert r.registry is not None
    assert "overlay wildfire.notch.json shipped on pign" in say("notch load wildfire.notch.json")
    assert len(r._overlays) == 1
    assert "no overlay file" in say("notch load nothing.json")
    assert "dropped (0 left)" in say("notch pop")
