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


def test_seeded_narrate_trace_is_priced_by_the_session():
    """`narrate SEED` builds its trace with the session's ports: flat while
    the base binding is prior-flat (pign = 0.5), the fitted law once the
    overlay is loaded — the same pricing `run` uses (the narrate-bug
    report: 34/80 k0 ignitions came from the un-notched world)."""
    import re

    def k0_fires(r):
        fired = total = 0
        for seed in range(1, 21):
            rows, _, _ = r._seeded_trace(seed)
            for _, _, ev in rows:
                m = re.match(r"catch-(\d)\[.*K=k0\]", ev.name)
                if m:
                    total += 1
                    fired += m.group(1) == "1"
        return fired, total
    flat = Repl(WILD, seed=1, terminal=True, stdout=io.StringIO())
    f0, n0 = k0_fires(flat)
    fitted = Repl(WILD, seed=1, terminal=True, stdout=io.StringIO())
    fitted.onecmd("notch load wildfire.notch.json")
    f1, n1 = k0_fires(fitted)
    assert n0 and n1
    assert f0 / n0 > 0.25          # pign(k0) = 0.5 under the base binding
    assert f1 / n1 < 0.05          # the fitted law: pign(k0) ≈ 0.009


def directed_copy(tmp_path):
    """wildfire.llp with '#notch wildfire.notch.json' beside its #plugin line."""
    d = tmp_path / "rddl"
    d.mkdir()
    (d / "wildfire.llp").write_text(WILD.read_text() + "\n#notch wildfire.notch.json\n")
    os.symlink(RDDL / "wildfire", d / "wildfire")
    os.symlink(RDDL / "wildfire.notch.json", d / "wildfire.notch.json")
    return d / "wildfire.llp"


def test_notch_directive_ships_at_serve_and_pop_drops_it(tmp_path):
    """#notch FILE ships the fitted overlay at serve, as `notch load` does;
    the pricing is named in run's footer, the ctx band and narrate user;
    `notch pop` drops it back to the base binding."""
    out = io.StringIO()
    r = Repl(directed_copy(tmp_path), seed=1, terminal=True, stdout=out)

    def say(line):
        out.truncate(0)
        out.seek(0)
        r.onecmd(line)
        return out.getvalue()
    assert r._overlay_names == ["wildfire.notch.json"]
    assert "priced by wildfire-prior-flat; overlays: wildfire.notch.json" in say("ctx")
    assert "(priced by wildfire-prior-flat; overlays: wildfire.notch.json)" in say("run 3")
    assert "overlays: wildfire.notch.json" in say("narrate user 1")
    say("notch pop")
    assert "priced by wildfire-prior-flat; overlays: none" in say("ctx")


def test_without_the_directive_the_base_binding_prices():
    out = io.StringIO()
    r = Repl(WILD, seed=1, terminal=True, stdout=out)
    out.truncate(0)
    out.seek(0)
    r.onecmd("ctx")
    assert "priced by wildfire-prior-flat; overlays: none" in out.getvalue()
