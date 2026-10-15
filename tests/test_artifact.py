"""The domain artifact (metis/kernel/artifact.py, runtime-binding step 2):
ground.wire + sidecar.json (+ program.ir) of the served program — the
program plus its packs, merged in order.

Goldens pin the kitchen pair's artifact directories (tests/goldens/
artifact/<n>.metis); the dog's and the plant's are in
test_artifact_catalog.py (they read metis-catalog). Regenerate
deliberately with METIS_BLESS=1 (and review the diff)."""

import json
import os
from pathlib import Path

import pytest

from metis.kernel.artifact import build, main, write

ROOT = Path(__file__).resolve().parents[1]
GOLDENS = Path(__file__).parent / "goldens" / "artifact"
KITCHEN = ROOT / "metis" / "domains" / "kitchen" / "catalogs"

LAMP = """\
% lamp — the artifact emitter's fixture: a program and two packs
catalog lamp 1.
provenance text "a lamp the person switches" "2026-10-15".
layers (dyn).
stages (main).

namespace lamp produce (dyn).

pred off : lamp.
pred on : lamp.
pred cmd_on : lamp.
pred broken : lamp.

stage main {
  %% You switch me on.
  switch_on [dyn] : off * cmd_on -o on.
  %% My bulb breaks.
  burn_out [dyn] : off * cmd_on -o broken.
}

#init off 1
#init cmd_on 1
#steps 2
#query on
"""
FLICKER = """\
catalog flicker 1.
extends lamp.
stage main {
  %% I flicker off.
  flicker [dyn] : on -o off.
}
"""
SURGE = """\
catalog surge 1.
extends lamp.
stage main {
  %% A surge switches me on.
  surge [dyn] : off -o on.
}
"""


@pytest.fixture
def lamp(tmp_path):
    (tmp_path / "packs").mkdir()
    (tmp_path / "lamp.llp").write_text(LAMP)
    (tmp_path / "packs" / "a_flicker.llp").write_text(FLICKER)
    (tmp_path / "packs" / "b_surge.llp").write_text(SURGE)
    return tmp_path


def _side(files):
    return json.loads(files["sidecar.json"])


def _wire_events(text):
    return [ln.split()[4] for ln in text.splitlines() if ln.startswith("E ")]


def test_every_pack_is_served_in_order(lamp):
    """Two packs: both merged, the second over the first (each pack is
    compiled against the world so far), the key the union's."""
    one = build(lamp / "lamp.llp", [lamp / "packs" / "a_flicker.llp"])
    two = build(lamp / "lamp.llp", [lamp / "packs" / "a_flicker.llp",
                                    lamp / "packs" / "b_surge.llp"])
    s = _side(two)
    assert s["catalog"] == "lamp+flicker+surge"
    assert s["packs"] == ["a_flicker.llp", "b_surge.llp"]
    assert [e["rule"] for e in s["events"]] == \
        ["switch_on", "burn_out", "flicker", "surge"]
    assert {e["rule"]: e["line"] for e in s["events"]}["surge"] == \
        "A surge switches me on."
    assert s["key"] != _side(one)["key"]


def test_the_sidecar_is_keyed_to_the_wire(lamp):
    files = build(lamp / "lamp.llp", [lamp / "packs" / "a_flicker.llp"])
    s = _side(files)
    assert s["format"] == "metis-sidecar/1"
    assert [e["name"] for e in s["events"]] == \
        _wire_events(files["ground.wire"])
    assert [e["i"] for e in s["events"]] == list(range(len(s["events"])))
    assert s["stages"] == ["main"]
    assert s["init"] == {"cmd_on": 1, "off": 1} and s["steps"] == 2
    assert s["queries"] == ["on"]
    assert s["ir"] is None and "program.ir" not in files
    # the served card first; no file paths travel in the artifact
    assert s["cards"][0]["relation"] == "served"
    assert "file" not in json.dumps(s["cards"])
    assert [c["name"] for c in s["script"][0]["clauses"]] == \
        ["switch_on", "burn_out"]


def test_a_pack_that_does_not_admit_raises(lamp):
    bad = lamp / "packs" / "c_bad.llp"
    bad.write_text("catalog bad 1.\nextends lamp.\nnamespace lamp produce "
                   "(dyn).\npred on : lamp.\n")
    with pytest.raises(Exception, match="redeclares"):
        build(lamp / "lamp.llp", [bad])


def test_program_ir_on_request(lamp):
    files = build(lamp / "lamp.llp", with_ir=True)
    ir = files["program.ir"].splitlines()
    assert ir[0] == "C lamp" and "A on on#0" in ir and ir[-1] == "."
    assert _side(files)["ir"] == "program.ir"


def test_write_replaces_the_directory_whole(tmp_path):
    out = tmp_path / "x.metis"
    write(out, {"a.txt": "1\n", "b.txt": "2\n"})
    write(out, {"a.txt": "3\n"})
    assert sorted(p.name for p in out.iterdir()) == ["a.txt"]
    assert (out / "a.txt").read_text() == "3\n"
    assert [p.name for p in tmp_path.iterdir()] == ["x.metis"]


def test_the_cli_writes_the_artifact(lamp, capsys):
    out = lamp / "art"
    assert main([str(lamp / "lamp.llp"), "-o", str(out),
                 "--pack", str(lamp / "packs" / "b_surge.llp")]) == 0
    assert sorted(p.name for p in out.iterdir()) == \
        ["ground.wire", "sidecar.json"]
    assert _side({"sidecar.json": (out / "sidecar.json").read_text()})[
        "packs"] == ["b_surge.llp"]
    assert "lamp+surge" in capsys.readouterr().out


def check_golden(name, files):
    d = GOLDENS / f"{name}.metis"
    if os.environ.get("METIS_BLESS"):
        write(d, files)
    have = sorted(p.name for p in d.iterdir())
    assert have == sorted(files), f"{name}: files {have}"
    for f, text in files.items():
        assert text == (d / f).read_text(), \
            f"{name}.metis/{f} drifted (METIS_BLESS=1 to regenerate, " \
            "then review the diff)"


@pytest.mark.parametrize("name, scene", [
    ("kitchen", KITCHEN / "kitchen_host.llp"),
    ("kitchen_shared", KITCHEN / "shared" / "kitchen_shared.llp")])
def test_the_kitchen_pair_golden(name, scene):
    """The pair's two cooks as their domain agents serve them (metis/
    domains/kitchen/pair.py SCENES): no packs, the weight ports in the
    wire's B manifest and w sections, the rules' lines from the include."""
    files = build(scene)
    s = _side(files)
    assert s["catalog"] == name
    assert all(e["line"] for e in s["events"])
    assert "B weight w_load" in files["ground.wire"]
    check_golden(name, files)
