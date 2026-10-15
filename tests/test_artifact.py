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
    # the pack's clauses follow the program's, tagged (ruling F3)
    assert [(c["name"], c.get("pack")) for c in s["script"][0]["clauses"]] \
        == [("switch_on", None), ("burn_out", None),
            ("flicker", "a_flicker.llp")]


def test_canon_is_the_samplers_order(lamp):
    """events[i].canon: wire event i's index in refsample.canonical_events
    (the law the live runtime samples by; the wire keeps its own order)."""
    from metis.kernel.artifact import served_doc
    from metis.kernel.refsample import canonical_events
    packs = [lamp / "packs" / "a_flicker.llp", lamp / "packs" / "b_surge.llp"]
    s = _side(build(lamp / "lamp.llp", packs))
    doc, case, _ = served_doc(lamp / "lamp.llp", packs)
    tbl = canonical_events(doc.catalog.ground(dict(case["init"])))
    assert sorted(e["canon"] for e in s["events"]) == \
        list(range(len(s["events"])))
    assert [tbl[e["canon"]][2].name for e in s["events"]] == \
        [e["name"] for e in s["events"]]
    # the lamp's canonical order differs from the wire's (bytewise lines)
    assert [e["canon"] for e in s["events"]] != list(range(len(s["events"])))


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


def test_write_swaps_a_link_to_a_versioned_directory(tmp_path):
    out = tmp_path / "x.metis"
    v1 = write(out, {"a.txt": "1\n", "b.txt": "2\n"})
    v2 = write(out, {"a.txt": "3\n"})
    assert out.is_symlink() and out.resolve() == v2.resolve()
    assert sorted(p.name for p in out.iterdir()) == ["a.txt"]
    assert (out / "a.txt").read_text() == "3\n"
    # the replaced version stays one swap longer, then goes
    assert v1.is_dir()
    write(out, {"a.txt": "4\n"})
    assert not v1.exists() and v2.is_dir()
    assert sorted(p.name for p in tmp_path.iterdir()
                  if not p.name.startswith("x.metis.v-")) == ["x.metis"]
    # the same content is the same version (no rewrite)
    assert write(out, {"a.txt": "4\n"}) == out.resolve()
    # R2: the directory's mode is the umask's, not mkdtemp's 0700
    m = os.umask(0o022)
    os.umask(m)
    assert out.resolve().stat().st_mode & 0o777 == 0o777 & ~m


def test_write_replaces_an_older_real_directory(tmp_path):
    out = tmp_path / "x.metis"
    out.mkdir()
    (out / "old.txt").write_text("old\n")
    write(out, {"a.txt": "1\n"})
    assert out.is_symlink() and sorted(p.name for p in out.iterdir()) == \
        ["a.txt"]


def test_a_reader_never_finds_no_artifact(tmp_path):
    """R1 (nina's probe): while the artifact is rewritten, `out` always
    resolves to a whole version, and both files come from it. A version
    is pruned two swaps after it was replaced, so a reader that resolved
    one and lost it to two swaps in between resolves again (counted, not
    a miss; 400 rewrites back to back are far denser than a reload)."""
    import threading
    out = tmp_path / "x.metis"
    write(out, {"ground.wire": "k 0\n", "sidecar.json": "k 0\n"})
    misses, mixed, again, stop = [0], [0], [0], threading.Event()

    def reader():
        while not stop.is_set():
            if not os.path.lexists(out):
                misses[0] += 1          # no artifact at all: R1's window
                continue
            try:
                d = out.resolve(strict=True)
            except OSError:
                again[0] += 1           # its target pruned: resolve again
                continue
            try:
                a = (d / "ground.wire").read_text()
                b = (d / "sidecar.json").read_text()
            except OSError:
                again[0] += 1           # pruned under the reader: resolve again
                continue
            if a != b:
                mixed[0] += 1
    t = threading.Thread(target=reader)
    t.start()
    for i in range(1, 400):
        write(out, {"ground.wire": f"k {i}\n", "sidecar.json": f"k {i}\n"})
    stop.set()
    t.join()
    assert misses[0] == 0 and mixed[0] == 0, (misses, mixed, again)


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
    if os.environ.get("METIS_BLESS"):           # plain files, committed
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
        for f, text in files.items():
            (d / f).write_text(text)
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
