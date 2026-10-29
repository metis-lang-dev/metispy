"""The language's standard units (metis-lang spec/17 amendment 5): metis-lang's iface/ is authoritative, metispy's
metis/catalog/iface/ its copy byte for byte, so every key of the four units (and the protocol's natures) is metis-lang's
at the checkout CI builds metisc from. A failure here: the vendored copy is stale (copy metis-lang's iface/ over it).
And both toolchains resolve an iface/ include to the standard place with no METIS_HOME."""
import os
import subprocess
from pathlib import Path

import pytest

from metis import catalog
from langpath import LANG

UNITS = ["protocol_domain.llp", "director_domain.llp", "director_tree_domain.llp", "director_tree_wake.llp",
         "protocol_domain.yaml"]


@pytest.mark.skipif(not (LANG / "iface").is_dir(), reason=f"no metis-lang iface/ at {LANG} (METIS_LANG)")
@pytest.mark.parametrize("unit", UNITS)
def test_the_vendored_unit_is_metis_langs(unit):
    ours, theirs = catalog.std() / "iface" / unit, LANG / "iface" / unit
    assert ours.read_bytes() == theirs.read_bytes(), \
        f"metis/catalog/iface/{unit} differs from {theirs}: the vendored copy is stale"


def test_the_vendored_units_are_all_of_them():
    if (LANG / "iface").is_dir():
        assert sorted(p.name for p in (LANG / "iface").iterdir()) == sorted(UNITS)
    assert sorted(p.name for p in (catalog.std() / "iface").iterdir() if p.suffix in (".llp", ".yaml")) == sorted(UNITS)


def test_an_iface_include_resolves_to_the_standard_units_without_metis_home(tmp_path, monkeypatch):
    monkeypatch.delenv("METIS_HOME", raising=False)
    monkeypatch.setenv("METIS_CATALOG", str(tmp_path))
    from metis.lang.compiler import include_path
    assert include_path("iface/protocol_domain.llp", tmp_path) == catalog.std() / "iface/protocol_domain.llp"
    assert [why for _, why in catalog.places()] == ["the catalog root", "the standard units", "METIS_HOME"]
    with pytest.raises(FileNotFoundError, match="the standard units"):
        include_path("iface/nowhere.llp", tmp_path)
    monkeypatch.setenv("METIS_HOME", str(tmp_path / "home"))
    with pytest.raises(FileNotFoundError, match="METIS_HOME") as e:
        catalog.find("image.py")                        # the standard units hold iface/ only
    assert "the standard units" not in str(e.value)     # nor are they named for another path
    monkeypatch.delenv("METIS_HOME")
    (tmp_path / "iface").mkdir()
    (tmp_path / "iface/protocol_domain.llp").write_text("catalog mine 1.\n")
    assert include_path("iface/protocol_domain.llp", tmp_path / "x") == tmp_path / "iface/protocol_domain.llp"   # the root first


@pytest.mark.skipif(not catalog.metisc(), reason="no metisc (METISC or PATH)")
def test_metisc_resolves_it_the_same_without_metis_home(tmp_path):
    (tmp_path / "p.llp").write_text('catalog p 1.\ninclude "iface/protocol_domain.llp".\n')
    env = {k: v for k, v in os.environ.items() if k != "METIS_HOME"}
    r = subprocess.run([catalog.metisc(), "--diagnostics-json", str(tmp_path / "p.llp")],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    (tmp_path / "q.llp").write_text('catalog q 1.\ninclude "iface/nowhere.llp".\n')
    r = subprocess.run([catalog.metisc(), "--diagnostics-json", str(tmp_path / "q.llp")],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 1 and "nowhere.llp" in r.stderr + r.stdout
