"""metis.catalog.iface (specialist-tree s-404d001f; metis-lang spec/15): the generic interface package — the
protocol module, its natures, included by its package path (the serves form) from any source; the manifest's
gating field."""
import re

import pytest
import yaml

from metis.catalog import iface
from metis.catalog import manifests as mc


def test_a_source_includes_the_protocol_by_its_package_path(tmp_path):
    from metis.lang.compiler import load_llp
    p = tmp_path / "member.llp"
    p.write_text('catalog member 1.\ninclude "metis.catalog:iface/protocol_domain.llp".\n'
                 "type domain {nav cook}.\ntype resource {motion pot}.\n")
    preds = set(load_llp(p).catalog.preds)                         # compiled through the package path, no copy
    assert {"tick__want", "tick__release", "tick__grant", "tick__status", "tick__revoke"} <= preds


def test_every_protocol_pred_has_its_nature():
    preds = set(re.findall(r"^pred (tick__\w+)\(", iface.path().read_text(), re.M))
    nature = yaml.safe_load(iface.path("protocol_domain.yaml").read_text())["nature"]
    assert preds == set(nature)
    assert {p for p, n in nature.items() if n == "report"} == {"tick__status", "tick__active", "tick__done", "tick__stuck",
                                                               "tick__want", "tick__release"}
    assert set(nature.values()) <= {"report", "signal", "resource"}


def test_copy_roots_the_package(tmp_path):
    out = iface.copy(tmp_path / "domains")
    assert out.read_text() == iface.path().read_text() and (tmp_path / "domains" / "protocol_domain.yaml").is_file()


@pytest.mark.parametrize("value, ok", [("host", True), ("program", True), ("rules", False), ("[host]", False)])
def test_the_gating_field(tmp_path, value, ok):
    d = tmp_path / "domains/cook"
    d.mkdir(parents=True)
    (d / "cook.llp").write_text("catalog cook 1.\n")
    (d / "domain.yaml").write_text("program: cook.llp\nshared: {pot: resource}\ngating: %s\n" % value)
    if ok:
        assert mc.get("cook", tmp_path).manifest["gating"] == value
    else:
        with pytest.raises(mc.ManifestError, match="gating is host or program"):
            mc.domains(tmp_path)


def test_metis_pkg_path_is_searched_first(tmp_path, monkeypatch):
    from metis.lang.compiler import include_path
    alt = tmp_path / "metis" / "catalog" / "iface"
    alt.mkdir(parents=True)
    (alt / "protocol_domain.llp").write_text(iface.path().read_text())
    monkeypatch.setenv("METIS_PKG_PATH", "/nowhere:%s" % tmp_path)
    assert include_path("metis.catalog:iface/protocol_domain.llp", "/x") == (alt / "protocol_domain.llp").resolve()
    monkeypatch.setenv("METIS_PKG_PATH", "/nowhere")
    assert include_path("metis.catalog:iface/protocol_domain.llp", "/x") == iface.path()   # else the imported package
    assert include_path("../a.llp", "/x/y") == __import__("pathlib").Path("/x/a.llp")
