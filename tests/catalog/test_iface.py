"""metis.catalog.iface (specialist-tree s-404d001f; metis-lang spec/15): the generic interface package — the
protocol module, its natures, included by its package path (the serves form) from any source; the manifest's
gating field."""
import io
import re

import pytest
import yaml

from metis.catalog import iface
from metis.catalog import manifests as mc


def test_a_source_includes_the_protocol_from_metis_home(tmp_path):
    from metis.lang.compiler import load_llp
    p = tmp_path / "member.llp"
    p.write_text('catalog member 1.\ninclude "iface/protocol_domain.llp".\n'
                 "type domain {nav cook}.\ntype resource {motion pot}.\n")
    preds = set(load_llp(p).catalog.preds)                         # found under METIS_HOME (metispy's own): no copy
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


def test_an_include_is_found_in_three_places_in_order(tmp_path, monkeypatch):
    """(1) beside the including file, (2) under the catalog root, (3) under METIS_HOME; a miss names all three."""
    from metis.lang.compiler import include_path
    own, root, home = tmp_path / "own", tmp_path / "root", tmp_path / "home"
    for d in (own, root, home):
        (d / "iface").mkdir(parents=True)
    monkeypatch.setenv("METIS_CATALOG", str(root))
    monkeypatch.setenv("METIS_HOME", str(home))
    (home / "iface" / "p.llp").write_text("x")
    assert include_path("iface/p.llp", own) == (home / "iface" / "p.llp").resolve()
    (root / "iface" / "p.llp").write_text("x")
    assert include_path("iface/p.llp", own) == (root / "iface" / "p.llp").resolve()
    (own / "iface" / "p.llp").write_text("x")
    assert include_path("iface/p.llp", own) == (own / "iface" / "p.llp").resolve()
    with pytest.raises(FileNotFoundError, match="the including file's directory.*the catalog root.*METIS_HOME"):
        include_path("iface/none.llp", own)
    monkeypatch.delenv("METIS_HOME")                                # default: metispy's own metis/catalog
    assert include_path("iface/protocol_domain.llp", own) == iface.path()


def _tree(tmp_path, priority=()):
    from metis.lang.repl import Repl
    p = tmp_path / "kitchen_director.llp"
    p.write_text("\n".join(['catalog kd 1.', 'provenance text "test" "2026-10-25".', 'include "iface/director_tree_domain.llp".',
                            "type domain {a b c}.", "type resource {pot}.", "mature_grant(a1).", "mature_grant(a2).",
                            "mature_grant(a3).", "young_grant(a0).", "mature_status(a1).", "mature_status(a2).",
                            "mature_status(a3)."] + ["priority(%s, %s)." % pq for pq in priority]
                           + ["#init free(pot) 1", "#init issue(pot,t0) 1", "#init serving(pot,t0) 1"]) + "\n")
    r = Repl(str(p), seed=1)
    r.stdout = io.StringIO()
    return r


def _do(r, *atoms):
    for a in atoms:
        r.onecmd("do %s 1" % a)
    r.stage = r.prog.init_stage                                     # a boundary revives the quiet director (the tick)
    r.run_steps(50)
    return sorted(a for a, n in r.counts.items() if n > 0 and a.startswith("holder("))


def test_the_tree_serves_requests_first_asked_first_served(tmp_path):
    r = _tree(tmp_path)
    assert _do(r, "+tick__want(b,pot)") == ["holder(b,pot)"]
    assert _do(r, "+tick__want(c,pot)") == ["holder(b,pot)"]          # c waits with ticket t1
    assert _do(r, "+tick__want(a,pot)") == ["holder(b,pot)"]          # a with t2, behind c
    assert _do(r, "+tick") == ["holder(b,pot)"]                        # b's grant ages a tick (hysteresis k = 1)
    assert _do(r, "+tick__release(b,pot)") == ["holder(c,pot)"]       # c's turn, not a's
    _do(r, "+tick")
    assert _do(r, "+tick__release(c,pot)") == ["holder(a,pot)"]


def test_a_priority_preempts_and_its_ticket_is_skipped(tmp_path):
    r = _tree(tmp_path, priority=[("a", "b")])
    assert _do(r, "+tick__want(b,pot)") == ["holder(b,pot)"]          # b: t0
    assert _do(r, "+tick__want(c,pot)") == ["holder(b,pot)"]          # c: t1
    assert _do(r, "+tick__want(a,pot)") == ["holder(a,pot)"]          # a: t2, preempts b (a young grant: the exception)
    assert r.counts.get("queued(b,pot,t3)") == 1 and r.counts.get("skipped(pot,t2)") == 1
    _do(r, "+tick")
    assert _do(r, "+tick__release(a,pot)") == ["holder(c,pot)"]       # c's t1 was serving
    _do(r, "+tick")
    assert _do(r, "+tick__release(c,pot)") == ["holder(b,pot)"]       # t2 skipped, b's t3
