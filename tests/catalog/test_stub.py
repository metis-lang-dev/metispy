"""The interface stub (metis-lang spec/15 §10.4, specialist-tree s-404d001f step 1d): a specialist run ALONE
against its own cases, driven by its manifest — the in-copies fed at boundaries, the outs collected and handed
over, the grants as the script says, the status from the manifest's status atoms. The fixtures are metis-worlds
specialist-tree-2's (8d172fe) alfworld nav and access and minihack act, copied; the dog is the catalog's own (a
single member: no shared fields)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metis.catalog import manifests as mc
from metis.catalog import stub

from catalogs import installed_catalog

FIX = Path(__file__).resolve().parents[1] / "fixtures" / "stub"
SPECIALISTS = ("alfworld_hier/nav", "alfworld_hier/access", "minihack_hier/act")


def domain(pkg):
    d = FIX / pkg
    return mc.Domain(d.name, d.resolve(), mc.check_manifest(d / "domain.yaml"))


@pytest.mark.parametrize("pkg", SPECIALISTS)
def test_a_specialists_own_cases_pass_alone(pkg):
    rs = stub.run_package(domain(pkg))
    assert len(rs) == 2
    assert all(r.passed for r in rs), stub.render(rs)


def test_the_boundary_record():
    (far, no_grant) = stub.run_package(domain("alfworld_hier/nav"))
    b = far.boundaries[0]
    assert b.delivered == ["goto(dest)"] and b.world == ["at(lhere)", "recat(dest,l1)"] and b.holding == ["motion"]
    assert b.outs == ["reached(dest)"] and b.status == "success"
    assert [f["rule"] for f in b.fired] == ["go"] and b.fired[0]["p"] == 1.0
    assert b.intents == [{"event": "go[R=dest,L=lhere,L2=l1]", "held": False}]
    held = no_grant.boundaries[1]
    assert held.intents[0]["held"] and held.outs == ["reached(dest)"]   # fired, but no body: not to the world
    assert far.boundaries[1].outs == []                                    # handed over at b0, not collected twice


def test_the_status_comes_from_the_manifest():
    r = stub.run_package(domain("alfworld_hier/access"))[1]
    assert [b.status for b in r.boundaries] == ["running", "success"]       # elsewhere: an out, neither success nor failure
    act = stub.run_package(domain("minihack_hier/act"))[1]
    assert [b.status for b in act.boundaries] == ["running", "running", "success"]


def test_a_case_that_does_not_fit_the_interface_is_refused(tmp_path):
    nav = domain("alfworld_hier/nav")
    bad = {"deliver: [reached(dest)]": "delivered reached: not in",           # an out is never fed
           "world: [goto(dest)]": "world reading goto: not world-private",   # a goal is no reading
           "grant: [hands]": "not a resource",
           "deliver: [goto(dest]": "unbalanced atom"}
    for line, msg in bad.items():
        f = tmp_path / "c.yaml"
        f.write_text("boundaries:\n  - %s\n" % line)
        with pytest.raises(stub.CaseError, match=msg.replace("(", r"\(")):
            stub.run_case(nav, f)
    f.write_text("boundaries:\n  - {deliver: [goto(dest)], expect: {status: done}}\n")
    with pytest.raises(stub.CaseError, match="one of running"):
        stub.run_case(nav, f)
    f.write_text("cases: []\nboundaries: []\n")
    with pytest.raises(stub.CaseError, match="unknown key cases"):
        stub.run_case(nav, f)


def test_a_failing_expectation_is_a_failing_case_not_an_error(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("boundaries:\n  - {world: [at(lhere), recat(dest, l1)], deliver: [goto(dest)],"
                 " expect: {out: [], status: idle, held: false}}\n")
    (r,) = [stub.run_case(domain("alfworld_hier/nav"), f)]
    assert not r.passed
    assert r.boundaries[0].failures == ["out ['reached(dest)'], expected []", "status success, expected idle",
                                        "held True, expected False"]


CAT = installed_catalog()


@pytest.mark.skipif(CAT is None or not (CAT / "domains" / "dog").is_dir(), reason="no installed catalog with the dog")
def test_the_dog_is_the_degenerate_single_member_case():
    dog = mc.get("dog", CAT)
    assert stub.Interface.of(dog.manifest).degenerate
    rs = [stub.run_case(dog, p) for p in sorted((FIX / "dog" / "cases").glob("*.yaml"))]
    assert [r.passed for r in rs] == [True, True], stub.render(rs)
    assert rs[1].boundaries[0].fired[0]["rule"] == "stand_low"              # the case's init replaced the program's


def test_the_cli_runs_a_packages_cases(tmp_path):
    import shutil
    root = tmp_path / "cat"
    shutil.copytree(FIX / "alfworld_hier", root / "domains" / "alfworld_hier")
    r = subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(root), "cases", "nav", "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert [c["passed"] for c in out] == [True, True]
    (root / "domains" / "alfworld_hier" / "nav" / "cases" / "far.yaml").write_text(
        "boundaries:\n  - {deliver: [goto(dest)], expect: {status: failure}}\n")
    r = subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(root), "cases", "nav"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "FAIL" in r.stdout


@pytest.mark.skipif(CAT is None or not (CAT / "domains" / "kitchen_pot" / "cook_a").is_dir(), reason="no catalog with kitchen_pot")
def test_a_member_that_gates_itself_is_never_held():
    # gating: program (the shared-pot kitchen's cook): its pot rules read the grant themselves, so what it fires without
    # the pot (its own hand: fetch_onion, ask_pot) is sent, not held; a host-gated member (nav, no grant) is held
    cook = mc.get("cook_a", CAT)
    assert stub.Interface.of(cook.manifest).gating == "program"
    rs = stub.run_package(cook)
    assert rs and all(r.passed for r in rs), stub.render(rs)
    first = [b for r in rs for b in r.boundaries if b.fired and not b.holding]
    assert first and not any(i["held"] for b in first for i in b.intents)
    nav = stub.run_package(domain("alfworld_hier/nav"))[1]
    assert nav.boundaries[1].intents[0]["held"]                              # host gating (the default): still held
