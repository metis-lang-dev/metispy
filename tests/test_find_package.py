"""metis.catalog's one package rule (metis-packages s-65778b98 step 3, the user's ruling): a package is
<root>/domains/NAME ($METIS_CATALOG, else the cwd), else $METIS_HOME/domains/NAME — the order the
compiler's include rule takes after the including file's own directory. A miss names both places; the
listing shows the root's domains and METIS_HOME's marked shared, a name in both being the root's."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from metis import catalog
from metis.lang.compiler import include_path

MANIFEST = "program: {name}.llp\ndisplay: the {name}\nline: the {name} line\n"
PROGRAM = "catalog {name} 1.\nlayers (dyn).\nstages (main).\nnamespace {name} produce (dyn).\npred on : {name}.\n"


def package(base, name, nested=()):
    d = base / "domains" / name
    d.mkdir(parents=True)
    (d / "domain.yaml").write_text(MANIFEST.format(name=name))
    (d / f"{name}.llp").write_text(PROGRAM.format(name=name))
    for n in nested:                           # a package of several domains (kitchen_pot's cooks)
        (d / n).mkdir()
        (d / n / "domain.yaml").write_text(MANIFEST.format(name=n))
        (d / n / f"{n}.llp").write_text(PROGRAM.format(name=n))
    return d


@pytest.fixture
def two(tmp_path, monkeypatch):
    root, home = tmp_path / "root", tmp_path / "home"
    package(root, "dog")
    package(root, "plant")
    package(home, "plant")                     # in both: the root's
    package(home, "lamp", nested=("lamp_a",))  # only home's
    monkeypatch.delenv("METIS_CATALOG", raising=False)
    monkeypatch.setenv("METIS_HOME", str(home))
    monkeypatch.chdir(root)
    return root, home


def test_home_is_metis_home_else_metispys_own_catalog(monkeypatch, tmp_path):
    monkeypatch.delenv("METIS_HOME", raising=False)
    assert catalog.home() == Path(catalog.__file__).resolve().parent
    monkeypatch.setenv("METIS_HOME", str(tmp_path))
    assert catalog.home() == tmp_path.resolve()


def test_the_root_first_then_home(two):
    root, home = two
    assert catalog.find_package("dog") == (root / "domains/dog").resolve()
    assert catalog.find_package("plant") == (root / "domains/plant").resolve()
    assert catalog.find_package("lamp") == (home / "domains/lamp").resolve()
    assert catalog.get("plant").dir.resolve() == (root / "domains/plant").resolve()
    assert catalog.get("lamp").dir.resolve() == (home / "domains/lamp").resolve()
    assert [d.name for d in catalog.package_domains("lamp")] == ["lamp", "lamp_a"]


def test_metis_catalog_is_the_root_over_the_cwd(two, tmp_path, monkeypatch):
    other = tmp_path / "other"
    package(other, "lamp")
    monkeypatch.setenv("METIS_CATALOG", str(other))
    assert catalog.find_package("lamp") == (other / "domains/lamp").resolve()
    with pytest.raises(FileNotFoundError):
        catalog.find_package("dog")            # the cwd is not the root now


def test_a_miss_names_both_places(two):
    root, home = two
    with pytest.raises(FileNotFoundError) as e:
        catalog.find_package("nosuch")
    assert str(root.resolve()) in str(e.value) and str(home.resolve()) in str(e.value)
    assert "the catalog root" in str(e.value) and "METIS_HOME" in str(e.value)
    with pytest.raises(KeyError) as e:
        catalog.get("nosuch")
    assert str(root.resolve()) in str(e.value) and str(home.resolve()) in str(e.value)


def test_the_listing_marks_homes_domains_shared(two):
    rows = {e["name"]: e for e in catalog.listing()}
    assert set(rows) == {"dog", "plant", "lamp", "lamp_a"}
    assert not rows["dog"].get("shared") and not rows["plant"].get("shared")
    assert rows["lamp"]["shared"] and Path(rows["lamp"]["path"] + ".llp").is_file()      # a path, no .llp: absolute
    assert [e["name"] for e in catalog.index() if e["kind"] == "domain"][:2] == ["lamp", "lamp_a"]


def test_a_given_root_sees_only_itself(two):
    root, _ = two
    assert {e["name"] for e in catalog.listing(root)} == {"dog", "plant"}
    with pytest.raises(KeyError):
        catalog.get("lamp", root)


def test_the_include_rule_is_the_same_order(two, tmp_path):
    root, home = two
    (home / "iface").mkdir()
    (home / "iface" / "x.llp").write_text("% home's\n")
    (root / "iface").mkdir()
    (root / "iface" / "y.llp").write_text("% the root's\n")
    (home / "iface" / "y.llp").write_text("% home's\n")
    here = tmp_path / "pkg"
    here.mkdir()
    assert include_path("iface/x.llp", here) == (home / "iface/x.llp").resolve()
    assert include_path("iface/y.llp", here) == (root / "iface/y.llp").resolve()
    (here / "iface").mkdir()
    (here / "iface" / "y.llp").write_text("% the includer's\n")
    assert include_path("iface/y.llp", here) == (here / "iface/y.llp").resolve()     # its own directory first
    with pytest.raises(FileNotFoundError) as e:
        include_path("iface/z.llp", here)
    assert all(w in str(e.value) for w in ("the including file's directory", "the catalog root", "METIS_HOME"))


def test_the_cli_find_and_home(two):
    root, home = two
    def cli(*a):
        r = subprocess.run([sys.executable, "-m", "metis.catalog", *a, "--json"], capture_output=True, text=True,
                           timeout=60)
        return r.returncode, json.loads(r.stdout)
    assert cli("home") == (0, {"home": str(home.resolve())})
    rc, out = cli("find", "lamp")
    assert rc == 0 and out == {"name": "lamp", "dir": str((home / "domains/lamp").resolve()), "from": "home",
                               "domains": ["lamp", "lamp_a"]}
    rc, out = cli("find", "dog")
    assert rc == 0 and out["from"] == "root" and out["domains"] == ["dog"]
    rc, err = cli("find", "nosuch")
    assert rc != 0 and "METIS_HOME" in err["error"]
