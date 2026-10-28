"""metis.catalog.image (linker s-5908724a step 3, metis-lang spec/17): metispy calls metisc's link and build and reads
the image they wrote; it never resolves a composition itself. Over metis-lang's corpus kitchen_pot (the sibling
checkout, as the parity job has it); skipped without metisc or the corpus."""
import shutil
from pathlib import Path

import pytest
import yaml

from metis import catalog
from metis.catalog import image as I
from metis.catalog.manifests import Domain

CORPUS = Path(__file__).resolve().parents[2] / "metis-lang" / "corpus" / "iface"
KITCHEN = CORPUS / "kitchen_pot"

pytestmark = pytest.mark.skipif(not catalog.metisc() or not (KITCHEN / "composition.yaml").is_file(),
                                reason="no metisc, or no metis-lang corpus beside metispy")


@pytest.fixture
def kitchen(tmp_path, monkeypatch):
    """A copy of the corpus kitchen_pot (its build/ never in the checkout), METIS_HOME the corpus's _home."""
    monkeypatch.setenv("METIS_HOME", str(CORPUS / "_home"))
    d = tmp_path / "kitchen_pot"
    shutil.copytree(KITCHEN, d)
    yield d
    I.unuse()


def members_of(comp):
    """The composition's members as the image names them: a Domain per directory beside it (their manifests)."""
    return lambda n: Domain(n, comp / n, yaml.safe_load((comp / n / "domain.yaml").read_text()))


def test_the_link_check_lists_and_writes_nothing(kitchen):
    listing, d = I.link(kitchen, check=True)
    assert "image kitchen_pot v-" in listing and d is None
    assert not (kitchen / "build").exists()


def test_a_built_image_is_read_not_derived(kitchen):
    img = I.image(kitchen)                                   # a composition dir: built into its build/
    assert img.dir.parent == kitchen / "build" and img.name == "kitchen_pot"
    assert img.names == ["pot_director", "cook_a", "cook_b"] and img.direct == "pot_director"
    assert img.clocks == [{"domain": n, "n": 1} for n in img.names]
    assert img.natures["tick__want"] == "report" and img.natures["tick__grant"] == "signal"
    assert img.member("cook_a")["gating"] == "program" and img.member("cook_a")["status"]["success"] == ["tick__done"]
    assert I.image(img.dir).hash == img.hash                 # the image dir itself
    assert I.build(kitchen) == img.dir                       # built again: every member reused, the same image


def test_the_members_sources_are_the_programs_the_image_linked(kitchen, tmp_path):
    img = I.image(kitchen)
    assert I.key_mismatches(img, members_of(kitchen)) == []
    other = tmp_path / "other"                               # cook_a's source changed in another copy: its key differs
    shutil.copytree(KITCHEN, other)
    p = other / "cook_a" / "cook_a.llp"
    text = p.read_text()
    assert "self(cook_a)." in text
    p.write_text(text.replace("self(cook_a).", "self(cook_b)."))
    bad = I.key_mismatches(img, members_of(other))
    assert [m for m, _, _ in bad] == ["cook_a"] and bad[0][1] == img.member("cook_a")["wire_key"]


def test_a_generated_director_is_a_package_with_the_pass_host(kitchen):
    img = I.image(kitchen / "composition_gen.yaml")          # direct: tree_director, no package: the link derives it
    assert img.direct == "tree_director" and img.generated("tree_director")
    I.use(img)
    d = catalog.get("tree_director")
    assert d.dir == img.bundle("tree_director") and d.program.name == "tree_director.llp"
    assert d.manifest["host"] == I.PASS_HOST
    from metis.domain.package import build
    ns = build("tree_director", dir=d.dir, module="pass_host_test", host=d.manifest["host"])
    assert type(ns["make"]("live").inner).__name__ == "PassHost"   # the module's factory, wrapped as every host is
    I.unuse(img)
    with pytest.raises(KeyError):
        catalog.get("tree_director")
