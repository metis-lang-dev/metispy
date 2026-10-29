"""A manifest's generated level declarations (metis-lang spec/16 amendment 1, toolchain-units s-ddaed776 step 2):
metis.lang.levels against metis-lang's parity corpus (corpus/levels: one program, a manifest per case) and its goldens
(corpus/goldens/levels, what `metisc --levels` prints, its refusals included), byte for byte; the level package's wire,
made by metispy's emitter with its manifest, == metis-lang's golden; without its manifest the program does not
compile (its level declarations are the manifest's)."""
from pathlib import Path

import pytest

from langpath import LANG

CORPUS, GOLD = LANG / "corpus/levels", LANG / "corpus/goldens/levels"
CASES = sorted(p.name for p in CORPUS.iterdir() if (p / "domain.yaml").is_file()) if CORPUS.is_dir() else []

pytestmark = pytest.mark.skipif(not CASES, reason=f"no metis-lang corpus/levels at {LANG} (METIS_LANG)")


def listing(case):
    """What `metisc --levels CASE/domain.yaml` prints, as its golden holds it (stdout, stderr, rc)."""
    from metis.lang import levels
    from metis.lang.compiler import LangError
    try:
        return "".join(l + "\n" for l in levels.lines(CORPUS / case / "domain.yaml")) + "rc 0\n"
    except LangError as e:
        return "".join("error: %s\n" % m for m in e.findings) + "rc 1\n"


@pytest.mark.parametrize("case", CASES)
def test_the_generated_declarations_are_metiscs(case):
    assert listing(case) == (GOLD / f"{case}.txt").read_text()


def test_the_level_package_compiles_with_its_manifest_only(monkeypatch):
    from metis.kernel import artifact
    from metis.kernel.catalog import CatalogError
    monkeypatch.delenv("METISC", raising=False)
    monkeypatch.setattr(artifact, "metisc", lambda: None)            # metispy's own emitter
    files = artifact.build(CORPUS / "alf.llp", [], manifest=CORPUS / "alf_view/domain.yaml")
    assert files["ground.wire"] == (GOLD / "alf_view.wire").read_text()
    with pytest.raises(CatalogError, match="undeclared predicate 'place_level'"):
        artifact.served_doc(CORPUS / "alf.llp")


def test_a_dict_manifest_generates_the_same(monkeypatch):
    """The loop's candidate carries its levels as a dict (no file): the same declarations, lines 0 in a refusal."""
    import yaml
    from metis.lang import levels
    from metis.lang.compiler import resolve_includes
    from metis.lang.parser import parse_file
    from metis.lang.ast import decl_line
    man = yaml.safe_load((CORPUS / "alf_view/domain.yaml").read_text())
    cat = resolve_includes(parse_file(str(CORPUS / "alf.llp")), CORPUS)
    assert [decl_line(d) for d in levels.level_decls(man, cat)] == levels.lines(CORPUS / "alf_view/domain.yaml")
