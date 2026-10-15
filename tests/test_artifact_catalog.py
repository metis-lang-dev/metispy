"""The domain artifact goldens of the catalog's dog and plant (metis-catalog
domains/<n>: the manifest's program plus packs/*.llp) — the two hosted
domains go-domain-agent serves first. Reads the sibling metis-catalog
checkout (CI: CATALOG_TESTS). Regenerate with METIS_BLESS=1."""

import json

import pytest

from metis.kernel.artifact import build, resolve
from test_artifact import check_golden


@pytest.mark.parametrize("name, packs", [("dog", []),
                                         ("plant", ["visitor.llp"])])
def test_the_catalog_domain_golden(name, packs):
    program, pack_files, out = resolve(name)
    assert out.parts[-2:] == ("build", f"{name}.metis")
    files = build(program, pack_files)
    s = json.loads(files["sidecar.json"])
    assert s["packs"] == packs
    check_golden(name, files)
