"""The domain artifact goldens of the catalog's dog and plant (metis-catalog
domains/<n>: the manifest's program plus extensions/*.llp) — the two hosted
domains go-domain-agent serves first. Reads the sibling metis-catalog
checkout (CI: CATALOG_TESTS). Regenerate with METIS_BLESS=1."""

import json

import pytest

from metis.kernel.artifact import build, resolve
from test_artifact import check_golden


@pytest.mark.parametrize("name, extensions", [("dog", []),
                                         ("plant", ["visitor.llp"])])
def test_the_catalog_domain_golden(name, extensions):
    program, pack_files, out = resolve(name)
    assert out.parts[-2:] == ("build", f"{name}.metis")
    files = build(program, pack_files)
    s = json.loads(files["sidecar.json"])
    assert s["extensions"] == extensions
    if extensions:              # the extensions' clauses in the served script (F3)
        assert [c["name"] for c in s["script"][0]["clauses"]
                if c.get("extension")] == ["visitor_waters"]
    check_golden(name, files)


@pytest.mark.parametrize("name, scene", [
    ("kitchen", "kitchen_host.llp"), ("kitchen_shared", "kitchen_shared.llp")])
def test_the_kitchen_pair_resolves_through_serves(name, scene):
    """Q1: the manifest's `serves` names the pair scene the agent serves;
    the emitted artifact is the kitchen pair golden's."""
    from test_artifact import GOLDENS
    program, extensions, _ = resolve(name)
    assert program.name == scene and extensions == []
    files = build(program, extensions)
    assert files["sidecar.json"] == \
        (GOLDENS / f"{name}.metis" / "sidecar.json").read_text()
