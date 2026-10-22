"""metis.catalog.manifests (moved from metis_catalog; metis-packages s-65778b98): the loader's tests, moved
from metis-catalog tests/test_loader.py; the catalog read is the installed metis-catalog's."""
import subprocess
import sys
from pathlib import Path

import pytest

from catalogs import installed_catalog  # noqa: E402
ROOT = installed_catalog()
pytestmark = pytest.mark.skipif(ROOT is None, reason="no metis-catalog installed (its domains)")
sys.path.insert(0, str(ROOT))
from metis.catalog import manifests as mc  # noqa: E402


def test_every_manifest_is_whole():
    ds = {d.name: d for d in mc.domains(ROOT)}
    assert {"dog", "kitchen", "kitchen_shared", "wildfire", "tamarisk", "wgc"} <= set(ds)
    for d in ds.values():
        if d.manifest.get("program"):
            assert d.program.is_file(), d.name
        if d.manifest.get("notch"):
            assert d.notch.is_file(), d.name
        h = d.host
        if h and h[0] is not None:
            assert h[0].is_file(), d.name
        if d.agent:
            assert d.agent[0].is_file(), d.name
        for k, v in d.intent.items():
            for p in (v.values() if isinstance(v, dict) else v if isinstance(v, list) else [v]):
                assert p.is_file(), (d.name, k, p)
        if d.partner_of:
            assert d.partner_of in ds and d.name in ds[d.partner_of].pair


def test_paths_and_the_host_spec():
    dog, wf, kit = mc.get("dog", ROOT), mc.get("wildfire", ROOT), mc.get("kitchen", ROOT)
    assert dog.program == ROOT / "domains/dog/dog.llp" and dog.voice == "own" and dog.display == "the dog"
    assert dog.host_spec_for() == "dog_host.py:make" and dog.intent["grammar"].name == "dog.gbnf"
    assert dog.agent is None and dog.has()["agent"] and dog.has()["intent"]          # the generic agent, from intent.yaml
    assert wf.program == ROOT / "domains/wildfire/wildfire.llp"                      # an RDDL world package (rddl-worlds)
    assert wf.host_spec_for() == "wildfire_host.py:make"                             # as wildfire.llp's own #plugin
    assert "#plugin wildfire_host.py:make" in wf.program.read_text()
    assert kit.host_spec_for() == "metis.domains.kitchen.host:make" and kit.agent[1] == "kitchen"
    assert kit.clock == {"beats": 2} and kit.pair == ["kitchen_shared"]
    with pytest.raises(KeyError):
        mc.get("nope", ROOT)


def test_the_index_lists_every_catalog_the_domains_first():
    es = mc.index(ROOT)
    llps = mc.walk(ROOT, "*.llp")          # the loader's walk: no dot directory, build/, records/ …
    assert len(es) == len(llps)                                                      # every program once
    kinds = [e["kind"] for e in es]
    assert kinds == sorted(kinds, key=lambda k: k != "domain")
    named = [e["name"] for e in es if e["manifest"]]
    assert named == sorted(named)                                                    # the manifests first, by name
    assert {"dog", "kitchen", "plant", "tamarisk", "wgc", "wildfire",                # the fiction ports: domains
            "herstory", "love_triangle", "talespin", "three_pigs", "three_pigs_connectives", "tragedy",
            "nav", "xtraffic", "gol", "sysadmin", "hanoi", "elevators",               # the RDDL worlds too
            "primes", "cellular_automata", "ll", "blocks_world", "tower_of_hanoi",    # the Ceptre example ports and
            "gossip", "fptp", "bridge", "bridge_decl"} <= set(named)                  # the puzzles (catalog-packages)
    assert "kitchen_shared" not in [e["name"] for e in es]                           # a partner is listed by its lead
    by = {e["path"]: e for e in es}
    assert by["domains/wildfire/wildfire"]["notch"] == "wildfire.notch.json"
    assert by["domains/ceptre/three_pigs/three_pigs"]["kind"] == "domain"
    for n in ("herstory", "love_triangle", "talespin", "three_pigs", "three_pigs_connectives", "tragedy"):
        h = mc.get(n, ROOT).has()                                            # no intents: the domain agent opens
        assert h["program"] and not h["intent"] and not h["agent"], n        # them with no verbs (agent-no-intent)
    assert by["domains/ceptre/ll/ll"]["kind"] == "domain"                             # a package now (catalog-packages)
    assert by["domains/road/scenes/nav4_instance"]["kind"] == "catalog"                           # program only, no manifest


def test_list_cli():
    out = subprocess.run([sys.executable, "-m", "metis.catalog.manifests_cli", "list", "--domains"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    rows = [l.split()[:2] for l in out.splitlines()[1:]]
    assert "a robot dog" in out and ["domain", "dog"] in rows                 # by name: blocks_world sorts first now
    assert {r[0] for r in rows} == {"domain"}


def test_the_extensions_are_the_programs_in_name_order(tmp_path):
    """human-loop step 1 (the architect's ruling): a domain's served
    program is its program plus every extensions/*.llp, in name order."""
    from metis.catalog import manifests as mc
    d = tmp_path / "domains/plant"
    (d / "extensions").mkdir(parents=True)
    (d / "domain.yaml").write_text("line: a plant\nprogram: plant.llp\n")
    (d / "plant.llp").write_text("catalog plant 1.\n")
    assert mc.get("plant", tmp_path).extensions() == [] and not mc.get("plant", tmp_path).has()["extensions"]
    for n in ("visitor", "a_sun", "notes.txt"):
        (d / "extensions" / (n if "." in n else n + ".llp")).write_text("catalog %s 1.\nextends plant.\n" % n)
    p = mc.get("plant", tmp_path)
    assert [x.name for x in p.extensions()] == ["a_sun.llp", "visitor.llp"] and p.has()["extensions"]
    row = next(e for e in mc.index(tmp_path) if e["name"] == "plant")
    assert row["has"]["extensions"]


def test_serves_is_the_program_plus_extensions_unless_the_manifest_lists_it(tmp_path):
    """runtime-binding Q1 (the architect's ruling): the served program's
    files are the manifest's `serves:` (relative to its directory, or
    pkg.mod:path), else [program] + extensions/*.llp — py, Go and the bundle
    emitter read only this."""
    from metis.catalog import manifests as mc
    d = tmp_path / "domains/plant"
    (d / "extensions").mkdir(parents=True)
    (d / "domain.yaml").write_text("line: a plant\nprogram: plant.llp\n")
    (d / "plant.llp").write_text("catalog plant 1.\n")
    (d / "extensions" / "visitor.llp").write_text("catalog visitor 1.\nextends plant.\n")
    assert [x.name for x in mc.get("plant", tmp_path).serves()] == ["plant.llp", "visitor.llp"]
    (d / "domain.yaml").write_text("line: a plant\nprogram: plant.llp\n"
                                   "serves: [other/scene.llp, 'metis_catalog:__init__.py']\n")
    s = mc.get("plant", tmp_path).serves()
    assert s[0] == (d / "other/scene.llp").resolve()
    assert s[1] == (ROOT / "metis_catalog" / "__init__.py").resolve()


def test_the_kitchen_pair_serves_its_pair_scenes():
    """The kitchen's agents serve metispy's pair scenes (metis.domains.kitchen
    pair.SCENES), not the loop console's kitchen.llp."""
    from metis.catalog import manifests as mc
    pytest.importorskip("metis.domains.kitchen")
    k, s = mc.get("kitchen").serves(), mc.get("kitchen_shared").serves()
    assert [x.name for x in k] == ["kitchen_host.llp"] and k[0].is_file()
    assert [x.name for x in s] == ["kitchen_shared.llp"] and s[0].is_file()
    assert k[0].parent.name == "catalogs" and s[0].parent.name == "shared"


def test_the_manifests_extensions_list_orders_them(tmp_path):
    d = tmp_path / "domains/plant"
    (d / "extensions").mkdir(parents=True)
    (d / "plant.llp").write_text("catalog plant 1.\n")
    for n in ("a_sun", "visitor"):
        (d / "extensions" / (n + ".llp")).write_text("catalog %s 1.\nextends plant.\n" % n)
    (d / "domain.yaml").write_text("program: plant.llp\nextensions: [extensions/visitor.llp]\n")
    p = mc.get("plant", tmp_path)
    assert [x.name for x in p.extensions()] == ["visitor.llp"]
    assert [x.name for x in p.serves()] == ["plant.llp", "visitor.llp"]


@pytest.mark.parametrize("yaml_text, why", [
    ("name: plant\nprogram: plant.llp\n", "unknown key name"),
    ("program: plant.llp\ndocs: README.md\n", "unknown key docs"),
    ("program: plant.llp\npacks: [a.llp]\n", "unknown key packs"),
    ("program: plant.llp\nintent: {spec: intent.yaml, grammar: intent/plant.gbnf}\n", "unknown intent key grammar"),
    ("program: plant.llp\nintent: {grammar: intent/plant.gbnf}\n", "intent is {spec"),
    ("program: plant.llp\nreload: sometimes\n", "reload is propose or auto"),
    ("program: plant.llp\nextensions: a.llp\n", "extensions is a list"),
    # every path stays inside the catalog root (nina's catalog-sdk review, note a)
    ("program: ../../../../etc/passwd\n", "program '../../../../etc/passwd' leaves the catalog root"),
    ("program: /etc/passwd\n", "program '/etc/passwd' is absolute"),
    ("program: plant.llp\nextensions: [../../../x.llp]\n", "extensions '../../../x.llp' leaves"),
    ("serves: [plant.llp, ../../../x.llp]\n", "serves '../../../x.llp' leaves"),
    ("program: plant.llp\nhost: ../../../h.py:make\n", "host '../../../h.py' leaves"),
    ("program: plant.llp\nagent: ../../../a.py:v\n", "agent '../../../a.py' leaves"),
    ("program: plant.llp\nnotch: ../../../n.notch.json\n", "notch '../../../n.notch.json' leaves"),
    ("program: plant.llp\nrecords: ../../../r/\n", "records '../../../r/' leaves"),
    ("program: plant.llp\nintent: {spec: ../../../i.yaml}\n", "intent.spec '../../../i.yaml' leaves"),
])
def test_the_loader_refuses_what_a_package_does_not_carry(tmp_path, yaml_text, why):
    """catalog-entry step 2: the package's keys only (CATALOG.md's table)."""
    d = tmp_path / "domains/plant"
    d.mkdir(parents=True)
    (d / "domain.yaml").write_text(yaml_text)
    with pytest.raises(mc.ManifestError, match=why):
        mc.domains(tmp_path)


def test_the_name_is_the_directorys_and_the_intent_files_are_derived(tmp_path):
    d = tmp_path / "domains/sub/plant"
    (d / "intent").mkdir(parents=True)
    (d / "domain.yaml").write_text("program: plant.llp\nintent: {spec: intent.yaml}\n")
    (d / "intent.yaml").write_text("{}\n")
    for f in ("plant.gbnf", "plant-prompt.txt", "plant-spoken.txt"):
        (d / "intent" / f).write_text("x\n")
    p = mc.get("plant", tmp_path)
    assert p.name == "plant" and p.display == "plant"
    assert {k: (v.name if hasattr(v, "name") else {t: x.name for t, x in v.items()}) for k, v in p.intent.items()} == \
        {"spec": "intent.yaml", "grammar": "plant.gbnf", "prompt": "plant-prompt.txt", "examples": {"gemma4": "plant-spoken.txt"}}


def test_a_package_may_name_its_neighbours_files_never_outside(tmp_path):
    """Containment is the catalog root's, not the package's: the kitchen
    pair's partner names ../kitchen/agent.py, an rddl package its program one
    level up (../wildfire.llp); a pkg.mod:path or pkg.mod:factory names a
    package, not a file. A manifest outside a domains/ tree is checked for
    its keys only (its root is unknown), unless the root is given."""
    k = tmp_path / "domains/kitchen_shared"
    k.mkdir(parents=True)
    (k / "domain.yaml").write_text('agent: ../kitchen/agent.py:kitchen_shared\n'
                                   'serves: ["metis.domains.kitchen:catalogs/shared/kitchen_shared.llp"]\n'
                                   'host: metis.domains.kitchen.host:make\n')
    w = tmp_path / "domains/rddl/wildfire"
    w.mkdir(parents=True)
    (w / "domain.yaml").write_text("program: ../wildfire.llp\nnotch: ../wildfire.notch.json\n")
    assert [d.name for d in mc.domains(tmp_path)] == ["kitchen_shared", "wildfire"]
    loose = tmp_path / "elsewhere/domain.yaml"
    loose.parent.mkdir()
    loose.write_text("program: ../../../x.llp\n")
    assert mc.check_manifest(loose)["program"] == "../../../x.llp"
    with pytest.raises(mc.ManifestError, match="leaves the catalog root"):
        mc.check_manifest(loose, root=tmp_path)


def test_a_package_is_a_manifest_anywhere_under_the_root(tmp_path):
    """cwd-root s-ddac2cbf rule 9: domains/ is a convention, not a
    requirement — a domain.yaml anywhere under the root is a package (its
    name the directory's), never one under a dot directory, build/, records/
    or __pycache__; the catalog's own domains/ceptre/ is a root of its 13."""
    for rel in ("lamp", "shelf/bell", "domains/clock"):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "domain.yaml").write_text("program: %s.llp\n" % d.name)
        (d / ("%s.llp" % d.name)).write_text("%% the %s\n" % d.name)
    for rel in (".metis/scratch/ghost", ".git/x", "lamp/build/copy", "lamp/records/r", "node_modules/n"):
        d = tmp_path / rel
        d.mkdir(parents=True)
        (d / "domain.yaml").write_text("program: x.llp\n")
        (d / "x.llp").write_text("% x\n")
    assert [d.name for d in mc.domains(tmp_path)] == ["bell", "clock", "lamp"]
    assert [e["name"] for e in mc.index(tmp_path)] == ["bell", "clock", "lamp"]
    (tmp_path / "other" / "lamp").mkdir(parents=True)
    (tmp_path / "other" / "lamp" / "domain.yaml").write_text("")
    with pytest.raises(ValueError, match="domain 'lamp' twice"):
        mc.domains(tmp_path)
    ceptre = ROOT / "domains" / "ceptre"
    if ceptre.is_dir():                                   # a checkout (the wheel has no ceptre/)
        assert len(mc.domains(ceptre)) == 13 and "three_pigs" in {d.name for d in mc.domains(ceptre)}
