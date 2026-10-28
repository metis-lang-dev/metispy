"""metis.catalog (catalog-sdk s-bc1718e3; moved from metis_loop_sdk.catalog by metis-packages s-65778b98, with
its tests: the widget's and the TS face's stay metis-loop's): the facade reads what the
loader reads, apply is the one validated write (into a scratch catalog here),
the CLI says the same in JSON, the widget ships as package data.

The fixtures are the installed metis-catalog's own domains (its checkout
installed -e, or the wheel's; $METIS_CATALOG overrides); the writes go to
a scratch copy of its dog and plant."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from metis import catalog

from catalogs import installed_catalog

CAT = installed_catalog()
if not (CAT / "domains").is_dir():
    pytest.skip(f"no metis-catalog domains at {CAT} (METIS_CATALOG)", allow_module_level=True)


@pytest.fixture
def scratch(tmp_path):
    """A scratch catalog: the dog and the plant, and the intent generator."""
    for d in ("dog", "plant"):
        shutil.copytree(CAT / "domains" / d, tmp_path / "domains" / d,
                        ignore=shutil.ignore_patterns("records", "__pycache__"))
    shutil.copytree(CAT / "tools", tmp_path / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


LAMP = """% lamp — a lamp that can be switched on
catalog lamp 1.
provenance text "a lamp that can be switched on" "2026-10-21".
layers (dyn).
stages (main).

namespace lamp produce (dyn).

pred off : lamp.
pred on : lamp.

stage main {
  %% I switch it on.
  switch_on [dyn] : off -o on.
}

#init off 1
"""


def cli(args, cwd):
    """The CLI run in `cwd`, without $METIS_CATALOG: its root is the cwd."""
    env = {k: v for k, v in os.environ.items() if k != "METIS_CATALOG"}
    r = subprocess.run([sys.executable, "-m", "metis.catalog", *args], capture_output=True, text=True,
                       env=env, cwd=cwd, timeout=120)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_root_is_the_env_else_the_cwd(monkeypatch, tmp_path):
    """cwd-root s-ddac2cbf: $METIS_CATALOG, else the current directory —
    `cd DIR` makes DIR the catalog; no configured root, no sibling."""
    (tmp_path / "a").mkdir()
    monkeypatch.setenv("METIS_CATALOG", str(tmp_path))
    monkeypatch.chdir(tmp_path / "a")
    assert catalog.root() == tmp_path.resolve()
    monkeypatch.delenv("METIS_CATALOG")
    assert catalog.root() == (tmp_path / "a").resolve()
    assert not hasattr(catalog, "SIBLING")
    with pytest.raises(TypeError):
        catalog.root("/elsewhere")                        # the config argument went


def test_an_empty_directory_is_an_empty_catalog(monkeypatch, tmp_path):
    """An empty cwd lists nothing; the first apply makes domains/<name>/,
    and the index then shows it; the intent generator is the installed
    metis-catalog's (the root has no tools/)."""
    monkeypatch.delenv("METIS_CATALOG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert catalog.domains() == [] and catalog.index() == [] and catalog.listing() == []
    out = catalog.apply("lamp", program=LAMP, manifest="display: the lamp\nline: a lamp\nprogram: lamp.llp\n")
    assert sorted(out["written"]) == ["domains/lamp/domain.yaml", "domains/lamp/lamp.llp"]
    assert [e["name"] for e in catalog.index()] == ["lamp"] and [d.name for d in catalog.domains()] == ["lamp"]
    gen = catalog.generator()
    assert not (tmp_path / "tools").exists() and Path(gen.__file__) == Path(__import__("metis.intents").intents.__file__)
    assert gen.ROOT == tmp_path.resolve()
    rows = cli(["list", "--json"], cwd=tmp_path)          # the CLI in the cwd: the same catalog
    assert [e["name"] for e in rows] == ["lamp"]


def test_state_is_the_roots_dot_metis_and_hidden(monkeypatch, tmp_path):
    """DIR/.metis/ holds the loop's own state: made on demand, ignored in a
    git checkout's .gitignore (once), and nothing under it is a catalog."""
    monkeypatch.delenv("METIS_CATALOG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert catalog.state() == tmp_path.resolve() / ".metis" and (tmp_path / ".metis").is_dir()
    assert not (tmp_path / ".gitignore").exists()             # not a checkout: nothing to ignore in
    (tmp_path / ".git").mkdir()
    (tmp_path / ".gitignore").write_text("build/")
    catalog.state(); catalog.state()
    assert (tmp_path / ".gitignore").read_text().splitlines().count("/.metis/") == 1
    assert (tmp_path / ".gitignore").read_text().startswith("build/\n")
    (tmp_path / "fresh").mkdir()
    (tmp_path / "fresh" / ".git").mkdir()
    catalog.state(tmp_path / "fresh")                         # a .gitignore it creates starts with its line
    assert (tmp_path / "fresh" / ".gitignore").read_text().startswith("# the metis loops")
    (tmp_path / ".metis" / "scratch").mkdir()
    (tmp_path / ".metis" / "scratch" / "lamp.llp").write_text(LAMP)
    assert catalog.index() == []
    assert cli(["state", "--json"], cwd=tmp_path)["state"] == str(tmp_path.resolve() / ".metis")


def test_installed_is_the_packages_own_never_the_root(monkeypatch, tmp_path):
    """cwd-root step 2: the installed metis-catalog's catalog and tools/, wherever
    the cwd or $METIS_CATALOG is — the lamp demo's plant and an RDDL host's tools."""
    monkeypatch.setenv("METIS_CATALOG", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    got = catalog.installed()
    assert got["catalog"] != tmp_path and (got["catalog"] / "domains" / "plant" / "domain.yaml").is_file()
    assert (got["tools"] / "rddl_host.py").is_file()
    assert cli(["installed", "--json"], cwd=tmp_path) == {k: str(v) for k, v in got.items()}


def test_metisc_is_the_env_else_the_sibling_else_path_else_a_refusal(monkeypatch, tmp_path):
    monkeypatch.delenv("METISC", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    sib = tmp_path / "metis-lang" / "src" / "metisc"
    monkeypatch.setattr(catalog, "SIBLING_METISC", sib)
    assert catalog.metisc() is None
    sib.parent.mkdir(parents=True)                       # the metis-lang checkout beside metispy's (the loop's rule)
    sib.write_text("#!/bin/sh\n")
    sib.chmod(0o755)
    assert catalog.metisc() == str(sib)
    sib.unlink()
    assert catalog.metisc() is None
    (tmp_path / "x.llp").write_text(LAMP)
    with pytest.raises(ValueError, match="set METISC or put metisc on PATH"):
        catalog.diagnostics(tmp_path / "x.llp")
    monkeypatch.setenv("METISC", "/opt/metisc")
    assert catalog.metisc() == "/opt/metisc"


def test_the_facade_reads_what_the_loader_reads():
    mc = catalog.loader()
    assert [d.name for d in catalog.domains(CAT)] == [d.name for d in mc.domains(CAT)]
    assert catalog.index(CAT) == mc.index(CAT)
    assert catalog.get("dog", CAT).program == mc.get("dog", CAT).program
    assert catalog.intents("dog", CAT) == mc.get("dog", CAT).intent
    dog = next(e for e in catalog.index(CAT) if e["name"] == "dog")
    assert catalog.readme_head(dog["readme"], r=CAT).startswith("# ")
    assert catalog.host_of(dog["path"], CAT) == dog["host"]


def test_listing_is_the_index_domains_then_each_partner():
    rows = catalog.listing(CAT)
    doms = [e["name"] for e in catalog.index(CAT) if e["kind"] == "domain"]
    assert [e["name"] for e in rows if e["kind"] == "domain"] == doms
    partners = [d.name for d in catalog.domains(CAT) if d.partner_of]
    assert [e["name"] for e in rows if e["kind"] == "partner"] == partners
    assert all(set(e) >= {"name", "kind", "path", "display", "line", "has"} for e in rows)


def test_the_generator_reads_its_root(scratch):
    gen = catalog.generator(scratch)
    assert gen.ROOT == scratch.resolve()
    assert gen.files("dog") == catalog.intent_files("dog", (scratch / "domains/dog/intent.yaml").read_text(), scratch)


def test_the_fingerprint_moves_with_what_a_reload_takes(scratch):
    fp = catalog.fingerprint("dog", scratch)
    (scratch / "domains/dog/records").mkdir(exist_ok=True)
    (scratch / "domains/dog/records/s.jsonl").write_text("{}\n")          # not watched
    (scratch / "domains/dog/notes.md").write_text("x")
    assert catalog.fingerprint("dog", scratch) == fp
    p = scratch / "domains/dog/dog.llp"
    p.write_text(p.read_text() + "\n% a comment\n")
    assert catalog.fingerprint("dog", scratch) != fp


def test_apply_writes_the_program_and_an_extension_never_outside(scratch):
    prog = (scratch / "domains/dog/dog.llp").read_text()
    new = prog + "\n% applied\n"
    files = catalog.plan("dog", program=new, extensions={"Pet Tricks": "% an extension\n"}, r=scratch)
    assert [f["path"] for f in files] == ["domains/dog/dog.llp", "domains/dog/extensions/pet_tricks.llp"]
    assert not (scratch / "domains/dog/extensions/pet_tricks.llp").exists()      # plan writes nothing
    before = catalog.fingerprint("dog", scratch)
    out = catalog.apply("dog", program=new, extensions={"Pet Tricks": "% an extension\n"}, r=scratch)
    assert out["written"] == [f["path"] for f in files]
    assert "+% applied" in out["diff"]
    assert (scratch / "domains/dog/dog.llp").read_text() == new
    assert out["fingerprint"] == catalog.fingerprint("dog", scratch) != before
    assert catalog.apply("dog", program=new, r=scratch)["written"] == []          # unchanged: nothing
    with pytest.raises(ValueError):
        catalog.plan("../../etc/x", program="", r=scratch)


def test_apply_intents_generates_and_names_the_spec(scratch):
    d = scratch / "domains/plant"
    (d / "intent.yaml").unlink(missing_ok=True)
    shutil.rmtree(d / "intent", ignore_errors=True)
    mf = d / "domain.yaml"
    mf.write_text("".join(l for l in mf.read_text().splitlines(True) if not l.startswith("intent:")))
    text = catalog.skeleton("plant", scratch)
    out = catalog.apply("plant", intents=text, r=scratch)
    gen = catalog.generator(scratch)
    made = {f: t for f, t in gen.files("plant").items() if t}  # an empty one is not written (no spoken: no -spoken.txt)
    assert set(out["written"]) == {"domains/plant/intent.yaml", "domains/plant/domain.yaml",
                                   *("domains/plant/intent/" + f for f in made)}
    assert mf.read_text().endswith("intent: {spec: intent.yaml}\n")
    for fn, t in made.items():                                # what `tools/intents.py gen` writes
        assert (d / "intent" / fn).read_text() == t
    assert not (d / "intent/plant-spoken.txt").exists()
    with pytest.raises(ValueError):
        catalog.plan("plant", intents="not: [a, mapping", r=scratch)


def test_the_manifest_goes_through_the_loaders_check(scratch):
    with pytest.raises(ValueError, match="unknown key"):
        catalog.plan("dog", manifest={"colour": "brown"}, r=scratch)
    mf = scratch / "domains/dog/domain.yaml"
    old = mf.read_text()
    files = catalog.plan("dog", manifest={"reload": "auto", "line": "a dog: it sits"}, r=scratch)
    text = files[0]["text"]
    assert text.endswith("reload: auto\n") and "line: 'a dog: it sits'\n" in text
    assert text.count("\n") == old.count("\n") + 1                # one line replaced, one appended
    with pytest.raises(ValueError, match="reload is propose or auto"):
        catalog.plan("dog", manifest={"reload": "sometimes"}, r=scratch)
    # the loader's containment, read where the manifest will live (needs metis-catalog ae1e3b2)
    if "root" in __import__("inspect").signature(catalog.loader().check_manifest).parameters:
        with pytest.raises(ValueError, match="leaves the catalog root"):
            catalog.plan("dog", manifest={"notch": "../../../../etc/passwd"}, r=scratch)
        assert catalog.plan("dog", manifest={"notch": "../plant/x.notch.json"}, r=scratch)   # a neighbour's file


def test_a_new_domain(scratch):
    out = catalog.apply("lamp", program="% the lamp\n", manifest={"display": "the lamp", "records": "records/"},
                        r=scratch)
    assert out["written"] == ["domains/lamp/lamp.llp", "domains/lamp/domain.yaml"]
    assert (scratch / "domains/lamp/domain.yaml").read_text() == \
        "display: the lamp\nrecords: records/\nprogram: lamp.llp\n"
    assert catalog.get("lamp", scratch).program == (scratch / "domains/lamp/lamp.llp").resolve()
    whole = "display: the bell\nprogram: bell.llp\n"
    catalog.apply("bell", program="% the bell\n", manifest=whole, r=scratch)
    assert (scratch / "domains/bell/domain.yaml").read_text() == whole


def test_take_brings_a_domain_over_from_a_scratch_catalog(scratch, tmp_path):
    dst = tmp_path.parent / (tmp_path.name + "-dst")      # beside the scratch root, not in it (a package anywhere under a root is its)
    shutil.copytree(scratch / "tools", dst / "tools")
    (dst / "domains").mkdir()
    kw = catalog.take("dog", scratch, dst)
    assert kw["host"] == (scratch / "domains/dog/dog_host.py").read_text() and kw["corpus"] and kw["readme"]
    out = catalog.apply("dog", r=dst, **kw)
    assert "domains/dog/dog_host.py" in out["written"] and "domains/dog/intent/dog-corpus.tsv" in out["written"]
    for f in ("dog.llp", "dog_host.py", "domain.yaml", "intent.yaml", "README.md", *(
            "intent/" + n for n in ("dog.gbnf", "dog-prompt.txt", "dog-spoken.txt", "dog-corpus.tsv"))):
        assert (dst / "domains/dog" / f).read_text() == (scratch / "domains/dog" / f).read_text(), f
    # the package whole (release-r02 step 2b): its goldens (its contract) and its cases, as they are; never build/
    own = [p.relative_to(scratch / "domains/dog").as_posix() for sub in ("goldens", "cases")
           for p in sorted((scratch / "domains/dog" / sub).rglob("*")) if p.is_file()]
    assert "goldens/day.jsonl" in own and "cases/sit.yaml" in own
    for f in own:
        assert (dst / "domains/dog" / f).read_text() == (scratch / "domains/dog" / f).read_text(), f
    assert not (dst / "domains/dog/build").exists() and not (dst / "domains/dog/records").exists()
    with pytest.raises(ValueError, match="carried file"):
        catalog.plan("dog", carried={"build/x": "x"}, r=dst)
    assert catalog.fingerprint("dog", dst) == catalog.fingerprint("dog", scratch)
    with pytest.raises(ValueError, match="already"):
        catalog.take("dog", scratch, dst)
    # the plant's extension, and a README given in place of its own
    kw = catalog.take("plant", scratch, dst, readme="# plant\n")
    assert list(kw["extensions"]) == ["visitor"]
    catalog.apply("plant", r=dst, **kw)
    assert (dst / "domains/plant/extensions/visitor.llp").read_text() == \
        (scratch / "domains/plant/extensions/visitor.llp").read_text()
    assert (dst / "domains/plant/README.md").read_text() == "# plant\n"
    # a manifest key whose file take does not carry
    mf = scratch / "domains/dog/domain.yaml"
    mf.write_text(mf.read_text() + "agent: agent.py\n")
    with pytest.raises(ValueError, match="agent"):
        catalog.take("dog", scratch, tmp_path / "other")
    # a host's text needs a file host to land in
    with pytest.raises(ValueError, match="file.py:factory"):
        catalog.plan("bell", program="% the bell\n", host="# a host\n", r=dst)


def _cli(*args, root=None, stdin=None):
    env = dict(os.environ)
    if root:
        env["METIS_CATALOG"] = str(root)
    r = subprocess.run([sys.executable, "-m", "metis.catalog", *args], capture_output=True, text=True,
                       env=env, input=stdin, timeout=120)
    return r.returncode, json.loads(r.stdout) if r.stdout.strip().startswith(("{", "[")) else r.stdout, r.stderr


def test_the_cli_in_json(scratch):
    rc, rows, _ = _cli("list", "--json", root=scratch)
    assert rc == 0 and json.loads(json.dumps(catalog.index(scratch), default=str)) == rows
    rc, rows, _ = _cli("list", "--consoles", "--json", root=scratch)
    assert rc == 0 and rows == json.loads(json.dumps(catalog.listing(scratch), default=str))
    rc, dog, _ = _cli("get", "dog", "--json", root=scratch)
    assert rc == 0 and dog["fingerprint"] == catalog.fingerprint("dog", scratch) and dog["has"]["intent"]
    rc, it, _ = _cli("intents", "dog", "--spec", "--json", root=scratch)
    import yaml
    assert rc == 0 and it["spec"] == yaml.safe_load((scratch / "domains/dog/intent.yaml").read_text())
    assert it["intent"]["spec"].endswith("domains/dog/intent.yaml")
    rc, sk, _ = _cli("intents", "plant", "--skeleton", "--json", root=scratch)
    assert rc == 0 and sk["skeleton"] == catalog.skeleton("plant", scratch)
    prog = (scratch / "domains/dog/dog.llp").read_text() + "% cli\n"
    rc, dry, _ = _cli("apply", "dog", "--program", "-", "--dry-run", "--json", root=scratch, stdin=prog)
    assert rc == 0 and dry["dry_run"] and [f["path"] for f in dry["files"]] == ["domains/dog/dog.llp"]
    assert "% cli" not in (scratch / "domains/dog/dog.llp").read_text()
    rc, out, _ = _cli("apply", "dog", "--program", "-", "--json", root=scratch, stdin=prog)
    assert rc == 0 and out["written"] == ["domains/dog/dog.llp"] and out["fingerprint"]
    rc, err, _ = _cli("apply", "dog", "--manifest", '{"colour": 1}', "--json", root=scratch)
    assert rc == 1 and "unknown key" in err["error"]
    rc, err, _ = _cli("get", "nosuch", "--json", root=scratch)
    assert rc == 1 and "nosuch" in err["error"]
    dst = scratch.parent / "dst"
    (dst / "domains").mkdir(parents=True)
    shutil.copytree(scratch / "tools", dst / "tools")
    (scratch / "domains/plant/intent/plant-spoken.txt").unlink()   # an empty generated file: take writes none
    rc, dry, _ = _cli("--root", str(dst), "take", "plant", "--from", str(scratch), "--dry-run", "--json",
                      root=scratch)                       # --root wins over $METIS_CATALOG (the source's, in pi)
    assert rc == 0 and "domains/plant/plant_host.py" in [f["path"] for f in dry["files"]]
    assert not (dst / "domains/plant").exists()
    rc, out, _ = _cli("--root", str(dst), "take", "plant", "--from", str(scratch), "--json")
    assert rc == 0 and out["fingerprint"] == catalog.fingerprint("plant", scratch)
    rc, err, _ = _cli("--root", str(dst), "take", "plant", "--from", str(scratch), "--json")
    assert rc == 1 and "already" in err["error"]


def test_readme_from_a_run_and_the_bundle(scratch, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(scratch, run / "catalog")
    (run / "catalog/domains/plant/README.md").unlink()
    (run / "records").mkdir()
    (run / "records/formalize-calls.jsonl").write_text(
        json.dumps({"model": "m/x", "cost": 0.25}) + "\n" + json.dumps({"model": "m/x", "cost": 0.5}) + "\n")
    text = catalog.readme("plant", run / "catalog", run)
    assert text.startswith("# plant — ") and "on `m/x` (formalize: 2 calls, 0.7500 USD" in text
    assert "`extensions/visitor.llp`: `visitor_waters [dyn] : dry * visitor -o moist.`" in text
    assert "`metis --plugin plant_host:make plant.llp`" in text
    (run / "seed.json").write_text(json.dumps({"seed": "domains/fiction/x.llp"}))
    assert "seeded from `domains/fiction/x.llp`" in catalog.readme("plant", run / "catalog", run)
    dst = tmp_path / "dst"
    (dst / "domains").mkdir(parents=True)
    shutil.copytree(scratch / "tools", dst / "tools")
    rc, out, _ = _cli("--root", str(dst), "take", "plant", "--from", str(run / "catalog"), "--readme-from-run", str(run),
                      "--json")
    assert rc == 0 and (dst / "domains/plant/README.md").read_text() == catalog.readme("plant", run / "catalog", run)
    rc, err, _ = _cli("--root", str(dst), "take", "dog", "--from", str(scratch), "--readme", "-",
                      "--readme-from-run", str(run), "--json", stdin="x")
    assert rc == 1 and "not both" in err["error"]
    rc, b, _ = _cli("--root", str(dst), "build", "plant", "--json")
    assert rc == 0 and "plant.metis" in b["bundle"] and (dst / "domains/plant/build/plant.metis").exists()
    rc, err, _ = _cli("--root", str(dst), "build", "nosuch", "--json")
    assert rc == 1 and "nosuch" in err["error"]


def test_a_host_stays_in_its_domain(scratch, tmp_path):
    # review R1: a manifest host out of the domain (../dog/victim.py) would make take and
    # plan write another domain's file
    (scratch / "domains/dog/victim.py").write_text("# the dog's own\n")
    mf = scratch / "domains/plant/domain.yaml"
    mf.write_text(mf.read_text().replace("host: plant_host.py:make", "host: ../dog/victim.py:make"))
    assert "../dog/victim.py" in mf.read_text()
    dst = tmp_path / "dst"
    (dst / "domains/dog").mkdir(parents=True)
    (dst / "domains/dog/victim.py").write_text("# the target's dog\n")
    with pytest.raises(ValueError, match="not beside its domain.yaml"):
        catalog.take("plant", scratch, dst)
    with pytest.raises(ValueError, match="not beside its domain.yaml"):
        catalog.plan("plant", host="# mine\n", r=scratch)
    with pytest.raises(ValueError, match="not beside its domain.yaml"):
        catalog.plan("bell", program="% b\n", manifest={"host": "../dog/victim.py:make"}, host="# x\n", r=scratch)
    with pytest.raises(ValueError, match="not beside its domain.yaml"):
        catalog.plan("bell", program="% b\n", manifest={"host": "sub/h.py:make"}, host="# x\n", r=scratch)
    assert (dst / "domains/dog/victim.py").read_text() == "# the target's dog\n"
    assert catalog.plan("bell", program="% b\n", manifest={"host": "bell_host.py:make"}, host="# x\n",
                        r=scratch)[-1]["path"] == "domains/bell/domain.yaml"


def test_apply_readme_host_corpus_and_the_readme_verb(scratch, tmp_path):
    # review R3: the CLI's apply writes what py's apply writes (readme, host, corpus)
    for n, t in (("r", "# plant, by hand\n"), ("h", "# the plant's host, again\n"), ("c", "water it\tWATER\n")):
        (tmp_path / n).write_text(t)
    rc, dry, _ = _cli("apply", "plant", "--readme", str(tmp_path / "r"), "--host", str(tmp_path / "h"), "--corpus", "-",
                      "--dry-run", "--json", root=scratch, stdin="water it\tWATER\n")
    assert rc == 0 and [f["path"] for f in dry["files"]] == [
        "domains/plant/README.md", "domains/plant/intent/plant-corpus.tsv", "domains/plant/plant_host.py"]
    rc, out, _ = _cli("apply", "plant", "--readme", str(tmp_path / "r"), "--host", str(tmp_path / "h"), "--corpus",
                      str(tmp_path / "c"), "--json", root=scratch)
    assert rc == 0 and (scratch / "domains/plant/README.md").read_text() == "# plant, by hand\n"
    assert (scratch / "domains/plant/plant_host.py").read_text() == "# the plant's host, again\n"
    assert (scratch / "domains/plant/intent/plant-corpus.tsv").read_text() == "water it\tWATER\n"
    rc, err, _ = _cli("apply", "plant", "--readme", "-", "--corpus", "-", "--json", root=scratch, stdin="x")
    assert rc == 1 and "one text from stdin" in err["error"]
    rc, rd, _ = _cli("readme", "plant", "--json", root=scratch)
    assert rc == 0 and rd["readme"] == catalog.readme("plant", scratch)
    run = tmp_path / "run"
    (run / "records").mkdir(parents=True)
    (run / "seed.json").write_text(json.dumps({"seed": "domains/fiction/x.llp"}))
    rc, rd, _ = _cli("readme", "plant", "--from", str(scratch), "--run", str(run), "--json")
    assert rc == 0 and rd["readme"] == catalog.readme("plant", scratch, run) and "seeded from" in rd["readme"]


# ── loop-fixes s-855d8067 ────────────────────────────────────────────────────

METISC = catalog.metisc()
if not METISC and os.environ.get("METIS_LOOP_REQUIRE_METISC"):     # CI builds it: an edit's check must run there
    raise RuntimeError("METIS_LOOP_REQUIRE_METISC: no metisc (METISC or PATH)")
needs_metisc = pytest.mark.skipif(not METISC, reason="no metisc (METISC or PATH)")


def changed(diff):
    """A unified diff's removed and added lines (no headers)."""
    ls = diff.splitlines()
    return ([l[1:] for l in ls if l.startswith("-") and not l.startswith("---")],
            [l[1:] for l in ls if l.startswith("+") and not l.startswith("+++")])


@needs_metisc
def test_an_edit_changes_exactly_its_text_and_keeps_every_comment(scratch):
    prog = scratch / "domains/plant/plant.llp"
    before = prog.read_text()
    old = "% Made in the loop console (new)."
    kw = catalog.edit("plant", "plant.llp", old, "% Made in the loop console (new), edited.", r=scratch)
    assert list(kw) == ["program"] and kw["program"] == before.replace(old, "% Made in the loop console (new), edited.")
    files = catalog.plan("plant", **kw, r=scratch)
    assert [f["path"] for f in files] == ["domains/plant/plant.llp"]
    assert changed(files[0]["diff"]) == (["% Made in the loop console (new). Say in prose what there is, what can"],
                                         ["% Made in the loop console (new), edited. Say in prose what there is, what can"])
    out = catalog.apply("plant", **kw, r=scratch)
    assert out["written"] == ["domains/plant/plant.llp"]
    after = prog.read_text()
    assert [l for l in after.splitlines() if l.startswith("%")] == \
        [l.replace("(new).", "(new), edited.") for l in before.splitlines() if l.startswith("%")]


@needs_metisc
def test_an_edit_is_refused_unless_it_is_exact_inside_and_adds_no_diagnostic(scratch):
    with pytest.raises(ValueError, match="not in it"):
        catalog.edit("plant", "plant.llp", "no such text", "x", r=scratch)
    with pytest.raises(ValueError, match="occurs [0-9]+ times"):
        catalog.edit("plant", "plant.llp", "pred", "PRED", r=scratch)
    with pytest.raises(ValueError, match="empty"):
        catalog.edit("plant", "plant.llp", "", "x", r=scratch)
    with pytest.raises(ValueError, match="the same"):
        catalog.edit("plant", "plant.llp", "catalog plant 1.", "catalog plant 1.", r=scratch)
    with pytest.raises(ValueError, match="not inside"):
        catalog.edit("plant", "../dog/dog.llp", "pred", "PRED", r=scratch)
    with pytest.raises(ValueError, match="generated from intent.yaml"):
        catalog.edit("plant", "intent/plant.gbnf", "root", "ROOT", r=scratch)
    with pytest.raises(ValueError, match="no records/x to edit"):
        catalog.edit("plant", "records/x", "a", "b", r=scratch)
    # a rule's atom misspelt: metisc's new diagnostic refuses it, and names it
    rule = next(l for l in (scratch / "domains/plant/plant.llp").read_text().splitlines() if " -o moist" in l)
    with pytest.raises(ValueError, match="(?s)adds 1 diagnostic.*pred-undeclared") as e:
        catalog.edit("plant", "plant.llp", rule, rule.replace("-o moist", "-o moistt"), r=scratch)
    assert "moistt" in str(e.value)
    # every occurrence with replace_all: the program's own rename of an atom passes …
    kw = catalog.edit("plant", "plant.llp", "sun", "light", replace_all=True, r=scratch)
    assert "sun" not in kw["program"]
    # … but not one its extension still reads (review R3): visitor.llp, against the edited program
    with pytest.raises(ValueError, match="(?s)extensions/visitor.llp: error pred-undeclared.*'moist'"):
        catalog.edit("plant", "plant.llp", "moist", "wet", replace_all=True, r=scratch)


@needs_metisc
def test_an_edit_routes_each_file_through_applys_own_checks(scratch):
    d = scratch / "domains/plant"
    assert list(catalog.edit("plant", "README.md", "plant", "Plant", r=scratch, replace_all=True)) == ["readme"]
    ext = catalog.edit("plant", "extensions/visitor.llp", "my soil is moist again", "my soil is wet again", r=scratch)
    assert list(ext) == ["as_is"] and [Path(k).name for k in ext["as_is"]] == ["visitor.llp"]
    assert [f["path"] for f in catalog.plan("plant", **ext, r=scratch)] == ["domains/plant/extensions/visitor.llp"]
    with pytest.raises(ValueError, match="(?s)adds .*undeclared predicate 'guest'"):   # checked against the program
        catalog.edit("plant", "extensions/visitor.llp", "* visitor -o", "* guest -o", r=scratch)
    it = catalog.edit("plant", "intent.yaml", "potted plant", "pot plant", replace_all=True, r=scratch)
    assert [f["path"] for f in catalog.plan("plant", **it, r=scratch)][0] == "domains/plant/intent/plant-prompt.txt"
    bad = catalog.edit("plant", "intent.yaml", "verbs:", "verbz:", r=scratch)
    with pytest.raises(ValueError, match="does not generate"):
        catalog.plan("plant", **bad, r=scratch)
    mf = catalog.edit("plant", "domain.yaml", "records: records/", "colour: 1\nrecords: records/", r=scratch)
    with pytest.raises(ValueError, match="unknown key"):                       # the loader's own check
        catalog.plan("plant", **mf, r=scratch)
    line = (d / "plant_host.py").read_text().splitlines()[0]
    assert list(catalog.edit("plant", "plant_host.py", line, line + " (edited)", r=scratch)) == ["host"]


@needs_metisc
def test_the_cli_edit_and_since(scratch, tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.write_text("% Made in the loop console (new).")
    new.write_text("% Made in the loop console.")
    rc, dry, _ = _cli("--root", str(scratch), "edit", "plant", "plant.llp", "--old", str(old), "--new", str(new),
                      "--dry-run", "--json")
    assert rc == 0 and dry["files"][0]["text"].count("% Made in the loop console. Say") == 1
    assert "(new)" in (scratch / "domains/plant/plant.llp").read_text()
    refused = dry["files"][0]["text"].replace("console.", "console, refused.")
    since = tmp_path / "since.json"
    since.write_text(json.dumps({"domains/plant/plant.llp": refused}))
    rc, dry, _ = _cli("--root", str(scratch), "edit", "plant", "plant.llp", "--old", str(old), "--new", str(new),
                      "--since", str(since), "--dry-run", "--json")
    sd = dry["files"][0]["since_diff"]
    assert rc == 0 and "-% Made in the loop console, refused." in sd and "+% Made in the loop console." in sd
    assert len(changed(sd)[0]) == 1 and "--- refused/domains/plant/plant.llp" in sd
    rc, out, _ = _cli("--root", str(scratch), "edit", "plant", "plant.llp", "--old", str(old), "--new", str(new), "--json")
    assert rc == 0 and out["written"] == ["domains/plant/plant.llp"]
    rc, err, _ = _cli("--root", str(scratch), "edit", "plant", "plant.llp", "--old", str(old), "--new", str(new), "--json")
    assert rc == 1 and "not in it" in err["error"]
    rc, err, _ = _cli("--root", str(scratch), "edit", "plant", "plant.llp", "--old", "-", "--new", "-", "--json")
    assert rc == 1 and "one text from stdin" in err["error"]


def test_since_on_a_plan(scratch):
    rd = "# plant\n\nOne.\nTwo.\n"
    f = catalog.plan("plant", readme=rd, r=scratch, since={"domains/plant/README.md": "# plant\n\nOne.\nTwi.\n"})[0]
    assert changed(f["since_diff"]) == (["Twi."], ["Two."]) and f["text"] == rd
    assert "since_diff" not in catalog.plan("plant", readme=rd, r=scratch)[0]


def test_display_and_line_go_into_the_manifest(scratch):
    files = catalog.plan("plant", display="the plant", line="a potted plant that dries out", r=scratch)
    text = files[0]["text"]
    assert [f["path"] for f in files] == ["domains/plant/domain.yaml"]
    import yaml
    m = yaml.safe_load(text)
    assert m["display"] == "the plant" and m["line"] == "a potted plant that dries out"
    new = catalog.plan("lamp2", program="catalog lamp2 1.\n", display="the lamp", line="a lamp", r=scratch)
    m = yaml.safe_load(next(f for f in new if f["path"].endswith("domain.yaml"))["text"])
    assert m == {"program": "lamp2.llp", "display": "the lamp", "line": "a lamp"}
    whole = catalog.plan("plant", manifest="program: plant.llp\n", display="the plant", r=scratch)
    assert yaml.safe_load(whole[0]["text"]) == {"program": "plant.llp", "display": "the plant"}
    with pytest.raises(ValueError, match="one line"):
        catalog.plan("plant", line="two\nlines", r=scratch)
    uni = catalog.plan("plant", line="the pigs: ⊗, &, 1 and ⊕", r=scratch)[0]["text"]   # as written, not \u escapes
    assert "line: 'the pigs: ⊗, &, 1 and ⊕'\n" in uni and yaml.safe_load(uni)["line"] == "the pigs: ⊗, &, 1 and ⊕"
    rc, out, _ = _cli("--root", str(scratch), "apply", "plant", "--display", "the plant", "--line", "dries out", "--json")
    assert rc == 0 and catalog.get("plant", scratch).line == "dries out"


NOCMD = """% a world that only happens: no commands
% (a header of two lines)
catalog quiet 1.
pred a : w.
pred b : w.
stage s {
  %% a becomes b.
  ab [w] : a -o b.
}
#init a 1
#steps 2
"""


def test_a_world_with_no_commands_gets_no_draft(scratch, monkeypatch):
    catalog.apply("quiet", program=NOCMD, r=scratch)
    assert catalog.commands("quiet", scratch) == [] and catalog.commands("plant", scratch)
    sk = catalog.skeleton("quiet", scratch, intentless=False)
    assert sk.startswith(catalog.NO_COMMANDS.format(n="quiet")) and "DRAFT" not in sk
    import yaml
    assert [v["verb"] for v in yaml.safe_load(sk)["verbs"]] == ["UNKNOWN"]
    assert "a DRAFT" in catalog.skeleton("plant", scratch, intentless=True).split("\n")[0]   # a real draft keeps it
    assert catalog.skeleton("quiet", scratch, intentless=True) is None
    files = catalog.plan("quiet", intents=sk, r=scratch)
    assert "domains/quiet/intent/quiet-spoken.txt" not in [f["path"] for f in files]      # empty: not written
    assert "domains/quiet/intent/quiet.gbnf" in [f["path"] for f in files]
    env = {k: v for k, v in os.environ.items() if k != "METIS_CATALOG_INTENTLESS"}   # the default: intentless
    p = subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(scratch), "apply", "quiet",
                        "--intents-skeleton", "--json"], capture_output=True, text=True, env=env)
    assert p.returncode == 1 and "takes no commands" in json.loads(p.stdout)["error"]
    p = subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(scratch), "intents", "quiet",
                        "--skeleton", "--json"], capture_output=True, text=True, env=env)
    assert json.loads(p.stdout) == {"name": "quiet", "skeleton": None, "commands": []}
    p = subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(scratch), "intents", "quiet",
                        "--skeleton", "--json"], capture_output=True, text=True, env=dict(env, METIS_CATALOG_INTENTLESS="0"))
    assert json.loads(p.stdout)["skeleton"] == sk                             # the switch off: UNKNOWN only


def test_the_readme_writes_its_mechanical_parts(scratch, tmp_path):
    catalog.apply("quiet", program=NOCMD, r=scratch)
    run = tmp_path / "run"
    run.mkdir()
    (run / "seed.json").write_text(json.dumps({"seed": "domains/x/quiet.llp", "name": "quiet"}))
    text = catalog.readme("quiet", scratch, run, summary="An a turns into a b.\n", title="a world that only happens")
    assert text.startswith("# quiet — a world that only happens\n\nAn a turns into a b.\n\n**Where it came from:**")
    assert "seeded from `domains/x/quiet.llp`" in text
    assert "```text\n% a world that only happens: no commands\n% (a header of two lines)\n```" in text
    assert "`metisc --run quiet.llp` runs the case its `#` lines give:\n\n```text\n#init a 1\n#steps 2\n```" in text
    assert "`metis quiet.llp`" in text
    assert text.rstrip().endswith("so it has no intent.yaml: the agents open it with no verbs.")
    catalog.apply("quiet", intents=catalog.skeleton("quiet", scratch, intentless=False), r=scratch)
    assert "`intent.yaml` knows only UNKNOWN" in catalog.readme("quiet", scratch)
    assert catalog.readme("quiet", scratch).startswith("# quiet\n\n**Where")       # no title, no prose: the name
    plant = catalog.readme("plant", scratch)
    assert "the verbs `WATER`, `CHECK` (and UNKNOWN). Its check: `python3 tools/intents.py check plant`." in plant
    rc, out, _ = _cli("--root", str(scratch), "readme", "quiet", "--summary", "-", "--title", "t", "--json",
                      stdin="S.\n")
    assert rc == 0 and out["readme"].startswith("# quiet — t\n\nS.\n")


@needs_metisc
def test_an_edit_of_a_program_that_includes(scratch):
    """review R1: the candidate is diagnosed beside the original, so its include resolves."""
    d = scratch / "domains/inc"
    d.mkdir()
    (d / "inc_rules.llp").write_text("% the lifted part\ncatalog inc-rules 1.\nlayers (story).\nstages (s).\n"
                                     "namespace w produce (story).\npred a : w.\npred b : w.\nstage s {\n"
                                     "  %% a becomes b.\n  ab [story] : a -o b.\n}\n")
    (d / "inc.llp").write_text('% the instance: its header\ncatalog inc 1.\nprovenance hand "t" "2026-10-18".\n'
                               'include "inc_rules.llp".\n#init a 1\n#steps 2\n')
    (d / "domain.yaml").write_text("program: inc.llp\n")
    assert catalog.diagnostics(d / "inc.llp") == []
    kw = catalog.edit("inc", "inc.llp", "its header", "its header, edited", r=scratch)
    assert catalog.apply("inc", **kw, r=scratch)["written"] == ["domains/inc/inc.llp"]
    with pytest.raises(ValueError, match="(?s)adds 1 diagnostic.*metisc-failed.*nope.llp"):
        catalog.edit("inc", "inc.llp", '"inc_rules.llp"', '"nope.llp"', r=scratch)
    assert sorted(f.name for f in d.iterdir()) == ["domain.yaml", "inc.llp", "inc_rules.llp"]   # no dot-file left


@needs_metisc
def test_an_extension_edit_writes_back_its_own_file(scratch):
    """review R2: extensions/Wear-2.llp stays Wear-2.llp (plan's extensions= would write wear_2.llp beside it)."""
    d = scratch / "domains/plant"
    (d / "extensions/visitor.llp").rename(d / "extensions/Wear-2.llp")
    assert [Path(e).name for e in catalog.get("plant", scratch).extensions()] == ["Wear-2.llp"]
    kw = catalog.edit("plant", "extensions/Wear-2.llp", "moist again", "wet again", r=scratch)
    files = catalog.plan("plant", **kw, r=scratch)
    assert [(f["path"], f["new"]) for f in files] == [("domains/plant/extensions/Wear-2.llp", False)]
    catalog.apply("plant", **kw, r=scratch)
    assert sorted(f.name for f in (d / "extensions").iterdir()) == ["Wear-2.llp"]
    with pytest.raises(ValueError, match="existing file of the domain"):
        catalog.plan("plant", as_is={scratch / "domains/dog/dog.llp": "x"}, r=scratch)


@needs_metisc
def test_an_edit_may_repair_a_file_that_does_not_compile(scratch):
    """review note 1: a failing metisc before the edit is a diagnostic, not a refusal of every edit."""
    prog = scratch / "domains/plant/plant.llp"
    good = prog.read_text()
    prog.write_text(good.replace("catalog plant 1.", "catalog plant 1"))       # a parse error
    assert [x["code"] for x in catalog.diagnostics(prog)] == ["metisc-failed"]
    kw = catalog.edit("plant", "plant.llp", "catalog plant 1\n", "catalog plant 1.\n", r=scratch)
    assert kw["program"] == good
    prog.write_text(good)
    with pytest.raises(ValueError, match="metisc-failed"):                     # and breaking it is refused
        catalog.edit("plant", "plant.llp", "catalog plant 1.\n", "catalog plant 1\n", r=scratch)


def test_one_line_means_no_line_break_of_any_kind(scratch):
    """review note 2: U+2028, U+0085 and the other breaks str.splitlines knows are refused."""
    for v in ("a\u2028b", "a\x85b", "a\rb", "a\x0cb"):
        with pytest.raises(ValueError, match="one line"):
            catalog.plan("plant", line=v, r=scratch)


def test_the_readme_refuses_or_survives_odd_inputs(scratch):
    """review note 4: a title of two lines; provenance text "."; an intent.yaml that does not read."""
    with pytest.raises(ValueError, match="title is one line"):
        catalog.readme("plant", scratch, title="two\nlines")
    prog = scratch / "domains/plant/plant.llp"
    prog.write_text(prog.read_text().replace('provenance text "a potted plant that dries out: the person waters it or asks how it is"',
                                             'provenance text "."'))
    assert catalog.readme("plant", scratch).startswith("# plant\n")
    (scratch / "domains/plant/intent.yaml").write_text("verbs: [}\n")
    assert "does not read as YAML" in catalog.readme("plant", scratch)
    (scratch / "domains/plant/intent.yaml").write_text("verbs: [WATER, {verb: [CHECK, 3]}]\n")
    assert "the verbs `CHECK` (and UNKNOWN)" in catalog.readme("plant", scratch)


def test_commands_follow_the_skeletons_rule(scratch):
    """review note 6: the SDK's commands() and intents.py's skeleton read the same atoms."""
    import yaml
    prog = scratch / "domains/plant/plant.llp"
    prog.write_text(prog.read_text() + "pred cmd_go(soil, soil) : plant.\npred cmd_nope.\npred  cmd_spaced  : plant.\n")
    verbs = [v["verb"] for v in yaml.safe_load(catalog.skeleton("plant", scratch))["verbs"] if v["verb"] != "UNKNOWN"]
    assert [c[4:].upper() for c in catalog.commands("plant", scratch)] == verbs
    assert "cmd_nope" not in catalog.commands("plant", scratch) and "cmd_go" in catalog.commands("plant", scratch)


def _inc_domain(root):
    d = root / "domains/inc"
    d.mkdir(parents=True)
    (d / "inc_rules.llp").write_text("% the lifted part\ncatalog inc-rules 1.\nlayers (story).\nstages (s).\n"
                                     "namespace w produce (story).\npred a : w.\npred b : w.\nbwd near(cell).\nstage s {\n"
                                     "  %% a becomes b.\n  ab [story] : a -o b.\n}\n")
    (d / "inc.llp").write_text('% the instance: its header\ncatalog inc 1.\nprovenance hand "t" "2026-10-18".\n'
                               'type cell {c1 c2}.\ninclude "inc_rules.llp".\nnear(c1).\n#init a 1\n#steps 2\n')
    (d / "domain.yaml").write_text("program: inc.llp\n")
    return d


def test_take_carries_the_files_a_program_includes(scratch, tmp_path):
    """rddl-worlds B: a package seed (nav: nav.llp includes nav_domain.llp) is taken whole."""
    _inc_domain(scratch)
    assert catalog.includes(scratch / "domains/inc", [scratch / "domains/inc/inc.llp"]) == \
        {"inc_rules.llp": (scratch / "domains/inc/inc_rules.llp").read_text()}
    dst = tmp_path.parent / (tmp_path.name + "-dst")      # beside the scratch root, not in it (a package anywhere under a root is its)
    (dst / "domains").mkdir(parents=True)
    out = catalog.apply("inc", **catalog.take("inc", scratch, dst), r=dst)
    assert sorted(out["written"]) == ["domains/inc/domain.yaml", "domains/inc/inc.llp", "domains/inc/inc_rules.llp"]
    assert (dst / "domains/inc/inc_rules.llp").read_text() == (scratch / "domains/inc/inc_rules.llp").read_text()
    (scratch / "domains/inc/inc.llp").write_text('catalog inc 1.\ninclude "../plant/plant.llp".\n')
    with pytest.raises(ValueError, match="outside its domain directory"):
        catalog.take("inc", scratch, tmp_path / "dst2")
    with pytest.raises(ValueError, match="a .llp inside the domain"):
        catalog.plan("inc", included={"../plant/x.llp": "x"}, r=scratch)


def test_take_carries_the_notch_its_manifest_names(scratch, tmp_path):
    """rddl-worlds step 3: wildfire and tamarisk are packages with their fitted notch."""
    src = CAT / "domains" / "wildfire"
    if not (src / "domain.yaml").is_file():
        pytest.skip("no wildfire package in the catalog")
    shutil.copytree(src, scratch / "domains/wildfire", ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    dst = tmp_path.parent / (tmp_path.name + "-dst")      # beside the scratch root, not in it (a package anywhere under a root is its)
    (dst / "domains").mkdir(parents=True)
    kw = catalog.take("wildfire", scratch, dst)
    assert kw["notch"] == (src / "wildfire.notch.json").read_text()
    files = {f["path"]: f for f in catalog.plan("wildfire", **kw, r=dst)}
    assert files["domains/wildfire/wildfire.notch.json"]["new"]       # shown at the gate, a file of its own
    catalog.apply("wildfire", **kw, r=dst)
    for f in ("wildfire.llp", "wildfire.notch.json", "wildfire_host.py", "domain.yaml", "README.md"):
        assert (dst / "domains/wildfire" / f).read_text() == (src / f).read_text(), f
    assert catalog.get("wildfire", dst).notch == (dst / "domains/wildfire/wildfire.notch.json").resolve()
    # a notch is carried, never edited (brief v2: no fitting): catalog_edit's edit() refuses it
    with pytest.raises(ValueError, match="not one of the domain's files an edit changes"):
        catalog.edit("wildfire", "wildfire.notch.json", '"rows"', '"rowz"', r=dst)
    # a manifest's notch that is not there
    (scratch / "domains/wildfire/wildfire.notch.json").unlink()
    with pytest.raises(ValueError, match="not there"):
        catalog.take("wildfire", scratch, tmp_path / "dst2")


def test_plan_writes_a_notch_where_the_manifest_names_it(scratch):
    n = '{"rows": []}\n'
    with pytest.raises(ValueError, match="not JSON"):
        catalog.plan("dog", notch="{rows", r=scratch)
    files = {f["path"]: f for f in catalog.plan("dog", notch=n, r=scratch)}   # no notch named: <name>.notch.json
    assert files["domains/dog/dog.notch.json"]["text"] == n
    assert "notch: dog.notch.json" in files["domains/dog/domain.yaml"]["text"]
    with pytest.raises(ValueError, match="beside its domain.yaml"):
        catalog.plan("dog", notch=n, manifest={"notch": "../plant/x.notch.json"}, r=scratch)
    with pytest.raises(ValueError, match="beside its domain.yaml"):
        catalog.plan("dog", notch=n, manifest={"notch": "dog.json"}, r=scratch)


def test_take_and_edit_keep_a_files_bytes(scratch, tmp_path):
    """rddl-worlds step 3: elevators' header is verbatim from its RDDL file, CRLF; take
    and apply wrote it LF (universal newlines). Now the bytes are kept, and an edit
    given in LF applies to a CRLF file and keeps its line ends."""
    d = scratch / "domains/dog"
    prog = d / "dog.llp"
    lines = prog.read_bytes().split(b"\n")
    prog.write_bytes(b"\r\n".join(lines[:3]) + b"\r\n" + b"\n".join(lines[3:]))   # a CRLF header, LF below
    dst = tmp_path / "dst"
    shutil.copytree(scratch / "tools", dst / "tools")
    (dst / "domains").mkdir()
    catalog.apply("dog", **catalog.take("dog", scratch, dst), r=dst)
    assert (dst / "domains/dog/dog.llp").read_bytes() == prog.read_bytes()
    assert catalog.plan("dog", program=catalog._text(prog), r=dst) == []          # the same bytes: nothing to write


@needs_metisc
def test_an_edit_in_lf_keeps_a_crlf_files_line_ends(scratch):
    """The edit half of the above (a program's edit is checked by metisc; sdk-alone has none)."""
    dst = scratch
    prog = dst / "domains/dog/dog.llp"
    lines = prog.read_bytes().split(b"\n")
    prog.write_bytes(b"\r\n".join(lines[:3]) + b"\r\n" + b"\n".join(lines[3:]))   # a CRLF header, LF below
    two = lines[0].decode() + "\n" + lines[1].decode()
    kw = catalog.edit("dog", "dog.llp", two, two + " (edited)", r=dst)
    catalog.apply("dog", **kw, r=dst)
    out = (dst / "domains/dog/dog.llp").read_bytes()
    assert out.count(b"\r\n") == 3 and b"(edited)\r\n" in out and out.endswith(b"\n".join(lines[3:]))


def test_the_gate_diff_marks_a_last_line_with_no_newline(scratch):
    """The gate shows the files' diffs joined: a file without a final newline ran
    on into the next file's header (`}--- a/…`)."""
    files = catalog.plan("dog", program="% the dog\ncatalog dog 9.", readme="# dog\n", r=scratch)
    prog = next(f for f in files if f["path"].endswith("dog.llp"))
    assert prog["diff"].endswith("+catalog dog 9.\n\\ No newline at end of file\n")
    joined = "".join(f["diff"] for f in files)
    assert "9.--- a/" not in joined and "9.\n\\ No newline at end of file\n--- a/" in joined


def test_take_says_how_to_name_a_program_it_does_not_carry(scratch, tmp_path):
    d = scratch / "domains/plant"
    (d / "plant.llp").rename(d / "plant2.llp")
    mf = d / "domain.yaml"
    mf.write_text(mf.read_text().replace("plant.llp", "plant2.llp"))
    with pytest.raises(ValueError, match=r"names plant2\.llp: rename the file to domains/plant/plant\.llp "
                                         r"and set program: plant\.llp"):
        catalog.take("plant", scratch, tmp_path / "dst")


@needs_metisc
def test_an_edit_of_an_included_file(scratch):
    """The included file is checked alone, and the program that includes it with the edit in its place."""
    d = _inc_domain(scratch)
    kw = catalog.edit("inc", "inc_rules.llp", "the lifted part", "the lifted part, edited", r=scratch)
    assert list(kw) == ["included"] and catalog.apply("inc", **kw, r=scratch)["written"] == ["domains/inc/inc_rules.llp"]
    # a rename the included file alone is fine with; the program's fact near(c1) is not: refused, naming the program
    assert catalog.diagnostics(d / "inc.llp") == []
    with pytest.raises(ValueError, match=r"(?s)adds 1 diagnostic.*\n  inc\.llp: error fact-not-bwd"):
        catalog.edit("inc", "inc_rules.llp", "bwd near(cell).", "bwd nearr(cell).", r=scratch)
    assert sorted(f.name for f in d.iterdir()) == ["domain.yaml", "inc.llp", "inc_rules.llp"]
