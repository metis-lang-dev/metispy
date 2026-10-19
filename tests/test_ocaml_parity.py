"""OCaml toolchain parity gates (docs/rewrite/).

Gate 1 (symbolic parity): the OCaml compiler (parse → compile →
typecheck → ground → emit → certify → serialize) must reproduce the
IR golden corpus byte-for-byte on every symbolic line — the referee
expectations (trailing M/Z floats) stay Python's.

Gate 2 (the triangle): the OCaml-emitted ARTIFACT, merged with
Python's referee expectations, is fed to the C++ interpreter, which
must reproduce every expectation by table ops along the
OCaml-shipped schedules. Compiler, referee and interpreter are then
three implementations meeting at one wire:
    OCaml compiles ≙ Python referees ≙ C++ interprets.

Gate 3 (canonical key): program_key now hashes the language-neutral
canonical serialization (docs/rewrite/03-canonical-key.md); the
OCaml toolchain must produce the SAME key for every corpus program
(plus the FIPS sha256 vectors via --selftest) — the precondition
for sharing the artifact cache across toolchains.

The corpus itself is manifest-driven: ocaml/parity/manifest.txt is
the single source (tests/corpus.py parses it; gen_ir_goldens.py and
the OCaml driver both consume it).

Skipped when the respective compilers are missing from PATH."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from langpath import LANG  # noqa: E402
OCAML = shutil.which("ocamlopt")
CXX = shutil.which("g++") or shutil.which("clang++")


def _strip(line):
    if line.startswith("M ") or line.startswith("Z "):
        return line.rsplit(" ", 1)[0]
    return line


def _ocaml_lines():
    subprocess.run(["make", "-s", "-C", str(LANG / "src")],
                   check=True)
    r = subprocess.run(
        [str(LANG / "src/metisc"),
         str(LANG / "corpus/manifest.txt")],
        capture_output=True, text=True, check=True)
    return [l for l in r.stdout.splitlines() if not l.startswith("#")]


def _golden_lines():
    return [l.rstrip("\n")
            for l in (LANG / "corpus/goldens/ir_goldens.txt").open()
            if not l.startswith("#")]


@pytest.mark.slow
@pytest.mark.skipif(OCAML is None, reason="no OCaml compiler")
def test_ocaml_symbolic_parity():
    mine = _ocaml_lines()
    gold = [_strip(l) for l in _golden_lines()]
    assert len(mine) == len(gold), (len(mine), len(gold))
    for i, (g, m) in enumerate(zip(gold, mine)):
        assert g == m, f"line {i}:\n  gold : {g}\n  ocaml: {m}"


@pytest.mark.slow
@pytest.mark.skipif(OCAML is None or CXX is None,
                    reason="needs OCaml and C++ compilers")
def test_ocaml_artifact_cpp_python_triangle(tmp_path):
    """OCaml artifact + Python expectations → C++ interpreter."""
    mine = _ocaml_lines()
    gold = _golden_lines()
    assert len(mine) == len(gold)
    merged = []
    for g, m in zip(gold, mine):
        if m.startswith(("M ", "Z ")):
            assert _strip(g) == m          # symbolic prefix agrees
            merged.append(m + " " + g.rsplit(" ", 1)[1])
        else:
            assert g == m
            merged.append(m)
    corpus = tmp_path / "ocaml_ir_corpus.txt"
    corpus.write_text("\n".join(merged) + "\n")
    exe = tmp_path / "metis_run"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/metis_run.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run([str(exe), str(corpus)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    scores = re.findall(r"IR PARITY (\d+)/(\d+)", r.stdout)
    assert len(scores) == 1, r.stdout
    ok, total = scores[0]
    assert ok == total and int(total) > 0, r.stdout


@pytest.mark.slow
@pytest.mark.skipif(OCAML is None, reason="no OCaml compiler")
def test_ocaml_program_key_parity():
    """The neutral canonical serialization: same program, same key,
    either toolchain — the artifact cache can be shared."""
    from corpus import build_case, parse_manifest
    from metis.kernel import program_key
    subprocess.run(["make", "-s", "-C", str(LANG / "src")],
                   check=True)
    r = subprocess.run([str(LANG / "src/metisc"), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "selftest ok" in r.stdout, \
        r.stdout + r.stderr
    manifest = LANG / "corpus/manifest.txt"
    r = subprocess.run(
        [str(LANG / "src/metisc"), "--keys", str(manifest)],
        capture_output=True, text=True, check=True)
    ocaml_keys = dict(l.split()[1:3]
                      for l in r.stdout.splitlines()
                      if l.startswith("K "))
    cases = parse_manifest(manifest)
    assert set(ocaml_keys) == {c.name for c in cases}
    for case in cases:
        prog, _ = build_case(case)
        assert ocaml_keys[case.name] == program_key(prog), case.name
    # include-splitting is identity-invariant: the instance file
    # adopting the domain file denotes the SAME program
    assert ocaml_keys["cascade-inc"] == ocaml_keys["cascade"]


@pytest.mark.slow
@pytest.mark.skipif(OCAML is None, reason="no OCaml compiler")
def test_ocaml_pretty_roundtrip():
    """The interpretability contract, parse(pretty(a)) == a, checked
    by the OCaml toolchain over every corpus source file."""
    subprocess.run(["make", "-s", "-C", str(LANG / "src")],
                   check=True)
    r = subprocess.run(
        [str(LANG / "src/metisc"), "--roundtrip",
         str(LANG / "corpus/manifest.txt")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith("R ")]
    assert lines and all(l.endswith(" ok") for l in lines), r.stdout


# -- gate 4 (bundle-compiler s-3985caf6 step 2): the bundle's ground.wire ----

def _packages():
    """(name, program, extensions) of every catalog package that serves a
    program metispy can ground (metis_catalog: installed, $METIS_CATALOG,
    else the sibling checkout)."""
    from metis.kernel.artifact import _catalog
    try:
        cat = _catalog()
    except (ImportError, RuntimeError):          # no metis-catalog: said by _catalog
        return []
    out = []
    for d in cat.domains():
        try:
            files = d.serves()
        except ValueError:
            continue
        if files:
            out.append((d.name, files[0], files[1:]))
    return out


@pytest.mark.skipif(OCAML is None, reason="no OCaml compiler")
def test_gate4_every_package_wire_is_metiscs(monkeypatch):
    """Every catalog package's ground.wire: `metisc --wire` (what
    metis.kernel.artifact serves when metisc is there) == metispy's
    emit_artifact (the cover), byte for byte; the whole bundle is the same
    either way, and the build says who compiled it."""
    from metis.kernel import artifact
    pkgs = _packages()
    if not pkgs:
        pytest.skip("metis-catalog not available")
    subprocess.run(["make", "-s", "-C", str(LANG / "src")], check=True)
    mc = str(LANG / "src/metisc")
    for name, program, exts in pkgs:
        monkeypatch.delenv("METISC", raising=False)
        monkeypatch.setenv("PATH", "/nonexistent")          # no metisc: the cover
        rep_py = {}
        py = artifact.build(program, exts, report=rep_py)
        monkeypatch.setenv("METISC", mc)
        rep_ml = {}
        ml = artifact.build(program, exts, report=rep_ml)
        assert rep_py["wire_by"].startswith("metispy") and rep_ml["wire_by"].startswith("metisc"), name
        assert ml["ground.wire"] == py["ground.wire"], f"{name}: metisc's wire differs from emit_artifact's"
        assert ml == py, name
    assert {"dog", "plant", "kitchen", "kitchen_shared"} <= {n for n, _, _ in pkgs}


def test_a_failing_metisc_is_an_error_not_a_cover(tmp_path, monkeypatch):
    from metis.kernel import artifact
    pkgs = [p for p in _packages() if p[0] == "plant"]
    if not pkgs:
        pytest.skip("metis-catalog not available")
    bad = tmp_path / "metisc"
    bad.write_text("#!/bin/sh\necho 'metisc: bad directive' >&2\nexit 2\n")
    bad.chmod(0o755)
    monkeypatch.setenv("METISC", str(bad))
    _, program, exts = pkgs[0]
    with pytest.raises(RuntimeError, match="metisc --wire plant.llp: exit 2: metisc: bad directive"):
        artifact.build(program, exts)
