"""The sampler-wire triangle (docs/rewrite/06-sampler-wire.md §5) —
the FULL-CLASS gate: python referees (goldens), `metisc --sample`
reproduces every line from the .llp sources, C++ sample_run
reproduces every trace from the canonical bytes alone. The corpus
includes the programs the compiled wire cannot carry (the love
triangle's multiplicities) — nothing Ceptre executes is outside
THIS triangle. Skipped without the respective compilers."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from langpath import LANG  # noqa: E402
OCAML = shutil.which("ocamlopt")
CXX = shutil.which("g++") or shutil.which("clang++")


def _golden_lines():
    return [l.rstrip("\n")
            for l in (LANG / "corpus/goldens/sample_goldens.txt").open()
            if not l.startswith("#")]


@pytest.mark.slow
@pytest.mark.skipif(OCAML is None, reason="no OCaml compiler")
def test_ocaml_sample_parity():
    subprocess.run(["make", "-s", "-C", str(LANG / "src")],
                   check=True)
    r = subprocess.run(
        [str(LANG / "src/metisc"), "--sample",
         str(LANG / "corpus/sample_manifest.txt")],
        capture_output=True, text=True, check=True)
    mine = r.stdout.splitlines()
    gold = _golden_lines()
    assert len(mine) == len(gold), (len(mine), len(gold))
    for i, (g, m) in enumerate(zip(gold, mine)):
        assert g == m, f"line {i}:\n  gold : {g}\n  ocaml: {m}"


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_sample_parity(tmp_path):
    exe = tmp_path / "sample_run"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/sample_run.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run(
        [str(exe), str(LANG / "corpus/goldens/sample_goldens.txt")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"SAMPLE PARITY (\d+)/(\d+)", r.stdout)
    assert m and m.group(1) == m.group(2) and int(m.group(2)) > 0, \
        r.stdout


@pytest.mark.slow
def test_goldens_are_current():
    """Regenerating the goldens is a no-op — the checked-in file is
    what python produces today."""
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    from corpus import build_program, parse_manifest
    from metis.kernel import canonical_bytes, program_key
    from metis.kernel.refsample import canonical_events, \
        final_line, run_trace
    out = []
    for case in parse_manifest(LANG / "corpus/"
                               "sample_manifest.txt"):
        prog = build_program(case)
        out.append(f"B {case.name} {case.steps} "
                   f"{program_key(prog)}")
        out.append(canonical_bytes(prog).decode().rstrip("\n"))
        out.append(f"E {len(canonical_events(prog))}")
        for seed in range(1, case.seeds + 1):
            trace, final = run_trace(prog, case.steps, seed)
            out.append(f"R {seed} {len(trace)} "
                       + " ".join(str(i) for i in trace)
                       if trace else f"R {seed} 0")
            out.append(final_line(seed, final))
        out.append(".")
    assert "\n".join(out).splitlines() == _golden_lines()
