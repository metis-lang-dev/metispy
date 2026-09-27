"""C++ parity gates: compile the golden CLI and run BOTH corpora —
the ground-program filter goldens (stage 16) and the IR corpus
(stage 19: factor-graph artifact + shipped schedules, the C++
interpreter reproducing python's marginals by table ops alone).
Skipped when no C++ compiler is on PATH."""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
from langpath import LANG  # noqa: E402
CXX = shutil.which("g++") or shutil.which("clang++")


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_parity_both_corpora(tmp_path):
    exe = tmp_path / "metis_run"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/metis_run.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run(
        [str(exe), str(LANG / "corpus/goldens/goldens.txt"),
         str(LANG / "corpus/goldens/ir_goldens.txt")],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    scores = re.findall(r"PARITY (\d+)/(\d+)", r.stdout)
    assert len(scores) == 2, r.stdout
    for ok, total in scores:
        assert ok == total and int(total) > 0, r.stdout


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_fw_smoke(tmp_path):
    """The BT.cpp-parity features (loader, blackboard, observer,
    substitution) — self-checking binary (docs/bt-parity.md)."""
    exe = tmp_path / "fw_smoke"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/fw_smoke.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "FW SMOKE OK" in r.stdout, \
        r.stdout + r.stderr


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_budget_smoke(tmp_path):
    """The wayplan-oom-hardening kernel API (max_states), pinned
    metis-side: default 0 = the exact measure; a crossed budget
    bails MID-step with overflowed() set and final() = the last
    COMPLETE slice (mass 1, never a partial slice); an uncrossed
    budget is inert."""
    exe = tmp_path / "budget_smoke"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/budget_smoke.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0 and "BUDGET SMOKE OK" in r.stdout, \
        r.stdout + r.stderr


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_evaluator(tmp_path):
    """The C++ Evaluator product surface (evaluator.hpp —
    fw.Evaluator parity; TODO 'C++ runtime / framework' item 1):
    artifact + B port manifest fail-fast, blackboard inputs/outputs,
    overlay-chained weight resolution, candidates-as-init-edits plan
    cycle, and the HOT-SWAP (v2 = one added rule recompiled by
    metisc flips the decision; corrupt artifact degrades to last
    good). Python fw.Evaluator generates the truth
    (gen_fw_artifact.py)."""
    subprocess.run(
        [sys.executable, str(ROOT / "tools/gen_fw_artifact.py"),
         str(tmp_path)],
        check=True, cwd=ROOT)
    exe = tmp_path / "evaluator_smoke"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/evaluator_smoke.cpp"), "-o",
         str(exe)],
        check=True)
    r = subprocess.run(
        [str(exe), str(tmp_path / "fw_rover_v1.txt"),
         str(tmp_path / "fw_rover_v2.txt"),
         str(tmp_path / "fw_rover_expect.txt"),
         str(tmp_path / "fw_rover_serve.txt")],
        capture_output=True, text=True)
    assert r.returncode == 0 and "OK (0 failures)" in r.stdout, \
        r.stdout + r.stderr


@pytest.mark.slow
@pytest.mark.skipif(CXX is None, reason="no C++ compiler")
def test_cpp_ir_eval(tmp_path):
    """The PRODUCT IR runtime (ir.hpp IrArtifact/IrEval — TODO 'C++
    runtime / framework' item 2): one loaded artifact with declared
    U slots + shipped schedules serves every per-tick evidence
    assignment — marginals, the likelihood surprise op (0 =
    unmodeled), the slot-candidate decision. Python FactoredProgram
    generates the truth (gen_ir_artifact.py)."""
    subprocess.run(
        [sys.executable, str(ROOT / "tools/gen_ir_artifact.py"),
         str(tmp_path)],
        check=True, cwd=ROOT)
    exe = tmp_path / "ir_eval_smoke"
    subprocess.run(
        [CXX, "-std=c++17", "-O2", f"-I{LANG / 'runtime/include'}",
         str(LANG / "runtime/tools/ir_eval_smoke.cpp"), "-o", str(exe)],
        check=True)
    r = subprocess.run(
        [str(exe), str(tmp_path / "ir_rover_artifact.txt"),
         str(tmp_path / "ir_rover_expect.txt"),
         str(tmp_path / "ir_rover_serve.txt")],
        capture_output=True, text=True)
    assert r.returncode == 0 and "OK (0 failures)" in r.stdout, \
        r.stdout + r.stderr
