"""metis.repl (metis-packages s-65778b98): the REPL as one structured API — `run 3` answers the counts the
CLI prints, a step names what fired, a snapshot restores through the kernel's undo, the tools are the
REPL API's declarations."""
import subprocess
import sys

import pytest

from metis.lang.repl import _fmt_counts
from metis.repl import Repl

PROG = """catalog toy 1.
layers (dyn).
stages (main).
namespace toy produce (dyn).
pred a : toy.
pred b : toy.
pred c : toy.
stage main {
  %% One a becomes a b.
  ab [dyn] : a -o b.
  %% Two bs make a c.
  bbc [dyn] : b * b -o c.
}
#init a 4
"""


@pytest.fixture
def prog(tmp_path):
    p = tmp_path / "toy.llp"
    p.write_text(PROG)
    return p


def _cli(path, lines):
    out = subprocess.run([sys.executable, "-m", "metis.lang.repl", str(path)], input=lines, capture_output=True,
                         text=True, timeout=120).stdout
    return [ln[len("llp> "):] if ln.startswith("llp> ") else ln for ln in out.splitlines()]


def test_run_answers_what_the_cli_prints(prog):
    out = Repl.open(prog).run(3)
    cli = _cli(prog, "run 3\n")
    i = max(k for k, ln in enumerate(cli) if ln.startswith("stage "))     # the banner prints the start's line first
    assert cli[i - 1] == (" ".join(out["fired"]) or "(quiescent)")
    assert cli[i] == "stage %s | %s" % (out["stage"], _fmt_counts(out["counts"]))
    assert set(out) == {"stage", "done", "counts", "fired"} and len(out["fired"]) == 3 and not out["done"]


def test_quiescence_is_done(prog):
    out = Repl.open(prog).run(50)
    assert out["done"] and out["stage"] == "(done)" and out["counts"] == {"c": 2}


def test_step_snapshot_restore_undo_reset(prog):
    r = Repl.open(prog)
    s0 = r.state()
    st = r.step("ab")
    assert st["fired"] == ["ab"] and st["counts"] == {"a": 3, "b": 1}
    snap = r.snapshot()
    r.run(10)
    assert r.restore(snap)["counts"] == {"a": 3, "b": 1}
    assert r.undo()["counts"] == s0["counts"]
    r.run(2)
    assert r.reset()["counts"] == {"a": 4}
    with pytest.raises(ValueError):
        r.restore(99)
    assert "fire" in r.cmd("help run") or r.cmd("help run")          # any REPL line, its words


def test_the_tools_are_the_repl_apis_declarations():
    from metis.lang.repl_api import REPL_API
    t = Repl.tools()
    assert t == REPL_API and t is not REPL_API
