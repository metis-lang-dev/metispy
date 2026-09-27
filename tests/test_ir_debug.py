"""The debug sidecar: D lines name each site's candidates in the
canonical (wire) order, so an interpreter can decode 'which rule'
from the artifact without recompiling."""
import os
from pathlib import Path

from metis.kernel.factorize import FactoredProgram
from metis.kernel.ir import ir_debug_lines
from metis.lang import load_llp

ROOT = Path(__file__).resolve().parents[1]
LANG = Path(os.environ.get(
    "METIS_LANG", ROOT.parent / "metis-lang")).resolve()


def test_debug_lines_name_site_candidates():
    doc = load_llp(LANG / "corpus/packbase.llp")
    prog = doc.catalog.ground({"coin": 1})
    fp = FactoredProgram(prog, 2)
    lines = ir_debug_lines("packbase", fp)
    assert lines[0] == "C packbase"
    d = [l for l in lines if l.startswith("D ")]
    assert d, lines
    # the roll site's candidates are h and t, in canonical order
    names = d[0].split()[3:]
    assert names == sorted(names) == ["h", "t"]
    # candidate count field agrees
    assert int(d[0].split()[2]) == len(names)
