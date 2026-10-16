"""specialist S1b (metis-agent docs/SPECIALIST.md §5b): `state` prints
every atom in the form why / necessity / sufficiency take, pred(args), and
a refusal of those three names the nearest held atom."""
from pathlib import Path

import pytest

from metis.lang.directives import atom_rows, nearest_held
from metis.lang.repl import Repl

CAT = Path(__file__).resolve().parents[3] / "metis-catalog"
BLOCKS = CAT / "domains/planning/blocks_world.llp"
pytestmark = pytest.mark.skipif(not BLOCKS.is_file(),
                                reason="metis-catalog not checked out")


def test_state_prints_the_instruments_form():
    r = Repl(str(BLOCKS))
    out = r.capture("state", "")
    assert out.splitlines()[1:] == ["  arm_free", "  clear(b) clear(c)",
                                    "  on(c,a)", "  on_table(a) on_table(b)"]
    for a in ("clear(b)", "on(c,a)", "on_table(a)"):          # each one why takes
        assert "never produced" not in r.capture("why", a)


def test_the_rows_count_and_wrap():
    assert atom_rows([("p(a)", 2), ("q", 1), ("r(x)", 0)]) == ["  p(a):2", "  q"]
    rows = atom_rows([(f"long_name(arg{i})", 1) for i in range(9)])
    assert len(rows) > 1 and all(len(x) <= 78 for x in rows)


def test_why_refusal_names_the_nearest_held_atom():
    r = Repl(str(BLOCKS))
    assert "the nearest held atom: on(c,a)" in r.capture("why", "on (c,b)")
    assert "the nearest held atom: clear(b)" in r.capture("why", "clear(zz)")
    assert "no held atom has the predicate holding" in r.capture("why", "holding(a)")


def test_cause_refusal_names_the_nearest_held_atom():
    r = Repl(str(BLOCKS))
    r.capture("run", "3")
    out = r.capture("necessity", "1 clear(q)")
    assert out.startswith("refused: clear(q) is no atom of this catalog — the nearest held atom: clear(")
    assert "refused:" in r.capture("sufficiency", "1 on (z,q)")


def test_nearest_is_the_same_predicate_fewest_differences():
    held = [("on(a,b)", 1), ("on(c,d)", 1), ("clear(a)", 1)]
    assert nearest_held("on(c,b)", held) == "on(a,b)"        # tie: sort order
    assert nearest_held("on(c,d)", held) == "on(c,d)"
    assert nearest_held("on (c, x)", held) == "on(c,d)"
    assert nearest_held("holding(a)", held) == ""
