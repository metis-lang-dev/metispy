"""Spec 11 (branch spec11-cone): the influence cone — the P1 sweep
closes (no out-of-cone event writes a cone atom; cone events read only
cone atoms), and the quotient filter agrees EXACTLY with an independent
full-program enumerator under the same readout (§5 gate 1)."""
import io
from dataclasses import replace
from pathlib import Path

import pytest

from metis.kernel.filter import StateBudgetExceeded
from metis.lang.cone import (check_closed, cone_readout, full_readout,
                             qkey, quotient, sweep)
from metis.lang.repl import Repl

FICTION = Path(__file__).resolve().parents[3] / \
    "metis-catalog/domains/fiction"
pytestmark = pytest.mark.skipif(not FICTION.exists(),
                                reason="metis-catalog not checked out")


def shell(name, seed, k):
    r = Repl(FICTION / name, seed=seed, stdout=io.StringIO())
    r.onecmd(f"run {k}")
    return r


def agree(r, outcomes, readouts, budget=200_000):
    full = replace(r.prog, init_stage=r.stage, init=dict(r.counts))
    n = 0
    for y in outcomes:
        c = sweep(r.prog, [y], r.stage)
        assert check_closed(r.prog, c, r.stage) == []
        q = quotient(r.prog, c, r.stage, r.counts)
        for ro in readouts:
            try:
                pc, _ = cone_readout(q, y, ro, budget, frame_cap=64)
                pf = full_readout(full, c.events, y, ro, budget,
                                  max_steps=200)
            except (StateBudgetExceeded, RuntimeError):
                continue
            assert pc == pytest.approx(pf, abs=1e-12), (y, ro)
            n += 1
    return n


@pytest.mark.parametrize("seed,k", [(1, 0), (1, 3), (3, 2), (7, 1)])
def test_3lp_cone_equals_full_exactly(seed, k):
    r = shell("3lp.llp", seed, k)
    atoms = sorted(r.atoms)
    n = agree(r, atoms + ["!" + a for a in atoms],
              [("cone", 1), ("cone", 3), ("frame",)])
    assert n >= 3 * len(atoms)


def test_love_triangle_cone_equals_full_on_answerable_cases():
    r = shell("love_triangle.llp", 1, 2)
    assert agree(r, ["anger(lysander,helena)", "!eros(helena,hermia)"],
                 [("cone", 1), ("cone", 3)]) == 4


def test_the_sweep_closes_and_the_quotient_has_its_own_key():
    r = shell("3lp.llp", 1, 0)
    c = sweep(r.prog, ["brickhouse"], r.stage)
    names = {r.prog.events()[i][2].name for i in c.events}
    # brickhouse is written by build_brick only (blow_brick only READS
    # it); build_brick consumes a pig, so every builder competing for
    # pigs joins; the blows write no cone atom and stay out
    assert names == {"build_brick", "build_stick", "build_straw"}
    assert c.atoms == {"brickhouse", "pig", "bricks", "sticks", "straw"}
    q = quotient(r.prog, c, r.stage, r.counts)
    assert qkey(q) != r.key and len(qkey(q)) == 64
    assert r.key == Repl(FICTION / "3lp.llp",
                         stdout=io.StringIO()).key     # keys never move
    # a hand-broken cone is caught (the construction-bug detector)
    broken = c._replace(events=frozenset())
    assert any("writes" in b for b in check_closed(r.prog, broken,
                                                   r.stage))
    # masks prune first: blocking the only writer leaves an empty cone
    c2 = sweep(r.prog, ["brickhouse"], r.stage,
               masked=lambda n: n == "build_brick")
    assert not c2.events and c2.atoms == {"brickhouse"}
    assert check_closed(r.prog, c2, r.stage,
                        masked=lambda n: n == "build_brick") == []


def test_tragedy_prospective_cone_is_the_whole_world():
    """The §5 finding, pinned: on tragedy every outcome's ground cone is
    (nearly) the whole program — `at(C,L)` is persisted by nearly every
    rule and written by travelTo; despair, murder, anger and love feed
    each other — so P1/P2 alone cannot reduce the filter there."""
    r = shell("tragedy.llp", 132, 74)
    c = sweep(r.prog, ["suicidal(mercutio)"], r.stage)
    assert check_closed(r.prog, c, r.stage) == []
    assert len(c.events) == c.n_events and len(c.atoms) >= c.n_atoms - 1
