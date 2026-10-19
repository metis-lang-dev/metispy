"""COUNTEREXAMPLES — the class-F boundary, pinned as tests.

Domain cases the factored emission (and hence the triangle corpus)
CANNOT carry, kept deliberately: each pins the EXACT rejection, so
the boundary is executable documentation, not folklore. The doctrine
under test is paper §8.2: conservative statics plus the runtime
Z = 1 check convert every out-of-class program into an ERROR —
never a silent approximation. If one of these ever starts passing,
the class grew: promote it to the corpus and update
docs/rewrite/04-domain-coverage.md."""

from pathlib import Path

from pytest import raises

from metis.kernel.factorize import FactoredProgram, FactorizeError
from metis.lang import load_llp

ROOT = Path(__file__).resolve().parents[1]
from langpath import LANG  # noqa: E402


def test_kitchen_multiplicity_rejected_at_admission():
    """The 3-onion soup is a multiplicity-3 premise: substitution
    collapse prices as falling factorials (paper Thm 1) — exact
    filter / sampler territory BY DESIGN, structurally outside the
    0/1 emitted class. Rejected before any table exists."""
    doc = load_llp(ROOT / "metis/domains/kitchen/catalogs/kitchen.llp")
    with raises(FactorizeError, match="multiplicity"):
        FactoredProgram(doc.catalog.ground({}), 1)


def test_road_mini_stalled_tier_rejected_at_evaluation():
    """The mini road cycle freezes the world on crash by CONSUMING
    the run token — from then on the clause tier of the active stage
    is empty on that branch (quiescence mid-schedule). Conservative
    statics ADMIT the program; the Z = 1 total-mass check catches
    the reachable stall at evaluation. This is Theorem 4's class
    boundary on a production-shaped catalog: the schedule is
    branch-DEPENDENT, so the measure does not descend to sites."""
    from metis.domains.road.mini import base_catalog, scenario
    prog = base_catalog().ground(scenario())
    fp = FactoredProgram(prog, 6)          # statics admit
    with raises(FactorizeError, match="mass"):
        fp.marginal("crash")               # runtime detects


def test_love_triangle_multiplicity_rejected_at_admission():
    """The ORIGIN lives at the boundary: the Ceptre tutorial's
    three-anger heartbreak is a multiplicity-3 premise — the native
    idiom of our lineage is falling-factorial territory (exact
    filter/sampler, Thm 1), structurally outside the emitted class.
    The lineage and the class boundary are the same fact."""
    from lovetriangle import INIT, love_triangle
    with raises(FactorizeError, match="multiplicity"):
        FactoredProgram(love_triangle().catalog.ground(INIT), 1)


def test_nav_depth_two_stalls_on_the_dead_branch():
    """Navigation at horizon >= 2: the absorbing `dead` branch has no
    enabled clause (no self-loop — adding one would change the
    pinned stage-12 measure), so the second step's tier stalls on
    exactly the dead mass. Horizon 1 is in the corpus; depth is the
    honest boundary."""
    from corpus import REGISTRIES, build_case, parse_manifest
    case = [c for c in parse_manifest(LANG / "corpus/"
                                      "manifest.txt")
            if c.name == "nav4-1step"][0]
    assert case.registry in REGISTRIES
    case.steps = 2
    prog, hooks = build_case(case)
    fp = FactoredProgram(prog, 2, hooks)   # statics admit
    with raises(FactorizeError, match="mass"):
        fp.marginal("dead")                # runtime detects
