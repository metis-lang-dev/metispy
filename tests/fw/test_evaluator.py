"""Stage 10 gates: unbound-port fail-fast; sysadmin through the SAME
Evaluator class via the text catalog + weight ports (matching the
domain's own greedy planner); learning overlays over the base
distribution; certificate-gated hot swap that never drops the serving
catalog; and (slow) the SUMO closed loop through the framework path
with tick-for-tick equality."""

import shutil
from pathlib import Path

import pytest
from pytest import approx

from metis.fw import Blackboard, Evaluator, PortError, Registry, \
    catalog_domain
from metis.kernel import CatalogError, ForwardProjection, program_key
from metis.lang import load_llp, parse
from metis.domains.road.mini import base_catalog, scenario, \
    wheelchair_pack
from metis.domains.sysadmin.model import greedy_plan

ROAD = Path("metis/domains/road/catalogs")
SYS = Path("metis/domains/sysadmin/catalogs")
EXACT = dict(abs=1e-12)
ADJ = {"a": ["b"], "b": ["a"]}


# ---- sysadmin registry + domain over the TEXT catalog ------------------

def sys_registry():
    reg = Registry()
    reg.input("state", lambda running: set(running))

    def cpt(args, counts):
        c = args[0]
        if f"rb({c})" in counts:
            return 1.0
        ins = ADJ[c]
        r = sum(1 for y in ins if f"prev({y})" in counts)
        return 0.45 + 0.5 * (1 + r) / (1 + len(ins))

    reg.weight("cpt", cpt)
    reg.weight("rp", lambda args, counts:
               1.0 if f"rb({args[0]})" in counts else 0.04)
    reg.output("action", lambda res, sit: res.action)
    return reg


def expected_running(pr):
    return sum(p * sum(1 for a, _ in k if a.startswith("run("))
               for (_, k), p in pr.final.items())


def sys_evaluator():
    ev = Evaluator(load_llp(SYS / "sysadmin2.llp"), sys_registry())

    def init_fn(sit, a):
        running = sit["state"]
        init = {"m(i0)": 1}
        for c in ("a", "b"):
            init[f"run({c})" if c in running else f"down({c})"] = 1
        if a is not None:
            init[f"rb({a})"] = 1
        return init

    def score_fn(pr, sit):
        rebooted = any(x.startswith("rb(") for x in pr.program.init)
        return ((0.75 if rebooted else 0.0) - expected_running(pr),)

    ev.domain = catalog_domain(
        ev,
        candidates_fn=lambda sit: [None] + sorted(
            c for c in ("a", "b") if c not in sit["state"]),
        init_fn=init_fn, steps_fn=lambda sit: 9, score_fn=score_fn)
    return ev


# ---- fail-fast ---------------------------------------------------------

def test_unbound_ports_fail_fast():
    with pytest.raises(PortError) as e:
        Evaluator(load_llp(SYS / "sysadmin2.llp"), Registry())
    msg = str(e.value)
    for port in ("weight:cpt", "weight:rp", "input:state",
                 "output:action"):
        assert port in msg


# ---- same Evaluator class, second domain -------------------------------

def test_sysadmin_through_the_evaluator_matches_greedy():
    ev = sys_evaluator()
    bb = Blackboard()
    bb["state"] = {"a"}
    res = ev.tick(bb)
    ref = greedy_plan(["a", "b"], ADJ, {"a"})
    assert res.action == ref.action == "b"
    assert bb["action"] == "b"
    assert res.scores[None] == approx(ref.scores[None], **EXACT)
    assert res.scores["b"] == approx(ref.scores["b"], **EXACT)


# ---- learning overlays -------------------------------------------------

def test_overlay_over_base_distribution():
    ev = sys_evaluator()
    sit = {"state": {"a"}}
    _, do, project, _, _ = ev.domain(sit)

    def marginal_b(pr):
        return sum(p for (_, k), p in pr.final.items()
                   if ("run(b)", 1) in k)

    assert marginal_b(project(do(None))) == approx(0.04, **EXACT)
    # experience overlay: b revives at 0.5; a falls through (None)
    ev.registry.overlay("rp", lambda args, counts:
                        0.5 if args[0] == "b" else None)
    assert marginal_b(project(do(None))) == approx(0.5, **EXACT)
    ev.registry.pop_overlay("rp")
    assert marginal_b(project(do(None))) == approx(0.04, **EXACT)


# ---- hot swap ----------------------------------------------------------

BAD_PACK = """
catalog badp 1.
extends road.
namespace pk produce (pack).
pred tokp : pk.
stage check {
  %% crafted violation
  boost [pack] : tokp -o st(ego,l0,c4,v2).
}
"""

PORTLESS_PACK = """
catalog needsport 1.
extends road.
weight newf(car).
"""


def test_hot_swap_certificate_gated():
    reg = Registry()
    reg.input("corridor", lambda v: v)
    reg.output("setpoint", lambda res, sit: res.action)
    ev = Evaluator(load_llp(ROAD / "mini.llp"), reg)

    # good pack applies; the merged program matches the python path
    ev.load_pack(str(ROAD / "wheelchair.llp"))
    key = program_key(ev.doc.catalog.ground(scenario(wc=True)))
    py_key = program_key(base_catalog().admit(wheelchair_pack())
                         .ground(scenario(wc=True)))
    assert key == py_key
    pr = ForwardProjection(ev.doc.catalog.ground(scenario(wc=True)), 8)
    assert pr.final_distribution("caution(c2)") == approx(
        [11 / 27, 16 / 27], **EXACT)

    # ill-typed pack refused, previous catalog still serving
    with pytest.raises(CatalogError, match="may not produce 'st'"):
        ev.load_pack(parse(BAD_PACK))
    assert program_key(ev.doc.catalog.ground(scenario(wc=True))) == key

    # pack declaring an unbound port refused at the PORT gate
    with pytest.raises(PortError, match="weight:newf"):
        ev.load_pack(parse(PORTLESS_PACK))
    assert program_key(ev.doc.catalog.ground(scenario(wc=True))) == key


# ---- the SUMO closed loop through the framework path -------------------

@pytest.mark.slow
@pytest.mark.skipif(shutil.which("sumo") is None
                    or shutil.which("netconvert") is None,
                    reason="sumo/netconvert not installed")
def test_sumo_closed_loop_through_framework(tmp_path):
    from metis.domains.road.fw import road_evaluator
    from metis.domains.road.profile import CAR
    from metis.domains.road import sumo_harness as sh
    ev = road_evaluator(ticks=2, objective="dof", behaviors=None,
                        ego_vcap=CAR.v_max,
                        dmin_pad=CAR.dmin_pad_cells)
    stats = sh.closed_loop(str(tmp_path), seed=11, steps=240,
                           evaluator=ev)
    # per-tick equality is asserted inside closed_loop; here: the
    # framework path really ran, was non-vacuous, and stayed safe
    assert stats["fw_ticks"] == stats["ticks"] > 100
    assert stats["actions"].get("brake", 0) > 10
    assert stats["ego_collisions"] == 0
