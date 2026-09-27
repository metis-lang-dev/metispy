"""gridcat gates, GOLDEN-PINNED (stage 42): the lldsl/planner oracle
retired with the simulator divergence — the recode-identity proof it
served is history (it held from stage 33 until retirement; the
constants below were captured from the then-equal production path).
The catalog is now the ONLY road encoding; these pins are what keep
future catalog edits honest in-repo:
- program_key per scene class (vehicles, others-start, directed
  walkers, holonomic ped discs, occlusion ghosts) — catalog drift
  changes a key;
- unpoliced P(crash) for the closing 2-car scene;
- the RSS-shield property: ZERO crash mass under the ports policy
  across all scene classes;
- the binding plan()'s decision, dof and scores per objective."""

from pytest import approx

from metis.kernel import ForwardProjection, program_key
from metis.domains.road import gridcat
from metis.domains.road.gridcat import build, gen_instance, \
    road_bindings, slices_for
from metis.domains.road.sim import Car, Road, RoadState
from metis.lang import weight_policy

EXACT = dict(abs=1e-12)


def scene(cars, lanes=2, length=24, v_max=3):
    return RoadState(Road(lanes, length, v_max), tuple(cars))


CASES = [
    ("2-car-ego", dict(
        state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 12, 1)]),
        ticks=2)),
    ("2-car-others", dict(
        state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 12, 1)]),
        ticks=2, start_stage="others")),
    ("oncoming-walker", dict(
        state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 8, 1),
                     Car("walker", 1, 14, 1)]),
        ticks=2, walkers={"walker": -1})),
    ("ped-disc", dict(
        state=scene([Car("ego", 0, 0, 1), Car("dog", 1, 6, 0)],
                    lanes=2, length=16, v_max=2),
        ticks=1, peds={"dog": 2})),
    ("ghost", dict(
        state=scene([Car("ego", 0, 0, 2)], lanes=2, length=20,
                    v_max=3),
        ticks=1, frontiers=[(1, 8)])),
]

KEYS = {
    "2-car-ego": "3318877ce2a72cc57fc1335216f31a90"
                 "c12f947a2c50dc8476e4240824cbde64",
    "2-car-others": "41e2b7a177d56ebc994b38e2520791a8"
                    "f541da7bb67156e81aa9b9d189558a32",
    "oncoming-walker": "cf22133a568ec9d9d10ab76d4e0d9d24"
                       "4a19c054c11410c4aa2542ba753476f1",
    "ped-disc": "60cb94cd91d4e118961fa1dd89042d51"
                "2ffff8359ddb87b4aac0a5e46ca11049",
    "ghost": "2762e22fe402f121adaafb9e847aa799"
             "780eaf02c4fbe2a01d9f59c89f0e5794",
}


def test_catalog_program_keys_pinned():
    for label, kw in CASES:
        _, prog = build(**kw)
        assert program_key(prog) == KEYS[label], label


def _p_crash(pr):
    return sum(q for (_, k), q in pr.final.items()
               if ("crash", 1) in k)


def test_unpoliced_p_crash_pinned():
    st = scene([Car("ego", 0, 0, 2), Car("truck", 0, 5, 1)])
    steps = slices_for(2, 1)
    _, prog = build(st, 2)
    pr = ForwardProjection(prog, steps)
    assert _p_crash(pr) == approx(0.140625, **EXACT)


def test_rss_shield_zero_crash_mass_under_ports():
    """The shield property, pinned by name: under the ports policy
    (RSS shield + behavior rows + danger response + manifest/ghost
    priors) the crash mass is exactly zero across the scene
    classes, and the measure stays normalized."""
    cases = [
        dict(state=scene([Car("ego", 0, 0, 2),
                          Car("truck", 0, 5, 1)]), ticks=2),
        dict(state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 7, 2),
                          Car("walker", 1, 12, 1)]),
             ticks=2, walkers={"walker": 1}),
        dict(state=scene([Car("ego", 0, 0, 1), Car("dog", 1, 5, 0)],
                         lanes=2, length=16, v_max=2),
             ticks=1, peds={"dog": 2}),
        dict(state=scene([Car("ego", 0, 0, 2)], lanes=2, length=20,
                         v_max=3), ticks=1, frontiers=[(1, 8)]),
    ]
    for kw in cases:
        st, ticks = kw["state"], kw["ticks"]
        walkers = kw.get("walkers", {})
        peds = kw.get("peds", {})
        frontiers = kw.get("frontiers")
        steps = slices_for(
            ticks, len(st.cars) - 1 + (1 if frontiers else 0))
        doc, prog = build(st, ticks, walkers=walkers, peds=peds,
                          frontiers=frontiers)
        reg = road_bindings(st, ticks, walkers=walkers, peds=peds,
                            frontiers=frontiers)
        pol = weight_policy(prog, doc.ports, reg)
        pr = ForwardProjection(prog, steps, policy=pol)
        assert _p_crash(pr) == 0.0, kw
        assert sum(pr.final.values()) == approx(1.0, **EXACT), kw


def test_merge_lane_relations_are_the_handle_vocabulary():
    from metis.domains.road.semantic import Lane, SemanticCompiler
    text = gen_instance(CASES[0][1]["state"], 2)
    assert "leftof(0,1)." in text and "rightof(1,0)." in text
    lanes = [Lane("0", 500.0, 27.8, right="1"),
             Lane("1", 500.0, 27.8, left="0")]
    hfacts = set(SemanticCompiler(lanes).facts())
    assert "leftof(0,1)." in hfacts and "rightof(1,0)." in hfacts
    dom = gridcat.GRID_DOMAIN.read_text()
    stage_part = dom[dom.index("stage ego"):]
    for tok in (" 12", "27.8", "0.55", "0.3", "0.7"):
        assert tok not in stage_part


PLANS = [
    (dict(state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 6, 1)]),
          ticks=2, objective="dof"),
     ("brake", False, 1, {"brake": (0.0, -9.336791667)})),
    (dict(state=scene([Car("ego", 0, 0, 2), Car("truck", 0, 9, 2)]),
          ticks=2, objective="dof+progress"),
     ("right", False, 3, {"brake": (0.0, -13.947961806),
                          "keep": (0.0, -15.202090278),
                          "right": (0.0, -16.594787326)})),
    (dict(state=scene([Car("ego", 0, 0, 2), Car("van", 0, 7, 1),
                       Car("walker", 1, 13, 1)]),
          ticks=2, objective="dof", walkers={"walker": 1}),
     ("brake", False, 2, {"brake": (7.576e-06, -9.584813352),
                          "right": (0.001306309, -8.433706504)})),
    (dict(state=scene([Car("ego", 0, 0, 1), Car("dog", 1, 5, 0)],
                      lanes=2, length=16, v_max=2),
          ticks=1, objective="dof", peds={"dog": 2}),
     ("brake", False, 4, {"accel": (0.054526749, -4.860596708),
                          "brake": (0.001028807, -5.748971193),
                          "keep": (0.017489712, -5.49537037),
                          "right": (0.009259259, -5.013888889)})),
    (dict(state=scene([Car("ego", 0, 0, 2), Car("wall", 1, 8, 0)],
                      lanes=2, length=20, v_max=3),
          ticks=1, objective="dof", occlusion="aware"),
     ("brake", False, 3, {"accel": (0.0, -6.0),
                          "brake": (0.0, -7.257),
                          "keep": (0.0, -6.333333333)})),
]


def test_binding_plan_decisions_pinned():
    """gridcat.plan across objectives, walkers, peds and occlusion:
    action, doomed, dof and every score to one round9 quantum."""
    for kw, (action, doomed, dof, scores) in PLANS:
        got = gridcat.plan(**kw)
        assert got["action"] == action, kw
        assert got["doomed"] == doomed and got["dof"] == dof, kw
        assert set(got["scores"]) == set(scores), kw
        for a, sc in scores.items():
            got_sc = tuple(got["scores"][a])
            assert got_sc == approx(sc, abs=1.5e-9), (kw, a)
