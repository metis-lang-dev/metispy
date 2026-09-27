"""Leg-1 gates: the surprise record. An outcome the BASE catalog
cannot produce logs as unmodeled (zero likelihood — the structural
signal); the record replayed against the base + a shadow-admitted
pack recovers a full explanation (likelihood, posterior, the MAP
derivation naming the pack's rule) while plan() stays on the base —
the shadow-pack usage pattern. Determinism: the record carries
pointers, the offline side re-derives everything."""

import json

from pytest import approx

from metis.kernel import program_key
from metis.kernel.record import recover, surprise_record
from metis.domains.road.mini import base_catalog, scenario, \
    wheelchair_pack

EXACT = dict(abs=1e-12)


def test_unmodeled_logged_then_explained_by_shadow_pack():
    base = base_catalog()
    base_prog = base.ground(scenario())
    # the wheelchair world produced a caution atom the base cannot
    rec = surprise_record(base_prog, 8, {"caution(c2)": 1},
                          meta={"pack": "base@1"})
    assert rec["kind"] == "unmodeled"
    assert rec["likelihood"] == 0.0
    json.dumps(rec)                        # the record ships
    assert rec["program_key"] == program_key(base_prog)

    # offline: shadow-admit the candidate pack, replay the record
    merged = base.admit(wheelchair_pack())
    shadow_prog = merged.ground(scenario(wc=True))
    out = recover(rec, shadow_prog)
    assert out["explains"] and out["likelihood"] == approx(
        16 / 27, **EXACT)
    assert any("caution" in n or "wc" in n or "slow" in n
               for n in out["explanation_events"]), \
        out["explanation_events"]
    # the base still cannot explain it — recover is honest
    out_base = recover(rec, base_prog)
    assert not out_base["explains"]


def test_rare_and_expected_verdicts():
    base_prog = base_catalog().ground(scenario())
    rec = surprise_record(base_prog, 6, {"crash": 1})
    assert rec["kind"] is None             # the crash is likely here
    assert rec["likelihood"] == approx(11 / 18, **EXACT)
    rec = surprise_record(base_prog, 6, {"crash": 0},
                          rare_ratio=0.45)
    assert rec["kind"] == "rare"           # survival below the bar
    assert rec["likelihood"] == approx(7 / 18, **EXACT)
