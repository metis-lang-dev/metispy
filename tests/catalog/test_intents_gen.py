"""metis.intents (was metis-catalog tools/intents.py; metis-packages s-65778b98): the draft's outcomes and asks (e2e-workflow step 4,
the architect's rulings), and the asks check (CATALOG.md `asks`)."""
import sys
from pathlib import Path

import yaml

from metis import intents as I  # noqa: E402

PROG = """catalog plant 1.
pred moist : plant.
pred dry : plant.
pred sun : plant.
pred cmd_water : plant.
pred cmd_check : plant.
pred done(water) : plant.
pred checked(soil) : plant.
stage main {
  %% My soil dries out.
  dry_out [dyn] : moist * sun -o dry.
  %% You water me.
  water [dyn] : dry * cmd_water -o moist * done(water).
  %% I am moist.
  check_moist [dyn] : cmd_check * $moist -o checked(moist).
  %% I am dry.
  check_dry [dyn] : cmd_check * $dry -o checked(dry).
}
"""


def test_outcomes_are_what_a_command_produces_and_no_rule_reads():
    rules = I.program_rules(PROG)
    assert [r[1] for r in rules] == ["dry_out", "water", "check_moist", "check_dry"]
    assert I.outcome_atoms(rules) == ["done", "checked"]          # moist is state: dry_out reads it


def test_an_ask_must_only_read_the_state():
    spec = {"verbs": [{"verb": "WATER"}, {"verb": "CHECK"}],
            "map": [{"verb": "WATER", "atom": "cmd_water"}, {"verb": "CHECK", "atom": "cmd_check"}]}
    assert I.asks_problems(dict(spec, asks=["CHECK"]), PROG) == []
    bad = I.asks_problems(dict(spec, asks=["WATER", "NOPE"]), PROG)
    assert bad[0].startswith("asks WATER: rule water consumes dry") and bad[1] == "asks NOPE: no such verb"


def test_the_catalogs_asks_hold_and_the_drafts_infer_them(monkeypatch):
    from catalogs import installed_catalog
    cat = installed_catalog()
    if cat is None:
        import pytest
        pytest.skip("no metis-catalog installed (its domains)")
    monkeypatch.setattr(I, "ROOT", cat)                 # metis.intents reads the root it is given
    for name in ("dog", "plant"):
        d = I.ROOT / "domains" / name
        if not (d / "intent.yaml").is_file():
            continue
        assert I.asks_problems(I.load(name), (d / ("%s.llp" % name)).read_text()) == []
    plant = I.ROOT / "domains/plant/plant.llp"
    if plant.is_file():
        sk = yaml.safe_load(I.skeleton("plant"))
        assert sk["asks"] == ["CHECK"] and sk["outcomes"] == {"done": "done", "checked": "done"}


def test_an_empty_generated_file_is_not_written(tmp_path, monkeypatch):
    """A spec without `spoken` makes an empty -spoken.txt: gen does not write it and
    check does not ask for it; one that is there is still checked (loop-fixes s-855d8067)."""
    d = tmp_path / "domains" / "plant"
    d.mkdir(parents=True)
    (d / "plant.llp").write_text(PROG)
    monkeypatch.setattr(I, "ROOT", tmp_path)
    assert "spoken" not in yaml.safe_load(I.skeleton("plant"))
    (d / "intent.yaml").write_text(I.skeleton("plant"))
    assert I.files("plant")["plant-spoken.txt"] == ""
    assert I.main(["gen", "plant"]) == 0
    assert sorted(p.name for p in (d / "intent").iterdir()) == ["plant-prompt.txt", "plant.gbnf"]
    assert I.main(["check", "plant"]) == 0
    (d / "intent" / "plant-spoken.txt").write_text("stale\n")
    assert I.main(["check", "plant"]) == 1
