"""The opt-in ground budget (policy-tuning-2 s-e075f47f, the sw-s2 OOM): METIS_GROUND_BUDGET refuses a schema that
grounds past it, with words that name the rule and its variables' types; unset, grounding is as before."""
import pytest

from metis.kernel.ground import GroundBudget
from metis.lang.compiler import load_llp

SRC = """catalog t 1.
layers (dyn).
stages (act).
type term {%s}.
type room {r1 r2}.
namespace t produce (dyn).
pred here(room) : t.
pred connected(term, term) : t.
pred holds : t.
stage act {
  %%%% take, reading a pair of terms nothing else binds
  take [dyn] : $here(R) * connected(T1, T2) -o holds.
}
#init here(r1) 1
""" % " ".join("t%d" % i for i in range(40))


def _ground(tmp_path):
    f = tmp_path / "t.llp"
    f.write_text(SRC)
    return load_llp(str(f)).catalog.ground({"here(r1)": 1})


def test_unset_grounds_as_before(tmp_path, monkeypatch):
    monkeypatch.delenv("METIS_GROUND_BUDGET", raising=False)
    prog = _ground(tmp_path)
    assert sum(len(c) for c in prog.stages.values()) == 2 * 40 * 40


def test_a_budget_refuses_with_the_rules_words(tmp_path, monkeypatch):
    monkeypatch.setenv("METIS_GROUND_BUDGET", "1000")
    with pytest.raises(GroundBudget, match=r"take grounds more than 1000 events .*T1: term 40, T2: term 40"):
        _ground(tmp_path)
    monkeypatch.setenv("METIS_GROUND_BUDGET", "5000")
    assert sum(len(c) for c in _ground(tmp_path).stages.values()) == 3200
