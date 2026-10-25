"""metis.kernel.evaluate, the one-tick instrument (metis-lang spec/16 §6, §7; spec16 s-b12d90f2 step 3): metispy's
parity with metisc --evaluate.

The gate: every fixture of metis-lang's corpus/evaluate (the embedded .llp form and every case YAML) gives the
golden's listing and exit code byte for byte, and every exact V line equals the brute-force referee's, as rationals —
metis-lang's ci gate 14, run on metispy's instrument. The corpus is the sibling metis-lang checkout's
($METIS_LANG_DIR overrides); without one those cases skip, and the self-contained cases below still run."""
import io
import os
import re
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path

import pytest

from metis.kernel import evaluate as E

ROOT = Path(__file__).resolve().parents[1]
LANG = Path(os.environ["METIS_LANG_DIR"]) if os.environ.get("METIS_LANG_DIR") else ROOT.parent / "metis-lang"
CORPUS = LANG / "corpus" / "evaluate"
GOLDENS = LANG / "corpus" / "goldens" / "evaluate"


def fixtures():
    if not CORPUS.is_dir():
        return []
    out = [(f.name, f.stem) for f in sorted(CORPUS.glob("*.llp")) if re.search(r"(?m)^#evaluate", f.read_text())]
    out += [(str(p.relative_to(CORPUS)), p.stem) for p in sorted(CORPUS.rglob("cases/*.yaml"),
                                                                   key=lambda p: str(p.relative_to(CORPUS)))]
    return out


def listing(path, *flags):
    """metisc --evaluate's listing as the gate takes it: stdout and stderr, then `rc N`."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = E.main([path, *flags])
    return out.getvalue() + err.getvalue() + "rc %d\n" % rc


@pytest.mark.skipif(not CORPUS.is_dir(), reason="no metis-lang checkout beside metispy (corpus/evaluate)")
@pytest.mark.parametrize("path, golden", fixtures(), ids=[g for _, g in fixtures()])
def test_the_listing_is_metiscs_and_brute_agrees(path, golden, monkeypatch):
    monkeypatch.chdir(CORPUS)
    monkeypatch.setenv("METIS_HOME", str(LANG / "corpus" / "iface" / "_home"))
    got = listing(path)
    assert got == (GOLDENS / ("%s.txt" % golden)).read_text()
    if got.endswith("rc 0\n"):
        exact = [l.split(" worlds ")[0] for l in got.splitlines() if l.startswith("V ") and " = " in l]
        brute = {l.split(" worlds ")[0] for l in listing(path, "--brute").splitlines() if l.startswith("V ")}
        assert exact and all(v in brute for v in exact)


def test_the_corpus_is_all_there():
    if not CORPUS.is_dir():
        pytest.skip("no metis-lang checkout beside metispy")
    assert {g for _, g in fixtures()} == {p.stem for p in GOLDENS.glob("*.txt")}     # every golden has its fixture


COIN = """catalog coin 1.
layers (dyn).
stages (play).
namespace c produce (dyn).
type side {heads tails}.
pred up(side) : c.
pred won : c.
pred lost : c.
pred tossed : c.
weight luck(S) reads {up(S)}.
stage play {
  %% Heads up: I win.
  win [dyn] : $up(heads) * tossed -o won.
  %% Tails up: I lose.
  lose [dyn] : $up(tails) * tossed -o lost.
}
#init tossed 1
#belief up one S by luck
#port luck(heads) 0.7
#port luck(tails) 0.3
#evaluate w reach won within 2
#evaluate l risk lost within 2
#expect w 7/10
#level odds from w at 0.5 names poor good
"""


def coin(tmp_path, text=COIN):
    p = tmp_path / "coin.llp"
    p.write_text(text)
    return listing(str(p))


def test_decimals_enter_as_their_shortest_round_trip(tmp_path):
    # 0.7 + 0.3 is exactly 1 (Fraction(repr(x))), so the one group passes §2's exact check; its binary64 sum would not
    assert E.number("x", 0.7) == Fraction(7, 10) and E.number("x", "0.3") + E.number("x", 0.7) == 1
    assert Fraction(0.7) + Fraction(0.3) != 1
    got = coin(tmp_path, COIN.replace("#level odds from w at 0.5 names poor good\n", ""))
    assert "V w reach won within 2 = 7/10 ~ 0.69999999999999996 worlds 2 states" in got
    assert "V l risk lost within 2 = 3/10 ~ 0.29999999999999999 worlds 2 states" in got
    assert got.startswith("N belief-read win up\nN belief-read lose up\n") and got.endswith("rc 0\n")


def test_a_group_that_does_not_sum_to_one_is_refused(tmp_path):
    got = coin(tmp_path, COIN.replace("luck(tails) 0.3", "luck(tails) 0.2"))
    assert got == ("refused belief-sum: belief up one S: the group up(_) sums to 9/10 (0.90000000000000002), not 1\n"
                   "rc 2\n")


def test_a_failed_expectation_exits_1_and_a_level_needs_its_type(tmp_path):
    got = coin(tmp_path, COIN.replace("#expect w 7/10", "#expect w 1/2").replace("#level odds from w at 0.5 names poor good\n", ""))
    assert "X w expected 1/2 got 0.69999999999999996" in got and got.endswith("rc 1\n")
    got = coin(tmp_path)                                       # odds: not a pred of the program
    assert got == "refused level-type: level odds: not a pred of the program\nrc 2\n"


def test_the_budget_cuts_to_an_interval_that_holds_the_value(tmp_path):
    text = COIN.replace("#evaluate w reach won within 2", "#evaluate w reach won within 2 budget 1") \
        .replace("#expect w 7/10", "#expect w 0.69999999999999996").replace("#level odds from w at 0.5 names poor good\n", "")
    got = coin(tmp_path, text)
    line = next(l for l in got.splitlines() if l.startswith("V w "))
    assert " in [" in line and " cut " in line and got.endswith("rc 0\n")


def test_the_instrument_directives_are_the_instruments_only(tmp_path):
    from metis.lang.directives import embedded_case
    case = embedded_case(COIN)
    assert case["init"] == {"tossed": 1} and case["instrument"][0] == ["belief", "up", "one", "S", "by", "luck"]


# -- the level port, live (spec16 step 4): the manifest's fields, and the host's hook -----------------------------

MAN = {"program": "coin.llp", "beliefs": {"up": {"port": "luck", "one": "S"}},
       "evaluate": {"w": {"reach": "won", "within": 2}},
       "levels": {"odds": {"from": {"evaluate": "w"}, "at": [0.5], "names": ["poor", "good"]}}}


def package(tmp_path):
    d = tmp_path / "domains" / "coin"
    d.mkdir(parents=True)
    text = COIN.replace("#level odds from w at 0.5 names poor good\n", "").replace(
        "pred tossed : c.", "pred tossed : c.\ntype odds_l {poor good}.\npred odds(odds_l) : c.")
    (d / "coin.llp").write_text("\n".join(l for l in text.split("\n") if not l.startswith("#")))
    import yaml
    (d / "domain.yaml").write_text(yaml.safe_dump(MAN, sort_keys=False))
    return d


def test_a_manifest_carries_beliefs_levels_and_evaluate(tmp_path):
    from metis.catalog import manifests as mc
    d = package(tmp_path)
    assert mc.get("coin", tmp_path).manifest["levels"]["odds"]["names"] == ["poor", "good"]
    import yaml
    for bad, msg in ((dict(MAN, levels={"odds": {"from": {"evaluate": "w"}, "at": [0.5, 0.2], "names": ["a", "b", "c"]}}), "levels is"),
                     (dict(MAN, levels={"odds": {"from": {"evaluate": "nope"}, "at": [0.5], "names": ["poor", "good"]}}), "no such query"),
                     (dict(MAN, evaluate={"w": {"reach": "won"}}), "evaluate is"),
                     (dict(MAN, beliefs={"up": {"port": "luck", "one": "S", "lone": "S"}}), "beliefs is")):
        (d / "domain.yaml").write_text(yaml.safe_dump(bad, sort_keys=False))
        with pytest.raises(mc.ManifestError, match=msg):
            mc.domains(tmp_path)


def test_the_hook_asserts_the_level_of_the_state_as_it_stands(tmp_path):
    d = package(tmp_path)
    hook = E.LevelHook(MAN, d)
    ports = {"luck": {("heads",): "0.7", ("tails",): "0.3"}}
    atoms, rep = hook.levels({"tossed": 1}, ports)
    assert atoms == {"odds(good)": 1} and rep["values"] == {"w": "7/10"} and rep["ms"] >= 0
    atoms, rep = hook.levels({"tossed": 1}, {"luck": {("heads",): "0.2", ("tails",): "0.8"}})
    assert atoms == {"odds(poor)": 1}
    atoms, rep = hook.levels({}, ports)                          # nothing tossed: the fetch never comes
    assert atoms == {"odds(poor)": 1} and rep["values"] == {"w": "0/1"}
    atoms, rep = hook.levels({"tossed": 1}, {"luck": {("heads",): "0.7", ("tails",): "0.7"}})
    assert atoms == {} and rep["refused"].startswith("belief-sum:")
