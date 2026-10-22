"""metis.gate's experimental policy (loop-policy s-dd842f2a, metis-agentic-loop docs/POLICY.md):
a proposal judged by its numbers per phase and confirmed on accept, refused and remembered
otherwise; the journal; the ask (a disengagement record) waiting for the person's answer; the
human gate unchanged when no policy is given. The gain hook is a fake: the numbers are the
test's."""
import json
import shutil
import subprocess
import sys

import pytest

from catalogs import installed_catalog
from metis import gate

CAT = installed_catalog()
EXT = "% an extension\n"


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "cat"
    shutil.copytree(CAT / "domains" / "plant", r / "domains" / "plant",
                    ignore=shutil.ignore_patterns("records", "build", "__pycache__"))
    return r


def fake(numbers):
    """A gain hook answering `numbers` and keeping what it was asked."""
    seen = []

    def gain(plan, phase):
        seen.append((plan, phase))
        return dict(numbers)
    gain.seen = seen
    return gain


ASK = {"phase": "exploit", "reason": "no_progress", "tried": ["revise open_fridge: refused twice"],
       "believes": ["open_fridge needs closed(fridge)"], "options": ["explore here", "skip this state"],
       "question": "The fridge stays shut after `open fridge 1`: is there a key or a lock I do not see?",
       "snapshot": "run/snap/7.json"}


@pytest.mark.parametrize("phase, numbers, verdict", [
    ("explore", {"fits_records": True, "likelihood": 0.0}, "accept"),
    ("explore", {"fits_records": False, "likelihood": 0.3}, "refuse"),     # contradicts a record
    ("explore", {"fits_records": True, "likelihood": -0.1}, "refuse"),     # held-out down
    ("exploit", {"likelihood": 0.2, "success": 0.0}, "accept"),
    ("exploit", {"likelihood": 0.0, "success": 0.1}, "accept"),            # success up, likelihood not down
    ("exploit", {"likelihood": -0.01, "success": 0.5}, "refuse"),
    ("exploit", {"likelihood": 0.0, "success": 0.0}, "refuse"),            # no gain
    ("refactor", {"replay_identical": True, "success": 0.0, "vector": -3}, "accept"),
    ("refactor", {"replay_identical": True, "success": 0.0, "vector": {"before": [9, 4], "after": [5, 4]}}, "accept"),
    ("refactor", {"replay_identical": False, "success": 0.0, "vector": -3}, "refuse"),
    ("refactor", {"replay_identical": True, "success": -0.1, "vector": -3}, "refuse"),
    ("refactor", {"replay_identical": True, "success": 0.0, "vector": [0, 1]}, "refuse"),   # vector up
    ("refactor", {"replay_identical": True, "success": 0.0, "vector": -3, "likelihood": -0.2}, "refuse"),
    ("exploit", {"likelihood": 0.2}, "refuse"),                            # success not measured
])
def test_the_verdict_per_phase(phase, numbers, verdict):
    v = gate.judge(phase, numbers)
    assert v["verdict"] == verdict, v["why"]
    if "success" not in numbers and phase == "exploit":
        assert "success: not measured" in v["why"]


def test_an_accept_is_confirmed_and_journaled(root, tmp_path):
    j = tmp_path / "state" / "journal.jsonl"
    gain = fake({"likelihood": 0.25, "success": 0.1})
    g = gate.Gate(policy="experimental", gain=gain, journal=j)
    plant = (root / "domains/plant/plant.llp").read_text()
    pl = g.propose("program", "plant", root, text=plant + "% a comment the candidate adds\n",
                   phase="exploit", cost=0.004)
    assert pl["verdict"]["verdict"] == "accept" and pl["written"] == ["domains/plant/plant.llp"]
    assert (root / "domains/plant/plant.llp").read_text().endswith("% a comment the candidate adds\n")
    assert g.store["plans"] == {}                                             # confirmed: no plan waits
    (plan, phase), = gain.seen
    assert phase == "exploit" and plan["files"][0]["path"] == "domains/plant/plant.llp" and "write" not in plan
    e, = g.journal()
    assert e["event"] == "proposal" and e["verdict"] == "accept" and e["phase"] == "exploit"
    assert e["numbers"] == {"likelihood": 0.25, "success": 0.1} and e["cost"] == 0.004
    assert e["written"] == ["domains/plant/plant.llp"] and e["files"] == ["domains/plant/plant.llp"]
    assert json.loads(j.read_text().splitlines()[0])["token"] == pl["token"]


def test_a_refuse_writes_nothing_and_is_remembered(root):
    g = gate.Gate(policy="experimental", gain=fake({"likelihood": -0.2, "success": 0.3}))
    pl = g.propose("extension", "plant", root, name="tired", text=EXT, phase="exploit")
    assert pl["verdict"]["verdict"] == "refuse" and pl["refused"] == ["domains/plant/extensions/tired.llp"]
    assert any(w.startswith("not: a generalization gain") for w in pl["verdict"]["why"])
    assert not (root / "domains/plant/extensions/tired.llp").exists()
    again = g.propose("extension", "plant", root, name="tired", text=EXT + "% more\n", phase="exploit")
    assert "+% more" in again["files"][0]["since_diff"]                       # since the refusal, as the person's
    assert [e["verdict"] for e in g.journal()] == ["refuse", "refuse"]


def test_numbers_given_instead_of_a_hook(root):
    """A driver that measured the numbers itself (the CLI, pi) passes them in the proposal."""
    g = gate.Gate(policy="experimental")
    pl = g.propose("extension", "plant", root, name="tired", text=EXT, phase="explore",
                   numbers={"fits_records": True, "likelihood": 0.0})
    assert pl["verdict"]["verdict"] == "accept" and (root / "domains/plant/extensions/tired.llp").is_file()
    with pytest.raises(gate.Refused, match="needs the numbers"):
        g.propose("extension", "plant", root, name="other", text=EXT, phase="explore")
    with pytest.raises(gate.Refused, match="in a phase"):
        g.propose("extension", "plant", root, name="other", text=EXT, numbers={})


def test_the_unjudged_kinds_still_wait_for_the_person(root):
    g = gate.Gate(policy="experimental", gain=fake({}))
    pl = g.propose("readme", "plant", root, text="A plant that dries out.")
    assert "verdict" not in pl and pl["token"] in g.store["plans"] and g.journal() == []


def test_the_human_gate_is_unchanged_without_a_policy(root):
    g = gate.Gate()
    pl = g.propose("extension", "plant", root, name="tired", text=EXT, phase="exploit", numbers={"likelihood": 1})
    assert "verdict" not in pl and not (root / "domains/plant/extensions/tired.llp").exists()
    assert pl["token"] in g.store["plans"] and g.journal() == []
    g.confirm(pl["token"])
    assert (root / "domains/plant/extensions/tired.llp").is_file()
    with pytest.raises(gate.Refused, match="policy"):
        gate.Gate(policy="lenient")


def test_ask_waits_then_the_answer_is_read_back(root, tmp_path):
    g = gate.Gate.at(tmp_path / "state", policy="experimental")
    a = g.propose("ask", "plant", root, **ASK)
    assert a["kind"] == "ask" and a["files"] == [] and a["ask"]["question"] == ASK["question"]
    assert sorted(p.name for p in (root / "domains/plant").iterdir()) == sorted(
        p.name for p in (CAT / "domains/plant").iterdir() if p.name not in ("records", "build", "__pycache__"))
    assert [x["token"] for x in g.asks()] == [a["token"]] and g.answers(a["token"]) is None
    with pytest.raises(gate.Refused, match="answered, not confirmed"):
        g.confirm(a["token"])
    with pytest.raises(gate.Refused, match="answered, not refused"):
        g.refuse(a["token"])
    out = g.answer(a["token"], {"kind": "phase", "phase": "explore", "text": "probe the fridge"})
    assert out["instruction"] == {"kind": "phase", "phase": "explore", "text": "probe the fridge"}
    # the loop re-engages in another process: the answer and the snapshot are in the state
    g2 = gate.Gate.at(tmp_path / "state")
    back = g2.answers(a["token"])
    assert back["instruction"]["phase"] == "explore" and back["ask"]["snapshot"] == "run/snap/7.json"
    assert g2.asks() == []
    assert [e["event"] for e in g2.journal()] == ["ask", "answer"]
    assert g2.journal()[0]["reason"] == "no_progress"
    with pytest.raises(gate.Refused, match="no such ask"):
        g2.answer(a["token"], "again")


def test_the_same_ask_twice_is_two_asks(root):
    g = gate.Gate()                                   # an ask needs no policy: the console's person answers it
    a = g.propose("ask", "plant", root, **ASK)
    g.answer(a["token"], "the fridge is locked: look for the key")
    b = g.propose("ask", "plant", root, **ASK)
    assert b["token"] != a["token"] and g.answers(a["token"])["instruction"] == {
        "kind": "note", "text": "the fridge is locked: look for the key"}


@pytest.mark.parametrize("bad, words", [
    ({"reason": "bored"}, "reason"),
    ({"phase": "dream"}, "phase"),
    ({"question": " "}, "question"),
    ({"options": ["a", "b", "c", "d"]}, "at most three"),
])
def test_an_ask_record_is_checked(root, bad, words):
    with pytest.raises(gate.Refused, match=words):
        gate.Gate().propose("ask", "plant", root, **dict(ASK, **bad))


@pytest.mark.parametrize("ins, words", [
    ({"kind": "shrug"}, "an instruction of kind"),
    ({"kind": "note"}, "note: text"),
    ({"kind": "edit", "path": "plant.llp"}, "path, old, new"),
    ({"kind": "budget", "beats": "many"}, "a number"),
    ({"kind": "phase", "phase": "nap"}, "explore"),
])
def test_an_answer_is_checked(root, ins, words):
    g = gate.Gate()
    a = g.propose("ask", "plant", root, **ASK)
    with pytest.raises(gate.Refused, match=words):
        g.answer(a["token"], ins)
    assert g.asks()                                                           # a bad answer leaves it waiting
    g.answer(a["token"], {"kind": "abandon"})


def test_the_cli_journals_and_answers_across_calls(root, tmp_path):
    st = tmp_path / "state"

    def cli(*args, stdin=""):
        r = subprocess.run([sys.executable, "-m", "metis.gate", "--state", str(st), *args, "--json"],
                           input=stdin, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return json.loads(r.stdout)

    pl = cli("--policy", "experimental", "propose", "extension", "plant", "--root", str(root),
             stdin=json.dumps({"name": "tired", "text": EXT, "phase": "explore", "cost": 0.01,
                               "numbers": {"fits_records": True, "likelihood": 0.0}}))
    assert pl["verdict"]["verdict"] == "accept" and pl["written"] == ["domains/plant/extensions/tired.llp"]
    a = cli("propose", "ask", "plant", "--root", str(root), stdin=json.dumps(ASK))
    assert cli("answers")["asks"][0]["token"] == a["token"]
    cli("answer", a["token"], stdin="skip the fridge for now")
    assert cli("answers", "--token", a["token"])["instruction"]["text"] == "skip the fridge for now"
    assert [e["event"] for e in cli("journal")] == ["proposal", "ask", "answer"]


@pytest.mark.parametrize("numbers, words", [
    ({"likelihood": "0.2", "success": 0.1}, "likelihood is a real number"),
    ({"likelihood": True, "success": 0.0}, "likelihood is a real number"),        # true is not +1
    ({"likelihood": 0.2, "success": float("nan")}, "success is a real number"),
    ({"replay_identical": 1, "success": 0.0, "vector": -1}, "replay_identical is true or false"),
    ({"replay_identical": True, "success": 0.0, "vector": "-3"}, "vector is the complexity vector"),
    ({"replay_identical": True, "success": 0.0, "vector": {"before": [9], "after": [True]}}, "vector is the complexity vector"),
    ({"fits_records": "yes", "likelihood": 0.0}, "fits_records is true or false"),
    ("good", "an object of the measures"),
])
def test_numbers_of_the_wrong_type_refuse_in_words(numbers, words):
    """opus's review F1: the numbers are type-checked before any comparison."""
    phase = "refactor" if isinstance(numbers, dict) and "replay_identical" in numbers else (
        "explore" if isinstance(numbers, dict) and "fits_records" in numbers else "exploit")
    with pytest.raises(gate.Refused, match=words):
        gate.judge(phase, numbers)


def test_a_bool_likelihood_writes_nothing_and_leaves_no_plan(root):
    """Under the experimental policy a refusal of the numbers writes nothing and leaves no plan
    waiting (from the hook as from the args), and the journal has no verdict."""
    for g in (gate.Gate(policy="experimental", gain=fake({"likelihood": True, "success": True})),
              gate.Gate(policy="experimental")):
        args = {} if g.gain else {"numbers": {"likelihood": True, "success": 0.0}}
        with pytest.raises(gate.Refused, match="likelihood is a real number"):
            g.propose("extension", "plant", root, name="tired", text=EXT, phase="exploit", **args)
        assert not (root / "domains/plant/extensions/tired.llp").exists()
        assert g.store["plans"] == {} and g.journal() == []


def test_the_cli_refuses_bad_numbers_as_json(root, tmp_path):
    """The CLI answers the refusal as its JSON error (exit 1), not a traceback."""
    r = subprocess.run([sys.executable, "-m", "metis.gate", "--state", str(tmp_path / "st"), "--policy", "experimental",
                        "propose", "extension", "plant", "--root", str(root), "--json"],
                       input=json.dumps({"name": "tired", "text": EXT, "phase": "exploit",
                                         "numbers": {"likelihood": "lots", "success": 0.1}}),
                       capture_output=True, text=True)
    assert r.returncode == 1 and "Traceback" not in r.stderr, r.stderr
    assert "likelihood is a real number" in json.loads(r.stdout)["error"]
