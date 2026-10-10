"""The RDDL experiment's step-1 loop findings, fixed (metis-loop
experiments/rddl-elevators/RESULTS-step1.md): a blank world's candidate
gates as a base edit in formalize's repair loop; a revision's own
#init/#steps survive accept; directives tolerate spaces inside atoms;
`save` writes the accepted base; a parse finding says the rest is
unchecked; and the terminal-only `transcript`."""
import io
from pathlib import Path

from metis.lang.directives import embedded_case
from metis.lang.formalize import admit_text_pack
from metis.lang.repl import Repl
from metis.loop.core import base_shaped, formalize_loop

BLANK = "catalog scratch 1.\nlayers (dyn).\nstages (z).\nstage z {\n}\n"

BORN = """catalog toy 1.
provenance text "a toy" "2026-10-09".
layers (dyn).
stages (run).
type room {r1 r2}.
namespace t produce (dyn).
pred at(room) : t.
pred tick : t.
stage run {
  %% "it moves between rooms"
  go [dyn] : at(r1) * tick -o at(r2).
}
#init at(r1) 1
#init tick 2
#steps 3
"""


def blank(tmp_path):
    p = tmp_path / "blank.llp"
    p.write_text(BLANK)
    out = io.StringIO()
    r = Repl(p, seed=1, stdout=out)

    def say(line):
        out.truncate(0)
        out.seek(0)
        r.onecmd(line)
        return out.getvalue()
    return r, say


def test_base_shaped_and_the_repair_loop_gates_a_blank_world_as_revise(
        tmp_path):
    assert base_shaped(BORN)
    assert not base_shaped("catalog p 1.\nextends toy.\nstages (run).\n")
    base = tmp_path / "blank.llp"
    base.write_text(BLANK)
    pack = formalize_loop("it moves between rooms", BLANK,
                          lambda prompt: BORN, base_path=base,
                          max_rounds=1, mode="pack")
    assert not pack.ok and any("stage" in f for f in pack.report.findings)
    rev = formalize_loop("it moves between rooms", BLANK,
                         lambda prompt: BORN, base_path=base,
                         max_rounds=1, mode="auto")
    assert rev.ok and rev.mode == "revise"


def test_formalize_on_a_blank_world_arms_accept_and_adopts_directives(
        tmp_path, monkeypatch):
    r, say = blank(tmp_path)
    r.cmd_text('"it moves between rooms"')
    import metis.loop.core as core
    monkeypatch.setattr(core, "run_backend", lambda cmd, prompt: BORN)
    r.llm = "command:true"
    out = say("formalize")
    assert '"gated_as": "revise (base edit)"' in out and '"ok": true' in out
    acc = say("accept")
    assert '"adopted_directives"' in acc
    assert r.case["init"] == {"at(r1)": 1, "tick": 2}
    assert r.case["steps"] == 3
    assert r.counts == {"at(r1)": 1, "tick": 2}     # the born world starts
    assert "go" in say("events")
    assert r.doc.catalog.name == "toy"


def test_directives_tolerate_spaces_inside_atoms(tmp_path):
    assert embedded_case("#init at(e0, f0) 1\n")["init"] == {"at(e0,f0)": 1}
    r, say = blank(tmp_path)
    assert "at(e0,f0):1" in say("#init at(e0, f0) 1")
    assert "bad directive" not in say("#init at( e1 ,f2 ) 2")
    assert r.counts["at(e1,f2)"] == 2


def test_save_after_a_blank_world_accept_writes_the_born_catalog(tmp_path):
    r, say = blank(tmp_path)
    r.cmd_text('"it moves between rooms"')
    r.cmd_llp(BORN)
    assert '"ok": true' in say("revise")
    say("accept")
    out = tmp_path / "born.llp"
    say(f"save {out}")
    text = out.read_text()
    assert "catalog toy 1." in text and "catalog scratch" not in text
    assert "go [dyn]" in text and "#init at(r1) 1" in text


def test_a_parse_finding_says_the_rest_is_unchecked():
    rep = admit_text_pack("x", "catalog a 1.\nfoo 1 2.\nlayers (d).\n"
                          "stages (s).\n")
    (f,) = [f for f in rep.findings if f.startswith("parse:")]
    assert "parsing stopped at line 2; lines 3..4 of the candidate are " \
        "UNCHECKED" in f


def test_transcript_is_terminal_only_and_logs_the_session(tmp_path):
    r, say = blank(tmp_path)
    assert "terminal-only" in say("transcript")       # a non-terminal
    r.terminal = True                                 # what main() sets
    log = tmp_path / "t.md"
    assert "transcript open" in say(f"transcript {log}")
    r.stdin = io.StringIO('"it moves between rooms"\nEOF\n')
    say("text <<EOF")
    r.stdin = io.StringIO("a session prompt\nEOF\n")
    say("formalize system <<EOF")
    r.stdin = io.StringIO(BORN + "EOF\n")
    say("llp <<EOF")
    say("revise")
    say("accept")
    say("transcript off")
    t = log.read_text()
    assert t.startswith("# metis transcript — blank.llp")
    assert "### system prompt `formalize` (in effect" in t
    assert "### system prompt `formalize` (changed; source: session)" in t
    assert "a session prompt" in t
    assert "## " in t and "`text <<EOF`" in t and "it moves between" in t
    assert "#### candidate snapshot (after `llp`)" in t and "go [dyn]" in t
    assert "**accepted** — the world is now key" in t
    assert "_transcript closed" in t
    # the web wire never reaches it: not an intent verb
    from metis.loop.wire import INTENT_VERBS
    assert "transcript" not in INTENT_VERBS


def test_accept_survives_an_over_budget_revision_diff(tmp_path):
    """Wildfire step 2: accepting an admitted revision computed the
    exact before/after diff and died on StateBudgetExceeded — the diff
    is a readout, never the gate (revise already said so)."""
    from metis.kernel import StateBudgetExceeded
    r, out = blank(tmp_path)
    r.cmd_text("a toy")
    r.cmd_llp(BORN)
    assert r.cmd_revise("")["ok"]

    def blow(rep):
        raise StateBudgetExceeded(3, 6, 11, 10)
    r._revision_diff = blow
    res = r.cmd_accept("")
    assert res["ok"] and res["diffs"] == []
    assert "too large" in res["diff_note"]
    assert r.doc.catalog.name == "toy"
