"""The loop, merged into the kernel (metis.loop): core-seam gates on
the stage-45 fixtures, and the lifecycle verbs mounted on the metis
shell — tabula rasa via `revise` of a blank served world, pack
extension with live re-serving, the intent policy, and the narrate
boundary with `cat` in the LLM seat (the material is then auditable:
the narration IS the prompt)."""
import os
import stat

import pytest
from pathlib import Path

from metis.loop import admit, formalize_loop
from metis.loop.wire import dispatch, parse_intent, run_intent

ROOT = Path(__file__).resolve().parents[2]
LANG = Path(os.environ.get(
    "METIS_LANG", ROOT.parent / "metis-lang")).resolve()
TEXT = "%% market coin: a mark trades on the coin flip outcome."

BLANK = """catalog blank 1.
layers (dyn).
stages (z).
stage z {
}
"""

MEADOW = """catalog meadow 1.
provenance text "repl gate: wolf and flock" "2026-09-27".
layers (dyn pack).
stages (hunt rest).
namespace md produce (dyn).
pred wolf : md.
pred sheep : md.
pred night : md.
pred fed : md.
pred hungry : md.
weight pk reads {}.
stage hunt {
  %% catch.
  catch [dyn] : wolf * sheep * night -o wolf * fed  @w pk.
  %% evade.
  evade [dyn] : wolf * night -o wolf * hungry  @w ~pk.
}
%% settle.
qui hr [dyn] : hunt -o rest.
"""

SHEPHERD = """catalog shepherd 1.
extends meadow.
provenance text "repl gate: the tally pack" "2026-09-27".
namespace sh produce (pack).
pred tally : sh.
pred loss : sh.
pred safe : sh.
stage rest {
  %% loss.
  n1 [pack] : tally * $fed -o loss.
  %% safe.
  n2 [pack] : tally * $hungry -o safe.
}
"""


def _fixture(name):
    return (LANG / "corpus" / name).read_text()


def blank_shell(tmp_path):
    import io

    from metis.fw.registry import Registry
    from metis.lang.repl import Repl
    p = tmp_path / "blank.llp"
    p.write_text(BLANK)
    # MEADOW's `@w pk / @w ~pk` pair: an unbound port is a gate finding
    # (PLUGIN-PORTS.md); pk = 0.5 keeps the pair's old 1:1 split
    return Repl(p, seed=1, stdout=io.StringIO(),
                registry=Registry().weight("pk", 0.5))


# -- core seams (unchanged from the metis-loop repo) -----------------

def test_admit_pack_ok():
    rep = admit(TEXT, _fixture("packmark.llp"),
                base_path=LANG / "corpus/packbase.llp",
                scenarios=[{"coin": 1, "mkt": 1}], steps=4)
    assert rep.ok, rep.findings
    assert rep.scenarios and rep.scenarios[0].key


def test_admit_vocabulary_drift_caught():
    bad = _fixture("packmark.llp").replace("mkt", "unlock", 1)
    rep = admit(TEXT, bad, base_path=LANG / "corpus/packbase.llp")
    assert not rep.ok and rep.findings


def test_repair_loop_converges(tmp_path):
    good = LANG / "corpus/packmark.llp"
    bad = tmp_path / "bad.llp"
    bad.write_text(_fixture("packmark.llp").replace("mkt", "unlock", 1))
    marker = tmp_path / "called"
    backend = tmp_path / "backend.sh"
    backend.write_text(
        "#!/bin/sh\n"
        f"if [ -e {marker} ]; then cat {good}; else touch {marker}; "
        f"cat {bad}; fi\n")
    backend.chmod(backend.stat().st_mode | stat.S_IEXEC)
    res = formalize_loop(TEXT, "coin, heads, tails, mkt",
                         [str(backend)],
                         base_path=LANG / "corpus/packbase.llp",
                         scenarios=[{"coin": 1, "mkt": 1}], steps=4)
    assert res.ok and res.rounds == 2, (res.rounds, res.history)
    assert res.history[0][1] != []            # round 1 had findings
    assert res.report and res.report.scenarios


# -- the lifecycle on the shell --------------------------------------

def test_shell_tabula_rasa_cycle(tmp_path):
    """revise a blank served world into meadow, extend with the
    shepherd pack — the session re-serves LIVE at each accept."""
    r = blank_shell(tmp_path)
    d = lambda op, arg="": dispatch(r, op, arg)   # noqa: E731
    assert d("init", "wolf=1,sheep=3,night=2,tally=2")["ok"]
    assert d("steps", "6")["ok"]
    key0 = r.key
    d("text", "%% meadow world")
    assert r.cmd_llp(MEADOW)["ok"]
    g1 = d("revise")
    assert g1["ok"] and g1["rules"], g1
    assert r.cmd_accept("")["ok"]
    assert r.key != key0                    # re-served live
    tr = d("trace", "7")
    assert tr["ok"] and tr["events"][0].startswith("hunt/"), tr
    mp = d("map")
    assert mp["ok"] and mp["events"][0] == "hunt/catch"
    mg = d("marginals", "fed hungry")
    assert abs(mg["marginals"]["hungry"] - 0.5) < 1e-12
    # the pack gates against the REVISED world, not the blank file
    d("text", "%% shepherd tally")
    r.cmd_llp(SHEPHERD)
    g2 = d("gate")
    assert g2["ok"], g2
    diffs = {x[0]: (x[1], x[2]) for x in g2["scenarios"][0]["diffs"]}
    assert diffs["loss"][0] == 0.0 and diffs["loss"][1] > 0.8
    assert r.cmd_accept("")["ok"]
    tr2 = d("trace", "7")
    assert any(e.startswith("rest/") for e in tr2["events"]), tr2
    assert not d("nosuch")["ok"]


def test_shell_revise_replaces_world_with_diff(tmp_path):
    r = blank_shell(tmp_path)
    d = lambda op, arg="": dispatch(r, op, arg)   # noqa: E731
    d("init", "wolf=1,sheep=1,night=1,tally=1")
    d("steps", "4")
    d("text", "%% base")
    r.cmd_llp(MEADOW)
    assert d("revise")["ok"] and r.cmd_accept("")["ok"]
    d("text", "%% tally")
    r.cmd_llp(SHEPHERD)
    assert d("gate")["ok"] and r.cmd_accept("")["ok"]
    rev = MEADOW.replace("-o wolf * fed ", "-o wolf * fed * fed ")
    d("text", "%% revised: a catch gorges the wolf")
    r.cmd_llp(rev)
    g = d("revise")
    assert g["ok"] and g["revision"], g
    a = r.cmd_accept("")
    assert a["ok"] and a["dropped_packs"] == 1, a
    dd = {x[0]: (x[1], x[2]) for x in a["diffs"]}
    assert dd["loss"][1] == 0.0 and dd["tally"] == (0.0, 1.0), a
    # the shepherd pack must be re-proposed against the new base
    d("text", "%% tally again")
    r.cmd_llp(SHEPHERD)
    assert d("gate")["ok"] and r.cmd_accept("")["ok"]


def test_short_name_collision_warned(tmp_path):
    collide = """catalog p 1.
extends meadow.
provenance text "t" "2026-09-27".
namespace px produce (pack).
pred mark : px.
stage rest {
  %% shadow name.
  catch [pack] : mark * $fed -o mark.
}
"""
    r = blank_shell(tmp_path)
    d = lambda op, arg="": dispatch(r, op, arg)   # noqa: E731
    d("init", "wolf=1,sheep=1,night=1,mark=1")
    d("steps", "4")
    d("text", "%% base")
    r.cmd_llp(MEADOW)
    assert d("revise")["ok"] and r.cmd_accept("")["ok"]
    d("text", "%% name collision")
    r.cmd_llp(collide)
    g = d("gate")
    assert g["ok"] and any("catch" in w and "short" in w
                           for w in g["warnings"]), g


# -- the wire / intent policy ----------------------------------------

def test_intent_policy():
    assert parse_intent(":narrate 2") is not None
    assert parse_intent(":llm anthropic:claude-opus-5-5") is not None
    assert parse_intent(":llm command:rm -rf /") is None
    assert parse_intent(":formalize command:sh") is None
    assert parse_intent(":accept") is None       # approval is human
    assert parse_intent(":export x") is None     # no file writes
    assert parse_intent("gate") is None          # ':' required


def test_wire_and_narrate_cat_backend(tmp_path):
    r = blank_shell(tmp_path)
    d = lambda op, arg="": dispatch(r, op, arg)   # noqa: E731
    d("init", "wolf=1,sheep=2,night=2")
    d("steps", "5")
    d("text", "%% meadow")
    r.cmd_llp(MEADOW)
    assert d("revise")["ok"] and r.cmd_accept("")["ok"]
    assert run_intent(r, ":llm command:cat")["ok"] is False
    assert d("llm", "command:cat")["ok"]          # raw wire: allowed
    out = d("narrate", "3")
    assert out["ok"], out
    txt = out["narration"]
    assert "beats" in txt and "== the rules" in txt
    assert "catch" in txt and "INDEPENDENT" in txt
    assert "⇒ gone:" in txt and "new:" in txt     # glossed effects


# -- D5: the diagnostics-only repair prompt (spec 08) ----------------

def test_repair_prompt_carries_diagnostics_not_the_base(tmp_path):
    """Repair rounds send the TEXT, the candidate's own source and the
    failing diagnostics' slices — never the base catalog; `full` keeps
    the pre-D5 prompt (vocabulary + findings)."""
    from metis.loop.core import render_prompt, render_repair_prompt
    base = LANG / "corpus/packbase.llp"
    bad = _fixture("packmark.llp").replace("-o seen.", "-o seen * heads.")
    rep = admit(TEXT, bad, base_path=base)
    p = render_repair_prompt(TEXT, bad, rep)
    assert "error containment-produce @" in p       # the slice
    assert "| layers (dyn pack)." in p               # its context
    assert bad.strip() in p                          # its own source
    assert "provenance-shape" not in p               # warnings stay out
    assert "stage roll" not in p                     # NO base source
    full = render_prompt(TEXT, base.read_text(), rep.findings)
    assert "stage roll" in full and "containment-produce" not in full


def test_repair_modes_and_start(tmp_path):
    """`start` repairs a given candidate (round 1 = its verdict); each
    mode builds its own repair prompt; an unknown mode is loud."""
    good = _fixture("packmark.llp")
    bad = good.replace("mkt", "unlock", 1)
    for mode, marker in (("diagnostics", "error pred-undeclared"),
                         ("full", "FAILED these gates")):
        seen = []
        res = formalize_loop(TEXT, "coin, heads, tails, mkt",
                             lambda p: (seen.append(p), good)[1],
                             base_path=LANG / "corpus/packbase.llp",
                             repair=mode, start=bad)
        assert res.ok and res.rounds == 2 and len(seen) == 1
        assert marker in seen[0] and res.prompts == seen
    with pytest.raises(ValueError, match="repair must be"):
        formalize_loop(TEXT, "", lambda p: good, repair="nope")


def test_repair_prompt_caps_a_runaway_candidate():
    """A runaway model output is echoed CAPPED, and the cut is said."""
    from metis.loop.core import CANDIDATE_CAP, render_repair_prompt
    base = LANG / "corpus/packbase.llp"
    junk = "concept " * 20000
    rep = admit(TEXT, junk, base_path=base)
    p = render_repair_prompt(TEXT, junk, rep)
    assert len(p) < CANDIDATE_CAP + 3000
    assert "candidate truncated" in p


def test_gloss_keeps_quoted_doc_verbatim():
    """'narrate user' (specialist step 1, nina): wgc's row doc is a verbatim
    quote starting with the article "A"; the event binds A=far. Role
    letters are substituted outside quotes only."""
    from metis.loop.narrate import gloss
    docs = {"row": '"A farmer with a wolf, a goat, and a cabbage must cross '
                   'a river by boat." (the boat crosses with the farmer alone too)',
            "warn": "C warns D about the gossip."}
    g = gloss("3 | row [A=far,B=near]", docs)
    assert '"A farmer with a wolf' in g and "far farmer" not in g
    assert "ada warns bo" in gloss("1 | warn [C=ada,D=bo]", docs)
