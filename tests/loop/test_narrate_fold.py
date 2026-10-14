"""narrate: machinery folds and the cut rule (the user's wildfire finding,
via the architect, 2026-10-14). A rule whose %% doc opens with [fold …]
is bookkeeping: its events become one line with the net-new atoms of the
named predicates; a narration the endpoint cut at its cap ends on a
sentence and says so."""
import io

from metis.loop import narrate as N

WALK = """catalog walk 1.
provenance text "narrate fold test" "2026-10-14".
layers (dyn).
stages (go).
namespace wk produce (dyn).
pred tick : wk.
pred a : wk.
pred b : wk.
pred c : wk.
pred lit : wk.
pred dark : wk.
weight pl reads {}.
stage go {
  %% [fold c] the walk passes a cell.
  w1 [dyn] : tick -o a.
  %% [fold c]
  w2 [dyn] : a -o b.
  %% [fold]
  w3 [dyn] : b -o c.
  %% the lamp is lit | the lamp stays dark
  light [dyn] : c -o (lit @w pl | dark @w ~pl).
}
#init tick 1
#steps 10
"""


def _session(tmp_path):
    from metis.fw.registry import Registry
    from metis.lang.repl import Repl
    p = tmp_path / "walk.llp"
    p.write_text(WALK)
    out = io.StringIO()
    r = Repl(p, seed=1, stdout=out, registry=Registry().weight("pl", 0.5))
    r.onecmd("run 10")
    out.truncate(0), out.seek(0)
    r.onecmd("narrate user")
    return r, out.getvalue()


def test_machinery_folds_into_one_line(tmp_path):
    _, txt = _session(tmp_path)
    beats = txt.split("== the final state ==")[0]
    assert "4 events" in beats and "(3 of them machinery, folded)" in beats
    fold = [l for l in beats.splitlines() if "machinery, folded —" in l]
    assert len(fold) == 1 and fold[0].lstrip().startswith("1–3 |")
    assert "the walk passes a cell" in fold[0] and "⇒ left: c" in fold[0]
    assert not any(f"| {w} " in beats for w in ("w1", "w2", "w3"))      # no folded event listed
    told = [l for l in beats.splitlines() if "light" in l]
    assert len(told) == 1 and told[0].lstrip().startswith("4 |")
    # a ⊕ branch tells its own phrase (light-1: lit, light-2: dark)
    assert ("the lamp is lit." in told[0]) != ("the lamp stays dark." in told[0])
    rules = txt.split("== the rules")[1]
    assert "light [dyn]" in rules and "w1 [dyn]" not in rules and "[fold" not in rules
    assert "in at most 80 words" in txt                                 # one told + one fold line


def test_a_bare_fold_run_with_nothing_left_is_not_told():
    docs = {"w": "[fold]", "s": "the story moves."}
    assert N.fold_rules(docs) == {"w": ("", ())}
    assert N.fold_rules({"x": "[fold tally hold] counts the fire."}) == {"x": ("counts the fire", ("tally", "hold"))}


def test_the_cut_ends_on_a_sentence():
    cut = "The fire caught at x1,y2. Then it spread to x2,y3 while the `scan"
    assert N.end_on_sentence(cut, {"cut": True, "tokens": 700}) == \
        "The fire caught at x1,y2.\n\n(narration cut at 700 tokens; ask fewer beats)"
    assert N.end_on_sentence(cut, {"cut": False}) == cut                 # not cut: as it came
    assert N.end_on_sentence("no sentence ends here", {"cut": True}) == \
        "(narration cut at the token cap; ask fewer beats)"
    assert N.end_on_sentence('He said "stop." and the', {"cut": True, "tokens": 5}).startswith('He said "stop."')


def test_narrate_applies_the_cut_from_the_endpoint(tmp_path, monkeypatch):
    r, _ = _session(tmp_path)

    def fake(spec, prompt, timeout=600, options=None, system=None, meta=None):
        meta.update(cut=True, tokens=12)
        return "The lamp was lit. Then the walk went on to the"
    monkeypatch.setattr(N.llm, "complete", fake)
    out = io.StringIO()
    r.stdout = out
    r.onecmd("narrate")
    assert out.getvalue().strip() == "The lamp was lit.\n\n(narration cut at 12 tokens; ask fewer beats)"


def test_ollama_reports_the_cap(monkeypatch):
    import json
    from metis.loop import ollama

    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass
    monkeypatch.setattr(ollama.urllib.request, "urlopen", lambda req, timeout=None: R(json.dumps(
        {"response": "half a sen", "done_reason": "length", "eval_count": 700}).encode()))
    meta = {}
    assert ollama.generate("p", model="m", host="http://x", meta=meta).strip() == "half a sen"
    assert meta == {"cut": True, "tokens": 700}
