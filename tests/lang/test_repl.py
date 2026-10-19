"""The metis shell: a stateful session over a .llp — stepping on the
reference sampler's normative selection (an unperturbed seeded session
== run_trace), assert/retract/undo, exact query from the current
state, reload-keeps-serving, and save-as-directives closing the loop
back to `metisc --run`."""

import io
from pathlib import Path

from metis.kernel.refsample import run_trace
from metis.lang import compile_catalog, parse
from metis.lang.directives import embedded_case
from metis.lang.repl import Repl

ROOT = Path(__file__).resolve().parents[2]
from lovetriangle import CATALOG, needs  # noqa: E402  (the catalog's Ceptre port)

pytestmark = needs


def shell(path=CATALOG, seed=1):
    return Repl(path, seed=seed, stdout=io.StringIO())


def run(r, *cmds):
    for c in cmds:
        r.onecmd(c)
    out, r.stdout = r.stdout.getvalue(), io.StringIO()
    return out


def test_sampled_session_is_the_reference_trace():
    r = shell()
    run(r, "run 6")
    text = CATALOG.read_text()
    case = embedded_case(text)
    prog = compile_catalog(parse(text)).catalog.ground(case["init"])
    _, (stage, kappa) = run_trace(prog, 6, 1)
    assert (r.stage, frozenset(r.counts.items())) == (stage, kappa)
    assert len(r.fired) == 6


def test_events_step_and_undo():
    r = shell()
    out = run(r, "events")
    assert out.count("jealousy[") == 5 and "p=0.2" in out
    before = dict(r.counts)
    out = run(r, "step 3", "undo")
    assert "fired: jealousy[" in out
    assert r.counts == before and r.fired == []
    out = run(r, "step nosuch")
    assert "no enabled candidate" in out
    assert r.undo_stack == []          # a refused step leaves no mark


def test_assert_query_prices_the_heartbreak():
    """The falling-factorial gate, interactively: with eros + anger^3
    for ONE pair only, P(move_on) = 3!/(3!+2) = 6/8 — the eros dies
    with p=0.75 in one step (mirrors test_love_triangle.py)."""
    r = shell()
    run(r, "reset")
    r.counts = {"eros(helena,hermia)": 1, "anger(helena,hermia)": 3,
                "eros(hermia,helena)": 1}
    out = run(r, "query eros(helena,hermia) 1")
    assert "= 0.25" in out
    out = run(r, "assert anger(helena,hermia)",
              "retract anger(helena,hermia) 5")
    assert "anger(helena,hermia):4" in out and "removed all" in out


def test_save_writes_a_runnable_directive_file(tmp_path):
    r = shell()
    run(r, "step 1", "step 1")
    dst = tmp_path / "session.llp"
    run(r, f"save {dst}")
    text = dst.read_text()
    case = embedded_case(text)
    assert case["init"] == r.counts and case["steps"] == 6
    # the saved file parses and grounds — replayable by either tool
    doc = compile_catalog(parse(text))
    assert doc.catalog.ground(case["init"]).init == r.counts


def test_reload_keeps_serving_on_a_broken_edit(tmp_path):
    dst = tmp_path / "lt.llp"
    dst.write_text(CATALOG.read_text())
    r = shell(dst)
    key = r.key
    dst.write_text("catalog broken 1.\nstage go {\n  r : a -o .\n}\n")
    out = run(r, "reload")
    assert "previous catalog still serving" in out
    assert r.key == key and r.counts  # session intact
    dst.write_text(CATALOG.read_text())
    assert "unchanged" in run(r, "reload")


def test_buffer_lines_apply_to_the_live_session():
    """shift+enter's contract: '#'-directive lines are live commands
    (#init ASSIGNS the count), '%' comments and blanks are inert, so
    sending any slice of the buffer is safe."""
    r = shell()
    out = run(r, "#init anger(helena,hermia) 3")
    assert "anger(helena,hermia):3" in out
    assert r.counts["anger(helena,hermia)"] == 3
    out = run(r, "#init anger(helena,hermia) 3")   # assign, not add
    assert r.counts["anger(helena,hermia)"] == 3
    run(r, "#steps 1")
    assert r.case["steps"] == 1
    out = run(r, "#init eros(helena,lysander) 0",   # assign 0 removes
              "#init eros(lysander,hermia) 0",
              "#query eros(helena,hermia)")        # default horizon: 1
    assert "eros(helena,lysander)" not in r.counts
    assert "= 0.25" in out                          # 6/8 heartbreak
    assert "eros(helena,hermia)" in r.case["queries"]
    out = run(r, "% a comment", "", "#init bad n",
              "#context init")
    lines = [l for l in out.splitlines() if l]
    assert lines == ["bad directive: #init bad n",
                     "bad directive: #context init"]
    run(r, "undo", "undo", "undo", "undo")          # marks unwind
    assert "anger(helena,hermia)" not in r.counts


def test_rule_text_gets_the_reload_hint_not_noise():
    """A selection of RULE source sent to the REPL (the user's
    shift+enter on the jealousy block) points at save+reload; `-o`
    never falls into the '-ATOM' retract shorthand; unknown commands
    say bare-words."""
    r = shell()
    out = run(r, "jealousy [story] : $eros(A,B) * $eros(C,A) * C <> A",
              "  -o anger(C,A) * anger(C,B).",
              "move_on [story] : eros(A,B) * anger(A,B)",
              "  -o one.", "}", "stage go {")
    lines = [l for l in out.splitlines() if l]
    assert len(lines) == 6
    assert all("LLP source" in l and "reload" in l for l in lines)
    assert "retract" not in out and "unknown command" not in out
    out = run(r, "frobnicate")
    assert "bare words" in out
    out = run(r, "-anger(hermia,helena)")       # shorthand still works
    assert "only 0 present" in out
    assert "reloaded" in run(r, "reload")


def test_list_shows_file_frame_and_stage_rules():
    """`list`: the file, the frame (stage, fired count, state) and
    the current stage's rules with source lines, * on enabled heads."""
    r = shell()
    out = run(r, "list")
    assert "file" in out and "frame  stage go | 0 fired" in out
    heads = [l for l in out.splitlines() if "jealousy [story]" in l]
    assert len(heads) == 1 and heads[0].lstrip().startswith("*")
    assert "move_on [story]" in out and "%%" in out


def test_edit_runs_the_editor_at_the_rule(monkeypatch, tmp_path):
    """`edit RULE` hands the editor +LINE FILE; outside VS Code a
    terminal editor blocks and a changed file reloads on exit."""
    log = tmp_path / "argv"
    ed = tmp_path / "ed"
    ed.write_text(f'#!/bin/sh\necho "$@" > {log}\n')
    ed.chmod(0o755)
    for v in ("TERM_PROGRAM", "VSCODE_IPC_HOOK_CLI", "EDITOR"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("VISUAL", str(ed))
    r = shell()
    assert "no rule" in run(r, "edit nope")
    run(r, "edit move_on")
    argv = log.read_text().split()
    assert argv[0].startswith("+") and argv[1] == str(r.path.resolve())
    assert r.text.splitlines()[int(argv[0][1:]) - 1].lstrip() \
        .startswith("move_on")


def test_empty_load_is_a_usage_hint_not_a_crash():
    """Bare `load` (meaning `reload`) used to read '.' and kill the
    shell; it now points at reload."""
    r = shell()
    assert "reload" in run(r, "load")


def test_trace_replays_the_case_without_touching_the_session():
    """ctrl+alt+enter's contract: `trace` = metisc --run, fresh from
    #init, while the interactive session keeps its state."""
    r = shell()
    run(r, "assert anger(helena,hermia) 2", "step 1")
    before = (r.stage, dict(r.counts), list(r.fired))
    out = run(r, "trace")
    assert out.splitlines()[0] == "case love_triangle steps=6 seeds=3"
    assert sum(l.startswith("T ") for l in out.splitlines()) == 3
    assert (r.stage, r.counts, r.fired) == before
    run(r, "#steps 0")
    assert "cannot trace" in run(r, "trace")


def test_llm_and_narrate_via_command_backend():
    """The shell's narrate command through metis-loop's endpoint with
    `cat` in the LLM seat: the output is the prompt, so the material
    (this session's Foata beats + the file's %% docs) is verifiable."""
    r = shell()
    out = run(r, "llm")
    assert "default" in out
    out = run(r, "llm sorcery:x")
    assert "unknown LLM backend" in out
    run(r, "llm command:cat", "run 5")
    out = run(r, "narrate")
    assert "beats" in out and "== the rules" in out
    assert "jealousy" in out
    # a seeded replay narration works without session history
    r2 = shell()
    out = run(r2, "llm command:cat", "narrate 2")
    assert "beats" in out
    out = run(r2, "narrate nonsense")
    assert "usage: narrate" in out
