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
CATALOG = ROOT / "metis/domains/fiction/catalogs/love_triangle.llp"


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
