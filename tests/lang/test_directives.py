"""Interpreter directives (00-language-spec §8): '#' lines are a
comment class — compiler-invisible, canonical-key-invariant — and the
embedded case they carry replays on the reference sampler with the
SAME seeds and I/T/F lines as `metisc --run`."""

from pathlib import Path

import pytest

from metis.kernel import program_key
from metis.lang import compile_catalog, parse
from metis.lang.directives import (DirectiveError, embedded_case,
                                   render_run)

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "metis/domains/fiction/catalogs/love_triangle.llp"

DIRECTIVES = """\
#init eros(helena,hermia) 1
# init eros(hermia,helena) 2
#steps 6
#seeds 3
#unknown x
#query anger(hermia,helena)
#lik a=1 b=0
"""


def _strip(text):
    return "\n".join(l for l in text.splitlines()
                     if not l.lstrip().startswith("#"))


def test_directives_are_compiler_invisible():
    text = CATALOG.read_text()
    assert "#init" in text                 # the origin trace, restored
    a, b = parse(text), parse(_strip(text))
    assert a == b
    ka = program_key(compile_catalog(a).catalog.ground(
        embedded_case(text)["init"]))
    kb = program_key(compile_catalog(b).catalog.ground(
        embedded_case(text)["init"]))
    assert ka == kb                        # canonical identity holds


def test_embedded_case_is_the_manifest_vocabulary():
    case = embedded_case(DIRECTIVES)
    assert case["init"] == {"eros(helena,hermia)": 1,
                            "eros(hermia,helena)": 2}
    assert case["steps"] == 6 and case["seeds"] == 3
    assert case["unknowns"] == ["x"]
    assert case["queries"] == ["anger(hermia,helena)"]
    assert case["liks"] == [[("a", 1), ("b", 0)]]
    with pytest.raises(DirectiveError):
        embedded_case("#steps 6 7\n")
    with pytest.raises(DirectiveError):
        embedded_case("#context init\n")


def test_render_run_replays_the_origin_trace():
    text = CATALOG.read_text()
    case = embedded_case(text)
    prog = compile_catalog(parse(text)).catalog.ground(case["init"])
    out = render_run("love_triangle", case, prog)
    lines = out.splitlines()
    assert lines[0] == "case love_triangle steps=6 seeds=3"
    assert lines[1] == "I eros(helena,hermia) 1"
    assert sum(l.startswith("T ") for l in lines) == 3
    assert sum(l.startswith("F ") for l in lines) == 3
    assert "jealousy[" in out and "move_on[" in out
    q = next(l for l in lines if l.startswith("Q anger(hermia,helena)"))
    assert " exact=" in q
    p = float(q.split("exact=")[1])
    assert 0.0 < p < 1.0
    # seeded replay: byte-stable (the metisc --run parity surface)
    assert render_run("love_triangle", case, prog) == out


def test_render_run_pretty_beats_and_state():
    text = CATALOG.read_text()
    case = embedded_case(text)
    prog = compile_catalog(parse(text)).catalog.ground(case["init"])
    out = render_run("love_triangle", case, prog, pretty=True)
    # the wire lines are replaced by the human form ...
    assert "beats" in out and " | " in out
    assert not any(l.startswith("T 1 jealousy") for l in out.splitlines())
    # ... the beat layering is the run_key layering (Foata), and the
    # state is tabulated by predicate
    assert any(l.strip().startswith("anger") for l in out.splitlines())
    # I and Q lines are identical in both modes
    wire = render_run("love_triangle", case, prog)
    for pre in ("I ", "Q "):
        assert [l for l in out.splitlines() if l.startswith(pre)] == \
               [l for l in wire.splitlines() if l.startswith(pre)]


def test_an_interactive_case_never_waits_on_stdin(tmp_path):
    """agent-no-intent s-7fd29000 B: `#interactive go` with no input. The
    py twin of `metisc --run` (render_run, the REPL's `trace`) never reads
    stdin — every choice is the sampler's — and metisc with stdin closed
    falls back to the sampler for the whole run (spec 07 P2): the same
    I/T/F lines. Timed: a run that waits on stdin fails in 60 s, never
    hangs the suite."""
    import os
    import subprocess
    import sys
    from metis.kernel.artifact import metisc
    p = tmp_path / "love_triangle.llp"
    p.write_text(CATALOG.read_text() + "#interactive go\n")
    assert embedded_case(p.read_text())["interactive"] == ["go"]
    code = ("import sys; from pathlib import Path; from metis.lang import compile_catalog, parse; "
            "from metis.lang.directives import embedded_case, render_run; "
            "t = Path(sys.argv[1]).read_text(); c = embedded_case(t); "
            "print(render_run('love_triangle', c, compile_catalog(parse(t)).catalog.ground(c['init'])))")
    # stdin a pipe that is never written nor closed: a read would block
    py = subprocess.Popen([sys.executable, "-c", code, str(p)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, cwd=ROOT,
                          env=dict(os.environ, PYTHONPATH=str(ROOT)))
    try:
        out, _ = py.communicate(timeout=60)
    finally:
        py.kill()
        py.stdin.close()
    assert py.returncode == 0
    wire = [l for l in out.splitlines() if l[:2] in ("I ", "T ", "F ")]
    assert sum(l.startswith("T ") for l in wire) == 3
    mc = metisc()
    if not mc:
        pytest.skip("no metisc ($METISC or PATH): the py half only")
    r = subprocess.run([mc, "--run", str(p)], stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert [l for l in r.stdout.splitlines() if l[:2] in ("I ", "T ", "F ")] == wire
    assert r.stderr.count("choose an event") == 1                     # one menu, then the sampler
