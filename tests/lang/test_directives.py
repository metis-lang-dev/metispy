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
