"""The exact filter is one number on every host (the architect's ruling,
runtime-binding step 4): ForwardProjection sums every state's tier in the
normative order (refsample.canonical_events), never in the hash order a
frozenset's counts come in — so a likelihood does not move with
PYTHONHASHSEED (the dog's 0.9999999999999999 was 1.0 under another seed).

Each case runs in subprocesses under different hash seeds and compares the
final distribution's exact floats. Reads metis-catalog (the dog)."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
import sys
sys.path.insert(0, %(tools)r)
from metis.kernel.artifact import resolve, served_doc
from metis.kernel.filter import ForwardProjection
from metis.kernel.record import surprise_record
program, packs, _ = resolve(%(name)r)
doc, case, _ = served_doc(program, packs)
counts = %(counts)r
prog = doc.catalog.ground(dict(counts), init_stage=doc.catalog.stages[0])
pr = ForwardProjection(prog, %(steps)d, keep_history=False)
for (stage, kappa), p in pr.final.items():      # insertion order too
    print(stage, sorted(kappa), repr(p))
print(repr(surprise_record(prog, %(steps)d, %(observed)r)["likelihood"]))
"""

CASES = [
    # the dog at a go command: the case whose likelihood varied
    ("dog", {"posture(standing)": 1, "at(door)": 1, "idle": 1, "battery_ok": 1,
             "floor_dry": 1, "free(left)": 1, "free(right)": 1, "free(around)": 1,
             "clear(forward)": 1, "wall(backward)": 1, "leg_ok": 1, "mapped(door)": 1,
             "open(door)": 1, "mapped(kitchen)": 1, "closed(kitchen)": 1,
             "cmd_go(kitchen)": 1, "cmd_sit": 1, "cmd_find(keys)": 1},
     4, {"idle": 1}),
    ("plant", {"dry": 1, "visitor": 1, "sun": 2, "cmd_check": 2, "moist": 1}, 3,
     {"moist": 1}),
]


def _run(seed, name, counts, steps, observed):
    import os
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    code = PROBE % {"tools": str(ROOT / "tools"), "name": name, "counts": counts,
                    "steps": steps, "observed": observed}
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("name, counts, steps, observed", CASES)
def test_a_projection_is_one_number_under_every_hash_seed(name, counts, steps,
                                                          observed):
    runs = {s: _run(s, name, counts, steps, observed) for s in (1, 2, 3, 7)}
    first = runs[1]
    assert first.strip()
    for s, out in runs.items():
        assert out == first, f"{name}: PYTHONHASHSEED={s} gives another projection"
