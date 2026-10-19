"""The watch on a world that steps in lockstep with its program
(rddl-watch s-cfe94dca): a step computed as a PASS of the stage pipeline
is read once, at the pass's end (a wrap back into the first stage, or
nothing left to fire); the beat's input restarts a pipeline that ended
quiet; every beat is the window's boundary. `follow` adopts a reading the
program explains, so a row-by-row window never grows. The sampled tier is
run_trace's draws, declared on the record, and agrees with the exact one
within its stated error. Off (the defaults), the watch is unchanged."""
import io
import sys

import pytest

from metis.kernel.record import surprise_record
from metis.kernel.refsample import TraceSampler, run_trace
from metis.lang.repl import Repl
from metis.loop.watch import Watch

# one toss per step: the pass tosses, settles the face, and wraps
COIN = """catalog coin 1.
layers (dyn).
stages (toss settle).
namespace c produce (dyn).
pred go : c.
pred heads : c.
pred tails : c.
pred nh : c.
pred nt : c.
stage toss {
  %% The coin is tossed.
  toss [dyn] : go -o (nh @w 0.7 | nt @w 0.3).
}
stage settle {
  %% Heads stays heads.
  hh [dyn] : nh * heads -o heads.
  %% Heads turns tails.
  ht [dyn] : nt * heads -o tails.
  %% Tails turns heads.
  th [dyn] : nh * tails -o heads.
  %% Tails stays tails.
  tt [dyn] : nt * tails -o tails.
}
%% Tossed: settle the face.
qui on [dyn] : toss -o settle.
{wrap}
#init go 1
#init heads 1
"""
WRAP = "%% Settled: the next toss.\nqui again [dyn] : settle -o toss * go."

# the world: a face per step, the program brought to it at each beat (an
# observing RDDL host's tick: sync, then the world steps), `go` handed in
# when the program does not hold it (a pipeline with no link back)
HOST = '''
FACES = ["tails", "tails", "heads", "tails", "heads", "heads", "tails", "heads"]


class Coin:
    def __init__(self, case):
        self.t, self.face = 0, "heads"

    def bind(self, reg):
        pass

    def reading(self):
        return {"heads": int(self.face == "heads"), "tails": int(self.face == "tails")}

    def observe(self, counts):
        return self.reading() if self.t else {}

    def tick(self, counts):
        if self.t >= len(FACES):
            return None
        dos = ["%s%s" % ("+" if n else "-", a) for a, n in self.reading().items() if n != counts.get(a, 0)]
        self.face = FACES[self.t]
        self.t += 1
        return dos + ([] if counts.get("go") else ["+go"])

    def on_event(self, name, args):
        pass

    def snapshot(self):
        return [self.t, self.face]

    def restore(self, s):
        self.t, self.face = s


def make(case):
    return Coin(case)
'''


def _shell(tmp_path, monkeypatch, wrap=True):
    (tmp_path / "coin.llp").write_text(COIN.replace("{wrap}", WRAP if wrap else ""))
    (tmp_path / "coin_host.py").write_text(HOST)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("coin_host", None)
    return Repl(str(tmp_path / "coin.llp"), seed=3, plugin="coin_host:make", host=True, stdout=io.StringIO())


def _run(sh, **kw):
    w = Watch(sh, **kw)
    sigs = []
    for _ in range(400):
        if w.done:
            break
        sigs += [(s, sh.fired[-1] if sh.fired else None, w.beat) for s in w.step()]
    return w, sigs


def test_a_wrapping_pass_is_read_at_its_end_only(tmp_path, monkeypatch):
    sh = _shell(tmp_path, monkeypatch)
    w, sigs = _run(sh, lockstep=True)
    reads = [(s, row) for s, row, _ in sigs if s.get("kind") in ("reading", "rare") or s["signal"] == "surprise"]
    assert reads, "the world differs from some draws"
    # every read follows the wrap's row: never mid-pass
    assert {row.split("[")[0] for _, row in reads} == {"again"}
    assert not any(s["signal"] == "surprise" for s, _, _ in sigs)
    # the window is one pass: 4 rows (toss, on, settle, again), priced over 4
    assert all(s["likelihood"] in (pytest.approx(0.7), pytest.approx(0.3)) for s, _ in reads)
    # one toss per beat: the wrapped pass waits for the beat
    tosses = [b for s, row, b in sigs if s.get("kind") == "clause" and s["name"].startswith("toss")]
    assert tosses == sorted(set(tosses)) and len(tosses) == 9      # the #init pass and one per face


def test_a_quiet_pipeline_restarts_at_the_beat(tmp_path, monkeypatch):
    """No link back: the pass ends quiet in its last stage. Lockstep: the
    beat's `go` starts the next toss at the first stage. Off: the input
    lands in `settle`, where nothing takes it — the second step stalls."""
    def tosses(sigs):
        return sum(1 for s, _, _ in sigs if s.get("kind") == "clause" and s["name"].startswith("toss"))
    w, sigs = _run(_shell(tmp_path, monkeypatch, wrap=False), lockstep=True)
    assert tosses(sigs) == 9
    assert not any(s["signal"] == "surprise" for s, _, _ in sigs)
    w, sigs = _run(_shell(tmp_path, monkeypatch, wrap=False))
    assert (w.lockstep, w.follow, w.tier) == (False, False, None)      # the defaults: off
    assert tosses(sigs) == 1


# a world the program follows row by row: arrivals it draws on its own rows,
# a world where nothing ever arrives (elevators' shape)
QUEUE = """catalog queue 1.
layers (dyn).
stages (run).
namespace q produce (dyn).
pred free : q.
pred waiting : q.
pred bell : q.
stage run {
  %% Someone may arrive.
  arrive [dyn] : free -o (waiting @w 0.4 | free @w 0.6).
  %% The bell rings.
  ring [dyn] : $bell -o () @w 0.2.
}
#init free 1
#init bell 1
"""

QHOST = '''
class Empty:
    beat = 0
    def __init__(self, case):
        pass
    def bind(self, reg):
        pass
    def observe(self, counts):
        return {"free": 1, "waiting": 0}
    def tick(self, counts):
        return None
    def on_event(self, name, args):
        pass
    def snapshot(self):
        return 0
    def restore(self, s):
        pass


def make(case):
    return Empty(case)
'''


def test_follow_adopts_an_explained_reading(tmp_path, monkeypatch):
    (tmp_path / "queue.llp").write_text(QUEUE)
    (tmp_path / "queue_host.py").write_text(QHOST)
    monkeypatch.syspath_prepend(str(tmp_path))
    windows = {}
    for follow in (False, True):
        sys.modules.pop("queue_host", None)
        sh = Repl(str(tmp_path / "queue.llp"), seed=5, plugin="queue_host:make", host=True, stdout=io.StringIO())
        w = Watch(sh, follow=follow)
        widest, sigs = 0, []
        for _ in range(30):
            sigs += w.step()
            widest = max(widest, len(sh.fired) - w._cp[2])
        windows[follow] = (widest, sigs, sh)
    widest, sigs, sh = windows[False]
    assert widest > 5                              # the window grows while the draw lasts
    widest, sigs, sh = windows[True]
    assert widest == 0                             # each read checkpoints: adopted, or consistent
    adopted = [s for s in sigs if s.get("adopted")]
    assert adopted and all(s["adopted"] == {"free": 1, "waiting": -1} for s in adopted)
    assert any(name.startswith("reading do") for name in sh.fired)
    assert not any(s["signal"] == "surprise" for s in sigs)


def test_the_trace_sampler_is_run_traces_draws():
    from metis.domains.fiction import INIT, love_triangle
    prog = love_triangle().catalog.ground(INIT)
    ts = TraceSampler(prog)
    for seed in range(1, 40):
        assert ts.run(6, seed) == run_trace(prog, 6, seed)[1], seed


def test_the_sampled_tier_is_declared_and_agrees_with_the_exact(tmp_path):
    from metis.lang import compile_catalog, parse
    doc = compile_catalog(parse(COIN.replace("{wrap}", WRAP)))
    prog = doc.catalog.ground({"go": 1, "heads": 1})
    for observed, p in (({"heads": 1, "tails": 0}, 0.7), ({"heads": 0, "tails": 1}, 0.3),
                        ({"heads": 0, "tails": 0}, 0.0)):
        exact = surprise_record(prog, 4, observed)
        assert "tier" not in exact and exact["likelihood"] == pytest.approx(p)
        within = surprise_record(prog, 4, observed, sampled={"max_states": 1000, "samples": 400, "seed": 9})
        assert within["tier"] == "exact" and within["likelihood"] == exact["likelihood"]
        rec = surprise_record(prog, 4, observed, sampled={"max_states": 2, "samples": 400, "seed": 9})
        assert (rec["tier"], rec["n"], rec["seed"], rec["max_states"]) == ("sampled", 400, 9, 2)
        assert abs(rec["likelihood"] - p) <= rec["ci"], (observed, rec["likelihood"], rec["ci"])
        again = surprise_record(prog, 4, observed, sampled={"max_states": 2, "samples": 400, "seed": 9})
        assert again["likelihood"] == rec["likelihood"]          # reproducible per seed
    assert exact["kind"] == "unmodeled" and rec["kind"] == "rare"   # a sampled zero is no proof
