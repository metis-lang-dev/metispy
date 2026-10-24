"""watch-history (s-e061fad2): the REPL's bounded history. A long-running
watch fires forever; Repl(history=N) keeps the last N rows as a Window over
the absolute history — len() and every position stay absolute, the causal
base rolls forward to the window's start — so undo, fork/back, whatif, why,
log and the watch's record keep working inside the window, and a position
before it answers "history kept: last N". The default (None) is unbounded."""
import copy
import io

import pytest

from metis.lang.repl import Repl
from metis.lang.window import Trimmed, Window
from metis.loop.watch import Watch

# a world that never ends: every shift settles, the env brings a job each beat
SHIFTS = """catalog shifts 1.
provenance text "work, then settle; the env brings a job every beat" "2026-10-24".
layers (dyn).
stages (work settle).
namespace sh produce (dyn).
pred job : sh.
pred did : sh.
stage work {
  %% "a job gets done"
  use [dyn] : job -o did.
}
%% "the shift settles"
qui rest [dyn] : work -o settle.
%% "a new shift starts"
qui wake [dyn] : settle -o work.
#plugin shifts_env.py:make
#init job 1
#steps 9
"""

SHIFTS_ENV = """
class Env:
    beat_stage = "settle"
    def __init__(self, case):
        self.beats = 0
    def bind(self, reg):
        pass
    def tick(self, counts):
        self.beats += 1
        return ["+job"]
    def on_event(self, name, args):
        pass
    def snapshot(self):
        return self.beats
    def restore(self, snap):
        self.beats = snap

def make(case):
    return Env(case)
"""


def _world(tmp_path):
    (tmp_path / "shifts_env.py").write_text(SHIFTS_ENV)
    p = tmp_path / "shifts.llp"
    p.write_text(SHIFTS)
    return p


def _repl(path, **kw):
    out = io.StringIO()
    return Repl(str(path), seed=3, stdout=out, terminal=True, **kw), out


def _say(r, out, line):
    out.truncate(0)
    out.seek(0)
    r.onecmd(line)
    return out.getvalue()


def _state(r):
    return r.stage, dict(r.counts), len(r.fired), r.plugin.beats


# -- the Window ---------------------------------------------------------------

def test_a_window_answers_absolute_positions():
    w = Window()
    for i in range(10):
        w.append(i)
    assert w.trim(6) == 6 and w.off == 6 and len(w) == 10 and list(w) == [6, 7, 8, 9]
    assert w[6] == 6 and w[-1] == 9 and w[8:] == [8, 9]
    assert w[2:8] == [6, 7]                       # a read before it: what is kept of that range
    with pytest.raises(Trimmed):
        w[3]
    with pytest.raises(Trimmed):
        del w[4:]                                 # nothing there to cut back to
    del w[8:]
    assert len(w) == 8 and list(w) == [6, 7]
    w[7] = "x"
    assert w[7] == "x" and w.index("x") == 7 and w.count(6) == 1 and 6 in w
    assert w.pop() == "x" and w.pop() == 6
    with pytest.raises(Trimmed):
        w.pop()                                   # empty, but there was a history
    c = copy.deepcopy(w)
    assert c == w and c is not w and c.off == 6
    assert Window([1, 2]) == [1, 2] and bool(Window((), 3)) and not Window()


# -- the REPL -----------------------------------------------------------------

def test_the_default_repl_keeps_everything(tmp_path):
    r, out = _repl(_world(tmp_path))
    _say(r, out, "run 300")
    assert r.history is None and r.fired.off == 0 and len(r.fired.items) == len(r.fired) > 300


def test_a_bounded_history_keeps_the_last_n_and_counts_them_all(tmp_path):
    r, out = _repl(_world(tmp_path), history=40)
    _say(r, out, "run 300")
    n = len(r.fired)
    assert n > 300 and len(r.fired.items) == 40 and r.fired.off == n - 40
    assert len(r.fired_rows) == len(r.fired_how) == n and len(r.fired_rows.items) == 40
    assert len(r._envs) == n + 1 and len(r._envs.items) == 41          # an env per position kept
    # the base rolled to the window's start, and base + the rows since is the state now
    k0, stage, counts = r._base
    assert k0 == r.fired.off
    assert r._rows_applied(k0, stage, counts, n) == (r.stage, r.counts)
    g, since = r.causal()
    assert since == 40 and min(nd.i for nd in g.nodes if nd.i) == k0 + 1     # the graph: the base, then the window


def test_the_bounded_session_is_the_unbounded_one_inside_the_window(tmp_path):
    a, oa = _repl(_world(tmp_path))
    b, ob = _repl(_world(tmp_path), history=40)
    for r, o in ((a, oa), (b, ob)):
        _say(r, o, "run 300")
    n = len(b.fired)
    assert _state(a) == _state(b) and list(a.fired)[-40:] == list(b.fired)
    # fork inside the window: exactly the unbounded session's post-#K, and back
    for r, o in ((a, oa), (b, ob)):
        _say(r, o, f"fork {n - 20}")
    assert _state(a) == _state(b) and len(b.fired) == n - 20
    for r, o in ((a, oa), (b, ob)):
        _say(r, o, "back")
    assert _state(a) == _state(b) and len(b.fired) == n
    # whatif and why inside it: the same answers
    wa, wb = _say(a, oa, f"whatif {n - 15} do(+job) run 3"), _say(b, ob, f"whatif {n - 15} do(+job) run 3")
    assert wa == wb and "history kept" not in wb and "error" not in wb and "usage" not in wb
    assert _state(a) == _state(b)                             # whatif comes back by itself
    # why: the same cone, counted over the graph it has (the window's)
    ya, yb = _say(a, oa, f"why {n - 10}"), _say(b, ob, f"why {n - 10}")
    assert yb.startswith(f"why event {n - 10}: ") and "error" not in yb
    rows = lambda t: {int(l.split("|")[1].split()[0]) for l in t.splitlines()[1:] if "|" in l} - {0}
    assert rows(yb) == {k for k in rows(ya) if k > b.fired.off}   # the unbounded cone, cut at the window
    # undo inside it: the command undone exactly
    before = _state(b)
    _say(b, ob, "run 5")
    _say(b, ob, "undo")
    assert _state(b) == before


def test_before_the_window_the_answer_is_the_history_kept(tmp_path):
    r, out = _repl(_world(tmp_path), history=40)
    _say(r, out, "run 300")
    n = len(r.fired)
    kept = f"history kept: last 40 (#{n - 40}..#{n})"
    assert kept in _say(r, out, "fork 10")
    assert r.shelf is None and len(r.fired) == n             # nothing forked
    assert kept in _say(r, out, "whatif 10 do(+job) run 3")
    log = _say(r, out, "log")
    assert kept in log and f" {n - 39}. " in log and f" {n - 40}. " not in log
    # the undo marks before the window went with it
    while r.undo_stack.items:
        r.undo_stack.pop()
    assert kept in _say(r, out, "undo")


def test_a_bounded_watch_runs_flat(tmp_path):
    """The agent's watch on a bounded REPL: the history stays at N however long
    it beats, and its signals still count every row."""
    r, _ = _repl(_world(tmp_path), history=30)
    w = Watch(r)
    sig = []
    for _ in range(500):
        sig += w.step()
    assert len(r.fired.items) <= 30 and len(r._envs.items) <= 31 and len(r.fired) > 400
    assert len(r.undo_stack.items) <= 30 and len(r.ledger.items) <= 30
    fired = [s["fired"] for s in sig if "fired" in s]
    assert fired == sorted(fired) and fired[-1] == len(r.fired)


def test_the_api_snapshot_tokens_inside_and_before_the_window(tmp_path):
    from metis.repl import Repl as Api
    r, _ = _repl(_world(tmp_path), history=40)
    api = Api(r)
    old = api.snapshot()
    r.onecmd("run 300")
    tok = api.snapshot()
    before = _state(r)
    r.onecmd("run 3")
    api.restore(tok)
    assert _state(r) == before
    with pytest.raises(ValueError, match="history kept: last 40"):
        api.restore(old)
