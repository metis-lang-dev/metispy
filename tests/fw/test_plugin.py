"""Port plugins in the REPL (metis-loop docs/PLUGIN-PORTS.md, subject
port-plugins): '#plugin' attachment (terminal only — D4), fail-fast on
unbound ports (serve, reload, gate), exogenous tick input as [env]
rows that revive a quiet world (G6), env snapshots beside the world in
fork/back/undo/reset/whatif (counterfactual honesty), the sampling
tier's G5 scope teeth, the session notch as an overlay, and the env
field of surprise records."""
import io

import pytest

from metis.fw.plugin import PluginError, exo_event, load_plugin
from metis.fw.registry import PortError
from metis.kernel.factorize import ScopeViolation
from metis.kernel.record import recover, surprise_record
from metis.lang import PortPricer, load_llp
from metis.lang.formalize import admit_text_pack
from metis.lang.repl import Repl
from metis.loop.wire import INTENT_VERBS

SENSOR = """catalog sensor 1.
provenance text "a hazard priced by a port reading the gap class" "2026-10-09".
layers (dyn).
stages (tick).
namespace sx produce (dyn).
pred ck : sx.
pred gclose : sx.
pred crash : sx.
pred ok : sx.
weight rss reads {READS}.
stage tick {
  %% "the hazard fires with the rss-priced probability"
  hit [dyn] : ck * $gclose -o crash  @w rss.
  %% "the tick passes clean"
  pass [dyn] : ck -o ok  @w ~rss.
}
#plugin sensor_env.py:make
#init ck 1
#init gclose 1
#steps 2
"""

SENSOR_ENV = """
class Env:
    GAPS = (18.0, 6.0)
    def __init__(self, case):
        self.beat = 0
    def rss(self, args, counts):
        close = counts.get("gclose", 0) > 0
        gap = self.GAPS[min(self.beat, len(self.GAPS) - 1)]
        return 0.9 if (close and gap < 10) else (0.3 if close else 0.01)
    def bind(self, reg):
        reg.weight("rss", self.rss)
    def tick(self, counts):
        if self.beat >= len(self.GAPS) - 1:
            return None
        self.beat += 1
        return []
    def on_event(self, name, args):
        pass
    def snapshot(self):
        return self.beat
    def restore(self, snap):
        self.beat = snap

def make(case):
    return Env(case)
"""

CLOCK = """catalog clock 1.
provenance text "a shop on the env's clock" "2026-10-09".
layers (dyn).
stages (trade).
namespace sh produce (dyn).
pred customer : sh.
pred dooropen : sh.
pred doorshut : sh.
pred sale : sh.
pred missed : sh.
stage trade {
  %% "a customer buys while the door is open"
  serve [dyn] : customer * $dooropen -o sale.
  %% "a shut door loses the customer"
  lose [dyn] : customer * $doorshut -o missed.
}
#plugin clock_env.py:make
#init doorshut 1
#steps 20
"""

CLOCK_ENV = """
class Clock:
    ARRIVALS = {10: 1, 11: 2, 14: 1}
    def __init__(self, case):
        self.hour = 8
        self.acted = []
    def bind(self, reg):
        pass
    def tick(self, counts):
        if self.hour >= 16:
            return None
        h, dos = self.hour, []
        if h == 9:
            dos += ["-doorshut", "+dooropen"]
        if h == 13:
            dos += ["-dooropen", "+doorshut"]
        if self.ARRIVALS.get(h):
            dos.append(f"+customer {self.ARRIVALS[h]}")
        self.hour += 1
        return dos
    def on_event(self, name, args):
        self.acted = self.acted + [name]
    def snapshot(self):
        return (self.hour, tuple(self.acted))
    def restore(self, snap):
        self.hour, self.acted = snap[0], list(snap[1])

def make(case):
    return Clock(case)
"""


def _world(tmp_path, llp, env_name, env_src, reads="gclose"):
    (tmp_path / env_name).write_text(env_src)
    p = tmp_path / "world.llp"
    p.write_text(llp.replace("READS", reads))
    return p


def _repl(path, **kw):
    out = io.StringIO()
    r = Repl(path, seed=3, stdout=out, terminal=True, **kw)
    return r, out


def _say(r, out, line):
    out.truncate(0)
    out.seek(0)
    r.onecmd(line)
    return out.getvalue()


def _p(text):
    """The P of a `query` answer."""
    return float(text.split("] = ")[1].split()[0])


# -- the clock world: time face, G6 revival, the ledger sees it ----------

def test_clock_runs_natively_and_the_env_revives_a_quiet_world(tmp_path):
    r, out = _repl(_world(tmp_path, CLOCK, "clock_env.py", CLOCK_ENV))
    assert r.plugin is not None
    # nothing can fire at the start: quiet, not dead (G6) — `run` ticks
    assert not r._candidates()
    _say(r, out, "run 100")
    assert r.counts.get("sale") == 3 and r.counts.get("missed") == 1
    log = _say(r, out, "log")
    assert "do +customer 2  [env]" in log
    assert "env rows — exogenous input, not chance" in log
    why = _say(r, out, "why missed")
    assert "do +customer" in why and "lose" in why
    # env rows are conditioning, never chance: out of the likelihood
    lik, ns, nf = r.likelihood()
    assert ns == 4 and nf == 0 and lik == 1.0
    # on_event saw every fired event, in order
    assert r.plugin.acted == ["serve", "serve", "serve", "lose"]


def test_step_at_quiescence_ticks_instead_of_latching_done(tmp_path):
    r, out = _repl(_world(tmp_path, CLOCK, "clock_env.py", CLOCK_ENV))
    said = _say(r, out, "step")          # hour 8: a quiet beat
    assert "the env ticked: tick" in said and r.stage == "trade"
    _say(r, out, "step")                 # hour 9: the door opens
    assert r.counts.get("dooropen") == 1
    _say(r, out, "undo")                 # undo restores the env too
    assert r.plugin.hour == 9 and "dooropen" not in r.counts


def test_exo_event_replays_its_own_edit_exactly():
    ev, after = exo_event(["-doorshut", "+dooropen", "+customer 2",
                           "-ghost"], {"doorshut": 1})
    assert after == {"dooropen": 1, "customer": 2}
    assert ev.consume == {"doorshut": 1}
    assert ev.produce == {"dooropen": 1, "customer": 2}
    assert ev.name == "do -doorshut +dooropen +customer 2 -ghost"


# -- counterfactual honesty: the env rides with the world ----------------

def test_fork_whatif_back_and_reset_restore_the_env(tmp_path):
    r, out = _repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV))
    assert _p(_say(r, out, "query crash")) == pytest.approx(0.3)
    _say(r, out, "tick")                 # the env closes the gap: 6 m
    assert r.plugin.beat == 1
    assert _p(_say(r, out, "query crash")) == pytest.approx(0.9)
    # whatif at #0: BOTH arms price under the env as it was at #0
    w = _say(r, out, "whatif 0 do(+ck) query crash @2")
    assert "P = 0.51" in w and "P = 0.3" in w          # 1 - 0.7^2
    assert r.plugin.beat == 1                          # live env back
    # a user fork at #0 sees the forked env; back restores the actual
    _say(r, out, "fork 0")
    assert r.plugin.beat == 0
    assert _p(_say(r, out, "query crash")) == pytest.approx(0.3)
    _say(r, out, "back")
    assert r.plugin.beat == 1
    assert _p(_say(r, out, "query crash")) == pytest.approx(0.9)
    # undo of the tick restores the env; reset returns to as-attached
    _say(r, out, "undo")
    assert r.plugin.beat == 0
    _say(r, out, "tick")
    _say(r, out, "reset")
    assert r.plugin.beat == 0


def test_sampling_follows_the_ports(tmp_path):
    r, out = _repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV))
    ws = dict((r.events_tbl[i][2].name, w) for i, w in r._candidates())
    assert ws == {"hit": pytest.approx(0.3), "pass": pytest.approx(0.7)}
    assert "rss() = 0.3" in _say(r, out, "port rss")
    assert "bound" in _say(r, out, "ports")


# -- G5: the sampling tier enforces `reads` like the factored tier -------

def test_out_of_scope_read_raises_in_the_sampling_tier(tmp_path):
    p = _world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV, reads="")
    r, out = _repl(p)
    said = _say(r, out, "step")
    assert "ScopeViolation" in said and "G5" in said
    assert any(d.code == "reads-undeclared" for d in r.notes)
    with pytest.raises(ScopeViolation):
        r._candidates()
    doc = load_llp(p)
    pricer = PortPricer(doc.ports, {"rss": lambda a, c: c.get("gclose")},
                        horn=doc.catalog.horn_db())
    with pytest.raises(ScopeViolation):
        pricer.factor("hit", {"ck": 1, "gclose": 1})


# -- fail fast: unbound ports refuse to serve ----------------------------

def test_unbound_port_refuses_to_serve(tmp_path):
    p = _world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV)
    p.write_text(p.read_text().replace("#plugin sensor_env.py:make\n", ""))
    with pytest.raises(PortError) as e:
        _repl(p)
    assert "unbound weight port 'rss'" in e.value.findings[0]


def test_reload_with_an_unbound_port_keeps_the_old_catalog(tmp_path):
    p = _world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV)
    r, out = _repl(p)
    key = r.key
    p.write_text(p.read_text().replace(
        "weight rss reads {gclose}.",
        "weight rss reads {gclose}.\nweight other reads {}.").replace(
        "@w ~rss", "@w other"))
    said = _say(r, out, "reload")
    assert "reload failed" in said and "unbound weight port 'other'" in said
    assert r.key == key


def test_plugin_off_is_refused_while_ports_need_it(tmp_path):
    r, out = _repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV))
    said = _say(r, out, "plugin off")
    assert "plugin not changed" in said and r.plugin is not None
    with pytest.raises(PluginError):
        load_plugin("nope_mod:make", {}, tmp_path)


# -- D4: plugins are terminal-only ----------------------------------------

def test_web_console_never_imports_the_plugin(tmp_path):
    p = _world(tmp_path, CLOCK, "clock_env.py", CLOCK_ENV)
    out = io.StringIO()
    r = Repl(p, seed=1, stdout=out)          # not the terminal shell
    assert r.plugin is None
    assert any("terminal-only" in w for w in r._port_warns)
    assert "terminal-only" in _say(r, out, "plugin clock_env.py:make")
    # a catalog whose ports need the plugin cannot be served there
    with pytest.raises(PortError):
        Repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV),
             stdout=io.StringIO())
    assert not {"plugin", "ports", "port", "tick", "notch"} & INTENT_VERBS


def test_a_hosted_run_posts_its_env_rows_on_the_wire(tmp_path):
    """A host attached as the operator's --host (the loop console's
    one-click open, metis-loop catalog_index): the wire's posting of a run
    lists the env rows (kind exo, no stage link) — it read ev.pre off them
    and failed every hosted run ("'ExoEvent' object has no attribute
    'pre'")."""
    from metis.loop.wire import intent_posting
    p = _world(tmp_path, CLOCK, "clock_env.py", CLOCK_ENV)
    r = Repl(p, seed=1, stdout=io.StringIO(), plugin="clock_env.py:make", host=True)
    assert r.plugin is not None
    post = intent_posting(r, ":run 100")
    assert post["status"] == "done", post
    rows = post["json"]["fired"]
    assert any(f["kind"] == "exo" and f["stage"] == r.fired_rows[f["i"] - 1][1] for f in rows)
    assert {f["kind"] for f in rows} >= {"exo", "clause"}


# -- gate: an unbound port in a candidate is a finding -------------------

def test_gate_unbound_port_is_a_finding_not_a_crash(tmp_path):
    src = (tmp_path / "x.llp")
    src.write_text(SENSOR.replace("READS", "gclose"))
    cand = src.read_text().split("#plugin")[0]
    rep = admit_text_pack("a hazard", cand, scenarios=({"ck": 1,
                                                        "gclose": 1},),
                          steps=2, registry={})
    assert not rep.ok
    assert any("unbound weight port 'rss'" in f for f in rep.findings)
    # bound: the scenario is PRICED by the session's bindings
    rep = admit_text_pack("a hazard", cand, scenarios=({"ck": 1,
                                                        "gclose": 1},),
                          steps=2, registry={"rss": lambda a, c: 0.25})
    assert rep.ok
    assert rep.scenarios[0].map_events == ["pass"]


# -- notch: fitted from the session, shipped as an overlay ---------------

def test_notch_fits_rows_and_ships_an_overlay(tmp_path):
    r, out = _repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV))
    for _ in range(30):                     # 30 exposures at beat 0
        _say(r, out, "step")
        _say(r, out, "reset")
    sheet = _say(r, out, "notch")
    assert "rss | gclose=1 : 0.300 ->" in sheet and "n=30" in sheet
    key = r.key
    _say(r, out, "notch accept")
    post = r._overlays and r.registry.weight_bindings()["rss"](
        (), {"gclose": 1})
    s = sum(1 for h in r.fired_how if h.get("site") and
            not h["site"][2])
    assert post == pytest.approx((10 * 0.3 + s) / (10 + 30))
    assert r.key == key                     # never a catalog edit
    _say(r, out, "notch pop")
    assert r.registry.weight_bindings()["rss"]((), {"gclose": 1}) == 0.3


# -- records carry the env; recover replays against it ------------------

def test_surprise_record_env_replays_honestly(tmp_path):
    r, out = _repl(_world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV))
    prog = r.doc.catalog.ground({"ck": 1, "gclose": 1})
    pol = r.pricer.policy(prog)
    rec = surprise_record(prog, 2, {"crash": 1}, policy=pol,
                          env=r.plugin.snapshot())
    assert rec["likelihood"] == pytest.approx(0.3) and "blob" in rec["env"]
    r.onecmd("tick")                        # the live env moves on
    assert recover(rec, prog, policy=pol)["likelihood"] == \
        pytest.approx(0.9)                  # dishonest without the env
    out_ = recover(rec, prog, policy=pol, plugin=r.plugin)
    assert out_["env"] == "restored"
    assert out_["likelihood"] == pytest.approx(0.3)
    assert r.plugin.beat == 1               # live env put back
    del rec["env"]["blob"]
    assert recover(rec, prog, policy=pol,
                   plugin=r.plugin)["env"] == "MISMATCH"


# -- tick-stage catalogs: one env beat per settle entry ------------------

BEAT = """catalog beat 1.
provenance text "work, then settle; the env brings jobs" "2026-10-10".
layers (dyn).
stages (work settle).
namespace bt produce (dyn).
pred job : bt.
pred did : bt.
stage work {
  %% "a job gets done"
  use [dyn] : job -o did.
}
%% "the shift settles"
qui rest [dyn] : work -o settle.
%% "a new shift starts"
qui wake [dyn] : settle -o work.
#plugin beat_env.py:make
#init job 1
#steps 9
"""

BEAT_ENV = """
class Env:
    beat_stage = "settle"
    def __init__(self, case):
        self.beats = 0
    def bind(self, reg):
        pass
    def tick(self, counts):
        self.beats += 1
        return ["+job"] if self.beats <= 2 else []
    def on_event(self, name, args):
        pass
    def snapshot(self):
        return self.beats
    def restore(self, snap):
        self.beats = snap

def make(case):
    return Env(case)
"""


def test_beat_stage_ticks_on_settle_entry(tmp_path):
    (tmp_path / "beat_env.py").write_text(BEAT_ENV)
    p = tmp_path / "world.llp"
    p.write_text(BEAT)
    r, out = _repl(p)
    _say(r, out, "run 12")
    # use, rest, (tick +job), wake, use, rest, (tick +job), wake, use,
    # rest, (tick: quiet beat) — every settle entry is one env beat
    assert r.fired[:4] == ["use", "rest", "do +job", "wake"]
    assert r.counts.get("did") == 3
    assert r.plugin.beats == r.fired.count("rest")


# -- the --plugin flag: catalog-served files attach at session level -----

def test_main_plugin_flag_serves_a_port_catalog(tmp_path, monkeypatch,
                                                capsys):
    from metis.lang.repl import main
    p = _world(tmp_path, SENSOR, "sensor_env.py", SENSOR_ENV)
    p.write_text(p.read_text().replace("#plugin sensor_env.py:make\n", ""))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # without the flag: refused, and the message teaches both routes
    assert main([str(p)]) == 1
    err = capsys.readouterr().err
    assert "metis --plugin pkg.mod:factory FILE" in err
    assert "'#plugin pkg.mod:factory' in a scene file" in err
    # with it: served, the banner and the transcript record it
    t = tmp_path / "t.md"
    monkeypatch.setattr("sys.stdin", io.StringIO(
        f"transcript {t}\nquery crash\ntranscript off\nquit\n"))
    assert main(["--plugin", "sensor_env.py:make", str(p)]) == 0
    out = capsys.readouterr().out
    assert "plugin sensor_env.py:make (Env) attached" in out
    assert "] = 0.3" in out
    assert "- plugin: sensor_env.py:make (Env)" in t.read_text()
    # the --json wire never imports operator code (D4)
    assert main(["--json", "--plugin", "sensor_env.py:make", str(p)]) == 2


def test_policy_index_space_matches_its_consumer(tmp_path):
    """The filter indexes program.events(); run_trace indexes
    canonical_events(program) — a pricer policy handed to the wrong one
    prices the wrong events. Both declaration orders, rss = 0: `hit`
    must never fire in either consumer."""
    from metis.kernel import ForwardProjection
    from metis.kernel.refsample import run_trace
    for order in ("hit-first", "pass-first"):
        src = SENSOR.replace("READS", "gclose").replace(
            "#plugin sensor_env.py:make\n", "")
        if order == "pass-first":
            a = src.index('  %% "the hazard fires')
            b = src.index('  %% "the tick passes clean"')
            c = src.index("}", b)
            src = src[:a] + src[b:c] + src[a:b] + src[c:]
        p = tmp_path / f"{order}.llp"
        p.write_text(src)
        doc = load_llp(p)
        pricer = PortPricer(doc.ports, {"rss": lambda a, c: 0.0},
                            horn=doc.catalog.horn_db())
        prog = doc.catalog.ground({"ck": 1, "gclose": 1})
        fins = ForwardProjection(prog, 2, policy=pricer.policy(prog)).final
        assert all("crash" not in str(s) for s in fins), order
        pol = pricer.policy(prog, canonical=True)
        for seed in range(50):
            _, (_, kappa) = run_trace(prog, 2, seed, policy=pol)
            assert "crash" not in dict(kappa), (order, seed)


def test_accept_with_unbound_ports_is_refused_atomically(tmp_path):
    """A revision whose catalog declares ports nobody binds: accept is
    refused with the fix in the message, and the loop state is exactly
    as before (the old world serves, the candidate still pending)."""
    p = tmp_path / "blank.llp"
    p.write_text("catalog scratch 1.\nlayers (dyn).\nstages (z).\n"
                 "stage z {\n}\n")
    r = Repl(p, stdout=io.StringIO())
    cand = SENSOR.replace("READS", "gclose").split("#plugin")[0]
    r.cmd_text("a hazard")
    r.cmd_llp(cand)
    from metis.fw.registry import Registry
    r.registry = Registry().weight("rss", 0.3)   # bound while gating
    assert r.cmd_revise("")["ok"]
    r.registry = None                           # ... gone at accept
    key, base_src = r.key, r.loop.base_src
    res = r.cmd_accept("")
    assert not res["ok"] and "unbound weight port 'rss'" in res["error"]
    assert "--plugin" in res["hint"]
    assert r.key == key and r.loop.base_src == base_src
    assert r.loop.revision and r.loop.cand == cand
    # bound in-process: the same accept now goes through
    r.registry = Registry().weight("rss", 0.3)
    assert r.cmd_accept("")["ok"] and r.doc.catalog.name == "sensor"
