"""G-D and G-E on the one-process wire (metis-agent docs/INTEGRATION.md
§2, DESIGN.md "Orchestrator contract"): the runtime drives its served
world under the attached host and PUSHES what happens — the harness
never polls.

Three signals up, each with the raising domain's path and a sequence
number (the order is the session's own):

  event       progress or a belief change: a fired intent (kind
              clause/link), an exogenous beat that carried input
              (kind exo — quiet beats say nothing), a sensor reading
              after a surprise when it changes (kind reading:
              consistent again, or a new diff), a
              soft surprise that only logs (kind rare)
  quiescence  nothing left the domain can do (once per entry, not per
              idle beat; `episode_over` when the host's script ends)
  surprise    a reading the served world gives zero probability
              (kernel surprise_record kind `unmodeled`), logged as a
              loop record (its id is the index `record N` selects);
              the session turns OBSERVE-ONLY — no intent fires, the
              env keeps ticking and being read (observe-only by the
              host, phase 1; G-C proper is phase 2)

The surprise test is the kernel's, verbatim: a reading that differs
from the session's own state is scored by surprise_record against the
served world grounded at the last consistent checkpoint, over the
events fired since (one, normally). A reading that matches is
explained by the very path that fired — no record.

The host's sensor face is optional: `observe(counts) -> {atom: n}`
over the atoms it can read. A plugin without it is never surprised.

The mirror (G-E): one line per domain — stage, status, surprise
state, and the catalog's own #query marginals projected from the
current state over the case horizon (#steps). Phase 1 serves one
domain (the served catalog), so the mirror has one line."""

from typing import Dict, List, Optional

from metis.kernel.refsample import DONE


class Watch:
    """The driver of one watched episode over a Repl session. step()
    advances the world by one thing — a fired intent, an entry into
    quiescence, or one env beat — and returns the signals it raised."""

    def __init__(self, shell, beats: Optional[int] = None,
                 rare_ratio: float = 0.05):
        self.shell = shell
        self.path = shell.doc.catalog.name
        self.beats = beats              # env beats before stopping
        self.rare_ratio = rare_ratio
        self.beat = 0
        self.seq = 0
        self.quiet = False              # in quiescence (signalled)
        self.done = False
        self.open: Optional[int] = None  # the open surprise's record id
        self._said: Dict[str, int] = {}  # the last reading diff signalled
        self._checkpoint()
        shell.__dict__["_watch"] = self  # the mirror reads its state

    @property
    def observe_only(self) -> bool:
        return self.open is not None

    # -- the driver ------------------------------------------------------

    def step(self) -> List[dict]:
        sh = self.shell
        if self.done:
            return []
        if not self.observe_only and sh._candidates():
            self.quiet = False
            sh._sample()
            out = [self._row_signal()]
            return out + self._read()
        if not self.observe_only and not self.quiet:
            self.quiet = True
            if sh.plugin is None:
                self.done = True
            return [self._signal("quiescence", stage=sh.stage,
                                 fired=len(sh.fired),
                                 episode_over=sh.plugin is None)]
        if sh.plugin is None or (self.beats is not None
                                 and self.beat >= self.beats):
            self.done = True
            return [self._signal("quiescence", stage=sh.stage,
                                 fired=len(sh.fired), episode_over=True,
                                 observe_only=self.observe_only)]
        out = self._read() if self.observe_only else []
        if sh._tick() is None:
            if sh.stage != DONE and not sh._candidates():
                sh._latch_done()
            self.done = True
            return out + [self._signal("quiescence", stage=sh.stage,
                                       fired=len(sh.fired),
                                       episode_over=True,
                                       observe_only=self.observe_only)]
        self.beat += 1
        ev = sh.fired_rows[-1][2]
        if ev.ops:
            self.quiet = False          # input arrived: maybe work again
            out.append(self._row_signal())
        if not self.observe_only:
            self._checkpoint()
        return out

    # -- the surprise test -------------------------------------------------

    def _checkpoint(self):
        sh = self.shell
        self._cp = (sh.stage, dict(sh.counts), len(sh.fired))

    def _read(self) -> List[dict]:
        sh = self.shell
        observe = getattr(sh.plugin, "observe", None)
        if observe is None:
            self._checkpoint()
            return []
        reading = {a: int(n) for a, n in dict(observe(dict(sh.counts)))
                   .items()}
        model = {a: sh.counts.get(a, 0) for a in reading}
        diff = {a: reading[a] - model[a] for a in reading
                if reading[a] != model[a]}
        if self.observe_only:
            # keep sensing; speak only when the reading changes (a
            # transient clears, a wrong world drifts further)
            if diff == self._said:
                return []
            self._said = diff
            return [self._signal("event", kind="reading",
                                 consistent=not diff, diff=diff,
                                 record=self.open)]
        if not diff:
            self._checkpoint()
            return []
        rec = self._record(reading)
        if rec["kind"] == "unmodeled":
            sh.loop.records.append(rec)
            self.open = len(sh.loop.records) - 1
            self._said = diff
            return [self._signal("surprise", record=self.open,
                                 kind=rec["kind"],
                                 likelihood=rec["likelihood"],
                                 steps=rec["steps"], observed=reading,
                                 model=model, diff=diff,
                                 observe_only=True)]
        if rec["kind"] == "rare":
            sh.loop.records.append(rec)
            return [self._signal("event", kind="rare",
                                 record=len(sh.loop.records) - 1,
                                 likelihood=rec["likelihood"], diff=diff)]
        # another path of the served world explains the reading: a
        # belief change, not a surprise
        return [self._signal("event", kind="reading", consistent=False,
                             likelihood=rec["likelihood"], diff=diff)]

    def _record(self, reading: Dict[str, int]) -> dict:
        from metis.kernel.record import surprise_record
        sh = self.shell
        stage, counts, n0 = self._cp
        prog = sh.doc.catalog.ground(dict(counts), init_stage=stage)
        steps = max(1, len(sh.fired) - n0)
        policy = (sh.pricer.policy(prog) if sh.pricer is not None
                  else None)
        snap = sh.plugin.snapshot() if sh.plugin is not None else None
        return surprise_record(
            prog, steps, reading, policy=policy,
            rare_ratio=self.rare_ratio,
            meta={"path": self.path, "beat": self.beat,
                  "fired": len(sh.fired), "stage": stage,
                  "since": sh.fired[n0:],
                  "host": sh.plugin_spec or None},
            env=snap)

    # -- signal lines --------------------------------------------------------

    def _signal(self, sig: str, **payload) -> dict:
        self.seq += 1
        return {"signal": sig, "seq": self.seq, "path": self.path,
                "beat": self.beat, **payload}

    def _row_signal(self) -> dict:
        sh = self.shell
        kind, stage, ev = sh.fired_rows[-1]
        out = {"kind": kind, "name": ev.name, "fired": len(sh.fired),
               "stage": sh.stage}
        if kind == "exo":
            out["ops"] = list(ev.ops)
        else:
            out["consume"] = dict(ev.consume)
            out["produce"] = dict(ev.produce)
        return self._signal("event", **out)

    def state(self) -> dict:
        return {"seq": self.seq, "beat": self.beat, "done": self.done,
                "observe_only": self.observe_only, "surprise": self.open}


# -- the mirror (G-E) ----------------------------------------------------

def mirror(shell, watch: Optional[Watch] = None) -> List[dict]:
    """One row per domain: path, stage, status, surprise state and the
    #query marginals P[atom > 0] at the case horizon from NOW."""
    from metis.kernel import (STATE_BUDGET, ForwardProjection,
                              StateBudgetExceeded)
    queries = list(shell.case.get("queries") or [])
    steps = shell.case.get("steps") or 6
    cands = shell._candidates()
    kinds = {shell.events_tbl[i][0] for i, _ in cands}
    if watch is not None and watch.observe_only:
        status = "observe-only"
    elif shell.stage == DONE:
        status = "done"
    elif not cands or kinds == {"link"}:
        status = "quiescent"
    else:
        status = "running"
    if watch is not None and watch.open is not None:
        rec = shell.loop.records[watch.open]
        surprise = {"record": watch.open, "kind": rec["kind"],
                    "beat": rec["meta"].get("beat")}
    else:
        surprise = None
    beliefs: Dict[str, Optional[float]] = {}
    if queries:
        stage = shell.stage if shell.stage != DONE else \
            (shell.__dict__.get("_quiet_stage") or shell.prog.init_stage)
        try:
            prog = shell.doc.catalog.ground(dict(shell.counts),
                                            init_stage=stage)
            pr = ForwardProjection(
                prog, steps, max_states=STATE_BUDGET, keep_history=False,
                policy=(shell.pricer.policy(prog)
                        if shell.pricer is not None else None))
            for q in queries:
                beliefs[q] = round(sum(
                    p for (_, kappa), p in pr.final.items()
                    if dict(kappa).get(q, 0) > 0), 4)
        except StateBudgetExceeded:
            beliefs = {q: None for q in queries}
    return [{"path": shell.doc.catalog.name, "stage": shell.stage,
             "status": status, "surprise": surprise, "horizon": steps,
             "beliefs": beliefs}]


def mirror_line(row: dict) -> str:
    s = row["surprise"]
    sur = (f"surprise #{s['record']} {s['kind']} @beat {s['beat']}"
           if s else "surprise none")
    bel = "  ".join(f"P[{q}]={'?' if p is None else f'{p:.2f}'}"
                    for q, p in row["beliefs"].items())
    return (f"{row['path']}  stage {row['stage']}  {row['status']}  "
            f"{sur}" + (f"  {bel} @{row['horizon']}" if bel else ""))
