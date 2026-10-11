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
        self.outbox: List[dict] = []    # intent results, flushed by the wire
        self.results: List[dict] = []   # every intent result (the ledger)
        self.intents = 0
        self._reset_sigs = {}           # surprise -> (record, beat) of the
                                        # reset that already met it
        self._explained = {}            # surprise -> the (card, move) that
                                        # explained it (adopted, not served)
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
        name = sh._tick(quiet_rows=False)
        if name is None:
            if sh.stage != DONE and not sh._candidates():
                sh._latch_done()
            self.done = True
            return out + [self._signal("quiescence", stage=sh.stage,
                                       fired=len(sh.fired),
                                       episode_over=True,
                                       observe_only=self.observe_only)]
        self.beat += 1
        if name:                        # input arrived: maybe work again
            self.quiet = False
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
        rec["meta"]["diff"] = diff
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

    # -- intents down (step 3): explain / reset / escalate -------------------
    #
    # Each intent is consumed exactly once: the op answers with its id and
    # the outcome returns as ONE `result` signal (admitted or refused, with
    # the reason). The horizon rule: the served catalog is scored at exactly
    # k over its own fired path (the record's verdict — it may not explain
    # by inventing its own moves); a candidate card gets k+1 — the fired
    # path plus ONE named move of the card, in any interleaving — the same
    # allowance for every candidate. An admitted explain ADOPTS the reading
    # (one exo row named for the card's move) and resumes; it does not
    # switch the served world — activate is case 2, phase 2 (G-A/G-G). A
    # recurring card move therefore surprises again, and a second explain
    # of the same surprise is refused with a mined_text naming the card:
    # explained, then escalated — the argument for G-G.

    VERBS = ("explain", "reset", "escalate")

    def intent(self, line: str) -> dict:
        self.intents += 1
        n = self.intents
        verb, _, rest = str(line).strip().partition(" ")
        if verb not in self.VERBS:
            self._result(n, verb or "?", False,
                         f"unknown intent '{verb}' — one of "
                         + ", ".join(self.VERBS))
        else:
            try:
                getattr(self, "_intent_" + verb)(n, rest.strip())
            except Exception as e:      # refused, never a crash
                self._result(n, verb, False, f"{type(e).__name__}: {e}")
        return {"ok": True, "intent": n}

    def _result(self, n, verb, admitted, reason, **payload):
        sig = self._signal(
            "result", intent=n, verb=verb,
            status="admitted" if admitted else "refused", reason=reason,
            **payload)
        self.outbox.append(sig)
        self.results.append(sig)

    def _open_record(self, n, verb, arg):
        try:
            r = int(arg)
        except ValueError:
            self._result(n, verb, False, f"want a record id, got '{arg}'")
            return None
        recs = self.shell.loop.records
        if not 0 <= r < len(recs):
            self._result(n, verb, False, f"no record {r}")
            return None
        if r != self.open:
            res = recs[r].get("resolution")
            self._result(n, verb, False, f"record {r} is not open"
                         + (f" ({res['by']} at beat {res['beat']})"
                            if res else ""), record=r)
            return None
        return r

    @staticmethod
    def _signature(rec):
        """The same surprise: the same domain, the same gap between the
        reading and the model — whatever fired just before it."""
        m = rec["meta"]
        return (m["path"], tuple(sorted(m["diff"].items())))

    def _intent_reset(self, n, arg):
        r = self._open_record(n, "reset", arg)
        if r is None:
            return
        rec = self.shell.loop.records[r]
        sig = self._signature(rec)
        if sig in self._reset_sigs:
            r0, b0 = self._reset_sigs[sig]
            gap = " ".join(f"{a} {d:+d}" for a, d in
                           sorted(rec["meta"]["diff"].items()))
            text = (f"the same misreading ({gap}) recurred after a reset "
                    f"of record {r0} at beat {b0}: not a one-off")
            self._refuse_fact(rec, "reset", text)
            self._result(n, "reset", False,
                         "the same surprise came back after a reset — no "
                         "more resets; escalate", record=r, mined_text=text)
            return
        self._reset_sigs[sig] = (r, self.beat)
        self._close(r, "reset")
        self._result(n, "reset", True,
                     "the reading is set aside as noise; resumed on the "
                     "model — the next reading checks it", record=r)

    def _intent_explain(self, n, arg):
        parts = arg.split()
        if len(parts) not in (2, 3):
            self._result(n, "explain", False,
                         "usage: explain RECORD CARD [MOVE]")
            return
        r = self._open_record(n, "explain", parts[0])
        if r is None:
            return
        from .cards import card_program, cards
        rec = self.shell.loop.records[r]
        card = next((c for c in cards(self.shell)
                     if c["name"] == parts[1]), None)
        if card is None:
            self._result(n, "explain", False,
                         f"no card '{parts[1]}' at this node", record=r)
            return
        move = parts[2] if len(parts) == 3 else None
        since = list(rec["meta"]["since"])
        k = rec["steps"]
        before = self._explained.get(self._signature(rec))
        if before is not None:
            c0, m0 = before
            text = (f"{c0} ({m0}) explained this surprise before and it "
                    f"came back: the served world {rec['meta']['path']} "
                    f"lacks {c0}'s {m0}")
            self._refuse_fact(rec, "explain", text)
            self._result(n, "explain", False,
                         f"the same surprise came back after {c0} "
                         f"explained it — the served world is not switched "
                         f"(activate is phase 2); escalate", record=r,
                         card=c0, move=m0, mined_text=text)
            return
        if card["relation"] == "served":
            self._result(n, "explain", False,
                         f"the served world is scored at exactly k={k} over "
                         "its own fired path: the record's own verdict "
                         "(unmodeled) — it may not invent its own moves",
                         record=r, card=card["name"],
                         horizon={"served": k})
            return
        if move is None or move not in card["moves"]:
            offered = ", ".join(card["moves"]) or "none"
            self._result(n, "explain", False,
                         f"a proposal names one move of {card['name']} "
                         f"beside the fired path ({offered})",
                         record=r, card=card["name"])
            return
        _, prog = card_program(self.shell, card, rec["init"],
                                 rec["meta"]["stage"])
        path = _witness(prog, rec["init"], since, move, rec["observed"])
        if path is None:
            self._result(n, "explain", False,
                         f"{card['name']} with {move} does not reach the "
                         f"reading from the record over its fired path "
                         f"{since} (k+1={k + 1})", record=r,
                         card=card["name"], move=move,
                         horizon={"served": k, "card": k + 1})
            return
        self._explained[self._signature(rec)] = (card["name"], move)
        adopted = self._adopt(by=f"{card['name']}:{move}")
        self._close(r, "explain", card=card["name"], move=move)
        self._result(n, "explain", True,
                     f"{card['name']} explains the reading: {' '.join(path)}"
                     f" — the reading is adopted, the served world stays "
                     f"{self.path} (activate is phase 2), resumed",
                     record=r, card=card["name"], move=move, witness=path,
                     horizon={"served": k, "card": k + 1},
                     adopted=adopted)

    def _intent_escalate(self, n, arg):
        rid, _, text = arg.partition(" ")
        r = self._open_record(n, "escalate", rid)
        if r is None:
            return
        if not text.strip():
            self._result(n, "escalate", False,
                         "escalate needs the description (mined_text)",
                         record=r)
            return
        rec = self.shell.loop.records[r]
        # the revision loop's input: the runtime's facts FIRST (a refusal
        # is authoritative; a model's sentence cannot be trusted to carry
        # it), then the model's description
        facts = rec.get("facts", [])
        rec["mined_text"] = "\n".join(facts + [text.strip()])
        rec["escalated"] = {"beat": self.beat, "facts": len(facts)}
        self._result(n, "escalate", True,
                     "stamped on the record for the revision loop; still "
                     "observe-only", record=r, chars=len(text.strip()))

    @staticmethod
    def _refuse_fact(rec, verb, text):
        """A bound's refusal is a fact about the surprise: kept on the
        record, ahead of whatever an escalate describes."""
        rec.setdefault("refusals", []).append(verb)
        if text not in rec.setdefault("facts", []):
            rec["facts"].append(text)

    def _close(self, r, by, **extra):
        rec = self.shell.loop.records[r]
        rec["resolution"] = {"by": by, "beat": self.beat, **extra}
        self.open = None
        self._said = {}
        self._checkpoint()

    def _adopt(self, by: str = "") -> Dict[str, int]:
        """The session takes the host's reading as its state — as ONE
        exogenous row (an edit the served world's events do not
        explain, first-class in log/why like a tick's input), named for
        the card move that explains it. -> the edit, {} when none."""
        from metis.fw.plugin import exo_event
        sh = self.shell
        observe = getattr(sh.plugin, "observe", None)
        if observe is None:
            return {}
        reading = dict(observe(dict(sh.counts)))
        diff = {a: n - sh.counts.get(a, 0) for a, n in reading.items()
                if n != sh.counts.get(a, 0)}
        if not diff:
            return {}
        lines = [f"{'+' if d > 0 else '-'}{a} {abs(d)}"
                 for a, d in sorted(diff.items())]
        ev, after = exo_event(lines, sh.counts)
        if by:
            ev.name = f"{by} {ev.name}"
        sh.counts = after
        sh.fired_how.append({"how": "exo", "p": None})
        sh.fired.append(ev.name)
        sh.fired_rows.append(("exo", sh.stage, ev))
        sh._env_push()
        self.quiet = False
        return diff

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


def _apply(counts: Dict[str, int], ev) -> Optional[Dict[str, int]]:
    """Fire `ev` on counts, or None if it is not enabled there."""
    need = dict(ev.consume)
    for a, m in ev.persist.items():
        need[a] = max(need.get(a, 0), m)
    if any(counts.get(a, 0) < m for a, m in need.items()):
        return None
    out = dict(counts)
    for a, m in ev.consume.items():
        out[a] -= m
    for a, m in ev.produce.items():
        out[a] = out.get(a, 0) + m
    return {a: m for a, m in out.items() if m > 0}


def _witness(prog, init, since, move, observed):
    """The horizon rule's test: some interleaving of the fired path with
    ONE `move` (k+1 events), each enabled where it stands in the card's
    world, reaches the reading on every observed atom. -> the path, or
    None."""
    events = {}
    for _, _, ev in prog.events():
        events.setdefault(ev.name, ev)
    if move not in events or any(n not in events for n in since):
        return None
    for i in range(len(since) + 1):
        path = since[:i] + [move] + since[i:]
        counts = dict(init)
        for name in path:
            counts = _apply(counts, events[name])
            if counts is None:
                break
        if counts is not None and all(counts.get(a, 0) == m
                                      for a, m in observed.items()):
            return path
    return None


def ledger(shell, watch: Optional[Watch] = None) -> dict:
    """The runtime's own account of a watched run (the step-3 report's
    ledger): every surprise record as logged — kind, the gap, what
    fired before it, its resolution, refusals and mined_text (the env
    blob elided to its hash) — and every intent result in order."""
    recs = []
    for i, r in enumerate(shell.loop.records):
        m = r.get("meta", {})
        recs.append({"record": i, "kind": r.get("kind"),
                     "likelihood": r.get("likelihood"),
                     "path": m.get("path"), "beat": m.get("beat"),
                     "since": m.get("since"), "diff": m.get("diff"),
                     "steps": r.get("steps"),
                     "env": (r.get("env") or {}).get("hash"),
                     "resolution": r.get("resolution"),
                     "refusals": r.get("refusals", []),
                     "escalated": r.get("escalated"),
                     "facts": r.get("facts", []),
                     "mined_text": r.get("mined_text")})
    out = {"records": recs,
           "results": list(watch.results) if watch is not None else []}
    if watch is not None:
        out["watch"] = watch.state()
    return out
