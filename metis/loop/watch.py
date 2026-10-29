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

Copy semantics (metis-lang spec/17 amendment 4, the user's ruling of
2026-10-29): a member reads the world as a snapshot taken at the start of
its own tick, and its own command shows in the world from its next tick.
So a beat begins with its tick (`begin`, the beat runners call it), and
the surprise test runs AT THE TICK, never after a firing: the reading
against the effects the member modelled for its own commands since its
last consistent reading — the atoms its fired events consumed or produced
— over that WINDOW (its diff, since and steps are the beat's). Another
member's change is the tick's input, not a surprise. A reading is never
older than a command it is compared for: in one process the world applies
a command at once, so the next tick's snapshot has it; across a split the
proxy adds the pending ones (a pending effect older than the L18 budget is
what tick__stuck names). Whether anything is left to do is known at a
tick too, so the quiescence is declared at the tick whose reading leaves
the member nothing to fire (once per entry, as before), not when it stops
mid-beat.

The surprise taxonomy (unmodeled, rare, a consistent or explained
reading; the kitchen fixtures' transient, wrong_domain, novel) is
unchanged in name; each is now stated over the window. A member that
chains its own modelled effects within a beat (the chained kitchen scene:
a whole soup in one beat) observes a partner's change only through its
consequence on its own commands, so for such a member the window shows a
belief change (fewer of its loads took), never a sibling's card; a member
that wants partner changes explained reads the shared state as a
world-owned reading at its tick (the tick form, catalogs/tick/: one action
per tick, the window one action). A member whose host steps the world on
an event (an RDDL observing world's move: the host's `world_stepped()`)
reads that world only at its tick, so each such event takes a tick: the
watch's `tick_due` ends the beat there. A program that chains such events
on its own modelled state would run ahead of a world it cannot see (the
old live read after a firing hid it: nav's planted-death test found its
10 of 10 that way). Evidence: toolchain-units s-ddaed776
step 3's 20-variant probe (a partner load at the tick, orders of 3-6
onions, five injection beats: never explained by kitchen_shared on the
chained scene).

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
domain (the served catalog), so the mirror has one line.

Lockstep (a domain's manifest `clock: {lockstep: true}`, rddl-watch
s-cfe94dca; off, the watch above is unchanged): the program computes one
step of its world as a PASS of its stage pipeline, many rows long (an
RDDL world's game-of-life or wildfire step). The watch then reads the
world once per step, when the pass is done, never mid-pass:

  done        the pass wrapped (a link back into the program's first
              stage, from another) or nothing is left to fire
  the beat    a wrapped pass holds (`held`) until the env's beat: its
              next pass starts after the host's tick, as a quiet one does
  restart     the beat's input starts the next pass at the first stage,
              wherever the last one stopped (a pipeline with no link back
              would otherwise sit in its last stage)
  checkpoint  at every beat, the step's boundary: the window priced is
              one pass, from the beat's input; a difference the program
              explained is not priced again (the host's sync brings the
              program to the world at the beat)

`tier` prices the window (record.surprise_record's `sampled`): exact
under a state budget, the declared sampled tier past it.

Follow (`clock: {follow: true}`; row by row, off: unchanged): a reading
that differs from the program's state but that the program explains (a
belief change, or a rare one) is ADOPTED at once — one exo row named
`reading`, the session's state is the world's — and the checkpoint moves
there; a host that reads nothing (its world has not moved since its last
step: an RDDL host's `read_steps`) leaves the window open, so it runs from
one world step to the next. Without it the checkpoint stays put while the
difference lasts,
and each row prices a window one row longer (elevators: 0.1 s to 22 s a
row by its 12th gym step, rddl-worlds finding (e))."""

from typing import Dict, List, Optional

from metis.kernel.refsample import DONE


class Watch:
    """The driver of one watched episode over a Repl session. step()
    advances the world by one thing — a fired intent, an entry into
    quiescence, or one env beat — and returns the signals it raised."""

    def __init__(self, shell, beats: Optional[int] = None,
                 rare_ratio: float = 0.05, blind: bool = False,
                 operator: Optional[int] = None, scope: str = "order",
                 explain_repeat_notice: int = 2, lockstep: bool = False,
                 tier: Optional[dict] = None, follow: bool = False):
        self.shell = shell
        self.path = shell.doc.catalog.name
        self.beats = beats              # env beats before stopping
        self.lockstep = lockstep        # one read per pass (module doc)
        self.held = False               # lockstep: a wrapped pass waits for the beat
        self.tier = tier                # the pricing tier (None: exact, unbudgeted)
        self.follow = follow            # an explained reading is adopted (module doc)
        self.rare_ratio = rare_ratio
        self.blind = blind              # arm A: no check, intents apply
        if scope not in ("order", "session"):
            raise ValueError(f"same_surprise_scope {scope!r}: order or "
                             "session")
        # same_surprise_scope: how long the bounds remember a surprise a
        # reset or explain resolved — "order" (the task open at the
        # resolution: cleared when the domain next reaches quiescence,
        # i.e. that work is done — the kitchen's order served — and at
        # episode_over) or "session" (for the whole watch)
        self.scope = scope
        # the silent-wrong-context smell: the SAME (card, move) admitted
        # in N consecutive orders still resumes, but the record gets a
        # notice the revision loop (and phase 2's activate) consumes
        self.notice_after = explain_repeat_notice
        self.tasks = 0                  # task boundaries so far
        self._streak = {}               # (card, move) -> (task, run)
        self.operator = operator        # beats until an escalation is
                                        # resumed by the operator
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
        return self.open is not None and not self.blind

    # -- the driver ------------------------------------------------------

    def step(self) -> List[dict]:
        sh = self.shell
        if self.done:
            return []
        if self.__dict__.get("_tick_due") and not self.observe_only:
            return self._tick_step()    # the world stepped on the last event: it is read before the next
        if not self.observe_only and not self.held and sh._candidates():
            self.quiet = False
            stage0 = sh.stage
            sh._sample()
            out = [self._row_signal()]
            if not self.lockstep:
                if self.follow:
                    return out + self._read()   # follow reads row by row (module doc)
                # copy semantics: a fired rule's world reads at the next tick. A host that steps its world on the
                # event (an RDDL observing world's move) makes that event take a tick: the next one waits for it
                inner = getattr(sh.plugin, "inner", sh.plugin)
                if callable(getattr(inner, "world_stepped", None)) and inner.world_stepped():
                    self._tick_due = True
                return out
            first = sh.prog.init_stage
            if sh.stage == first and stage0 != first:
                self.held = True        # the pass wrapped: read, then the beat
                return out + self._read()
            if not sh._candidates():
                return out + self._read()   # the pass ended quiet
            return out                  # mid-pass: nothing to compare yet
        if not self.observe_only and not self.quiet and not self.held:
            self.quiet = True
            if sh.plugin is not None and not (self.lockstep or self.follow):
                # copy semantics: whether anything is left to do is known at the next tick, from its reading (a
                # turn, an order); the quiescence is declared there, or not at all (_tick_rest)
                self._quiet_due = True
                return []
            self._task_done()
            if sh.plugin is None:
                self.done = True
            return [self._signal("quiescence", stage=sh.stage,
                                 fired=len(sh.fired),
                                 episode_over=sh.plugin is None)]
        return self._tick_step()

    def begin(self) -> List[dict]:
        """A beat's start (metis-lang spec/17 amendment 4, copy semantics): its tick comes first, so the member
        reads the world as the snapshot at its tick's start, before any rule fires, and a command it fires shows
        in the world from its next tick. Lockstep and follow keep their order (module doc); an observing session,
        a hostless one and a done one start as before."""
        sh = self.shell
        if self.done or self.lockstep or self.follow or self.held or self.observe_only or sh.plugin is None:
            return []
        return self._tick_step()

    def _tick_step(self) -> List[dict]:
        """One env beat: the window's check, the host's tick (its do-lines one exo row), then what the beat's input
        asks. The check (copy semantics): the reading against the effects the member modelled for its own commands
        since the last checkpoint (the atoms its fired events consumed or produced), never against another
        member's change, which is the tick's input. A reading is never older than a command it is compared for: in
        one process the world applies a command at once, so this tick's snapshot has it (a split's proxy adds the
        pending ones)."""
        sh = self.shell
        if sh.plugin is not None and not (self.lockstep or self.follow or self.observe_only):
            pre = self._read(own=True)
        else:
            pre = []
        return pre + self._tick_rest()

    @property
    def tick_due(self) -> bool:
        """The world stepped on the last event: the beat ends here, its next one starts with the tick (the beat
        runners break on it as on a quiet session)."""
        return bool(self.__dict__.get("_tick_due"))

    def _tick_rest(self) -> List[dict]:
        sh = self.shell
        self._tick_due = False
        if sh.plugin is None or (self.beats is not None
                                 and self.beat >= self.beats):
            self.done = True
            return [self._signal("quiescence", stage=sh.stage,
                                 fired=len(sh.fired), episode_over=True,
                                 observe_only=self.observe_only)]
        out = self._read() if self.observe_only else []
        if self.observe_only:
            out += self._operator()
        name = sh._tick(quiet_rows=False, restart=self.lockstep)
        self.held = False
        due, self._quiet_due = self.__dict__.get("_quiet_due"), False
        if name is None:
            if sh.stage != DONE and not sh._candidates():
                sh._latch_done()
            if due:
                self._task_done()
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
            if name or self.lockstep:
                # input arrived: the program answers it first (a pack's rule
                # consuming the input atom: the plant's visitor_waters); the
                # reading comes after its row, as for any fired row. Lockstep:
                # every beat is a step's boundary, input or not (the reading
                # comes at the pass's end)
                self._checkpoint()
            else:
                # nothing arrived, nothing fires: the env moved on its own
                # (the plant's visitor waters it while nothing fires —
                # e2e-workflow s-143c3088 A5): a quiet domain is read on
                # every beat; a consistent reading checkpoints
                out += self._read()
        if due and not self.observe_only and not sh._candidates():
            # the tick's reading left nothing to do: the quiescence the last beat entered, declared now
            self.quiet = True
            self._task_done()
            out.append(self._signal("quiescence", stage=sh.stage, fired=len(sh.fired), episode_over=False))
        return out

    # -- the surprise test -------------------------------------------------

    def _task_done(self):
        """The task boundary: with scope "order", the bounds forget the
        surprises resolved during the task just finished."""
        self.tasks += 1
        if self.scope == "order":
            self._reset_sigs.clear()
            self._explained.clear()

    def _checkpoint(self):
        sh = self.shell
        self._cp = (sh.stage, dict(sh.counts), len(sh.fired))

    def _read(self, own=False) -> List[dict]:
        """The surprise test on the host's reading. own=True (begin, at a tick): only the atoms the member's own
        events since the checkpoint consumed or produced, those it modelled for its commands."""
        sh = self.shell
        observe = getattr(sh.plugin, "observe", None)
        if observe is None:
            self._checkpoint()
            return []
        reading = {a: int(n) for a, n in dict(observe(dict(sh.counts)))
                   .items()}
        if own:
            if len(sh.fired) == self.__dict__.get("_own_at"):
                return []               # nothing fired since the last tick's check: nothing new to compare
            self._own_at = len(sh.fired)
            touched = set()
            for kind, _, ev in sh.fired_rows[self._cp[2]:]:
                if kind != "exo":
                    touched |= set(ev.consume) | set(ev.produce)
            if not touched:
                return []               # nothing of its own to compare: the tick's input checkpoints
            if self._shared_world():
                # a world other members change too (kitchen_pot's pot): only the effects it modelled for its own
                # commands are compared; another member's change is the tick's input
                reading = {a: n for a, n in reading.items() if a in touched}
                self._scope = touched   # an open surprise keeps being sensed on the same atoms
        elif self.open is not None and self.__dict__.get("_scope") is not None:
            reading = {a: n for a, n in reading.items() if a in self._scope}
        else:
            self._scope = None
        if not reading and (self.lockstep or self.follow):
            return []                   # nothing read (the world has not moved): the window goes on
        model = {a: sh.counts.get(a, 0) for a in reading}
        diff = {a: reading[a] - model[a] for a in reading
                if reading[a] != model[a]}
        if self.open is not None:
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
            if self.blind:              # arm A: the reading, no verdict
                return [self._signal("event", kind="reading",
                                     consistent=False, diff=diff,
                                     observed=reading, model=model,
                                     record=self.open)]
            return [self._signal("surprise", record=self.open,
                                 kind=rec["kind"],
                                 likelihood=rec["likelihood"],
                                 steps=rec["steps"], observed=reading,
                                 model=model, diff=diff,
                                 observe_only=True)]
        if rec["kind"] == "rare":
            sh.loop.records.append(rec)
            out = [self._signal("event", kind="rare",
                                record=len(sh.loop.records) - 1,
                                likelihood=rec["likelihood"], diff=diff)]
        else:
            # another path of the served world explains the reading: a
            # belief change, not a surprise
            out = [self._signal("event", kind="reading", consistent=False,
                                likelihood=rec["likelihood"], diff=diff)]
        if self.follow:
            # the program follows its world: the reading it explained becomes
            # its state (one exo row) and the next window starts there
            out[-1]["adopted"] = self._adopt(by="reading")
            self._checkpoint()
        return out

    def _shared_world(self) -> bool:
        """The host says other members act on its world too (`shared_world = True`, on the host or the agent host's
        inner one). Then the tick's check compares only the member's own modelled effects; in a world only this member
        acts on, every difference the snapshot shows is its commands' doing, compared whole."""
        p = self.shell.plugin
        return bool(getattr(p, "shared_world", False) or getattr(getattr(p, "inner", None), "shared_world", False))

    def _record(self, reading: Dict[str, int]) -> dict:
        sh = self.shell
        stage, counts, n0 = self._cp
        prog = self.program_at(counts, stage)
        steps = max(1, len(sh.fired) - n0)
        snap = sh._env_snap()
        return self.price(
            prog, steps, reading,
            meta={"path": self.path, "beat": self.beat,
                  "fired": len(sh.fired), "stage": stage,
                  "since": sh.fired[n0:],
                  "host": sh.plugin_spec or None},
            env=snap)

    def program_at(self, counts, stage):
        """The served program grounded at (counts, stage): the session's
        own ground with that start (the events do not depend on it; a
        ground of a big lockstep world costs a second)."""
        from dataclasses import replace
        return replace(self.shell.prog, init=dict(counts), init_stage=stage)

    def price(self, prog, steps: int, reading: Dict[str, int],
              meta: Optional[dict] = None, env=None) -> dict:
        """surprise_record of `reading` over `steps` from `prog`, under
        the session's pricer and the watch's tier: exact, or (`tier`
        {max_states, samples, seed}) exact within the budget and the
        declared sampled tier past it, its runs seeded from the tier's
        seed and the session's position (reproducible per seed). The
        domain agent's scored beat prices through here too."""
        from metis.kernel.record import env_record, surprise_record
        sh = self.shell
        policy = (sh.pricer.policy(prog) if sh.pricer is not None
                  else None)
        sampled = None
        if self.tier is not None:
            # the same window priced twice (the watch's record at the pass's
            # end, the scored beat after it): once
            key = (prog.init_stage, tuple(sorted(prog.init.items())), steps,
                   tuple(sorted(reading.items())), len(sh.fired))
            memo = self.__dict__.get("_priced")
            if memo is not None and memo[0] == key:
                return dict(memo[1], meta=dict(meta or {}),
                            **({"env": env_record(env)} if env is not None
                               else {}))
            sampled = {"max_states": int(self.tier["max_states"]),
                       "samples": int(self.tier["samples"]),
                       "seed": int(self.tier.get("seed", 1)) * 1_000_003
                       + len(sh.fired),
                       "policy": (sh.pricer.policy(prog, canonical=True)
                                  if sh.pricer is not None else None)}
        rec = surprise_record(prog, steps, reading, policy=policy,
                              rare_ratio=self.rare_ratio, meta=meta,
                              env=env, sampled=sampled)
        if sampled is not None:
            self._priced = (key, {k: v for k, v in rec.items()
                                  if k not in ("meta", "env")})
        return rec

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

    def _verdict(self, n, verb, ok, reason, apply, **payload):
        """One intent's outcome. Checked (arms B/C): applied only when
        admitted. Blind (arm A, `watch arm=blind`): applied ALWAYS — the
        runtime's verdict rides along as `shadow`, so an illegal action
        is an applied intent whose shadow says refused."""
        if ok or self.blind:
            apply()
        if self.blind:
            sig = self._signal("result", intent=n, verb=verb,
                               status="applied", reason=reason,
                               shadow="admitted" if ok else "refused",
                               **payload)
            self.outbox.append(sig)
            self.results.append(sig)
        else:
            self._result(n, verb, ok, reason, **payload)

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
            self._verdict(n, "reset", False,
                          "the same surprise came back after a reset — no "
                          "more resets; escalate",
                          lambda: self._close(r, "reset"),
                          record=r, mined_text=text)
            return

        def apply():
            self._reset_sigs[sig] = (r, self.beat)
            self._close(r, "reset")
        self._verdict(n, "reset", True,
                      "the reading is set aside as noise; resumed on the "
                      "model — the next reading checks it", apply, record=r)

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
        out = {"adopted": {}}

        def apply():                    # adopt the reading as the card's
            out["adopted"] = self._adopt(by=f"{card['name']}:{move}")
            self._close(r, "explain", card=card["name"], move=move)
        before = self._explained.get(self._signature(rec))
        if before is not None:
            c0, m0 = before
            text = (f"{c0} ({m0}) explained this surprise before and it "
                    f"came back: the served world {rec['meta']['path']} "
                    f"lacks {c0}'s {m0}")
            self._refuse_fact(rec, "explain", text)
            self._verdict(n, "explain", False,
                          f"the same surprise came back after {c0} "
                          f"explained it — the served world is not switched "
                          f"(activate is phase 2); escalate", apply,
                          record=r, card=c0, move=m0, mined_text=text)
            return
        if card["relation"] == "served":
            self._verdict(n, "explain", False,
                          f"the served world is scored at exactly k={k} over "
                          "its own fired path: the record's own verdict "
                          "(unmodeled) — it may not invent its own moves",
                          apply, record=r, card=card["name"],
                          horizon={"served": k})
            return
        if move is None or move not in card["moves"]:
            offered = ", ".join(card["moves"]) or "none"
            self._verdict(n, "explain", False,
                          f"a proposal names one move of {card['name']} "
                          f"beside the fired path ({offered})", apply,
                          record=r, card=card["name"])
            return
        _, prog = card_program(self.shell, card, rec["init"],
                               rec["meta"]["stage"])
        path = _witness(prog, rec["init"], since, move, rec["observed"])
        if path is None:
            self._verdict(n, "explain", False,
                          f"{card['name']} with {move} does not reach the "
                          f"reading from the record over its fired path "
                          f"{since} (k+1={k + 1})", apply, record=r,
                          card=card["name"], move=move,
                          horizon={"served": k, "card": k + 1})
            return

        def admit():
            self._explained[self._signature(rec)] = (card["name"], move)
            key = (card["name"], move)
            last, run = self._streak.get(key, (None, 0))
            run = run + 1 if last == self.tasks - 1 else \
                (run if last == self.tasks else 1)
            self._streak[key] = (self.tasks, run)
            if self.notice_after and run >= self.notice_after:
                out["notice"] = (
                    f"{card['name']} ({move}) explained this diff in {run} "
                    f"consecutive orders: the served world {self.path} "
                    f"lacks it; activate is phase 2")
                rec["notice"] = {"card": card["name"], "move": move,
                                 "orders": run}
                rec["mined_text"] = out["notice"]
            apply()
        self._verdict(n, "explain", True,
                      f"{card['name']} explains the reading: {' '.join(path)}"
                      f" — the reading is adopted, the served world stays "
                      f"{self.path} (activate is phase 2), resumed", admit,
                      record=r, card=card["name"], move=move, witness=path,
                      horizon={"served": k, "card": k + 1},
                      adopted=out["adopted"])
        # the adopted edit is known only after apply(): patch the signal
        sig = self.outbox[-1]
        sig["adopted"] = out["adopted"]
        if out.get("notice"):
            if sig["status"] == "admitted":
                sig["status"] = "admitted+notice"
            sig["notice"] = rec["notice"]
            sig["mined_text"] = out["notice"]

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

        def apply():
            # the revision loop's input: the runtime's facts FIRST (a
            # refusal is authoritative; a model's sentence cannot be
            # trusted to carry it), then the model's description
            facts = rec.get("facts", [])
            rec["mined_text"] = "\n".join(facts + [text.strip()])
            rec["escalated"] = {"beat": self.beat, "facts": len(facts)}
        self._verdict(n, "escalate", True,
                      "stamped on the record for the revision loop; still "
                      "observe-only", apply, record=r,
                      chars=len(text.strip()))

    def _operator(self):
        """`watch operator=N`: an ESCALATED surprise still open N beats
        after its escalation is resumed by the operator (the human the
        escalation reached) — the reading adopted as one exo row
        'operator …', the record closed by 'operator'. Never an intent:
        the model cannot do this."""
        if self.operator is None or self.open is None:
            return []
        rec = self.shell.loop.records[self.open]
        esc = rec.get("escalated")
        if not esc or self.beat - esc["beat"] < self.operator:
            return []
        r = self.open
        adopted = self._adopt(by="operator")
        self._close(r, "operator")
        return [self._signal("event", kind="operator", record=r,
                             adopted=adopted, resumed=True)]

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
        sh._keep()                      # the bounded history (watch-history)
        self.quiet = False
        return diff

    def state(self) -> dict:
        return {"seq": self.seq, "beat": self.beat, "done": self.done,
                "scope": self.scope, "explain_repeat_notice":
                self.notice_after,
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
    elif (not cands or kinds == {"link"}) and (watch is None or (watch.quiet and not watch.__dict__.get("_quiet_due"))):
        status = "quiescent"            # as the watch declared it: a member waiting for its next tick is running
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
    # RACK v2 (finding 1): progress — the #query atoms' counts NOW — and
    # the last thing the domain did (its last fired event, not an env row)
    last = next((ev.name for kind, _, ev in reversed(shell.fired_rows)
                 if kind != "exo"), None)
    return [{"path": shell.doc.catalog.name, "stage": shell.stage,
             "status": status, "surprise": surprise, "horizon": steps,
             "beliefs": beliefs,
             "now": {q: shell.counts.get(q, 0) for q in queries},
             "last": last}]


def mirror_line(row: dict) -> str:
    s = row["surprise"]
    sur = (f"surprise #{s['record']} {s['kind']} @beat {s['beat']}"
           if s else "surprise none")
    bel = "  ".join(f"P[{q}]={'?' if p is None else f'{p:.2f}'}"
                    for q, p in row["beliefs"].items())
    now = " ".join(f"{q}={n}" for q, n in row.get("now", {}).items())
    return (f"{row['path']}  stage {row['stage']}  {row['status']}  "
            f"{sur}" + (f"  {bel} @{row['horizon']}" if bel else "")
            + (f"  now {now}" if now else "")
            + (f"  last {row['last']}" if row.get("last") else ""))


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
                     "notice": r.get("notice"),
                     "mined_text": r.get("mined_text")})
    out = {"records": recs,
           "results": list(watch.results) if watch is not None else []}
    plugin = shell.plugin
    if plugin is not None and hasattr(plugin, "schedule"):
        out["host"] = {"schedule": plugin.schedule(),
                       "injected": [dict(e) for e in
                                    getattr(plugin, "injected", [])]}
    observe = getattr(plugin, "observe", None)
    if observe is not None:
        reading = dict(observe(dict(shell.counts)))
        out["final"] = {
            "diff": {a: n - shell.counts.get(a, 0)
                     for a, n in reading.items()
                     if n != shell.counts.get(a, 0)},
            "served_model": shell.counts.get("served", 0),
            "served_host": reading.get("served", 0),
            "fired": len(shell.fired)}
    if watch is not None:
        out["watch"] = {**watch.state(), "blind": watch.blind,
                        "operator": watch.operator}
    return out
