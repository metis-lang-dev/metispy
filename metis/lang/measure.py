"""Spec 11 P2-P5: answering an instrument question, honestly tiered.

A question is one or more ARMS — snapshots of a world (stage, counts,
masks) after the question's interventions — and one OUTCOME (atom or
`!atom`) read at one READOUT:

    ("cone", N)   @N — after N cone events, or when no cone event can
                  fire if that is sooner (the default; interleaving-
                  insensitive: out-of-cone scheduling cannot move it)
    ("frame",)    @frame — when no cone event can fire any more
    ("global", N) plain N — N events of the whole world: kept for
                  validation, flagged scheduling-sensitive

Tiers, auto-selected:

- EXACT on the cone quotient (P2): reachability-pruned (events that can
  never fire from the arm's state are dropped — sound), swept to its
  cone (P1/P1a), answered by the kernel ForwardProjection unchanged; the
  quotient's canonical key is printed with the answer. `full` forces
  the unrestricted path (validation): the independent full enumerator
  for @N/@frame, the full filter for plain N.
- SAMPLED (P5) when the exact tier would exceed the interactive state
  budget (or is forced with `sampled`): Monte Carlo on the same quotient
  (the global readout samples the pruned full world). Arms are paired by
  COMMON RANDOM NUMBERS — a VARIANCE-REDUCTION device for the difference
  between arms, never a cross-world coupling claim: each arm estimates
  its own marginal, and spec 10's no-canonical-coupling stance stands.
  n grows in batches until every arm's 95% Wilson half-width meets the
  target or the time budget runs out; the answer says n and ± ci.

Every answer carries its tier and why, the cone line, the quotient key
and the readout.

WEIGHT PORTS (PLUGIN-PORTS.md): with a `price` (a PortPricer, or one
per arm — each arm prices under ITS env snapshot), the exact tier runs
on the masked, reachability-pruned WHOLE world (the cone still names
the readout's events): a port's declared read scope may name atoms the
quotient drops, so the quotient is never priced. Port values are
constants per arm snapshot — no env tick happens inside a measure."""
import math
import time
from dataclasses import replace
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from metis.kernel import ForwardProjection
from metis.kernel.filter import DONE, StateBudgetExceeded
from metis.kernel.program import Program
from metis.kernel.refsample import SplitMix64
from metis.kernel.weight import choice_weight

from .cone import (Cone, NotSettled, cone_line, cone_readout,
                   full_readout, priced, qkey, quotient, sweep)

import re


class Arm(NamedTuple):
    stage: str
    counts: Dict[str, int]
    masks: frozenset
    spent: int = 0          # forced events already applied (global)
    spent_names: tuple = ()  # their names (cone-step accounting)
    env: object = None       # the env snapshot the arm prices under


class Prepared(NamedTuple):
    arm: Arm
    masked: Callable[[str], bool]
    live: frozenset         # names of events that may still fire
    cone: Cone
    q: Program
    key: str


def masked_by(masks) -> Callable[[str], bool]:
    """Spec 10 D2: a mask names a ground event, or a rule (all its
    ground instances and ⊕ branches)."""
    masks = frozenset(masks)

    def masked(name):
        if not masks:
            return False
        rule = name.partition("[")[0]
        return (name in masks or rule in masks
                or re.sub(r"-\d+$", "", rule) in masks)
    return masked


def live_events(prog: Program, counts, masked) -> frozenset:
    """Sound pruning: the events that MAY still fire from `counts` (a
    forward may-reach fixpoint over premises); every other ground event
    can never be enabled, so dropping it changes no probability."""
    evs = [(ev.name, ev) for _, _, ev in prog.events()
           if not masked(ev.name)]
    reach = {a for a, n in counts.items() if n > 0}
    live, grew = set(), True
    while grew:
        grew = False
        for name, ev in evs:
            if name in live:
                continue
            if all(a in reach for a in (*ev.consume, *ev.persist)):
                live.add(name)
                grew = True
                reach.update(ev.produce)
    return frozenset(live)


def prepare(prog: Program, arm: Arm, outcome: str) -> Prepared:
    masked = masked_by(arm.masks)
    live = live_events(prog, arm.counts, masked)
    gone = lambda name: masked(name) or name not in live  # noqa: E731
    cone = sweep(prog, [outcome], arm.stage, gone)
    q = quotient(prog, cone, arm.stage, arm.counts)
    return Prepared(arm, masked, live, cone, q, qkey(q))


def cone_names(prog: Program, cone: Cone) -> frozenset:
    evs = prog.events()
    return frozenset(evs[i][2].name for i in cone.events)


def arm_readout(readout, prep: Prepared, cone_ns: frozenset):
    """The readout as THIS arm reads it: forced events already applied
    used up steps — every one for plain N, the in-cone ones for @N."""
    if readout[0] == "global":
        return ("global", max(0, readout[1] - prep.arm.spent))
    if readout[0] == "cone":
        used = sum(1 for n in prep.arm.spent_names if n in cone_ns)
        return ("cone", max(0, readout[1] - used))
    return readout


def readout_text(readout) -> str:
    if readout[0] == "frame":
        return "@frame (read when the cone settles)"
    if readout[0] == "cone":
        return (f"@{readout[1]} cone step{'' if readout[1] == 1 else 's'}"
                " (or when the cone settles, if sooner)")
    return (f"{readout[1]} global event{'' if readout[1] == 1 else 's'} "
            "(scheduling-sensitive — validation readout)")


def _world(prog: Program, prep: Prepared) -> Program:
    """The arm's whole world, masked and pruned (both sound)."""
    keep = lambda c: c.name in prep.live  # noqa: E731
    return replace(prog, init_stage=prep.arm.stage,
                   init=dict(prep.arm.counts),
                   stages={s: [c for c in cs if keep(c)]
                           for s, cs in prog.stages.items()},
                   links=[lk for lk in prog.links if keep(lk)])


def _holds_final(final, outcome):
    a, neg = outcome.lstrip("!"), outcome.startswith("!")
    p = sum(q for (_, k), q in final.items()
            if any(x == a and n > 0 for x, n in k))
    return 1.0 - p if neg else p


def exact_arm(prog, prep, outcome, readout, budget, full=False,
              price=None):
    """(p, how) on the exact tier; StateBudgetExceeded propagates."""
    if readout[0] == "global":
        w = _world(prog, prep)
        if readout[1] <= 0 or w.init_stage == DONE:
            return (float(_holds_counts(prep.arm.counts, outcome)),
                    "exact (now)")
        pr = ForwardProjection(w, readout[1], max_states=budget,
                               keep_history=False,
                               policy=price.policy(w) if price else None)
        return _holds_final(pr.final, outcome), "exact (full world)"
    if full or price is not None:
        w = replace(_world(prog, prep))
        return (full_readout(w, cone_names(prog, prep.cone), outcome,
                             readout, budget, price=price),
                "exact (full world, validation)" if price is None
                else "exact (priced world)")
    p, _ = cone_readout(prep.q, outcome, readout, budget)
    return p, "exact (cone)"


def _holds_counts(counts, outcome):
    return (counts.get(outcome.lstrip("!"), 0) > 0) != \
        outcome.startswith("!")


# -- the sampled tier (P5) ----------------------------------------------

class _Sim:
    """One arm's sampler: the quotient (cone/frame readouts) or the
    pruned whole world (global readout), the normative CHOICE."""

    def __init__(self, prog: Program, outcome, readout, frame_cap=1024,
                 price=None, cone_ns=None):
        # priced: the pruned whole world, stepping per CONE event
        # (cone_ns) like full_readout — the quotient is never priced
        self.price, self.cone_ns = price, cone_ns
        self.events = prog.events()
        self.by = {}
        for i, (k, s, ev) in enumerate(self.events):
            self.by.setdefault((s, k), []).append(ev)
        self.stage0, self.init = prog.init_stage, dict(prog.init)
        self.outcome, self.readout = outcome, readout
        self.steps = (readout[1] if readout[0] != "frame" else frame_cap)

    def draw(self, rng) -> bool:
        stage, c = self.stage0, dict(self.init)
        settled = self.readout[0] != "frame"
        cone_ns = self.cone_ns
        n_cone, budget = 0, self.steps
        if cone_ns is not None:
            budget = 64 * max(1, self.steps) + 4096   # out-of-cone slack
        for _ in range(budget):
            if stage == DONE:
                settled = True
                break
            if cone_ns is not None and self.readout[0] != "frame" \
                    and n_cone >= self.steps:
                break
            pick, ws, kind = None, [], None
            for kind in ("clause", "link"):
                ws = [(ev, choice_weight(ev, c))
                      for ev in self.by.get((stage, kind), ())]
                ws = [(ev, w) for ev, w in ws if w > 0]
                if ws and self.price is not None and kind == "clause":
                    ws = priced(ws, lambda ev: ev.name, c, self.price)
                if ws:
                    break
            if ws and cone_ns is not None and not any(
                    ev.name in cone_ns for ev, _ in ws):
                ws = []          # no cone event can fire: settled
            if ws:
                t = rng.unit() * sum(w for _, w in ws)
                acc = 0.0
                pick = ws[-1][0]
                for ev, w in ws:
                    acc += w
                    if t < acc:
                        pick = ev
                        break
                if kind == "link":
                    stage = pick.post
            if pick is None:
                settled = True
                break                         # quiescent: absorbed
            if cone_ns is not None and pick.name in cone_ns:
                n_cone += 1
            for q, n in pick.consume.items():
                c[q] = c.get(q, 0) - n
            for q, n in pick.produce.items():
                c[q] = c.get(q, 0) + n
        if not settled:
            raise NotSettled(
                f"a sampled cone run had not settled after {self.steps} "
                "cone steps — @frame has no answer here; read at @N")
        return _holds_counts(c, self.outcome)


def wilson_half(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 1.0
    p = k / n
    d = 1 + z * z / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d


def sampled(prog, preps, outcome, readouts, target=0.01, seconds=5.0,
            n_min=400, n_max=200_000, seed=1, batch=200, prices=None):
    """CRN-paired Monte Carlo over the arms -> ([(p, half)], n, secs).
    Draw i of every arm uses SplitMix64 seeded from (seed, i): the SAME
    random numbers in each arm — variance reduction for the difference
    estimator only; each arm's estimate is of its own marginal."""
    sims = []
    prices = prices or [None] * len(preps)
    for prep, ro, pz in zip(preps, readouts, prices):
        if pz is not None:
            sims.append(_Sim(_world(prog, prep), outcome, ro, price=pz,
                             cone_ns=None if ro[0] == "global"
                             else cone_names(prog, prep.cone)))
            continue
        w = _world(prog, prep) if ro[0] == "global" else prep.q
        sims.append(_Sim(w, outcome, ro))
    hits = [0] * len(sims)
    n, t0 = 0, time.time()
    while n < n_max:
        for _ in range(batch):
            for j, sim in enumerate(sims):
                hits[j] += sim.draw(SplitMix64(seed * 1_000_003 + n))
            n += 1
        if n >= n_min and (all(wilson_half(h, n) <= target
                               for h in hits)
                           or time.time() - t0 >= seconds):
            break
    return ([(h / n, wilson_half(h, n)) for h in hits], n,
            time.time() - t0)


class Answer(NamedTuple):
    ps: List[float]
    cis: List[Optional[float]]     # None on the exact tier
    tier: str                      # "exact (cone)" | "sampled …" | …
    why: str                       # why this tier ("" for exact)
    preps: List[Prepared]
    readouts: list
    n: int = 0
    refusal: Optional[Exception] = None   # the exact tier's, if sampled


AUTO_BUDGET = 100_000   # the auto tier's exact attempt: fail FAST, then
                        # sample (a forced `exact` gets the full budget)


def measure(prog: Program, arms: List[Arm], outcome: str, readout,
            tier: str = "auto", budget: int = 500_000,
            cache: Optional[dict] = None, seed: int = 1,
            price=None) -> Answer:
    """Answer one question over its arms (the instruments' core). tier:
    auto (exact below budget, else sampled) | exact (may refuse: raises
    StateBudgetExceeded) | sampled | full (exact, unrestricted).
    `price`: a PortPricer, or a list of them (one per arm)."""
    prices = (list(price) if isinstance(price, (list, tuple))
              else [price] * len(arms))
    if any(p is not None for p in prices):
        cache = None        # port values ride on env state: never cached
    preps = [prepare(prog, a, outcome) for a in arms]
    base_names = cone_names(prog, preps[0].cone)
    if callable(readout):        # D1's default: from the question's cone
        readout = readout(base_names)
    ros = [arm_readout(readout, p, base_names) for p in preps]
    why, refusal = "", None
    if tier != "sampled":
        try:
            ps, hows = [], []
            for prep, ro, pz in zip(preps, ros, prices):
                # D4: the quotient key identifies a QUOTIENT computation
                # only — global and full answers depend on the whole
                # world (two arms can share a quotient and differ
                # outside it), so they are never cached by it
                ck = (prep.key, ro, outcome) if (
                    tier != "full" and ro[0] != "global") else None
                if cache is not None and ck in cache:
                    p, how = cache[ck]
                else:
                    p, how = exact_arm(
                        prog, prep, outcome, ro,
                        min(budget, AUTO_BUDGET) if tier == "auto"
                        else budget, full=tier == "full", price=pz)
                    if cache is not None and ck is not None:
                        cache[ck] = (p, how)
                ps.append(p)
                hows.append(how)
            return Answer(ps, [None] * len(ps), hows[0], "", preps, ros)
        except StateBudgetExceeded as e:
            if tier in ("exact", "full"):
                raise
            why, refusal = f"exact refused: {e}", e
    res, n, secs = sampled(prog, preps, outcome, ros, seed=seed,
                           prices=prices)
    return Answer([p for p, _ in res], [h for _, h in res],
                  f"sampled n={n}, 95% ± ci ({secs:.1f} s)", why, preps,
                  ros, n, refusal)


def parse_tail(tokens: List[str]) -> Tuple[Optional[tuple], str]:
    """`[@frame|@N|N] [exact|sampled|full]` (any order) ->
    (readout or None for the default, tier). ValueError on junk."""
    readout, tier = None, "auto"
    for t in tokens:
        if t in ("exact", "sampled", "full"):
            tier = t
        elif t == "@frame":
            readout = ("frame",)
        elif t.startswith("@") and t[1:].isdigit():
            readout = ("cone", int(t[1:]))
        elif t.isdigit():
            readout = ("global", int(t))
        else:
            raise ValueError(t)
    return readout, tier


def structural(prep: Prepared, outcome: str) -> str:
    """A cone with NO events: nothing (under the active masks, from
    this state) can produce or consume the outcome's atom — its value
    is fixed, a structural PROOF, not a probability. -> "" or the
    phrase to print beside P."""
    if prep.cone.events:
        return ""
    a, neg = outcome.lstrip("!"), outcome.startswith("!")
    present = prep.arm.counts.get(a, 0) > 0
    holds = present != neg
    what = ("no event can consume" if present else
            "no event can produce") + f" {a} under the active masks"
    return (f"certain — {what}" if holds else f"impossible — {what}")


def pval(p: float, ci: Optional[float]) -> str:
    if abs(p) < 1e-12:
        p = 0.0
    elif abs(1 - p) < 1e-12:
        p = 1.0
    return f"{p:.4g}" + (f" ± {ci:.2g}" if ci is not None else "")


def footer(ans: Answer, labels: List[str]) -> List[str]:
    """The lines every cone answer carries: tier (and why), the readout,
    and per arm its cone line + quotient key."""
    out = [f"  tier         {ans.tier}" + (f" — {ans.why}" if ans.why
                                           else "")]
    if ans.cis[0] is not None:
        out.append("               (arms paired by common random numbers "
                   "— variance reduction only, no cross-world coupling)")
    ro = ans.readouts[0]
    out.append(f"  readout      {readout_text(ro)}")
    for lab, prep in zip(labels, ans.preps):
        out.append(f"  cone {lab:<8}{cone_line(prep.cone)} · key "
                   f"{prep.key[:12]}")
    return out
