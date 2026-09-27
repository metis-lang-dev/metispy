"""Generic synchronous-Bernoulli compilation — the stage-7 SysAdmin
pattern, promoted: a bank of binary units updated SYNCHRONOUSLY (each
unit an independent Bernoulli whose parameter reads the pre-tick
snapshot) compiles to the interleaved engine as three chained stages

    mark    on(u) -> p_on(u) + tok(u)     (off -> p_off; the snapshot)
    update  tok(u) consumed, outcome n_on(u)/n_off(u); the CHOICE
            weights come from a per-unit probability function reading
            snapshot atoms only, so order cannot matter
    sweep   snapshot + outcome -> on/off  (commit)

with a sequencing-chain token per stage (2^k states instead of 3^n) and
cyclic links re-arming the next tick. Exactness of this compilation is
pinned by the sysadmin and game-of-life analytic gates.

A `force(u)` atom in the initial multiset marks unit u as acted-on;
its EFFECT is the domain's business (p_on reads the atom): SysAdmin's
reboot is KronDelta(1), Game of Life's set is Bernoulli(1-noise)."""

from typing import Callable, Dict, Iterable, List, Optional

from .clause import Clause, Link
from .program import Program


def sync_program(units: List[str], on: str = "on", off: str = "off",
                 state_on: Iterable[str] = (),
                 force: Optional[str] = None) -> Program:
    n = len(units)
    pon, poff = f"p{on}", f"p{off}"
    non, noff = f"n{on}", f"n{off}"
    mark, update, sweep = [], [], []
    for i, u in enumerate(units):
        m0, m1 = f"m{i}", f"m{i + 1}"
        u0, u1 = f"u{i}", f"u{i + 1}"
        s0, s1 = f"s{i}", f"s{i + 1}"
        mark.append(Clause(f"mr({u})", {m0: 1, f"{on}({u})": 1},
                           {f"{pon}({u})": 1, f"tok({u})": 1, m1: 1}))
        mark.append(Clause(f"md({u})", {m0: 1, f"{off}({u})": 1},
                           {f"{poff}({u})": 1, f"tok({u})": 1, m1: 1}))
        for name, snap, out in (
                (f"up({u})", pon, non), (f"dn({u})", pon, noff),
                (f"rv({u})", poff, non), (f"st({u})", poff, noff)):
            update.append(Clause(name, {u0: 1, f"tok({u})": 1},
                                 {f"{out}({u})": 1, u1: 1},
                                 persist={f"{snap}({u})": 1}))
        for snap, out, dest in (
                (pon, non, on), (pon, noff, off),
                (poff, non, on), (poff, noff, off)):
            sweep.append(Clause(
                f"sw({u},{snap},{out})",
                {s0: 1, f"{snap}({u})": 1, f"{out}({u})": 1},
                {f"{dest}({u})": 1, s1: 1}))
    links = [Link("mu", pre="mark", post="update",
                  consume={f"m{n}": 1}, produce={"u0": 1}),
             Link("us", pre="update", post="sweep",
                  consume={f"u{n}": 1}, produce={"s0": 1}),
             Link("sm", pre="sweep", post="mark",
                  consume={f"s{n}": 1}, produce={"m0": 1})]
    init = {"m0": 1}
    state_on = set(state_on)
    for u in units:
        init[f"{on}({u})" if u in state_on else f"{off}({u})"] = 1
    if force is not None:
        init[f"force({force})"] = 1
    return Program(stages={"mark": mark, "update": update,
                           "sweep": sweep},
                   links=links, init_stage="mark", init=init)


def sync_policy(program: Program,
                p_on: Callable[[str, Dict[str, int], bool], float]):
    """CHOICE weights for the update stage. p_on(unit, counts, was_on)
    returns P(unit is ON next) reading snapshot (and force) atoms
    only — force semantics are the domain's."""
    kind_of = {}
    for i, (k, s, ev) in enumerate(program.events()):
        if k == "clause" and s == "update":
            kind_of[i] = (ev.name[:2], ev.name[3:-1])

    def policy(stage, counts, ws):
        if stage != "update":
            return ws
        out = []
        for i, w in ws:
            kind, u = kind_of[i]
            was_on = kind in ("up", "dn")
            p = p_on(u, counts, was_on)
            w2 = p if kind in ("up", "rv") else 1.0 - p
            if w2 > 0:
                out.append((i, w * w2))
        return out

    return policy


def steps_per_tick(n_units: int) -> int:
    return 3 * n_units + 3


def marginal_on(pr, unit: str, on: str = "on") -> float:
    return sum(p for (_, k), p in pr.final.items()
               if (f"{on}({unit})", 1) in k)


def expected_on(pr, on: str = "on") -> float:
    pref = f"{on}("
    return sum(p * sum(1 for a, c in k if a.startswith(pref))
               for (_, k), p in pr.final.items())
