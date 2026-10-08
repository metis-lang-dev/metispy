"""Spec 11 — the influence cone and the cone-restricted filter.

P1 (the sweep): from the question's outcome atoms, backward over the
GROUND events after masks prune them: every event whose consume or
produce touches a cone atom joins, and its consume + persist premises
join the atoms; repeat to closure. By construction only cone events
write cone atoms and cone events read only cone atoms, so the cone is
an AUTONOMOUS sub-program: out-of-cone events can neither change it
nor enable/disable it ($-reads of cone atoms do not disturb it).

STAGE CLOSURE (an extension P1 needs, found in implementation): the
tier is "this stage's clauses, else its links" — a link fires only
when EVERY clause of its stage is disabled, a negative premise over
out-of-cone atoms too. So when a link is reachable from the current
stage, the cone must hold every event of the reachable stages (exact,
but no reduction); single-stage catalogs (the fiction set) restrict.

P2: the quotient is a Program like any other — the cone's clauses
(and links, when stage-closed), its state projected to the cone atoms
— answered by the kernel ForwardProjection UNCHANGED, with its own
canonical key (program_key). Readouts (P3): @N = N quotient steps
(each one a cone event; quiescence absorbs as DONE), @frame = until
the quotient is quiescent.

`full_readout` is the validation side (§5 gate 1): an independent
enumerator over the FULL program that counts cone events and absorbs
at the same readout — kept deliberately separate from the quotient
path so agreement is evidence, not a tautology."""
from typing import Callable, Dict, FrozenSet, Iterable, NamedTuple, \
    Optional, Tuple

from metis.kernel import ForwardProjection
from metis.kernel.filter import DONE, StateBudgetExceeded, apply_event
from metis.kernel.nets import program_key
from metis.kernel.program import Program, pack_counts
from metis.kernel.weight import choice_weight


class NotSettled(RuntimeError):
    """@frame asked of a cone that does not settle within the cap — an
    @frame answer does not exist there (read at @N instead); never a
    probability read off an unsettled distribution."""


class Cone(NamedTuple):
    atoms: FrozenSet[str]
    events: FrozenSet[int]       # indices into program.events()
    stage_closed: bool           # links reachable: the whole reachable
    n_atoms: int                 # totals, for the cone line
    n_events: int


def _all_atoms(prog: Program) -> set:
    out = set(prog.init)
    for _, _, ev in prog.events():
        out |= set(ev.consume) | set(ev.produce) | set(ev.persist)
    return out


def _reachable_stages(prog, stage, live):
    seen, todo = {stage}, [stage]
    evs = prog.events()
    while todo:
        s = todo.pop()
        for i in live:
            k, pre, ev = evs[i]
            if k == "link" and pre == s and ev.post not in seen:
                seen.add(ev.post)
                todo.append(ev.post)
    return seen


def sweep(prog: Program, outcomes: Iterable[str], stage: str,
          masked: Callable[[str], bool] = lambda name: False) -> Cone:
    """P1 to closure (+ the stage closure). `outcomes` are atom names
    (an `!atom` outcome tracks the same atom)."""
    evs = prog.events()
    live = [i for i, (_, _, ev) in enumerate(evs) if not masked(ev.name)]
    n_atoms, n_events = len(_all_atoms(prog)), len(evs)
    stages = _reachable_stages(prog, stage, live)
    if any(evs[i][0] == "link" and evs[i][1] in stages for i in live):
        events = frozenset(i for i in live if evs[i][1] in stages)
        atoms = set()
        for i in events:
            ev = evs[i][2]
            atoms |= set(ev.consume) | set(ev.produce) | set(ev.persist)
        return Cone(frozenset(atoms), events, True, n_atoms, n_events)
    # single reachable stage, no links: only its clauses can ever fire
    writers: Dict[str, list] = {}
    for i in live:
        k, s, ev = evs[i]
        if k == "clause" and s == stage:
            for a in (*ev.consume, *ev.produce):
                writers.setdefault(a, []).append(i)
    atoms = {a.lstrip("!") for a in outcomes}
    events = set()
    todo = list(atoms)
    while todo:
        a = todo.pop()
        for i in writers.get(a, ()):
            if i in events:
                continue
            events.add(i)
            ev = evs[i][2]
            for b in (*ev.consume, *ev.persist):
                if b not in atoms:
                    atoms.add(b)
                    todo.append(b)
    return Cone(frozenset(atoms), frozenset(events), False, n_atoms,
                n_events)


def check_closed(prog: Program, cone: Cone, stage: str,
                 masked: Callable[[str], bool] = lambda name: False):
    """The closure property, checked — [] when it holds, else the
    violations (an out-of-cone live event of the stage writing a cone
    atom, or a cone event reading an out-of-cone atom): a construction
    bug, never acceptable (spec 11 §5)."""
    bad = []
    for i, (k, s, ev) in enumerate(prog.events()):
        if masked(ev.name) or (not cone.stage_closed and s != stage):
            continue
        writes = set(ev.consume) | set(ev.produce)
        if i not in cone.events and writes & cone.atoms:
            bad.append(f"out-of-cone {ev.name} writes "
                       f"{sorted(writes & cone.atoms)}")
        if i in cone.events:
            reads = set(ev.consume) | set(ev.persist)
            if reads - cone.atoms:
                bad.append(f"cone {ev.name} reads out-of-cone "
                           f"{sorted(reads - cone.atoms)}")
    return bad


def quotient(prog: Program, cone: Cone, stage: str,
             counts: Dict[str, int]) -> Program:
    """The cone as a Program: its clauses/links, the state projected to
    the cone atoms. Kernel-plain — program_key applies."""
    pos = {id(ev): i for i, (_, _, ev) in enumerate(prog.events())}
    stages = {s: [c for c in cs if pos[id(c)] in cone.events]
              for s, cs in prog.stages.items()}
    links = [lk for lk in prog.links if pos[id(lk)] in cone.events]
    init = {a: n for a, n in counts.items() if a in cone.atoms and n > 0}
    return Program(stages=stages, links=links, init_stage=stage,
                   init=init)


def _holds_p(final, outcome: str) -> float:
    a, neg = outcome.lstrip("!"), outcome.startswith("!")
    p = sum(q for (_, k), q in final.items()
            if any(x == a and n > 0 for x, n in k))
    return 1.0 - p if neg else p


def cone_readout(q: Program, outcome: str, readout, budget: int,
                 frame_cap: int = 1024) -> Tuple[float, dict]:
    """P[outcome] on the quotient under readout ("cone", N) or
    ("frame",). Returns (p, info{steps, settled})."""
    if readout[0] == "cone":
        n = max(0, int(readout[1]))
        if n == 0:
            return _holds_p({(q.init_stage, pack_counts(q.init)): 1.0},
                            outcome), {"steps": 0, "settled": None}
        pr = ForwardProjection(q, n, max_states=budget,
                               keep_history=False)
        return _holds_p(pr.final, outcome), {"steps": n, "settled": None}
    s = 8
    while True:
        pr = ForwardProjection(q, s, max_states=budget,
                               keep_history=False)
        done = sum(p for (st, _), p in pr.final.items() if st == DONE)
        if done >= 1.0 - 1e-12:
            return _holds_p(pr.final, outcome), {
                "steps": s, "settled": done}
        if s >= frame_cap:
            raise NotSettled(
                f"the cone has not settled after {s} cone steps (settled "
                f"mass {done:.4g}) — @frame has no answer here; read at "
                "@N cone steps")
        s *= 2


def _tier(events, by_stage, stage, counts):
    """The normative CHOICE (clauses of the stage, else its links) —
    re-stated plainly for the validation enumerator."""
    if stage == DONE:
        return [(None, 1.0)]
    for kind in ("clause", "link"):
        ws = [(i, choice_weight(ev, counts))
              for i, ev in by_stage.get((stage, kind), ())]
        ws = [(i, w) for i, w in ws if w > 0]
        tot = sum(w for _, w in ws)
        if tot > 0:
            return [(i, w / tot) for i, w in ws]
    return [(None, 1.0)]


def full_readout(prog: Program, cone_names: FrozenSet[str],
                 outcome: str, readout, budget: int,
                 max_steps: int = 5_000) -> float:
    """VALIDATION (§5 gate 1): the FULL program enumerated exactly,
    counting cone events; mass absorbs at the readout — ("cone", N): N
    cone events fired, or none can fire (settled) if that comes first;
    ("frame",): none can fire — or at full quiescence. Independent of `quotient`/`cone_readout`. Cone events
    are named (ground names are unique), so a masked or pruned program
    keeps its membership."""
    events = prog.events()
    cone_events = frozenset(i for i, (_, _, ev) in enumerate(events)
                            if ev.name in cone_names)
    by_stage: Dict[tuple, list] = {}
    for i, (k, s, ev) in enumerate(events):
        by_stage.setdefault((s, k), []).append((i, ev))
    a, neg = outcome.lstrip("!"), outcome.startswith("!")
    start = (prog.init_stage, pack_counts(prog.init), 0)
    dist = {start: 1.0}
    hit = 0.0
    retained = 1
    for step in range(max_steps):
        if not dist:
            break
        new: Dict[tuple, float] = {}
        for (stage, kappa, c), p in dist.items():
            counts = dict(kappa)
            choices = _tier(events, by_stage, stage, counts)
            # @N reads after N cone events OR when no cone event can
            # fire, whichever is first (the cone has settled: the
            # quotient's DONE); @frame reads at the latter only
            settled = not any(e in cone_events for e, _ in choices)
            absorb = choices == [(None, 1.0)] or settled or (
                readout[0] == "cone" and c >= readout[1])
            if absorb:
                hit += p * (counts.get(a, 0) > 0)
                continue
            for e, pe in choices:
                nst, nk = apply_event(events, stage, counts, e)
                key = (nst, nk, c + (e in cone_events))
                new[key] = new.get(key, 0.0) + p * pe
        retained += len(new)
        if budget and retained > budget:
            raise StateBudgetExceeded(step + 1, max_steps, retained,
                                      budget)
        dist = new
    if dist:
        raise NotSettled(f"the full world did not reach the readout within "
                         f"{max_steps} events — read at @N cone steps")
    return 1.0 - hit if neg else hit


def cone_line(cone: Cone) -> str:
    return (f"tracking {len(cone.atoms)} of {cone.n_atoms} atoms · "
            f"{len(cone.events)} of {cone.n_events} events"
            + (" (stage-closed: links reachable)" if cone.stage_closed
               else ""))


def qkey(q: Program) -> str:
    return program_key(q)
