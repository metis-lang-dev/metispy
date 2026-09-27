"""SysAdmin (IPPC 2011 MDP) as a metis domain — the genericity proof.

Dynamics (verbatim from the competition domain file):
    running'(x) = reboot(x)      -> true
                  running(x)     -> Bernoulli(.45 + .5*(1 + running
                                    in-neighbors) / (1 + in-neighbors))
                  else           -> Bernoulli(REBOOT-PROB)
    reward      = sum running(c) - REBOOT-PENALTY * sum reboot(c)

Since stage 7 this file carried its own copy of the synchronous ->
interleaved compilation (mark/update/sweep + chain tokens); kernel/
sync.py promoted that pattern to the engine, and this module is now
what it should be: the DOMAIN ONLY — the CPT (p_on) + the reward,
over the generic sync compilation. The reboot action is sync's
`force` atom; its KronDelta effect lives in p_on, per the sync
doctrine ("force semantics are the domain's business").

Exactness of the compilation is pinned by tests/sysadmin (analytic
values, multi-tick totals) exactly as before the migration — the
gates did not move, the implementation under them deduplicated."""

from typing import Dict, Iterable, List, Optional, Set

from metis.kernel import ForwardProjection, plan, steps_per_tick, \
    sync_policy, sync_program

REBOOT_PROB = 0.04        # instance non-fluents override these
REBOOT_PENALTY = 0.75


def sysadmin_p_on(in_neighbors: Dict[str, List[str]],
                  reboot_prob: float = REBOOT_PROB):
    """The CPT: P(running next) from the snapshot — the Bernoulli
    branch per computer; a forced (rebooted) computer is KronDelta
    up."""
    def p_on(c, counts, was_on):
        if f"force({c})" in counts:
            return 1.0
        if was_on:
            ins = in_neighbors.get(c, [])
            r = sum(1 for y in ins if f"prun({y})" in counts)
            return 0.45 + 0.5 * (1 + r) / (1 + len(ins))
        return reboot_prob

    return p_on


def build_program(computers: List[str], running: Set[str],
                  reboot: Optional[str] = None):
    return sync_program(computers, on="run", off="down",
                        state_on=running, force=reboot)


def make_policy(program, in_neighbors: Dict[str, List[str]],
                reboot_prob: float = REBOOT_PROB):
    return sync_policy(program,
                       sysadmin_p_on(in_neighbors, reboot_prob))


def project(computers: List[str], in_neighbors: Dict[str, List[str]],
            running: Iterable[str], ticks: int = 1,
            reboot: Optional[str] = None,
            reboot_prob: float = REBOOT_PROB) -> ForwardProjection:
    prog = build_program(computers, set(running), reboot)
    return ForwardProjection(
        prog, ticks * steps_per_tick(len(computers)),
        policy=make_policy(prog, in_neighbors, reboot_prob))


def running_marginal(pr: ForwardProjection, c: str) -> float:
    return sum(p for (_, kappa), p in pr.final.items()
               if (f"run({c})", 1) in kappa)


def expected_running(pr: ForwardProjection) -> float:
    ev = 0.0
    for (_, kappa), p in pr.final.items():
        ev += p * sum(1 for a, k in kappa if a.startswith("run("))
    return ev


def greedy_plan(computers: List[str], in_neighbors, running: Set[str],
                reboot_prob: float = REBOOT_PROB,
                penalty: float = REBOOT_PENALTY):
    """One-step-lookahead exact-measure agent: argmax over {noop} +
    reboot(down computer) of E[sum running'] - penalty*reboot, via the
    generic kernel loop. The reboot cost is read off the projection's
    own init (the force atom) — score sees only the measure."""
    candidates = [None] + sorted(c for c in computers
                                 if c not in running)

    def proj(a):
        return project(computers, in_neighbors, running, ticks=1,
                       reboot=a, reboot_prob=reboot_prob)

    def score(pr):
        rebooted = any(atom.startswith("force(")
                       for atom in pr.program.init)
        return ((penalty if rebooted else 0.0) - expected_running(pr),)

    return plan(candidates, do=proj, project=lambda pr: pr, score=score,
                doomed_action=None)
