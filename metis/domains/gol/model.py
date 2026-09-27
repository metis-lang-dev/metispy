"""Game of Life (IPPC 2011 MDP) on the generic sync compilation.

Dynamics (competition domain file): alive'(x,y) = Bernoulli(1-noise)
when [alive ∧ 2<=live-neighbors<=3] ∨ [dead ∧ live-neighbors==3] ∨
set(x,y), else Bernoulli(noise). Reward = Σ alive. The set action ORs
into the CONDITION — a forced cell gets 1-noise, not certainty.

The whole domain is: the neighbor topology + this p_on function +
kernel.sync. Everything else (program, policy, filter, planning) is
the unchanged engine."""

from typing import Dict, List, Optional, Set

from metis.kernel import ForwardProjection, expected_on, plan, \
    steps_per_tick, sync_policy, sync_program


def gol_p_on(neighbors: Dict[str, List[str]],
             noise: Dict[str, float]):
    def p_on(u, counts, was_on):
        if f"force({u})" in counts:
            cond = True
        else:
            live = sum(1 for v in neighbors[u]
                       if f"palive({v})" in counts)
            cond = (2 <= live <= 3) if was_on else (live == 3)
        nz = noise[u]
        return 1.0 - nz if cond else nz

    return p_on


def gol_program(cells: List[str], alive: Set[str],
                force: Optional[str] = None):
    return sync_program(cells, on="alive", off="dead",
                        state_on=alive, force=force)


def gol_project(cells, neighbors, noise, alive, ticks=1,
                force=None) -> ForwardProjection:
    prog = gol_program(cells, set(alive), force)
    return ForwardProjection(
        prog, ticks * steps_per_tick(len(cells)),
        policy=sync_policy(prog, gol_p_on(neighbors, noise)))


def greedy_set_factored(cells, neighbors, noise, alive):
    """Beyond-cliff one-step agent (instance 5 = 16 cells, past the
    2^16 support cliff): the exact E[Σ alive'] lookahead through the
    FACTORED evaluator. One base pass suffices — at one tick, set(c)
    changes only c's OWN next-state Bernoulli (neighbors read c's
    current value, which is fixed), so the candidate's gain is the
    closed-form delta (1 - noise(c)) - P_base(alive'(c)); the argmax
    is exact. Ties/no-gain resolve to noop, earliest cell first."""
    from metis.kernel.factored import FactoredSync

    def p(u, snap, was_on):
        live = sum(snap[v] for v in neighbors[u])
        cond = (2 <= live <= 3) if was_on else (live == 3)
        nz = noise[u]
        return 1.0 - nz if cond else nz

    ev = FactoredSync(cells, neighbors, p, set(alive)).tick()
    best, gain = None, 1e-12
    for c in sorted(cells):
        d = (1.0 - noise[c]) - ev.marginal(c)
        if d > gain:
            best, gain = c, d
    return best


def greedy_set(cells, neighbors, noise, alive):
    """One-step exact-measure agent: argmax over {noop} ∪ {set(c)} of
    E[Σ alive'] (no action cost in this domain), via the kernel loop."""
    def proj(a):
        return gol_project(cells, neighbors, noise, alive, ticks=1,
                           force=a)

    return plan([None] + list(cells), do=proj,
                project=lambda pr: pr,
                score=lambda pr: (-expected_on(pr, on="alive"),),
                doomed_action=None)


# -- the text recode (stage 42): the catalog IS the domain ---------------

def gen_catalog(cells: List[str],
                neighbors: Dict[str, List[str]]) -> str:
    """GoL as generated .llp — the sync compilation authored as text
    (sysadmin2/ring pattern), removing the last code-side program
    builder. The `life(C)` port reads the snapshot neighborhood plus
    the force atom; pinned against the kernel.sync path at 1e-9
    (tests/gol) and corpus-gated across the toolchain triangle."""
    n = len(cells)
    lines = [
        "% GENERATED — Game of Life, the sync shape authored as text.",
        "catalog gol 2.",
        'provenance hand "text recode of the sync compilation" '
        '"2026-07-22".',
        "layers (dyn).",
        "stages (mark update sweep).",
        f"type cell {{{' '.join(cells)}}}.",
        f"type idx {{i0 .. i{n}}}.",
        "namespace state produce (dyn).",
    ]
    for pr in ("a", "d", "pa", "pd", "tok", "na", "nd", "force"):
        lines.append(f"pred {pr}(cell) : state.")
    for pr in ("m", "u", "s"):
        lines.append(f"pred {pr}(idx) : state.")
    lines += ["bwd cidx(cell,idx).", "bwd succ(idx,idx).",
              "bwd adj(cell,cell)."]
    for k, c in enumerate(cells):
        lines.append(f"cidx({c},i{k}).")
    for k in range(n):
        lines.append(f"succ(i{k},i{k + 1}).")
    for c in cells:
        for v in neighbors[c]:
            lines.append(f"adj({c},{v}).")
    lines += ["input grid.", "output action.",
              "weight life(C) reads {force(C), pa(C), "
              "pa(D) if adj(C,D)}."]
    lines += [
        "stage mark {",
        "  %% snapshot an alive cell.",
        "  mr [dyn] : m(I) * a(C) * cidx(C,I) * succ(I,I2)"
        " -o pa(C) * tok(C) * m(I2).",
        "  %% snapshot a dead cell.",
        "  md [dyn] : m(I) * d(C) * cidx(C,I) * succ(I,I2)"
        " -o pd(C) * tok(C) * m(I2).",
        "}",
        "stage update {",
        "  %% was alive, stays alive.",
        "  up [dyn] : u(I) * tok(C) * $pa(C) * cidx(C,I) * succ(I,I2)"
        " -o na(C) * u(I2)  @w life(C).",
        "  %% was alive, dies.",
        "  dn [dyn] : u(I) * tok(C) * $pa(C) * cidx(C,I) * succ(I,I2)"
        " -o nd(C) * u(I2)  @w ~life(C).",
        "  %% was dead, is born.",
        "  rv [dyn] : u(I) * tok(C) * $pd(C) * cidx(C,I) * succ(I,I2)"
        " -o na(C) * u(I2)  @w life(C).",
        "  %% was dead, stays dead.",
        "  st [dyn] : u(I) * tok(C) * $pd(C) * cidx(C,I) * succ(I,I2)"
        " -o nd(C) * u(I2)  @w ~life(C).",
        "}",
        "stage sweep {",
        "  %% commit: was alive, next alive.",
        "  swaa [dyn] : s(I) * pa(C) * na(C) * cidx(C,I) * succ(I,I2)"
        " -o a(C) * s(I2).",
        "  %% commit: was alive, next dead.",
        "  swad [dyn] : s(I) * pa(C) * nd(C) * cidx(C,I) * succ(I,I2)"
        " -o d(C) * s(I2).",
        "  %% commit: was dead, next alive.",
        "  swda [dyn] : s(I) * pd(C) * na(C) * cidx(C,I) * succ(I,I2)"
        " -o a(C) * s(I2).",
        "  %% commit: was dead, next dead.",
        "  swdd [dyn] : s(I) * pd(C) * nd(C) * cidx(C,I) * succ(I,I2)"
        " -o d(C) * s(I2).",
        "}",
        "%% snapshot done -> update pass.",
        f"qui mu [dyn] : mark * m(i{n}) -o update * u(i0).",
        "%% update done -> commit.",
        f"qui us [dyn] : update * u(i{n}) -o sweep * s(i0).",
        "%% committed -> next tick.",
        f"qui sm [dyn] : sweep * s(i{n}) -o mark * m(i0).",
    ]
    return "\n".join(lines) + "\n"


def life_bindings(neighbors: Dict[str, List[str]],
                  noise: Dict[str, float]):
    """The life(C) port binding — gol_p_on over the catalog's
    snapshot vocabulary (pa instead of palive; was_on = pa(C))."""
    def life(args, counts):
        c = args[0]
        if f"force({c})" in counts:
            cond = True
        else:
            live = sum(1 for v in neighbors[c]
                       if f"pa({v})" in counts)
            was_on = f"pa({c})" in counts
            cond = (2 <= live <= 3) if was_on else (live == 3)
        nz = noise[c]
        return 1.0 - nz if cond else nz

    return {"life": life}


def catalog_init(cells: List[str], alive: Set[str],
                 force: Optional[str] = None) -> Dict[str, int]:
    init = {"m(i0)": 1}
    for c in cells:
        init[f"a({c})" if c in alive else f"d({c})"] = 1
    if force is not None:
        init[f"force({force})"] = 1
    return init
