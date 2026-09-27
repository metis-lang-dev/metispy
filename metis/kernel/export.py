"""BN/factor-graph export — IR item 4 (docs/ir-plan.md): dump the
emitted occurrence net as a standard factor graph for EXTERNAL tools
— the EFD direction (their compiler goes BN -> proof net; ours goes
program -> net -> BN), and free validation: any UAI-speaking solver
must reproduce the kernel's marginals.

Format: UAI MARKOV (inference-competition standard; pgmpy, libDAI,
merlin, toulbar2 all read it). Variables are the net's site-outcome
and SSA atom-occurrence variables; a sidecar name list keeps the
mapping (the UAI format itself is index-only). Evidence (the
`unknowns` point factors) is baked into the export so the file is
self-contained. `to_dot` renders the same net for humans."""

from typing import Dict, List, Optional, Tuple

from .factored import Factor
from .factorize import FactoredProgram


def _factors(fp: FactoredProgram, init=None) -> List[Factor]:
    """The complete factor list (tables + evidence points), built —
    and Z-checked — through the same path evaluation uses."""
    fp._calibration(init)
    points = [Factor.point(fp._unknown_vars[a], val)
              for a, val in sorted(fp._cal_key)]
    return list(fp._tables) + points


def to_uai(fp: FactoredProgram,
           init=None) -> Tuple[str, List[str]]:
    """UAI MARKOV text + the index->variable-name sidecar. Table
    values are row-major with the LAST scope variable fastest — the
    UAI convention, which is exactly numpy's reshape(-1)."""
    fs = _factors(fp, init)
    names = sorted(fp._cards)
    idx = {v: i for i, v in enumerate(names)}
    lines = ["MARKOV", str(len(names)),
             " ".join(str(fp._cards[v]) for v in names),
             str(len(fs))]
    for f in fs:
        lines.append(f"{len(f.vars)} "
                     + " ".join(str(idx[v]) for v in f.vars))
    for f in fs:
        flat = f.table.reshape(-1)
        lines.append(str(flat.size))
        lines.append(" ".join(f"{x:.17g}" for x in flat))
    return "\n".join(lines) + "\n", names


def to_dot(fp: FactoredProgram, init=None) -> str:
    """The net as a bipartite factor graph (GraphViz): ellipses are
    variables (site outcomes, SSA atom occurrences), boxes are
    factors."""
    fs = _factors(fp, init)
    out = ["graph metis_net {", "  rankdir=LR;",
           "  node [fontsize=9];"]
    for v in sorted(fp._cards):
        out.append(f'  "{v}" [shape=ellipse];')
    for i, f in enumerate(fs):
        out.append(f'  f{i} [shape=box, label="f{i}"];')
        for v in f.vars:
            out.append(f'  f{i} -- "{v}";')
    out.append("}")
    return "\n".join(out) + "\n"
