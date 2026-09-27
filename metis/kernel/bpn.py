"""Bayesian-proof-net typing of the emitted IR — the mechanical check
behind the paper's Proposition 6 (Conjecture 7 reduction). The
factored emission (factorize.py) claims its output is MLL^B-shaped:
sites are probabilistic boxes, SSA occurrences are atomic wires,
fan-out is contraction, unread occurrences are the net's conclusions.
This module CHECKS that claim against the formal side — Ehrhard/
Faggian/Di Guardia (arXiv:2602.04045, in-repo docs/LL-PGM.pdf),
Definitions 3.1 + 3.2 and Lemma 3.1:

  box rule      ⊢ Y1-,...,Yk-, X+ : all conclusions atomic, exactly
                one positive (the main conclusion), X ∉ {Yi};
  typed graph   every edge the conclusion of at most one node and
                the premise of at most one node (contraction trees
                absorb fan-out — negative atoms only, Remark 3.1);
  pending edges every edge the conclusion of SOME node — a premise
                wired to no producer is ill-formed;
  correctness   for atomic nets, Lemma 3.1: correct iff the
                polarized orientation (downward on positive, upward
                on negative) is a DAG over the boxes;
  Bayesian      box conclusions pairwise distinct (Def 3.2.1 — our
                SSA single-writer); all net conclusions positive
                (Def 3.2.2 — pending edges are box main conclusions).

The boxes of the emitted net: one per CHOICE site (premises = its
parent occurrences, conclusion = the site variable), one per SSA
write (premises = site outcome + old occurrence, conclusion = the
new occurrence), one parentless prior box per `unknowns` lift, and
one observed constraint box per double-production guard.

The structural pass reads the SYMBOLIC emission only (no tables, no
binding calls — admission-time, same doctrine as certificate()).
`tables=True` additionally builds the CPTs and checks row
stochasticity: every site row normalized or zero (zero = stall row,
backed by the runtime Z==1 check), writes deterministic one-hot,
guards 0/1 indicators."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

TOL = 1e-9


class BpnError(ValueError):
    pass


@dataclass(frozen=True)
class BpnReport:
    boxes: int                     # box-nodes in the net
    wires: int                     # atomic edges (box conclusions)
    contractions: int              # c-nodes needed: Σ (consumers-1)
    pending: Tuple[str, ...]       # the net's conclusions Δ (unread)
    stall_rows: int                # zero-mass CPT rows (tables=True)


def bpn_boxes(fp) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """The emitted net read as MLL^B boxes: (label, main conclusion,
    negative premises). Guard constraints become observed boxes with
    a fresh conclusion (the standard BN encoding of an indicator);
    '!' cannot occur in emitted variable names, so no capture."""
    boxes = []
    for s in fp._sites:
        boxes.append((f"site {s.var}", s.var, tuple(s.parents)))
    for w in fp._writes:
        prem = (w.x,) + ((w.old,) if w.old else ())
        boxes.append((f"write {w.new}", w.new, prem))
    for a, v in sorted(fp._unknown_vars.items()):
        boxes.append((f"prior {v}", v, ()))
    for k, (x, _card, old) in enumerate(fp._guards):
        boxes.append((f"guard !g{k}", f"!g{k}", (x, old)))
    return boxes


def bpn_check(fp, tables: bool = False) -> BpnReport:
    """Raise BpnError unless the emitted net is a Bayesian proof-net
    (EFD Def 3.1 + 3.2); return the net's shape."""
    boxes = bpn_boxes(fp)

    # box rule: the main conclusion is not among the premises
    for label, concl, prem in boxes:
        if concl in prem:
            raise BpnError(f"{label}: its own premise (box rule "
                           f"requires X ∉ {{Yi}})")

    # Def 3.2.1 + typed graph: conclusions pairwise distinct
    producer: Dict[str, str] = {}
    for label, concl, _prem in boxes:
        if concl in producer:
            raise BpnError(f"'{concl}' concluded twice: "
                           f"{producer[concl]} and {label} — the "
                           f"emission is not SSA")
        producer[concl] = label

    # Def 3.1.2 pending edges: every premise is some box's conclusion
    for label, _concl, prem in boxes:
        for p in prem:
            if p not in producer:
                raise BpnError(f"{label}: premise '{p}' is the "
                               f"conclusion of no box")

    # every emitted variable is a wire of the net (drift guard)
    for v in fp._cards:
        if v not in producer:
            raise BpnError(f"variable '{v}' has a cardinality but "
                           f"no producing box")

    # Lemma 3.1: the polarized orientation is a DAG over the boxes
    consumers: Dict[str, int] = {c: 0 for c in producer}
    out: Dict[str, List[str]] = {c: [] for c in producer}
    indeg: Dict[str, int] = {c: 0 for c in producer}
    for _label, concl, prem in boxes:
        for p in prem:
            out[p].append(concl)
            indeg[concl] += 1
            consumers[p] += 1
    ready = [c for c, d in indeg.items() if d == 0]
    seen = 0
    while ready:
        c = ready.pop()
        seen += 1
        for d in out[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
    if seen != len(boxes):
        cyc = sorted(c for c, d in indeg.items() if d > 0)
        raise BpnError(f"polarized orientation is not a DAG "
                       f"(Lemma 3.1): cycle through {cyc}")

    # Def 3.2.2 holds by construction once the above do: the pending
    # edges are exactly the unread box conclusions — positive atoms.
    pending = tuple(sorted(c for c, n in consumers.items() if n == 0))

    stall_rows = 0
    if tables:
        stall_rows = _check_tables(fp)

    return BpnReport(
        boxes=len(boxes),
        wires=len(producer),
        contractions=sum(n - 1 for n in consumers.values() if n > 1),
        pending=pending,
        stall_rows=stall_rows)


def _check_tables(fp) -> int:
    """CPT semantics: site rows normalized or zero-mass (stall),
    write rows one-hot deterministic, guard entries 0/1. Builds
    tables — evaluation-time, calls bindings."""
    import numpy as np
    fs = fp._build()
    ns, nw = len(fp._sites), len(fp._writes)
    stall = 0
    for s, f in zip(fp._sites, fs[:ns]):
        rows = f.table.reshape(-1, f.table.shape[-1])
        for i, row in enumerate(rows):
            tot = row.sum()
            if abs(tot - 1.0) <= TOL:
                continue
            if abs(tot) <= TOL:
                stall += 1
                continue
            raise BpnError(f"site {s.var}: row {i} has mass "
                           f"{tot:.6g} — neither a distribution nor "
                           f"a stall row")
    for w, f in zip(fp._writes, fs[ns:ns + nw]):
        rows = f.table.reshape(-1, 2)
        if not np.all(np.isin(rows, (0.0, 1.0))) or \
                not np.allclose(rows.sum(axis=1), 1.0, atol=TOL):
            raise BpnError(f"write {w.new}: not deterministic "
                           f"one-hot")
    for (x, _c, old), f in zip(fp._guards, fs[ns + nw:]):
        if not np.all(np.isin(f.table, (0.0, 1.0))):
            raise BpnError(f"guard ({x},{old}): entries outside "
                           f"{{0,1}}")
    return stall
