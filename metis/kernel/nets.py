"""Canonical identity — the proof-net layer (docs/nets.md).

Two objects get canonical forms, at two different levels:

PROGRAMS (the theory): `canonical()` sorts clauses/links by semantic
content — names are presentation, order is presentation — and
`program_key()` hashes the result. Two programs with the same key
denote the same measure; the key is the projection-cache key. Stage
ORDER is kept (it is the declared contract for stage_distribution
indexing); clause names are excluded (clauses with identical content
are fully symmetric in the measure, so even their relative event order
cannot matter).

RUNS (the derivations): a run is presented as an event SEQUENCE, but
its identity is the partial order induced by resource dependencies —
the occurrence net, which is the MLL proof-net of the derivation.
`run_key()` computes the Foata normal form of the Mazurkiewicz trace:
each event's level = 1 + the deepest earlier event it depends on;
levels are sorted by event content and hashed. Two interleavings of the
same concurrent run get one key — the O1 quotient of MORPHISM-REVIEW,
executable.

Independence is footprint disjointness. Stage discipline enters
uniformly by treating the active stage as an implicit resource: a
clause of stage s READS stage@s; a link WRITES stage@pre and
stage@post. Writers conflict with writers and readers; readers never
conflict with readers.

CACHING: ProjectionCache canonicalizes before projecting, so cached
projections carry the canonical event table. State-level queries
(final_distribution, stage_distribution, p_done) are always safe;
event-level queries and policies must be built against the returned
projection's `.program` (the canonical presentation). Policies are
cacheable only under an explicit `policy_key` naming their identity."""

import hashlib
from typing import List, NamedTuple, Optional, Tuple

from .filter import ForwardProjection
from .program import Program


# -- content keys (names excluded) --------------------------------------

def _items(d):
    return tuple(sorted(d.items()))


def _clause_key(c):
    return ("clause", _items(c.consume), _items(c.produce),
            _items(c.persist), float(c.weight))


def _link_key(lk):
    return ("link", lk.pre, lk.post, _items(lk.consume),
            _items(lk.produce), _items(lk.persist), float(lk.weight))


def _event_key(kind, stage, ev):
    return (_link_key(ev) if kind == "link"
            else (stage,) + _clause_key(ev))


def _digest(payload) -> str:
    return hashlib.sha256(repr(payload).encode()).hexdigest()


# -- program identity ----------------------------------------------------

def canonical(program: Program) -> Program:
    """Content-sorted presentation; semantics untouched (CHOICE is
    order-free; identical-content clauses are symmetric)."""
    return Program(
        stages={s: sorted(body, key=_clause_key)
                for s, body in program.stages.items()},
        links=sorted(program.links, key=_link_key),
        init_stage=program.init_stage,
        init=dict(program.init))


# -- the neutral canonical serialization (docs/rewrite/03) ---------------
#
# program_key hashes CANONICAL BYTES, not a python repr — the key must
# be reproducible from any implementation language (the OCaml toolchain
# shares the artifact cache). Format: line-based UTF-8, multiset items
# `atom:count` (zero counts are absence), lines content-sorted BYTEWISE,
# weights %.17g, stage order retained (it is semantics).

def _mset_line(d) -> str:
    items = sorted(f"{a}:{n}" for a, n in d.items() if n > 0)
    return ",".join(items) or "-"


def _clause_line(c) -> str:
    return (f"c {_mset_line(c.consume)} ; {_mset_line(c.produce)} ; "
            f"{_mset_line(c.persist)} ; w {c.weight:.17g}")


def _link_line(lk) -> str:
    return (f"l {lk.pre} {lk.post} ; {_mset_line(lk.consume)} ; "
            f"{_mset_line(lk.produce)} ; {_mset_line(lk.persist)} ; "
            f"w {lk.weight:.17g}")


def canonical_bytes(program: Program) -> bytes:
    """The language-neutral canonical form: what program_key hashes,
    byte-for-byte reproducible by the OCaml toolchain (ocaml/nets.ml)
    and pinned cross-language by tests/test_ocaml_parity.py."""
    lines = ["metis-canonical 1"]
    for s in program.stages:
        lines.append(f"stage {s}")
        lines += sorted(_clause_line(c) for c in program.stages[s])
    lines += sorted(_link_line(lk) for lk in program.links)
    lines.append(f"init-stage {program.init_stage}")
    lines += sorted(f"i {a}:{n}" for a, n in program.init.items()
                    if n > 0)
    return ("\n".join(lines) + "\n").encode()


def program_key(program: Program) -> str:
    return hashlib.sha256(canonical_bytes(program)).hexdigest()


# -- run identity (the proof-net quotient) -------------------------------

def _footprint(kind, stage, ev):
    writes = set(ev.consume) | set(ev.produce)
    reads = set(ev.persist)
    if kind == "link":
        writes.add(("@stage", ev.pre))
        writes.add(("@stage", ev.post))
    else:
        reads.add(("@stage", stage))
    return frozenset(reads), frozenset(writes)


def independent_events(program: Program, i: int, j: int) -> bool:
    """May events i and j (indices into program.events()) be swapped
    when adjacent — disjoint footprints, writer-vs-anything conflicts."""
    table = program.events()
    r1, w1 = _footprint(*table[i])
    r2, w2 = _footprint(*table[j])
    return not (w1 & (w2 | r2)) and not (w2 & (w1 | r1))


def foata_levels(rows) -> List[int]:
    """The Foata normal form's layering: for each position of a trace
    (rows = (kind, stage, event) triples in firing order), the causal
    level — 1 + the max level of any earlier DEPENDENT event (footprint
    conflict, writer-vs-anything). Positions sharing a level are
    pairwise independent: the concurrent "beats" of the run. This is
    the naturally induced grouping of a trace — exactly what run_key
    hashes, exposed for rendering."""
    fps = [_footprint(kind, stage, ev) for kind, stage, ev in rows]
    level: List[int] = []
    for i in range(len(rows)):
        ri, wi = fps[i]
        lvl = 0
        for j in range(i):
            rj, wj = fps[j]
            if (wi & (wj | rj)) or (wj & (wi | ri)):
                lvl = max(lvl, level[j] + 1)
        level.append(lvl)
    return level


def run_key(program: Program, events: List[int]) -> str:
    """Foata normal form of the trace, hashed. Invariant under
    permutation of adjacent independent events; discriminates different
    event multisets and different causal orders."""
    table = program.events()
    rows = [table[e] for e in events]
    keys = [_event_key(kind, stage, ev) for kind, stage, ev in rows]
    level = foata_levels(rows)
    depth = max(level, default=-1) + 1
    layers: List[List[Tuple]] = [[] for _ in range(depth)]
    for i, l in enumerate(level):
        layers[l].append(keys[i])
    payload = tuple(tuple(sorted(layer, key=repr)) for layer in layers)
    return _digest(payload)


# -- the causality graph (spec 09 P1) ----------------------------------

class CausalNode(NamedTuple):
    i: int            # occurrence number (log numbering; 0 = the base)
    name: str
    kind: str         # "init" | "clause" | "link"
    stage: str
    beat: int         # Foata level; the base is beat 0
    key: str          # repr of the content key (canonical ordering)
    episode: Optional[int] = 0   # stage episode (D5); None = a link


class CausalEdge(NamedTuple):
    src: int
    dst: int
    atoms: Tuple[str, ...]   # the conflicting atoms of this class
    kind: str                # "flow" | "read" | "order"


class CausalGraph(NamedTuple):
    nodes: Tuple[CausalNode, ...]
    edges: Tuple[CausalEdge, ...]
    frontier: Tuple[int, ...]


def _atom_str(a) -> str:
    return f"{a[0]} {a[1]}" if isinstance(a, tuple) else a


def causal_graph(rows, init, init_stage: str,
                 first: int = 1) -> CausalGraph:
    """The causality graph of a trace (spec 09 P1): a pure function of
    (init, init_stage, rows). Occurrence 0 is the base — it writes the
    initial multiset and ("@stage", init_stage), reads nothing; rows
    (firing order) are occurrences first, first+1, ... Dependence is
    the footprint conflict run_key quotients by — never a new
    relation. Edges are the Hasse diagram of the causal order (D1):
    covering pairs only, each a direct conflict, classed by
    precedence flow (produce ∩ consume) > read (produce ∩ persist) >
    order (any other write conflict: competition, stage switches).
    Beats are Foata levels; maximal nodes are the frontier. Episodes
    (D5): the runs between quiescence links, numbered from 0 (the
    base's); links write both @stage markers, so no beat spans two
    episodes; a link node itself has episode None — it stands
    between them."""
    base = {a: n for a, n in dict(init).items() if n > 0}
    fps = [(frozenset(), frozenset(base) | {("@stage", init_stage)})]
    prod, cons, pers = [set(base)], [set()], [set()]
    for kind, stage, ev in rows:
        fps.append(_footprint(kind, stage, ev))
        prod.append(set(ev.produce))
        cons.append(set(ev.consume))
        pers.append(set(ev.persist))
    n = len(fps)
    anc = [0] * n                     # strict ancestors, as bitsets
    beat = [0] * n
    parents: List[List[int]] = []
    for j in range(n):
        rj, wj = fps[j]
        direct = [i for i in range(j)
                  if (wj & (fps[i][0] | fps[i][1]))
                  or (fps[i][1] & rj)]
        reach = covered = 0
        for i in direct:
            reach |= anc[i] | (1 << i)
            covered |= anc[i]
        anc[j] = reach
        parents.append([i for i in direct if not (covered >> i) & 1])
        beat[j] = 1 + max((beat[i] for i in direct), default=-1)
    num = [0] + [first + k for k in range(n - 1)]
    nodes = [CausalNode(0, "#init", "init", init_stage, 0, "", 0)]
    ep = 0                     # D5: a link closes episode ep, opens ep+1
    for k, (kind, stage, ev) in enumerate(rows, 1):
        nodes.append(CausalNode(num[k], ev.name, kind,
                                ev.pre if kind == "link" else stage,
                                beat[k], repr(_event_key(kind, stage, ev)),
                                None if kind == "link" else ep))
        ep += kind == "link"
    edges = []
    for j in range(n):
        for i in parents[j]:
            flow = prod[i] & cons[j]
            read = prod[i] & pers[j]
            if flow:
                kind, atoms = "flow", flow
            elif read:
                kind, atoms = "read", read
            else:
                ri, wi = fps[i]
                rj, wj = fps[j]
                kind, atoms = "order", (wi & (wj | rj)) | (wj & ri)
            edges.append(CausalEdge(num[i], num[j],
                                    tuple(sorted(map(_atom_str, atoms))),
                                    kind))
    edges.sort(key=lambda e: (e.src, e.dst))
    frontier = tuple(num[j] for j in range(n)
                     if not any((anc[k] >> j) & 1 for k in range(j + 1, n)))
    return CausalGraph(tuple(nodes), tuple(edges), frontier)


def cone(g: CausalGraph, roots) -> List[int]:
    """The causal cone (spec 09 P3): roots and all their ancestors
    in g, by occurrence number. The Hasse edges' closure is the full
    causal order, so walking edges back is exact."""
    par = {nd.i: [] for nd in g.nodes}
    for e in g.edges:
        par[e.dst].append(e.src)
    seen, todo = set(), list(roots)
    while todo:
        i = todo.pop()
        if i not in seen:
            seen.add(i)
            todo.extend(par[i])
    return sorted(seen)


def live_roots(rows, init, init_stage: str, counts, stage,
               first: int = 1) -> List[int]:
    """Occurrences the CURRENT state directly depends on: a virtual
    'now' reads every present atom and the active stage; the roots
    are the occurrences whose writes conflict with that read (0 = the
    base). Atoms are indistinguishable — this is dependence on the
    multiset, not token lineage. cone(g, live_roots(...)) is bare
    `why`: what the state rests on; everything else is pruned."""
    present = {a for a, n in dict(counts).items() if n > 0}
    if stage:                         # None/"" once the run is done
        present.add(("@stage", stage))
    base = {a for a, n in dict(init).items() if n > 0}
    hits = [0] if (base | {("@stage", init_stage)}) & present else []
    for k, row in enumerate(rows):
        if _footprint(*row)[1] & present:
            hits.append(first + k)
    return hits


def canonical_graph(g: CausalGraph) -> CausalGraph:
    """The graph with occurrences renumbered by (beat, content key):
    two traces with equal run_key give byte-equal canonical graphs —
    beat-internal firing order is presentation, like clause names."""
    order = sorted(g.nodes, key=lambda nd: (nd.beat, nd.key, nd.i))
    new = {nd.i: k for k, nd in enumerate(order)}
    nodes = tuple(nd._replace(i=new[nd.i]) for nd in order)
    edges = tuple(sorted((e._replace(src=new[e.src], dst=new[e.dst])
                          for e in g.edges),
                         key=lambda e: (e.src, e.dst)))
    return CausalGraph(nodes, edges,
                       tuple(sorted(new[i] for i in g.frontier)))


# -- the projection cache ------------------------------------------------

class ProjectionCache:
    """Projections keyed by (program_key, steps, policy_key). A policy
    without a policy_key is never cached (function identity is not
    content identity)."""

    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    def project(self, program: Program, steps: int, policy=None,
                policy_key=None,
                policy_builder=None) -> ForwardProjection:
        """policy_builder, when given, is called on the CANONICAL
        program to build the policy — the index-safety rule
        (canonical() reorders events, so a policy built against the
        presentation program would misindex). On a cache hit it is
        never called at all — the point of content-keyed policies."""
        if (policy is not None or policy_builder is not None) \
                and policy_key is None:
            self.misses += 1
            canon = canonical(program)
            if policy_builder is not None:
                policy = policy_builder(canon)
            return ForwardProjection(canon, steps, policy=policy)
        key = (program_key(program), steps, policy_key)
        hit = self._store.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        canon = canonical(program)
        if policy_builder is not None:
            policy = policy_builder(canon)
        pr = ForwardProjection(canon, steps, policy=policy)
        self._store[key] = pr
        return pr
