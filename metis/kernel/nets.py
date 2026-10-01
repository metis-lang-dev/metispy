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
from typing import List, Tuple

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
