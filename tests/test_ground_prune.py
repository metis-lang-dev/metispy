"""The grounder's pruned enumeration and the REPL's candidate index are optimizations of the SAME
answers (alf-demo 3b timing: a planner instance checked 440k guards one by one to keep 28k clauses,
and weighed all 28k events every beat): every program grounds to the same clauses in the same order
as filtering the full product of the types, and the firing tier at every step of a walk is the one a
full scan of the event table gives."""

import itertools
import random
from pathlib import Path

from metis.kernel import ground
from metis.kernel.weight import choice_weight
from metis.lang import compile_catalog, parse

CATALOG = Path(__file__).resolve().parents[2] / "metis-catalog/domains"

SRC = """catalog prune 1.
layers (dyn).
stages (act).
type thing {a b c d}.
type place {p q r}.
namespace w produce (dyn).
bwd at(thing, place).
bwd near(place, place).
bwd big(thing).
bwd free(thing).
pred go(thing, place) : w.
pred moved(thing, place) : w.
pred tok : w.
at(a, p). at(b, q). at(c, q). at(d, r).
near(p, q). near(q, r). near(r, p).
big(X) :- at(X, q).
free(X).
stage act {
  %% a guard per variable, a binary guard and a distinct pair
  move [dyn] : go(T, P) * $at(T, P0) * $near(P0, P) * P0 <> P -o moved(T, P).
  %% a unary guard by a Horn rule
  grow [dyn] : go(T, P) * $big(T) -o moved(T, P).
  %% a quantified fact: no set of constants answers it (HornNonground), each constant is checked
  loose [dyn] : go(T, P) * $free(T) -o moved(T, P).
  %% no guard
  any [dyn] : go(T, P) -o moved(T, P).
  %% a guard with no variable
  closed [dyn] : tok * $near(p, q) -o tok.
}
"""


def _reference(vars_, distinct, guards, sig, horn):
    """The grounder before the pruning: the full product, each combination filtered."""
    names = list(vars_)
    for combo in itertools.product(*[sig.types[vars_[v]] for v in names]):
        s = dict(zip(names, combo))
        if any(s[x] == s[y] for x, y in distinct):
            continue
        if guards and not all(horn.derivable((g[0], tuple(s.get(a, a) for a in g[1]))) for g in guards):
            continue
        yield names, s


def _clauses(cat, subs):
    old = ground._substitutions
    ground._substitutions = subs
    try:
        prog = cat.ground({})
    finally:
        ground._substitutions = old
    return [(st, c.name, sorted(c.consume.items()), sorted(c.produce.items()), sorted(c.persist.items()))
            for st, cl in prog.stages.items() for c in cl]


def test_the_pruned_grounding_is_the_full_products():
    cats = [compile_catalog(parse(SRC)).catalog]
    for p in sorted(CATALOG.glob("*/*.llp"))[:40]:
        try:
            cat = compile_catalog(parse(p.read_text(), file=str(p))).catalog
        except Exception:  # noqa: BLE001 — includes, packs, ports: not a standalone program
            continue
        if not cat.abstract:            # a lifted file (spec 12) grounds through its instance
            cats.append(cat)
    assert len(cats) > 5
    for cat in cats:
        assert _clauses(cat, ground._substitutions) == _clauses(cat, _reference)
    names = [n for _, n, *_ in _clauses(cats[0], ground._substitutions)]
    assert "move[T=a,P=q,P0=p]" in names and "grow[T=b,P=p]" in names and "closed" in names
    assert sum(n.startswith("loose[") for n in names) == 12


def test_the_candidate_index_is_the_full_scan(tmp_path):
    from metis.lang.repl import Repl, DONE
    src = tmp_path / "prune.llp"
    src.write_text(SRC + "#init go(a, q) 1\n#init go(b, p) 2\n#init tok 1\n#steps 6\n#seeds 1\n")
    r = Repl(str(src), seed=1)
    rng = random.Random(3)
    for _ in range(8):
        full = [] if r.stage == DONE else [
            (i, w) for i, w in ((i, choice_weight(ev, r.counts)) for i, (k, s, ev) in enumerate(r.events_tbl)
                                if k == "clause" and s == r.stage) if w > 0]
        assert r._candidates() == full
        if not full:
            break
        r._fire(rng.choices([i for i, _ in full], weights=[w for _, w in full])[0], ws=full)


def test_the_candidate_index_follows_a_new_table(tmp_path):
    """The index is keyed on the events table (its identity and its length): a reload's new table, or
    a table that grew, is indexed again — the tier is the full scan's after each."""
    from metis.lang.repl import Repl
    src = tmp_path / "prune.llp"
    src.write_text(SRC + "#init go(a, q) 1\n#steps 6\n#seeds 1\n")
    r = Repl(str(src), seed=1)
    r._candidates()
    old = r.events_tbl
    src.write_text(SRC + "#init go(a, q) 1\n#init go(c, p) 1\n#steps 6\n#seeds 1\n")
    r._load()
    tbl = r.events_tbl
    full = [(i, w) for i, w in ((i, choice_weight(ev, r.counts)) for i, (k, s, ev) in enumerate(tbl)
                                if k == "clause" and s == r.stage) if w > 0]
    assert tbl is not old and r._candidates() == full and r.__dict__["_cidx"][0] is tbl   # the reload's table, indexed
    tbl.append(("clause", r.stage, tbl[full[0][0]][2]))          # grown in place: indexed again
    assert r._candidates()[-1][0] == len(tbl) - 1
