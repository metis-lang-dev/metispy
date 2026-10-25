"""The one-tick instrument (metis-lang spec/16 §6, §7; spec16 s-b12d90f2 step 3): metisc --evaluate in metispy.

The number is the forward projection's (filter.py: spec 06 §4's step, apply_event, transition_count, the canonical
event order) started from the WORLD MIXTURE instead of one state, the outcome absorbing, in exact rationals:

    μ₀     = Σ_w P(w) · δ((init stage, init ⊎ w))    every belief group's draw a world (§3)
    step   = tier 0 the stage's enabled clauses, W = prior × φ × port factor, normalised; tier 1 the links; DONE
    result = the mass that reached the outcome within h (or holds at h)

with R1 (a group no event touches and no query names is dropped), R2 (a group is drawn at the first state whose
stage's events touch it; the query's groups at step 0), R3 (absorption), R4 (merging), and the state budget as a CUT:
states expanded in canonical order (the state's key, as metisc's), the mass that cannot be expanded within the budget
set aside, the answer [lo, hi] = [HIT, HIT + CUT].

Numbers enter as the exact decimal of their binary64's shortest round-trip representation, Fraction(repr(x))
(§6.4, as ruled): 0.7 is 7/10, so a distribution written in decimals sums to exactly 1. The listing is metisc's,
line for line: N belief-read notes, V lines, L level atoms, X failed expectations; C cost lines with cost=True.

    run(doc, prog, init, directives, brute=False) -> Result(lines, costs, mismatches)    raises Refused(code, text)
    python -m metis.kernel.evaluate FILE.llp|CASE.yaml [--brute] [--cost]                (metisc --evaluate's exit codes)

`--brute` is the referee: every world drawn up front, every event path enumerated, no merging, no budget."""

import math
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .filter import DONE
from .weight import transition_count

DEFAULT_BUDGET = 500_000


class Refused(Exception):
    """A case the instrument does not evaluate: (code, text), metisc's `refused <code>: <text>`."""

    def __init__(self, code: str, text: str):
        self.code, self.text = code, text
        super().__init__("refused %s: %s" % (code, text))


@dataclass
class Result:
    lines: List[str]
    costs: List[str]
    mismatches: int


def _refuse(code, text):
    raise Refused(code, text)


ONE, ZERO = Fraction(1), Fraction(0)


def split_atom(s: str) -> Tuple[str, List[str]]:
    """'on(wolf,near)' -> ('on', ['wolf', 'near'])."""
    i = s.find("(")
    if i < 0:
        return s, []
    return s[:i], [a.strip() for a in s[i + 1:-1].split(",")]


def render(p: str, args) -> str:
    return "%s(%s)" % (p, ",".join(args)) if args else p


def dec(r: Fraction) -> str:
    """The correctly rounded binary64 of a rational, %.17g."""
    return "%.17g" % float(r)


def rat(r: Fraction) -> str:
    return "%d/%d" % (r.numerator, r.denominator)


def number(what: str, s) -> Fraction:
    """A number as its binary64's shortest round-trip decimal, exactly (§6.4): Fraction(repr(float(s)))."""
    try:
        f = float(s)
    except (TypeError, ValueError):
        _refuse("port-value", "%s: %s is no number" % (what, s))
    if not math.isfinite(f) or f < 0:
        _refuse("port-value", "%s: %s is no number >= 0" % (what, s))
    return Fraction(repr(f))


# -- the case's directives (spec/16 §5) -----------------------------------

@dataclass
class Belief:
    pred: str
    mode: str                     # "indep" | "one" | "lone"
    var: Optional[str]
    port: str


@dataclass
class Query:
    name: str
    kind: str                     # reach | risk
    conj: List[Tuple[str, List[str]]]
    within: bool
    h: int
    budget: int


@dataclass
class Level:
    pred: str
    query: Optional[str]          # From_query
    port: Optional[str]           # From_belief
    at: List[Fraction]
    names: List[str]


def parse(dirs: List[List[str]]):
    beliefs, tbl, queries, expects, levels = [], [], [], [], []

    def bad(toks):
        raise ValueError("bad instrument directive: " + " ".join(toks))
    for toks in dirs:
        if len(toks) == 4 and toks[0] == "belief" and toks[2] == "by":
            beliefs.append(Belief(toks[1], "indep", None, toks[3]))
        elif len(toks) == 6 and toks[0] == "belief" and toks[2] in ("one", "lone") and toks[4] == "by":
            beliefs.append(Belief(toks[1], toks[2], toks[3], toks[5]))
        elif len(toks) == 3 and toks[0] == "port":
            name, args = split_atom(toks[1])
            try:
                float(toks[2])
            except ValueError:            # a binding that computes: the host's function form (§10)
                _refuse("port-function", "port %s bound to %s: a binding that computes is the host's function form "
                        "(spec/16 §10, later); a case binds numbers or a table of them" % (toks[1], toks[2]))
            tbl.append((name, tuple(args) if "(" in toks[1] else None, number("port " + toks[1], toks[2])))
        elif len(toks) >= 3 and toks[0] == "evaluate" and toks[2] in ("reach", "risk"):
            name, kind, rest = toks[1], toks[2], toks[3:]
            if any(q.name == name for q in queries):
                raise ValueError("evaluate: the query %s twice" % name)
            conj, i = [], 0
            while i < len(rest) and rest[i] not in ("within", "at"):
                if rest[i] != "*":
                    conj.append(split_atom(rest[i]))
                i += 1
            if i + 1 >= len(rest) or not conj:
                bad(toks)
            tail = rest[i + 2:]
            if tail == []:
                budget = DEFAULT_BUDGET
            elif len(tail) == 2 and tail[0] == "budget":
                budget = int(tail[1])
            else:
                bad(toks)
            queries.append(Query(name, kind, conj, rest[i] == "within", int(rest[i + 1]), budget))
        elif len(toks) == 3 and toks[0] == "expect":
            expects.append((toks[1], toks[2]))
        elif len(toks) >= 3 and toks[0] == "level" and toks[2] == "from":
            pred, rest = toks[1], toks[3:]
            if rest[:1] == ["belief"] and len(rest) >= 2:
                src, rest = (None, rest[1]), rest[2:]
            elif rest:
                src, rest = (rest[0], None), rest[1:]
            else:
                bad(toks)
            if rest[:1] != ["at"] or "names" not in rest:
                bad(toks)
            k = rest.index("names")
            levels.append(Level(pred, src[0], src[1], [number("level " + pred, t) for t in rest[1:k]], rest[k + 1:]))
        else:
            bad(toks)
    return beliefs, tbl, queries, expects, levels


# -- states -----------------------------------------------------------------

@dataclass
class St:
    stage: str
    counts: Dict[str, int]
    undrawn: Tuple[int, ...]

    def key(self) -> str:
        """metisc's key, byte for byte: the cut's canonical order is the sort of these strings."""
        return "%s|%s|%s" % (self.stage, "".join("%s:%d " % (a, self.counts[a]) for a in sorted(self.counts)),
                             "".join("%d " % g for g in self.undrawn))


def matches(pat, a: str) -> bool:
    p, args = pat
    p2, args2 = split_atom(a)
    return p == p2 and len(args) == len(args2) and all(x == "_" or x == y for x, y in zip(args, args2))


def holds(counts: Dict[str, int], conj) -> bool:
    for pat in conj:
        p, args = pat
        if "_" not in args:
            if counts.get(render(p, args), 0) <= 0:
                return False
        elif not any(n > 0 and matches(pat, a) for a, n in counts.items()):
            return False
    return True


@dataclass
class Group:
    name: str
    worlds: List[Tuple[List[str], Fraction]]     # the atoms present, P > 0
    atoms: List[str]                             # its touched atoms
    stages: List[str]                            # stages whose events touch them


def _product(pairs):
    """[(formal, [consts])] -> every assignment [(formal, const)], the first formal outermost."""
    if not pairs:
        return [[]]
    (f, cs), rest = pairs[0], pairs[1:]
    tails = _product(rest)
    return [[(f, c)] + t for c in cs for t in tails]


def run(doc, prog, init: Dict[str, int], dirs: List[List[str]], brute: bool = False) -> Result:
    """The instrument over a compiled doc (metis.lang CatalogDoc), its ground program and the case's init."""
    from metis.lang.compiler import ground_reads
    from metis.lang import ast
    from .refsample import canonical_events
    beliefs, tbl, queries, expects, levels = parse(dirs)
    cat, ports = doc.catalog, doc.ports
    sigs = {d.name: list(d.args) for d in (doc.source.decls if doc.source else ()) if isinstance(d, ast.PredDecl)}

    def consts(ty):
        return list(cat.sig.types.get(ty, []))
    horn = cat.horn_db()
    for n, key, _ in tbl:
        if n not in ports.weights:
            _refuse("port-unknown", "port %s: not a weight port of the program (%s)" % (n, ", ".join(ports.weights)))
        if key is not None and len(key) != len(ports.weights[n]):
            _refuse("port-unknown", "port %s: %d arguments, %s takes %d" % (render(n, key), len(key), n, len(ports.weights[n])))

    def port_value(port, args):
        args = tuple(args)
        for n, a, v in tbl:
            if n == port and a == args:
                return v
        for n, a, v in tbl:
            if n == port and a is None:
                return v
        _refuse("port-unbound", "weight port %s is unbound at %s: the case binds it (#port %s VALUE, or a value "
                "per argument tuple)" % (port, render(port, args), port))
    # the events, their frozen weights: prior × the port factor
    rows = canonical_events(prog)
    w0 = []
    for kind, stage, ev in rows:
        w = number("rule " + ev.name, "%.17g" % ev.weight)
        if kind == "clause":
            base = ev.name.split("[", 1)[0]
            ref = ports.weight_refs.get(base)
            if ref is not None:
                tag = ev.name.split("[", 1)[1].rstrip("]") if "[" in ev.name else ""
                subst = dict(kv.split("=") for kv in tag.split(",")) if tag else {}
                args = [subst.get(a, a) for a in ref.args]
                v = port_value(ref.factor, args)
                if ref.complement:
                    if v > ONE:
                        _refuse("port-value", "~%s: %s above 1" % (ref.factor, render(ref.factor, args)))
                    v = ONE - v
                w = w * v
        w0.append(w)
    tiers: Dict[Tuple[str, str], List[int]] = {}
    for i, (kind, stage, _) in enumerate(rows):
        tiers.setdefault((kind, stage), []).append(i)
    touched = set()
    for _, _, ev in rows:
        touched.update(ev.consume, ev.produce, ev.persist)

    def mentioned(a):
        return any(matches(c, a) for q in queries for c in q.conj)
    # the beliefs: checked, then their groups
    bpreds = [b.pred for b in beliefs]
    for a in init:
        if split_atom(a)[0] in bpreds:
            _refuse("belief-init", "init %s: a belief atom — the instrument draws it from its port, the case's "
                    "init must not list it" % a)
    groups: List[Group] = []
    for b in beliefs:
        p = b.pred
        if p in cat.bwd:
            _refuse("belief-bwd", "belief %s: a bwd pred is a guard, eliminated at grounding; every world shares "
                    "one ground program, so a belief pred is a resource pred ($-read it)" % p)
        if p not in cat.preds:
            _refuse("belief-port", "belief %s: not a pred of the program" % p)
        formals = ports.weights.get(b.port)
        if formals is None:
            _refuse("belief-port", "belief %s: %s is not a weight port of the program" % (p, b.port))
        formals = list(formals)
        pats = ports.reads.get(b.port)
        if not pats or len(pats) != 1 or pats[0].atom.pred != p:
            _refuse("belief-port", "belief %s: port %s must read exactly one pattern over %s (reads {%s(…)})"
                    % (p, b.port, p, p))
        pat = pats[0]
        vs = []
        for t in pat.atom.terms:
            if t.kind != "var":
                _refuse("belief-port", "belief %s: %s's read pattern has the constant %s; its arguments are the "
                        "port's formals" % (p, b.port, t.value))
            vs.append(t.value)
        if sorted(vs) != sorted(formals) or len(set(vs)) != len(vs):
            _refuse("belief-port", "belief %s: %s's read pattern %s must name each formal of %s(%s) once"
                    % (p, b.port, str(pat.atom), b.port, ", ".join(formals)))
        types = sigs.get(p, [])

        def ftype(f, vs=vs, types=types):
            return types[vs.index(f) if f in vs else 0]

        def atom_of(asg, vs=vs, p=p):
            d = dict(asg)
            return render(p, [d[v] for v in vs])

        def args_of(asg, formals=formals):
            d = dict(asg)
            return [d[f] for f in formals]

        def in_support(asg, pat=pat, port=b.port):
            return not pat.guards or atom_of(asg) in ground_reads(ports, port, tuple(args_of(asg)), horn)

        def rel(a):
            return a in touched or mentioned(a)

        def stages_of(atoms):
            return sorted({stage for _, stage, ev in rows
                           if any(a in atoms for a in (*ev.consume, *ev.produce, *ev.persist))})

        def mk(name, worlds, atoms):
            return Group(name, [(w, q) for w, q in worlds if q != 0], atoms, stages_of(atoms))
        if b.mode == "indep":
            for asg in _product([(f, consts(ftype(f))) for f in formals]):
                if not in_support(asg):
                    continue
                a = atom_of(asg)
                q = port_value(b.port, args_of(asg))
                if q > ONE:
                    _refuse("belief-sum", "belief %s: %s = %s (%s) is above 1" % (p, render(b.port, args_of(asg)), rat(q), dec(q)))
                if rel(a):
                    groups.append(mk(a, [([a], q), ([], ONE - q)], [a]))
        else:
            v = b.var
            if v not in formals:
                _refuse("belief-port", "belief %s: %s is not a formal of %s(%s)" % (p, v, b.port, ", ".join(formals)))
            others = [f for f in formals if f != v]
            for kasg in _product([(f, consts(ftype(f))) for f in others]):
                members = [m for m in ([(v, c)] + kasg for c in consts(ftype(v))) if in_support(m)]
                ps = [(atom_of(m), port_value(b.port, args_of(m))) for m in members]
                total = sum((q for _, q in ps), ZERO)
                kd = dict(kasg)
                gname = render(p, ["_" if x == v else kd[x] for x in vs])
                if b.mode == "one" and total != ONE:
                    _refuse("belief-sum", "belief %s one %s: the group %s sums to %s (%s), not 1" % (p, v, gname, rat(total), dec(total)))
                if b.mode == "lone" and total > ONE:
                    _refuse("belief-sum", "belief %s lone %s: the group %s sums to %s (%s), above 1" % (p, v, gname, rat(total), dec(total)))
                live = [(a, q) for a, q in ps if rel(a)]
                if not live:                                   # R1: weakening
                    continue
                rest = ONE - sum((q for _, q in live), ZERO)
                groups.append(mk(gname, [([a], q) for a, q in live] + [([], rest)], [a for a, _ in live]))
    ng = len(groups)
    worlds = 1
    for g in groups:
        worlds *= len(g.worlds)

    # the choice at a state: spec 06 §4's tiers, exactly
    def choice(st):
        if st.stage == DONE:
            return [(None, ONE)]

        def weigh(kind):
            out = []
            for i in tiers.get((kind, st.stage), ()):
                n = transition_count(rows[i][2], st.counts)
                if n:
                    w = w0[i] * n
                    if w:
                        out.append((i, w))
            return out
        ws = weigh("clause") or weigh("link")
        if not ws:
            return [(None, ONE)]
        tot = sum((w for _, w in ws), ZERO)
        return [(i, w / tot) for i, w in ws]

    def apply(st, e):
        if e is None:
            return St(DONE, st.counts, st.undrawn)
        kind, _, ev = rows[e]
        c = dict(st.counts)
        for a, n in ev.consume.items():
            m = c.get(a, 0) - n
            if m > 0:
                c[a] = m
            else:
                c.pop(a, None)
        for a, n in ev.produce.items():
            c[a] = c.get(a, 0) + n
        return St(ev.post if kind == "link" and ev.post else st.stage, c, st.undrawn)

    # R2: a group is drawn where its presence could first matter
    def settle(must, st, m):
        gid = next((g for g in st.undrawn if must(groups[g], st)), None)
        if gid is None:
            return [(st, m)]
        undrawn = tuple(g for g in st.undrawn if g != gid)
        out = []
        for atoms, q in groups[gid].worlds:
            c = dict(st.counts)
            for a in atoms:
                c[a] = c.get(a, 0) + 1
            out += settle(must, St(st.stage, c, undrawn), m * q)
        return out
    st0 = St(prog.init_stage, {a: n for a, n in init.items() if n > 0}, tuple(range(ng)))
    maxbits = [0]

    def note(m):
        maxbits[0] = max(maxbits[0], m.denominator.bit_length())

    def instrument(q):
        def qg(g):
            return any(matches(c, a) for a in g.atoms for c in q.conj)

        def must(g, st):
            return qg(g) or st.stage in g.stages
        hit, cut = [ZERO], [ZERO]

        def admit(tbl_, sm):
            st, m = sm
            note(m)
            if q.within and holds(st.counts, q.conj):
                hit[0] += m
            elif q.within and st.stage == DONE:
                pass                                           # a miss, frozen
            else:
                k = st.key()
                if k in tbl_:
                    tbl_[k] = (tbl_[k][0], tbl_[k][1] + m)
                else:
                    tbl_[k] = (st, m)
        cur: Dict[str, Tuple[St, Fraction]] = {}
        for sm in settle(must, st0, ONE):
            admit(cur, sm)
        retained = len(cur)
        t = 0
        while t < q.h and cur:
            t += 1
            out: Dict[str, Tuple[St, Fraction]] = {}
            for _, (st, m) in sorted(cur.items()):
                if retained + len(out) >= q.budget:
                    cut[0] += m
                else:
                    for e, pe in choice(st):
                        for sm in settle(must, apply(st, e), m * pe):
                            admit(out, sm)
            retained += len(out)
            cur = out
        lo = hit[0] if q.within else sum((m for st, m in cur.values() if holds(st.counts, q.conj)), ZERO)
        return lo, lo + cut[0], cut[0], retained

    def referee(q):
        hit, paths = [ZERO], [0]

        def dfs(st, m, t):
            if q.within:
                if holds(st.counts, q.conj):
                    hit[0] += m
                    paths[0] += 1
                elif t == q.h or st.stage == DONE:
                    paths[0] += 1
                else:
                    for e, pe in choice(st):
                        dfs(apply(st, e), m * pe, t + 1)
            else:
                if t == q.h or st.stage == DONE:
                    paths[0] += 1
                    if holds(st.counts, q.conj):
                        hit[0] += m
                else:
                    for e, pe in choice(st):
                        dfs(apply(st, e), m * pe, t + 1)
        for st, m in settle(lambda g, st: True, st0, ONE):
            dfs(st, m, 0)
        return hit[0], paths[0]

    # the listing: the belief-read notes first, in catalog order
    notes: List[str] = []
    for e in cat.entries:
        if e.kind not in ("schema", "link"):
            continue
        pl = e.payload
        reads = {pt[0] for pt in (*pl.consume, *pl.persist, *pl.guards)}
        rule = e.name.rsplit("/", 1)[-1]
        for p in bpreds:
            line = "N belief-read %s %s" % (rule, p)
            if p in reads and line not in notes:
                notes.append(line)
    results: Dict[str, Tuple[Fraction, Fraction, Fraction]] = {}
    costs, vlines = [], []
    for q in queries:
        pat = " * ".join(render(p, a) for p, a in q.conj)
        head = "V %s %s %s %s %d" % (q.name, q.kind, pat, "within" if q.within else "at", q.h)
        maxbits[0] = 0
        t0 = time.process_time()
        if brute:
            v, paths = referee(q)
            lo, hi, cut = v, v, ZERO
            line = "%s = %s ~ %s worlds %d paths %d" % (head, rat(v), dec(v), worlds, paths)
        else:
            lo, hi, cut, n = instrument(q)
            if cut == 0:
                line = "%s = %s ~ %s worlds %d states %d" % (head, rat(lo), dec(lo), worlds, n)
            else:
                line = "%s in [%s, %s] ~ [%s, %s] cut %s worlds %d states %d" % (
                    head, rat(lo), rat(hi), dec(lo), dec(hi), dec(cut), worlds, n)
        ms = (time.process_time() - t0) * 1000.0
        costs.append("C %s h %d groups %d worlds %d bits %d digits %d ms %.1f"
                     % (q.name, q.h, ng, worlds, maxbits[0], len(str(hi.denominator)), ms))
        results[q.name] = (lo, hi, cut)
        vlines.append(line)
    # expectations
    mism, xlines = 0, []
    for name, v in expects:
        if name not in results:
            mism += 1
            xlines.append("X %s: no such query" % name)
            continue
        lo, hi, _ = results[name]
        if "/" in v:
            ok = lo == hi and rat(lo) == v
        else:
            try:
                f = float(v)
                ok = float(lo) <= f <= float(hi)
            except ValueError:
                ok = False
        if not ok:
            mism += 1
            xlines.append("X %s expected %s got %s" % (name, v, dec(lo) if lo == hi else "[%s, %s]" % (dec(lo), dec(hi))))

    # level ports (§4)
    def level_of(lv, r):
        return lv.names[sum(1 for t in lv.at if t <= r)]
    llines = []
    for lv in levels:
        types = sigs.get(lv.pred)
        if types is None:
            _refuse("level-type", "level %s: not a pred of the program" % lv.pred)
        if not (types and consts(types[0]) == lv.names):
            _refuse("level-type", "level %s: its first argument's type must have the constants %s, in order"
                    % (lv.pred, " ".join(lv.names)))
        if len(lv.names) != len(lv.at) + 1:
            _refuse("level-type", "level %s: %d names for %d thresholds" % (lv.pred, len(lv.names), len(lv.at)))
        if any(a >= b for a, b in zip(lv.at, lv.at[1:])):
            _refuse("level-type", "level %s: thresholds not increasing" % lv.pred)
        if lv.query is not None:
            if lv.query not in results:
                _refuse("level-type", "level %s: no query %s" % (lv.pred, lv.query))
            lo, hi, _ = results[lv.query]
            a, b = level_of(lv, lo), level_of(lv, hi)
            llines.append("L %s" % render(lv.pred, [a]) if a == b else "L %s cut" % render(lv.pred, [a + ".." + b]))
        else:
            if not any(b.port == lv.port for b in beliefs):
                _refuse("level-type", "level %s: no belief by %s" % (lv.pred, lv.port))
            formals = list(ports.weights[lv.port])
            if len(types) != 1 + len(formals):
                _refuse("level-type", "level %s: %d arguments, the level and %s's %d formals"
                        % (lv.pred, len(types), lv.port, len(formals)))
            pat = ports.reads[lv.port][0]
            vs = [t.value for t in pat.atom.terms]
            for g in groups:
                if not any(split_atom(a)[0] == pat.atom.pred for a in g.atoms):
                    continue
                for atoms, q in g.worlds:
                    if len(atoms) == 1 and split_atom(atoms[0])[0] == pat.atom.pred:
                        asg = dict(zip(vs, split_atom(atoms[0])[1]))
                        llines.append("L %s" % render(lv.pred, [level_of(lv, q)] + [asg[f] for f in formals]))
    return Result(notes + vlines + llines + xlines, costs, mism)


# -- the case and manifest YAML (spec/16 §5): metisc's subset, byte for byte -----------------------------------

class YamlError(ValueError):
    pass


def _flow(file, lno, s):
    """A flow value: a bare or quoted scalar, [a, b] or {k: v}, nested; an atom's parentheses hold its commas.
    → str | list | list of (key, value) pairs (a map, in order)."""
    n = len(s)

    def err(m):
        raise YamlError("%s:%d: %s" % (file, lno, m))

    def ws(i):
        while i < n and s[i] in " \t":
            i += 1
        return i

    def scalar(i, stops):
        i = ws(i)
        if i < n and s[i] == '"':
            b, j = [], i + 1
            while True:
                if j >= n:
                    err("unterminated quote")
                if s[j] == "\\" and j + 1 < n:
                    b.append(s[j + 1])
                    j += 2
                elif s[j] == '"':
                    return "".join(b), j + 1
                else:
                    b.append(s[j])
                    j += 1
        j, depth = i, 0
        while j < n and (depth > 0 or s[j] not in stops):
            if s[j] == "(":
                depth += 1
            elif s[j] == ")" and depth > 0:
                depth -= 1
            j += 1
        return s[i:j].strip(), j

    def value(i):
        i = ws(i)
        if i < n and s[i] == "[":
            return lst(i + 1, [])
        if i < n and s[i] == "{":
            return mp(i + 1, [])
        return scalar(i, ",]}")

    def lst(i, acc):
        i = ws(i)
        if i < n and s[i] == "]":
            return acc, i + 1
        v, j = value(i)
        j = ws(j)
        if j < n and s[j] == ",":
            return lst(j + 1, acc + [v])
        if j < n and s[j] == "]":
            return acc + [v], j + 1
        err("expected , or ] in a flow list")

    def mp(i, acc):
        i = ws(i)
        if i < n and s[i] == "}":
            return _Map(acc), i + 1
        k, j = scalar(i, ":,}")
        if not (j < n and s[j] == ":"):
            err("expected key: value in a flow map")
        v, j = value(j + 1)
        j = ws(j)
        if j < n and s[j] == ",":
            return mp(j + 1, acc + [(k, v)])
        if j < n and s[j] == "}":
            return _Map(acc + [(k, v)]), j + 1
        err("expected , or } in a flow map")
    v, j = value(0)
    if ws(j) != n:
        err("trailing text after a flow value")
    return v


class _Map(list):
    """A YAML map: (key, value) pairs in order."""

    def get(self, k, default=None):
        return next((v for kk, v in self if kk == k), default)


def _strip_comment(rest):
    q = False
    for j, c in enumerate(rest):
        if c == '"':
            q = not q
        elif not q and c == "#" and (j == 0 or rest[j - 1] in " \t"):
            return rest[:j].strip()
    return rest.strip()


def _leaf(file, lno, rest):
    if rest[0] in "[{":
        return _flow(file, lno, rest)
    if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
        return rest[1:-1]
    return rest


def read_yaml_block(file):
    """metisc's instrument reader (Iface.read_yaml_block): `key: value` lines and block MAPPINGS (`key:` alone opens
    a map indented below it); leaves are scalars and flow values; block lists are refused."""
    lines = []
    with open(file, "rb") as f:
        raw = f.read().decode().split("\n")
    if raw and raw[-1] == "":
        raw = raw[:-1]
    for lno, l in enumerate(raw, 1):
        t = l.strip()
        if t == "" or t[0] == "#":
            continue
        ind = len(l) - len(l.lstrip(" "))
        if ind < len(l) and l[ind] == "\t":
            raise YamlError("%s:%d: a tab in the indentation" % (file, lno))
        if t[0] == "-":
            raise YamlError("%s:%d: a block list is not read here (a flow list [a, b])" % (file, lno))
        i = t.find(":")
        if i < 0:
            raise YamlError("%s:%d: expected key: value" % (file, lno))
        key = t[:i].strip()
        if len(key) >= 2 and key[0] == '"' and key[-1] == '"':
            key = key[1:-1]
        lines.append((lno, ind, key, _strip_comment(t[i + 1:])))

    def map_at(ind, k):
        acc = _Map()
        while k < len(lines):
            lno, i, key, rest = lines[k]
            if i > ind:
                raise YamlError("%s:%d: unexpected indentation" % (file, lno))
            if i < ind:
                break
            if rest != "":
                acc.append((key, _leaf(file, lno, rest)))
                k += 1
            elif k + 1 < len(lines) and lines[k + 1][1] > ind:
                child, k = map_at(lines[k + 1][1], k + 1)
                acc.append((key, child))
            else:
                raise YamlError("%s:%d: %s: an empty value" % (file, lno, key))
        return acc, k
    m, k = map_at(0, 0)
    if k < len(lines):
        raise YamlError("%s:%d: unexpected indentation" % (file, lines[k][0]))
    return m


def _squash(a):
    out, depth = [], 0
    for c in a:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if not (depth > 0 and c in " \t"):
            out.append(c)
    return "".join(out)


def case_of_yaml(path):
    """A case YAML and its manifest (found by `manifest:`, the cases/ convention, or beside it) → (the program's
    path, the init [(atom, n)], the directives in the embedded form's order): beliefs, ports, the manifest's
    queries then the case's, expects, levels (Evaluate.case_of_yaml)."""
    import os

    def bad(m):
        raise YamlError("%s: %s" % (path, m))

    def scal(what, v):
        if not isinstance(v, str):
            bad("%s: expected a scalar" % what)
        return v

    def lst(what, v):
        if not isinstance(v, list) or isinstance(v, _Map):
            bad("%s: expected a flow list" % what)
        return v

    def mp(what, v):
        if not isinstance(v, _Map):
            bad("%s: expected a map" % what)
        return v
    case = read_yaml_block(path)
    for k, _ in case:
        if k not in ("case", "manifest", "init", "ports", "evaluate", "expect"):
            bad("unknown key %s (case, manifest, init, ports, evaluate, expect)" % k)
    d = os.path.dirname(path)
    if case.get("manifest") is not None:
        mpath = os.path.join(d, scal("manifest", case.get("manifest")))
    elif os.path.basename(d) == "cases":
        mpath = os.path.join(os.path.dirname(d), "domain.yaml")
    else:
        mpath = os.path.join(d, "domain.yaml")
    man = read_yaml_block(mpath)
    if man.get("program") is None:
        bad("%s: no program" % mpath)
    program = os.path.join(os.path.dirname(mpath), scal("program", man.get("program")))
    init = []
    for x in lst("init", case.get("init")) if case.get("init") is not None else []:
        t = _squash(scal("init", x))
        i = t.rfind(" ")
        if i >= 0 and t[i + 1:].lstrip("-").isdigit():
            init.append((t[:i], int(t[i + 1:])))
        else:
            init.append((t, 1))
    dirs = []
    for pred, f in mp("beliefs", man.get("beliefs")) if man.get("beliefs") is not None else []:
        f = mp("beliefs " + pred, f)
        if f.get("port") is None:
            bad("beliefs %s: no port" % pred)
        port = scal("port", f.get("port"))
        one, lone = f.get("one"), f.get("lone")
        if one is not None and lone is None:
            dirs.append(["belief", pred, "one", scal("one", one), "by", port])
        elif lone is not None and one is None:
            dirs.append(["belief", pred, "lone", scal("lone", lone), "by", port])
        elif one is None and lone is None:
            dirs.append(["belief", pred, "by", port])
        else:
            bad("beliefs %s: one or lone, not both" % pred)
    for port, b in mp("ports", case.get("ports")) if case.get("ports") is not None else []:
        if isinstance(b, str):
            dirs.append(["port", port, b])
        elif isinstance(b, _Map):
            for atom, x in b:
                a = _squash(atom)
                if split_atom(a)[0] != port:
                    bad("ports %s: the entry %s is not an argument tuple of %s" % (port, atom, port))
                dirs.append(["port", a, scal("ports " + atom, x)])
        else:
            _refuse("port-function", "port %s bound to a list: a binding that computes is the host's function form "
                    "(spec/16 §10, later); a case binds numbers or a table of them" % port)

    def queries(src):
        out = []
        for name, f in mp("evaluate", src.get("evaluate")) if src.get("evaluate") is not None else []:
            f = mp("evaluate " + name, f)
            reach, risk = f.get("reach"), f.get("risk")
            if reach is not None and risk is None:
                kind, pat = "reach", scal("reach", reach)
            elif risk is not None and reach is None:
                kind, pat = "risk", scal("risk", risk)
            else:
                bad("evaluate %s: one of reach, risk" % name)
            within, at = f.get("within"), f.get("at")
            if within is not None and at is None:
                q, h = "within", scal("within", within)
            elif at is not None and within is None:
                q, h = "at", scal("at", at)
            else:
                bad("evaluate %s: one of within, at" % name)
            budget = ["budget", scal("budget", f.get("budget"))] if f.get("budget") is not None else []
            out.append(["evaluate", name, kind] + [t for t in _squash(pat).split(" ") if t] + [q, h] + budget)
        return out
    dirs += queries(man) + queries(case)
    for n, x in mp("expect", case.get("expect")) if case.get("expect") is not None else []:
        dirs.append(["expect", n, scal("expect", x)])
    for pred, f in mp("levels", man.get("levels")) if man.get("levels") is not None else []:
        f = mp("levels " + pred, f)
        src = f.get("from")
        if isinstance(src, _Map) and len(src) == 1 and src[0][0] == "evaluate":
            s = [scal("from", src[0][1])]
        elif isinstance(src, _Map) and len(src) == 1 and src[0][0] == "belief":
            s = ["belief", scal("from", src[0][1])]
        else:
            bad("levels %s: from: {evaluate: Q} or {belief: PORT}" % pred)

        def get(k):
            if f.get(k) is None:
                bad("levels %s: no %s" % (pred, k))
            return [scal(k, x) for x in lst(k, f.get(k))]
        dirs.append(["level", pred, "from"] + s + ["at"] + get("at") + ["names"] + get("names"))
    return program, init, dirs


def evaluate_file(path: str, brute: bool = False):
    """metisc --evaluate FILE.llp: the file's embedded case (its #init and its instrument directives). → Result."""
    from pathlib import Path
    from metis.lang.compiler import compile_catalog, resolve_includes
    from metis.lang.directives import embedded_case
    from metis.lang.parser import parse as parse_llp
    if str(path).endswith(".yaml"):
        llp, init_l, dirs = case_of_yaml(str(path))         # a case YAML and its manifest (§5)
        init = {}
        for a, n in init_l:
            init[a] = n
        llp = Path(llp)
    else:
        llp = Path(path)
        case = embedded_case(llp.read_text())
        init, dirs = case["init"], case["instrument"]
    doc = compile_catalog(resolve_includes(parse_llp(llp.read_text(), file=str(llp)), llp.parent))
    prog = doc.catalog.ground(init)
    return run(doc, prog, init, dirs, brute=brute)


def main(argv=None) -> int:
    """python -m metis.kernel.evaluate FILE.llp [--brute] [--cost]: V/N/L/X lines on stdout, C lines on stderr
    with --cost; exit 0, 1 on a failed #expect, 2 on a refusal (metisc's)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(main.__doc__, file=sys.stderr)
        return 2
    path, flags = argv[0], set(argv[1:])
    bad = flags - {"--brute", "--cost"}
    if bad:
        print("--evaluate: unknown argument %s" % sorted(bad)[0], file=sys.stderr)
        return 2
    try:
        r = evaluate_file(path, brute="--brute" in flags)
    except YamlError as e:
        print("error: %s" % e, file=sys.stderr)
        return 1
    except Refused as e:
        print("refused %s: %s" % (e.code, e.text), file=sys.stderr)
        return 2
    for line in r.lines:
        print(line)
    if "--cost" in flags:
        for line in r.costs:
            print(line, file=sys.stderr)
    return 1 if r.mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
