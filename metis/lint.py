"""The linter (patterns CHECKS.md §3): shape findings over a program
that COMPILES, never gating. Each is a spec/08 diagnostic of phase
`lint`, severity warning; `metisc --lint` mirrors the codes.

    python -m metis.lint FILE.llp [--manifest domain.yaml] [--strict]
    lint                                 (the REPL verb, on the served file)

- state-dead-end: a state pred the program's own rules produce and no
  rule consumes (a hold with no way on: Duty-scoped class with a way on;
  Bounded yield). Exempt: protocol atoms (`__`), preds declared by an
  interface package (a path with an `iface` component), family members
  (the host's), and with a manifest its `shared` out/resource/
  world-private preds and its status atoms.
- state-unproduced: a pred some rule reads or consumes that no rule
  produces, no `#init` holds, and nothing outside supplies (the same
  exemptions, plus `shared` in). The REPL's `list` marks the same atoms
  `?` per session; this is the static version.
- rules-twin-body: two rules of one stage with the same premises
  (variables renamed by first occurrence) and different heads, not one
  rule with alternative heads (Choice site).
- report-ungated: a rule consuming an atom whose last argument is a
  generation (`tick__gen`) without reading `$tick__gen(…, G)` with the
  same variable (Generation guard). The stub's rules pass by
  construction; this is for the hand-written ones.
- stages-twice: an included unit states `stages` or `layers` that the
  includer also states, differently: the includer's wins silently (Stage
  block merge).
- belief-in-safety: with the manifest's `safety: [stages]`, a rule of
  one of them reads or consumes a belief pred: the manifest's `belief:`
  list and the keys of its spec/16 `beliefs:` and `levels:` maps
  (Intent-blind safety).
- enter-equals-hold: a HOLD rule (it reads `$S` and its head is empty:
  an act) and the ENTRY rule of S (it makes S and does not read it) both
  read a member of one declared `family`: the act can move that class,
  and the hold lets go mid-act (Begin hard, hold firm). Silent with no
  family.
- name-not-a-command: with the manifest's `commands: [...]`, a rule with
  an empty head (acted on by its name) whose name is no entry and does
  not begin with an entry and `_` (The name is the command).
- invariant-init (runtime phase, error): the session's or the file's
  `#init` holds a sum over an invariant's members other than the
  declared one (the init half of `invariant`; the rules' half is the
  compiler's).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metis.kernel.diagnostics import Diagnostic, Loc, Source, make
from metis.lang import ast

_INIT = re.compile(r"^\s*#init\s+([A-Za-z][A-Za-z0-9_-]*)")


def init_preds(text: str):
    """The predicates the file's `#init` lines hold (compiler-invisible
    directives, spec/00 §8)."""
    return {m.group(1) for m in map(_INIT.match, text.splitlines()) if m}


def manifest_exempt(manifest) -> Tuple[set, set]:
    """(supplied, taken): preds a manifest says come from outside (in,
    world-private) and preds it says leave or are read outside (out,
    resource, world-private, the status atoms)."""
    if not manifest:
        return set(), set()
    import yaml
    m = yaml.safe_load(open(manifest)) or {}
    shared = m.get("shared") or {}
    supplied = {p for p, d in shared.items() if d in ("in", "world-private")}
    taken = {p for p, d in shared.items()
             if d in ("out", "resource", "world-private")}
    for v in (m.get("status") or {}).values():
        taken.update(v or ())
    return supplied, taken


def manifest_lists(manifest) -> Tuple[List[str], List[str], Optional[List[str]]]:
    """(safety stages, belief preds, commands or None) from a manifest:
    belief = its `belief:` list and the keys of `beliefs:` and `levels:`."""
    if not manifest:
        return [], [], None
    import yaml
    m = yaml.safe_load(open(manifest)) or {}
    belief = list(m.get("belief") or [])
    for key in ("beliefs", "levels"):
        belief += [p for p in (m.get(key) or {}) if p not in belief]
    cmds = m.get("commands")
    return list(m.get("safety") or []), belief, \
        (list(cmds) if cmds is not None else None)


def _iface_pred(d) -> bool:
    return "iface" in Path(getattr(d, "file", "") or "").parts


def _norm(atoms) -> Tuple:
    """Premises with variables renamed by first occurrence, sorted."""
    names: Dict[str, str] = {}

    def t(term):
        if term.kind != "var":
            return term.value
        return names.setdefault(term.value, f"V{len(names)}")
    out = []
    for a in atoms:
        if isinstance(a, ast.Distinct):
            out.append(("<>", names.get(a.a, a.a), names.get(a.b, a.b)))
        else:
            out.append((a.pred, a.persist, tuple(t(x) for x in a.terms)))
    return tuple(sorted(map(str, out)))


def _src(node, decl, line_txt, doc="") -> Source:
    subject = tuple(f"%% {ln}" for ln in doc.split("\n") if ln) \
        + (line_txt,)
    return Source(Loc(getattr(node, "file", ""), getattr(node, "line", 0),
                      getattr(node, "col", 0), decl), subject, ())


def lint(doc, text: str = "", manifest=None, path=None) -> List[Diagnostic]:
    """The lint findings of a compiled CatalogDoc (its `source` AST):
    TEXT is the file's text (for `#init`), MANIFEST its domain.yaml,
    PATH the file (for stages-twice across its includes)."""
    cat = doc.source
    out: List[Diagnostic] = []
    if cat is None:
        return out
    supplied, taken = manifest_exempt(manifest)
    inits = init_preds(text)
    bwd = set(doc.catalog.bwd)
    family_members = {m for _, ms in doc.families for m in ms}
    sigs = {d.name: d.args for d in cat.decls
            if isinstance(d, (ast.PredDecl, ast.BwdDecl))}
    iface = {d.name for d in cat.decls
             if isinstance(d, ast.PredDecl) and _iface_pred(d)}
    decl_of = {d.name: d for d in cat.decls if isinstance(d, ast.PredDecl)}
    produced: Dict[str, List] = {}
    consumed: Dict[str, List] = {}
    read: Dict[str, List] = {}
    clauses = []
    for d in cat.decls:
        if isinstance(d, ast.Stage):
            for r in d.rules:
                clauses.append((d.name, r, [a for a in r.body
                                            if isinstance(a, ast.Atom)],
                                list(r.head), ast.rule_line(r)))
        elif isinstance(d, ast.LinkDecl):
            clauses.append(("qui", d, list(d.pre_atoms), list(d.post_atoms),
                            ast.decl_line(d)))
    for stage, node, body, head, line in clauses:
        for a in body:
            if a.pred in bwd:
                continue
            (read if a.persist else consumed).setdefault(a.pred, []).append(
                node.name)
        for a in head:
            if a.pred != "one":
                produced.setdefault(a.pred, []).append(node.name)
    exempt = {p for p in sigs if "__" in p} | iface | family_members
    # state-dead-end / state-unproduced, in declaration order
    for name, d in decl_of.items():
        if name in exempt:
            continue
        if name in produced and name not in consumed and name not in taken:
            out.append(make(
                "state-dead-end",
                f"{name}: produced by {', '.join(sorted(set(produced[name])))}"
                f" and consumed by no rule: a hold with no way on",
                _src(d, f"pred/{name}", ast.decl_line(d)),
                (("pred", name),
                 ("producers", " ".join(sorted(set(produced[name])))))))
        users = sorted(set(consumed.get(name, []) + read.get(name, [])))
        if users and name not in produced and name not in inits \
                and name not in supplied:
            out.append(make(
                "state-unproduced",
                f"{name}: read by {', '.join(users)} and produced by no "
                f"rule, #init or manifest",
                _src(d, f"pred/{name}", ast.decl_line(d)),
                (("pred", name), ("readers", " ".join(users)))))
    # rules-twin-body, per stage
    seen: Dict[Tuple, Tuple] = {}
    for stage, node, body, head, line in clauses:
        if stage == "qui":
            continue
        key = (stage, _norm(node.body))
        if key in seen and _norm(head) != _norm(seen[key][0].head):
            first = seen[key][0]
            out.append(make(
                "rules-twin-body",
                f"{node.name}: the same premises as {first.name} with "
                f"another head: twin bodies the sampler weighs apart",
                _src(node, f"{stage}/{node.name}", line, node.doc),
                (("rule", node.name), ("twin", first.name))))
        else:
            seen.setdefault(key, (node,))
    # report-ungated
    for stage, node, body, head, line in clauses:
        gens = [a for a in body if not a.persist and a.pred != "tick__gen"
                and sigs.get(a.pred) and sigs[a.pred][-1] == "tick__gen"
                and a.terms and a.terms[-1].kind == "var"]
        if not gens:
            continue
        guarded = {a.terms[-1].value for a in body
                   if a.pred == "tick__gen" and a.terms
                   and a.terms[-1].kind == "var"}
        for a in gens:
            if a.terms[-1].value not in guarded:
                out.append(make(
                    "report-ungated",
                    f"{node.name}: consumes {a.pred} without reading "
                    f"$tick__gen for its generation {a.terms[-1].value}",
                    _src(node, f"{stage}/{node.name}", line, node.doc),
                    (("rule", node.name), ("pred", a.pred))))
    safety, belief, commands = manifest_lists(manifest)
    # belief-in-safety
    for stage, node, body, head, line in clauses:
        if stage not in safety:
            continue
        for p in dict.fromkeys(a.pred for a in body if a.pred in belief):
            out.append(make(
                "belief-in-safety",
                f"{node.name}: a safety-stage rule reads the belief {p}",
                _src(node, f"{stage}/{node.name}", line, node.doc),
                (("rule", node.name), ("pred", p), ("stage", stage))))
    # enter-equals-hold
    family_of = {m: n for n, ms in doc.families for m in ms}
    acts = [c for c in clauses                    # `-o ()`: the head is `one`
            if c[0] != "qui" and all(a.pred == "one" for a in c[3])]
    for stage, node, body, head, line in acts:
        for s in dict.fromkeys(a.pred for a in body
                               if a.persist and a.pred not in bwd):
            for _, enode, ebody, ehead, _ in clauses:
                if s not in {a.pred for a in ehead} \
                        or s in {a.pred for a in ebody}:
                    continue
                theirs = {a.pred for a in ebody}
                for c in dict.fromkeys(a.pred for a in body):
                    if c != s and c in theirs and c in family_of:
                        out.append(make(
                            "enter-equals-hold",
                            f"{node.name} holds {s} on {c}, which "
                            f"{enode.name} tests to enter {s}: the act can "
                            f"move {c}, so the hold lets go mid-act",
                            _src(node, f"{stage}/{node.name}", line,
                                 node.doc),
                            (("rule", node.name), ("entry", enode.name),
                             ("state", s), ("cls", c))))
    # name-not-a-command
    if commands is not None:
        for stage, node, body, head, line in acts:
            n = node.name
            if not any(n == e or n.startswith(e + "_") for e in commands):
                out.append(make(
                    "name-not-a-command",
                    f"{n}: acts by its name, and no command the host "
                    f"takes begins it: the rule fires and nothing moves",
                    _src(node, f"{stage}/{n}", line, node.doc),
                    (("rule", n),)))
    # stages-twice, across the file's includes
    if path:
        out.extend(_stages_twice(Path(path)))
    return out


def _stages_twice(path: Path, seen=None) -> List[Diagnostic]:
    from metis.lang.compiler import include_path
    from metis.lang.parser import parse_file
    seen = seen or set()
    if path in seen:
        return []
    seen.add(path)
    out: List[Diagnostic] = []
    try:
        cat = parse_file(path)
    except Exception:
        return out
    for d in cat.decls:
        if not isinstance(d, ast.IncludeDecl):
            continue
        try:
            sub = include_path(d.path, path.parent)
            inc = parse_file(sub)
        except Exception:
            continue
        for what, mine, theirs in (("stages", cat.stages, inc.stages),
                                   ("layers", cat.layers, inc.layers)):
            if mine and theirs and mine != theirs:
                where = f"{Path(sub).name}"
                out.append(make(
                    "stages-twice",
                    f"{what} ({' '.join(mine)}) stated here and "
                    f"({' '.join(theirs)}) in {where}: the first wins",
                    _src(d, f"include/{d.path}", ast.decl_line(d)),
                    (("decl", what), ("where", where))))
        out.extend(_stages_twice(Path(sub), seen))
    return out


def invariant_init(doc, counts: Dict[str, int]) -> List[Diagnostic]:
    """The init half of each `invariant`: COUNTS (atom -> tokens) must
    hold exactly the declared sum over the members (all ground atoms of
    each member pred)."""
    out: List[Diagnostic] = []
    for name, members, total in doc.invariants:
        held = sum(n for atom, n in counts.items()
                   if atom.split("(", 1)[0] in members)
        if held != total:
            out.append(make(
                "invariant-init",
                f"invariant {name}: the state holds {held} token(s) over "
                f"{' '.join(members)}, declared {total}",
                Source(Loc("", 0, 0, f"invariant/{name}"),
                       (f"invariant {name} ({' '.join(members)}) : {total}.",),
                       ()),
                (("invariant", name), ("members", " ".join(members)),
                 ("total", str(total)), ("held", str(held)))))
    return out


def init_counts(text: str) -> Dict[str, int]:
    """`#init ATOM [N]` lines as counts (the last assignment wins)."""
    counts: Dict[str, int] = {}
    for ln in text.splitlines():
        m = re.match(r"^\s*#init\s+(.+?)(?:\s+(\d+))?\s*$", ln)
        if m:
            # an atom written with inner spaces (`at(e0, f0)`) is the
            # same atom the shell squashes
            counts[m.group(1).replace(" ", "")] = int(m.group(2) or 1)
    return counts


def parser():
    """The verb's argparse: the CLI's, and its row of metis/trace_tools.json (tools/gen_trace_tools.py)."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m metis.lint",
        description="shape findings over a program that compiles "
                    "(patterns CHECKS.md §3); exit 1 with --strict when "
                    "any fires")
    ap.add_argument("file")
    ap.add_argument("--manifest", help="the member's domain.yaml")
    ap.add_argument("--strict", action="store_true")
    return ap


def main(argv=None):
    import sys
    from metis.kernel.diagnostics import render_text
    from metis.lang.compiler import CatalogError, LangError, load_llp
    a = parser().parse_args(argv)
    try:
        doc = load_llp(a.file)
    except (LangError, CatalogError) as e:
        for d in e.diagnostics:
            sys.stderr.write(render_text(d))
        print(f"{a.file}: does not compile; lint reads a compiled program",
              file=sys.stderr)
        return 2
    text = open(a.file).read()
    found = lint(doc, text, a.manifest, a.file) \
        + invariant_init(doc, init_counts(text))
    for d in found:
        sys.stdout.write(render_text(d))
    if not found:
        print(f"{a.file}: no lint finding")
    return 1 if found and a.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
