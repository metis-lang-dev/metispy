"""A manifest's generated level declarations (metis-lang spec/16 amendment 1, toolchain-units s-ddaed776 step 2; the
rulings of 2026-10-29): the declarations a `levels:` block fully determines are the toolchain's, added to the program at
build, byte-equal with metisc's (iface.ml level_decls; `metisc --levels DOMAIN.YAML` prints them).

For a BELIEF level (`from: {belief: PORT}`), in the belief pred's namespace, level by level and in this order:
`type level {names}` (unless a type level is there), the belief's weight port `weight PORT(V..) reads {PRED(V..)}`
(unless the program has the port; V.. as the first rule reading $NAME(_, V..) names them, else the types' initials, the
belief's one/lone formal last) and `pred NAME(level, <PRED's types>) : <PRED's namespace>`. A hand declaration agrees
structurally: NAME's first argument's type has exactly the names, in order, and its rest are PRED's types. An EVALUATE
level's pred is hand-declared (nothing fixes its namespace) and checked the same; a missing one is level-undeclared.
Every refusal is a LangError naming the manifest line: `domain.yaml:LINE: level-CODE: text`."""
import re
from dataclasses import replace
from pathlib import Path

from metis.lang import ast
from metis.lang.compiler import LangError, resolve_includes
from metis.lang.parser import parse_file

HEADER = "levels"


def _is_key(k, line):
    return re.search(r"(^|[ \t{,])%s *:" % re.escape(k), line) is not None


def level_line(text, name):
    """The manifest line naming level NAME (1-based): from the line that opens levels:, the first where NAME is a
    key; 0 when none (metisc's level_line)."""
    opened = False
    for i, l in enumerate(text.split("\n"), 1):
        opened = opened or _is_key(HEADER, l)
        if opened and _is_key(name, l):
            return i
    return 0


def _manifest(manifest):
    """(the manifest as a dict, its text or "", its file name for the messages)."""
    if isinstance(manifest, dict):
        return manifest, "", "domain.yaml"
    import yaml
    text = Path(manifest).read_text()
    return yaml.safe_load(text) or {}, text, Path(manifest).name


def level_decls(manifest, cat: ast.Catalog):
    """The declarations the manifest's levels: block determines and the RESOLVED program `cat` lacks: [decl]
    (ast.decl_line prints each as metisc --levels does). `manifest`: a domain.yaml path, or its dict (a candidate's
    levels, as the loop has them; its lines are then 0)."""
    man, text, fname = _manifest(manifest)
    levels, beliefs = man.get("levels") or {}, man.get("beliefs") or {}
    decls = cat.decls
    preds = {}
    for d in decls:
        if isinstance(d, ast.PredDecl) and d.name not in preds:
            preds[d.name] = (tuple(d.args), d.namespace)
    types = {}
    for d in decls:
        if isinstance(d, ast.TypeDecl) and d.name not in types:
            types[d.name] = tuple(d.consts)
    ports = {d.name for d in decls if isinstance(d, ast.PortDecl) and d.kind == "weight"}
    rules = [r for d in decls if isinstance(d, ast.Stage) for r in d.rules]

    def fail(name, code, msg):
        raise LangError(["%s:%d: %s: %s" % (fname, level_line(text, name) if text else 0, code, msg)])

    out, gen_type = [], None
    for name, spec in levels.items():
        spec = spec or {}
        names = tuple(str(x) for x in (spec.get("names") or ["low", "mid", "high"]))

        def agrees(ts, rest):
            return bool(ts) and types.get(ts[0]) == names and tuple(ts[1:]) == tuple(rest)

        src = spec.get("from") or {}
        if list(src) == ["evaluate"]:
            if name not in preds:
                fail(name, "level-undeclared", "level %s from evaluate %s: no pred %s (an evaluate level's pred is "
                     "the program's: pred %s(T) with type T {%s})" % (name, src["evaluate"], name, name, " ".join(names)))
            if not agrees(preds[name][0], ()):
                fail(name, "level-mismatch", "pred %s(%s) does not agree with level %s: pred %s(T) with type T {%s}"
                     % (name, ", ".join(preds[name][0]), name, name, " ".join(names)))
        elif list(src) == ["belief"]:
            port = str(src["belief"])
            bel = next(((p, b) for p, b in beliefs.items() if str((b or {}).get("port")) == port), None)
            if bel is None:
                fail(name, "level-belief", "level %s from belief %s: no belief has port %s" % (name, port, port))
            bpred, b = bel[0], bel[1] or {}
            if bpred not in preds:
                fail(name, "level-belief", "level %s from belief %s: the belief %s is not a pred of the program"
                     % (name, port, bpred))
            btypes, ns = preds[bpred]
            hand = preds.get(name)
            if hand is not None:
                if not agrees(hand[0], btypes):
                    fail(name, "level-mismatch", "pred %s(%s) does not agree with level %s: pred %s(T, %s) with type "
                         "T {%s}" % (name, ", ".join(hand[0]), name, name, ", ".join(btypes), " ".join(names)))
            elif "level" in types:
                if types["level"] != names:
                    fail(name, "level-type", "type level {%s} is not level %s's names [%s]"
                         % (" ".join(types["level"]), name, " ".join(names)))
            elif gen_type is not None:
                if gen_type[1] != names:
                    fail(name, "level-names", "levels %s [%s] and %s [%s] name different levels: one type level is "
                         "generated; declare both by hand" % (gen_type[0], " ".join(gen_type[1]), name, " ".join(names)))
            else:
                gen_type = (name, names)
                out.append(ast.TypeDecl("level", names))
            if port not in ports:
                read = next((a.terms[1:] for r in rules for a in r.body
                             if isinstance(a, ast.Atom) and a.persist and a.pred == name and a.terms), None)
                if read is not None and len(read) == len(btypes) and all(t.kind == "var" for t in read):
                    vs = [t.value for t in read]
                else:
                    vs = []
                    for t in btypes:
                        v = t[:1].upper() or "X"
                        while v in vs:
                            v += "2"
                        vs.append(v)
                lone = b.get("lone", b.get("one"))
                if lone is not None and vs and str(lone) not in vs:
                    vs[-1] = str(lone)
                atom = ast.Atom(bpred, tuple(ast.Term("var", v) for v in vs))
                out.append(ast.PortDecl("weight", port, tuple(vs), (ast.ReadPattern(atom),)))
            if hand is None:
                out.append(ast.PredDecl(name, ("level",) + tuple(btypes), ns))
        else:
            fail(name, "level-from", "level %s: from is {evaluate: Q} or {belief: PORT}" % name)
    return out


def with_levels(cat: ast.Catalog, manifest=None) -> ast.Catalog:
    """The program with its manifest's generated level declarations after its own (none without a manifest, or with a
    path that is not there)."""
    if manifest is None or (not isinstance(manifest, dict) and not Path(manifest).is_file()):
        return cat
    gen = level_decls(manifest, cat)
    return replace(cat, decls=tuple(cat.decls) + tuple(gen)) if gen else cat


def load_program(path, manifest=None) -> ast.Catalog:
    """A program file, its includes resolved, with its manifest's generated level declarations."""
    return with_levels(resolve_includes(parse_file(str(path)), Path(path).parent), manifest)


def lines(manifest, program=None):
    """The generated declarations as metisc --levels prints them (`program`: default the manifest's)."""
    man, _, _ = _manifest(manifest)
    program = Path(program) if program else Path(manifest).parent / man["program"]
    return [ast.decl_line(d) for d in level_decls(manifest, resolve_includes(parse_file(str(program)), program.parent))]
