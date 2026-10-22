"""metis.intents — a domain's intents (e2e-workflow s-143c3088, WORKFLOW.md stage 3): the
intent pass's grammar and prompts GENERATED from domains/<name>/intent.yaml,
never written by hand, and the check that maps the domain's corpus through
them.

Moved from metis-catalog's tools/intents.py (metis-packages s-65778b98): the runtime's own;
`python -m metis.intents …` is the CLI (from a catalog's directory, or with $METIS_CATALOG);
the generated files keep the header that names tools/intents.py, byte for byte.

    python3 tools/intents.py gen dog            # → domains/dog/intent/{dog.gbnf, dog-prompt.txt, dog-spoken.txt}
    python3 tools/intents.py check dog          # the generated files are what intent.yaml makes (CI)
    python3 tools/intents.py schema dog         # the JSON schema the grammar is made from
    python3 tools/intents.py skeleton plant     # a first intent.yaml from plant.llp's cmd_* atoms and %% lines

intent.yaml (CATALOG.md "intent.yaml"): persona, verbs, slots, guidance,
examples, spoken — the intent pass; map, values, handles, fallback,
standing, enders — the mapper; outcomes, alternatives — the replier.

The grammar is the JSON schema's (verbs → the "intent" enum, slots → the
optional fields in order) in the GBNF shape llama.cpp's
json_schema_to_grammar.py wrote in 2024-04 (ab9a3240a): one-line JSON
(space ::= " "?), enums without trailing space, rules sorted by name.
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = None        # the catalog: set by metis.catalog.generator(r); else metis.catalog.root()


def _root():
    if ROOT is not None:
        return Path(ROOT)
    from metis import catalog
    return catalog.root()
GENERATED = "generated from intent.yaml by tools/intents.py — do not edit"

PRIMITIVES = {
    "char": r'''[^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])''',
    "decimal-part": "[0-9] " + "([0-9] " * 15 + ")?" * 15,
    "integral-part": "[0-9] | [1-9] " + "([0-9] " * 15 + ")?" * 15,
    "number": '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
    "space": '" "?',
    "string": '"\\"" char* "\\"" space',
}
PRIMITIVES["decimal-part"] = PRIMITIVES["decimal-part"].replace(" )", ")")
PRIMITIVES["integral-part"] = PRIMITIVES["integral-part"].replace(" )", ")")


def program_text(path):
    """The program as the compiler reads it: each `include "x.llp".` line replaced by that
    file's text, resolved beside the including file, transitively, each file once (a lifted
    instance: the ui domain's ui.llp includes its base ui_domain.llp, every rule there)."""
    import re
    seen = set()

    def splice(p):
        p = Path(p).resolve()
        if p in seen:
            return ""
        seen.add(p)
        return re.sub(r'^\s*include\s+"([^"]+)"\s*\.\s*$', lambda m: splice(p.parent / m.group(1)),
                      p.read_text(), flags=re.M)
    return splice(path)


def load(name):
    return load_from(_root() / "domains" / name)


def load_from(d):
    """A domain directory's intent.yaml (a scratch copy too)."""
    return yaml.safe_load((Path(d) / "intent.yaml").read_text())


def verb_names(spec):
    out = []
    for row in spec["verbs"]:
        v = row["verb"]
        out += v if isinstance(v, list) else [v]
    return out


# ── the JSON schema ──────────────────────────────────────────────────────────

def _ref(types):
    return {"type": "object", "properties": {
        "type": {"enum": types}, "description": {"type": "string"},
        "attributes": {"type": "object", "properties": {"color": {"type": "string"}, "size": {"type": "string"},
                                                         "material": {"type": "string"}}}},
        "required": ["type", "description"]}


def _names(vs):
    """An enum's values are names: YAML 1.1 reads a bare ON / OFF / YES / NO
    as a boolean — quote it ("ON")."""
    bad = [v for v in vs if not isinstance(v, str)]
    if bad:
        raise ValueError("enum value(s) %r are not names: quote them in intent.yaml" % bad)
    return list(vs)


def slot_schema(s):
    if s == "number" or s == "string":
        return {"type": s}
    if "enum" in s:
        return {"enum": _names(s["enum"])}
    if "object" in s:
        return _ref(_names(s["object"]))
    if "relations" in s:
        return {"type": "array", "items": {"type": "object", "properties": {
            "relation": {"enum": _names(s["relations"])}, "reference": _ref(_names(s["reference"]))},
            "required": ["relation", "reference"]}}
    raise ValueError("slot %r: number | string | {enum} | {object} | {relations, reference}" % (s,))


def schema(spec):
    props = {"intent": {"enum": verb_names(spec)}}
    for k, s in spec.get("slots", {}).items():
        props[k] = slot_schema(s)
    return {"type": "object", "properties": props, "required": ["intent"]}


# ── the grammar ──────────────────────────────────────────────────────────────

class Gbnf:
    def __init__(self):
        self.rules = dict(PRIMITIVES)

    def add(self, name, body):
        self.rules[name] = body
        return name

    def value(self, name, s):
        if "enum" in s:
            return self.add(name, " | ".join('"\\"%s\\""' % v for v in s["enum"]))
        t = s.get("type")
        if t == "number":
            return "number"
        if t == "string":
            return "string"
        if t == "array":
            item = self.value(name + "-item", s["items"])
            return self.add(name, '"[" space (%s ("," space %s)*)? "]" space' % (item, item))
        if t == "object":
            return self.obj(name, s)
        raise ValueError(s)

    def kv(self, name, key, s):
        return self.add(name + "-kv", '"\\"%s\\"" space ":" space %s' % (key, self.value(name, s)))

    def obj(self, name, s):
        props = s.get("properties", {})
        req = [k for k in props if k in s.get("required", [])]
        opt = [k for k in props if k not in req]
        rn = lambda k: (name + "-" if name != "root" else "") + k.replace("_", "-")
        kvs = {k: self.kv(rn(k), k, props[k]) for k in props}
        # the optional chain: each optional field's -rest offers the ones after it
        for i, k in enumerate(opt[:-1]):
            nxt = opt[i + 1]
            tail = " %s-rest" % rn(nxt) if i + 1 < len(opt) - 1 else ""
            self.add(rn(k) + "-rest", '( "," space %s )?%s' % (kvs[nxt], tail))
        alts = [("%s %s-rest" % (kvs[k], rn(k))) if i < len(opt) - 1 else kvs[k] for i, k in enumerate(opt)]
        if req:
            body = '"{" space ' + ' "," space '.join(kvs[k] for k in req)
            if opt:
                body += ' ( "," space ( %s ) )?' % " | ".join(alts)
            return self.add(name, body + ' "}" space')
        return self.add(name, '"{" space  (%s )? "}" space' % " | ".join(alts))

    def text(self):
        return "\n".join("%s ::= %s" % (k, self.rules[k]) for k in sorted(self.rules))


def grammar(spec):
    g = Gbnf()
    g.obj("root", schema(spec))
    return g.text()


# ── the prompts ──────────────────────────────────────────────────────────────

def _j(o):
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def prompt(spec):
    out = [spec["persona"].rstrip("\n"), "", "Intent categories:", "| intent | meaning |", "|--------|---------|"]
    for row in spec["verbs"]:
        v = row["verb"]
        out.append("| %s | %s |" % (" / ".join(v) if isinstance(v, list) else v, row["meaning"]))
    out += ["", spec["guidance"].rstrip("\n"), "", "## Examples (utterance → JSON)"]
    for e in spec["examples"]:
        out.append(e["note"] if isinstance(e, dict) else '"%s" → %s' % (e[0], _j(e[1])))
    return "\n".join(out) + "\n"


def spoken(spec):
    s = spec.get("spoken")
    if not s:
        return ""
    out = ["", "## " + s["heading"], s["text"].rstrip("\n")]
    for u, j, said in s["examples"]:
        out += ['"%s" → %s' % (u, _j(j)), "Spoken response: " + said]
    return "\n".join(out) + "\n"


# ── the draft: intent.yaml's skeleton from the program ──────────────────────

def program_rules(prog):
    """The program's rules: (its %% line, name, premises, products), atoms
    as written ("$moist" a premise kept, "cmd_check" one consumed)."""
    import re
    out, doc = [], None
    for line in prog.splitlines():
        t = line.strip()
        if t.startswith("%%"):
            doc = t[2:].strip()
        elif " -o " in t and ":" in t:
            name = t.split()[0]
            lhs, rhs = t.split(":", 1)[1].split(" -o ", 1)
            atoms = lambda s: [a.strip().rstrip(".").strip() for a in s.split("*") if a.strip().rstrip(".").strip()]
            out.append((doc, name, atoms(lhs), atoms(rhs)))
            doc = None
        elif t and not t.startswith("%"):
            doc = None
    return out


def _pred(atom):
    return atom.lstrip("$").split("(")[0].strip()


def command_rules(rules, stem):
    """The rules that consume the command cmd_<stem>."""
    return [r for r in rules if any(_pred(a) == "cmd_" + stem and not a.startswith("$") for a in r[2])]


def outcome_atoms(rules):
    """What a command's rule produces that no rule reads: the outcome tokens
    the host narrates and retracts (done(water), checked(moist)) — the
    replier's `outcomes` keys; the domain's state (moist, which dry_out
    consumes) is not one."""
    read = {_pred(a) for r in rules for a in r[2]}
    out = []
    for _, _, lhs, rhs in rules:
        if any(_pred(a).startswith("cmd_") for a in lhs):
            for a in rhs:
                p = _pred(a)
                if not p.startswith("cmd_") and p not in read and p not in out:
                    out.append(p)
    return out


def asks_problems(spec, prog):
    """An ask must ask: every rule of each command its verb maps to keeps
    the state ($-premises only, besides the command) — else it changes the
    world and is a command, not a question (CATALOG.md `asks`)."""
    rules, out = program_rules(prog), []
    names = verb_names(spec)
    for v in spec.get("asks") or []:
        if v not in names:
            out.append("asks %s: no such verb" % v)
            continue
        for row in spec.get("map") or []:
            if row.get("verb") != v:
                continue
            stem = row["atom"].split("(")[0][len("cmd_"):]
            rs = command_rules(rules, stem)
            if not rs:
                out.append("asks %s: %s has no rule in the program" % (v, row["atom"].split("(")[0]))
            for _, rn, lhs, _ in rs:
                taken = [a for a in lhs if not a.startswith("$") and _pred(a) != "cmd_" + stem]
                if taken:
                    out.append("asks %s: rule %s consumes %s — it changes the state, so %s is a command, not a question"
                               % (v, rn, ", ".join(taken), v))
    return out


def skeleton(name):
    """A first intent.yaml from the program alone (the loop console's
    `intents` hands it to Claude, who writes the meanings, examples and
    perception vocabulary): one verb per command atom `pred cmd_X[(T)]`, its
    meaning the %% line of the rule that completes it (done(...)), an
    argument's type as an enum of the type's constants, one example per
    verb; the replier's defaults. Not a finished domain: a start."""
    import re
    prog = program_text(_root() / "domains" / name / ("%s.llp" % name))
    types = {m.group(1): m.group(2).split() for m in re.finditer(r"^type\s+(\w+)\s*\{([^}]*)\}", prog, re.M)}
    about = re.search(r'^provenance\s+text\s+"([^"]*)"', prog, re.M)
    cmds = [(m.group(1), [a.strip() for a in m.group(2).split(",")] if m.group(2) else [])
            for m in re.finditer(r"^pred\s+cmd_(\w+)(?:\(([^)]*)\))?\s*:", prog, re.M)]
    rules, doc = [], None
    for line in prog.splitlines():
        t = line.strip()
        if t.startswith("%%"):
            doc = t[2:].strip()
        elif " -o " in t and ":" in t:
            rules.append((doc, t))
            doc = None
        elif t and not t.startswith("%"):
            doc = None

    def meaning(stem):
        mine = [(d, r) for d, r in rules if re.search(r"\bcmd_%s\b" % stem, r.split(" -o ")[0])]
        done = [d for d, r in mine if d and "done(" in r.split(" -o ")[1]]
        return (done or [d for d, _ in mine if d] or [stem.replace("_", " ")])[0]

    verbs, slots, values, rows, examples = [], {}, {}, [], []
    for stem, args in cmds:
        verb = stem.upper()
        verbs.append({"verb": verb, "meaning": meaning(stem)})
        atom, ex = "cmd_" + stem, {"intent": verb}
        if args:
            binds = []
            for t in args:
                vals = types.get(t, [])
                slots.setdefault(t, {"enum": [v.upper() for v in vals]})
                values.setdefault(t, {v.upper(): v for v in vals})
                binds.append("{%s:%s}" % (t, t))
                if vals:
                    ex[t] = vals[0].upper()
            atom += "(%s)" % ", ".join(binds)
        rows.append({"verb": verb, "atom": atom, "act": stem})
        examples.append([" ".join([stem.replace("_", " ")] + [ex[t].lower() for t in args if t in ex]), ex])
    asks = [s.upper() for s, _ in cmds if s not in ("time", "date", "chat", "chitchat", "unknown", "stop", "cancel")
            and command_rules(program_rules(prog), s)
            and all(all(a.startswith("$") or _pred(a) == "cmd_" + s for a in lhs)
                    for _, _, lhs, _ in command_rules(program_rules(prog), s))]
    spec = {
        "persona": "You map what the person says about the %s to exactly one intent JSON per utterance.\n"
                   "Respond with only the JSON.%s\n" % (name, ("\n\nThe %s: %s." % (name, about.group(1))) if about else ""),
        "verbs": verbs + [{"verb": "UNKNOWN", "meaning": "anything the %s cannot do" % name}],
        "slots": slots,
        "guidance": "Use only the intents and values listed. Omit fields you don't need.\n",
        "examples": examples + [["order me a pizza", {"intent": "UNKNOWN"}]],
        "map": rows,
        "values": values,
        "enders": ["cmd_%s" % s for s, _ in cmds if s in ("stop", "cancel")],
        # every atom a command's rule produces that no rule reads: done unless it names its status
        "outcomes": {o: (o if o in ("done", "refused", "progress", "expired", "cancelled") else "done")
                     for o in outcome_atoms(program_rules(prog))},
        "alternatives": {"max": 2, "rules": {}},
    }
    if asks:   # candidates: the commands whose rules only read the state; keep the questions
        spec["asks"] = asks
    head = ("# %s's intents — a DRAFT from %s.llp (tools/intents.py skeleton): one verb per command\n"
            "# atom; write the meanings, more examples, open-vocabulary slots and handles,\n"
            "# then `python3 tools/intents.py gen %s` and the check. CATALOG.md \"intent.yaml\".\n"
            "# outcomes: what a command's rule produces that no rule reads. asks: CANDIDATES, the\n"
            "# commands whose rules only read the state — keep the ones the person asks as a question.\n") % (name, name, name)
    class Dumper(yaml.SafeDumper):
        pass
    Dumper.add_representer(str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))
    return head + yaml.dump(spec, Dumper=Dumper, sort_keys=False, allow_unicode=True, width=100)


def files(name):
    spec = load(name)
    return {"%s.gbnf" % name: grammar(spec), "%s-prompt.txt" % name: prompt(spec),
            "%s-spoken.txt" % name: spoken(spec)}


def main(argv):
    if len(argv) != 2 or argv[0] not in ("gen", "check", "schema", "skeleton"):
        print(__doc__)
        return 2
    verb, name = argv
    if verb == "schema":
        print(json.dumps(schema(load(name)), indent=1))
        return 0
    if verb == "skeleton":
        print(skeleton(name), end="")
        return 0
    d = _root() / "domains" / name / "intent"
    bad = 0
    prog = _root() / "domains" / name / ("%s.llp" % name)
    if verb == "check" and prog.is_file():
        for msg in asks_problems(load(name), program_text(prog)):
            print("ASKS", msg)
            bad += 1
    for fn, text in files(name).items():
        p = d / fn
        if not text and not p.exists():   # an empty file is not written (a spec without spoken: no -spoken.txt)
            continue
        if verb == "gen":
            d.mkdir(exist_ok=True)
            p.write_text(text)
            print("wrote", p.relative_to(_root()))
        elif not p.exists() or p.read_text() != text:
            print("STALE", p.relative_to(_root()), "(python3 tools/intents.py gen %s)" % name)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
