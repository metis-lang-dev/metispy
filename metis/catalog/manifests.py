"""metis.catalog.manifests — the catalog's loader (domain-catalog s-2a2bc758; CATALOG.md), moved
from metis_catalog (metis-packages s-65778b98: the runtime's own distribution; metis_catalog
re-exports it for one release).

Each domain package is one directory with a manifest, domain.yaml, anywhere
under the catalog root (its name is the directory's; domains/<name>/ is the
convention of a repository-sized catalog, and where a new one is written;
cwd-root s-ddac2cbf); every key is optional and the loader refuses any other
(check_manifest; CATALOG.md lists each key's reader):

    program: dog.llp                  # paths are relative to the manifest's directory
    extensions: [extensions/a.llp]    # absent: extensions/*.llp in name order (each `extends` the program)
    serves: [a.llp, …]                # the served program's files, in order (pkg.mod:path allowed);
                                      #   absent: [program] + extensions — Domain.serves()
    host: dog_host.py:make            # file.py:factory (relative) or pkg.mod:factory
    intent: {spec: intent.yaml}       # J only; the generated files at fixed names in intent/
    clock: {beats: 1}                 # env beats per orchestrator tick (+ steps, and py's watch:
                                      #   lockstep / follow, exact_states, samples: CATALOG.md)
    reload: propose                   # propose | auto
    records: records/                 # the agents' surprise records (records/<session>.jsonl)
    display: the dog                  # the narrator's name
    line: "a robot dog: …"            # the one display line (consoles, the orchestrator's INSTALLED list)
    voice: own                        # own | third
    agent: agent.py[:VARIANT]         # the domain-agent module; absent with intent.spec: the generic agent
    actor: {max_steps: 40, task: "…"} # act without a rule (metis-agent's actor): each quiet beat the model types an
                                      #   intent from the card for the task, the host's act() runs it (acts: false: off)
    pair: [kitchen_shared]            # partners stepped inside its clock
    partner_of: kitchen               # never loaded alone
    notch: dog.notch.json             # a fitted overlay

A directory whose only catalog is one .llp needs no manifest (a
program-only domain); every other .llp is listed as a plain catalog, with
what it has read off the checkout (#plugin, #notch, a README).

    from metis.catalog import manifests as mc
    mc.root()                 # the catalog ($METIS_CATALOG, else the cwd)
    mc.domains()              # [Domain] — every manifest, by name
    mc.get("dog")             # Domain: .program .extensions() .serves() .host .host_spec_for(program_dir) .intent … .has()
    mc.index()                # every catalog for a console's list: the domains first (display line, has)

    python -m metis_catalog list [--json]      (the loader's CLI: metis.catalog.manifests_cli)
"""
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

def data_dir():
    """The installed metis_catalog package's directory (the domains' data: the wheel carries
    domains/ and tools/ in it; a checkout or an editable install has them beside it), or None
    when it is not installed. Found without importing it."""
    import importlib.util
    try:
        spec = importlib.util.find_spec("metis_catalog")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def root():
    """The catalog: $METIS_CATALOG, else the cwd (the user's rule, cwd-root s-ddac2cbf: the
    catalog is the pwd), as metis.catalog.root(). The installed metis_catalog's own domains
    are found by data_dir() (metis.catalog.installed()), never as a default root (release-r02)."""
    env = os.environ.get("METIS_CATALOG")
    return Path(env).resolve() if env else Path.cwd().resolve()


@dataclass
class Domain:
    """One manifest, its paths resolved against its directory."""
    name: str
    dir: Path
    manifest: dict = field(default_factory=dict)

    def path(self, rel):
        return (self.dir / rel).resolve() if rel else None

    @property
    def display(self):
        return self.manifest.get("display") or self.name

    @property
    def line(self):
        return self.manifest.get("line") or ""

    @property
    def voice(self):
        return self.manifest.get("voice") or "third"

    @property
    def program(self):
        return self.path(self.manifest.get("program"))

    @property
    def notch(self):
        return self.path(self.manifest.get("notch"))

    @property
    def host(self):
        """The host as the manifest names it: (absolute file, factory) for
        file.py:factory, (None, "pkg.mod:factory") for a module spec, or None."""
        h = self.manifest.get("host")
        if not h:
            return None
        mod, _, fac = h.rpartition(":")
        if mod.endswith(".py"):
            return (self.dir / mod).resolve(), fac
        return None, h

    def host_spec_for(self, program_dir=None):
        """The host spec as metispy's Repl takes it (plugin=…): a file.py
        relative to the program's directory, or the module spec as is."""
        h = self.host
        if h is None:
            return None
        f, fac = h
        if f is None:
            return fac
        base = Path(program_dir) if program_dir else (self.program.parent if self.program else self.dir)
        return "%s:%s" % (os.path.relpath(f, base), fac)

    @property
    def agent(self):
        """(the agent module's file, its variant) or None — absent with an
        intent.spec, the generic agent (metis-agent intent_domain) is the agent."""
        a = self.manifest.get("agent")
        if not a:
            return None
        f, _, var = a.partition(":")
        return self.path(f), var or None

    @property
    def intent(self):
        """The intent material's paths: the manifest names only the spec
        (intent: {spec: intent.yaml}); the generated files sit at fixed names
        in intent/ and are listed when present — grammar <n>.gbnf, prompt
        <n>-prompt.txt, examples {gemma4: <n>-spoken.txt}, corpus
        <n>-corpus.tsv (catalog-entry step 2). J only: the REPL's tool
        syntax (T) is metispy's, never the catalog's."""
        i = self.manifest.get("intent") or {}
        if not i.get("spec"):
            return {}
        out, d = {"spec": self.path(i["spec"])}, self.dir / "intent"
        for k, f in (("grammar", "%s.gbnf"), ("prompt", "%s-prompt.txt"), ("corpus", "%s-corpus.tsv")):
            if (d / (f % self.name)).is_file():
                out[k] = (d / (f % self.name)).resolve()
        if (d / ("%s-spoken.txt" % self.name)).is_file():
            out["examples"] = {"gemma4": (d / ("%s-spoken.txt" % self.name)).resolve()}
        return out

    def extensions(self):
        """The domain's extensions, in order: the manifest's `extensions:
        [files]` (relative), else extensions/*.llp beside the manifest in
        name order — each `extends` the program (the language's word). The
        served program is the program PLUS every extension (human-loop step
        1, the architect's ruling): the domain agent builds and reloads from
        the union, the loop console's one-click open loads them, apply
        writes them there."""
        e = self.manifest.get("extensions")
        if e:
            return [self.path(x) for x in e]
        d = self.dir / "extensions"
        return sorted(p.resolve() for p in d.glob("*.llp")) if d.is_dir() else []

    def serves(self):
        """The served program's files, in order: the program first, then
        what extends it (runtime-binding Q1, the architect's ruling). The
        manifest's `serves: [files]` — each relative to the manifest's
        directory, or `pkg.mod:path` inside an importable package (the
        kitchen pair's scenes live in metispy) — else [program] + extensions().
        py, Go and the artifact emitter read the served program here only."""
        s = self.manifest.get("serves")
        if not s:
            return ([self.program] if self.program else []) + self.extensions()
        return [_served_file(self.dir, x) for x in s]

    @property
    def records(self):
        return self.dir / (self.manifest.get("records") or "records")

    @property
    def readme(self):
        r = self.dir / "README.md"
        return r.resolve() if r.is_file() else None

    @property
    def clock(self):
        return dict(self.manifest.get("clock") or {"beats": 1})

    @property
    def pair(self):
        return list(self.manifest.get("pair") or [])

    @property
    def partner_of(self):
        return self.manifest.get("partner_of")

    def has(self):
        return {"program": bool(self.program and self.program.is_file()), "notch": bool(self.notch and self.notch.is_file()),
                "host": self.host is not None, "agent": self.agent is not None or bool(self.intent.get("spec")),
                "intent": bool(self.intent), "readme": self.readme is not None, "extensions": bool(self.extensions())}


def _served_file(base, entry):
    """One `serves` entry: a path relative to `base`, or pkg.mod:path."""
    mod, sep, rel = str(entry).partition(":")
    if sep and rel and "/" not in mod and not mod.endswith(".llp"):
        import importlib.util
        try:
            spec = importlib.util.find_spec(mod)
        except ModuleNotFoundError:     # its parent package is not installed
            spec = None
        if spec is None:
            raise ValueError("serves %r: no package %r" % (entry, mod))
        root = (Path(next(iter(spec.submodule_search_locations)))
                if spec.submodule_search_locations else Path(spec.origin).parent)
        return (root / rel).resolve()
    return (Path(base) / entry).resolve()


# The package's manifest: every key and its reader (CATALOG.md has the table).
KEYS = ("program", "extensions", "serves", "host", "intent", "clock", "reload", "records",
        "display", "line", "voice", "agent", "actor", "pair", "partner_of", "notch",
        "shared", "status", "world", "resources",      # metis-lang spec/15's interface (§2)
        "gating",                                      # who holds a resource's commands: host (default) | program
        "beliefs", "levels", "evaluate")               # metis-lang spec/16 §2, §4, §5: the one-tick instrument's
DIRECTIONS = ("in", "out", "resource", "world-private")
INTENT_KEYS = ("spec",)
ACTOR_KEYS = ("acts", "max_steps", "task")


class ManifestError(ValueError):
    pass


def _catalog_root(p):
    """The catalog root a manifest lives in when no root is given: the parent
    of the outermost `domains` directory above it, or None outside a
    domains/ tree (domains() always passes its root)."""
    ups = [a for a in Path(os.path.abspath(p)).parents if a.name == "domains"]
    return ups[-1].parent if ups else None


def _paths(m):
    """(key, path text) for every value of a manifest that names a file: the
    program, its extensions, the served files (not pkg.mod:path), a file.py
    host, the agent module, the notch, the records, the intent spec."""
    for k in ("program", "notch", "records"):
        if isinstance(m.get(k), str):
            yield k, m[k]
    for name, f in (m.get("evaluate") or {}).items() if isinstance(m.get("evaluate"), dict) else ():
        if isinstance(f, dict) and isinstance(f.get("model"), str):      # a query's world model (spec/16 §3)
            yield "evaluate %s model" % name, f["model"]
    for k in ("extensions", "serves"):
        for x in m.get(k) or []:
            mod, sep, rel = str(x).partition(":")
            if k == "serves" and sep and rel and "/" not in mod and not mod.endswith(".llp"):
                continue                         # pkg.mod:path, inside an importable package
            yield k, str(x)
    if isinstance(m.get("host"), str):
        mod, _, _ = m["host"].rpartition(":")
        if mod.endswith(".py"):
            yield "host", mod
    if isinstance(m.get("agent"), str):
        yield "agent", m["agent"].partition(":")[0]
    if isinstance(m.get("intent"), dict) and isinstance(m["intent"].get("spec"), str):
        yield "intent.spec", m["intent"]["spec"]


def _check_interface(p, m):
    """spec/15's fields, their shapes: shared {pred: in|out|resource|world-private}, status {success: [..],
    failure: [..]} of its out preds, world {out: [..], in: [..]}, resources {name: {count: n >= 1, held_by: [..],
    free: pred}} (free: the pred whose atom over the resource's constant is the free token; absent, the resource's
    name is a 0-ary pred)."""
    def strs(v):
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    sh = m.get("shared")
    if sh is not None:
        if not isinstance(sh, dict) or any(not isinstance(k, str) or v not in DIRECTIONS for k, v in sh.items()):
            raise ManifestError("%s: shared is {pred: %s}" % (p, " | ".join(DIRECTIONS)))
    st = m.get("status")
    if st is not None:
        if not isinstance(st, dict) or set(st) - {"success", "failure"} or not all(strs(v) for v in st.values()):
            raise ManifestError("%s: status is {success: [pred…], failure: [pred…]}" % p)
        outs = {k for k, v in (sh or {}).items() if v == "out"}
        bad = sorted(x for v in st.values() for x in v if x not in outs)
        if bad:
            raise ManifestError("%s: status names %s, not an out pred of shared" % (p, ", ".join(bad)))
    w = m.get("world")
    if w is not None and (not isinstance(w, dict) or set(w) - {"out", "in"} or not all(strs(v) for v in w.values())):
        raise ManifestError("%s: world is {out: [pred…], in: [pred…]}" % p)
    if m.get("gating") not in (None, "host", "program"):
        raise ManifestError("%s: gating is host or program (host: the agent holds a command until its grant; "
                            "program: the program reads tick__grant and gates itself)" % p)
    rs = m.get("resources")
    if rs is not None:
        ok = isinstance(rs, dict) and all(isinstance(v, dict) and not set(v) - {"count", "held_by", "free"}
                                          and isinstance(v.get("count"), int) and v["count"] >= 1
                                          and strs(v.get("held_by", [])) and isinstance(v.get("free", ""), str)
                                          for v in rs.values())
        if not ok:
            raise ManifestError("%s: resources is {name: {count: n >= 1, held_by: [pred…], free: pred}}" % p)


def _check_belief(p, m):
    """spec/16's three fields, their shapes: beliefs {pred: {port, one|lone?}}; levels {pred: {from: {evaluate: Q} |
    {belief: PORT}, at: [increasing in (0, 1)], names: [|at| + 1]}}; evaluate {name: {reach|risk: OUTCOME,
    within|at: h, budget?, model?, tick?}} (model: the world model the query runs over, spec/16 §3, else the program;
    tick: the stage whose entries count h and the step rows, spec/16 §10.1)."""
    def num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    b = m.get("beliefs")
    if b is not None:
        ok = isinstance(b, dict) and all(isinstance(f, dict) and isinstance(f.get("port"), str)
                                         and not set(f) - {"port", "one", "lone"} and not ("one" in f and "lone" in f)
                                         for f in b.values())
        if not ok:
            raise ManifestError("%s: beliefs is {pred: {port: PORT, one|lone: VAR}} (spec/16 §2)" % p)
    e = m.get("evaluate")
    if e is not None:
        ok = isinstance(e, dict) and all(
            isinstance(f, dict) and not set(f) - {"reach", "risk", "within", "at", "budget", "model", "tick"}
            and (("reach" in f) != ("risk" in f)) and (("within" in f) != ("at" in f))
            and isinstance(f.get("within", f.get("at")), int) and isinstance(f.get("model", ""), str)
            and isinstance(f.get("tick", ""), str)
            for f in e.values())
        if not ok:
            raise ManifestError("%s: evaluate is {name: {reach|risk: OUTCOME, within|at: h, budget?, model?, tick?}} (spec/16 §5)" % p)
    lv = m.get("levels")
    if lv is not None:
        def good(f):
            if not isinstance(f, dict) or set(f) - {"from", "at", "names"}:
                return False
            src, at, names = f.get("from"), f.get("at"), f.get("names")
            return (isinstance(src, dict) and len(src) == 1 and next(iter(src)) in ("evaluate", "belief")
                    and isinstance(at, list) and all(num(x) and 0 < x < 1 for x in at)
                    and all(a < b for a, b in zip(at, at[1:]))
                    and isinstance(names, list) and len(names) == len(at) + 1 and all(isinstance(x, str) for x in names))
        if not (isinstance(lv, dict) and all(good(f) for f in lv.values())):
            raise ManifestError("%s: levels is {pred: {from: {evaluate: Q} | {belief: PORT}, at: [increasing in (0, 1)], "
                                "names: [one more than at]}} (spec/16 §4)" % p)
        bad = sorted(q for f in lv.values() for k, q in f["from"].items() if k == "evaluate" and q not in (e or {}))
        if bad:
            raise ManifestError("%s: levels from evaluate %s: no such query" % (p, ", ".join(bad)))


def check_manifest(p, root=None):
    """A domain.yaml read and checked: a mapping of the known keys only,
    intent {spec} only (catalog-entry step 2: the loader refuses an unknown
    key). The package's name is its directory's. Every path it names stays
    inside the catalog root (`root`, else the parent of the domains/ tree
    the manifest is in; a manifest outside one is checked for its keys
    only): relative, and normalized it does not climb out of the root — a
    package may name its neighbour's file (../kitchen/agent.py), never one
    outside the catalog."""
    m = yaml.safe_load(Path(p).read_text()) or {}
    if not isinstance(m, dict):
        raise ManifestError("%s: a manifest is a mapping" % p)
    bad = sorted(k for k in m if k not in KEYS)
    if bad:
        raise ManifestError("%s: unknown key%s %s (the package's keys: %s)"
                            % (p, "s" if len(bad) > 1 else "", ", ".join(bad), ", ".join(KEYS)))
    i = m.get("intent")
    if i is not None:
        if not isinstance(i, dict) or not i.get("spec"):
            raise ManifestError("%s: intent is {spec: intent.yaml}" % p)
        bad = sorted(k for k in i if k not in INTENT_KEYS)
        if bad:
            raise ManifestError("%s: unknown intent key%s %s (intent is {spec}; the generated files sit "
                                "at fixed names in intent/)" % (p, "s" if len(bad) > 1 else "", ", ".join(bad)))
    for k in ("extensions", "serves", "pair"):
        if k in m and not isinstance(m[k], list):
            raise ManifestError("%s: %s is a list" % (p, k))
    _check_interface(p, m)
    _check_belief(p, m)
    if m.get("reload") not in (None, "propose", "auto"):
        raise ManifestError("%s: reload is propose or auto" % p)
    a = m.get("actor")
    if a is not None:
        if not isinstance(a, dict):
            raise ManifestError("%s: actor is {max_steps, task, acts?}" % p)
        bad = sorted(k for k in a if k not in ACTOR_KEYS)
        if bad:
            raise ManifestError("%s: unknown actor key%s %s (actor is {%s})" % (p, "s" if len(bad) > 1 else "", ", ".join(bad),
                                                                            ", ".join(ACTOR_KEYS)))
        if "max_steps" in a and not (isinstance(a["max_steps"], int) and not isinstance(a["max_steps"], bool) and a["max_steps"] > 0):
            raise ManifestError("%s: actor max_steps is a positive number" % p)
        if "task" in a and not isinstance(a["task"], str):
            raise ManifestError("%s: actor task is a line of text" % p)
        if "acts" in a and not isinstance(a["acts"], bool):
            raise ManifestError("%s: actor acts is true or false" % p)
    r = Path(os.path.abspath(root)) if root else _catalog_root(p)
    if r is not None:
        base = Path(os.path.abspath(p)).parent
        for k, v in _paths(m):
            if os.path.isabs(v) or v.startswith("~"):
                raise ManifestError("%s: %s %r is absolute: a package names its files relative to its manifest"
                                    % (p, k, v))
            q = os.path.normpath(base / v)
            if os.path.commonpath([q, str(r)]) != str(r):
                raise ManifestError("%s: %s %r leaves the catalog root %s" % (p, k, v, r))
    return m


# never a catalog's: a dot directory (.metis/ the loops' state, .git/, .venv/ …) and these
SKIP = ("build", "records", "__pycache__", "node_modules")


def walk(r, name):
    """Every file matching `name` (a glob) under the root `r`, sorted, never
    under a dot directory or one of SKIP (bundles, surprise records, caches):
    the root is a workspace too, and may be any directory."""
    import fnmatch
    out = []
    for d, dirs, files in os.walk(r):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in SKIP]
        out += [Path(d) / f for f in fnmatch.filter(files, name)]
    return sorted(out)


def _manifests(r=None):
    r = Path(r) if r else root()
    for p in walk(r, "domain.yaml"):
        yield Domain(p.parent.name, p.parent.resolve(), check_manifest(p, r))


def domains(r=None):
    """Every domain with a manifest, by name (a name twice is an error)."""
    out = {}
    for d in _manifests(r):
        if d.name in out:
            raise ValueError("domain %r twice: %s and %s" % (d.name, out[d.name].dir, d.dir))
        out[d.name] = d
    return [out[k] for k in sorted(out)]


def get(name, r=None):
    for d in domains(r):
        if d.name == name:
            return d
    raise KeyError("no domain %r in %s" % (name, Path(r) if r else root()))


# ── the console's list: every catalog, the domains first ──────────────────────

_PLUGIN = re.compile(r"^#plugin\s+(\S+)", re.M)
_NOTCH = re.compile(r"^#notch\s+(\S+)", re.M)
_DRAFT = re.compile(r"(^|/)(iter\d+-world|step\d|.*_born$|.*-candidate|.*LOSSY)")


def _title(readme):
    for line in readme.read_text(errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _table(r):
    rows, readme = {}, r / "README.md"
    if readme.is_file():
        for m in re.finditer(r"^\|\s*`([^`]+)/`\s*\|\s*(.+?)\s*\|\s*$", readme.read_text(errors="replace"), re.M):
            rows[m.group(1).strip("/")] = re.sub(r"\s+", " ", m.group(2))
    return rows


def _first_comment(text):
    for line in text.splitlines():
        if line.startswith("%"):
            s = line.lstrip("% ").strip()
            if s and not s.startswith(("SOURCE:", "//", "/")):
                return s
        elif line.strip():
            break
    return ""


def _short(s, n=140):
    s = re.sub(r"\*\*|`", "", s or "")
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + "…"


def _inferred(r, path, table):
    """A catalog with no manifest: what it has, read off the checkout."""
    rel = path.relative_to(r).with_suffix("").as_posix()
    text = path.read_text(errors="replace")
    host = (m.group(1) if (m := _PLUGIN.search(text)) else
            "%s_host.py:make" % path.stem if (path.parent / ("%s_host.py" % path.stem)).is_file() else None)
    notch = (m.group(1) if (m := _NOTCH.search(text)) else
             "%s.notch.json" % path.stem if (path.parent / ("%s.notch.json" % path.stem)).is_file() else None)
    readme = next((x for x in (path.parent / path.stem / "README.md", path.parent / "README.md")
                   if x.is_file() and (x.parent.name == path.stem or len(list(x.parent.glob("*.llp"))) == 1)), None)
    row = table.get(path.parent.relative_to(r).as_posix(), "")
    alone = len(list(path.parent.glob("*.llp"))) == 1
    line = (_title(readme) if readme else "") or (row if alone else "") or _first_comment(text) or row
    domain = bool(host or notch or readme) and not _DRAFT.search(rel)
    return {"name": path.stem, "path": rel, "line": _short(line), "kind": "domain" if domain else "catalog",
            "manifest": None, "host": host, "notch": notch,
            "readme": readme.relative_to(r).as_posix() if readme else None,
            "has": {"program": True, "host": bool(host), "notch": bool(notch), "readme": bool(readme),
                    "agent": False, "intent": False, "extensions": False}}


def index(r=None):
    """Every catalog of the checkout for a console's list: the manifests'
    domains first (their display line, what they have, the host spec as
    the program's Repl takes it), then the other .llp files — a program
    with a host, a notch or its own README is still a domain; the rest are
    plain catalogs."""
    r = (Path(r) if r else root()).resolve()
    table = _table(r)
    out, covered = [], set()
    for d in domains(r):
        if d.partner_of:                       # never loaded alone: the pair's lead lists it
            continue
        prog = d.program
        if prog is not None:
            covered.add(prog)
        has = d.has()
        line = d.line or (_title(d.readme) if d.readme else "") or (_first_comment(prog.read_text()) if prog and prog.is_file() else "")
        out.append({"name": d.name, "path": prog.relative_to(r).with_suffix("").as_posix() if prog and prog.is_relative_to(r) else None,
                    "line": _short(line), "kind": "domain", "manifest": (d.dir / "domain.yaml").relative_to(r).as_posix(),
                    "display": d.display, "host": d.host_spec_for() if prog else None,
                    "notch": os.path.relpath(d.notch, prog.parent) if d.notch and prog else None,
                    "readme": d.readme.relative_to(r).as_posix() if d.readme else None, "has": has,
                    "pair": d.pair})
    for p in walk(r, "*.llp"):
        if p.resolve() in covered:
            continue
        out.append(_inferred(r, p, table))
    return sorted(out, key=lambda e: (e["kind"] != "domain", e["manifest"] is None, e["name"] if e["kind"] == "domain" else e["path"]))


def readme_head(rel, limit=1600, r=None):
    """A README's opening (its title and first paragraphs, cut at a paragraph)."""
    p = (Path(r) if r else root()) / rel
    if not p.is_file():
        return ""
    out, n = [], 0
    for para in p.read_text(errors="replace").split("\n\n"):
        if n + len(para) > limit and out:
            break
        out.append(para)
        n += len(para) + 2
    return "\n\n".join(out)
