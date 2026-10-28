"""metis.catalog — the one access to a catalog (catalog-sdk s-bc1718e3; moved into the
runtime's own distribution by metis-packages s-65778b98, with the loader beside it, metis.catalog.manifests,
and the intent generator, metis.intents). Every consumer reads and
writes the catalog here, so a change to the catalog's layout or to domain.yaml touches this
module and no consumer. metis_loop_sdk.catalog and metis_catalog re-export it for one release.

    from metis import catalog
    catalog.root()                  # $METIS_CATALOG, else the cwd (cd DIR && metis-loop: DIR is the catalog)
    catalog.home()                  # $METIS_HOME, else metispy's own metis/catalog: the shared packages
    catalog.find_package("dog")     # <root>/domains/dog, else <home>/domains/dog (the include rule's order)
    catalog.shared()                # METIS_HOME's domains the root lacks: index()/listing() mark them shared
    catalog.state()                 # DIR/.metis/: the loop's own state (git-ignored when DIR is a checkout)
    catalog.installed()             # the installed metis-catalog's own {catalog, tools}: never the root
    catalog.domains()               # [Domain] — every manifest, by name (the loader's Domain)
    catalog.get("dog")              # Domain: .program .serves() .intent .has() …
    catalog.index()                 # every catalog for a console's list, the domains first
    catalog.listing()               # the agent console's rows: the domains, then each partner as its own row
    catalog.readme_head(rel)        # a README's opening
    catalog.intents("dog")          # the domain's intent material: {spec, grammar, prompt, examples, corpus}
    catalog.generator()             # the intent generator (metis.intents) as a module, its ROOT the root
    catalog.fingerprint("dog")      # the stale detection: a hash over what a reload takes
    catalog.plan("dog", program=…)  # what apply would write, with each file's diff
    catalog.apply("dog", program=…, extensions={…}, intents=…, manifest={…}, display=…, line=…)
                                    # the one validated write: never a commit
    catalog.edit("dog", "dog.llp", old, new)   # apply()'s arguments for one exact edit of a file on disk
    catalog.take("lamp", src)       # apply()'s arguments that bring a domain over from another catalog
    catalog.readme("lamp", src, run, summary=…)  # a README for a domain an agentic-loop run made
    catalog.build("lamp")           # the domain's bundle (build/, metispy's emitter)

    python -m metis.catalog list|get|intents [--spec|--skeleton]|fingerprint|apply|edit|take|readme|build|root|state|installed … --json

The TypeScript face and the catalog widget stay metis-loop's (metis_loop_sdk.catalog: TS_DIR,
WIDGET_DIR, widget_file, and its CLI's `ts` and `widget` verbs over this one).
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

STATE = ".metis"                      # DIR/.metis/: the loop's own state in the catalog it runs in

from .manifests import SKIP, Domain, ManifestError, data_dir  # noqa: E402,F401 — the loader's types, here too
import importlib  # noqa: E402


# ── the loader and the root ───────────────────────────────────────────────────

def loader():
    """The catalog's loader, metis.catalog.manifests (was metis_catalog). The one place every
    function here reaches it."""
    return importlib.import_module("metis.catalog.manifests")


def root() -> Path:
    """The catalog root (cwd-root s-ddac2cbf): $METIS_CATALOG, else the
    current directory — as `cd DIR && claude` makes DIR the project, `cd DIR
    && metis-loop` makes DIR the catalog. Any directory is one: an empty
    one is an empty catalog, and domains/ appears at the first apply."""
    c = os.environ.get("METIS_CATALOG")
    return Path(c).resolve() if c else Path.cwd().resolve()


def home() -> Path:
    """METIS_HOME: the one directory of the shared packages (the user's ruling, specialist-tree
    s-404d001f), by default metispy's own metis/catalog (its iface/ interface packages)."""
    h = os.environ.get("METIS_HOME")
    return Path(h).resolve() if h else Path(__file__).resolve().parent


def places(r=None):
    """Where a catalog path is looked for, in order: [(dir, why)] — the catalog root, then METIS_HOME
    (the compiler's include rule after the including file's own directory: metis.lang.compiler
    include_path)."""
    return [(_r(r), "the catalog root"), (home(), "METIS_HOME")]


def find(rel, r=None, kind="file") -> Path:
    """`rel` (a file, or with kind="dir" a directory) under the first of places(): FileNotFoundError
    naming every place searched when it is in none."""
    test = Path.is_dir if kind == "dir" else Path.is_file
    ps = places(r)
    for d, _ in ps:
        if test(d / rel):
            return (d / rel).resolve()
    raise FileNotFoundError("%r: not under %s" % (str(rel), ", ".join("%s (%s)" % (d, why) for d, why in ps)))


def find_package(name, r=None) -> Path:
    """The package `name`: <root>/domains/<name>, else <home>/domains/<name> (metis-packages s-65778b98
    step 3: what --pkg NAME loads and the widget picks); a miss names both places."""
    return find(Path("domains") / name, r, kind="dir")


def package_domains(name, r=None):
    """The domains of the package `name` (its manifests, the package's own and any nested; a
    partner is stepped with its pair, so it is not listed): what --pkg NAME installs."""
    d = find_package(name, r)
    base = next(p for p, _ in places(r) if d.is_relative_to(p))
    return [x for x in loader().domains(base) if x.dir.resolve().is_relative_to(d) and not x.partner_of]


def shared(r=None):
    """METIS_HOME's domains the root `r` (default the root) does not have (a name in both is the
    root's); none when the root is METIS_HOME or METIS_HOME has no domains/."""
    h, r = home(), _r(r)
    if h == r or not (h / "domains").is_dir():
        return []
    own = {d.name for d in loader().domains(r)}
    return [d for d in loader().domains(h) if d.name not in own]


def shared_index(r=None):
    """index() rows of shared(r), marked shared, their path absolute (no .llp, as index's)."""
    names = {d.name for d in shared(r)}
    if not names:
        return []
    return [dict(e, shared=True, path=str(home() / e["path"]), **({"readme": str(home() / e["readme"])}
                                                                    if e.get("readme") else {}))
            for e in loader().index(home()) if e["kind"] == "domain" and e["name"] in names]


def _shared(r):
    """shared() for a call with no root given; a given root sees only itself."""
    return [] if r else shared()


def state(r=None) -> Path:
    """DIR/.metis/, the loop's own state in the catalog DIR it runs in
    (records, sessions, scratch files), made when missing. When DIR is a
    git checkout and its .gitignore does not ignore .metis/ yet, the line
    is added (never a commit)."""
    r = _r(r)
    d = r / STATE
    d.mkdir(exist_ok=True)
    if (r / ".git").exists():
        gi = r / ".gitignore"
        text = gi.read_text() if gi.is_file() else ""
        if not any(ln.strip().strip("/") == STATE for ln in text.splitlines()):
            sep = ("" if text.endswith("\n") else "\n") + "\n" if text else ""
            gi.write_text(text + sep + f"# the metis loops' own state (records, sessions, scratch)\n/{STATE}/\n")
    return d


def _r(r):
    return Path(r).resolve() if r else root()


def domains(r=None):
    """Every domain with a manifest, by name (metis_catalog Domain): the root's (METIS_HOME's shared
    ones are listing()'s and get()'s, marked shared)."""
    return loader().domains(_r(r))


def get(name, r=None):
    """The domain `name`: the root's, else (no root given) METIS_HOME's, as find_package resolves a
    package (KeyError naming both when neither has a manifest of it). A generated member of an image in use
    (metis.catalog.image.use: the derived director, whose source the link wrote) is found first."""
    from .image import USED
    if name in USED and r is None:
        return USED[name]
    try:
        return loader().get(name, _r(r))
    except KeyError:
        if r is not None:
            raise
        for d in _shared(r):
            if d.name == name:
                return d
        raise KeyError("no domain %r in %s" % (name, ", ".join("%s (%s)" % (p, why) for p, why in places()))) from None


def index(r=None):
    """Every catalog of the root for a console's list: the domains first
    (their display line, what they have, the host spec), then every other
    .llp (metis_catalog.index) — none under a dot directory (.metis/, the
    loop's own state; .git/, .venv/, .pi/ …): the root is a workspace too."""
    out = [e for e in loader().index(_r(r)) if not hidden(e.get("path"))]
    if not r:                                  # METIS_HOME's domains, marked shared, after the root's
        rows = shared_index()
        if rows:
            out = sorted(out + rows, key=lambda e: e["kind"] != "domain")
    return out


def walk(pattern, r=None):
    """Every file matching `pattern` under the root, by the loader's own walk
    (never under a dot directory, build/, records/, __pycache__, node_modules/)."""
    return loader().walk(_r(r), pattern)


def hidden(rel) -> bool:
    """A path relative to the root under a dot directory (.metis/, .git/ …):
    never a catalog of the root."""
    return bool(rel) and any(part.startswith(".") for part in Path(rel).parts[:-1])


def listing(r=None):
    """The agent console's rows (py service and Go console alike): the
    index's domains, then each partner (never listed alone in the index) as
    its own row, kind partner."""
    home_ = _shared(r)
    out = [dict({k: e.get(k) for k in ("name", "kind", "path", "display", "line", "has")},
                **({"shared": True} if e.get("shared") else {}))
           for e in index(r) if e["kind"] == "domain"]
    for d in domains(r) + home_:
        if d.partner_of:
            out.append(dict({"name": d.name, "kind": "partner", "path": str(d.dir), "display": d.display,
                             "line": d.line, "has": d.has(), "partner_of": d.partner_of},
                            **({"shared": True} if any(d is x for x in home_) else {})))
    return out


def readme_head(rel, limit=1600, r=None):
    """A README's opening (its title and first paragraphs, cut at a paragraph)."""
    return loader().readme_head(rel, limit, r=_r(r))


def host_of(rel, r=None):
    """The host the index names for the catalog `rel` (its path, no .llp), or None."""
    return next((e["host"] for e in index(r) if e.get("path") == rel), None)


def check_manifest(path, root=None):
    """A domain.yaml read and checked by the loader's own rule (ManifestError):
    its paths inside `root` (else the domains/ tree it is in)."""
    return loader().check_manifest(path, root)


# ── intents ───────────────────────────────────────────────────────────────────

def generator(r=None):
    """The intent generator (metis.intents) as a module reading the root `r`: a
    fresh module per call, its ROOT the root. A catalog's own tools/intents.py is
    a wrapper over metis.intents now (metis-packages step 2), never loaded here."""
    r = _r(r)
    from metis import intents as _intents
    p = Path(_intents.__file__)
    spec = importlib.util.spec_from_file_location("metis_catalog_intents", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ROOT = r
    return mod


def installed() -> dict:
    """The installed metis-catalog's own catalog and tools/ (cwd-root s-ddac2cbf
    step 2): {catalog, tools} — the wheel's (metis_catalog/domains, metis_catalog/tools)
    or, installed -e, its checkout's; where the package is, never $METIS_CATALOG
    nor the cwd. What a loop copies a known domain from (the lamp demo's plant),
    and the directory an RDDL world's host imports rddl_host from when its
    catalog has no tools/ of its own (loop.sh puts it on PYTHONPATH)."""
    own = loader().data_dir()
    if own is None:
        raise ImportError("no metis-catalog installed (its domains): pip install metis-catalog, "
                          "or -e a metis-catalog checkout")
    base = own if (own / "domains").is_dir() else own.parent
    return {"catalog": base, "tools": base / "tools"}


def intents(name, r=None):
    """The domain's intent material: {spec, grammar, prompt, examples:
    {template: path}, corpus} — the paths that exist ({} without intent: {spec})."""
    return get(name, r).intent


def intent_files(name, text, r=None):
    """The files the generator makes of intent.yaml `text` for `name`
    ({file name in intent/: text}); raises when the text does not generate."""
    gen = generator(r)
    with tempfile.TemporaryDirectory(prefix="intents-") as tmp:
        d = Path(tmp) / "domains" / name
        d.mkdir(parents=True)
        (d / "intent.yaml").write_text(text)
        gen.ROOT = Path(tmp)
        try:
            if not isinstance(yaml.safe_load(text), dict):
                raise ValueError("intent.yaml is a mapping")
            return gen.files(name)
        except Exception as e:      # the author's text: any failure is the finding
            raise ValueError(f"intent.yaml does not generate: {type(e).__name__}: {e}") from None


# A world whose program has no command atoms (pred cmd_*) takes no commands. With
# INTENTLESS (the default) it gets no intent.yaml at all: the agents open a domain without
# intents with no verbs (agent-no-intent s-7fd29000, metis-agent a0a19e6). Without it (for
# agents before that), it gets an intent.yaml that knows only UNKNOWN, headed as what it
# is rather than as a draft. $METIS_CATALOG_INTENTLESS=0 turns it off.
INTENTLESS = os.environ.get("METIS_CATALOG_INTENTLESS", "1") != "0"
NO_COMMANDS = ("# {n}'s intents: the world takes no commands (its program has no cmd_* atom), so this\n"
               "# file knows only UNKNOWN. It is here because the domain agent opens no domain without\n"
               "# an intent.yaml; a domain with commands drafts its verbs from them (tools/intents.py skeleton).\n")


# a command atom's declaration, by tools/intents.py skeleton's own rule (`pred cmd_X[(T, …)] :`)
COMMAND = r"^pred\s+(cmd_\w+)(?:\([^)]*\))?\s*:"


def commands(name, r=None):
    """The command atoms of the domain's program (pred cmd_X … :): ["cmd_X", …], by the
    rule the intent generator's skeleton drafts its verbs from (COMMAND)."""
    import re
    prog, _ = _program_of(name, _r(r))
    return re.findall(COMMAND, prog.read_text(), re.M) if prog.is_file() else []


def skeleton(name, r=None, intentless=None):
    """A first intent.yaml drafted from the domain's program (its cmd_* atoms and %% lines);
    for a world with no commands, None with INTENTLESS (`intentless`), else one that knows
    only UNKNOWN (not a draft: NO_COMMANDS is its header)."""
    text = generator(r).skeleton(name)
    if commands(name, r):
        return text
    if INTENTLESS if intentless is None else intentless:
        return None
    body = text.split("\n")
    while body and body[0].startswith("#"):
        body.pop(0)
    return NO_COMMANDS.format(n=name) + "\n".join(body)


# ── the stale detection ──────────────────────────────────────────────────────

def watched(name):
    """What a domain directory holds that a reload takes: the program and its
    extensions, the intents, the host, the overlay, the manifest (metis-agent
    v3reload.go's watched — Go keeps its own, by this rule)."""
    return (Path(name).suffix in (".llp", ".py", ".gbnf") or name in ("intent.yaml", "domain.yaml")
            or name.endswith((".notch.json", "-prompt.txt", "-spoken.txt")))


def snapshot(d):
    """{path relative to the domain directory d: text} of its watched files
    (d, d/intent, d/extensions)."""
    out = {}
    if not d:
        return out
    d = Path(d)
    for sub in (d, d / "intent", d / "extensions"):
        try:
            entries = sorted(sub.iterdir())
        except OSError:
            continue
        for e in entries:
            if e.is_file() and watched(e.name):
                try:
                    out[str(e.relative_to(d))] = e.read_text(errors="replace")
                except OSError:
                    pass
    return out


def fingerprint(name, r=None):
    """A hash over the domain's watched files (names and contents): it
    changes exactly when a reload would read something else."""
    d = get(name, r).dir
    h = hashlib.sha256()
    for k, v in sorted(snapshot(d).items()):
        h.update(k.encode() + b"\0" + hashlib.sha256(v.encode()).digest())
    return h.hexdigest()


def extensions_of(program, r=None):
    """A program file's extensions, by the manifest beside it only
    (runtime-binding Q1): the domain's served files after the program when
    the program is the first of them (Domain.serves()), else none — none for
    a program no manifest names, or a manifest that does not read (mid-write)."""
    program = Path(program).resolve()
    mf = program.parent / "domain.yaml"
    if not mf.is_file():
        return []
    try:
        d = loader().Domain(program.parent.name, program.parent, check_manifest(mf))
        served = d.serves()
    except Exception:                       # noqa: BLE001 — a manifest mid-write, a missing package
        return []
    return list(served[1:]) if served and served[0] == program else []


def _program_of(name, r):
    """(the program file, the domain directory or None) for a domain name or
    a catalog's path relative to the root (no .llp)."""
    if "/" in name or name.endswith(".llp"):
        rel = name[:-4] if name.endswith(".llp") else name
        p = (r / (rel + ".llp")).resolve()
        if r not in p.parents:
            raise ValueError(f"{name!r} resolves outside the catalog root")
        d = p.parent if (p.parent / "domain.yaml").is_file() else None
        return p, d
    lc = loader()
    try:
        dom = lc.get(name, r)
    except KeyError:                       # a new domain: domains/<name>/<name>.llp
        d = r / "domains" / name
        return d / f"{name}.llp", d
    return (dom.program or dom.dir / f"{name}.llp"), dom.dir


def _manifest_text(old, keys):
    """domain.yaml's text with `keys` set: a key it has with another value has
    its line replaced (a one-line key only), a key it lacks is appended; its
    comments and order kept."""
    import re
    lines = old.splitlines(True) if old else []
    have = (yaml.safe_load(old) or {}) if old else {}
    for k, v in keys.items():
        if k in have and have[k] == v:
            continue
        line = "%s: %s\n" % (k, yaml.safe_dump(v, default_flow_style=True, width=1 << 16, allow_unicode=True).strip()
                             .removesuffix("\n...").strip())
        at = [i for i, l in enumerate(lines) if re.match(r"^%s\s*:" % re.escape(k), l)]
        if at:
            i = at[0]
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt[:1] in (" ", "\t", "-"):
                raise ValueError(f"domain.yaml: {k} spans lines; set it by hand")
            lines[i] = line
        else:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines.append(line)
    return "".join(lines)


def _text(p):
    """A file's text as its bytes are: its line ends kept (a CRLF header stays CRLF;
    read_text's universal newlines would make it LF, and take or an edit would write it so)."""
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def _write(p, text):
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _diff(old, new, a, b):
    """A unified diff of two texts, as diff -u writes it: a last line with no newline is
    marked `\\ No newline at end of file`, so the next file's diff, joined after it, does
    not run on from it."""
    return "".join(l if l.endswith("\n") else l + "\n\\ No newline at end of file\n"
                   for l in difflib.unified_diff(old.splitlines(True), new.splitlines(True), a, b))


def _host_file(d, keys, whole):
    """The file a domain's host is in (`file.py:factory`), by the manifest
    plan() writes, else the one there is: beside its domain.yaml, never a
    path out of the domain (`../dog/x.py` would write another domain's file)."""
    spec = (yaml.safe_load(whole) or {}).get("host") if whole is not None else keys.get("host")
    if spec is None and (d / "domain.yaml").is_file():
        spec = (yaml.safe_load((d / "domain.yaml").read_text()) or {}).get("host")
    f, sep, _ = str(spec or "").partition(":")
    if not (sep and f.endswith(".py")):
        raise ValueError(f"{d.name}: a host's text needs a manifest host of the form file.py:factory (it has {spec!r})")
    p = (Path(d) / f).resolve()
    if p.parent != Path(d).resolve():
        raise ValueError(f"{d.name}: the host {f!r} is not beside its domain.yaml: a host's file stays in its domain")
    return p


def _notch_file(d, keys, whole):
    """The file a domain's notch is in, by the manifest plan() writes, else the one
    there is, else <name>.notch.json (set in keys, so the manifest names it): a
    .notch.json beside its domain.yaml."""
    spec = (yaml.safe_load(whole) or {}).get("notch") if whole is not None else keys.get("notch")
    if spec is None and (d / "domain.yaml").is_file():
        spec = (yaml.safe_load((d / "domain.yaml").read_text()) or {}).get("notch")
    if spec is None:
        if whole is not None:
            raise ValueError(f"{d.name}: a notch's text needs the manifest's notch: <file>.notch.json")
        spec = keys["notch"] = f"{d.name}.notch.json"
    p = (Path(d) / str(spec)).resolve()
    if p.parent != Path(d).resolve() or not p.name.endswith(".notch.json"):
        raise ValueError(f"{d.name}: the notch {spec!r} is not a .notch.json beside its domain.yaml")
    return p


def plan(name, program=None, extensions=None, intents=None, manifest=None, readme=None, host=None, corpus=None,
         r=None, display=None, line=None, since=None, as_is=None, included=None, notch=None, carried=None):
    """What apply would write: [{path (relative to the root), new, diff,
    text}] for each file whose text changes. `name` is a domain (or a
    catalog's path, no .llp); program the program's text; extensions
    {name: text} written as <dir>/extensions/<name>.llp; intents
    intent.yaml's text (its generated files in intent/, and intent: {spec}
    named in the manifest); manifest {key: value} set in domain.yaml (or
    its whole text, a str); readme the domain's README.md; host the
    host's Python, written as the file the manifest's host names
    (file.py:factory); corpus intent/<name>-corpus.tsv; display and
    line the manifest's two keys of that name (the console's row: a short
    name, one line on the world). The manifest is checked with the loader's
    check_manifest and the intents must generate before anything is
    planned. A generated intent file whose text is empty is not written
    (a domain's spec without `spoken` has no -spoken.txt).

    since {path: text}: an earlier proposal of a file (the one the person
    refused); a planned file with one gets since_diff, its diff from that
    text, so the person reads what changed since, not the whole file again.

    as_is {path: text}: existing files of the domain directory written back where they
    are, as given (edit()'s write of an extension: extensions= would clean its name and
    write a second file beside it).

    included {path relative to the domain directory: text}: the .llp files the program
    and its extensions include (take()'s include closure), written there.

    notch: the fitted overlay's JSON text, written as the file the manifest's notch
    names (beside domain.yaml, a .notch.json), else <name>.notch.json, which the
    manifest then names.

    carried {path relative to the domain directory: text}: the package's own files take()
    brings as they are (goldens/, cases/, intent/*-spoken.txt), written there after the
    generated intent files, so the package's own text is what lands."""
    import re
    r = _r(r)
    prog, d = _program_of(name, r)
    files = {}
    if program is not None:
        files[prog] = program
    safe = re.compile(r"[^a-z0-9_]+")
    for k, (en, text) in enumerate((extensions or {}).items(), 1):
        stem = safe.sub("_", str(en or f"extension{k}").lower()).strip("_") or f"extension{k}"
        files[prog.parent / "extensions" / f"{stem}.llp"] = text
    for rel_, text in (included or {}).items():
        if d is None:
            raise ValueError(f"{name}: included files belong to a domain directory")
        ip = (Path(d) / rel_).resolve()
        if Path(d).resolve() not in ip.parents or ip.suffix != ".llp":
            raise ValueError(f"{name}: an included file is a .llp inside the domain's directory, not {rel_!r}")
        files[ip] = text
    for ap, text in (as_is or {}).items():
        ap = (Path(ap) if Path(ap).is_absolute() else r / ap).resolve()
        if d is None or Path(d).resolve() not in ap.parents or not ap.is_file():
            raise ValueError(f"{name}: as_is writes back an existing file of the domain's directory, not {ap}")
        files[ap] = text
    whole = manifest if isinstance(manifest, str) else None      # domain.yaml's whole text
    keys = {} if whole is not None else dict(manifest or {})
    shown = {k: v for k, v in (("display", display), ("line", line)) if v is not None}
    for k, v in shown.items():
        if not isinstance(v, str) or not v.strip() or v.splitlines() != [v]:   # any line break (U+2028, U+0085 …)
            raise ValueError(f"{name}: {k} is one line of text")
    if whole is not None and shown:
        whole = _manifest_text(whole, shown)
    keys.update(shown)
    if readme is not None:
        if d is None:
            raise ValueError(f"{name}: a README belongs to a domain directory")
        files[d / "README.md"] = readme
    if corpus is not None:
        if d is None:
            raise ValueError(f"{name}: a corpus belongs to a domain directory")
        files[d / "intent" / f"{d.name}-corpus.tsv"] = corpus
    if host is not None:
        if d is None:
            raise ValueError(f"{name}: a host belongs to a domain directory")
        files[_host_file(d, keys, whole)] = host
    if notch is not None:
        if d is None:
            raise ValueError(f"{name}: a notch belongs to a domain directory")
        try:
            json.loads(notch)
        except ValueError as e:
            raise ValueError(f"{name}: the notch is not JSON ({e})") from None
        files[_notch_file(d, keys, whole)] = notch
    if intents is not None:
        if d is None:
            raise ValueError(f"{name}: intents belong to a domain directory (domains/<name>/)")
        n = d.name
        for fn, text in intent_files(n, intents, r).items():
            if text or (d / "intent" / fn).is_file():     # an empty one that is not there stays away
                files[d / "intent" / fn] = text
        files[d / "intent.yaml"] = intents
        cur = {}
        if (d / "domain.yaml").is_file():
            cur = yaml.safe_load((d / "domain.yaml").read_text()) or {}
        if not isinstance(cur.get("intent"), dict):
            keys.setdefault("intent", {"spec": "intent.yaml"})
    for rel_, text in (carried or {}).items():
        if d is None:
            raise ValueError(f"{name}: carried files belong to a domain directory")
        cp_ = (Path(d) / rel_).resolve()
        if Path(d).resolve() not in cp_.parents or Path(rel_).parts[0] in SKIP:
            raise ValueError(f"{name}: a carried file stays inside the domain's directory, outside {', '.join(SKIP)}: "
                             f"not {rel_!r}")
        files[cp_] = text
    if whole is not None or keys or (d is not None and program is not None and not (d / "domain.yaml").is_file()
                                     and d.parent == r / "domains"):
        if d is None:
            raise ValueError(f"{name}: a manifest belongs to a domain directory")
        mf = d / "domain.yaml"
        if whole is None and not mf.is_file() and "program" not in keys:   # a new domain: its program named
            keys["program"] = prog.relative_to(d).as_posix()
        text = _manifest_text(whole if whole is not None else (_text(mf) if mf.is_file() else ""), keys)
        # checked where it will live: a scratch tree with the package at its
        # depth under the root (domains/… or anywhere: rule 9), so the loader's
        # containment rule (its paths stay inside the catalog root) reads it as it will
        with tempfile.TemporaryDirectory(prefix="manifest-") as tmp:
            t = Path(tmp) / Path(d).resolve().relative_to(r) / "domain.yaml"
            t.parent.mkdir(parents=True)
            t.write_text(text)
            try:
                check_manifest(t, tmp)
            except Exception as e:
                raise ValueError(str(e).replace(str(t), f"{name}/domain.yaml").replace(tmp, str(r))) from None
        files[mf] = text
    out = []
    for p, text in files.items():
        p = Path(p).resolve()
        if r not in p.parents:
            raise ValueError(f"{p} is outside the catalog root {r}")
        old = _text(p) if p.is_file() else ""
        if p.is_file() and old == text:
            continue
        rel = p.relative_to(r).as_posix()
        diff = _diff(old, text, f"a/{rel}", f"b/{rel}")
        f = {"path": rel, "new": not p.is_file(), "diff": diff, "text": text}
        if since and rel in since:
            f["since_diff"] = _diff(since[rel], text, f"refused/{rel}", f"b/{rel}")
        out.append(f)
    return out


def apply(name, program=None, extensions=None, intents=None, manifest=None, readme=None, host=None, corpus=None,
          r=None, display=None, line=None, as_is=None, included=None, notch=None, carried=None):
    """The one validated write into a catalog root (a working copy or a
    scratch catalog): plan()'s files written, never a commit. Returns
    {written: [paths], diff, fingerprint} — the fingerprint of the domain
    after the write (None for a catalog no manifest names)."""
    r = _r(r)
    files = plan(name, program, extensions, intents, manifest, readme, host, corpus, r, display, line, as_is=as_is,
                 included=included, notch=notch, carried=carried)
    for f in files:
        p = r / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        _write(p, f["text"])
    _, d = _program_of(name, r)
    fp = None
    if d is not None and (d / "domain.yaml").is_file():
        fp = fingerprint(d.name, r)
    return {"written": [f["path"] for f in files], "diff": "".join(f["diff"] for f in files), "fingerprint": fp}


# ── one exact edit of a file on disk ─────────────────────────────────────────

def metisc():
    """The metis compiler for an edit's diagnostics: $METISC, else metisc on PATH;
    None when there is none (diagnostics() then refuses with NO_METISC). The link and the build resolve their own
    (metis.catalog.image.metisc: the sibling checkout too)."""
    import shutil
    return os.environ.get("METISC") or shutil.which("metisc")


def metisc_env():
    """The environment for a metisc this process starts: METIS_HOME defaulted to metispy's own (metis/catalog, where
    the shared iface/ packages are), so both toolchains resolve an include in the same three places."""
    env = dict(os.environ)
    env.setdefault("METIS_HOME", str(Path(__file__).resolve().parent))
    return env


NO_METISC = ("no metisc to check a program's edit: set METISC or put metisc on PATH "
             "(build it: make -C <a metis-lang checkout> metisc, then src/metisc)")


def diagnostics(path, base=None):
    """metisc's diagnostics of the program file `path` (an extension's against `base`, the
    program file it extends): [{code, severity, decl, text, line, hint}]. A metisc that
    fails on it (a parse error, a missing include) is one diagnostic, code metisc-failed,
    so a file that does not compile is still compared, and an edit may repair it. Raises
    ValueError without a metisc."""
    import subprocess
    m = metisc()
    if not m:
        raise ValueError(NO_METISC)
    p = subprocess.run([m, "--diagnostics-json", str(path), *(["--base", str(base)] if base else [])],
                       capture_output=True, text=True, timeout=120, env=metisc_env())
    if p.returncode:
        why = (p.stderr or p.stdout).strip().splitlines()
        return [{"code": "metisc-failed", "severity": "error", "decl": None, "line": None, "hint": None,
                 "text": (why[-1] if why else f"exit {p.returncode}").replace(str(path), Path(path).name)}]
    out = []
    for l in p.stdout.splitlines():
        if l.strip().startswith("{"):
            d = json.loads(l)
            loc = d.get("loc") or {}
            out.append({"code": d.get("code"), "severity": d.get("severity"), "decl": loc.get("decl"),
                        "text": d.get("text"), "line": loc.get("line"), "hint": d.get("hint")})
    return out


class _Beside:
    """The text `text` as a dot-file beside `at` (`.<name>.edit-…`, no .llp suffix: no
    reload watches it), so its includes resolve as the original's; removed on exit."""
    def __init__(self, at, text):
        self.at, self.text, self.path = Path(at), text, None

    def __enter__(self):
        fd, t = tempfile.mkstemp(prefix=f".{self.at.name}.edit-", dir=self.at.parent)
        with os.fdopen(fd, "w") as f:
            f.write(self.text)
        self.path = Path(t)
        return self.path

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _added(before, after):
    """The diagnostics of `after` that `before` does not have (a multiset of code,
    severity, declaration and text: shifted lines do not count; metisc-failed by its
    code alone)."""
    from collections import Counter
    key = lambda x: (x["code"], x["severity"]) + ((None, None) if x["code"] == "metisc-failed" else (x["decl"], x["text"]))
    left = Counter(key(x) for x in after) - Counter(key(x) for x in before)
    out = []
    for x in after:
        if left[key(x)] > 0:
            left[key(x)] -= 1
            out.append(x)
    return out


def edited(text, old, new, replace_all=False, what="the file"):
    """`text` with `old` replaced by `new`: exactly one occurrence, or every one with
    replace_all. Refused (ValueError) when old is empty, is not there, is there more than
    once without replace_all, or old == new."""
    if not old:
        raise ValueError(f"{what}: old_string is empty (an edit replaces text that is there)")
    if old == new:
        raise ValueError(f"{what}: old_string and new_string are the same: nothing to change")
    k = text.count(old)
    if k == 0:
        raise ValueError(f"{what}: old_string is not in it (it must match exactly, whitespace and all)")
    if k > 1 and not replace_all:
        raise ValueError(f"{what}: old_string occurs {k} times: give more of the text around it, or replace_all")
    return text.replace(old, new)


def edit(name, path, old, new, replace_all=False, r=None):
    """apply()'s arguments for one exact edit of a file of the domain `name` as it is on
    disk: `path` relative to the domain directory (the program, an extensions/*.llp,
    README.md, intent.yaml, domain.yaml, the host's file, the corpus), old replaced by new
    (edited()). It never goes through formalize or revise: every other byte, comments and
    all, stays. A .llp edit adds no diagnostic (metisc's, before and after; an extension
    against the program it extends). Then `apply(name, **edit(…))`: the program
    and intent.yaml go through apply's own checks, intent/ generated again."""
    r = _r(r)
    dom = get(name, r)
    d = Path(dom.dir).resolve()
    p = (d / path).resolve()
    if d not in p.parents:
        raise ValueError(f"{name}: {path!r} is not inside the domain's directory")
    rel = p.relative_to(d).as_posix()
    if rel.startswith("intent/") and rel != f"intent/{name}-corpus.tsv":
        raise ValueError(f"{name}: {rel} is generated from intent.yaml: edit intent.yaml")
    if not p.is_file():
        raise ValueError(f"{name}: there is no {rel} to edit (a new file is written by apply)")
    prog = Path(dom.program).resolve() if dom.program else None
    exts = {Path(e).resolve(): Path(e).stem for e in dom.extensions()}
    host = None
    if dom.manifest.get("host"):
        try:
            host = _host_file(d, {}, (d / "domain.yaml").read_text())
        except ValueError:
            host = None
    old_text = _text(p)
    if old and "\r" not in old and old not in old_text and old.replace("\n", "\r\n") in old_text:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")   # a CRLF file, the edit given in LF
    text = edited(old_text, old, new, replace_all, f"{name}/{rel}")
    inc = includes(d, [f for f in [prog, *exts] if f is not None])
    if p == prog:
        kw = {"program": text}
    elif p in exts:
        kw = {"as_is": {p: text}}       # written back where it is (plan's extensions= would clean its name)
    elif rel == "README.md":
        kw = {"readme": text}
    elif rel == "intent.yaml":
        kw = {"intents": text}
    elif rel == "domain.yaml":
        kw = {"manifest": text}
    elif p == host:
        kw = {"host": text}
    elif rel == f"intent/{name}-corpus.tsv":
        kw = {"corpus": text}
    elif rel in inc:
        kw = {"included": {rel: text}}          # a file the program includes (rddl's nav_domain.llp)
    else:
        raise ValueError(f"{name}: {rel} is not one of the domain's files an edit changes (its program, "
                         "extensions/*.llp, the files they include, README.md, intent.yaml, domain.yaml, its host, its corpus)")
    if p.suffix == ".llp":
        found = []   # (the file, its added diagnostics)
        with _Beside(p, text) as cand:
            if p in exts:                                   # an extension, against the program it extends
                found.append((rel, _added(diagnostics(p, prog), diagnostics(cand, prog))))
            else:
                found.append((rel, _added(diagnostics(p), diagnostics(cand))))
                if p == prog and not found[0][1]:           # the program's extensions, against the edited program
                    for e in exts:
                        found.append((e.relative_to(d).as_posix() if d in e.parents else str(e),
                                      _added(diagnostics(e, prog), diagnostics(e, cand))))
                if rel in inc and prog is not None and not found[0][1]:
                    # the program that includes it, with the edited file in its place: the domain's .llp
                    # files mirrored (their includes are the domain's own), the one replaced
                    with tempfile.TemporaryDirectory(prefix="edit-") as tmp:
                        for f in d.rglob("*.llp"):
                            t = Path(tmp) / f.relative_to(d)
                            t.parent.mkdir(parents=True, exist_ok=True)
                            _write(t, text if f.resolve() == p else _text(f))
                        found.append((prog.relative_to(d).as_posix(),
                                      _added(diagnostics(prog), diagnostics(Path(tmp) / prog.relative_to(d)))))
        found = [(f, a) for f, a in found if a]
        if found:
            n = sum(len(a) for _, a in found)
            raise ValueError(f"{name}/{rel}: the edit adds {n} diagnostic(s), so it is refused:\n" + "\n".join(
                f"  {f}: {x['severity']} {x['code']}" + (f" @ line {x['line']}" if x.get("line") else "")
                + (f" ({x['decl']})" if x.get("decl") else "") + f": {x['text']}"
                + (f" — {x['hint']}" if x.get("hint") else "") for f, a in found for x in a))
    return kw


# what take() carries: the manifest's keys whose files it brings, and the ones with no file
TAKEN = ("program", "extensions", "host", "intent", "notch", "display", "line", "voice", "clock", "reload", "records")
CARRIED = ("goldens", "cases")      # a package's own directories take() brings whole, as they are


def take(name, src, r=None, readme=None):
    """apply()'s arguments that bring the domain `name` over from the catalog
    `src` (a scratch catalog: the agentic loop's) into the root r: its
    domain.yaml as it is, the program, the extensions, the files they
    include (includes(): inside the domain directory only), the host, the
    notch (its fitted overlay), intent.yaml (apply generates intent/ again)
    and a corpus, its README (or `readme`), and as they are (`carried`) its goldens/ (the
    package's contract: what both runtimes are held to), its cases/ (its hand-written
    specialist tests) and intent/*-spoken.txt. Never records/ or build/ (the agents' own) or
    another document beside its README (a transcript). Refused when r has the domain already, or the domain names a
    file take does not carry (serves, agent, pair: bring those by hand) or keeps one elsewhere
    than its directory's layout. Then `apply(name, **take(name, src, r))`."""
    r, src = _r(r), Path(src).resolve()
    s = get(name, src)
    if (r / "domains" / name).exists():
        raise ValueError(f"{name}: {r} has domains/{name} already")
    if name in {x.name for x in domains(r)}:              # a package of that name anywhere under r (rule 9)
        raise ValueError(f"{name}: {r} has a package {name} already ({get(name, r).dir})")
    if s.dir.resolve() != src / "domains" / name:         # a nested one (domains/ceptre/<name>) keeps its license beside it
        raise ValueError(f"{name}: take brings a domain from domains/{name}/ (it is at {s.dir})")
    other = sorted(set(s.manifest) - set(TAKEN))
    if other:
        raise ValueError(f"{name}: its manifest names {', '.join(other)}, which take does not carry")
    d = s.dir.resolve()
    if s.program is None or Path(s.program).resolve() != d / f"{name}.llp":
        named = (Path(s.program).relative_to(d).as_posix() if s.program and d in Path(s.program).resolve().parents
                 else s.program)
        raise ValueError(f"{name}: take carries a domain's program only as {name}.llp beside its domain.yaml, and "
                         f"its manifest names {named}: rename the file to domains/{name}/{name}.llp and set "
                         f"program: {name}.llp in its domain.yaml (by hand, in the catalog it is taken from)")
    ext = {}
    for e in s.extensions():
        e = Path(e).resolve()
        if e.parent != d / "extensions":
            raise ValueError(f"{name}: the extension {e} is not in extensions/")
        ext[e.stem] = _text(e)
    host = None
    if s.manifest.get("host"):
        host = _text(_host_file(d, {}, (d / "domain.yaml").read_text()))
    notch = None
    if s.manifest.get("notch"):
        nf = _notch_file(d, {}, (d / "domain.yaml").read_text())
        if not nf.is_file():
            raise ValueError(f"{name}: its manifest names the notch {s.manifest['notch']}, which is not there")
        notch = _text(nf)
    included = includes(d, [Path(s.program).resolve(), *(Path(e).resolve() for e in s.extensions())])
    it, cp = d / "intent.yaml", d / "intent" / f"{name}-corpus.tsv"
    if readme is None and (d / "README.md").is_file():
        readme = _text(d / "README.md")
    carried = {f.relative_to(d).as_posix(): _text(f) for f in [
        *(f for sub in CARRIED for f in sorted((d / sub).rglob("*")) if f.is_file() and "__pycache__" not in f.parts),
        *sorted((d / "intent").glob("*-spoken.txt"))]}
    return dict(program=_text(s.program), extensions=ext, manifest=_text(d / "domain.yaml"),
                intents=_text(it) if it.is_file() else None, host=host,
                corpus=_text(cp) if cp.is_file() else None, readme=readme, included=included or None,
                notch=notch, carried=carried or None)


def header(prog):
    """A program's leading `%` block, verbatim: its comment lines (and the blank lines
    between them) before the first line that is neither; `%%` (a rule's doc) ends it."""
    out = []
    for l in prog.splitlines():
        if l.startswith("%") and not l.startswith("%%"):
            out.append(l)
        elif not l.strip():
            if out:
                out.append(l)
        else:
            break
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def intents_status(name, r=None):
    """One paragraph on the domain's intents, from its files: the verbs intent.yaml maps
    to (a DRAFT said so), only UNKNOWN for a world with no commands, or no intent.yaml."""
    d = Path(get(name, r).dir)
    it = d / "intent.yaml"
    cmds = commands(name, r)
    if not it.is_file():
        if cmds:
            return (f"The program has command atoms ({', '.join(f'`{c}`' for c in cmds)}) but no intent.yaml yet: "
                    f"draft one with `python3 tools/intents.py skeleton {name}`.")
        return ("The world takes no commands (its program has no `cmd_*` atom), so it has no intent.yaml: "
                "the agents open it with no verbs.")
    text = it.read_text()
    try:
        spec = yaml.safe_load(text) or {}
    except yaml.YAMLError as e:
        return f"`intent.yaml` does not read as YAML ({type(e).__name__}): `python3 tools/intents.py check {name}`."
    verbs = []
    for v in (spec.get("verbs") if isinstance(spec, dict) else None) or []:
        verb = v.get("verb") if isinstance(v, dict) else None
        for x in (verb if isinstance(verb, list) else [verb]):
            if isinstance(x, str) and x and x != "UNKNOWN":
                verbs.append(x)
    if not verbs:
        return ("The world takes no commands (its program has no `cmd_*` atom). `intent.yaml` knows only UNKNOWN: "
                "it is there because the domain agent opens no domain without one.")
    draft = "a DRAFT" in text.split("\n", 1)[0]
    return (f"`intent.yaml` maps what the person says to the verbs {', '.join(f'`{v}`' for v in verbs)} (and UNKNOWN)."
            + (" It is the loop's first draft (its DRAFT header): write the meanings and more examples, then "
               f"`python3 tools/intents.py gen {name}`." if draft else "")
            + f" Its check: `python3 tools/intents.py check {name}`.")


INCLUDE = r'^\s*include\s+"([^"]+)"\s*\.'


def includes(d, files):
    """{path relative to the domain directory d: text} of every file the .llp `files`
    include, transitively (`include "x.llp".`, resolved beside the including file). An
    include out of d is refused (ValueError): take carries a domain's own files only."""
    import re
    d, out, todo, seen = Path(d).resolve(), {}, list(files), set()
    while todo:
        f = Path(todo.pop()).resolve()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        for m in re.finditer(INCLUDE, f.read_text(), re.M):
            inc = (f.parent / m.group(1)).resolve()
            if d not in inc.parents:
                raise ValueError(f"{d.name}: {f.name} includes {m.group(1)!r}, outside its domain directory: "
                                 "take carries a domain's own files only")
            if not inc.is_file():
                raise ValueError(f"{d.name}: {f.name} includes {m.group(1)!r}, which is not there")
            out[inc.relative_to(d).as_posix()] = _text(inc)
            todo.append(inc)
    return out


def readme(name, src, run=None, summary=None, title=None):
    """A README for the domain `name` of the catalog `src`, written from what an
    agentic-loop run left. The SDK writes every mechanical part: the title line, where it
    came from (from the run directory `run`, loop.sh --scratch DIR: records/formalize-
    calls.jsonl's model and cost, or seed.json's seed), the program's own header (its
    leading `%` block, verbatim), its extensions' rules, how to run the case (its `#`
    lines) and the REPL, and the intents' status. A model writes only `summary` (the
    paragraph after the title) and `title` (what the world is, after "<name> — "). For
    take(readme=…) when the domain has none of its own."""
    import re
    src = Path(src).resolve()
    run = Path(run).resolve() if run else None
    d = get(name, src).dir                 # anywhere under src (rule 9)
    prog = (d / f"{name}.llp").read_text()
    m = re.search(r'provenance text "((?:[^"\\]|\\.)*)"', prog)
    prose = m.group(1) if m else ""
    calls = run / "records" / "formalize-calls.jsonl" if run else None
    rows = [json.loads(l) for l in calls.read_text().splitlines() if l.strip()] if calls and calls.is_file() else []
    seed = run / "seed.json" if run else None
    seed = json.loads(seed.read_text()) if seed and seed.is_file() else None
    title = (title or "").strip()
    if title and title.splitlines() != [title]:
        raise ValueError(f"{name}: a README's title is one line")
    first = prose.split(". ")[0].rstrip(".").strip()
    if title:
        head = f"# {name} — {title}"
    elif first:
        head = f"# {name} — {first[0].lower() + first[1:]}"
    else:
        head = f"# {name}"
    out = [head, ""]
    if summary and summary.strip():
        out += [summary.strip(), ""]
    label = f", run `{run.name}`" if run else ""
    if seed:
        how = (f"seeded from `{seed['seed']}`: the program as it was (its header keeps its origin), made a "
               "catalog domain around it")
    elif rows:
        how = (f"the kernel's formalize drafted the program on `{rows[0]['model']}` (formalize: {len(rows)} calls, "
               f"{sum(r.get('cost') or 0 for r in rows):.4f} USD; pi's own calls are on the key's usage)")
    else:
        how = "the kernel's formalize drafted the program"
    out += [f"**Where it came from:** the agentic loop (metis-agentic-loop){label}: pi drove, and {how}. "
            "Every file was approved at pi's apply gate, and taken into this catalog from the run's.", ""]
    host = (d / f"{name}_host.py").is_file()
    out += [f"**What:** `{name}.llp`" + (" formalizes the person's prose (its provenance line)." if prose else
                                          " is the program.")
            + (f" `{name}_host.py` is the world, written before any program: the loop exercised the program against"
               " it, and the extensions explain what the world did that the program did not say." if host else ""), ""]
    h = header(prog)
    if h:
        out += ["## The program's header", "", f"As it is at the top of `{name}.llp`:", "", "```text", h, "```", ""]
    ext = sorted((d / "extensions").glob("*.llp")) if (d / "extensions").is_dir() else []
    if ext:
        out += ["## The extensions", ""]
        for e in ext:
            rules = [l.strip() for l in e.read_text().splitlines() if re.match(r"\s*\w+\s*\[\w+\]\s*:", l)]
            out.append(f"- `extensions/{e.name}`: " + "; ".join(f"`{r}`" for r in rules))
        out.append("")
    out += ["## Run it", ""]
    case = [l.rstrip() for l in prog.splitlines() if re.match(r"#[a-z]", l)]
    if case:
        out += [f"`metisc --run {name}.llp` runs the case its `#` lines give:", "", "```text", *case, "```", ""]
    out.append(f"- the program in the REPL: `metis {'--plugin ' + name + '_host:make ' if host else ''}{name}.llp`;")
    out += ["- the loop console (metis-loop `run.sh`) and the agents serve the program with its extensions "
            "(domain.yaml: `extensions/*.llp` in name order); the console lists it at its next catalog read.", ""]
    out += ["## Intents", "", intents_status(name, src)]
    return "\n".join(out) + "\n"


def build(name, r=None):
    """The package's build (its build/, git-ignored): a COMPOSITION package (composition.yaml in its directory) is
    linked into an image by `metisc build` (spec/17 §7; the stale members recompiled), its one line
    `image NAME v-HASH DIR`; a lone domain gets its bundle by metispy's emitter in a child (python -m
    metis.kernel.artifact), its one-line account. Raises ValueError with the compiler's or the emitter's error."""
    import subprocess
    r = _r(r)
    try:
        pkg = find_package(name, r)
    except FileNotFoundError:
        pkg = None
    if pkg is not None and (pkg / "composition.yaml").is_file():
        from . import image
        d = image.build(pkg)
        return "image %s v-%s %s" % (d.name.split(".image.")[0], d.name.rpartition(".v-")[2], d)
    get(name, r)
    p = subprocess.run([sys.executable, "-m", "metis.kernel.artifact", name], cwd=r, capture_output=True, text=True,
                       env={**os.environ, "METIS_CATALOG": str(r)}, timeout=600)
    if p.returncode:
        raise ValueError(f"{name}: the bundle did not build: {(p.stderr or p.stdout).strip()[-600:]}")
    return p.stdout.strip()


# ── JSON forms (the CLI) ─────────────────────────────────────────────────────

def _jsonable(o):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def describe(name, r=None):
    """One domain as JSON: its manifest and what the loader derives of it."""
    d = get(name, r)
    return _jsonable({"name": d.name, "dir": d.dir, "manifest": d.manifest, "display": d.display, "line": d.line,
                      "voice": d.voice, "program": d.program, "extensions": d.extensions(), "serves": d.serves(),
                      "host": d.host_spec_for(), "notch": d.notch, "intent": d.intent, "records": d.records,
                      "clock": d.clock, "pair": d.pair, "partner_of": d.partner_of, "readme": d.readme,
                      "has": d.has(), "fingerprint": fingerprint(name, r)})


def main(argv=None, extra=None, prog="python -m metis.catalog"):
    """The CLI. `extra` {verb: (help, fn)}: a consumer's own verbs beside these (metis-loop's
    `ts` and `widget`: its TypeScript face and its catalog widget), each answered by fn()."""
    extra = dict(extra or {})
    ap = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="the catalog root (default: $METIS_CATALOG, else the current directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="every catalog, the domains first (index); --consoles: the agent console's rows")
    ls.add_argument("--domains", action="store_true", help="the domains only")
    ls.add_argument("--consoles", action="store_true", help="the agent console's rows (partners as their own rows)")
    g = sub.add_parser("get", help="one domain: its manifest and what the loader derives")
    g.add_argument("name")
    it = sub.add_parser("intents", help="the domain's intent material; --spec: with intent.yaml read; "
                                        "--skeleton: a first intent.yaml")
    it.add_argument("name")
    it.add_argument("--skeleton", action="store_true")
    it.add_argument("--spec", action="store_true", help="intent.yaml itself, read (null without one)")
    fp = sub.add_parser("fingerprint", help="the hash over what a reload takes")
    fp.add_argument("name")
    a_ = sub.add_parser("apply", help="the one validated write (never a commit); --dry-run: the plan and its diff")
    a_.add_argument("name", help="a domain, or a catalog's path relative to the root (no .llp)")
    a_.add_argument("--program", metavar="FILE", help="the program's text from FILE (- = stdin)")
    a_.add_argument("--extension", action="append", default=[], metavar="NAME=FILE",
                    help="an extension, written as extensions/NAME.llp")
    a_.add_argument("--intents", metavar="FILE", help="intent.yaml's text from FILE (its files generated)")
    a_.add_argument("--intents-skeleton", action="store_true", help="intent.yaml drafted from the program")
    a_.add_argument("--manifest", metavar="JSON", help="domain.yaml keys to set, as a JSON object")
    a_.add_argument("--readme", metavar="FILE", help="the domain's README.md from FILE")
    a_.add_argument("--host", metavar="FILE", help="the host's Python from FILE, written as the file the manifest's host names")
    a_.add_argument("--corpus", metavar="FILE", help="intent/<name>-corpus.tsv from FILE")
    a_.add_argument("--display", metavar="TEXT", help="domain.yaml's display: the world's short name")
    a_.add_argument("--line", metavar="TEXT", help="domain.yaml's line: one line on what the world is")
    a_.add_argument("--dry-run", action="store_true")
    ed = sub.add_parser("edit", help="one exact edit of a domain's file as it is on disk (never formalize); "
                                     "--dry-run: the plan and its diff")
    ed.add_argument("name")
    ed.add_argument("path", help="the file, relative to the domain directory (its program, extensions/X.llp, "
                                 "README.md, intent.yaml, domain.yaml, the host, the corpus)")
    ed.add_argument("--old", required=True, metavar="FILE", help="the text to replace, from FILE (- = stdin)")
    ed.add_argument("--new", required=True, metavar="FILE", help="its replacement, from FILE (- = stdin)")
    ed.add_argument("--replace-all", action="store_true", help="every occurrence (else exactly one)")
    ed.add_argument("--dry-run", action="store_true")
    tk = sub.add_parser("take", help="bring a domain over from another catalog (a scratch one) into the root; "
                                     "--dry-run: the plan and its diff")
    tk.add_argument("name")
    tk.add_argument("--from", dest="src", required=True, metavar="ROOT", help="the catalog the domain is in")
    tk.add_argument("--readme", metavar="FILE", help="its README.md from FILE (- = stdin), else the one it has")
    tk.add_argument("--readme-from-run", metavar="RUN_DIR",
                    help="without a README of its own: readme() from the agentic-loop run in RUN_DIR")
    tk.add_argument("--dry-run", action="store_true")
    rd = sub.add_parser("readme", help="readme(): a README for a domain an agentic-loop run made (from ROOT, the run in DIR)")
    rd.add_argument("name")
    rd.add_argument("--from", dest="src", metavar="ROOT", help="the catalog the domain is in (default: the root)")
    rd.add_argument("--run", metavar="DIR", help="the run's directory (records/formalize-calls.jsonl, seed.json)")
    rd.add_argument("--summary", metavar="FILE", help="the paragraph after the title, from FILE (- = stdin)")
    rd.add_argument("--title", metavar="TEXT", help="what the world is, after '<name> — '")
    bd = sub.add_parser("build", help="the package's build: a composition's image (metisc build), else the domain's "
                                      "bundle (build/, metispy's emitter)")
    bd.add_argument("name")
    lk = sub.add_parser("link", help="metisc --link DIR/composition.yaml: the image (DIR/build, or -o OUT); --check: "
                                     "the listing only")
    lk.add_argument("dir")
    lk.add_argument("-o", dest="out", metavar="OUT", help="the image's parent directory (default: DIR/build)")
    lk.add_argument("--check", action="store_true", help="the link's checks and the image line, nothing written")
    cs = sub.add_parser("cases", help="the domain's own cases (cases/*.yaml) through the interface stub: the "
                                      "specialist alone, its manifest's interface closed by the cases' facts "
                                      "(exit 1 when one fails)")
    cs.add_argument("name")
    gd = sub.add_parser("golden", help="the domain's golden (goldens/: day.jsonl and intents.jsonl in, their "
                                       ".out.jsonl written by metis.domain.day); --check: regenerate and diff "
                                       "(exit 1 on a difference)")
    gd.add_argument("name")
    gd.add_argument("--check", action="store_true")
    rt = sub.add_parser("root", help="the catalog root")
    hm = sub.add_parser("home", help="METIS_HOME: the shared packages")
    fd = sub.add_parser("find", help="a package: <root>/domains/NAME, else <home>/domains/NAME; its domains")
    fd.add_argument("name")
    st = sub.add_parser("state", help="the root's .metis/ (the loop's own state), made when missing")
    inst = sub.add_parser("installed", help="the installed metis-catalog's own catalog and tools/ (never the root)")
    more = [sub.add_parser(v, help=h) for v, (h, _) in extra.items()]
    for p in (ap, ls, g, it, fp, a_, ed, tk, rd, bd, lk, cs, gd, rt, hm, fd, st, inst, *more):     # --json after any verb
        p.add_argument("--json", action="store_true", help="JSON on stdout")
    for p in (a_, ed, tk):
        p.add_argument("--since", metavar="FILE", help="a JSON object {path: text}: earlier proposals (refused); "
                                                      "--dry-run's files then carry since_diff, the diff from each")
    a = ap.parse_args(argv)
    r = Path(a.root).resolve() if a.root else None        # --root wins over $METIS_CATALOG (root() reads it first)

    def read(f):
        return sys.stdin.read() if f == "-" else Path(f).read_text()

    def dry(name, kw):
        since = json.loads(read(a.since)) if getattr(a, "since", None) else None
        files = plan(name, **kw, since=since)
        return {"dry_run": True, "files": files, "diff": "".join(f["diff"] for f in files)}

    try:
        if a.cmd == "list":
            rows = listing(r) if a.consoles else [e for e in index(r) if not a.domains or e["kind"] == "domain"]
            if not a.json:
                print("# %s" % _r(r))
                for e in rows:
                    has = " ".join(k for k, v in e["has"].items() if v and k != "program")
                    print("%-7s %-16s %-44s %-28s %s" % (e["kind"], e["name"], e.get("path") or "—", has,
                                                         (e.get("line") or "")[:70]))
                return 0
            out = rows
        elif a.cmd == "get":
            out = describe(a.name, r)
        elif a.cmd == "intents":
            if a.skeleton:
                out = {"name": a.name, "skeleton": skeleton(a.name, r), "commands": commands(a.name, r)}
            else:
                it_ = intents(a.name, r)
                out = {"name": a.name, "intent": _jsonable(it_)}
                if a.spec:
                    sp = it_.get("spec") or get(a.name, r).dir / "intent.yaml"
                    out["spec"] = yaml.safe_load(sp.read_text()) if sp.is_file() else None
        elif a.cmd == "fingerprint":
            out = {"name": a.name, "fingerprint": fingerprint(a.name, r)}
        elif a.cmd == "root":
            out = {"root": str(_r(r))}
        elif a.cmd == "home":
            out = {"home": str(home())}
        elif a.cmd == "find":
            d = find_package(a.name, r)
            out = {"name": a.name, "dir": str(d), "from": "root" if d.is_relative_to(_r(r)) else "home",
                   "domains": [x.name for x in package_domains(a.name, r)]}
        elif a.cmd == "state":
            out = {"state": str(state(r))}
        elif a.cmd == "installed":
            out = {k: str(v) for k, v in installed().items()}
        elif a.cmd in extra:
            out = extra[a.cmd][1]()
        elif a.cmd == "readme":
            out = {"name": a.name, "readme": readme(a.name, a.src or _r(r), a.run,
                                                    read(a.summary) if a.summary else None, a.title)}
        elif a.cmd == "edit":
            if a.old == "-" and a.new == "-":
                raise ValueError("edit: one text from stdin (-) at most")
            kw = edit(a.name, a.path, read(a.old), read(a.new), a.replace_all, r)
            out = dry(a.name, {**kw, "r": r}) if a.dry_run else apply(a.name, r=r, **kw)
        elif a.cmd == "build":
            out = {"name": a.name, "bundle": build(a.name, r)}
        elif a.cmd == "link":
            from . import image
            text, d = image.link(a.dir, a.out, a.check)       # (not `listing`: that is the module's function)
            if not a.json:
                print(text, end="")
                return 0
            out = {"dir": str(Path(a.dir).resolve()), "listing": text, "image": str(d) if d else None}
        elif a.cmd == "cases":
            from . import stub
            res = stub.run_package(get(a.name, r))
            print(json.dumps([c.as_dict() for c in res], indent=1) if a.json else stub.render(res))
            return 0 if all(c.passed for c in res) else 1
        elif a.cmd == "golden":
            from metis.domain import day
            d = get(a.name, r)
            if a.check:
                bad = day.check(d)
                if a.json:
                    print(json.dumps({"name": a.name, "ok": not bad, "problems": [{"file": f, "problem": w} for f, w in bad]}))
                else:
                    for f, w in bad:
                        print("%s goldens/%s: %s" % (a.name, f, w))
                    if not bad:
                        print("%s: the goldens are what the package makes" % a.name)
                return 1 if bad else 0
            files = day.golden(d)
            if not files:
                raise ValueError(f"{a.name}: no goldens/day.jsonl or goldens/intents.jsonl to run")
            for f, text in files.items():
                (d.dir / day.GOLDENS / f).write_text(text)
            out = {"name": a.name, "wrote": sorted(str(d.dir / day.GOLDENS / f) for f in files)}
        elif a.cmd == "take":
            if a.readme and a.readme_from_run:
                raise ValueError("take: --readme or --readme-from-run, not both")
            text = read(a.readme) if a.readme else None
            if a.readme_from_run and not (get(a.name, Path(a.src).resolve()).dir / "README.md").is_file():
                text = readme(a.name, a.src, a.readme_from_run)
            kw = take(a.name, a.src, r, text)
            if a.dry_run:
                out = dry(a.name, {**kw, "r": r})
            else:
                out = apply(a.name, r=r, **kw)
        else:
            ext, ext_files = {}, []
            for e in a.extension:
                n, sep, f = e.partition("=")
                if not sep:
                    raise ValueError(f"--extension {e!r}: NAME=FILE")
                ext_files.append(f)
            if sum(1 for f in [a.program, a.intents, a.readme, a.host, a.corpus, *ext_files] if f == "-") > 1:
                raise ValueError("apply: one text from stdin (-) at most")
            for e, f in zip(a.extension, ext_files):
                ext[e.partition("=")[0]] = read(f)
            itext = read(a.intents) if a.intents else None
            if a.intents_skeleton:
                itext = skeleton(a.name, r)
                if itext is None:
                    raise ValueError(f"{a.name}: the world takes no commands (its program has no cmd_* atom): "
                                     "no intent.yaml (the agents open it with no verbs)")
            kw = dict(program=read(a.program) if a.program else None, extensions=ext, intents=itext,
                      manifest=json.loads(a.manifest) if a.manifest else None,
                      readme=read(a.readme) if a.readme else None, host=read(a.host) if a.host else None,
                      corpus=read(a.corpus) if a.corpus else None, r=r, display=a.display, line=a.line)
            if a.dry_run:
                out = dry(a.name, kw)
            else:
                out = apply(a.name, **kw)
    except (KeyError, ValueError, FileNotFoundError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        if a.json:
            json.dump({"error": msg}, sys.stdout)
            print()
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 1
    if a.json or a.cmd != "list":
        json.dump(out, sys.stdout, indent=1, default=str)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
