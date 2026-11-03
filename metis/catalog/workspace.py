"""metis.catalog.workspace — the workspace (metis-loop docs/WORKSPACE.md §2, loop-workspace s-5b4be5fa step 1):
one root DIR that IS the catalog (DIR/domains/), holds at most one bundle (a composition) and keeps the loops'
state in DIR/.metis/. The backend (the human loop's gateway), the pi loop and the judge never talk to each other:
they meet here, in files, and the UI tails one journal.

    from metis.catalog.workspace import workspace
    ws = workspace(DIR=None)          # DIR, else metis.catalog.root(): one object, no daemon
    ws.root, ws.state                 # DIR, DIR/.metis/ (made by the first write, through metis.catalog.state:
                                      #   git-ignored in a checkout; a read never makes it)
    ws.bundle(name=None)              # {name, composition, members, image, golden, config, build, stale, why} or None
    ws.context(patterns=None)         # the text a brain is given first (paths, never the patterns' text)
    ws.traces(), ws.trace(run)        # .metis/traces/<run>/ (metis.trace's Segments layout as it is)
    ws.surprises(status=None)         # .metis/surprises/<id>.json, each with its status: open | taken | ruled
    ws.drop_surprise(rec)             # the backend's write: → the id
    ws.take_surprise(id, pid=None)    # the pi loop's: .metis/surprises/<id>.taken
    ws.rule(id, ruling, verdict, …)   # the pi loop's (the judge's document): .metis/rulings/<id>.md + .json
    ws.rulings()                      # the rulings, each with its verdict and fields
    ws.journal(who, kind, **ptrs)     # append one row to .metis/loop/journal.jsonl (pointers, never bodies)
    ws.events(since=0)                # the journal's rows from row SINCE on (the UI keeps the count it read)
    ws.cost()                         # .metis/cost.jsonl summed: total, calls, by model, by who
    ws.loops()                        # {pi: lock or None, backend: lock or None} (§2.3)
    ws.lock(kind, pid=None, **f)      # one pi loop, one backend: a second is refused with the holder's line
    ws.unlock(kind, pid=None)
    tools(), tool_argv(name, args)    # the trace verbs and lint as declared tools (metis/trace_tools.json, §3.1)

    ws.marks(run)                     # a run's marks as windows (traces/<run>/marks.jsonl: the world's rows, at_k)
    ws.keep_window(run, k0, k1, name) # the window copied to traces/<run>-<name>/, past the writer's ring: its path

    python -m metis.catalog.workspace [--root DIR] layout|bundle|context|traces|trace|marks|keep|surprises|drop|take|rule|
                                      rulings|journal|events|cost|loops|lock|unlock|tools|argv … --json
    (a CLI lock/take's pid defaults to the caller's: the CLI's parent, a shell or the node face)

The layout lives ONCE, in LAYOUT below: this module's paths are read from it, and the TypeScript face (metis-loop
sdk/ts/workspace.ts) gets it from `layout --json`. The rules (WORKSPACE.md §2.2): every file is written whole then
renamed (no partial reads); ids are YYYYMMDDTHHMMSS-<6 hex>; a status is a file's existence (.taken, a ruling),
never a field edited in place; the journal is the one append-only file, and it carries pointers, never bodies.
A world runner's own run dir under .metis/ (the road's .metis/road/) is never read or written here.
"""
import argparse
import json
import os
import re
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from metis import catalog

FORMAT = "metis-workspace/1"
WHO = ("backend", "pi", "judge", "world")
LOCKS = ("pi", "backend")
ID = r"^\d{8}T\d{6}-[0-9a-f]{6}$"
STATUS = ("open", "taken", "ruled")
# The face: the names both faces have, py's (Workspace's methods, this module's functions) and the TypeScript
# face's (metis-loop sdk/ts/workspace.ts), one for one: its test reads them from layout()["face"].
FACE = ("layout", "root", "path", "bundle", "context", "traces", "trace", "surprises", "surprise", "drop_surprise",
        "take_surprise", "rule", "rulings", "journal", "events", "cost", "loops", "lock", "unlock", "tools",
        "tool_argv", "marks", "keep_window")

# The layout: name → (path under .metis/, kind, who writes it, what it is). {id}, {run}, {kind} are filled in.
LAYOUT = {
    "loop":       ("loop/", "dir", "", "the loops' own: their locks and the journal"),
    "lock":       ("loop/{kind}.json", "json", "pi|backend",
                   "a loop's lock, one per kind per workspace: pi {pid, started, mode, cwd, model, socket?}, "
                   "backend {pid, started, port}"),
    "journal":    ("loop/journal.jsonl", "jsonl", "every part",
                   "{ts, who, kind, ...pointers}: the UI's feed, the one append-only file"),
    "surprises":  ("surprises/", "dir", "", "the surprises dropped to the loop"),
    "surprise":   ("surprises/{id}.json", "json", "backend",
                   "{id, ts, who, member, trace?, k?, k0?, window?, reading | atoms, words?, status: open}"),
    "taken":      ("surprises/{id}.taken", "json", "pi", "{pid, ts}: the pi loop started on it; removed on a ruling"),
    "rulings":    ("rulings/", "dir", "", "the rulings"),
    "ruling_md":  ("rulings/{id}.md", "text", "pi", "the pack verb's document (propose.py document())"),
    "ruling":     ("rulings/{id}.json", "json", "pi",
                   "{id, ts, verdict, md, pack?, moved?, lint?, reads_absent?, spent?}: its existence is the status ruled"),
    "packs":      ("packs/", "dir", "", "the candidate packs judged"),
    "pack":       ("packs/{id}.llp", "text", "pi", "the candidate pack judged (applied only through the gate)"),
    "traces":     ("traces/", "dir", "", "the traces, one dir per run"),
    "trace":      ("traces/{run}/", "dir", "backend|world",
                   "the Segments writer's dir (metis.trace: seg-<k0>.jsonl, checkpoints/k<k>.json)"),
    "marks":      ("traces/{run}/marks.jsonl", "jsonl", "world",
                   "{mark, edge: start|stop, at_k, at_tick, ...}: the person's marks on the run, the world's log rows with the "
                   "trace's k added (at_k: the writer shifts)"),
    "run_config": ("traces/{run}/config.yaml", "text", "world", "the service config the run was played with (the judge replays it)"),
    "kept":       ("traces/{run}-{name}/", "dir", "backend",
                   "a window kept past the writer's ring: keep_window's copy of the segments and checkpoints that hold it"),
    "records":    ("records/", "dir", "backend", "the REPL's records (metis.catalog.state, as before)"),
    "sessions":   ("sessions/", "dir", "backend", "the REPL's sessions (as before)"),
    "seed":       ("seed/", "dir", "pi", "the seeded start's files (as before)"),
    "seed_json":  ("seed.json", "json", "pi", "the seeded start's record (as before)"),
    "cost":       ("cost.jsonl", "jsonl", "pi|judge", "one line per paid call: {ts, who, model, cost, usage_before}"),
}


def layout():
    """The layout as data: what `layout --json` prints and the TypeScript face reads."""
    return {"format": FORMAT, "state": catalog.STATE, "id": ID, "who": list(WHO), "locks": list(LOCKS),
            "status": list(STATUS), "face": list(FACE),
            "paths": {n: {"path": p, "kind": k, "writer": w, "what": t} for n, (p, k, w, t) in LAYOUT.items()}}


_LAST_ID = [""]


def new_id(now=None):
    """YYYYMMDDTHHMMSS-<6 hex>: the second, then random hex that keeps two writers in one second apart. Monotonic
    within a process: an id in the same second as this process's last one is that one's hex plus one, so a
    process's ids sort in the order it made them. Across processes the order is the record's ts (sub-second):
    surprises() sorts by (ts, id)."""
    sec = time.strftime("%Y%m%dT%H%M%S", time.localtime(now))
    hx = secrets.token_hex(3)
    last = _LAST_ID[0]
    if last[:15] == sec and int(hx, 16) <= int(last[16:], 16):
        hx = "%06x" % min(int(last[16:], 16) + 1, 0xFFFFFF)
    _LAST_ID[0] = sec + "-" + hx
    return _LAST_ID[0]


def _ts():
    """Now, ISO 8601 with microseconds and the offset: what orders records written in one second."""
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def _order(rec):
    """(when, id): a record's ts as an instant (an older ts without sub-seconds, or none, sorts at its second),
    then its id."""
    try:
        when = datetime.fromisoformat(str(rec.get("ts"))).timestamp()
    except ValueError:
        when = float("inf")
    return (when, str(rec.get("id") or ""))


def _check_id(i):
    if not isinstance(i, str) or not re.match(ID, i):
        raise ValueError("%r: not a workspace id (YYYYMMDDTHHMMSS-<6 hex>)" % (i,))
    return i


def _check_run(run):
    if not isinstance(run, str) or not run or run in (".", "..") or "/" in run or "\\" in run:
        raise ValueError("%r: not a run name (one directory under traces/)" % (run,))
    return run


def write_whole(path, text):
    """Write TEXT to PATH whole: a temporary file beside it, then a rename (a reader never sees half)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)       # (the workspace made .metis/ first: Workspace.made)
    tmp = path.parent / (".%s.%s.tmp" % (path.name, secrets.token_hex(4)))
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def create_whole(path, text):
    """Write TEXT to PATH whole, only if PATH is not there: a temporary file, then a hard link (which refuses an
    existing name), so a reader never sees half and two writers never both win. → False when PATH was there."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (".%s.%s.tmp" % (path.name, secrets.token_hex(4)))
    tmp.write_text(text)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink()


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _rows(path):
    """A jsonl file's rows; a line that does not parse (a writer cut short) is skipped."""
    out = []
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for ln in text.splitlines():
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    return out


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError):
        return False
    return True


# ── the trace verbs as declared tools (WORKSPACE.md §3.1): metis/trace_tools.json, generated from their argparse ──
TOOLS_FORMAT = "metis-trace-tools/1"
TOOLS_FILE = Path(__file__).resolve().parents[1] / "trace_tools.json"
_TYPES = {int: "INTEGER", float: "NUMBER", None: "STRING", str: "STRING"}


def _param(act, pos):
    """One argparse action as a declared parameter: its type, its help, and how it goes on the command line
    (positional: its place; else flag: its option string)."""
    if act.nargs == 0:                                           # store_true
        p = {"type": "BOOLEAN"}
    elif act.__class__.__name__ == "_AppendAction":
        p = {"type": "ARRAY", "items": {"type": _TYPES.get(act.type, "STRING")}}
    else:
        p = {"type": _TYPES.get(act.type, "STRING")}
    if act.help:
        p["description"] = act.help
    if act.option_strings:
        p["flag"] = max(act.option_strings, key=len)
    else:
        p["positional"] = pos
    return p


def _tool(name, module, verb, description, ap):
    props, req, pos = {}, [], 0
    for act in ap._actions:
        if act.dest in ("help", "json") or act.__class__.__name__ == "_SubParsersAction":
            continue
        props[act.dest] = _param(act, pos)
        if not act.option_strings:
            pos += 1
        if act.required:
            req.append(act.dest)
    return {"name": name, "module": module, "verb": verb, "description": description,
            "parameters": {"type": "OBJECT", "properties": props, "required": req}}


def declarations():
    """The trace verbs and lint as tools, read off their argparse (metis.trace.parser, metis.lint.parser): what
    tools/gen_trace_tools.py writes into metis/trace_tools.json and tests/catalog/test_workspace.py holds it to."""
    from metis import lint, trace
    out = []
    ap = trace.parser()
    sub = next(a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction")
    helps = {c.dest: c.help for c in sub._choices_actions}
    for verb, p in sub.choices.items():
        out.append(_tool("trace_" + verb, "metis.trace", verb, helps.get(verb) or "", p))
    lp = lint.parser()
    out.append(_tool("lint", "metis.lint", None, lp.description or "", lp))
    return {"format": TOOLS_FORMAT, "tools": out}


def tools_json():
    return json.dumps(declarations(), indent=1, ensure_ascii=False) + "\n"


def tools():
    """The declaration table as committed (metis/trace_tools.json): what the backend's trace channel serves and
    the pi loop's extension exposes, nothing more."""
    return json.loads(TOOLS_FILE.read_text())


def tool_argv(name, args):
    """The command line of the declared tool NAME with ARGS {param: value}: [python, -m, module, verb?, ...]. An
    argument the tool does not declare, or a required one missing, is refused."""
    t = next((x for x in tools()["tools"] if x["name"] == name), None)
    if t is None:
        raise ValueError("%r: not a declared trace tool" % (name,))
    props, args = t["parameters"]["properties"], dict(args or {})
    extra = sorted(set(args) - set(props))
    if extra:
        raise ValueError("%s: no parameter %s" % (name, ", ".join(extra)))
    missing = [k for k in t["parameters"]["required"] if args.get(k) in (None, "", [])]
    if missing:
        raise ValueError("%s: %s required" % (name, ", ".join(missing)))
    pos = sorted((p["positional"], k) for k, p in props.items() if "positional" in p)
    argv = [sys.executable, "-m", t["module"]] + ([t["verb"]] if t["verb"] else [])
    argv += [str(args[k]) for _, k in pos]
    for k, p in props.items():
        if "flag" not in p or args.get(k) in (None, False, []):
            continue
        if p["type"] == "BOOLEAN":
            argv.append(p["flag"])
        elif p["type"] == "ARRAY":
            for v in args[k]:
                argv += [p["flag"], str(v)]
        else:
            argv += [p["flag"], str(args[k])]
    return argv


class Workspace:
    """One workspace: DIR and DIR/.metis/. No state of its own beyond the two paths: every answer is read from
    the files when asked, so any number of processes may hold one."""

    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else catalog.root()
        self.state = self.root / catalog.STATE

    def made(self):
        """DIR/.metis/ made (and git-ignored when DIR is a checkout: metis.catalog.state) before a write."""
        return catalog.state(self.root)

    def path(self, name, **fill):
        """The layout's path NAME under .metis/, its {id}/{run}/{kind} filled in."""
        return self.state / LAYOUT[name][0].format(**fill)

    def rel(self, p):
        return str(Path(p).relative_to(self.state))

    # ── the bundle ──
    def _composition(self, name=None):
        d = self.root / "domains"
        if name:
            p = catalog.find_package(name, self.root) / "composition.yaml"
            if not p.is_file():
                raise ValueError("%s: not a composition (no composition.yaml in %s)" % (name, p.parent))
            return p
        if (d / "composition.yaml").is_file():
            return d / "composition.yaml"
        found = sorted(d.glob("*/composition.yaml")) if d.is_dir() else []
        return found[0] if len(found) == 1 else None

    def bundle(self, name=None):
        """The workspace's bundle: DIR/domains/composition.yaml, else the one composition under domains/*/ (NAME:
        that package's); None when there is none (several and no NAME: None too). config: the service config a
        run of the bundle takes (propose.sh --config), absolute: the path the composition names under `service:`
        (relative to it), else service.yaml beside it, else None; the bundle's, never .metis/'s. image: the hash of the last
        `metisc build` (build/<name>.build); golden: image.golden beside the composition; stale: no build yet, or
        a member's .llp or domain.yaml, the composition or its own .llp newer than the build record (the
        project Makefile's test, spec/17 §7: metisc build itself compares hashes, so a touched file reads stale
        here and is reused there)."""
        comp = self._composition(name)
        if comp is None:
            return None
        cdir = comp.parent
        spec = yaml.safe_load(comp.read_text()) or {}
        bname = str(spec.get("name") or cdir.name)
        members = ([spec["direct"]] if spec.get("direct") else []) + list(spec.get("specialists") or [])
        record = cdir / "build" / (bname + ".build")
        image, dirs = None, {}
        if record.is_file():
            for ln in record.read_text().splitlines():
                if ln.startswith("image "):
                    image = ln.split()[1].rpartition(".v-")[2]
            lock = _json(cdir / "build" / ("%s.image.v-%s" % (bname, image)) / "image.lock") if image else None
            for m, v in ((lock or {}).get("members") or {}).items():
                p = (v.get("dir") or {}).get("path") if isinstance(v, dict) and isinstance(v.get("dir"), dict) else None
                if p:
                    dirs[m] = Path(p)
        golden = (cdir / "image.golden").read_text().strip() if (cdir / "image.golden").is_file() else None
        svc = spec.get("service")
        config = (cdir / str(svc)).resolve() if svc else (cdir / "service.yaml").resolve() \
            if (cdir / "service.yaml").is_file() else None
        why = None
        if not record.is_file():
            why = "no build yet (%s)" % record.relative_to(self.root)
        else:
            built = record.stat().st_mtime
            files = [comp, *sorted(cdir.glob("*.llp"))]
            for m in members:
                md = dirs.get(m) or cdir / m
                if md.is_dir():
                    files += [f for f in sorted(md.rglob("*")) if f.is_file() and (f.suffix == ".llp" or f.name == "domain.yaml")
                              and not {"build", "records", ".metis"} & set(f.relative_to(md).parts)]
            newer = [f for f in files if f.stat().st_mtime > built]
            if newer:
                why = "%s is newer than the build" % newer[0].relative_to(self.root) if newer[0].is_relative_to(self.root) \
                    else "%s is newer than the build" % newer[0]
        return {"name": bname, "composition": str(comp.relative_to(self.root)), "members": members, "image": image,
                "golden": golden, "config": str(config) if config else None,
                "build": str(record.relative_to(self.root)) if record.is_file() else None,
                "stale": why is not None, "why": why}

    # ── the context ──
    def context(self, patterns=None, readme_limit=4000):
        """The text a brain is given first: DIR, its README.md (up to README_LIMIT characters), the bundle (its
        composition's leading comment, each member's domain.yaml line) and the patterns catalog's PATH
        (PATTERNS, else $METIS_PATTERNS), never its text."""
        out = ["# the workspace %s" % self.root, ""]
        rd = self.root / "README.md"
        if rd.is_file():
            t = rd.read_text()
            out += ["## README.md", "", t[:readme_limit].rstrip() + ("\n…(cut at %d characters)" % readme_limit
                                                                    if len(t) > readme_limit else ""), ""]
        b = self.bundle()
        if b:
            comp = self.root / b["composition"]
            head = [ln[1:].strip() for ln in comp.read_text().splitlines()
                    if ln.startswith("#")][:12]
            out += ["## the bundle %s (%s)" % (b["name"], b["composition"]), ""] + head + [""]
            for m in b["members"]:
                try:
                    d = catalog.get(m, comp.parent)
                    line = d.line or d.display or ""
                except (KeyError, ValueError, FileNotFoundError):
                    line = "(no domain.yaml found)"
                out.append("- %s: %s" % (m, line))
            out += ["", "image %s%s" % (b["image"] or "(not built)", ", STALE: %s" % b["why"] if b["stale"] else ""), ""]
        else:
            out += ["## no bundle (no composition.yaml in domains/)", ""]
        p = patterns or os.environ.get("METIS_PATTERNS")
        if p:
            out += ["## the patterns catalog", "", str(Path(p).resolve()), ""]
        return "\n".join(out)

    # ── traces ──
    def traces(self):
        """[{run, path, segments}] under traces/, by name."""
        d = self.path("traces")
        if not d.is_dir():
            return []
        return [{"run": t.name, "path": self.rel(t), "segments": len(list(t.glob("seg-*.jsonl")))}
                for t in sorted(d.iterdir()) if t.is_dir()]

    def trace(self, run):
        """One trace: {run, path, k0, k1, checkpoints, header} read by metis.trace (k1: the last tick written)."""
        from metis import trace as T
        d = self.path("trace", run=_check_run(run))
        segs = T.read(d)
        last = [t["k"] for t in segs[-1].ticks if "k" in t]
        return {"run": run, "path": self.rel(d), "k0": segs[0].k0, "k1": last[-1] if last else segs[-1].k0,
                "checkpoints": [s.k0 for s in segs], "header": segs[0].header}

    def marks(self, run):
        """The run's marks (traces/<run>/marks.jsonl, the world's writer: its log rows with `at_k`, the trace's k), as
        windows: [{mark, k0, k1, start, stop}] by mark number; a mark not ended yet has k1 None. Rows without an at_k
        are skipped (a tick is not a k: the writer shifts, nothing here computes it)."""
        rows = _rows(self.path("marks", run=_check_run(run)))
        by = {}
        for r in rows:
            if not isinstance(r, dict) or not isinstance(r.get("at_k"), int) or r.get("edge") not in ("start", "stop"):
                continue
            m = by.setdefault(r.get("mark"), {"mark": r.get("mark"), "k0": None, "k1": None})
            m["k0" if r["edge"] == "start" else "k1"] = r["at_k"]
            m["start" if r["edge"] == "start" else "stop"] = r
        return sorted(by.values(), key=lambda m: (m["k0"] if m["k0"] is not None else -1, str(m["mark"])))

    def keep_window(self, run, k0, k1, name, who="backend"):
        """The window [K0, K1] of traces/<RUN>/ kept past the writer's ring: the segments that hold it, each with the
        checkpoint it starts from, and the run's marks.jsonl and config.yaml when there are, copied to
        traces/<RUN>-<NAME>/ (whole, then renamed). Refused when the window is not in the trace any more, with the k
        range it holds, or when the copy is there already. → "traces/<RUN>-<NAME>"."""
        import shutil
        from metis import trace as T
        src = self.path("trace", run=_check_run(run))
        dst_run = _check_run("%s-%s" % (run, name))
        if not isinstance(k0, int) or not isinstance(k1, int) or k0 > k1:
            raise ValueError("window [%r, %r]: two ticks, k0 <= k1" % (k0, k1))
        segs = T.read(src)
        last = [t["k"] for t in segs[-1].ticks if "k" in t]
        hi = last[-1] if last else segs[-1].k0
        if k0 < segs[0].k0 or k1 > hi:
            raise ValueError("window [%d, %d] is not in traces/%s any more: it holds %d..%d" % (k0, k1, run, segs[0].k0, hi))
        dst = self.path("trace", run=dst_run)
        if dst.exists():
            raise ValueError("traces/%s is there already" % dst_run)
        keep = [s for i, s in enumerate(segs)
                if s.k0 <= k1 and (i + 1 == len(segs) or segs[i + 1].k0 > k0)]
        self.made()
        tmp = dst.with_name(".%s.tmp%d" % (dst.name, os.getpid()))
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "checkpoints").mkdir(parents=True)
        for sg in keep:
            shutil.copy2(src / ("seg-%d.jsonl" % sg.k0), tmp / ("seg-%d.jsonl" % sg.k0))
            shutil.copy2(src / sg.header["checkpoint"], tmp / sg.header["checkpoint"])
        for f in ("marks.jsonl", "config.yaml"):
            if (src / f).is_file():
                shutil.copy2(src / f, tmp / f)
        os.rename(tmp, dst)
        self.journal(who, "keep", trace="traces/" + dst_run, source="traces/" + run, window=[k0, k1])
        return "traces/" + dst_run

    # ── the journal ──
    def journal(self, who, kind, **ptrs):
        """Append one row {ts, who, kind, ...PTRS} to the journal (one write of one line: O_APPEND keeps rows
        whole between writers). → the row."""
        if who not in WHO:
            raise ValueError("who %r: one of %s" % (who, ", ".join(WHO)))
        if not isinstance(kind, str) or not kind:
            raise ValueError("a journal row needs a kind")
        row = {"ts": _ts(), "who": who, "kind": kind, **ptrs}
        self.made()
        p = self.path("journal")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(row, separators=(",", ":"), default=str) + "\n").encode())
        finally:
            os.close(fd)
        return row

    def events(self, since=0):
        """The journal's rows from row SINCE on (0: all); the UI passes the count it has read."""
        return _rows(self.path("journal"))[int(since):]

    # ── surprises and rulings ──
    def status(self, sid):
        """ruled (its ruling's json exists), taken (a .taken whose pid lives), else open: a take whose pid is gone
        is stale, the lock's own rule, and the surprise is open again (the next take_surprise re-takes it)."""
        if self.path("ruling", id=sid).is_file():
            return "ruled"
        tk = _json(self.path("taken", id=sid))
        return "taken" if isinstance(tk, dict) and _alive(tk.get("pid")) else "open"

    def _with_take(self, rec):
        """REC with its take: `taken` {pid, ts} when taken, `stale_take` when open over a dead pid's .taken."""
        tk = _json(self.path("taken", id=rec["id"]))
        if isinstance(tk, dict) and rec["status"] != "ruled":
            rec["taken" if rec["status"] == "taken" else "stale_take"] = tk
        return rec

    def surprises(self, status=None):
        """Every surprise, oldest first (by its ts, sub-second, then its id), each with its status (open | taken | ruled) read from the files."""
        if status is not None and status not in STATUS:
            raise ValueError("status %r: one of %s" % (status, ", ".join(STATUS)))
        d = self.path("surprises")
        out = []
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            rec = _json(f)
            if not isinstance(rec, dict) or not re.match(ID, f.stem):
                continue
            rec = self._with_take(dict(rec, id=f.stem, status=self.status(f.stem)))
            if status is None or rec["status"] == status:
                out.append(rec)
        return sorted(out, key=_order)

    def surprise(self, sid):
        """One surprise with its status, and its ruling when ruled."""
        f = self.path("surprise", id=_check_id(sid))
        rec = _json(f)
        if not isinstance(rec, dict):
            raise KeyError("no surprise %s" % sid)
        rec = self._with_take(dict(rec, id=sid, status=self.status(sid)))
        if rec["status"] == "ruled":
            rec["ruling"] = _json(self.path("ruling", id=sid))
        return rec

    def drop_surprise(self, rec, who="backend"):
        """Write a surprise: REC needs member and reading or atoms; trace ("traces/<run>") with k, k0 and window
        ([k_start, k_end]) and words (the person's text) when there are. → its id."""
        if not isinstance(rec, dict):
            raise ValueError("a surprise is an object")
        if not rec.get("member"):
            raise ValueError("a surprise names its member")
        if not rec.get("reading") and not rec.get("atoms"):
            raise ValueError("a surprise carries a reading or atoms")
        tr = rec.get("trace")
        if tr is not None:
            if not isinstance(tr, str) or not tr.startswith("traces/"):
                raise ValueError("trace %r: traces/<run>, under the workspace's .metis/" % (tr,))
            _check_run(tr[len("traces/"):].rstrip("/"))
        w = rec.get("window")
        if w is not None and not (isinstance(w, list) and len(w) == 2 and all(isinstance(x, int) for x in w)
                                  and w[0] <= w[1]):
            raise ValueError("window %r: [k_start, k_end], k_start <= k_end" % (w,))
        for k in ("k", "k0"):
            if rec.get(k) is not None and not isinstance(rec[k], int):
                raise ValueError("%s %r: a tick (an integer)" % (k, rec[k]))
        self.made()
        sid = new_id()
        while self.path("surprise", id=sid).exists():
            sid = new_id()
        rec = {**{k: v for k, v in rec.items() if k not in ("id", "ts", "status")}, "id": sid, "ts": _ts(),
               "who": rec.get("who") or who, "status": "open"}
        if rec["who"] not in WHO:
            raise ValueError("who %r: one of %s" % (rec["who"], ", ".join(WHO)))
        write_whole(self.path("surprise", id=sid), json.dumps(rec, indent=1) + "\n")
        self.journal(rec["who"], "surprise", id=sid, member=rec["member"], **({"trace": tr} if tr else {}))
        return sid

    def take_surprise(self, sid, pid=None, who="pi"):
        """The pi loop starts on SID: surprises/<id>.taken {pid, ts}. Refused when it is ruled, or taken by a
        live pid. A dead pid's take is stale (the surprise reads open) and is re-taken: journal kind `retake`,
        naming the dead pid."""
        rec = self.surprise(sid)
        if rec["status"] == "ruled":
            raise ValueError("%s: ruled already" % sid)
        pid = int(pid if pid is not None else os.getpid())
        f = self.path("taken", id=sid)
        body = json.dumps({"pid": pid, "ts": _ts()}) + "\n"
        if create_whole(f, body):
            self.journal(who, "take", id=sid, pid=pid)
            return sid
        old = _json(f) or {}
        if _alive(old.get("pid")):
            raise ValueError("%s: taken by pid %s at %s" % (sid, old.get("pid"), old.get("ts")))
        write_whole(f, body)
        self.journal(who, "retake", id=sid, pid=pid, dead_pid=old.get("pid"))
        return sid

    def rule(self, sid, ruling, verdict, who="pi", pack=None, **fields):
        """The ruling of SID: RULING (the pack verb's document, a path or its text) into rulings/<id>.md, PACK
        (a path or its text) into packs/<id>.llp, then rulings/<id>.json {id, ts, verdict, md, pack, **FIELDS}
        (moved, lint, reads_absent, spent …) last: its existence is the status. The .taken goes. → the json's path."""
        self.surprise(sid)
        if not isinstance(verdict, str) or not verdict:
            raise ValueError("a ruling has a verdict")
        if self.path("ruling", id=sid).exists():
            raise ValueError("%s: ruled already" % sid)

        def text(x):
            p = Path(x) if isinstance(x, (str, Path)) and "\n" not in str(x) and len(str(x)) < 4096 else None
            return p.read_text() if p is not None and p.is_file() else str(x)
        self.made()
        md = write_whole(self.path("ruling_md", id=sid), text(ruling))
        out = {**fields, "id": sid, "ts": _ts(), "verdict": verdict, "md": self.rel(md)}
        if pack is not None:
            out["pack"] = self.rel(write_whole(self.path("pack", id=sid), text(pack)))
        j = write_whole(self.path("ruling", id=sid), json.dumps(out, indent=1, default=str) + "\n")
        self.path("taken", id=sid).unlink(missing_ok=True)
        self.journal(who, "ruling", id=sid, verdict=verdict, ruling=self.rel(j))
        return j

    def rulings(self):
        """Every ruling's json, oldest first (by its ts, then id)."""
        d = self.path("rulings")
        return sorted([r for r in (_json(f) for f in d.glob("*.json")) if isinstance(r, dict)], key=_order) \
            if d.is_dir() else []

    # ── the locks: one active pi loop, one backend (WORKSPACE.md §2.3) ──
    def loops(self):
        """{pi: lock or None, backend: lock or None}; a lock carries alive: false when its pid is gone (stale: the
        next lock() of its kind takes it over)."""
        out = {}
        for k in LOCKS:
            rec = _json(self.path("lock", kind=k))
            out[k] = dict(rec, alive=_alive(rec.get("pid"))) if isinstance(rec, dict) else None
        return out

    def lock(self, kind, pid=None, **fields):
        """Take KIND's lock (loop/<kind>.json {pid, started, **FIELDS}: pi's mode, cwd, model, socket; the
        backend's port). Refused, in the holder's words, while a live pid holds it; a dead pid's lock is taken
        over with a journal row. → the lock."""
        if kind not in LOCKS:
            raise ValueError("lock %r: one of %s" % (kind, ", ".join(LOCKS)))
        self.made()
        pid = int(pid if pid is not None else os.getpid())
        rec = {**fields, "pid": pid, "started": _ts()}
        f = self.path("lock", kind=kind)
        body = json.dumps(rec, indent=1) + "\n"
        if create_whole(f, body):
            self.journal(kind, "lock", lock=self.rel(f), pid=pid)
            return rec
        old = _json(f) or {}
        if _alive(old.get("pid")) and int(old.get("pid")) != pid:
            more = ", ".join("%s %s" % (k, old[k]) for k in ("mode", "model", "port", "cwd") if old.get(k) is not None)
            raise ValueError("the %s loop holds this workspace: pid %s since %s%s (%s)"
                             % (kind, old.get("pid"), old.get("started"), ": " + more if more else "", self.rel(f)))
        write_whole(f, body)
        self.journal(kind, "lock", lock=self.rel(f), pid=pid,
                     **({"stale_pid": old.get("pid")} if old.get("pid") != pid else {}))
        return rec

    def unlock(self, kind, pid=None):
        """Release KIND's lock: only its holder (PID, default this process) does, or anyone once the holder is
        gone. → True when a lock went."""
        if kind not in LOCKS:
            raise ValueError("lock %r: one of %s" % (kind, ", ".join(LOCKS)))
        f = self.path("lock", kind=kind)
        old = _json(f)
        if not isinstance(old, dict):
            return False
        pid = int(pid if pid is not None else os.getpid())
        if int(old.get("pid") or 0) != pid and _alive(old.get("pid")):
            raise ValueError("the %s lock is pid %s's, not %d's" % (kind, old.get("pid"), pid))
        f.unlink(missing_ok=True)
        self.journal(kind, "unlock", lock=self.rel(f), pid=old.get("pid"))
        return True

    # ── cost ──
    def cost(self):
        """cost.jsonl summed: {total, calls, by_model, by_who}."""
        rows = _rows(self.path("cost"))
        out = {"total": 0.0, "calls": 0, "by_model": {}, "by_who": {}}
        for r in rows:
            c = float(r.get("cost") or 0)
            out["total"] += c
            out["calls"] += 1
            for k, f in (("by_model", "model"), ("by_who", "who")):
                key = str(r.get(f) or "?")
                out[k][key] = out[k].get(key, 0.0) + c
        out["total"] = round(out["total"], 6)
        return out


def workspace(root=None):
    """The workspace at ROOT, else metis.catalog.root() ($METIS_CATALOG, else the cwd)."""
    return Workspace(root)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m metis.catalog.workspace", description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="the workspace (default: $METIS_CATALOG, else the current directory)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {"layout": sub.add_parser("layout", help="the .metis/ layout as data (the one table both faces read)"),
          "root": sub.add_parser("root", help="the workspace's root and state")}
    ps["bundle"] = b = sub.add_parser("bundle", help="the bundle: members, image, stale")
    b.add_argument("name", nargs="?")
    ps["context"] = c = sub.add_parser("context", help="the text a brain is given first")
    c.add_argument("--patterns", help="the patterns catalog's path (else $METIS_PATTERNS)")
    ps["traces"] = sub.add_parser("traces", help="the traces under .metis/traces/")
    ps["trace"] = t = sub.add_parser("trace", help="one trace: its k range, checkpoints and header")
    ps["marks"] = mk = sub.add_parser("marks", help="a run's marks as windows (traces/<run>/marks.jsonl)")
    mk.add_argument("run")
    ps["keep"] = kp = sub.add_parser("keep", help="keep a window of a trace past its ring: traces/<run>-<name>/")
    kp.add_argument("run")
    kp.add_argument("k0", type=int)
    kp.add_argument("k1", type=int)
    kp.add_argument("name")
    t.add_argument("run")
    ps["surprises"] = s = sub.add_parser("surprises", help="the surprises, each with its status")
    s.add_argument("--status", choices=STATUS)
    ps["surprise"] = s1 = sub.add_parser("surprise", help="one surprise, with its ruling when ruled")
    s1.add_argument("id")
    ps["drop"] = d = sub.add_parser("drop", help="drop a surprise: its record as JSON from FILE (- = stdin)")
    d.add_argument("record", metavar="FILE")
    d.add_argument("--who", default="backend", choices=WHO)
    ps["take"] = tk = sub.add_parser("take", help="the pi loop takes a surprise")
    tk.add_argument("id")
    tk.add_argument("--pid", type=int, help="the taker's pid (default: the caller, this process's parent)")
    tk.add_argument("--who", default="pi", choices=WHO)
    ps["rule"] = r = sub.add_parser("rule", help="a surprise's ruling: the document, the verdict, the pack")
    r.add_argument("id")
    r.add_argument("--ruling", required=True, metavar="FILE", help="the pack verb's document (- = stdin)")
    r.add_argument("--verdict", required=True)
    r.add_argument("--pack", metavar="FILE", help="the candidate pack judged")
    r.add_argument("--fields", metavar="JSON", help="more of the ruling's json: moved, lint, reads_absent, spent …")
    r.add_argument("--who", default="pi", choices=WHO)
    ps["rulings"] = sub.add_parser("rulings", help="every ruling")
    ps["journal"] = j = sub.add_parser("journal", help="append one row to the journal")
    j.add_argument("--who", required=True, choices=WHO)
    j.add_argument("--kind", required=True)
    j.add_argument("--fields", metavar="JSON", help="the row's pointers, as a JSON object")
    ps["events"] = e = sub.add_parser("events", help="the journal's rows from row SINCE on")
    e.add_argument("--since", type=int, default=0)
    ps["cost"] = sub.add_parser("cost", help="cost.jsonl summed")
    ps["loops"] = sub.add_parser("loops", help="the pi loop's and the backend's locks")
    ps["lock"] = lk = sub.add_parser("lock", help="take a loop's lock (refused while a live pid holds it)")
    lk.add_argument("kind", choices=LOCKS)
    lk.add_argument("--pid", type=int, help="the holder (default: the caller, this process's parent)")
    lk.add_argument("--fields", metavar="JSON", help="the lock's fields: pi {mode, cwd, model, socket}, backend {port}")
    ps["unlock"] = ul = sub.add_parser("unlock", help="release a loop's lock (its holder, or anyone once it is gone)")
    ul.add_argument("kind", choices=LOCKS)
    ul.add_argument("--pid", type=int, help="the holder (default: the caller, this process's parent)")
    ps["tools"] = sub.add_parser("tools", help="the trace verbs and lint as declared tools (metis/trace_tools.json)")
    ps["argv"] = av = sub.add_parser("argv", help="a declared tool's command line from its arguments")
    av.add_argument("name")
    av.add_argument("--args", metavar="JSON", help="the arguments, as a JSON object")
    for p in (ap, *ps.values()):
        p.add_argument("--json", action="store_true", help="JSON on stdout (the only form; kept for the faces)")
    a = ap.parse_args(argv)

    def read(f):
        return sys.stdin.read() if f == "-" else Path(f).read_text()

    def obj(s):
        v = json.loads(s) if s else {}
        if not isinstance(v, dict):
            raise ValueError("%r: a JSON object" % s)
        return v

    try:
        if a.cmd == "layout":
            out = layout()
        elif a.cmd == "tools":
            out = tools()
        elif a.cmd == "argv":
            out = {"argv": tool_argv(a.name, obj(a.args))}
        else:
            ws = workspace(a.root)
            if a.cmd == "root":
                out = {"root": str(ws.root), "state": str(ws.state), "made": ws.state.is_dir()}
            elif a.cmd == "bundle":
                out = ws.bundle(a.name)
            elif a.cmd == "context":
                out = {"context": ws.context(a.patterns)}
            elif a.cmd == "traces":
                out = ws.traces()
            elif a.cmd == "trace":
                out = ws.trace(a.run)
            elif a.cmd == "marks":
                out = ws.marks(a.run)
            elif a.cmd == "keep":
                out = {"trace": ws.keep_window(a.run, a.k0, a.k1, a.name)}
            elif a.cmd == "surprises":
                out = ws.surprises(a.status)
            elif a.cmd == "surprise":
                out = ws.surprise(a.id)
            elif a.cmd == "drop":
                out = {"id": ws.drop_surprise(obj(read(a.record)), who=a.who)}
            elif a.cmd == "take":
                out = {"id": ws.take_surprise(a.id, a.pid or os.getppid(), a.who)}
            elif a.cmd == "rule":
                ruling = read(a.ruling)
                pack = read(a.pack) if a.pack else None
                out = {"ruling": ws.rel(ws.rule(a.id, ruling, a.verdict, a.who, pack, **obj(a.fields)))}
            elif a.cmd == "rulings":
                out = ws.rulings()
            elif a.cmd == "journal":
                out = ws.journal(a.who, a.kind, **obj(a.fields))
            elif a.cmd == "events":
                out = ws.events(a.since)
            elif a.cmd == "loops":
                out = ws.loops()
            elif a.cmd == "lock":
                out = ws.lock(a.kind, a.pid or os.getppid(), **obj(a.fields))
            elif a.cmd == "unlock":
                out = {"released": ws.unlock(a.kind, a.pid or os.getppid())}
            else:
                out = ws.cost()
    except (KeyError, ValueError, FileNotFoundError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        json.dump({"error": msg}, sys.stdout)
        print()
        return 1
    json.dump(out, sys.stdout, indent=1, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
