"""metis.gate — the loop's policy, one for both loops (loop-parity s-72d9d36b).
(Moved from metis_loop_sdk.gate into the runtime's distribution: metis-packages s-65778b98;
metis_loop_sdk.gate re-exports it for one release, and its CLI stays for the TS face.)

The catalog's mechanics are metis.catalog; its rules for a loop that
writes into it are here, so the person's console and pi's tools follow one
policy: every write is a proposal the person reads, then confirms.

    from metis import gate
    g = gate.Gate()                                  # the memory in this process (a console)
    g = gate.Gate.at(state_dir)                      # … or in state_dir/gate.json (a CLI per call)
    pl = g.propose("program", "lamp", root, text=…, display=…, line=…, run=state_dir)
    pl["files"]          # [{path, new, diff, text, since_diff?}]: what confirm writes, each with its diff
    g.confirm(pl["token"])   # writes exactly that plan; {written, fingerprint, serve?, bundle?}
    g.refuse(pl["token"])    # the person said no: its texts are remembered, per root and file

    python -m metis.gate --state DIR propose KIND DOMAIN --root R [--run RUN] --json < args.json
    python -m metis.gate --state DIR confirm TOKEN --json
    python -m metis.gate --state DIR refuse TOKEN --json

The kinds (propose's args):
  program    text (the candidate the loop gated), display, line: the domain's program; a new
             domain gets a domain.yaml (program, its host when <d>_host.py sits there, clock, records)
  served     text (the program the REPL serves, as it is: a seed), display, line; a package seed
             (run/seed.json package: true) comes whole from run/seed/, its own domain.yaml kept
  extension  name, text: domains/<d>/extensions/<name>.llp
  intents    text (an intent.yaml), else the program's skeleton (no model); a world with no
             commands gets a note and no write
  readme     text (the summary paragraph), title: the README, the SDK writing every other part
  edit       path, old, new, replace_all: one exact edit of a file on disk
  take       src (the loop's catalog): the domain into root (another catalog), a README of the
             run (run) when it has none of its own; confirm builds its bundle

The rules:
  - confirm writes exactly the plan the person read: a plan is named by its token, and a
    source that changed since (the same proposal now plans other texts) refuses;
  - a refused proposal is remembered per root and file: the next plan of that file carries
    since_diff, what changed since the refusal; a write of the file forgets it;
  - seed first: in a seeded run (run/seed.json), until the seed's domain is in the catalog,
    no program and no extension of a domain the catalog lacks, and the seed is served under
    its own name only;
  - README refused: an edit of a file that was never written because its proposal was
    refused goes through the proposal again (its gate shows what changed since);
  - a plan with nothing to write (a note) needs no confirmation of the person.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from metis import catalog

KINDS = ("program", "served", "extension", "intents", "readme", "edit", "take")
NAME = __import__("re").compile(r"^[a-z][a-z0-9_]*$")
NOTE_NO_COMMANDS = ("no intent.yaml: the world takes no commands (its program has no cmd_* atom), "
                    "so the agents open it with no verbs")


class Refused(ValueError):
    """A rule of the gate said no: its words are the message."""


def _state_dir(d):
    """The gate's directory, made when missing: a loop's .metis/ through catalog.state()
    (git-ignored in a checkout), any other directory as it is."""
    d = Path(d)
    if not d.is_dir():
        if d.name == catalog.STATE:
            catalog.state(d.parent)
        else:
            d.mkdir(parents=True, exist_ok=True)
    return d


def _seed(run):
    try:
        return json.loads((Path(run) / "seed.json").read_text()) if run else None
    except (OSError, ValueError):
        return None


def _known(domain, root):
    try:
        catalog.get(domain, root)
        return True
    except KeyError:
        return False


def _serve(domain, root):
    """The program the catalog's loader names for the domain, and its host as a
    REPL's --host takes it (a file.py:factory beside the program, made absolute)."""
    d = catalog.get(domain, root)
    host = d.host_spec_for() if d.program else None
    if host and ".py:" in host:                       # file.py:factory, relative to the program's directory
        host = str(Path(d.program).parent / host)
    return {"program": str(d.program), "host": host}


class Gate:
    """The refused-proposal memory and the plans waiting for the person, in `store`
    ({refused: {root: {path: text}}, plans: {token: plan}}); `save` persists it."""

    def __init__(self, store=None, save=None):
        self.store = store if store is not None else {}
        self.store.setdefault("refused", {})
        self.store.setdefault("plans", {})
        self._save = save or (lambda s: None)

    @classmethod
    def at(cls, state_dir):
        """A gate whose memory is state_dir/gate.json (the loop's state: DIR/.metis/),
        written whole and atomically (a temp file, then os.replace)."""
        import os
        f = Path(state_dir) / "gate.json"
        try:
            store = json.loads(f.read_text())
        except (OSError, ValueError):
            store = {}

        def save(s):
            _state_dir(f.parent)
            tmp = f.with_name(f.name + ".tmp")
            tmp.write_text(json.dumps(s))
            os.replace(tmp, f)
        return cls(store, save)

    @classmethod
    @__import__("contextlib").contextmanager
    def locked(cls, state_dir):
        """Gate.at under an exclusive lock on state_dir/gate.lock for the whole of one
        use (load, change, save): two loops in one DIR never lose each other's memory."""
        import fcntl
        d = _state_dir(Path(state_dir))
        with open(d / "gate.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield cls.at(d)
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    # -- the memory ---------------------------------------------------------

    def refused(self, root):
        """{path: text}: the last refused proposal of each file of the root."""
        return dict(self.store["refused"].get(str(Path(root).resolve()), {}))

    def _remember(self, root, files, ok):
        k = str(Path(root).resolve())
        m = self.store["refused"].setdefault(k, {})
        for f in files:
            if ok:
                m.pop(f["path"], None)
            else:
                m[f["path"]] = f["text"]

    # -- propose / confirm / refuse -------------------------------------------

    def propose(self, kind, domain, root=None, run=None, **args):
        """A plan of the write `kind` for the domain into root: {token, kind, domain, root,
        files, reopen, note?, say?}. Raises Refused with the rule's words."""
        root = str(Path(root).resolve() if root else catalog.root())
        pl = self._plan(kind, domain, root, run, args, since=self.refused(root))
        pl["token"] = hashlib.sha256(json.dumps(
            [kind, domain, root, [(f["path"], f["text"]) for f in pl["files"]], pl.get("note")]).encode()).hexdigest()[:16]
        self.store["plans"][pl["token"]] = {"kind": kind, "domain": domain, "root": root, "run": run, "args": args,
                                            "files": [[f["path"], f["text"]] for f in pl["files"]]}
        self._save(self.store)
        return pl

    def refuse(self, token):
        """The person refused the plan: its texts are remembered (the next plan of each file
        diffs from it); the plan is dropped."""
        p = self.store["plans"].pop(token, None)
        if p is None:
            raise Refused("no such plan: propose it again")
        self._remember(p["root"], [{"path": a, "text": b} for a, b in p["files"]], False)
        self._save(self.store)
        return {"refused": [a for a, _ in p["files"]]}

    def confirm(self, token):
        """Write exactly the plan `token` names: {written, fingerprint, note?, serve?, bundle?}.
        A source that changed since the plan refuses (nothing written)."""
        p = self.store["plans"].pop(token, None)
        self._save(self.store)
        if p is None:
            raise Refused("not approved: no such plan (propose it again)")
        kind, domain, root, run, args = p["kind"], p["domain"], p["root"], p["run"], p["args"]
        now = self._plan(kind, domain, root, run, args)
        if [[f["path"], f["text"]] for f in now["files"]] != p["files"]:
            if kind == "edit":
                raise Refused(f"{args.get('path')} changed after the person approved the edit: nothing written; propose it again")
            raise Refused(f"the catalog changed after the person approved the write ({domain}): nothing written; propose it again")
        if now.get("note"):
            return {"written": [], "fingerprint": None, "note": now["note"]}
        out = now["write"]()
        self._remember(root, now["files"], True)
        self._save(self.store)
        res = {"written": out["written"], "fingerprint": out.get("fingerprint")}
        if now["reopen"]:
            res["serve"] = _serve(domain, root)
        if kind == "take":
            try:
                res["bundle"] = catalog.build(domain, root)
            except ValueError as e:
                res["bundle"] = str(e)
        return res

    # -- the kinds ------------------------------------------------------------

    def _plan(self, kind, domain, root, run, a, since=None):
        if kind not in KINDS:
            raise Refused(f"kind: {' | '.join(KINDS)}, not {json.dumps(kind)}")
        if not NAME.match(domain or ""):
            raise Refused(f"domain {json.dumps(domain)}: lowercase letters, digits, _")
        since = since or {}

        def planned(w, reopen=False, r=root, say=None):
            def write():
                return catalog.apply(domain, r=r, **w)
            out = {"kind": kind, "domain": domain, "root": r, "reopen": reopen,
                   "files": catalog.plan(domain, r=r, since=since, **w), "write": write}
            if say:
                out["say"] = say
            return out

        if kind == "take":
            src = a.get("src")
            if not src:
                raise Refused("take: src, the catalog the domain is taken from")
            catalog.get(domain, src)                         # the loop's catalog has it (the SDK's error otherwise)
            text = None
            if run and not (catalog.get(domain, src).dir / "README.md").is_file():
                text = catalog.readme(domain, src, run)
            kw = catalog.take(domain, src, root, text)
            return planned(kw)
        if kind == "edit":
            d = catalog.get(domain, root)
            import posixpath
            # the file as the plan names it: relative to the root, wherever the package sits (rule 9: DIR/lamp/ too)
            pkg = Path(d.dir).resolve().relative_to(Path(root).resolve()).as_posix()
            rel = posixpath.normpath(f"{pkg}/{a['path']}")
            if not (Path(d.dir) / a["path"]).exists() and rel in since:
                raise Refused(f"{a['path']} was never written: the person refused its proposal. Send it again through "
                              "catalog_apply with only the fix; its gate shows what changed since")
            kw = catalog.edit(domain, a["path"], a["old"], a["new"], bool(a.get("replace_all")), root)
            reopen = bool(d.program) and (Path(d.dir) / a["path"]).resolve() == Path(d.program).resolve()
            return planned(kw, reopen)

        known = _known(domain, root)
        seed = _seed(run)
        shown = {k: a[k] for k in ("display", "line") if a.get(k) is not None}
        if (not known and seed and not _known(seed.get("name", ""), root)
                and kind in ("program", "extension")):
            raise Refused(f"a seeded run applies the seed first (catalog_apply domain={domain} what=served); "
                          "a change to it is then its diff against the seed")
        if kind == "readme":
            if not known:
                raise Refused(f"no domain {domain} in the loop's catalog yet: its README follows its program")
            if not (a.get("text") or "").strip():
                raise Refused("what=readme: text is the summary paragraph (what the program tells; the SDK writes the rest)")
            return planned({"readme": catalog.readme(domain, root, run, a["text"], a.get("title"))})
        if kind == "intents":
            text = a.get("text")
            if text is None:
                text = catalog.skeleton(domain, root)
                if text is None:
                    return {"kind": kind, "domain": domain, "root": root, "reopen": False, "files": [],
                            "note": NOTE_NO_COMMANDS, "write": lambda: {"written": [], "fingerprint": None}}
            return planned({"intents": text})
        if (kind == "served" and seed and not seed.get("package") and domain != seed.get("name")
                and not _known(seed.get("name", ""), root)):
            raise Refused(f"the seed is {seed['name']}: catalog_apply domain={seed['name']} what=served")
        if kind == "served" and seed and seed.get("package"):
            if domain != seed["name"]:
                raise Refused(f"the seed is the package {seed['name']}: catalog_apply domain={seed['name']} what=served")
            src = Path(run) / "seed"
            say = None
            if shown:
                own = catalog.get(domain, src)
                say = (f" The package's own domain.yaml is used as it is, so the display and line given here were not: "
                       f"it has display {json.dumps(own.display)} and line {json.dumps(own.line)}.")
            return planned(catalog.take(domain, src, root), True, say=say)
        text = a.get("text")
        if text is None and kind == "served":
            raise Refused("the REPL serves a blank world: there is no program to apply (loop.sh --seed starts it on one)")
        if text is None:
            raise Refused(f"{kind}: text, the program's text")
        if kind == "extension":
            name = a.get("name") or ""
            if not NAME.match(name):
                raise Refused("an extension needs a name (lowercase): it is written to domains/<domain>/extensions/<name>.llp")
            return planned({"extensions": {name: text}})
        # the console writes a program with its extensions in one plan (loop-parity step 2)
        exts = {k: v for k, v in (a.get("extensions") or {}).items()}
        for k in exts:
            if not NAME.match(k):
                raise Refused(f"extension {json.dumps(k)}: lowercase letters, digits, _ (domains/<domain>/extensions/<name>.llp)")
        if exts:
            shown["extensions"] = exts
        if known:
            return planned({"program": text, **shown}, True)
        # a new domain: its manifest, the host beside the program when the world brought one
        host = (Path(root) / "domains" / domain / f"{domain}_host.py").is_file()
        manifest = (f"# written by the agentic loop's catalog_apply (docs/LOOP-AGENT.md)\nprogram: {domain}.llp\n"
                    + (f"host: {domain}_host.py:make\n" if host else "") + "clock: {beats: 1}\nrecords: records/\n")
        return planned({"program": text, "manifest": manifest, **shown}, True)


def public(pl):
    """A plan as it leaves the gate (JSON): no write function."""
    return {k: v for k, v in pl.items() if k != "write"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m metis.gate", description=__doc__.split("\n")[0])
    ap.add_argument("--state", required=True, metavar="DIR", help="the loop's state (DIR/.metis/): its gate.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("propose", help="a plan of a write: its files and their diffs, its token (args: JSON on stdin)")
    pr.add_argument("kind", choices=KINDS)
    pr.add_argument("domain")
    pr.add_argument("--root", help="the catalog written (default: $METIS_CATALOG, else the cwd)")
    pr.add_argument("--run", metavar="DIR", help="the loop's state with its seed.json and records (seed first, README of a run)")
    cf = sub.add_parser("confirm", help="write exactly the plan TOKEN names")
    cf.add_argument("token")
    rf = sub.add_parser("refuse", help="the person refused the plan TOKEN: its texts are remembered")
    rf.add_argument("token")
    for p in (ap, pr, cf, rf):
        p.add_argument("--json", action="store_true", help="JSON on stdout")
    a = ap.parse_args(argv)
    raw = sys.stdin.read() if a.cmd == "propose" else ""
    try:
        with Gate.locked(a.state) as g:
            if a.cmd == "propose":
                args = json.loads(raw) if raw.strip() else {}
                out = public(g.propose(a.kind, a.domain, a.root, a.run, **args))
            elif a.cmd == "confirm":
                out = g.confirm(a.token)
            else:
                out = g.refuse(a.token)
    except (KeyError, ValueError, FileNotFoundError) as e:     # as the catalog's CLI says it
        msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
        json.dump({"error": msg}, sys.stdout)
        print()
        return 1
    json.dump(out, sys.stdout, default=str)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
