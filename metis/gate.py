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

The experimental policy (loop-policy s-dd842f2a, metis-agentic-loop docs/POLICY.md): a loop
running unattended has its proposals judged by numbers, not by the person, and stops to ask
when it is cornered. The human gate above is the default and is unchanged.

    g = gate.Gate(policy="experimental", gain=fn, journal=state_dir / "journal.jsonl")
    g = gate.Gate.at(state_dir, policy="experimental", gain=fn)    # the journal beside gate.json
    pl = g.propose("program", "alf", root, text=…, phase="exploit", cost=0.004)
    pl["verdict"]    # {verdict: accept|refuse, phase, numbers, why}; accept is confirmed at once
                     # (pl["written"]), refuse is remembered as the person's refusal is (since_diff)
    a = g.propose("ask", "alf", root, phase="exploit", reason="no_progress", tried=[…],
                  believes=[…], options=[…], question="…", snapshot="…")    # writes nothing; waits
    g.answer(a["token"], {"kind": "note", "text": "…"})      # the person's instruction
    g.answers(); g.journal()                                  # what the loop reads at re-engagement

  - the judged kinds are program, extension and edit; the others (served, intents, readme,
    take) wait for the person under either policy;
  - numbers come from `gain(plan, phase)` (the run driver measures the candidate on the
    held-out split), or as `numbers` in propose's args (a driver that measured them itself:
    the CLI). They are changes, candidate minus current: likelihood, success; and
    replay_identical (bool), vector (the complexity vector's change: a number, a list or a
    dict summed, or {before, after}), fits_records (bool: the candidate fires on its record
    and on no record it contradicts);
  - the verdict per phase (POLICY.md, the experimental gate):
      explore   fits_records, and the likelihood not down
      exploit   the likelihood up, or the success up with the likelihood not down
      refactor  replay_identical, the success (and a given likelihood) not down, the vector down
    a number the phase needs and the plan lacks refuses ("not measured");
  - kind ask, the disengagement record: phase (explore|exploit|refactor), reason (budget,
    no_progress, out_of_vocabulary, contradiction, stuck), tried, believes, options (at most
    three), ONE question, snapshot (what the loop re-engages from). No file is written and
    nothing confirms it: answer(token, instruction) does, with an instruction of kind note
    (text), edit (path, old, new: proposed again through the gate at re-engagement), rule
    (text), budget (beats, proposals, spend), phase (explore|exploit|refactor|skip, text) or
    abandon; a plain string is a note;
  - the journal (one JSON line each): every judged proposal (kind, phase, files, numbers,
    verdict, why, cost, written), every ask and every answer.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from metis import catalog

KINDS = ("program", "served", "extension", "intents", "readme", "edit", "take", "ask")
POLICIES = ("human", "experimental")
JUDGED = ("program", "extension", "edit")
PHASES = ("explore", "exploit", "refactor")
REASONS = ("budget", "no_progress", "out_of_vocabulary", "contradiction", "stuck")
ANSWERS = ("note", "edit", "rule", "budget", "phase", "abandon")
NAME = __import__("re").compile(r"^[a-z][a-z0-9_]*$")
NOTE_NO_COMMANDS = ("no intent.yaml: the world takes no commands (its program has no cmd_* atom), "
                    "so the agents open it with no verbs")


class Refused(ValueError):
    """A rule of the gate said no: its words are the message."""


def _vector(v):
    """The complexity vector's change as one number: a number as it is, a list or a dict
    summed, {before, after} as after minus before."""
    if isinstance(v, dict) and set(v) == {"before", "after"}:
        return _vector(v["after"]) - _vector(v["before"])
    if isinstance(v, dict):
        return sum(v.values())
    if isinstance(v, (list, tuple)):
        return sum(v)
    return v


def judge(phase, numbers):
    """The experimental gate's verdict on a proposal's numbers in a phase (POLICY.md):
    {verdict: accept|refuse, phase, numbers, why: [the criteria, each met or not]}."""
    if phase not in PHASES:
        raise Refused(f"phase: {' | '.join(PHASES)}, not {json.dumps(phase)}")
    n = dict(numbers or {})
    why, ok = [], True

    def need(key):
        nonlocal ok
        if n.get(key) is None:
            ok = False
            why.append(f"{key}: not measured")
            return None
        return n[key]

    def check(met, text):
        nonlocal ok
        ok = ok and met
        why.append(("" if met else "not: ") + text)

    if phase == "explore":
        f, lk = need("fits_records"), need("likelihood")
        if f is not None:
            check(bool(f), "the candidate fires on its record and on no record it contradicts")
        if lk is not None:
            check(lk >= 0, f"the held-out likelihood not down ({lk:+g})")
    elif phase == "exploit":
        lk, s = need("likelihood"), need("success")
        if lk is not None and s is not None:
            check(lk > 0 or (s > 0 and lk >= 0),
                  f"a generalization gain: the likelihood up, or the success up with the likelihood not down "
                  f"(likelihood {lk:+g}, success {s:+g})")
    else:
        r, s, v = need("replay_identical"), need("success"), need("vector")
        if r is not None:
            check(bool(r), "the records replay the same")
        if s is not None:
            check(s >= 0, f"the held-out success not down ({s:+g})")
        if n.get("likelihood") is not None:
            check(n["likelihood"] >= 0, f"the held-out likelihood not down ({n['likelihood']:+g})")
        if v is not None:
            dv = _vector(v)
            check(dv < 0, f"the complexity vector down ({dv:+g})")
    return {"verdict": "accept" if ok else "refuse", "phase": phase, "numbers": n, "why": why}


def _ask_record(a):
    """The disengagement record of an ask, its fields checked (POLICY.md, the minimal-risk
    condition)."""
    if a.get("phase") not in PHASES:
        raise Refused(f"ask: phase {' | '.join(PHASES)}, not {json.dumps(a.get('phase'))}")
    if a.get("reason") not in REASONS:
        raise Refused(f"ask: reason {' | '.join(REASONS)}, not {json.dumps(a.get('reason'))}")
    q = a.get("question")
    if not isinstance(q, str) or not q.strip():
        raise Refused("ask: question, the one question for the person")
    options = list(a.get("options") or [])
    if len(options) > 3:
        raise Refused(f"ask: at most three options, not {len(options)}")
    return {"phase": a["phase"], "reason": a["reason"], "tried": list(a.get("tried") or []),
            "believes": a.get("believes") or [], "options": options, "question": q.strip(),
            "snapshot": a.get("snapshot")}


def _instruction(ins):
    """The person's answer to an ask, checked: {kind, …} (a plain string is a note)."""
    if isinstance(ins, str):
        ins = {"kind": "note", "text": ins}
    if not isinstance(ins, dict) or ins.get("kind") not in ANSWERS:
        raise Refused(f"answer: an instruction of kind {' | '.join(ANSWERS)}")
    k = ins["kind"]
    if k in ("note", "rule") and not (ins.get("text") or "").strip():
        raise Refused(f"answer {k}: text")
    if k == "edit" and not all(isinstance(ins.get(x), str) for x in ("path", "old", "new")):
        raise Refused("answer edit: path, old, new (proposed again through the gate)")
    if k == "budget" and not any(isinstance(ins.get(x), (int, float)) for x in ("beats", "proposals", "spend")):
        raise Refused("answer budget: beats, proposals or spend (a number)")
    if k == "phase" and ins.get("phase") not in PHASES + ("skip",):
        raise Refused(f"answer phase: {' | '.join(PHASES + ('skip',))}")
    return dict(ins)


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
    ({refused: {root: {path: text}}, plans: {token: plan}, answers: {token: answer}});
    `save` persists it. `policy` "experimental" judges program/extension/edit by numbers
    (`gain(plan, phase)`, or `numbers` given); `journal` is the JSON-lines file of the
    verdicts, the asks and the answers (kept in memory too: journal())."""

    def __init__(self, store=None, save=None, policy=None, gain=None, journal=None):
        if (policy or "human") not in POLICIES:
            raise Refused(f"policy: {' | '.join(POLICIES)}, not {json.dumps(policy)}")
        self.store = store if store is not None else {}
        self.store.setdefault("refused", {})
        self.store.setdefault("plans", {})
        self.store.setdefault("answers", {})
        self._save = save or (lambda s: None)
        self.policy = policy or "human"
        self.gain = gain
        self.journal_path = Path(journal) if journal else None
        self._journal = []

    @classmethod
    def at(cls, state_dir, policy=None, gain=None):
        """A gate whose memory is state_dir/gate.json (the loop's state: DIR/.metis/),
        written whole and atomically (a temp file, then os.replace); the journal is
        state_dir/journal.jsonl."""
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
        return cls(store, save, policy, gain, Path(state_dir) / "journal.jsonl")

    @classmethod
    @__import__("contextlib").contextmanager
    def locked(cls, state_dir, policy=None, gain=None):
        """Gate.at under an exclusive lock on state_dir/gate.lock for the whole of one
        use (load, change, save): two loops in one DIR never lose each other's memory."""
        import fcntl
        d = _state_dir(Path(state_dir))
        with open(d / "gate.lock", "a") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                yield cls.at(d, policy, gain)
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

    # -- the journal ----------------------------------------------------------

    def _log(self, entry):
        entry = dict(entry, ts=round(time.time(), 3))
        self._journal.append(entry)
        if self.journal_path:
            _state_dir(self.journal_path.parent)
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return entry

    def journal(self):
        """The journal's entries: the file's when the gate has one, else this process's."""
        if self.journal_path:
            try:
                return [json.loads(x) for x in self.journal_path.read_text().splitlines() if x.strip()]
            except OSError:
                return []
        return list(self._journal)

    # -- propose / confirm / refuse -------------------------------------------

    def propose(self, kind, domain, root=None, run=None, **args):
        """A plan of the write `kind` for the domain into root: {token, kind, domain, root,
        files, reopen, note?, say?}; under the experimental policy a judged kind also has
        its verdict (and written, or refused). Raises Refused with the rule's words."""
        root = str(Path(root).resolve() if root else catalog.root())
        if kind == "ask":
            return self._ask(domain, root, args)
        judged = self.policy == "experimental" and kind in JUDGED
        phase, cost, numbers = (args.pop(k, None) for k in ("phase", "cost", "numbers"))
        if judged and phase not in PHASES:
            raise Refused(f"the experimental policy judges a {kind} in a phase: phase {' | '.join(PHASES)}")
        pl = self._plan(kind, domain, root, run, args, since=self.refused(root))
        pl["token"] = hashlib.sha256(json.dumps(
            [kind, domain, root, [(f["path"], f["text"]) for f in pl["files"]], pl.get("note")]).encode()).hexdigest()[:16]
        self.store["plans"][pl["token"]] = {"kind": kind, "domain": domain, "root": root, "run": run, "args": args,
                                            "files": [[f["path"], f["text"]] for f in pl["files"]]}
        self._save(self.store)
        if not judged:
            return pl
        if numbers is None:
            if self.gain is None:
                raise Refused("the experimental policy needs the numbers: a gain hook, or numbers in the proposal")
            numbers = self.gain(public(pl), phase)
        v = judge(phase, numbers)
        entry = {"event": "proposal", "token": pl["token"], "kind": kind, "domain": domain, "root": root,
                 "phase": phase, "files": [f["path"] for f in pl["files"]], "numbers": v["numbers"],
                 "verdict": v["verdict"], "why": v["why"], "cost": cost}
        pl = public(pl)
        pl["verdict"] = v
        if v["verdict"] == "accept":
            try:
                pl.update(self.confirm(pl["token"]))
            except Refused as e:
                self._log(dict(entry, error=str(e)))
                raise
            entry["written"] = pl["written"]
        else:
            pl["refused"] = self.refuse(pl["token"])["refused"]
        self._log(entry)
        return pl

    def _ask(self, domain, root, args):
        """kind ask: the disengagement record waits for the person's answer; nothing is written."""
        if not NAME.match(domain or ""):
            raise Refused(f"domain {json.dumps(domain)}: lowercase letters, digits, _")
        rec = _ask_record(args)
        n = sum(1 for e in self.store["plans"].values() if e["kind"] == "ask") + len(self.store["answers"])
        token = hashlib.sha256(json.dumps(["ask", domain, root, rec, n], default=str).encode()).hexdigest()[:16]
        self.store["plans"][token] = {"kind": "ask", "domain": domain, "root": root, "run": None,
                                      "args": rec, "files": []}
        self._save(self.store)
        self._log({"event": "ask", "token": token, "domain": domain, "root": root, **rec})
        return {"token": token, "kind": "ask", "domain": domain, "root": root, "files": [], "reopen": False,
                "ask": rec}

    def answer(self, token, instruction):
        """The person's instruction for the ask `token`: {token, domain, ask, instruction},
        kept (answers()) for the loop to read at re-engagement; the ask is done."""
        p = self.store["plans"].get(token)
        if p is None or p["kind"] != "ask":
            raise Refused("no such ask: it was answered, or never asked")
        ins = _instruction(instruction)
        self.store["plans"].pop(token)
        out = {"token": token, "domain": p["domain"], "root": p["root"], "ask": p["args"], "instruction": ins}
        self.store["answers"][token] = out
        self._save(self.store)
        self._log({"event": "answer", "token": token, "domain": p["domain"], "phase": p["args"]["phase"],
                   "reason": p["args"]["reason"], "instruction": ins})
        return out

    def asks(self):
        """The asks waiting for the person: [{token, domain, root, ask}]."""
        return [{"token": t, "domain": p["domain"], "root": p["root"], "ask": p["args"]}
                for t, p in self.store["plans"].items() if p["kind"] == "ask"]

    def answers(self, token=None):
        """The answered asks (one by token, or None when it has no answer yet)."""
        if token is not None:
            return self.store["answers"].get(token)
        return list(self.store["answers"].values())

    def refuse(self, token):
        """The person refused the plan: its texts are remembered (the next plan of each file
        diffs from it); the plan is dropped."""
        p = self.store["plans"].get(token)
        if p is not None and p["kind"] == "ask":
            raise Refused("an ask is answered, not refused: answer it (abandon ends the run)")
        p = self.store["plans"].pop(token, None)
        if p is None:
            raise Refused("no such plan: propose it again")
        self._remember(p["root"], [{"path": a, "text": b} for a, b in p["files"]], False)
        self._save(self.store)
        return {"refused": [a for a, _ in p["files"]]}

    def confirm(self, token):
        """Write exactly the plan `token` names: {written, fingerprint, note?, serve?, bundle?}.
        A source that changed since the plan refuses (nothing written)."""
        p = self.store["plans"].get(token)
        if p is not None and p["kind"] == "ask":
            raise Refused("an ask is answered, not confirmed: answer it with the person's instruction")
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
    ap.add_argument("--policy", choices=POLICIES, default="human",
                    help="experimental: program/extension/edit judged by the numbers in the args (phase, numbers, cost)")
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
    an = sub.add_parser("answer", help="the person's instruction for the ask TOKEN (JSON on stdin, or a note's text)")
    an.add_argument("token")
    ls = sub.add_parser("answers", help="the answered asks (one with --token), and the asks waiting")
    ls.add_argument("--token")
    jl = sub.add_parser("journal", help="the journal's entries")
    for p in (ap, pr, cf, rf, an, ls, jl):
        p.add_argument("--json", action="store_true", help="JSON on stdout")
    a = ap.parse_args(argv)
    raw = sys.stdin.read() if a.cmd in ("propose", "answer") else ""
    try:
        with Gate.locked(a.state, a.policy) as g:
            if a.cmd == "propose":
                args = json.loads(raw) if raw.strip() else {}
                out = public(g.propose(a.kind, a.domain, a.root, a.run, **args))
            elif a.cmd == "confirm":
                out = g.confirm(a.token)
            elif a.cmd == "answer":
                try:
                    ins = json.loads(raw)
                except ValueError:
                    ins = raw.strip()
                out = g.answer(a.token, ins)
            elif a.cmd == "answers":
                out = g.answers(a.token) if a.token else {"answers": g.answers(), "asks": g.asks()}
            elif a.cmd == "journal":
                out = g.journal()
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
