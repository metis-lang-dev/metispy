"""The loop verbs, mounted on the metis shell (one kernel REPL;
every UI a wrapper). Each verb is a `cmd_*` returning a dict — the
shell prints them for humans (its `do_*` wrappers), the JSON wire
(metis.loop.wire) ships them to the console verbatim.

State mapping onto the shell: the WORLD is the served catalog plus
accepted packs (`accept` re-serves the merged catalog live — the
interactive session immediately plays the extended world, keys and
events included); the SCENE is the served file's #init/#steps case;
the loop's own state (approved TEXT, CANDIDATE, REPORT, records) is
a LoopState the shell carries."""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from metis.kernel import (STATE_BUDGET, ForwardProjection,
                          StateBudgetExceeded)
from metis.kernel.record import recover
from metis.lang import compile_catalog, parse, resolve_includes
from metis.lang.ast import pretty
from metis.lang.formalize import admit_text_pack


@dataclass
class LoopState:
    text: str = ""                   # the approved unit (doc text)
    cand: str = ""                   # candidate LLP source
    base_src: str = ""               # revised world source ('' = the
                                     # served file is the base)
    report: object = None
    accepted: List[str] = field(default_factory=list)  # pack sources
    records: List[dict] = field(default_factory=list)
    rec_idx: Optional[int] = None
    revision: bool = False           # next accept REPLACES the base


class LoopVerbs:
    """Mixin for the shell Repl: the loop lifecycle as cmd_* verbs.
    Requires from the host: self.loop (LoopState), self.doc,
    self.path, self.case, self.text (served source), self.llm, and
    `_reserve(catalog)` to re-serve a merged/revised catalog live."""

    # -- scene helpers -------------------------------------------------

    def _scene_init(self) -> Dict[str, int]:
        return {a: n for a, n in self.case["init"].items() if n > 0}

    def _scene_steps(self) -> int:
        return self.case["steps"] or 6

    def _world(self):
        """The accepted world: served catalog + accepted packs (the
        shell re-serves on accept, so self.doc.catalog IS it)."""
        return self.doc.catalog

    def _base_src(self) -> str:
        """The base the next pack gates against: the revised world's
        source if a revision was accepted, else the served file."""
        return self.loop.base_src or self.text

    def _gate_base(self):
        """The world the next pack gates against: the base source with
        the packs accepted so far, each compiled against the world so far
        and admitted over it, in order — the artifact emitter's chain
        (metis.kernel.artifact.served_doc), so a second pack's merge
        keeps the first."""
        from dataclasses import replace
        from metis.lang.compiler import merge_ports
        doc = compile_catalog(resolve_includes(parse(self._base_src()),
                                               self.path.parent))
        for src in self.loop.accepted:
            pack = compile_catalog(resolve_includes(parse(src),
                                                    self.path.parent),
                                   base=doc)
            doc = replace(doc, catalog=doc.catalog.admit(pack.catalog),
                          ports=merge_ports(doc.ports, pack.ports))
        return doc

    # -- the candidate -------------------------------------------------

    def cmd_text(self, arg):
        """Set the approved doc text (TEXT) formalize drafts from —
        the person's words of what the world should do."""
        self.loop.text = arg
        return {"ok": True, "chars": len(arg)}

    def cmd_llp(self, arg):
        self.loop.cand = arg
        return {"ok": True, "lines": arg.count("\n") + 1}

    def cmd_load(self, arg):
        """Candidate from a .llp FILE (includes spliced); its leading
        comment block becomes the TEXT when none is set."""
        from pathlib import Path
        if not arg.strip():
            return {"ok": False,
                    "error": "usage: load FILE.llp (a candidate pack) "
                             "— to re-read the served file, reload"}
        p = Path(arg.strip()).expanduser()
        try:
            src = p.read_text()
        except OSError as e:
            return {"ok": False, "error": f"cannot load {p}: "
                                          f"{e.strerror or e}"}
        parsed = parse(src)
        cat = resolve_includes(parsed, p.parent)
        # VERBATIM unless includes were spliced: the pretty-printer
        # drops %% docs (the approval units) and '#' directives, and
        # what is gated, accepted and exported must be what was written
        self.loop.cand = src if cat == parsed else pretty(cat)
        out = {"ok": True, "file": str(p),
               "lines": self.loop.cand.count("\n") + 1}
        from metis.lang.directives import embedded_case
        case = embedded_case(src)
        adopted = {}
        if case["init"] and not self.case["init"]:
            self.case["init"] = {a: n for a, n in case["init"].items()
                                 if n > 0}
            adopted["init"] = self.case["init"]
        if case["steps"] and not self.case["steps"]:
            self.case["steps"] = case["steps"]
            adopted["steps"] = case["steps"]
        if adopted:
            out["adopted_directives"] = adopted
        if not self.loop.text.strip():
            head = []
            for line in src.splitlines():
                if line.startswith("%"):
                    head.append(line)
                elif line.strip():
                    break
            if head:
                self.loop.text = "\n".join(head)
                out["adopted_text_chars"] = len(self.loop.text)
        return out

    def cmd_formalize(self, arg):
        """Draft the candidate from the TEXT through the configured
        llm endpoint (repair loop: gate findings feed back; each
        round gates against the served world, exactly as `gate`
        does)."""
        import shlex

        from .core import formalize_loop
        from .llm import resolve
        if not self.loop.text.strip():
            return {"ok": False, "error": "no text — mine/text first"}
        spec = (arg.strip() or self.llm or "ollama")
        ep = resolve(spec)
        if ep.backend == "command":
            backend = shlex.split(ep.arg)
        else:
            from .llm import complete

            def backend(prompt, _ep=ep):
                return complete(_ep, prompt)
        init = self._scene_init() or self._cand_case_init()
        base_path = self.path
        tmp = None
        # a blank world has nothing to extend: its candidate IS a base
        blank = not getattr(self, "events_tbl", True)
        mode = "revise" if blank else "auto"
        if self.loop.base_src:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                "w", suffix=".llp", delete=False)
            tmp.write(self.loop.base_src)
            tmp.close()
            base_path = tmp.name
        try:
            res = formalize_loop(
                self.loop.text, self._base_src(), backend,
                base_path=base_path,
                scenarios=[init] if init else (),
                steps=self._scene_steps(),
                system=getattr(self, "sys_prompts",
                               {}).get("formalize"),
                mode=mode)
        finally:
            if tmp is not None:
                import os
                os.unlink(tmp.name)
        self.loop.cand, self.loop.report = res.llp, res.report
        # gated as a base edit -> accept REPLACES the world, as `revise`
        self.loop.revision = res.ok and res.mode == "revise"
        out = {"ok": res.ok, "backend": spec, "rounds": res.rounds,
               "gated_as": ("revise (base edit)" if res.mode == "revise"
                            else "pack"),
               "findings": (res.report.findings if res.report
                            else [])}
        # the repair rounds gate PRIOR-ONLY (the seat cannot bind ports,
        # a binding-demanding loop would spin) — but say what accept
        # will need, so the green report is not a surprise refusal later
        doc = getattr(res.report, "doc", None) if res.ok else None
        if doc is not None and doc.ports.weight_refs:
            from metis.lang.compiler import merge_ports
            ports = (doc.ports if res.mode == "revise"
                     else merge_ports(self.doc.ports, doc.ports))
            bound = set(self._gate_registry())
            unbound = sorted({r.factor for r in ports.weight_refs.values()}
                             - bound)
            if unbound:
                w = (f"unbound weight port(s) {', '.join(unbound)} — "
                     "gated prior-only; accept needs them bound "
                     "(`plugin SPEC` / `metis --plugin SPEC`)")
                res.report.warnings.append(w)
                out["warnings"] = [w]
        return out

    def _cand_case_init(self) -> Dict[str, int]:
        """The candidate's own #init (a born world's scenario), when it
        carries one."""
        from metis.lang.directives import DirectiveError, embedded_case
        try:
            return {a: n for a, n in
                    embedded_case(self.loop.cand)["init"].items() if n > 0}
        except (DirectiveError, ValueError):
            return {}

    def _adopt_cand_case(self) -> Dict:
        """F5: a revision's own #init/#steps become the session's case
        when the session has none — and a blank world starts there."""
        from metis.lang.directives import DirectiveError, embedded_case
        try:
            case = embedded_case(self.loop.cand)
        except (DirectiveError, ValueError):
            return {}
        adopted = {}
        init = {a: n for a, n in case["init"].items() if n > 0}
        if init and not self._scene_init():
            self.case["init"] = dict(init)
            adopted["init"] = dict(init)
            if not any(n > 0 for n in getattr(self, "counts", {}).values()):
                self.counts = dict(init)       # the born world starts here
                if hasattr(self, "_rebase"):
                    self._rebase()
        if case["steps"] and not self.case["steps"]:
            self.case["steps"] = case["steps"]
            adopted["steps"] = case["steps"]
        return adopted

    def cmd_gate(self, arg):
        """Gate the candidate as a PACK against the served world (the
        base plus the packs accepted so far)."""
        if not self.loop.cand.strip():
            return {"ok": False,
                    "error": "no candidate — llp/load/formalize first"}
        base = self._gate_base()
        init = self._scene_init()
        rep = admit_text_pack(self.loop.text, self.loop.cand,
                              base=base,
                              scenarios=(init,) if init else (),
                              steps=self._scene_steps(),
                              registry=self._gate_registry())
        self.loop.report, self.loop.revision = rep, False
        return self._gate_out(rep)

    def cmd_revise(self, arg):
        """Gate the candidate as a BASE EDIT (accept then REPLACES
        the served world; measure diff on the scene shown)."""
        if not self.loop.cand.strip():
            return {"ok": False, "error": "no candidate — llp first"}
        init = self._scene_init() or self._cand_case_init()
        rep = admit_text_pack(self.loop.text, self.loop.cand,
                              base=None,
                              scenarios=(init,) if init else (),
                              steps=self._scene_steps(),
                              registry=self._gate_registry())
        self.loop.report, self.loop.revision = rep, rep.ok
        out = self._gate_out(rep)
        out["revision"] = True
        if rep.ok:
            try:
                out["diff_preview"] = self._revision_diff(rep)
            except StateBudgetExceeded as e:
                out["diff_preview"] = []
                out["diff_note"] = (f"scene too large for the exact "
                                    f"before/after diff: {e}")
        return out

    def _gate_registry(self):
        """The session's resolved weight bindings for gate scenarios
        (PLUGIN-PORTS.md point 5): the plugin's registry, overlays
        included; {} without one — a candidate's @w port is then
        unbound, a finding."""
        reg = getattr(self, "registry", None)
        return reg.weight_bindings() if reg is not None else {}

    def _gate_out(self, rep):
        import json
        from metis.kernel.diagnostics import to_json
        out = {"ok": rep.ok, "classification": rep.classification,
               "findings": rep.findings, "warnings": rep.warnings,
               "rules": [list(r) for r in rep.rules],
               # spec 08: the structured slices behind the findings
               # (findings stay the text rendering)
               "diagnostics": [json.loads(to_json(d))
                               for d in rep.diagnostics]}
        if rep.scenarios:
            out["scenarios"] = [
                {"key": s.key, "support": s.support,
                 "diffs": [list(d) for d in s.diffs],
                 "map_events": s.map_events} for s in rep.scenarios]
        return out

    def _revision_diff(self, rep):
        init = self._scene_init()
        if not init:
            return []

        def marg(cat):
            pr = ForwardProjection(cat.ground(dict(init)),
                                   self._scene_steps(),
                                   max_states=STATE_BUDGET,
                                   keep_history=False)
            m: Dict[str, float] = {}
            for (_, kappa), p in pr.final.items():
                for a, n in kappa:
                    if n > 0:
                        m[a] = m.get(a, 0.0) + p
            return m
        old, new = marg(self._world()), marg(rep.doc.catalog)
        return [[a, round(old.get(a, 0.0), 12),
                 round(new.get(a, 0.0), 12)]
                for a in sorted(set(old) | set(new))
                if abs(old.get(a, 0.0) - new.get(a, 0.0)) > 1e-12]

    def cmd_accept(self, arg):
        """Admitted candidate joins the served world — the session
        re-serves the merged catalog LIVE (new key, new events;
        counts and stage carry over)."""
        rep = self.loop.report
        if not (rep and rep.ok):
            return {"ok": False, "error": "nothing admitted — gate first"}
        out = {"ok": True}
        if self.loop.revision:
            new_world, ports = rep.doc.catalog, rep.doc.ports
        else:
            new_world = (rep.merged if rep.merged is not None
                         else rep.doc.catalog)
            from metis.lang.compiler import merge_ports
            ports = merge_ports(self.doc.ports, rep.doc.ports)
        if self.loop.revision:
            try:
                out["diffs"] = self._revision_diff(rep)
            except StateBudgetExceeded as e:
                # the diff is a readout, never the gate: an over-budget
                # scene must not block an admitted revision (as revise)
                out["diffs"] = []
                out["diff_note"] = (f"scene too large for the exact "
                                    f"before/after diff: {e}")
        # re-serve FIRST: a refusal (an unbound weight port) must leave
        # the loop state untouched — accept is atomic
        try:
            self._reserve(new_world, ports)
        except Exception as e:
            findings = getattr(e, "findings", None) or [str(e)]
            return {"ok": False,
                    "error": "accept refused — the world was not "
                             "re-served: " + "; ".join(findings),
                    "hint": "bind the ports first (`plugin SPEC`, or "
                            "`metis --plugin SPEC`), then accept again"}
        if self.loop.revision:
            out["dropped_packs"] = len(self.loop.accepted)
            self.loop.accepted = []
            self.loop.revision = False
            self.loop.base_src = self.loop.cand
            adopted = self._adopt_cand_case()
            if adopted:
                out["adopted_directives"] = adopted
        else:
            self.loop.accepted.append(self.loop.cand)
        self.loop.cand, self.loop.report = "", None
        out["accepted_packs"] = len(self.loop.accepted)
        out["key"] = self.key
        return out

    # -- the surprise / refinement loop --------------------------------

    def _rec_line(self, i, r):
        return {"idx": i, "kind": r.get("kind"),
                "likelihood": r.get("likelihood"),
                "steps": r.get("steps"),
                "observed": r.get("observed", {}),
                "source": (r.get("meta") or {}).get("source", ""),
                "mined": bool(r.get("mined_text"))}

    def cmd_sink(self, arg):
        from .core import load_records
        if not arg.strip():
            return {"ok": False, "error": "usage: sink FILE.jsonl"}
        self.loop.records = load_records(arg.strip())
        self.loop.rec_idx = None
        return {"ok": True, "file": arg.strip(),
                "records": [self._rec_line(i, r)
                            for i, r in enumerate(self.loop.records)]}

    def cmd_records(self, arg):
        """The loaded surprise records, one line each (`record N`
        selects one)."""
        return {"ok": True,
                "records": [self._rec_line(i, r)
                            for i, r in enumerate(self.loop.records)]}

    def cmd_record(self, arg):
        """Select a record (default newest); its init/steps become
        the live scene (#init/#steps), its mined text the TEXT."""
        if not self.loop.records:
            return {"ok": False, "error": "no records — sink first"}
        i = int(arg) if arg.strip() else len(self.loop.records) - 1
        if not 0 <= i < len(self.loop.records):
            return {"ok": False, "error":
                    f"record {i} out of 0..{len(self.loop.records)-1}"}
        self.loop.rec_idx = i
        r = self.loop.records[i]
        self.case["init"] = {a: n for a, n in r.get("init", {}).items()
                             if n > 0}
        if r.get("steps"):
            self.case["steps"] = r["steps"]
        if r.get("mined_text"):
            self.loop.text = r["mined_text"]
        return {"ok": True, **self._rec_line(i, r),
                "init": self.case["init"],
                "program_key": r.get("program_key", "")}

    def cmd_replay(self, arg):
        """recover() the selected record against the served world —
        explanation before a fix, verification after (fixed: the
        likelihood moved positive)."""
        if self.loop.rec_idx is None:
            return {"ok": False, "error": "no record — record first"}
        r = self.loop.records[self.loop.rec_idx]
        prog = self._world().ground(self._scene_init())
        pricer = getattr(self, "pricer", None)
        try:
            out = recover(r, prog, steps=r.get("steps"),
                          max_states=STATE_BUDGET,
                          policy=pricer.policy(prog) if pricer else None,
                          plugin=getattr(self, "plugin", None))
        except StateBudgetExceeded as e:
            return {"ok": False, "error": str(e)}
        logged = r.get("likelihood", 0.0)
        now = out["likelihood"]
        return {"ok": True, "logged_likelihood": logged,
                "likelihood": now, "explains": out["explains"],
                "explanation_events": out.get("explanation_events"),
                **({"env": out["env"]} if out.get("env") else {}),
                "fixed": bool(now > 0 >= logged)}

    def cmd_mine(self, arg):
        """With TEXT: the approved unit, stamped on the record.
        Bare: the observed-vs-modeled gap to mine."""
        if self.loop.rec_idx is None:
            return {"ok": False, "error": "no record — record first"}
        r = self.loop.records[self.loop.rec_idx]
        if arg.strip():
            self.loop.text = arg
            r["mined_text"] = arg
            return {"ok": True, "chars": len(arg)}
        return {"ok": True, "kind": r.get("kind"),
                "likelihood": r.get("likelihood"),
                "init": r.get("init", {}),
                "observed": r.get("observed", {}),
                "hint": "state what happened that the world cannot "
                        "derive; quote atoms; that text is the "
                        "approved unit (mine <text>)"}

    # -- loop views ------------------------------------------------------

    def cmd_show(self, arg):
        """The loop's state: bare (or `state`) the served file, key,
        accepted packs, whether a candidate exists and passed the gate,
        records and the scene; `show text` the approved text; `show
        base` the base and accepted packs; `show llp` the candidate."""
        which = arg.strip() or "state"
        if which == "text":
            return {"ok": True, "text": self.loop.text}
        if which == "base":
            return {"ok": True, "base": self._base_src(),
                    "packs": self.loop.accepted,
                    "revised": bool(self.loop.base_src)}
        if which in ("llp", "candidate"):
            return {"ok": True, "llp": self.loop.cand}
        return {"ok": True, "file": str(self.path), "key": self.key,
                "accepted_packs": len(self.loop.accepted),
                "candidate": bool(self.loop.cand.strip()),
                "gated": bool(self.loop.report and self.loop.report.ok),
                "records": len(self.loop.records),
                "init": self._scene_init(),
                "steps": self._scene_steps()}

    def cmd_export(self, arg):
        """The accepted world as files: base.llp + packs + text —
        the raw material of a catalog commit."""
        from pathlib import Path
        d = Path(arg.strip() or "session")
        d.mkdir(parents=True, exist_ok=True)
        (d / "base.llp").write_text(self._base_src())
        for k, src in enumerate(self.loop.accepted, 1):
            (d / f"pack{k}.llp").write_text(src)
        if self.loop.text.strip():
            (d / "text.md").write_text(self.loop.text + "\n")
        return {"ok": True, "dir": str(d),
                "files": 1 + len(self.loop.accepted)
                + bool(self.loop.text.strip())}


def fmt(res: dict) -> str:
    """Human rendering of a verb result for the shell: narration and
    other long text raw, everything else compact JSON."""
    if res.get("ok") and "narration" in res:
        return res["narration"]
    if res.get("ok") and "help" in res:
        return res["help"]
    return json.dumps(res, indent=1)
