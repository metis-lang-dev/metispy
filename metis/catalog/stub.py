"""metis.catalog.stub — the INTERFACE STUB (metis-lang spec/15 §10.4, specialist-tree s-404d001f step 1d): one
generic driver that runs a specialist ALONE against its own cases, parameterized by its manifest. No director,
no world, no package-specific harness code: the case closes the interface with facts.

A package's cases are YAML files in its `cases/` directory. A case:

    case: nav goes to a receptacle elsewhere      # its name (default: the file's stem)
    seed: 1                                       # the session's RNG seed (default 1)
    init: [at(lhere)]                             # the state before the first boundary: it REPLACES the program's
                                                  #   #init (absent: the program's #init stands)
    steps: 100                                    # the most events a boundary runs to quiescence (default 100)
    boundaries:                                   # the script, one entry per tick boundary
      - lines: ["+at(lhere) 1"]                   # do-lines applied verbatim, first (a journaled session's exo rows:
                                                  #   F1's replay of a member alone from its input log)
        run: true                                 # false: the inputs only, no rule runs (the Live's take of a grant)
        world: [at(lhere), recat(target, l1)]     # the world-private readings NOW (synced: the listed atoms of
                                                  #   those preds stand, every other atom of them is retracted;
                                                  #   `atom 0` reads an atom gone)
        deliver: [goto(target)]                   # in-copies, asserted with this boundary's stamp (linear: a
                                                  #   trigger consumes its copy)
        grant: [motion]                           # resources granted from this boundary on (revoke: [...] ends it)
        expect:                                   # what the boundary must show (each key optional)
          out: [reached(target)]                  #   the out atoms collected at this boundary (exactly)
          status: success                         #   running | success | failure | idle
          fired: [go]                             #   the rules fired, in order (short names)
          held: false                             #   intents fired without the resource held (a host-gated member's;
                                                  #   gating: program never holds: its rules read the grant)

Per boundary the stub:
1. checks every atom against the manifest's `shared` directions: `world` atoms must be world-private, `deliver`
   atoms in, `grant`/`revoke` names resource (or a `resources` entry). A case that feeds anything else is not a
   case of this specialist (CaseError). A package without `shared` (a single member, the dog) is the degenerate
   case: nothing to check, every atom a reading or a delivery;
2. syncs the world readings, asserts the deliveries, applies the grants (the resource's token atom asserted
   when the program declares its pred — the manifest's `free` pred over the resource, else the 0-ary pred of
   its name — else the grant is the holder flag only, as the tick's host enforces it);
3. runs the session to quiescence (at most `steps` events), recording each fired row with its probability;
4. collects the OUT atoms produced (they are handed over and retracted: the Moore latch's release to the
   director) and the INTENTS (the fired events), marked `held` when the specialist requests a resource it does
   not hold (only the holder's commands would reach the world);
5. derives the boundary's status from the manifest's `status` atoms (spec/15 amendment 1b: success when a
   success atom was put out, failure when a failure atom was, else running when anything fired, else idle);
6. scores the boundary against `expect`.

    from metis.catalog import stub
    r = stub.run_case(domain, "cases/far.yaml")     # → CaseResult: .passed, .boundaries [BoundaryResult]
    stub.run_package(domain)                        # → [CaseResult], every case in cases/

    python -m metis.catalog cases NAME [--json]     (exit 1 when a case fails)"""
import contextlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

STATUSES = ("running", "success", "failure", "idle")


class CaseError(ValueError):
    """A case that does not fit the specialist's interface (or is malformed): not a failing case, a wrong one."""


def pred_of(atom):
    return atom.split("(")[0].strip()


def _atom(s):
    """'goto(target)' or 'goto(target) 2' or 'motion' → (the atom with no spaces, its count)."""
    m = re.match(r"^(.*?)\s+(\d+)$", str(s).strip())
    atom, n = (m.group(1), int(m.group(2))) if m and (")" in m.group(1) or "(" not in m.group(1)) else (str(s), 1)
    return "".join(atom.split()), n


def _atoms(items):
    """A case's atom list → [(atom, count)]. A YAML flow list splits `recat(dest, l1)` at its comma; the pieces
    are joined again until the parentheses balance, so a case needs no quotes."""
    out, cur = [], ""
    for x in items or ():
        cur = (cur + "," + str(x)) if cur else str(x)
        if cur.count("(") == cur.count(")"):
            out.append(_atom(cur))
            cur = ""
    if cur:
        raise CaseError("unbalanced atom %r" % cur)
    return out


@dataclass
class Interface:
    """What the stub reads of a manifest: the directions, the status atoms, the resources, the gating."""
    shared: Dict[str, str]
    success: List[str]
    failure: List[str]
    resources: Dict[str, dict]
    gating: str = "host"                # host: its commands held without the grant | program: its rules gate themselves

    @classmethod
    def of(cls, manifest):
        sh = dict(manifest.get("shared") or {})
        st = manifest.get("status") or {}
        res = {k: {} for k, v in sh.items() if v == "resource"}
        res.update(manifest.get("resources") or {})
        return cls(sh, list(st.get("success") or ()), list(st.get("failure") or ()), res, manifest.get("gating") or "host")

    @property
    def degenerate(self):
        return not self.shared

    def ins(self, direction):
        return {p for p, d in self.shared.items() if d == direction}

    @property
    def outs(self):
        return self.ins("out")


@dataclass
class BoundaryResult:
    b: int
    world: List[str]
    delivered: List[str]
    holding: List[str]
    fired: List[dict]                       # [{"event", "rule", "p"}]
    outs: List[str]
    intents: List[dict]                     # [{"event", "held"}]
    status: str
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self):
        return not self.failures


@dataclass
class CaseResult:
    name: str
    file: str
    boundaries: List[BoundaryResult]

    @property
    def passed(self):
        return all(b.passed for b in self.boundaries)

    def as_dict(self):
        return {"case": self.name, "file": self.file, "passed": self.passed,
                "boundaries": [dict(vars(b), passed=b.passed) for b in self.boundaries]}


class Stub:
    """One specialist's session, driven boundary by boundary."""

    def __init__(self, program, manifest, seed=1):
        from metis.lang.repl import Repl
        self.iface = Interface.of(manifest)
        with contextlib.redirect_stdout(io.StringIO()):
            self.repl = Repl(str(program), seed=seed)
        self.preds = set(self.repl.doc.catalog.preds) | set(self.repl.doc.catalog.bwd)
        self.holding = set()

    def _do(self, line):
        with contextlib.redirect_stdout(io.StringIO()):
            self.repl.onecmd(line)

    def _count(self, atom):
        return self.repl.counts.get(atom, 0)

    def _check(self, atoms, allowed, what):
        bad = sorted({pred_of(a) for a in atoms} - allowed)
        if bad:
            raise CaseError("%s %s: not %s in the manifest's shared (those are: %s)"
                            % ("delivered" if what == "in" else "world reading", ", ".join(bad),
                               "in" if what == "in" else "world-private", ", ".join(sorted(allowed)) or "none"))

    def _token(self, r):
        """The resource's token atom when the program declares it: free(r) over the manifest's free pred, else the
        0-ary pred of its name; None: the grant is the holder flag only."""
        free = (self.iface.resources.get(r) or {}).get("free")
        if free and free in self.preds:
            return "%s(%s)" % (free, r)
        return r if r in self.preds else None

    def boundary(self, b, spec, steps=100):
        world = _atoms(spec.get("world"))
        deliver = _atoms(spec.get("deliver"))
        grant, revoke = list(spec.get("grant") or ()), list(spec.get("revoke") or ())
        if not self.iface.degenerate:
            self._check([a for a, _ in world], self.iface.ins("world-private"), "world")
            self._check([a for a, _ in deliver], self.iface.ins("in"), "in")
            badr = sorted(set(grant + revoke) - set(self.iface.resources))
            if badr:
                raise CaseError("grant/revoke %s: not a resource of the manifest (%s)"
                                % (", ".join(badr), ", ".join(sorted(self.iface.resources)) or "none"))
        # 2. the journaled do-lines as they are (no direction check: a replay's inputs are what the member was
        # given), then the world's readings (synced per pred), the in-copies, the grants
        for line in spec.get("lines") or ():
            line = str(line).strip()
            if not line or line[0] not in "+-":
                raise CaseError("lines: %r is no do-line (+atom n | -atom n)" % line)
            self._do(line)
        if world:
            preds = {pred_of(a) for a, _ in world}
            want = dict(world)
            for a, n in sorted(self.repl.counts.items()):
                if n > 0 and pred_of(a) in preds and a not in want:
                    self._do("-%s %d" % (a, n))
            for a, n in world:
                have = self._count(a)
                if n > have:
                    self._do("+%s %d" % (a, n - have))
                elif n < have:
                    self._do("-%s %d" % (a, have - n))
        for a, n in deliver:
            self._do("+%s %d" % (a, n))
        for r in revoke:
            self.holding.discard(r)
            t = self._token(r)
            if t and self._count(t):
                self._do("-%s %d" % (t, self._count(t)))
        for r in grant:
            self.holding.add(r)
            t = self._token(r)
            if t and not self._count(t):
                self._do("+%s 1" % t)
        # 3. to quiescence (run: false — the inputs only, as the Live's take of a grant: no rule fires until the
        # member's next beat)
        before = {a: n for a, n in self.repl.counts.items() if n > 0}
        fired = []
        with contextlib.redirect_stdout(io.StringIO()):
            while spec.get("run", True) and len(fired) < steps and self.repl._candidates():
                name = self.repl._sample()
                fired.append({"event": name, "rule": name.split("[")[0], "p": self.repl.fired_how[-1].get("p")})
        # 4. the outs handed over (retracted), the intents
        outs = []
        for a, n in sorted(self.repl.counts.items()):
            if n > before.get(a, 0) and (pred_of(a) in self.iface.outs if not self.iface.degenerate else False):
                outs += [a] * (n - before.get(a, 0))
        for a in sorted(set(outs)):
            self._do("-%s %d" % (a, outs.count(a)))
        needs = sorted(self.iface.resources)
        # gating: program — the program reads its grant in its own rules, so whatever fired is sent (only the host gate
        # holds intents: a host-gated member's without the resource)
        held = self.iface.gating != "program" and bool(needs) and not all(r in self.holding for r in needs)
        intents = [{"event": f["event"], "held": held} for f in fired]
        # 5. the status
        put = {pred_of(a) for a in outs}
        if put & set(self.iface.failure):
            status = "failure"
        elif put & set(self.iface.success):
            status = "success"
        elif fired:
            status = "running"
        else:
            status = "idle"
        return BoundaryResult(b, [a for a, _ in world], [a for a, _ in deliver], sorted(self.holding), fired, outs,
                              intents, status)


def _score(res, expect):
    fails = []
    if "out" in expect:
        want = sorted(a for a, n in _atoms(expect["out"]) for _ in range(n))
        if sorted(res.outs) != want:
            fails.append("out %s, expected %s" % (sorted(res.outs), want))
    if "status" in expect:
        if expect["status"] not in STATUSES:
            raise CaseError("status %r: one of %s" % (expect["status"], ", ".join(STATUSES)))
        if res.status != expect["status"]:
            fails.append("status %s, expected %s" % (res.status, expect["status"]))
    if "fired" in expect:
        got = [f["rule"] for f in res.fired]
        if got != list(expect["fired"] or ()):
            fails.append("fired %s, expected %s" % (got, list(expect["fired"] or ())))
    if "held" in expect:
        got = any(i["held"] for i in res.intents)
        if got != bool(expect["held"]):
            fails.append("held %s, expected %s" % (got, bool(expect["held"])))
    return fails


def load_case(path):
    c = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(c, dict) or not isinstance(c.get("boundaries"), list):
        raise CaseError("%s: a case is a mapping with boundaries: [...]" % path)
    unknown = sorted(set(c) - {"case", "seed", "init", "steps", "boundaries"})
    if unknown:
        raise CaseError("%s: unknown key%s %s" % (path, "s" if len(unknown) > 1 else "", ", ".join(unknown)))
    for i, b in enumerate(c["boundaries"]):
        bad = sorted(set(b or {}) - {"lines", "world", "deliver", "grant", "revoke", "run", "expect"})
        if bad:
            raise CaseError("%s: boundary %d: unknown key%s %s" % (path, i, "s" if len(bad) > 1 else "", ", ".join(bad)))
    return c


def run_case(domain, case_path):
    """Run one case file against DOMAIN (a manifests.Domain, or (program path, manifest dict)). → CaseResult."""
    program, manifest = (domain.program, domain.manifest) if hasattr(domain, "manifest") else domain
    c = load_case(case_path)
    s = Stub(program, manifest, seed=c.get("seed", 1))
    if "init" in c:                                    # the case's state replaces the program's #init
        for a, n in sorted(s.repl.counts.items()):
            if n > 0:
                s._do("-%s %d" % (a, n))
    for atom, n in _atoms(c.get("init")):
        s._do("+%s %d" % (atom, n))
    out = []
    for i, b in enumerate(c["boundaries"]):
        r = s.boundary(i, b or {}, steps=c.get("steps", 100))
        r.failures = _score(r, (b or {}).get("expect") or {})
        out.append(r)
    return CaseResult(c.get("case") or Path(case_path).stem, str(case_path), out)


def cases_of(domain):
    """The package's case files: cases/*.yaml in name order."""
    d = Path(domain.dir) / "cases"
    return sorted(d.glob("*.yaml")) if d.is_dir() else []


def run_package(domain):
    return [run_case(domain, p) for p in cases_of(domain)]


def render(results):
    """A result list as text: one line per boundary, the failures under it."""
    lines = []
    for r in results:
        lines.append("%s %s (%s)" % ("PASS" if r.passed else "FAIL", r.name, Path(r.file).name))
        for b in r.boundaries:
            lines.append("  b%d %-8s out %s fired %s%s%s" % (
                b.b, b.status, b.outs or "[]", [f["rule"] for f in b.fired] or "[]",
                " held" if any(i["held"] for i in b.intents) else "",
                " holding %s" % b.holding if b.holding else ""))
            lines += ["    ! " + f for f in b.failures]
    return "\n".join(lines)
