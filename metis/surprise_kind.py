"""metis.surprise_kind — a surprise record's KIND (metis-lang spec/20 §4, §8; surprise-kinds s-7a02c2ac step 1): the
replay-time triage over a trace's three columns, truth, facts and law, and the one-guard-away listing the law column
needs. Nothing here changes a runtime: it reads a trace and writes two fields of a record.

    one_guard_away(trace, member, k0, k1, program)   per ground rule of the member's current stage, at each probe of
                                                     the window's beats: the rules every premise of which holds but one
                                                     → [{rule, ground, missing, stage, ticks, fired_at}]
    columns(trace, member, window, program=None)     {truth, facts, law} of the window, each None when the trace lacks it
    triage(record, trace_dir, program=None, hosts=()) spec 20 §4 on the columns → {kind, evidence, response, also}

    python -m metis.trace one-guard-away TRACE MEMBER K0 K1 --program P [--json]
    python -m metis.trace kind RECORD [--root WS] [--program P] [--host TRACE ...] [--write]

THE TRUTH COLUMN (the hosts' contract; spec 20 §10.2, written by the world, read here): TRACE/truth.jsonl, one JSON
object a line.
    {"truth": 1, "cannot": ["sidewalk", ...]}        optional first row: the words of things the world cannot hold
                                                     (a world capability gap names one of them: §4 step 1)
    {"k": K, "ego": {"lane": ..., "kind": ..., "contact": [ids]},
     "things": [{"id": ..., "class": ..., "as": TYPE | null, "role": CONST | null,
                 "lane": ..., "s": ..., "v": ..., <descriptive attributes: kind, direction, priority ...>}]}
    `as`: the member signature's TYPE the host projects this class onto ("obj" for a mover), null when it has none;
    `role`: the constant the host bound the thing to at this tick, null when unbound (unseen, out of range, passive).
    The bodies within the run's sensor range, and the ego's own lane, every tick the host snapshots.

THE PROCEDURE (§4, the first step that fires names the kind; the later ones that also fire are `also`):
    0 confusion      the record carries a person's mark: agreed is a closed mark; disagreed runs 1-6 on the words
    1 world          a word of the record is one the truth's `cannot` lists
    2 ontological    a word of the record names a value the truth holds (a thing's class or attribute) that no
                     symbol of the member's signature (a type, a constant, a pred) can hold
      perception     a named thing is in the truth and in no fact at any tick of the window, and its `as` type is in
                     the signature: the host failed to project it (a range, an occluder, a relevance rule)
    3 probabilistic  a named thing's facts flip inside the window with no truth event between the flips; or the
                     truth has the ego in contact with a thing whose role's response the facts left at none (a bound
                     crossed); or a second host's trace of the same card fired other commands in the window
    4 structural     the named things are in the facts, stable, and no command rule fired in the window, or the
                     one-guard-away listing is not empty
    5 computational  a beat in the window ended with rules still enabled (the step cap), or the record names a budget
    6 evidential     none of the above could be read: the evidence names the columns that are missing
`words` are the record's words (a confusion record's: the person's), matched as lower-case tokens against the truth's
ids, classes and string attributes; no model reads them.
"""
import json
import re
from pathlib import Path

KINDS = ("ontological", "perception", "structural", "probabilistic", "computational", "world", "evidential", "confusion")
TRUTH = "truth.jsonl"
RESPONSE = {       # spec 20 §5, the row each kind selects
    "ontological": "new vocabulary: a pred with its namespace and host contract, by hand with the architect's word; then "
                   "the clauses over it through the judge",
    "perception": "a host parameter or a host fix (sensor range, occluders, the relevance of standing movers), by hand "
                  "with the architect's word; both hosts move together and the tables rerun",
    "structural": "a pack: written by the agent through the judge when obvious, through the paid loop when not",
    "probabilistic": "a port value: a weight, a fold bound, a belief or sensor parameter; never a rule; a safety bound "
                     "is a person's word",
    "computational": "a form, a role count, a horizon or the budget itself (spec 14), the architect's",
    "world": "a world capability, recorded; no proposal",
    "evidential": "no proposal: the triage asks for the columns it lacks",
    "confusion": "the fragment kept with the person's words and the narration they rejected; then the row of the kind "
                 "the triage finds on the words, or a narration fix",
}
LON = "resp_lon"                # the ego's longitudinal response, resp_lon(R) on the road: R = none is no brake asked
LAT = ("resp_lat", "resp_side")  # the lateral ones, told beside it
NONE = "none"
TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")      # a word: a letter first, three characters at least (a lane's 2 or a speed's 25 names nothing)


def _tokens(text):
    return set(TOKEN.findall(str(text or "").lower()))


def _args(atom):
    """The arguments of a ground atom 'p(a,b)': ['a', 'b']; [] for a 0-ary atom."""
    if "(" not in atom:
        return []
    return [x.strip() for x in atom[atom.index("(") + 1: atom.rindex(")")].split(",") if x.strip()]


def _pred(atom):
    return atom.split("(", 1)[0]


# ── the law column: the one-guard-away listing ──
def one_guard_away(path, member, k0, k1, program, check_key=True):
    """MEMBER replayed over the trace at PATH on PROGRAM; at every probe of a beat in [K0, K1] (before each rule it
    fires, and at its end) each ground clause of the session's current stage that needs exactly one atom more than the
    state holds. → rows {rule, ground, missing, stage, ticks, fired_at}: one per (ground rule, missing atom), the ticks
    it was one guard away, and the ticks of the window it fired at (a rule may be one away early in a beat and fire later)."""
    from metis import trace as T
    rows, fired_at = {}, {}

    def probe(k, sh):
        if not (k0 <= k <= k1):
            return
        for kind, stage, ev in sh.events_tbl:
            if kind != "clause" or stage != sh.stage:
                continue
            need = dict(ev.persist)
            for a, n in ev.consume.items():
                need[a] = need.get(a, 0) + n
            short = [a for a, n in need.items() if sh.counts.get(a, 0) < n]
            if len(short) == 1:
                r = rows.setdefault((ev.name, short[0]), {"rule": T.rule_base(ev.name), "ground": ev.name, "missing": short[0],
                                                          "stage": stage, "ticks": [], "have": len(need) - 1})
                if not r["ticks"] or r["ticks"][-1] != k:
                    r["ticks"].append(k)

    segs = T.read(path)
    r = T.Replay(member)
    for seg in segs:
        last = seg.ticks[-1]["k"] if seg.ticks else seg.k0
        if last < k0:
            continue
        if seg.k0 > k1:
            break
        T.replay_segment(seg, member, program, r, check_key, probe=probe)
        if r.divergence:
            raise T.TraceError("the replay left the record at tick %s (%s): the program is not the trace's"
                               % (r.divergence.get("k"), r.divergence.get("what")))
        for k, names in r.fired.items():
            if k0 <= k <= k1:
                for n in names:
                    fired_at.setdefault(n, set()).add(k)
    out = []
    for (name, _), row in sorted(rows.items()):
        row["fired_at"] = sorted(fired_at.get(name, ()))
        out.append(row)
    return out


# ── the three columns ──
def read_truth(path):
    """TRACE/truth.jsonl as ({k: row}, header) — (None, None) when the trace has no truth column."""
    f = Path(path) / TRUTH
    if not f.is_file():
        return None, None
    rows, header = {}, {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "k" in r:
            rows[int(r["k"])] = r
        else:
            header.update(r)
    return rows, header


def _fired_in(path, member, k0, k1):
    """The rules MEMBER fired at each tick of [K0, K1], from the trace's rows alone (T8's reading, no program), and
    whether a beat in the window ended with rules still enabled (`left`: the step cap)."""
    from metis import trace as T
    fired, left = {}, []
    for seg in T.read(path):
        for t in seg.ticks:
            k = t["k"]
            if not (k0 <= k <= k1):
                continue
            e = (t.get("members") or {}).get(member) or {}
            for item in e.get("seq") or ():
                names = list(item.get("fired") or ()) + list(item.get("step") or ())
                if names:
                    fired.setdefault(k, []).extend(names)
                if item.get("left"):
                    left.append(k)
    return fired, left


def columns(path, member, window, program=None, check_key=True):
    """The window's three columns: truth {k: row} (None without truth.jsonl), facts {k: {atom: n}} (the member's
    state, timeline's reconstruction), law {fired, left, one_guard_away (None without PROGRAM), commands}."""
    from metis import trace as T
    k0, k1 = window
    truth, header = read_truth(path)
    try:
        empty = not T.read(path)
    except T.TraceError:
        empty = True
    if empty:                                        # the trace holds no segment (overwritten, or never written)
        return {"truth": {k: r for k, r in (truth or {}).items() if k0 <= k <= k1} if truth is not None else None,
                "truth_header": header or {}, "facts": None,
                "law": {"fired": {}, "left": [], "one_guard_away": None, "commands": [], "one_guard_away_error": "the trace holds no segment"}}
    if truth is not None:
        truth = {k: r for k, r in truth.items() if k0 <= k <= k1}
    ticks, _ = T.timeline(path, member, program)
    facts = {t.k: t.counts for t in ticks if k0 <= t.k <= k1}
    fired, left = _fired_in(path, member, k0, k1)
    segs = T.read(path)
    commands = set((segs[0].header or {}).get("commands") or ()) if segs else set()
    oga, why = None, None
    if program:
        try:
            oga = one_guard_away(path, member, k0, k1, program, check_key)
        except T.TraceError as e:                    # the law column is partial, the triage goes on
            why = str(e)
    return {"truth": truth, "truth_header": header or {}, "facts": facts or None,
            "law": {"fired": fired, "left": left, "one_guard_away": oga, "commands": sorted(commands),
                    **({"one_guard_away_error": why} if why else {})}}


# ── the signature ──
def signature(program):
    """The member's vocabulary: its type names, their constants and its preds (the compiled catalog's declarations,
    includes resolved). → set of lower-case symbols."""
    from metis import trace as T
    sh = T.open_session(program)
    out = set()
    for d in getattr(sh.doc.source, "decls", ()):
        name = type(d).__name__
        if name == "TypeDecl":
            out.add(d.name.lower())
            out.update(c.lower() for c in d.consts)
        elif name in ("PredDecl", "BwdDecl"):
            out.add(d.name.lower())
    return out


def signature_of_facts(facts):
    """A stand-in signature when no program is given: every pred and argument the facts ever held."""
    out = set()
    for counts in (facts or {}).values():
        for a in counts:
            out.add(_pred(a).lower())
            out.update(x.lower() for x in _args(a))
    return out


# ── the triage ──
def _named(truth, words):
    """The truth's things the words name: {id: (thing, the matching tokens, the attribute each token was found in)}."""
    toks, out = _tokens(words), {}
    for k in sorted(truth or {}):
        for th in truth[k].get("things") or ():
            hit = {}
            for key, val in th.items():
                if key in ("role", "as", "s", "v") or not isinstance(val, str):
                    continue
                for t in _tokens(val) & toks:
                    hit.setdefault(t, key)
            if hit:
                cur = out.setdefault(th["id"], [th, {}])
                cur[1].update(hit)
        ego = truth[k].get("ego") or {}
        for key, val in ego.items():
            if isinstance(val, str):
                for t in _tokens(val) & toks:
                    out.setdefault("ego:" + key, [dict(ego, id="ego:" + key, **{"class": "ego " + key}), {}])[1].setdefault(t, key)
    return out


def _thing_facts(th, facts):
    """The member's atoms that mention TH (its role at that tick, or one of its string values), per tick."""
    out = {}
    vals = {str(v).lower() for key, v in th.items() if isinstance(v, str) and key not in ("id", "as")}
    for k, counts in (facts or {}).items():
        hit = [a for a in counts if vals & {x.lower() for x in _args(a)}]
        if hit:
            out[k] = sorted(hit)
    return out


def _roles(truth, tid):
    """{k: role} of the thing TID wherever the host bound it."""
    out = {}
    for k, r in (truth or {}).items():
        for th in r.get("things") or ():
            if th.get("id") == tid and th.get("role"):
                out[k] = th["role"]
    return out


def _role_facts(roles, facts):
    out = {}
    for k, role in roles.items():
        hit = [a for a in (facts or {}).get(k, {}) if role in _args(a)]
        if hit:
            out[k] = sorted(hit)
    return out


def _flips(atoms_by_tick, ticks):
    """Atoms that come and go at least twice over TICKS (present, absent, present ...): {atom: [the ticks it changed]}."""
    seen = sorted({a for v in atoms_by_tick.values() for a in v})
    out = {}
    for a in seen:
        prev, changes = None, []
        for k in ticks:
            now = a in atoms_by_tick.get(k, ())
            if prev is not None and now != prev:
                changes.append(k)
            prev = now
        if len(changes) >= 2:
            out[a] = changes
    return out


def triage(rec, path, program=None, hosts=(), check_key=True):
    """spec 20 §4 on RECORD (a dict) over the trace at PATH. → {kind, evidence, response, also, [then, verdict]}."""
    member, window = rec.get("member"), rec.get("window")
    if not window:
        k = rec.get("k")
        window = [rec.get("k0", k), k] if k is not None else None
    mark = rec.get("mark")
    if mark is not None or rec.get("verdict"):
        verdict = rec.get("verdict")
        if verdict == "agreed":
            return {"kind": "confusion", "verdict": "agreed", "then": None, "response": "a closed mark: the narration was "
                    "accepted; nothing is recorded as a surprise", "evidence": {}, "also": []}
        if mark and mark.get("k0") is not None and mark.get("k1") is not None:
            window = [int(mark["k0"]), int(mark["k1"])]
        words = (mark or {}).get("said") or rec.get("words") or ""
        inner = _triage(member, window, words, rec, path, program, hosts, check_key)
        return {"kind": "confusion", "verdict": verdict or "disagreed", "then": inner["kind"] if inner["kind"] != "evidential" else
                "evidential", "evidence": inner["evidence"], "response": RESPONSE["confusion"] + " (here: %s)" % inner["response"],
                "also": [inner["kind"]] + inner["also"]}
    return _triage(member, window, rec.get("words") or rec.get("reading") or "", rec, path, program, hosts, check_key)


def _triage(member, window, words, rec, path, program, hosts, check_key):
    missing = []
    if not window:
        return _evidential(["window"], {})
    cols = columns(path, member, window, program, check_key)
    truth, facts, law = cols["truth"], cols["facts"], cols["law"]
    if truth is None:
        missing.append("truth")
    if facts is None:
        missing.append("facts")
    if law["one_guard_away"] is None:
        missing.append("law (one guard away: %s)" % (law.get("one_guard_away_error") or "no program"))
    sig = signature(program) if program else signature_of_facts(facts)
    toks = _tokens(words)
    ticks = list(range(window[0], window[1] + 1))
    fired_cmds = sorted({n for names in law["fired"].values() for n in names if _base(n) in law["commands"]})
    evidence = {"truth": None, "facts": None,
                "law": {"fired": {k: v for k, v in sorted(law["fired"].items())}, "commands_fired": fired_cmds,
                        "one_guard_away": law["one_guard_away"]},
                "hosts": None, "missing": missing}
    found = []                                                             # (step, kind, why) in §4's order

    # 1 world
    cannot = {str(w).lower() for w in cols["truth_header"].get("cannot") or ()}
    if cannot & toks:
        found.append(("world", "the words name %s, which the world cannot hold" % ", ".join(sorted(cannot & toks))))

    named = _named(truth, words) if truth is not None else {}
    if truth is not None:
        evidence["truth"] = [{"id": tid, "class": th.get("class"), "named_by": sorted(hit), **{key: th[key] for key in ("as", "lane", "kind") if key in th}}
                             for tid, (th, hit) in sorted(named.items())]
    # 2 ontological / perception
    if truth is not None:
        # a word found in a thing's class is held when the host says the class projects onto a type of the signature
        # (`as`); a word found in its id names the thing, never a value (cr-harvest: a world's record names its things by id);
        # a word found in any other attribute is held when the signature has it as a symbol
        unheld = sorted({t for _, (th, hit) in named.items() for t, key in hit.items()
                         if (key == "class" and not _class_held(th, sig, program)) or (key not in ("class", "id") and t not in sig)})
        if unheld:
            found.append(("ontological", "the truth holds %s and no symbol of the member's signature can hold it"
                          % ", ".join(unheld)))
        thing_facts = {}
        for tid, (th, hit) in named.items():
            f = _role_facts(_roles(truth, tid), facts) or (_thing_facts(th, facts) if any(key != "class" for key in hit.values()) else {})
            thing_facts[tid] = f
            if not f and _class_held(th, sig, program):
                found.append(("perception", "%s (%s) is in the truth and in no fact of the window, and its type %s is in "
                              "the signature: the host did not project it" % (tid, th.get("class"), th.get("as"))))
        evidence["facts"] = {tid: ({str(k): v for k, v in sorted(f.items())} if f else "none") for tid, f in thing_facts.items()}

        # 3 probabilistic
        prob = []
        for tid, f in thing_facts.items():
            fl = _flips(f, ticks)
            if fl:
                lanes = {r.get("lane") for k, r in ((k, next((t for t in truth.get(k, {}).get("things") or () if t.get("id") == tid), {})) for k in ticks) if r}
                if len(lanes) <= 1:
                    prob.append("%s's facts flip with no truth event: %s" % (tid, ", ".join("%s at %s" % (a, c) for a, c in sorted(fl.items())[:3])))
        for k in ticks:
            ego = (truth.get(k) or {}).get("ego") or {}
            hit = None
            for oid in ego.get("contact") or ():
                role = _roles(truth, oid).get(k) or _roles(truth, oid).get(k - 1)
                here = (facts or {}).get(k) or (facts or {}).get(k - 1) or {}
                lon = [a for a in here if _pred(a) == LON]
                if role and lon and all(_args(a)[-1] == NONE for a in lon):
                    lat = sorted(a for a in here if _pred(a) in LAT and _args(a)[-1] != NONE)
                    hit = "a bound crossed: at tick %d the ego is in contact with %s (%s) and the fold asked no longitudinal response%s" % (
                        k, oid, role, " (only a lateral one: %s)" % ", ".join(lat[:3]) if lat else "")
                    break
            if hit:
                prob.append(hit)
                break
        if hosts:
            other = []
            for h in hosts:
                f2, _ = _fired_in(h, member, *window)
                c2 = sorted({n for names in f2.values() for n in names if _base(n) in law["commands"]})
                other.append({"trace": str(h), "commands_fired": c2})
                if {_base(n) for n in c2} != {_base(n) for n in fired_cmds}:
                    prob.append("a second host fired other commands in the window: %s against %s" % (c2 or "none", fired_cmds or "none"))
            evidence["hosts"] = other
        for p in prob:
            found.append(("probabilistic", p))

        # 4 structural
        oga = law["one_guard_away"] or []
        if named and all(thing_facts.get(tid) for tid in named):
            if not fired_cmds:
                found.append(("structural", "the named things are in the facts and no command rule fired in the window"))
            elif oga:
                found.append(("structural", "one guard away: %s" % ", ".join(sorted({"%s (%s)" % (r["rule"], r["missing"]) for r in oga})[:6])))
    # 5 computational
    if law["left"] or rec.get("budget"):
        found.append(("computational", "a beat ended with rules still enabled at %s" % law["left"][:3] if law["left"] else "the record names a budget"))

    if not found:
        why = missing or (["words name nothing in the truth"] if truth is not None and not named else ["no step of §4 fired"])
        return _evidential(why, evidence)
    order = {k: i for i, k in enumerate(("world", "ontological", "perception", "probabilistic", "structural", "computational"))}
    found.sort(key=lambda x: order[x[0]])
    kinds = []
    for kind, _ in found:
        if kind not in kinds:
            kinds.append(kind)
    evidence["why"] = ["%s: %s" % kw for kw in found]
    return {"kind": kinds[0], "also": kinds[1:], "evidence": evidence, "response": RESPONSE[kinds[0]]}


def _class_held(th, sig, program):
    """A thing's class is held by the member's vocabulary when the host names a type for it (`as`) that the signature
    has; without the program the signature's type names are not known, and a type the host names is taken as held."""
    a = (th.get("as") or "").lower()
    return bool(a) and (a in sig or not program)


def _base(name):
    from metis.trace import rule_base
    return rule_base(name)


def _evidential(missing, evidence):
    evidence = dict(evidence or {}, missing=list(missing))
    return {"kind": "evidential", "also": [], "evidence": evidence,
            "response": RESPONSE["evidential"] + ": " + "; ".join(missing)}
