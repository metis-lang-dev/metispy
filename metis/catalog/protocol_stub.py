"""metis.catalog.protocol_stub — a member's CANCEL rules generated from its own rules (specialist-tree s-404d001f
step 3c, the architect's v8 §13.2 and rulings on the draft; spec/15 amendment 7; moved here from metis-worlds'
tools/ so a catalog package can generate its stub too): a stop is a cancel — the director's
tick__cancel(D) takes D's goal (a revoke takes the body and keeps it), and the member drops the goal itself. No
world's names in here: the member's rules and its manifest's goal kinds say everything.

The states a member can hold while it works are walked from what takes its goal:
- a member with the protocol stub's idle/busy (the tree's begin rules): from every rule that consumes idle (begin_K,
  and a member's own trigger), its state the tokens it puts out and the goal it reads;
- a member with no idle (a tree's task): from every rule that consumes a goal kind.
A state is the set of TRACKED tokens: the goal kinds and the member's private linear tokens that a rule consuming a
tracked token (or idle) puts out — a marker a rule puts out from the world alone is no state. A rule
whose tracked body the state (with the goal kinds, which may arrive at any time) covers moves it on; one giving back
idle or putting out an end ends it. The minimal states (a bigger one's cancel would race a smaller one's: the sampled
order) get one rule each, consuming all of it with the cancel:
    cancel_<tokens> [dyn] : tick__cancel(D) * busy * goto(X0) -o idle.      (one = the unit, for the task)
and a member with idle gets cancel_none, `idle * tick__cancel(D) -o idle`: the real race where the member's done and
the stop land in one director beat (path_pop and stop_cancel both enabled, the sampled order) — not for idle members
in general, who get no cancel.

    cancel_rules(text, member, goal_kinds, ends=(), decls="", idle="idle", namespace=None, world=(), bind="")
        → [line…]   (doc line + rule, in pairs)
    refresh(text, member, goal_kinds, …) → text   (a catalog source's generated block, between BEGIN and END)

It refuses (StubError, by name): a member whose goal kinds reach no state (no rule takes the goal: nothing a cancel
could drop — a wrong goal kind, or a member with no begin), and a tracked token whose arity neither the base nor
the family package declares.
"""
import re


class StubError(ValueError):
    """A member the generator cannot write a cancel stub for."""

RULE = re.compile(r"^\s+(\w+) \[dyn\] : (.*) -o (.*?)\s*(?:@w [^.]*)?\.(\s*%.*)?$")
PRED = re.compile(r"^pred (\w+)(?:\(([^)]*)\))?\s*:\s*(\w+)\s*\.")


def _atoms(side):
    out = []
    for a in side.split("*"):
        a = a.strip()
        if not a or "<>" in a or a in ("()", "one"):
            continue
        read = a.startswith("$")
        a = a.lstrip("$")
        out.append((a.split("(")[0], read))
    return out


def rules(text):
    """[(name, [(pred, read)…] body, [pred…] head)] of TEXT's [dyn] rules."""
    out = []
    for line in text.splitlines():
        m = RULE.match(line)
        if m:
            out.append((m.group(1), _atoms(m.group(2)), [p for p, _ in _atoms(m.group(3))]))
    return out


def arities(text):
    return {m.group(1): len([a for a in (m.group(2) or "").split(",") if a.strip()])
            for m in (PRED.match(l.strip()) for l in text.splitlines()) if m}


def private(text, member):
    """The preds TEXT declares in MEMBER's own namespace."""
    return {m.group(1) for m in (PRED.match(l.strip()) for l in text.splitlines()) if m and m.group(3) == member}


def states(text, member, goal_kinds, ends=(), crossing=(), idle="idle", namespace=None, world=()):
    """The sets of tracked tokens MEMBER can hold while it works, the minimal ones (sorted tuples, sorted): a rule
    that gives back IDLE or puts out an end (ENDS: the task's done, stuck) ends the work; a state that holds a smaller
    one is left out — its cancel would race the smaller's in the sampled order (one cancel, one rule).
    NAMESPACE: the base's own (default MEMBER's; a lifted base several members instantiate has its own); WORLD: the
    world's preds the member keeps a copy of in that namespace (its manifest's world-private) — the world's, never
    a member's state: a cancel drops the member's bookkeeping only."""
    rs = rules(text)
    mine = private(text, namespace or member)
    has_idle = idle in mine
    goals = set(goal_kinds)
    own = mine - {idle} - goals - set(crossing) - set(world)   # a crossing pred is no member's own state
    tracked = set(goals) | ({"busy"} if has_idle and "busy" in mine else set())
    for _, body, head in rs:                      # a token put out by a rule that consumes what is tracked (or idle)
        if any(not r and (p in tracked or p == idle) for p, r in body):
            tracked |= {p for p in head if p in own}
    for _ in range(4):                            # … and by the rules that consume those
        for _, body, head in rs:
            if any(not r and p in tracked for p, r in body):
                tracked |= {p for p in head if p in own}
    start = []
    for _, body, head in rs:
        takes = [p for p, r in body if not r]
        if (has_idle and idle in takes) or (not has_idle and set(takes) & goals):
            st = {p for p in head if p in tracked} | ({p for p, r in body if r and p in goals} if has_idle else set())
            if st:
                start.append(frozenset(st))
    seen, todo = set(), list(start)
    while todo:
        st = todo.pop()
        if st in seen:
            continue
        seen.add(st)
        for _, body, head in rs:
            takes = {p for p, r in body if not r and p in tracked}
            needs = {p for p, r in body if p in tracked} - goals
            if not takes or not needs <= st or idle in [p for p, r in body if not r]:
                continue
            if idle in head or set(head) & set(ends):   # it gives back idle, or ends: the work ends
                continue
            nxt = (st - takes) | {p for p in head if p in tracked}
            if nxt and nxt != st:
                todo.append(frozenset(nxt))
    return sorted(tuple(sorted(s)) for s in seen if not any(o < s for o in seen))


def triggers(text, idle="idle"):
    """The names of TEXT's rules that consume IDLE: a member's begin rules and its own triggers."""
    return [n for n, body, _ in rules(text) if any(p == idle and not r for p, r in body)]


def cancel_rules(text, member, goal_kinds, ends=(), decls="", idle="idle", namespace=None, world=(), bind=""):
    """MEMBER's cancel rules over TEXT (its base; DECLS: the family package it includes, for the goal kinds'
    arities): one per state, and cancel_none for a member with IDLE. A lifted base (NAMESPACE its own, MEMBER a
    variable) names its member by BIND, a read the rules bind it with (`$self(S)`); an idle token with an argument
    takes the member (`calm(S)`). A member with neither goal kinds nor a trigger gets none: nothing to cancel."""
    ar = dict(arities(decls), **arities(text))
    has_idle = idle in private(text, namespace or member)
    out = []
    sts = states(text, member, goal_kinds, ends, crossing=arities(decls), idle=idle, namespace=namespace, world=world)
    if not sts:
        if not goal_kinds and not (has_idle and triggers(text, idle)):
            return []
        raise StubError("%s: its goal kinds %s reach no state (no rule takes the goal)" % (member, ", ".join(goal_kinds) or "(none)"))
    unknown = sorted({p for st in sts for p in st if p not in ar and p != "busy"})
    if unknown:
        raise StubError("%s: no declaration for %s (the base or the family package)" % (member, ", ".join(unknown)))
    if ar.get(idle, 0) > 1:
        raise StubError("%s: its idle %s takes %d arguments (none, or the member)" % (member, idle, ar[idle]))
    back = "%s(%s)" % (idle, member) if ar.get(idle) else idle
    me = "tick__cancel(%s)%s" % (member, " * " + bind if bind else "")
    for st in sts:
        toks, n = [], 0
        for p in st:
            k = ar.get(p, 0)
            toks.append("%s(%s)" % (p, ", ".join("V%d" % (n + i) for i in range(k))) if k else p)
            n += k
        name = "cancel_" + "_".join(p for p in st if p != "busy") if len(st) > 1 or not has_idle else "cancel"
        out += ["  %% Stopped: I drop my goal.",
                "  %s [dyn] : %s * %s -o %s." % (name, me, " * ".join(toks), back if has_idle else "one")]
    if has_idle:
        out += ["  %% Stopped as I finished: nothing left to drop.",
                "  cancel_none [dyn] : %s * %s -o %s.   "
                "%% the race: its done and the stop in one director beat (path_pop, stop_cancel: the sampled order)" % (back, me, back)]
    return out


BEGIN = "  % >>> generated by metis.catalog.protocol_stub: the stop's cancels (regenerate, do not edit)"
END = "  % <<< generated"


def refresh(text, member, goal_kinds=(), **kw):
    """TEXT with its generated block (BEGIN … END) written again from the rest of TEXT, after its last trigger (a
    rule consuming the idle token): for a source file kept in a catalog, which no installer regenerates. → text."""
    lines = text.split("\n")
    if BEGIN in lines:
        i = lines.index(BEGIN)
        lines = lines[:i] + lines[lines.index(END, i) + 1:]
    base = "\n".join(lines)
    cancels = cancel_rules(base, member, list(goal_kinds), **kw)
    if not cancels:
        return base
    last = triggers(base, kw.get("idle", "idle"))
    if not last:
        raise StubError("%s: no trigger to write the cancels after" % member)
    at = max(k for k, m in enumerate(map(RULE.match, lines)) if m and m.group(1) in last)
    return "\n".join(lines[:at + 1] + [BEGIN] + cancels + [END] + lines[at + 1:])


def main(argv=None):
    """python -m metis.catalog.protocol_stub FILE --member S [--idle calm] [--namespace cook] [--bind '$self(S)']
    [--manifest domain.yaml] [--goal K …] [--write | --check]: FILE's generated cancels, printed, written in place,
    or checked (exit 1 when FILE is not as its generator writes it). --manifest: its world-private preds are the
    world's (no member state)."""
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="python -m metis.catalog.protocol_stub")
    ap.add_argument("file")
    ap.add_argument("--member", required=True)
    ap.add_argument("--idle", default="idle")
    ap.add_argument("--namespace")
    ap.add_argument("--bind", default="")
    ap.add_argument("--manifest")
    ap.add_argument("--goal", action="append", default=[])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = open(a.file).read()
    out = refresh(text, a.member, a.goal, idle=a.idle, namespace=a.namespace, bind=a.bind,
                  world=world_private(a.manifest) if a.manifest else ())
    if a.write:
        open(a.file, "w").write(out)
    elif a.check:
        if out != text:
            print("%s: not as protocol_stub writes it (run with --write)" % a.file, file=sys.stderr)
            return 1
    else:
        print(out)
    return 0


def world_private(manifest):
    """The preds a manifest's `shared` marks world-private: the world's, kept in the member's copy."""
    import yaml
    shared = (yaml.safe_load(open(manifest)) or {}).get("shared") or {}
    return sorted(p for p, d in shared.items() if d == "world-private")


if __name__ == "__main__":
    raise SystemExit(main())
