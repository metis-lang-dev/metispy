"""runtime-wire step 5 scaffolding (metis-agent DESIGN.md "Evaluation"):
the seeded injected-surprise episodes, the episode host, and the scorer.

    python -m metis.domains.kitchen.eval episodes --seed 1 --n 30 -o ep.jsonl
    python -m metis.domains.kitchen.eval run --policy oracle ep.jsonl > o.jsonl
    python -m metis.domains.kitchen.eval score o.jsonl [more.jsonl ...]
           [--harness wakes.jsonl]

An EPISODE is one short kitchen run with at most one surprise kind
(transient, wrong_domain, novel, partner, transient_twice) or none
(clean): a fresh runtime process per episode, so the runtime's
same-surprise memory starts empty (escalate keeps a session observe-only,
so one long run cannot mix kinds; the hour run is make_hour). The
harness serves an episode with

    METIS_KITCHEN_EPISODE='<one episode line>' \\
        metis --json --host metis.domains.kitchen.host:make_episode \\
        metis/domains/kitchen/catalogs/tick/kitchen_host.llp
    {"op":"watch","arg":"every=0.5 beats=<beats> operator=<operator>
                         scope=<scope> [arm=blind]"}
    ... triage ...
    {"op":"ledger"}

and writes one line per episode: {"episode": id, "arm": "A"|"B"|"C",
"ledger": <the ledger answer>}. Arms B and C differ only in the
harness's context (the runtime is the same, checked); arm A runs
`arm=blind` (no surprise check, every intent applied, the runtime's
verdict kept as `shadow`).

The scorer reads those lines (+ an optional harness log of wakes and
tokens: {"episode", "arm", "wakes", "tokens"} per episode) and prints,
per arm: triage accuracy by kind, false resumes, illegal actions,
refusals, task success (soups served / orders), wakes/hour and
tokens/hour, plus the always-escalate baseline and DESIGN.md's stop
conditions."""

import argparse
import io
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from .host import BEATS_PER_MIN, KINDS, KitchenHost, entry

# the tick form (catalogs/tick/: one action per world turn): the window the surprise taxonomy is defined for (copy
# semantics, metis-lang spec/17 amendment 4; metis/loop/watch.py)
SCENE = Path(__file__).parent / "catalogs" / "tick" / "kitchen_host.llp"
HOST = "metis.domains.kitchen.host:make_episode"

# the intended verdict per record of each kind, in order
EXPECTED = {"clean": [], "transient": ["reset"],
            "wrong_domain": ["explain"], "novel": ["escalate"],
            "partner": ["explain", "escalate"],
            "transient_twice": ["reset", "escalate"]}
# same_surprise_scope=order (the default): each partner onion lands on a
# new order — the bound forgot the first explain — so both are explained
EXPECTED_BY_SCOPE = {"order": {**EXPECTED,
                               "partner": ["explain", "explain"]},
                     "session": EXPECTED}
RESUMES = ("reset", "explain")
OPERATOR = 20     # beats until the operator resumes an escalation (the
                  # harness runs `watch ... operator=<episode operator>`)


# -- episodes -----------------------------------------------------------------

def episodes(seed: int, n: int, kinds=("clean",) + KINDS,
             scope: str = "order") -> List[dict]:
    """n seeded episodes, kinds balanced (round robin, shuffled): 3-4
    orders 40-70 beats apart; the surprise arms after a random order
    and fires on the first or second matching intent from there."""
    rng = random.Random(seed)
    order = [kinds[i % len(kinds)] for i in range(n)]
    rng.shuffle(order)
    out = []
    for i, kind in enumerate(order):
        orders, b = [], 0
        for _ in range(rng.randint(3, 4)):
            orders.append(b)
            b += rng.randint(40, 70)
        plan = []
        if kind != "clean":
            at = orders[rng.randint(0, 1)]
            skip = rng.randint(0, 1) if kind in (
                "transient", "transient_twice") else 0
            plan = [entry(kind, after=at, skip=skip)]
        out.append({"id": f"s{seed}-{i:03d}", "kind": kind, "seed": seed,
                    "orders": orders, "beats": b + 40, "plan": plan,
                    "operator": OPERATOR, "scope": scope})
    return out


def make_episode(case):
    """The host for one episode: METIS_KITCHEN_EPISODE is its JSON line."""
    ep = json.loads(os.environ["METIS_KITCHEN_EPISODE"])
    return KitchenHost(case, ep["plan"], ep["orders"], beats=ep["beats"], turns=True)


# -- in-process runs (the runtime-only baselines; and the scorer's check) ----

def _policy_oracle(s):
    d = s["diff"]
    if d == {"pot_onion": 1}:
        return "reset"
    if d == {"pot_onion": 1, "pot_space": -1}:
        return "explain kitchen_shared partner_load"
    return "escalate"


def _policy_escalate(s):
    return "escalate"


POLICIES = {"oracle": _policy_oracle, "escalate": _policy_escalate}


def run_episode(ep: dict, policy: str = "oracle", blind: bool = False,
                arm: str = "") -> dict:
    """One episode in-process with a fixed (non-LLM) policy: on each
    surprise (blind: each first inconsistent reading) the policy's verb;
    a refusal is followed by an escalate (the triage's bound)."""
    from metis.lang.repl import Repl
    from metis.loop.watch import Watch, ledger
    from metis.loop.wire import dispatch
    os.environ["METIS_KITCHEN_EPISODE"] = json.dumps(ep)
    sh = Repl(SCENE, plugin=HOST, host=True, stdout=io.StringIO())
    w = Watch(sh, beats=ep["beats"], blind=blind,
              operator=ep.get("operator"),
              scope=ep.get("scope", "order"))
    w.every = 0.5
    pick = POLICIES[policy]
    wakes = 0
    while not w.done:
        for s in w.step():
            wake = (s["signal"] == "surprise" or blind and
                    s.get("kind") == "reading" and not s["consistent"]
                    and s.get("record") == w.open and s.get("observed"))
            if not wake:
                continue
            wakes += 1
            r, verb = s["record"], pick(s)
            text = "" if verb != "escalate" else " the reading is wrong"
            dispatch(sh, "intent", f"{verb.split()[0]} {r}"
                     f"{' ' + ' '.join(verb.split()[1:]) if ' ' in verb else ''}"
                     f"{text}")
            if w.outbox and w.outbox[-1]["status"] == "refused":
                dispatch(sh, "intent", f"escalate {r} refused, escalated")
            w.outbox.clear()
    return {"episode": ep["id"], "arm": arm or
            (f"{policy}-blind" if blind else policy),
            "ledger": {"ok": True, **ledger(sh, w)},
            "harness": {"wakes": wakes, "tokens": 0}}


# -- the scorer ------------------------------------------------------------------

def _verdict(rec: dict) -> str:
    """What the triage did with one record, as the runtime saw it."""
    res = rec.get("resolution") or {}
    by = res.get("by")
    if by in RESUMES:
        return by
    if rec.get("escalated") or by == "operator":
        return "escalate"
    return "none"


def score_episode(ep_kind: str, line: dict, scope: str = "order") -> dict:
    led = line["ledger"]
    recs = led.get("records", [])
    got = [_verdict(r) for r in recs]
    expected = EXPECTED_BY_SCOPE[scope]
    want = list(expected[ep_kind])
    # a surprise the triage resumed wrongly comes back (or the world
    # stays wrong): extra records are judged as further repeats
    want += ["escalate"] * max(0, len(got) - len(want))
    false_resumes = sum(1 for g, x in zip(got, want)
                        if g in RESUMES and g != x)
    results = led.get("results", [])
    illegal = sum(1 for x in results if x.get("status") == "applied"
                  and x.get("shadow") == "refused")
    refused = sum(1 for x in results if x.get("status") == "refused")
    final = led.get("final", {})
    sched = (led.get("host") or {}).get("schedule") or {}
    w = led.get("watch", {})
    hours = (w.get("beat", 0) + final.get("fired", 0)) * 0.5 / 3600
    return {"kind": ep_kind, "correct": got == want[:len(got)]
            and len(got) >= len(expected[ep_kind]),
            "got": got, "want": want, "false_resumes": false_resumes,
            "diverged": bool(final.get("diff")) and w.get("surprise") is None,
            "illegal": illegal, "refused": refused,
            "served": final.get("served_host", 0),
            "orders": len(sched.get("orders", [])), "hours": hours,
            "wakes": (line.get("harness") or {}).get("wakes", 0),
            "tokens": (line.get("harness") or {}).get("tokens", 0)}


def always_escalate(kinds: List[str], scope: str = "order") \
        -> Dict[str, float]:
    """The baseline: escalate every surprise — right only where escalate
    is the whole answer (novel; clean has nothing to triage)."""
    by = defaultdict(list)
    for k in kinds:
        by[k].append(EXPECTED_BY_SCOPE[scope][k] in ([], ["escalate"]))
    return {k: sum(v) / len(v) for k, v in by.items()}


def score(lines: List[dict], eps: Dict[str, dict],
          harness: Dict[tuple, dict] = None) -> dict:
    harness = harness or {}
    arms = defaultdict(list)
    for ln in lines:
        h = harness.get((ln["episode"], ln["arm"]))
        if h:
            ln = {**ln, "harness": h}
        ep = eps[ln["episode"]]
        arms[ln["arm"]].append(score_episode(ep["kind"], ln,
                                             ep.get("scope", "order")))
    table = {}
    for arm, rows in arms.items():
        by = defaultdict(list)
        for r in rows:
            by[r["kind"]].append(r["correct"])
        hours = sum(r["hours"] for r in rows) or 1e-9
        surprised = [r for r in rows if r["kind"] != "clean"]
        table[arm] = {
            "episodes": len(rows),
            "triage": {k: sum(v) / len(v) for k, v in sorted(by.items())},
            "triage_all": (sum(r["correct"] for r in surprised)
                           / max(1, len(surprised))),
            "false_resumes": sum(r["false_resumes"] for r in rows),
            "diverged": sum(r["diverged"] for r in rows),
            "illegal": sum(r["illegal"] for r in rows),
            "refused": sum(r["refused"] for r in rows),
            "task": (sum(r["served"] for r in rows)
                     / max(1, sum(r["orders"] for r in rows))),
            "wakes_per_hour": sum(r["wakes"] for r in rows) / hours,
            "tokens_per_hour": sum(r["tokens"] for r in rows) / hours,
            "misses": [(r["kind"], r["got"], r["want"]) for r in rows
                       if not r["correct"]]}
    kinds = [e["kind"] for e in eps.values()]
    scopes = {e.get("scope", "order") for e in eps.values()}
    base = always_escalate(kinds, scopes.pop() if len(scopes) == 1
                           else "order")
    sur = [k for k in kinds if k != "clean"]
    base_all = sum(base[k] for k in sur) / max(1, len(sur))
    out = {"arms": table, "always_escalate": {"triage": base,
                                              "triage_all": base_all}}
    stops = []
    t = {a: v["triage_all"] for a, v in table.items()}
    if "A" in t and "B" in t:
        stops.append(f"B<=A (premise fails): {t['B'] <= t['A']}")
    if "B" in t and "C" in t:
        stops.append(f"C~B (drop the journal; |C-B|<=0.05): "
                     f"{abs(t['C'] - t['B']) <= 0.05}")
    for a in ("B", "C"):
        if a in t:
            stops.append(f"{a} triage <= always-escalate (narrator only): "
                         f"{t[a] <= base_all}")
    out["stop_conditions"] = stops
    return out


def render(res: dict) -> str:
    kinds = sorted({k for v in res["arms"].values() for k in v["triage"]})
    head = ["arm", "n"] + kinds + ["triage", "false resumes", "diverged",
                                   "illegal", "refused", "task",
                                   "wakes/h", "tokens/h"]
    rows = ["| " + " | ".join(head) + " |",
            "|" + "---|" * len(head)]
    for arm, v in sorted(res["arms"].items()):
        rows.append("| " + " | ".join(
            [arm, str(v["episodes"])]
            + [f"{v['triage'].get(k, float('nan')):.2f}" for k in kinds]
            + [f"{v['triage_all']:.2f}", str(v["false_resumes"]),
               str(v["diverged"]), str(v["illegal"]), str(v["refused"]),
               f"{v['task']:.2f}", f"{v['wakes_per_hour']:.0f}",
               f"{v['tokens_per_hour']:.0f}"]) + " |")
    b = res["always_escalate"]
    rows.append("| always-escalate | – | " + " | ".join(
        f"{b['triage'].get(k, float('nan')):.2f}" for k in kinds)
        + f" | {b['triage_all']:.2f} | – | – | – | – | – | – | – |")
    out = "\n".join(rows)
    if res["stop_conditions"]:
        out += "\n\nstop conditions:\n" + "\n".join(
            "  " + s for s in res["stop_conditions"])
    for arm, v in sorted(res["arms"].items()):
        for m in v["misses"][:10]:
            out += f"\n  {arm} miss: {m[0]} got {m[1]} want {m[2]}"
    return out


def _read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()
            if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="kitchen-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episodes")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--n", type=int, default=30)
    e.add_argument("-o", "--out")
    e.add_argument("--scope", choices=("order", "session"),
                   default="order")
    r = sub.add_parser("run")
    r.add_argument("episodes")
    r.add_argument("--policy", choices=sorted(POLICIES), default="oracle")
    r.add_argument("--blind", action="store_true")
    r.add_argument("--arm", default="")
    s = sub.add_parser("score")
    s.add_argument("runs", nargs="+")
    s.add_argument("--episodes", required=True)
    s.add_argument("--harness")
    s.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "episodes":
        text = "".join(json.dumps(x) + "\n"
                       for x in episodes(a.seed, a.n, scope=a.scope))
        if a.out:
            Path(a.out).write_text(text)
        else:
            sys.stdout.write(text)
        return 0
    if a.cmd == "run":
        for ep in _read_jsonl(a.episodes):
            print(json.dumps(run_episode(ep, a.policy, a.blind, a.arm)),
                  flush=True)
        return 0
    eps = {x["id"]: x for x in _read_jsonl(a.episodes)}
    lines = [ln for p in a.runs for ln in _read_jsonl(p)]
    harness = ({(h["episode"], h["arm"]): h
                for h in _read_jsonl(a.harness)} if a.harness else None)
    res = score(lines, eps, harness)
    print(json.dumps(res, indent=1) if a.json else render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
