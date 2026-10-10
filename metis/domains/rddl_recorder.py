"""A generic pyRDDLGym episode recorder and surprise sink (the RDDL
formalization experiment, metis-loop docs/EXPERIMENT-RDDL.md step 2).

Pointing it at another domain is CONFIG, not code: a problem id from
rddlrepository, an instance, an agent ("random" | "noop" | a callable
obs -> action dict), and paths.

Two passes, kept apart on purpose:

1. `record_episodes` runs REAL environment episodes (pyRDDLGym
   RDDLEnv; its own `log_path` — and `debug_path` for one episode — sit
   beside ours) and writes the RAW-TRANSITION JSONL: per step the true
   boolean state fluents, the actions taken, the reward, the next
   state, stamped with domain/instance/episode/seed/t. That file is the
   surprise-sink input, the notching dataset, and provenance at once.
   Nothing about any catalog is assumed here.
2. `surprise_sink` replays each transition against a serving catalog
   and writes metis surprise records (kernel `surprise_record`) where
   the likelihood says surprise. The mapping is MECHANICAL: a grounded
   fluent `elevator-at-floor___e0__f2` is the atom
   `elevator_at_floor(e0,f2)` (hyphens -> underscores, `___` opens the
   argument list, `__` separates arguments); actions taken are atoms of
   the same shape added to the transition's initial state (a rule may
   read them: `$move_current_dir(E)`). The OBSERVED vocabulary is every
   grounded boolean state fluent: present -> 1, absent -> 0.

The horizon per transition (`steps`) is a knob, default 1: one tick
read as one metis step — the naive mapping, deliberately NOT the
synchronous-tick idiom (the experiment records whether the loop
discovers it). `steps="frame"` reads the transition when the catalog
SETTLES (nothing can fire; ForwardProjection to a cap, settled mass
required) — the readout iteration 1's handshake asked for.

Two more generic, catalog-agnostic options (iteration 1's findings):
`negatives` maps a FALSE boolean fluent to `not_<name>(args)` in both
the initial state and the observed vocabulary (linear logic cannot
test absence — the formalizer invented complement atoms for exactly
this); `tick_token` adds one `tick` atom to every transition's initial
state (a resource a catalog may use to structure a step, or ignore).

Pricing tier (the Wildfire experiment's finding: a lockstep catalog
that walks every cell interleaves its per-cell work, and the exact
filter's intermediate states multiply past any budget): `tier`
"exact" (the default — the kernel filter, as before), "sampled"
(Monte Carlo: `samples` reference-sampler runs from the transition's
state, likelihood = the fraction whose readout matches the
observation, with a 95% Wilson half-width), or "auto" (exact, falling
back to sampled when the filter refuses its budget). Every record says
which tier priced it. A sampled zero is "no sampled run matched" — the
record carries n and ci so a reader can tell it from a proof.
`frame_cap` bounds the settle readout (a step of a big lockstep world
can take more than the old fixed 64 events).

Weight ports (PLUGIN-PORTS.md): `plugin` (a port-plugin spec, resolved
beside the catalog) binds the catalog's ports; `overlay` (a notch table
— {"rows": [{port, args, row, notched}]}, as notch writes it) rides on
top, newest first. Both tiers price with them; a catalog with an
unbound port is refused up front (the BT.cpp rule)."""
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union


def atom_of(key: str) -> str:
    """`elevator-at-floor___e0__f2` -> `elevator_at_floor(e0,f2)`."""
    name, _, args = key.partition("___")
    name = name.replace("-", "_")
    if not args:
        return name
    return f"{name}({','.join(a.replace('-', '_') for a in args.split('__'))})"


def _is_bool(v) -> bool:
    """Python bool or a numpy boolean (`numpy.bool_`; numpy 2 names it
    `numpy.bool`)."""
    return type(v).__name__ in ("bool", "bool_")


def _true_bools(d: Dict) -> List[str]:
    return sorted(k for k, v in d.items() if _is_bool(v) and bool(v))


def load_problem(problem: str, instance: str, **env_kwargs):
    import pyRDDLGym
    from rddlrepository.core.manager import RDDLRepoManager
    prob = RDDLRepoManager().get_problem(problem)
    env = pyRDDLGym.make(prob.get_domain(), prob.get_instance(instance),
                         **env_kwargs)
    return prob, env


def non_fluents(env) -> Dict[str, object]:
    m = env.model
    return {k: (v.item() if hasattr(v, "item") else v)
            for k, v in m.ground_vars_with_values(m.non_fluents).items()}


def state_vocabulary(env) -> List[str]:
    """Every grounded BOOLEAN state fluent (the observed vocabulary)."""
    s, _ = env.reset(seed=0)
    return sorted(k for k, v in s.items() if _is_bool(v))


def _agent(env, agent, seed):
    if callable(agent):
        return agent
    if agent == "noop":
        return lambda obs: {}
    if agent == "random":
        from pyRDDLGym.core.policy import RandomAgent
        ra = RandomAgent(action_space=env.action_space,
                         num_actions=env.max_allowed_actions, seed=seed)
        return lambda obs: ra.sample_action(obs)
    raise ValueError(f"agent: 'random' | 'noop' | callable, not {agent!r}")


def record_episodes(problem: str, instance: str, episodes: int,
                    out_jsonl: Union[str, Path], seed0: int = 1,
                    agent="random", log_dir: Optional[str] = None,
                    debug_first: bool = True) -> Dict:
    """Run REAL episodes; append raw transitions to `out_jsonl`.
    Returns a summary {episodes, steps, total_reward, path, logs}."""
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logs = []
    n_steps, total = 0, 0.0
    with out_jsonl.open("a") as f:
        for ep in range(episodes):
            seed = seed0 + ep
            kw = {}
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                kw["log_path"] = os.path.join(
                    log_dir, f"{problem}_i{instance}_ep{seed}")
                if debug_first and ep == 0:
                    kw["debug_path"] = os.path.join(
                        log_dir, f"{problem}_i{instance}_debug")
                logs.append(kw["log_path"])
            _, env = load_problem(problem, instance, **kw)
            act = _agent(env, agent, seed)
            obs, _ = env.reset(seed=seed)
            for t in range(env.horizon):
                a = act(obs)
                nxt, r, term, trunc, _ = env.step(a)
                f.write(json.dumps({
                    "domain": problem, "instance": instance,
                    "episode": ep, "seed": seed, "t": t,
                    "agent": agent if isinstance(agent, str) else "custom",
                    "state": _true_bools(obs),
                    "action": sorted(k for k, v in a.items() if v),
                    "reward": float(r),
                    "next": _true_bools(nxt)}) + "\n")
                total += float(r)
                n_steps += 1
                obs = nxt
                if term or trunc:
                    break
            env.close()
    return {"episodes": episodes, "steps": n_steps,
            "total_reward": total, "path": str(out_jsonl), "logs": logs}


def read_transitions(path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def neg_of(key: str) -> str:
    return "not_" + atom_of(key)


FRAME_CAP = 64


def _overlay_table(path):
    """A notch table file -> {port: fn(args, counts)} overlays keyed by
    (args, read-scope presence row)."""
    rows = json.loads(Path(path).read_text())["rows"]
    tables: Dict[str, Dict] = {}
    for r in rows:
        key = (tuple(r["args"]), tuple(tuple(x) for x in r["row"]))
        tables.setdefault(r["port"], {})[key] = float(r["notched"])

    def make(t):
        def ov(args, counts):
            for (a, row), v in t.items():
                if a == tuple(args) and all(
                        (counts.get(x, 0) > 0) == bool(b) for x, b in row):
                    return v
            return None
        return ov
    return {f: make(t) for f, t in tables.items()}


def port_pricer(doc, plugin=None, base_dir=None, overlay=None):
    """The catalog's PortPricer from a plugin spec (+ a notch overlay),
    or None for a catalog without weight ports."""
    if not doc.ports.weight_refs:
        return None
    from metis.fw.registry import Registry
    from metis.lang import PortPricer
    reg = Registry()
    if plugin:
        from metis.fw.plugin import load_plugin
        load_plugin(plugin, {}, base_dir).bind(reg)
    for f, ov in (_overlay_table(overlay).items() if overlay else ()):
        reg.overlay(f, ov)
    return PortPricer(doc.ports, reg.weight_bindings(),
                      horn=doc.catalog.horn_db())


def _sampled_likelihood(prog, horizon, frame, observed, samples, seed,
                        pricer=None):
    """(likelihood, half-width, unsettled runs): reference-sampler runs
    from the program's initial state; a run matches when every observed
    atom's presence equals the observation at the readout."""
    import math
    from metis.kernel.filter import DONE
    from metis.kernel.refsample import run_trace
    hits = unsettled = 0
    pol = pricer.policy(prog, canonical=True) if pricer else None
    for k in range(samples):
        _, (stage, kappa) = run_trace(prog, horizon, seed + k, policy=pol)
        if frame and stage != DONE:
            unsettled += 1
            continue
        c = dict(kappa)
        hits += all((c.get(a, 0) > 0) == bool(v)
                    for a, v in observed.items())
    n = samples
    p = hits / n
    z = 1.96
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            / (1 + z * z / n))
    return p, half, unsettled


def surprise_sink(catalog_path, transitions: Iterable[dict],
                  vocabulary: List[str], out_jsonl, steps=1,
                  rare: float = 0.05, max_states: int = 200_000,
                  with_actions: bool = True, negatives: bool = False,
                  tick_token: bool = False, tier: str = "exact",
                  samples: int = 2000, frame_cap: int = FRAME_CAP,
                  seed: int = 1, plugin: Optional[str] = None,
                  overlay=None) -> Dict:
    """Replay each transition against the catalog; write surprise
    records (kind unmodeled|rare) to `out_jsonl` (overwritten). Returns
    {transitions, surprises, per_episode {ep: n}, kinds, tiers}."""
    from metis.kernel.filter import StateBudgetExceeded
    from metis.kernel.nets import program_key
    from metis.kernel.record import surprise_record
    if tier not in ("exact", "sampled", "auto"):
        raise ValueError(f"tier: exact | sampled | auto, not {tier!r}")
    from metis.lang import compile_catalog, parse, resolve_includes
    p = Path(catalog_path)
    doc = compile_catalog(resolve_includes(parse(p.read_text()),
                                           p.parent))
    pricer = port_pricer(doc, plugin, p.parent, overlay)
    vocab_atoms = [atom_of(k) for k in vocabulary]
    if negatives:
        vocab_atoms += [neg_of(k) for k in vocabulary]
    frame = steps == "frame"
    horizon = int(frame_cap) if frame else int(steps)
    tiers: Dict[str, int] = {}
    per_ep: Dict[int, int] = {}
    kinds: Dict[str, int] = {}
    unsettled = 0
    n = 0
    Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(out_jsonl, "w") as f:
        for tr in transitions:
            n += 1
            init = {atom_of(k): 1 for k in tr["state"]}
            nxt = {atom_of(k) for k in tr["next"]}
            if negatives:
                on, on2 = set(tr["state"]), set(tr["next"])
                init.update({neg_of(k): 1 for k in vocabulary
                             if k not in on})
                nxt |= {neg_of(k) for k in vocabulary if k not in on2}
            if with_actions:
                init.update({atom_of(k): 1 for k in tr["action"]})
            if tick_token:
                init["tick"] = 1
            observed = {a: (1 if a in nxt else 0) for a in vocab_atoms}
            prog = doc.catalog.ground(init)
            meta = {"source": f"{tr['domain']} i{tr['instance']} "
                              f"ep{tr['episode']} seed{tr['seed']} "
                              f"t{tr['t']}",
                    "episode": tr["episode"], "t": tr["t"],
                    "action": tr["action"], "reward": tr["reward"],
                    "catalog": str(p)}
            rec = None
            if tier != "sampled":
                try:
                    if frame:
                        from metis.kernel import ForwardProjection
                        from metis.kernel.filter import DONE
                        pr = ForwardProjection(
                            prog, horizon, max_states=max_states,
                            keep_history=False,
                            policy=pricer.policy(prog) if pricer else None)
                        if sum(q for (st, _), q in pr.final.items()
                               if st == DONE) < 1 - 1e-9:
                            unsettled += 1
                    rec = surprise_record(
                        prog, horizon, observed, rare_ratio=rare,
                        max_states=max_states, meta=meta,
                        policy=pricer.policy(prog) if pricer else None)
                    rec["tier"] = "exact"
                except StateBudgetExceeded:
                    if tier == "exact":
                        raise
            if rec is None:
                like, half, uns = _sampled_likelihood(
                    prog, horizon, frame, observed, samples,
                    seed * 1_000_003 + n, pricer)
                unsettled += uns > 0
                kind = ("unmodeled" if like == 0.0 else
                        "rare" if like < rare else None)
                rec = {"format": "metis-surprise", "version": 1,
                       "program_key": program_key(prog),
                       "steps": horizon, "init": dict(prog.init),
                       "observed": dict(observed), "likelihood": like,
                       "kind": kind, "meta": meta, "tier": "sampled",
                       "n": samples, "ci": half}
            tiers[rec["tier"]] = tiers.get(rec["tier"], 0) + 1
            per_ep.setdefault(tr["episode"], 0)
            if rec["kind"]:
                per_ep[tr["episode"]] += 1
                kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
                f.write(json.dumps(rec) + "\n")
    return {"transitions": n, "surprises": sum(per_ep.values()),
            "per_episode": per_ep, "kinds": kinds,
            "readout": "frame" if frame else f"{horizon} step(s)",
            "unsettled": unsettled, "tiers": tiers}


def _batch(ns) -> int:
    d = Path(ns.dir)
    tag = ns.tag or (f"i{ns.instance}-{ns.agent}-s{ns.seed}"
                     f"-n{ns.episodes}")
    tr_path = d / "transitions" / f"{tag}.jsonl"
    if tr_path.exists():
        print(json.dumps({"error": f"{tr_path} exists — pick another "
                                   "--seed or --tag (batches never "
                                   "overwrite recorded episodes)"}))
        return 1
    out = {"record": record_episodes(ns.problem, ns.instance, ns.episodes,
                                     tr_path, ns.seed, ns.agent,
                                     str(d / "gym-logs"))}
    if ns.catalog:
        trs = list(read_transitions(tr_path))
        _, env = load_problem(ns.problem, ns.instance)
        steps = ns.steps if ns.steps == "frame" else int(ns.steps)
        sink = d / "sinks" / f"{tag}--{Path(ns.catalog).stem}.jsonl"
        out["sink"] = surprise_sink(
            ns.catalog, trs, state_vocabulary(env), sink, steps, ns.rare,
            negatives=ns.negatives, tick_token=ns.tick_token,
            tier=ns.tier, samples=ns.samples, frame_cap=ns.frame_cap,
            plugin=ns.plugin, overlay=ns.overlay)
        out["sink"]["path"] = str(sink)
        out["sink"]["surprises_per_episode"] = round(
            out["sink"]["surprises"] / max(1, ns.episodes), 3)
    print(json.dumps(out))
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m metis.domains.rddl_recorder",
        description="record REAL pyRDDLGym episodes (raw transitions "
                    "JSONL) and/or replay them against a catalog into a "
                    "surprise sink")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--problem", required=True)
    r.add_argument("--instance", default="1")
    r.add_argument("--episodes", type=int, default=5)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--agent", default="random")
    r.add_argument("--out", required=True)
    r.add_argument("--log-dir")
    b = sub.add_parser(
        "batch", help="one episodic batch: record REAL episodes, then "
        "(with --catalog) replay them into a surprise sink")
    b.add_argument("--problem", required=True)
    b.add_argument("--instance", default="1")
    b.add_argument("--episodes", type=int, default=5)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--agent", default="random")
    b.add_argument("--dir", required=True,
                   help="experiment directory: gym-logs/, transitions/, "
                        "sinks/ are written under it")
    b.add_argument("--tag", default="",
                   help="file-name tag (default i<inst>-<agent>-s<seed>-n<N>)")
    b.add_argument("--catalog")
    b.add_argument("--steps", default="1")
    b.add_argument("--rare", type=float, default=0.05)
    b.add_argument("--negatives", action="store_true")
    b.add_argument("--tick-token", action="store_true")
    s = sub.add_parser("sink")
    s.add_argument("--catalog", required=True)
    s.add_argument("--transitions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", default="1",
                   help="N metis steps per tick, or 'frame' (settle)")
    s.add_argument("--rare", type=float, default=0.05)
    s.add_argument("--negatives", action="store_true")
    s.add_argument("--tick-token", action="store_true")
    for q in (b, s):
        q.add_argument("--tier", default="exact",
                       choices=("exact", "sampled", "auto"),
                       help="exact filter | sampled (Monte Carlo) | auto "
                            "(exact, sampled when it refuses its budget)")
        q.add_argument("--samples", type=int, default=2000)
        q.add_argument("--frame-cap", type=int, default=FRAME_CAP)
        q.add_argument("--plugin", default=None,
                       help="port plugin binding the catalog's weight "
                            "ports (pkg.mod:factory | file.py:factory, "
                            "beside the catalog)")
        q.add_argument("--overlay", default=None,
                       help="a notch table (JSON) over the plugin's base")
    ns = ap.parse_args(argv)
    if ns.cmd == "batch":
        return _batch(ns)
    if ns.cmd == "record":
        print(json.dumps(record_episodes(
            ns.problem, ns.instance, ns.episodes, ns.out, ns.seed,
            ns.agent, ns.log_dir)))
        return 0
    trs = list(read_transitions(ns.transitions))
    if not trs:
        print(json.dumps({"error": "no transitions"}))
        return 1
    _, env = load_problem(trs[0]["domain"], trs[0]["instance"])
    steps = ns.steps if ns.steps == "frame" else int(ns.steps)
    print(json.dumps(surprise_sink(ns.catalog, trs, state_vocabulary(env),
                                   ns.out, steps, ns.rare,
                                   negatives=ns.negatives,
                                   tick_token=ns.tick_token,
                                   tier=ns.tier, samples=ns.samples,
                                   frame_cap=ns.frame_cap,
                                   plugin=ns.plugin,
                                   overlay=ns.overlay)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
