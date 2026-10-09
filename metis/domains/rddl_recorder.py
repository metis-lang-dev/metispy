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
discovers it)."""
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


def surprise_sink(catalog_path, transitions: Iterable[dict],
                  vocabulary: List[str], out_jsonl, steps: int = 1,
                  rare: float = 0.05, max_states: int = 200_000,
                  with_actions: bool = True) -> Dict:
    """Replay each transition against the catalog; write surprise
    records (kind unmodeled|rare) to `out_jsonl` (overwritten). Returns
    {transitions, surprises, per_episode {ep: n}, kinds}."""
    from metis.kernel.record import surprise_record
    from metis.lang import compile_catalog, parse, resolve_includes
    p = Path(catalog_path)
    doc = compile_catalog(resolve_includes(parse(p.read_text()),
                                           p.parent))
    vocab_atoms = [atom_of(k) for k in vocabulary]
    per_ep: Dict[int, int] = {}
    kinds: Dict[str, int] = {}
    n = 0
    with open(out_jsonl, "w") as f:
        for tr in transitions:
            n += 1
            init = {atom_of(k): 1 for k in tr["state"]}
            if with_actions:
                init.update({atom_of(k): 1 for k in tr["action"]})
            nxt = {atom_of(k) for k in tr["next"]}
            observed = {a: (1 if a in nxt else 0) for a in vocab_atoms}
            prog = doc.catalog.ground(init)
            rec = surprise_record(
                prog, steps, observed, rare_ratio=rare,
                max_states=max_states,
                meta={"source": f"{tr['domain']} i{tr['instance']} "
                                f"ep{tr['episode']} seed{tr['seed']} "
                                f"t{tr['t']}",
                      "episode": tr["episode"], "t": tr["t"],
                      "action": tr["action"], "reward": tr["reward"],
                      "catalog": str(p)})
            per_ep.setdefault(tr["episode"], 0)
            if rec["kind"]:
                per_ep[tr["episode"]] += 1
                kinds[rec["kind"]] = kinds.get(rec["kind"], 0) + 1
                f.write(json.dumps(rec) + "\n")
    return {"transitions": n, "surprises": sum(per_ep.values()),
            "per_episode": per_ep, "kinds": kinds}


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
    s = sub.add_parser("sink")
    s.add_argument("--catalog", required=True)
    s.add_argument("--transitions", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=1)
    s.add_argument("--rare", type=float, default=0.05)
    ns = ap.parse_args(argv)
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
    print(json.dumps(surprise_sink(ns.catalog, trs, state_vocabulary(env),
                                   ns.out, ns.steps, ns.rare)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
