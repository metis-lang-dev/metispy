"""The surprise record — leg 1's logging contract (docs/learning.md):
evaluation is DETERMINISTIC given (artifact, init, observation), so
the runtime logs pointers, not traces; the offline side re-derives
the full context (support, posteriors, MAP explanation) by re-running
the exact machinery — here.

A record is JSON-able and small: the artifact identity (program_key +
caller-supplied meta such as pack versions), the tick's init
multiset, the partially-observed outcome (atom -> count over the
OBSERVED vocabulary only; absent atoms may be asserted with count 0),
the horizon, and the verdict computed at logging time. Zero
likelihood is the STRUCTURAL signal (no parameter can create missing
support — a missing behavior is a missing rule).

`recover` replays a record against a CANDIDATE artifact — typically
the base catalog plus a shadow-admitted pack — returning likelihood,
the smoothed posterior and the MAP derivation (the explanation
object). The shadow-pack mode is a usage pattern, not an API:
condition and explain with the candidate, keep plan() on the base
until the pack is approved."""

import hashlib
import json
from typing import Dict, Optional

from .filter import ForwardProjection
from .nets import program_key
from .posterior import Conditioned, map_derivation


def _evidence(observed: Dict[str, int], steps: int):
    def pred(s):
        counts = dict(s[1])
        return all(counts.get(a, 0) == n for a, n in observed.items())
    return {steps: pred}


def surprise_record(program, steps: int, observed: Dict[str, int],
                    policy=None, rare_ratio: float = 0.05,
                    meta: Optional[dict] = None,
                    max_states=None, env=None) -> dict:
    """Log an observed (partial) outcome against the deployed
    artifact. kind: 'unmodeled' (zero likelihood — the leg-1
    trigger), 'rare' (likelihood below rare_ratio of the modal
    matching mass... below rare_ratio outright), or None.

    `env` (a port plugin's snapshot, taken when the policy priced
    this record): the record carries its hash and, when it is small,
    the snapshot itself as JSON data — so recover() replays against
    the SAME env (a port-priced likelihood is only honest under the
    env it saw)."""
    pr = ForwardProjection(program, steps, policy=policy,
                           max_states=max_states,
                           keep_history=False)   # only .final read
    like = sum(p for s, p in pr.final.items()
               if _evidence(observed, steps)[steps](s))
    kind = "unmodeled" if like == 0.0 else \
        ("rare" if like < rare_ratio else None)
    return {
        "format": "metis-surprise", "version": 1,
        "program_key": program_key(program),
        "steps": steps,
        "init": dict(program.init),
        "observed": dict(observed),
        "likelihood": like,
        "kind": kind,
        "meta": dict(meta or {}),
        **({"env": env_record(env)} if env is not None else {}),
    }


# -- env snapshots (port plugins): the neutral JSON env ---------------------
# A host's snapshot is JSON data (metis-agent RUNTIME-PARITY row 22, ruling
# (d)): what py and Go write into a record and read back alike. The record's
# env is {"format": "json", "hash", "data"}: the hash is sha256 over the
# canonical text (keys sorted, no whitespace, ASCII), the data the snapshot
# as JSON — a tuple is carried as an array, so a host's restore takes lists.
# Records written before carried a pickle ("blob"): never read here;
# `python -m metis.kernel.env_migrate` rewrites them.

class EnvError(TypeError):
    """A snapshot JSON cannot carry (a set, an object, NaN, mixed keys)."""


def env_text(snap) -> str:
    """The canonical JSON text of a snapshot (what env_hash hashes)."""
    try:
        return json.dumps(snap, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise EnvError("a host's snapshot must be JSON data (%s): %s"
                       % (type(snap).__name__, e)) from None


def env_json(snap):
    """The neutral form of a snapshot: the JSON value it reads back as
    (tuples become lists, int keys strings). Every restore gets this
    form, on the checkpoint and the replay alike."""
    return json.loads(env_text(snap))


def env_hash(snap) -> str:
    """A stable fingerprint of an env snapshot: sha256 of env_text."""
    return hashlib.sha256(env_text(snap).encode()).hexdigest()


def env_record(snap, data_cap: int = 65536) -> dict:
    """The `env` field of a surprise record: the snapshot's hash, plus
    the snapshot as JSON data when its text is small enough to carry."""
    text = env_text(snap)
    out = {"format": "json",
           "hash": hashlib.sha256(text.encode()).hexdigest()}
    if len(text) <= data_cap:
        out["data"] = json.loads(text)
    return out


def env_data(env: dict):
    """The snapshot a record's env carries (KeyError when only its hash
    is, or the record is pickled: see env_migrate)."""
    return env["data"]


def recover(record: dict, program, steps: Optional[int] = None,
            policy=None, max_states=None, plugin=None) -> dict:
    """Replay a record against a candidate artifact (offline). If the
    candidate is the RECORDED artifact, verify the key; a different
    key means a shadow candidate is being scored — allowed, that is
    the point. Returns the full re-derived context; likelihood 0
    keeps explanation None (the candidate does not explain it
    either).

    A record carrying `env` and a `plugin` given: the plugin's env is
    restored from the record's data for the replay (the policy's
    bindings read it) and put back after — out["env"] says
    "restored", "matches" (no data, same hash), "MISMATCH" (no data,
    the live env differs: the likelihood is NOT the recorded world's),
    or "legacy" (a pickled env, not read: the replay runs on the live
    env; `python -m metis.kernel.env_migrate` rewrites the record)."""
    steps = steps if steps is not None else record["steps"]
    env_rec = record.get("env")
    if env_rec and plugin is not None:
        live = env_json(plugin.snapshot())
        if "data" in env_rec:
            plugin.restore(env_json(env_rec["data"]))
            try:
                out = recover({k: v for k, v in record.items()
                               if k != "env"}, program, steps, policy,
                              max_states)
            finally:
                plugin.restore(live)
            out["env"] = "restored"
            return out
        out = recover({k: v for k, v in record.items() if k != "env"},
                      program, steps, policy, max_states)
        out["env"] = ("legacy" if "blob" in env_rec else
                      "matches" if env_hash(live) == env_rec["hash"]
                      else "MISMATCH")
        return out
    evidence = _evidence(record["observed"], steps)
    pr = ForwardProjection(program, steps, policy=policy,
                           max_states=max_states)
    try:
        cond = Conditioned(pr, evidence)
    except ValueError:
        return {"likelihood": 0.0, "explains": False,
                "posterior": None, "explanation": None}
    m = map_derivation(pr, evidence)
    names = [pr.events[e][2].name for e in m.events if e is not None]
    return {"likelihood": cond.likelihood, "explains": True,
            "posterior": cond, "explanation": m,
            "explanation_events": names}
