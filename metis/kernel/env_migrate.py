"""Rewrite records whose env is a pickle into the neutral JSON env.

Records written before the JSON env (metis.kernel.record) carried the
host's snapshot as a base64 pickle ("blob"). The runtime no longer reads
it; this one-off does, once, for records YOU wrote (a pickle runs code
when it is loaded: never point this at a file from elsewhere):

    python -m metis.kernel.env_migrate domains/*/records/*.jsonl

Each line whose env has a blob gets env_record(snapshot): the hash over
the canonical JSON and the snapshot as data (a tuple becomes an array —
the hosts' restore takes that form). A metis-agent AgentHost's (inner,
queues) pair, sunk before the sink kept the catalog host's own snapshot,
is cut to the inner one, as the sink does now. A blob that does not
load, or a snapshot JSON cannot carry, keeps the hash only, with why
under "unread" (the replay then says MISMATCH, as for a record too big
to carry its data). A line without a blob is left as written. The file
is replaced atomically; --keep saves the original beside it as
<file>.pickle-env."""

import argparse
import base64
import json
import os
import pickle
import sys
from pathlib import Path

from .record import EnvError, env_record


def migrate_env(env: dict) -> dict:
    try:
        snap = pickle.loads(base64.b64decode(env["blob"]))
    except Exception as e:  # noqa: BLE001 — the record keeps its hash only
        return {"format": "json", "hash": env.get("hash", ""),
                "unread": "pickle: %s" % e}
    if (isinstance(snap, tuple) and len(snap) == 2 and isinstance(snap[1], tuple)
            and len(snap[1]) == 5 and isinstance(snap[1][4], set)):
        snap = snap[0]          # AgentHost's (inner, (queue, inflight, acted, world, waits))
    try:
        return env_record(snap)
    except EnvError as e:
        return {"format": "json", "hash": env.get("hash", ""),
                "unread": str(e)}


def migrate_file(path: Path, keep: bool = False) -> dict:
    lines = path.read_text().splitlines()
    out, n = [], {"lines": len(lines), "migrated": 0, "unread": 0}
    for ln in lines:
        try:
            rec = json.loads(ln)
        except ValueError:
            out.append(ln)
            continue
        env = rec.get("env") if isinstance(rec, dict) else None
        if isinstance(env, dict) and "blob" in env:
            rec["env"] = migrate_env(env)
            n["unread" if "unread" in rec["env"] else "migrated"] += 1
            ln = json.dumps(rec, default=str)
        out.append(ln)
    if n["migrated"] or n["unread"]:
        if keep:
            path.with_name(path.name + ".pickle-env").write_text(
                "\n".join(lines) + "\n")
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out) + "\n")
        os.replace(tmp, path)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m metis.kernel.env_migrate",
        description="pickled record envs -> the neutral JSON env")
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--keep", action="store_true",
                    help="save each original as <file>.pickle-env")
    ns = ap.parse_args(argv)
    total = {"lines": 0, "migrated": 0, "unread": 0}
    for f in ns.files:
        n = migrate_file(f, keep=ns.keep)
        for k in total:
            total[k] += n[k]
        print("%s: %d lines, %d migrated, %d unread"
              % (f, n["lines"], n["migrated"], n["unread"]))
    print("total: %(lines)d lines, %(migrated)d migrated, %(unread)d unread"
          % total)
    return 1 if total["unread"] else 0


if __name__ == "__main__":
    sys.exit(main())
