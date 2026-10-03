"""One interpreter on the bus: wraps `metis --json WORLD.llp` and
serves envelopes addressed to its name.

The shim holds no semantics and no policy: an envelope's {op, arg}
goes to the child's stdin verbatim, the one result line comes back,
and is posted to the bus addressed to whoever asked. Ops run
strictly in order (the wire answers in order), so a node is a
sequential seat — parallelism is BETWEEN nodes, per SPEC §10.
"""
import argparse
import json
import subprocess
import sys

from .bus import BusClient


def serve(name, world, metis="metis", dir=None, timeout=None):
    child = subprocess.Popen(
        [metis, "--json", world], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1)
    client = BusClient(name, dir)
    print(f"node {name}: {world} on the bus", file=sys.stderr)
    try:
        while True:
            env = client.recv(timeout=timeout)
            op = env.get("op")
            if op == "superseded":
                print(f"node {name}: superseded by a newer hello",
                      file=sys.stderr)
                return 1
            child.stdin.write(json.dumps(
                {"op": op, "arg": env.get("arg") or ""}) + "\n")
            child.stdin.flush()
            line = child.stdout.readline()
            result = (json.loads(line) if line else
                      {"ok": False, "error": "wire closed"})
            client.send(env.get("src", "bus"), "result", result,
                        re=env.get("seq"))
            if not line:
                return 1
    except (KeyboardInterrupt, TimeoutError):
        return 0
    finally:
        client.close()
        child.terminate()
        child.wait(timeout=5)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m metis.loop.node",
        description="attach one metis --json interpreter to the "
                    "loop bus")
    ap.add_argument("name", help="this node's bus name")
    ap.add_argument("world", help="the .llp world it serves")
    ap.add_argument("--metis", default="metis",
                    help="metis executable (default: metis)")
    ap.add_argument("--dir", default=None, help="bus directory")
    a = ap.parse_args(argv)
    return serve(a.name, a.world, metis=a.metis, dir=a.dir)


if __name__ == "__main__":
    raise SystemExit(main())
