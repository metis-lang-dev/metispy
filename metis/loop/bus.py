"""The loop bus — multi-interpreter message passing over a
dict-as-blackboard, REPL-side (same machine). Kernel-owned like
every loop seam (one kernel REPL, every UI a wrapper); metis-loop
re-exports this as `loop.bus`.

Mechanism only, no policy: a broker owns ONE well-known named pipe
(the shared ingress) and the blackboard; `metis --json` interpreters
attach through the node shim (metis.loop.node); any attached src may
read/write the blackboard or address an {op, arg} line to another
node. WHO moves WHAT between worlds is the driver's business (a
script, a session, a test) — the bus never invents data flow. This
is the SPEC §10 doctrine at toy scale: the blackboard is each
world's environment, coupling is explicit, no joint measure is
claimed.

Topology: every participant WRITES envelopes to the single ingress
fifo, one JSON per line (keep a line under PIPE_BUF so concurrent
writers never interleave); each participant OWNS a private reply
fifo, announced at hello, where everything addressed to it arrives.

Envelope: {"src", "dst", "epoch", "seq", "op", "arg"}
  dst "bus":   hello (arg = reply fifo path) | bye
  dst "bb":    set {key, value} | get KEY | dump
  dst <node>:  delivered to that node, which runs {op, arg} on its
               metis --json wire and posts back
               {"src": node, "dst": <orig src>, "op": "result",
                "re": <orig seq>, "arg": <wire result>}

Staleness protection: a channel is (src, epoch), opened by hello
with a FRESH epoch. The broker drops, with a stderr note: envelopes
from a src with no hello; envelopes with a non-current epoch (a
stale repl from a previous run still writing — a new hello for the
same src SUPERSEDES the old channel, which is told so and closed);
and seq <= the last accepted seq of the channel (duplicated or
replayed lines). Blackboard entries keep (src, epoch, seq)
provenance, so every value's origin stays auditable.
"""
import argparse
import json
import os
import select
import sys
import time
import uuid

PIPE_BUF = 4096  # atomic-write bound for one envelope line


def bus_dir(path=None):
    d = path or os.environ.get("METIS_BUS_DIR") or os.path.join(
        os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
        "metis-bus-" + (os.environ.get("USER") or str(os.getuid())))
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


def ingress_path(d=None):
    return os.path.join(bus_dir(d), "bus.fifo")


def _mkfifo(path):
    try:
        os.mkfifo(path, 0o600)
    except FileExistsError:
        pass
    return path


class Rx:
    """Read side of a fifo that never sees EOF: we hold our own
    write end open, so writers may come and go."""

    def __init__(self, path):
        _mkfifo(path)
        self.rfd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self._keep = os.open(path, os.O_WRONLY)
        self._buf = b""

    def readline(self, timeout=None):
        while b"\n" not in self._buf:
            r, _, _ = select.select([self.rfd], [], [], timeout)
            if not r:
                raise TimeoutError(
                    f"bus: nothing within {timeout}s")
            try:
                chunk = os.read(self.rfd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise EOFError("bus: fifo closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode()

    def close(self):
        for fd in (self.rfd, self._keep):
            try:
                os.close(fd)
            except OSError:
                pass


class Tx:
    """Write side of a fifo. The open is non-blocking so a reply
    fifo whose reader died (a stale repl) fails fast (ENXIO)
    instead of hanging the broker."""

    def __init__(self, path):
        self.path = path
        self._f = None

    def send(self, obj):
        line = json.dumps(obj) + "\n"
        if len(line) > PIPE_BUF:
            raise ValueError(
                f"bus: envelope over PIPE_BUF ({len(line)} bytes) — "
                "pass bulk values through a file, key on the bb")
        if self._f is None:
            fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
            os.set_blocking(fd, True)
            self._f = os.fdopen(fd, "w")
        self._f.write(line)
        self._f.flush()

    def close(self):
        if self._f is not None:
            try:
                self._f.close()
            except OSError:
                pass
            self._f = None


class Broker:
    """The bus: one ingress fifo in, per-channel reply fifos out,
    the blackboard in the middle."""

    def __init__(self, dir=None):
        self.dir = bus_dir(dir)
        self.path = _mkfifo(ingress_path(self.dir))
        self.bb = {}    # key -> {value, src, epoch, seq, ts}
        self.chan = {}  # src -> {epoch, seq, reply, tx}

    def _drop(self, why, env):
        print(f"bus: drop ({why}): {json.dumps(env)[:200]}",
              file=sys.stderr)

    def _send(self, src, obj):
        ch = self.chan.get(src)
        if ch is None:
            return
        try:
            ch["tx"].send(obj)
        except OSError as e:
            # dead reader = stale repl on the OTHER end: evict it
            self._drop(f"reply fifo {e}; evicting '{src}'", obj)
            ch["tx"].close()
            self.chan.pop(src, None)

    def _result(self, src, re, arg):
        self._send(src, {"src": "bus", "dst": src, "op": "result",
                         "re": re, "arg": arg})

    def handle(self, env):
        src, dst = env.get("src"), env.get("dst")
        op, arg = env.get("op"), env.get("arg")
        epoch, seq = env.get("epoch"), env.get("seq")
        if not src or not dst or not epoch or not isinstance(seq, int):
            return self._drop("malformed", env)
        if dst == "bus" and op == "hello":
            old = self.chan.get(src)
            if old is not None and old["epoch"] == epoch:
                return  # idempotent re-hello
            if old is not None:  # SUPERSEDE the stale channel
                self._send(src, {"src": "bus", "dst": src,
                                 "op": "superseded", "arg": epoch})
                old["tx"].close()
            self.chan[src] = {"epoch": epoch, "seq": seq,
                              "reply": arg, "tx": Tx(arg)}
            self._send(src, {"src": "bus", "dst": src,
                             "op": "hello-ok", "re": seq,
                             "arg": {"nodes": sorted(self.chan),
                                     "bb": len(self.bb)}})
            return
        ch = self.chan.get(src)
        if ch is None:
            return self._drop("no hello", env)
        if epoch != ch["epoch"]:
            return self._drop("stale epoch", env)
        if seq <= ch["seq"]:
            return self._drop("duplicate seq", env)
        ch["seq"] = seq
        if dst == "bus":
            if op == "bye":
                ch["tx"].close()
                del self.chan[src]
                return
            return self._drop("unknown bus op", env)
        if dst == "bb":
            if op == "set":
                key = arg["key"]
                self.bb[key] = {"value": arg["value"], "src": src,
                                "epoch": epoch, "seq": seq,
                                "ts": time.time()}
                return self._result(src, seq, {"ok": True,
                                               "key": key})
            if op == "get":
                e = self.bb.get(arg)
                out = {"ok": e is not None, "key": arg}
                if e is not None:
                    out.update(value=e["value"], src=e["src"])
                return self._result(src, seq, out)
            if op == "dump":
                return self._result(src, seq, {
                    "ok": True,
                    "bb": {k: v["value"]
                           for k, v in sorted(self.bb.items())}})
            return self._drop("unknown bb op", env)
        if dst not in self.chan:
            return self._result(src, seq, {
                "ok": False, "error": f"no such node '{dst}'",
                "nodes": sorted(self.chan)})
        self.chan[dst]["tx"].send(env)  # verbatim; node answers back

    def serve(self):
        rx = Rx(self.path)
        try:
            while True:
                line = rx.readline()
                try:
                    env = json.loads(line)
                except ValueError:
                    self._drop("bad json", {"line": line[:120]})
                    continue
                self.handle(env)
        finally:
            rx.close()


class BusClient:
    """One attached participant: a fresh (src, epoch) channel. Used
    by the node shim and by drivers/tests alike."""

    def __init__(self, name, dir=None, timeout=30.0):
        self.name = name
        self.dir = bus_dir(dir)
        self.epoch = uuid.uuid4().hex[:12]
        self.seq = 0
        self.timeout = timeout
        self.reply_path = os.path.join(
            self.dir, f"{name}.{self.epoch}.fifo")
        self.rx = Rx(self.reply_path)
        self.tx = Tx(ingress_path(self.dir))
        self.send("bus", "hello", self.reply_path)
        ack = self.recv()
        if ack.get("op") != "hello-ok":
            raise RuntimeError(f"bus: bad hello ack: {ack}")

    def send(self, dst, op, arg=None, re=None):
        self.seq += 1
        env = {"src": self.name, "dst": dst, "epoch": self.epoch,
               "seq": self.seq, "op": op, "arg": arg}
        if re is not None:
            env["re"] = re
        self.tx.send(env)
        return self.seq

    def recv(self, timeout=None):
        return json.loads(
            self.rx.readline(timeout or self.timeout))

    def request(self, dst, op, arg=None):
        """Send and wait for the matching result. Anything else
        arriving meanwhile (an op addressed to us) is refused back
        to its sender — a pure driver is not a node."""
        seq = self.send(dst, op, arg)
        while True:
            m = self.recv()
            if m.get("op") == "superseded":
                raise EOFError("bus: channel superseded")
            if m.get("re") == seq:
                return m.get("arg")
            if m.get("op") != "result":
                self.send(m.get("src", "bus"), "result",
                          {"ok": False,
                           "error": f"'{self.name}' is not a node"},
                          re=m.get("seq"))

    def bb_set(self, key, value):
        return self.request("bb", "set", {"key": key, "value": value})

    def bb_get(self, key):
        return self.request("bb", "get", key)

    def bb_dump(self):
        return self.request("bb", "dump")

    def op(self, node, op, arg=""):
        """One {op, arg} line on a named interpreter's wire."""
        return self.request(node, op, arg)

    def close(self):
        try:
            self.send("bus", "bye")
        except OSError:
            pass
        self.tx.close()
        self.rx.close()
        try:
            os.unlink(self.reply_path)
        except OSError:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m metis.loop.bus",
        description="the loop bus: blackboard broker on one named "
                    "pipe")
    ap.add_argument("--dir", default=None,
                    help="bus directory (default: "
                         "$XDG_RUNTIME_DIR/metis-bus-$USER)")
    a = ap.parse_args(argv)
    b = Broker(a.dir)
    print(f"bus: serving {b.path}", file=sys.stderr)
    try:
        b.serve()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
