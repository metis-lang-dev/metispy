"""The transport between the processes of one image (metis-lang spec/17 §5.3–5.4, linker s-5908724a step 5b): the
carrier-independent contract and its first carrier, two sockets.

A MESSAGE is one tick's atoms from one process:

    M image=<h12> plan=<h12> from=<process> tick=<n>
    + tick__grant(cook_a,pot) 1
    - tick__want(cook_a,pot) 1
    .

one line per atom, a sign and the atom exactly as a wire's I line writes it (`p(x,y)` and its multiplicity),
read back with the wire's atom reader (`atom`). The HANDSHAKE, before any tick:

    H image=<h12> plan=<h12> lock=<h> process=<name>

and a process ticks only when the peer's H matches its own on image, plan and lock (HandshakeRefused names what
differs). Ordering: a process commits tick n, then sends tick n's message; a receiver takes tick n's message only
after its own tick n-1; a message missing past the plan's budget (in ticks: the receiver's beats waited) is a
`stuck` — the caller reports it to the director (spec 15 §9), never a silent wait.

    SocketCarrier(process, listen=(host, port), peer=(host, port), budget=1)
        .hello(image, plan, lock)        H out, H in, compared
        .send(tick, diff)                 diff: {atom: signed count}
        .receive(tick, beat, wait)       the peer's tick-n message, or Stuck after the budget
        .trace                            [{dir, from, tick, beat, atoms}]: the sender's tick, the latch beat
"""
import re
import socket
import threading
import time

# a wire atom: a name, its arguments in parentheses with no spaces (the I line's form)
ATOM = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\([^()\s]*\))?")
_KV = re.compile(r"(\w+)=(\S+)")


class TransportError(ValueError):
    pass


class HandshakeRefused(TransportError):
    pass


class Stuck(TransportError):
    """No message for the tick within the budget: the receiver reports `stuck` (spec 15 §9)."""

    def __init__(self, process, tick, waited):
        super().__init__("stuck: no message from %s for tick %d after %d beat(s)" % (process, tick, waited))
        self.process, self.tick, self.waited = process, tick, waited


def atom(text):
    """A wire atom as the I line writes it: `p(x,y)` (no spaces) → the same text; TransportError otherwise."""
    if not ATOM.fullmatch(text):
        raise TransportError("not a wire atom: %r" % text)
    return text


def message(image, plan, frm, tick, diff, sequence=()):
    """Tick `tick`'s message from process `frm`: diff {atom: signed count, 0 left out}, in atom order; then
    `sequence`, [(atom, signed count)] kept in its order and never folded — a command port's acts, whose order the
    world's effects depend on (a cook's load before its start), one line each."""
    lines = ["M image=%s plan=%s from=%s tick=%d" % (image, plan, frm, tick)]
    for a in sorted(diff):
        n = int(diff[a])
        if n:
            lines.append("%s %s %d" % ("+" if n > 0 else "-", atom(a), abs(n)))
    for a, n in sequence:
        if n:
            lines.append("%s %s %d" % ("+" if n > 0 else "-", atom(a), abs(int(n))))
    return "\n".join(lines + ["."]) + "\n"


def parse_message(text):
    """→ {image, plan, from, tick, diff, lines}: diff the atoms folded, lines [(atom, signed count)] in the
    message's order; TransportError on a malformed header, atom line or end."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith("M ") or lines[-1] != ".":
        raise TransportError("not a message (M … .): %r" % text[:80])
    head = dict(_KV.findall(lines[0]))
    missing = [k for k in ("image", "plan", "from", "tick") if k not in head]
    if missing:
        raise TransportError("M header without %s" % ", ".join(missing))
    diff, seq = {}, []
    for l in lines[1:-1]:
        parts = l.split()
        if len(parts) != 3 or parts[0] not in "+-" or not parts[2].isdigit():
            raise TransportError("not an atom line (+|- atom n): %r" % l)
        n = (1 if parts[0] == "+" else -1) * int(parts[2])
        diff[atom(parts[1])] = diff.get(parts[1], 0) + n
        seq.append((parts[1], n))
    return {"image": head["image"], "plan": head["plan"], "from": head["from"], "tick": int(head["tick"]),
            "diff": diff, "lines": seq}


def hello(image, plan, lock, process):
    return "H image=%s plan=%s lock=%s process=%s\n" % (image, plan, lock, process)


def parse_hello(line):
    if not line.startswith("H "):
        raise TransportError("not a handshake (H …): %r" % line[:80])
    h = dict(_KV.findall(line))
    missing = [k for k in ("image", "plan", "lock", "process") if k not in h]
    if missing:
        raise TransportError("H without %s" % ", ".join(missing))
    return h


def compare(mine, theirs):
    """The handshake's differences: [(field, mine, theirs)] over image, plan and lock."""
    return [(k, mine[k], theirs[k]) for k in ("image", "plan", "lock") if mine[k] != theirs[k]]


class SocketCarrier:
    """Two sockets, one per direction (§5.3: the first carrier), line-delimited text. Each side LISTENS on its
    own address for the peer's outbound connection and CONNECTS to the peer's address for its own; the inbound
    side is read by a thread into per-tick slots."""

    def __init__(self, process, listen, peer, budget=1, connect_timeout=30.0):
        self.process, self.listen, self.peer, self.budget = process, tuple(listen), tuple(peer), int(budget)
        self.connect_timeout = connect_timeout
        self.trace = []
        self.mine = self.theirs = None
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(self.listen)
        self._srv.listen(1)
        self._out = self._in = None
        self._got, self._hello = {}, None
        self._cv = threading.Condition()
        self._reader = None
        self.closed = False

    # the two connections
    def _connect(self):
        deadline = time.time() + self.connect_timeout
        while True:
            try:
                self._out = socket.create_connection(self.peer, timeout=self.connect_timeout)
                return
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.05)

    def _accept(self):
        self._srv.settimeout(self.connect_timeout)
        self._in, _ = self._srv.accept()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        buf = []
        f = self._in.makefile("r", encoding="utf-8", newline="\n")
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("H "):
                with self._cv:
                    self._hello = line
                    self._cv.notify_all()
                continue
            buf.append(line)
            if line == ".":
                m = parse_message("\n".join(buf))
                buf = []
                with self._cv:
                    self._got[m["tick"]] = m
                    self._cv.notify_all()
        with self._cv:
            self.closed = True
            self._cv.notify_all()

    def _send(self, text):
        self._out.sendall(text.encode("utf-8"))

    def hello(self, image, plan, lock):
        """Connect both ways, send our H, wait for the peer's: HandshakeRefused when image, plan or lock differ."""
        t = threading.Thread(target=self._accept)
        t.start()
        self._connect()
        t.join()
        self.mine = {"image": image, "plan": plan, "lock": lock, "process": self.process}
        self._send(hello(image, plan, lock, self.process))
        with self._cv:
            if not self._cv.wait_for(lambda: self._hello is not None or self.closed, timeout=self.connect_timeout):
                raise TransportError("no handshake from %s:%d" % self.peer)
        if self._hello is None:
            raise TransportError("the peer closed before its handshake")
        self.theirs = parse_hello(self._hello)
        bad = compare(self.mine, self.theirs)
        if bad:
            raise HandshakeRefused("handshake refused by %s: %s" % (self.process, "; ".join(
                "%s %s here, %s at %s" % (k, a, b, self.theirs["process"]) for k, a, b in bad)))
        return self.theirs

    def send(self, tick, diff, sequence=()):
        """Tick `tick`'s committed atoms (diff {atom: signed count}; sequence: ordered lines, see message) to the
        peer."""
        self._send(message(self.mine["image"], self.mine["plan"], self.process, tick, diff, sequence))
        self.trace.append({"dir": "send", "from": self.process, "tick": tick,
                           "atoms": len([n for n in diff.values() if n]) + len(list(sequence))})

    def receive(self, tick, beat, wait=None, lines=False):
        """The peer's tick-`tick` message, latched at our beat `beat`: its diff (with lines=True, (diff, the lines in
        order)). A message not here after `wait`
        seconds is one beat waited; past the budget, Stuck. A message for another image or plan is refused."""
        waited = 0
        while True:
            with self._cv:
                ok = self._cv.wait_for(lambda: tick in self._got or self.closed, timeout=wait)
                m = self._got.pop(tick, None)
            if m is not None:
                if (m["image"], m["plan"]) != (self.mine["image"], self.mine["plan"]):
                    raise TransportError("a message for image %s plan %s, not ours (%s, %s)"
                                         % (m["image"], m["plan"], self.mine["image"], self.mine["plan"]))
                self.trace.append({"dir": "recv", "from": m["from"], "tick": m["tick"], "beat": beat,
                                   "atoms": len(m["lines"])})
                return (m["diff"], m["lines"]) if lines else m["diff"]
            if self.closed and not ok:
                raise Stuck(self.theirs["process"] if self.theirs else "?", tick, waited)
            waited += 1
            if waited > self.budget or self.closed:
                raise Stuck(self.theirs["process"] if self.theirs else "?", tick, waited)

    def close(self):
        for s in (self._out, self._in, self._srv):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
