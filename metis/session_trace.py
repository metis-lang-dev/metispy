"""A REPL session's trace (metis-loop docs/WORKSPACE.md §3, §10 two terminals): the Segments writer in
.metis/traces/<session>/, the session as one member — one class for the console's REPL sessions (metis-loop
loop/workspace_ch.py imports it) and the terminal REPL started in a workspace (metis.lang.repl), so a surprise
dropped from either points at a trace `python -m metis.trace replay` reproduces (two-terminals s-065d46d5)."""
import time
from pathlib import Path

from metis import trace as T

# verbs that move a REPL session off its recorded course: the trace cuts (a new segment from the session's own state)
CUT_VERBS = frozenset(["undo", "reset", "whatif", "load", "seed", "accept"])


class SessionTrace:
    """A REPL session's trace (WORKSPACE.md §3): metis.trace's Segments writer in .metis/traces/<session>/, the
    session as ONE member. A command is a tick: the rows it fired, as py Live's Tracer writes a member's rows
    between beats ({"take": do-lines} for an input, {"step": [rules]} for rules fired), so `metis.trace replay`
    reproduces it on the session's program. A command that moves the session off that course (undo, reset, whatif,
    a recompile, or one that changed the state with no row) starts a new segment from the session's own state,
    marked cut: the replay resumes from its checkpoint and never reads across it."""

    def __init__(self, path, shell, member, program=None, extensions=()):
        """PROGRAM: the session's checkout .llp (the workspace owns the file: no in-memory sessions,
        workspace-files s-9ea4cfa4); EXTENSIONS: the names merged into the session (the header says so: a
        replay needs them merged too)."""
        self.path, self.member = Path(path), member
        self.w = T.Segments(self.path, every=0)
        self.k = 0
        self.program = Path(program) if program else None
        self.extensions = [str(e) for e in extensions]
        self._attach(shell, cut=None)

    def _header(self, cut):
        m = {"seed": getattr(self.shell, "seed", 1) or 1, "weights": T.weights_of(self.shell), "out": []}
        if self.program is not None and self.program.is_file():
            m["key"], m["program"] = T.program_key(self.program), str(self.program)
        if self.extensions:
            m["extensions"] = list(self.extensions)       # merged into the session: a replay needs them too
        h = {"clock": "command", "composition": None, "image": None, "protocol": T.protocol_key(), "commands": [],
             "clocks": [{"domain": self.member, "n": 1}], "live": T.SESSION_LIVE, "members": {self.member: m}}
        if cut:
            h["cut"] = cut
        return h

    @staticmethod
    def _key(shell):
        return getattr(shell, "key", None)

    def _attach(self, shell, cut):
        self.shell = shell
        self.key = self._key(shell)
        self.rows = len(shell.fired_rows)
        self.w.open_segment(self.k, self._header(cut),
                            {"sessions": {self.member: T.session_state(shell)}, "copies": {}, "world": None})

    def before(self, shell):
        if shell is not self.shell or self._key(shell) != self.key:   # recompiled: a new interpreter or program
            self.k += 1
            self._attach(shell, "reload")
        self.state = T.session_state(shell)
        self.rows = min(self.rows, len(shell.fired_rows))

    def after(self, shell, cmd):
        """The command's tick, or a cut when the command left the recorded course."""
        verb = cmd.strip().lstrip(":").split(" ", 1)[0]
        n = len(shell.fired_rows)
        now = T.session_state(shell)
        moved = now != self.state
        take = self._take_of(cmd, verb, now) if shell is self.shell and n == self.rows and moved else None
        if take is not None:                             # an intervention: the session's input, a take (loop-chat F2)
            self.k += 1
            self.w.line({"k": self.k, "t": round(time.time(), 3),
                         "members": {self.member: {"why": "command", "cmd": cmd[:200],
                                                   "seq": [{"take": take, "by": "person"}], "out": []}}})
            self.state = now
            return
        reloaded = shell is not self.shell or self._key(shell) != self.key
        if reloaded or verb in CUT_VERBS or n < self.rows or (n == self.rows and moved):
            self.k += 1
            self._attach(shell, "reload" if reloaded else "cut")
            return
        if n == self.rows:
            return                                       # a read: nothing moved, no tick
        seq = []
        for kind, _, ev in shell.fired_rows[self.rows:n]:
            if kind == "exo":
                seq.append({"take": list(ev.ops), "by": "tool"})
            elif seq and "step" in seq[-1]:
                seq[-1]["step"].append(ev.name)
            else:
                seq.append({"step": [ev.name], "by": "tool"})
        self.rows = n
        self.k += 1
        self.w.line({"k": self.k, "t": round(time.time(), 3),
                     "members": {self.member: {"why": "command", "cmd": cmd[:200], "seq": seq, "out": []}}})

    def _take_of(self, cmd, verb, now):
        """`do +ATOM [N]` / `do -ATOM [N]` as the take a replay applies ([its do-line]), when that line on the counts
        before gives exactly the counts now and nothing else of the state moved (stage, quiet, rng); else None (a
        cut, as any other move with no row). The REPL's do appends no row, so without this the session's input
        lived only in a checkpoint and the verbs (replay, fired, the judge) never saw it."""
        if verb != "do":
            return None
        line = cmd.strip().lstrip(":").split(" ", 1)[1].strip() if " " in cmd.strip() else ""
        if not line or line[0] not in "+-":
            return None
        from metis.fw.plugin import exo_event
        try:
            _, after = exo_event([line], dict(self.state["counts"]))
        except Exception:                                # noqa: BLE001 — a line the do-line grammar does not read: a cut
            return None
        same = all(now[k] == self.state[k] for k in now if k != "counts")
        return [line] if same and after == now["counts"] else None

    def last_k(self):
        return self.k

    def close(self):
        self.w.close()
