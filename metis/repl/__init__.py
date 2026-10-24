"""metis.repl — the REPL as one structured API (metis-packages s-65778b98): every driver of
the REPL (the `metis` CLI, metis-loop's console, metis-agent's tools, the worlds' experiments)
reads the same answers, as data, instead of parsing what the CLI prints. The CLI prints what
these return (`run N` is Repl.run).

    from metis.repl import Repl
    r = Repl.open("domains/plant/plant.llp")             # plugin="pkg.mod:factory" attaches a host
    r.run(3)        # {"stage": "main", "done": False, "counts": {"dry": 1, …}, "fired": ["dry_out", …]}
    r.step()        # one sampled step; step("water") fires by name: the same shape
    r.state()       # {"stage", "done", "counts"}
    s = r.snapshot(); r.run(10); r.restore(s)           # back to the snapshot (the kernel's undo)
    r.undo(); r.reset()                                  # back one mutating command; to the file's #init
    r.cmd("why dry")                                     # any REPL line, its words as text
    Repl.tools()    # the REPL API's tool declarations (metis.lang.repl_api.REPL_API), a copy

The session is metis.lang.repl.Repl (`r.shell`); this class adds nothing to its rules.
"""
import copy
import io

from metis.kernel.refsample import DONE


class Repl:
    """One REPL session, its answers as data."""

    def __init__(self, shell):
        self.shell = shell

    @classmethod
    def open(cls, path, plugin=None, seed=1, **kw):
        """A session on the program at `path`. `plugin` (pkg.mod:factory, or a path's file:factory)
        attaches a host: the caller's own code names it, as the operator's argv does (D4), so it
        attaches as the runtime's host does (host=True)."""
        from metis.lang.repl import Repl as Shell
        shell = Shell(str(path), seed=seed, plugin=plugin, host=bool(plugin), stdout=io.StringIO(), **kw)
        return cls(shell)

    def _quiet(self, fn, *a):
        """Run a shell command with its words kept off the caller's stdout; → the words."""
        out, self.shell.stdout = self.shell.stdout, io.StringIO()
        try:
            fn(*a)
            return self.shell.stdout.getvalue()
        finally:
            self.shell.stdout = out

    def state(self):
        sh = self.shell
        return {"stage": sh.stage, "done": sh.stage == DONE, "counts": dict(sh.counts)}

    def run(self, n=1):
        """Up to n sampled steps (stops at quiescence): {stage, done, counts, fired}."""
        out, self.shell.stdout = self.shell.stdout, io.StringIO()
        try:
            return self.shell.run_steps(int(n))
        finally:
            self.shell.stdout = out

    def step(self, pick=None):
        """One step: sampled, or `pick` (a candidate's number or name). {stage, done, counts,
        fired, said} — fired [] when nothing fired (said: the REPL's words why)."""
        k = len(self.shell.fired)
        said = self._quiet(self.shell.do_step, "" if pick is None else str(pick))
        return dict(self.state(), fired=list(self.shell.fired[k:]), said=said.strip())

    def undo(self):
        """Back one mutating command: the state after it."""
        self._quiet(self.shell.do_undo, "")
        return self.state()

    def reset(self):
        """Back to the file's #init directives."""
        self._quiet(self.shell.do_reset, "")
        return self.state()

    def snapshot(self):
        """An opaque token for the session as it is now (valid in this session): restore(token)
        comes back to it, through the kernel's own undo (the fired rows, the env, the masks)."""
        self.shell._mark()
        return len(self.shell.undo_stack)

    def restore(self, token):
        sh = self.shell
        if not isinstance(token, int) or token < 1 or token > len(sh.undo_stack):
            raise ValueError("restore: a snapshot of this session, not %r" % (token,))
        if token <= getattr(sh.undo_stack, "off", 0):          # a bounded history (watch-history)
            raise ValueError("restore: %s" % sh._kept_note())
        while len(sh.undo_stack) > token:
            sh.undo_stack.pop()
        self._quiet(sh.do_undo, "")
        return self.state()

    def cmd(self, line):
        """Any REPL line: its words as text (the shell's own)."""
        return self._quiet(self.shell.onecmd, str(line))

    @staticmethod
    def tools():
        """The REPL API's tool declarations (repl-tools: published once by metispy)."""
        from metis.lang.repl_api import REPL_API
        return copy.deepcopy(REPL_API)


def open(path, plugin=None, seed=1, **kw):   # noqa: A001 — metis.repl.open(...) reads as the verb
    return Repl.open(path, plugin=plugin, seed=seed, **kw)
