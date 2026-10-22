"""metis.host — the host contract and the parts every world's host repeats (metis-packages
s-65778b98): the PortPlugin faces a domain agent drives, the JSON-lines child that runs a
world's gym under its own interpreter, and the shim a generated package carries to reach its
world's host. Taken from metis-catalog's tools/rddl_host.py (GymClient, interpreter) and the
two worlds' alf_host.py / mh_host.py / package_host.py, which carry the same code three times;
each world keeps its own gym body (rddl_gym.py, alf_gym.py, mh_gym.py), its atoms and its host.

    from metis.host import PortPlugin, GymChild, find_interpreter, world_host

    class MyHost(PortPlugin):
        def tick(self, counts): ...                 # one beat: the do-lines, or None (the episode is over)
        def on_event(self, name, args): ...         # a fired rule: the world acts on it
    gym = GymChild("world/my_gym.py", find_interpreter("METIS_MY_PY", candidates=[...]),
                   what="gym", game=..., max_steps=60)   # the child's "open" answer is gym.info
    gym.call("step", action=...)                    # one request, one answer: {ok: true, ...} | an error
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod


class PortPlugin(ABC):
    """A catalog host as the domain agent drives it (the kernel's plugin faces, the RDDL host's
    docstring): its world moves on beats and on the program's fired rules; it reads back what the
    world is (observe), so the watch can score the program against it. A host's `make(case)`
    returns one.

    tick(counts)        one beat: a list of do-lines (`+atom`, `-atom N`, …) for the program, or
                        None when the world's episode is over
    on_event(name, args) a fired rule (`go_obj[O=x,L=y]`): the world acts on it at once, so the
                        reading right after the row is its answer
    observe(counts)     the reading {atom: n}; {} when there is nothing new to read
    bind(reg)           the weight ports the host binds (reg.weight(port, fn))
    world(lines)        driver lines, applied to the world now (sync gives the do-lines after)
    sync(counts)        the do-lines that bring the program's state to the world's, no step
    ports()             what the host reads and drives (for a console's ports line)
    snapshot()/restore(snap)  the episode as data, replayed on restore (deterministic per seed)
    reseed(seed)        the domain agent's seed is the episode's

    The hosted day's fields the domain agent reads: beat, due, script, last."""

    beat, due, script, last = 0, (), {}, None

    @abstractmethod
    def tick(self, counts):
        """One beat: the do-lines, or None (the episode is over)."""

    @abstractmethod
    def on_event(self, name, args):
        """A fired rule: the world acts on it."""

    def observe(self, counts):
        return {}

    def bind(self, reg):
        pass

    def world(self, lines):
        raise ValueError("this host takes no world lines")

    def sync(self, counts):
        return []

    def ports(self):
        return {}

    def snapshot(self):
        return None

    def restore(self, snap):
        pass

    def reseed(self, seed):
        pass


def find_interpreter(env=None, module=None, candidates=(), default=True):
    """The interpreter a world's gym runs under: $`env` when set; else this one, when it imports
    `module` (None: skip the check); else the first of `candidates` that exists (a world repo's
    own .venv, a shared venv beside the repos); else this interpreter (`default`), or an error
    naming the variable to set."""
    p = os.environ.get(env) if env else None
    if p:
        return p
    if module and importlib.util.find_spec(module) is not None:
        return sys.executable
    for c in candidates:
        if c and os.path.exists(str(c)):
            return str(c)
    if default:
        return sys.executable
    raise RuntimeError("no interpreter with %s: set %s" % (module or "the world's gym", env or "its variable"))


class GymChild:
    """A world's gym as a child process: one JSON request per line in, one JSON answer per line
    out ({"ok": true, …} or {"ok": false, "error": …}). The child's stderr is kept: its last
    lines name a crash. `what` is the world's word for it in the errors (gym, game, level).
    The constructor sends `open` with `open_args` (keywords, or a dict when one of them is named
    like a parameter here: minihack's level has an `env`); its answer is `info`."""

    def __init__(self, script, interpreter=None, what="gym", env=None, open_args=None, **kw):
        open_args = dict(open_args or {}, **kw)
        self.what = what
        self.err = tempfile.TemporaryFile("w+")
        self.proc = subprocess.Popen([interpreter or sys.executable, str(script)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.err, text=True,
                                     **({"env": env} if env is not None else {}))
        self.info = self.call("open", **open_args)

    def call(self, op, **kw):
        self.proc.stdin.write(json.dumps(dict(kw, op=op)) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            self.err.seek(0)
            tail = self.err.read().strip().splitlines()[-3:]
            raise RuntimeError("the %s process ended (%s): %s" % (self.what, self.proc.poll(), " | ".join(tail)))
        out = json.loads(line)
        if not out.pop("ok"):
            raise RuntimeError("%s %s: %s" % (self.what, op, out["error"]))
        return out

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write('{"op": "close"}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self.proc.kill()

    def __del__(self):
        if getattr(self, "proc", None) is not None:
            self.close()


def world_host(here, module, env=None, world=None, up=6):
    """The shim a generated package carries (the worlds' package_host.py): its world's host module
    `module` (alf_host, mh_host), found in $`env`, else the generator's `world` path, else a
    world/ directory up to `up` levels above the package `here`; that directory is put first on
    sys.path and the module imported. → the module (its host class, made with the package's
    directory)."""
    here = os.path.dirname(os.path.abspath(here)) if os.path.isfile(here) else os.path.abspath(here)
    cands = [os.environ.get(env) or "" if env else "", world or ""]
    d = here
    for _ in range(up):
        d = os.path.dirname(d)
        cands.append(os.path.join(d, "world"))
    for c in cands:
        if c and os.path.exists(os.path.join(c, module + ".py")):
            if c in sys.path:                      # already there but behind the package's own directory: a shim
                sys.path.remove(c)                 # named like the world module would import itself (opus, N2)
            sys.path.insert(0, c)
            break
    return importlib.import_module(module)
