"""The kitchen pair (metis-agent docs/RACK.md §5, rack step 1): ONE
physical kitchen, two cooks, two domains — each a runtime serving its
own cook's view, both hosted on the same env.

- `kitchen` (catalogs/kitchen_host.llp): the first cook. It does not
  model the second cook, so an onion the second cook drops into its pot
  is a SURPRISE on this tape (unmodeled; the kitchen_shared card
  explains it).
- `kitchen_shared` (catalogs/shared/kitchen_shared.llp): the second
  cook, with its own orders and its own pot, plus partner_load — it
  drops the onion it holds into the first cook's pot. On this tape that
  is an ordinary EVENT: the fact that explains the first tape's surprise.

The pair script fixes the orders on both and the one helping order:
the second cook helps on the first cook's 2nd order, right after the
first cook's first load of it (the pot has room). `run_pair(seed)` drives
both runtimes in one process, interleaved step for step on the shared
env, and returns both signal streams on one clock (`t` = the global
step) plus mirror rows at the points the E7 corpus asks its questions.

Separate processes (the harness's phase-1b wiring, RACK §5): the hosts
must share the env; `make_pair_kitchen` / `make_pair_shared` with
METIS_PAIR_DIR set share it through a JSON file under that directory
(a lock per access). Without it each process gets its own env, coupled
only by the script."""

import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from .host import ATOMS, EFFECTS, _weights

CATALOGS = Path(__file__).parent / "catalogs"
SCENES = {"kitchen": CATALOGS / "kitchen_host.llp",
          "kitchen_shared": CATALOGS / "shared" / "kitchen_shared.llp"}

# the pair script: env beats at which each cook's order arrives, and the
# first cook's order (1-based) the second cook helps with
SCRIPT = {"orders": {"kitchen": [0, 30, 60], "kitchen_shared": [8, 38, 68]},
          "help_order": 2, "beats": 90}


class PairEnv:
    """The one kitchen: two cooks' stations and the bookkeeping the
    script needs. Plain data (snapshot = a deepcopy-able dict)."""

    def __init__(self, script=SCRIPT):
        self.script = script
        self.k = {"kitchen": {"hand_free": 1}, "kitchen_shared": {"hand_free": 1}}
        self.beat = {"kitchen": 0, "kitchen_shared": 0}
        self.order = {"kitchen": 0, "kitchen_shared": 0}   # orders arrived
        self.loads = {"kitchen": 0, "kitchen_shared": 0}   # loads this order
        self.helped = []                                   # orders helped
        self.injected = []                                 # ground truth

    def state(self):
        return json.loads(json.dumps(self.__dict__))

    def load_state(self, d):
        self.__dict__.update(json.loads(json.dumps(d)))

    def help_due(self) -> bool:
        """The script's one helping move: on the first cook's help order,
        after its first load, while its pot has room."""
        n = self.script["help_order"]
        a = self.k["kitchen"]
        return (self.order["kitchen"] == n and n not in self.helped
                and self.loads["kitchen"] >= 1 and a.get("pot_space", 0) > 0)


class _FileEnv:
    """PairEnv shared through a JSON file (separate processes)."""

    def __init__(self, d):
        self.path = Path(d) / "kitchen-pair.json"
        if not self.path.exists():
            self.path.write_text(json.dumps(PairEnv().state()))

    @contextmanager
    def env(self):
        import fcntl
        with open(self.path, "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            e = PairEnv()
            e.load_state(json.loads(fh.read() or "{}") or e.state())
            yield e
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(e.state()))


class _MemEnv:
    def __init__(self, env):
        self._env = env

    @contextmanager
    def env(self):
        yield self._env


class CookHost:
    """One cook's PortPlugin over the shared env."""

    def __init__(self, path: str, shared):
        self.path, self.shared, self.name = path, shared, path

    def bind(self, reg):
        _weights(reg)
        if self.path == "kitchen_shared":
            def w_partner(args, counts):
                with self.shared.env() as e:
                    return 10.0 if e.help_due() else 0.0
            reg.weight("w_partner", w_partner)

    def tick(self, counts):
        with self.shared.env() as e:
            if e.beat[self.path] >= e.script["beats"]:
                return None
            dos = []
            if e.beat[self.path] in e.script["orders"][self.path]:
                k = e.k[self.path]
                k["pot_space"] = k.get("pot_space", 0) + 3
                e.order[self.path] += 1
                e.loads[self.path] = 0
                dos.append("+pot_space 3")
            e.beat[self.path] += 1
            return dos

    def on_event(self, name, args):
        with self.shared.env() as e:
            k = e.k[self.path]
            if name == "partner_load":           # into the OTHER pot
                k["held_onion"] = k.get("held_onion", 0) - 1
                k["hand_free"] = k.get("hand_free", 0) + 1
                a = e.k["kitchen"]
                a["pot_space"] -= 1
                a["pot_onion"] = a.get("pot_onion", 0) + 1
                e.helped.append(e.order["kitchen"])
                e.injected.append({"kind": "partner_load",
                                   "by": "kitchen_shared",
                                   "order": e.order["kitchen"],
                                   "beat": e.beat[self.path]})
            else:
                consume, produce = EFFECTS[name]
                for a_, n in consume.items():
                    k[a_] = max(0, k.get(a_, 0) - n)
                for a_, n in produce.items():
                    k[a_] = k.get(a_, 0) + n
                if name == "load":
                    e.loads[self.path] += 1
            e.k[self.path] = {a_: n for a_, n in k.items() if n > 0}

    def observe(self, counts) -> Dict[str, int]:
        with self.shared.env() as e:
            k = e.k[self.path]
            return {a: k.get(a, 0) for a in ATOMS}

    def snapshot(self):
        with self.shared.env() as e:
            return e.state()

    def restore(self, snap):
        with self.shared.env() as e:
            e.load_state(snap)

    def schedule(self):
        with self.shared.env() as e:
            return {"orders": e.script["orders"][self.path],
                    "beats": e.script["beats"],
                    "help_order": e.script["help_order"],
                    "surprises": []}

    @property
    def injected(self):
        with self.shared.env() as e:
            return list(e.injected)


def _shared():
    d = os.environ.get("METIS_PAIR_DIR")
    return _FileEnv(d) if d else _MemEnv(PairEnv())


def make_pair_kitchen(case):
    return CookHost("kitchen", _shared())


def make_pair_shared(case):
    return CookHost("kitchen_shared", _shared())


# -- the in-process pair run (the E7 source) ----------------------------------

def run_pair(seed: int = 1, ask_at=None, triage=True):
    """Both runtimes on one env, stepped alternately until both are done.
    -> {"signals": [...], "mirrors": [...], "truth": [...]}: every signal
    tagged with `t` (the global step); `ask_at(t, sigs) -> label|None`
    snapshots both mirror lines when it returns a label. triage: the
    first cook's surprise is explained by the kitchen_shared card (the
    runtime's own explain op) so its run goes on."""
    from metis.lang.repl import Repl
    from metis.loop.watch import Watch, mirror
    from metis.loop.wire import dispatch
    env = _MemEnv(PairEnv())
    shells, watches = {}, {}
    for path, scene in SCENES.items():
        sh = Repl(scene, seed=seed, plugin="metis.domains.kitchen.host:make",
                  host=True, stdout=io.StringIO())
        # attach the pair host in place of the stub (in-process: no import)
        host = CookHost(path, env)
        from metis.fw.registry import Registry
        from metis.lang.compiler import PortPricer
        reg = Registry()
        host.bind(reg)
        sh.plugin, sh.registry, sh.plugin_spec = host, reg, \
            f"metis.domains.kitchen.pair:{'make_pair_kitchen' if path == 'kitchen' else 'make_pair_shared'}"
        sh.pricer = PortPricer(sh.doc.ports, reg.weight_bindings(),
                               horn=sh.doc.catalog.horn_db())
        sh._env_attach()
        shells[path] = sh
        watches[path] = Watch(sh, beats=SCRIPT["beats"] * 4)
    sigs, mirrors, t = [], [], 0
    while not all(w.done for w in watches.values()):
        for path in SCENES:
            w = watches[path]
            if w.done:
                continue
            t += 1
            for s in w.step():
                sigs.append({**s, "t": t})
                if triage and s["signal"] == "surprise" and path == "kitchen":
                    dispatch(shells[path], "intent",
                             f"explain {s['record']} kitchen_shared partner_load")
                    for r in w.outbox:
                        sigs.append({**r, "t": t})
                    w.outbox.clear()
            if ask_at is not None:
                label = ask_at(t, sigs)
                if label:
                    mirrors.append({"label": label, "t": t, "rows": [
                        mirror(shells[p], watches[p])[0] for p in SCENES]})
    return {"signals": sigs, "mirrors": mirrors,
            "truth": list(env._env.injected),
            "served": {p: shells[p].counts.get("served", 0) for p in SCENES}}
