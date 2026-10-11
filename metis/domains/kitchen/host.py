"""The kitchen as a scripted PortPlugin host (runtime-wire step 1,
metis-agent docs/INTEGRATION.md): the env the harness's runtime
process serves `catalogs/kitchen_host.llp` against.

- bind: the six task weights — the KitchenModel base policy (dishes
  while the soup cooks, onions otherwise; small off-weights).
- tick (the INPUT face, one beat at quiescence): the order script —
  "+pot_space 3" when the next order is due, [] on a quiet beat, None
  when the script is over.
- on_event (the OUTPUT face): the env executes the fired intent on its
  OWN kitchen (the same multiset effect the catalog declares — a
  faithful actuator until a surprise is injected).
- observe (the sensor face, optional in the protocol; the watch driver
  reads it after every fired event and on every observe-only beat):
  the env's kitchen over the full vocabulary.

Three injectable surprises, one factory each (plus make_transient_twice,
the misread recurring on the 3rd load — the bounded-reset fixture) (`--host
metis.domains.kitchen.host:make_<kind>`):

- transient: right after the 2nd `load` the sensor misreads the pot
  once (one onion too many); the kitchen itself is unchanged and the
  next reading agrees with the model again.
- wrong_domain: right after the 1st `load` a second cook drops an onion
  into the pot. Nothing in `kitchen` derives it; the sibling card
  `kitchen_shared` (partner_load) does.
- novel: right after the 1st `plate` the soup is spilled — the dish is
  gone and the hand free, nothing served. No card derives it.

Env state mutates only in tick/on_event (the purity contract); a
reading is a pure function of the env (the transient's glitch is env
state: set by the botched on_event, cleared by the next tick or
on_event, so observe() stays repeatable)."""

from typing import Dict, List, Optional

ATOMS = ("hand_free", "held_onion", "held_dish", "held_soup",
         "pot_space", "pot_onion", "cooking", "ready", "served")

# the catalog's declared effects (catalogs/kitchen.llp), consume/produce
# — `cook` reads hand_free ($) and consumes three onions
EFFECTS = {
    "fetch_onion": ({"hand_free": 1}, {"held_onion": 1}),
    "load": ({"held_onion": 1, "pot_space": 1},
             {"hand_free": 1, "pot_onion": 1}),
    "cook": ({"pot_onion": 3}, {"cooking": 1}),
    "fetch_dish": ({"hand_free": 1}, {"held_dish": 1}),
    "plate": ({"held_dish": 1, "cooking": 1}, {"held_soup": 1}),
    "serve": ({"held_soup": 1}, {"hand_free": 1, "served": 1}),
    # kitchen_shared's move, when that card is served: the simulated
    # second cook obliges
    "partner_load": ({"pot_space": 1}, {"pot_onion": 1}),
}

# kind -> (intent, the occurrences of it the env botches)
INJECT = {"transient": ("load", (2,)), "wrong_domain": ("load", (1,)),
          "novel": ("plate", (1,)), "transient_twice": ("load", (2, 3))}


def _weights(reg):
    def port(name):
        def w(args, counts):
            if name == "fetch_dish":
                return 1.0 if counts.get("cooking", 0) > 0 else 0.05
            if name == "fetch_onion":
                return 0.05 if counts.get("cooking", 0) > 0 else 1.0
            return 1.0
        return w
    for name in EFFECTS:
        reg.weight(f"w_{name}", port(name))
    # the sibling card's port (kitchen_shared): a second cook loads
    # rarely next to this one's own moves
    reg.weight("w_partner", 0.1)


class KitchenHost:
    """Scripted orders + a faithful actuator + full sensing, with at
    most one injected surprise."""

    name = "kitchen"

    def __init__(self, case, inject: Optional[str] = None,
                 orders: List[int] = (0, 14), beats: int = 30):
        if inject is not None and inject not in INJECT:
            raise ValueError(f"inject {inject!r}: want one of "
                             f"{', '.join(INJECT)}")
        self.inject = inject
        self.orders = tuple(orders)     # beat -> an order arrives
        self.beats = beats              # the script's length
        self.beat = 0
        self.kitchen: Dict[str, int] = {
            a: n for a, n in dict(case.get("init", {})).items() if n > 0}
        self.seen: Dict[str, int] = {}  # intent -> times executed
        self.glitch = False             # the transient's one misread
        self.injected: List[str] = []   # what the env did off-model

    # -- the protocol ----------------------------------------------------

    def bind(self, reg):
        _weights(reg)

    def tick(self, counts):
        if self.beat >= self.beats:
            return None
        dos = []
        if self.beat in self.orders:
            self.kitchen["pot_space"] = self.kitchen.get("pot_space", 0) + 3
            dos.append("+pot_space 3")
        self.beat += 1
        self.glitch = False
        return dos

    def on_event(self, name, args):
        self.glitch = False
        eff = EFFECTS.get(name)
        if eff is None:
            return
        consume, produce = eff
        for a, n in consume.items():
            self.kitchen[a] = max(0, self.kitchen.get(a, 0) - n)
        for a, n in produce.items():
            self.kitchen[a] = self.kitchen.get(a, 0) + n
        self.kitchen = {a: n for a, n in self.kitchen.items() if n > 0}
        self.seen[name] = self.seen.get(name, 0) + 1
        if self.inject:
            what, when = INJECT[self.inject]
            if name == what and self.seen[name] in when:
                self._surprise()

    def observe(self, counts) -> Dict[str, int]:
        reading = {a: self.kitchen.get(a, 0) for a in ATOMS}
        if self.glitch:
            reading["pot_onion"] += 1
        return reading

    def snapshot(self):
        return (self.beat, dict(self.kitchen), dict(self.seen),
                self.glitch, list(self.injected))

    def restore(self, snap):
        beat, kitchen, seen, glitch, injected = snap
        self.beat, self.kitchen, self.seen = beat, dict(kitchen), dict(seen)
        self.glitch, self.injected = glitch, list(injected)

    # -- the injected surprises -----------------------------------------

    def _surprise(self):
        k = self.kitchen
        if self.inject in ("transient", "transient_twice"):
            self.glitch = True
        elif self.inject == "wrong_domain" and k.get("pot_space", 0) > 0:
            k["pot_space"] -= 1
            k["pot_onion"] = k.get("pot_onion", 0) + 1
        elif self.inject == "novel" and k.get("held_soup", 0) > 0:
            k["held_soup"] -= 1
            k["hand_free"] = k.get("hand_free", 0) + 1
        else:
            return
        self.kitchen = {a: n for a, n in k.items() if n > 0}
        self.injected.append(self.inject)


def make(case):
    return KitchenHost(case)


def make_transient(case):
    return KitchenHost(case, inject="transient")


def make_wrong_domain(case):
    return KitchenHost(case, inject="wrong_domain")


def make_novel(case):
    return KitchenHost(case, inject="novel")


def make_transient_twice(case):
    """The misread comes back on the 3rd load: a 'transient' that
    recurs after a reset (the bounded-reset fixture)."""
    return KitchenHost(case, inject="transient_twice")
