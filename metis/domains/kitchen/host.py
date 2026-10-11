"""The kitchen as a scripted PortPlugin host (runtime-wire, metis-agent
docs/INTEGRATION.md): the env the harness's runtime process serves
`catalogs/kitchen_host.llp` against.

- bind: the six task weights — the KitchenModel base policy (dishes
  while the soup cooks, onions otherwise; small off-weights).
- tick (the INPUT face, one beat at quiescence): the order script —
  "+pot_space 3" when the next order is due, [] on a quiet beat, None
  when the script is over (never, for an open-ended script).
- on_event (the OUTPUT face): the env executes the fired intent on its
  OWN kitchen (the same multiset effect the catalog declares — a
  faithful actuator until a surprise is injected).
- observe (the sensor face): the env's kitchen over the full vocabulary.

Surprises come from a SCHEDULE: entries {kind, after, skip, count} — the
entry arms at env beat `after` and fires on the (skip+1)-th matching
intent from then on, `count` times. Four kinds (+ the recurring fifth):

- transient: right after a `load` the sensor misreads the pot once (one
  onion too many); the kitchen is unchanged, the next reading agrees.
- wrong_domain: right after a `load` (pot capacity left) a second cook
  drops an onion in. `kitchen` cannot derive it; the sibling card
  `kitchen_shared` (partner_load) does.
- novel: right after a `plate` the soup is spilled — dish gone, hand
  free, nothing served. No card derives it.
- partner: wrong_domain on the first load of `count` successive orders
  (explained, then escalated: the served world is not switched).
- transient_twice: the misread on `count` consecutive loads (the
  bounded-reset fixture).

`schedule()` is the plan as data and `injected` what fired (kind, beat,
intent) — the eval's ground truth, read by the ledger.

Env state mutates only in tick/on_event (the purity contract); a
reading is a pure function of the env (the glitch is env state: set by
the botched on_event, cleared by the next tick or on_event)."""

import random
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

KINDS = ("transient", "wrong_domain", "novel", "partner",
         "transient_twice")
TRIGGER = {"transient": "load", "wrong_domain": "load", "novel": "plate",
           "partner": "load", "transient_twice": "load"}


def entry(kind, after=0, skip=0, count=None):
    if kind not in KINDS:
        raise ValueError(f"surprise kind {kind!r}: want one of "
                         f"{', '.join(KINDS)}")
    if count is None:
        count = {"partner": 2, "transient_twice": 2}.get(kind, 1)
    return {"kind": kind, "after": after, "skip": skip, "count": count}


def _weights(reg):
    def port(name):
        def w(args, counts):
            if name == "fetch_dish":
                # 0, not KitchenModel's 0.05: the catalog has no way to
                # put a dish down, so a dish fetched with nothing cooking
                # deadlocks the cook for good (found in the hour run)
                return 1.0 if counts.get("cooking", 0) > 0 else 0.0
            if name == "fetch_onion":
                return 0.05 if counts.get("cooking", 0) > 0 else 1.0
            return 1.0
        return w
    for name in EFFECTS:
        if name != "partner_load":
            reg.weight(f"w_{name}", port(name))
    # the sibling card's port (kitchen_shared): a second cook loads
    # rarely next to this one's own moves
    reg.weight("w_partner", 0.1)


class KitchenHost:
    """Scripted orders + a faithful actuator + full sensing + a
    surprise schedule."""

    name = "kitchen"

    def __init__(self, case, plan=(), orders=(0, 14),
                 beats: Optional[int] = 30):
        self.plan = [dict(e) for e in plan]
        self.orders = tuple(orders)     # beats at which an order arrives
        self.beats = beats              # the script's length (None: open)
        self.beat = 0
        self.kitchen: Dict[str, int] = {
            a: n for a, n in dict(case.get("init", {})).items() if n > 0}
        self.glitch = False             # the transient's one misread
        self.injected: List[dict] = []  # what the env did off-model
        self._plan0 = [dict(e) for e in self.plan]

    def schedule(self) -> dict:
        return {"orders": list(self.orders), "beats": self.beats,
                "surprises": [dict(e) for e in self._plan0]}

    # -- the protocol ----------------------------------------------------

    def bind(self, reg):
        _weights(reg)

    def tick(self, counts):
        if self.beats is not None and self.beat >= self.beats:
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
        for e in self.plan:
            if e["count"] <= 0 or e["after"] > self.beat \
                    or TRIGGER[e["kind"]] != name:
                continue
            if e["skip"] > 0:
                e["skip"] -= 1
                continue
            if self._surprise(e["kind"]):
                self.injected.append({"kind": e["kind"], "beat": self.beat,
                                      "intent": name})
                e["count"] -= 1
                if e["kind"] == "partner":   # the next order's first load
                    e["after"] = min((b + 1 for b in self.orders
                                      if b >= self.beat), default=1 << 60)
            break                            # one surprise per event

    def observe(self, counts) -> Dict[str, int]:
        reading = {a: self.kitchen.get(a, 0) for a in ATOMS}
        if self.glitch:
            reading["pot_onion"] += 1
        return reading

    def snapshot(self):
        return (self.beat, dict(self.kitchen), self.glitch,
                [dict(e) for e in self.injected],
                [dict(e) for e in self.plan])

    def restore(self, snap):
        beat, kitchen, glitch, injected, plan = snap
        self.beat, self.kitchen, self.glitch = beat, dict(kitchen), glitch
        self.injected = [dict(e) for e in injected]
        self.plan = [dict(e) for e in plan]

    # -- the injected surprises -----------------------------------------

    def _surprise(self, kind) -> bool:
        k = self.kitchen
        if kind in ("transient", "transient_twice"):
            self.glitch = True
            return True
        if kind in ("wrong_domain", "partner") and k.get("pot_space", 0) > 0:
            k["pot_space"] -= 1
            k["pot_onion"] = k.get("pot_onion", 0) + 1
        elif kind == "novel" and k.get("held_soup", 0) > 0:
            k["held_soup"] -= 1
            k["hand_free"] = k.get("hand_free", 0) + 1
        else:
            return False                     # not possible here: stay armed
        self.kitchen = {a: n for a, n in k.items() if n > 0}
        return True


# -- the fixtures (runtime-wire steps 1-3) ------------------------------------

def make(case):
    return KitchenHost(case)


def make_transient(case):
    return KitchenHost(case, [entry("transient", skip=1)])


def make_wrong_domain(case):
    return KitchenHost(case, [entry("wrong_domain")])


def make_novel(case):
    return KitchenHost(case, [entry("novel")])


def make_transient_twice(case):
    """The misread on the 2nd and 3rd load: a 'transient' that recurs
    after a reset (the bounded-reset fixture)."""
    return KitchenHost(case, [entry("transient_twice", skip=1)])


def make_partner(case):
    """The second cook loads on the first load of each of two orders: an
    explained wrong-domain reading recurs while the served world stays
    the one-cook kitchen (explained, then escalated)."""
    return KitchenHost(case, [entry("partner")])


# -- the hour (runtime-wire step 4) and the eval schedule (step 5) -----------

BEATS_PER_MIN = 120                       # at watch every=0.5


def orders_every(seed, lo_min=2.0, hi_min=3.0, until_min=60.0, first=0):
    """Order beats, one every lo..hi minutes (seeded), from `first`."""
    rng = random.Random(seed)
    out, b = [], first
    while b < until_min * BEATS_PER_MIN:
        out.append(b)
        b += int(rng.uniform(lo_min, hi_min) * BEATS_PER_MIN)
    return out


HOUR_PLAN = [entry("transient", after=5 * BEATS_PER_MIN),
             entry("wrong_domain", after=15 * BEATS_PER_MIN),
             entry("novel", after=25 * BEATS_PER_MIN),
             entry("partner", after=35 * BEATS_PER_MIN, count=3),
             entry("transient_twice", after=45 * BEATS_PER_MIN)]


def make_hour(case):
    """An open-ended kitchen hour (never ends by itself — `watch off`):
    an order every 2-3 min (seed 7) and the five kinds at ~5/15/25/35/45
    min (partner on three successive orders: two recurrences). Pair with
    `watch every=0.5 operator=N` so an escalated surprise is resumed by
    the operator and the rest of the schedule can fire."""
    return KitchenHost(case, HOUR_PLAN, orders_every(7), beats=None)


def make_episode(case):
    """One eval episode (runtime-wire step 5): METIS_KITCHEN_EPISODE is
    its JSON line, from `python -m metis.domains.kitchen.eval episodes`."""
    from .eval import make_episode as _make
    return _make(case)
