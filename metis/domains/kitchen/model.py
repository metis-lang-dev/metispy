"""Overcooked strategic layer: choose the next medium-level intent by
projecting the recipe catalog forward and minimizing expected total
DURATION to the next served soup (durations = real MLAM motion-plan
costs, bound as weight ports at zero model cost — the BT.cpp move once
more).

Remaining limitation, honest: cooking time is folded into the plate
duration. The v1 partner gap is CLOSED (stage 30): partner.py holds
the online posterior over partner styles, and choose() prices
overlap against its prediction — the stage-5 hook, attached where
the v1 note said it would go."""

from pathlib import Path
from typing import Dict

from metis.kernel import ForwardProjection, plan
from metis.lang import load_llp, weight_policy

CATALOG = Path(__file__).parent / "catalogs" / "kitchen.llp"

INTENTS = ("fetch_onion", "load", "cook", "fetch_dish", "plate",
           "serve")


def abstract(pot_onions: int, cooking: bool, ready: bool,
             held: str) -> Dict[str, int]:
    """The abstract scene -> initial multiset. `cooking` here means
    soup underway OR ready (plate covers both; MLAM waits as needed)."""
    init = {}
    init["hand_free" if held == "none" else f"held_{held}"] = 1
    if cooking or ready:
        init["cooking"] = 1
    else:
        if pot_onions:
            init["pot_onion"] = pot_onions
        if pot_onions < 3:
            init["pot_space"] = 3 - pot_onions
    return init


class KitchenModel:
    def __init__(self, durations: Dict[str, float]):
        self.doc = load_llp(CATALOG)
        self.durations = dict(durations)

    def _policy(self, prog, first: str = None):
        """Uniform CHOICE over enabled tasks; a forced FIRST intent
        gets weight 1 and everything else 0 until it fires (tracked by
        a counter closure — one-shot forcing without touching state)."""
        forced = {"pending": first is not None}

        def base_weight(name, counts):
            """The base task policy AS weights (state-aware ports):
            while the soup cooks, dishes dominate; otherwise onions.
            Small (not zero) off-weights keep dead-end branches priced
            instead of stalling the CHOICE — found when uniform
            branching let cheap early-stall paths beat completions."""
            if name == "fetch_dish":
                return 1.0 if "cooking" in counts else 0.05
            if name == "fetch_onion":
                return 0.05 if "cooking" in counts else 1.0
            return 1.0

        def port(name):
            def w(args, counts):
                if forced["pending"]:
                    return 1.0 if name == first else 0.0
                return base_weight(name, counts)
            return w

        bindings = {f"w_{i}": port(i) for i in INTENTS}
        pol = weight_policy(prog, self.doc.ports, bindings)

        def policy(stage, counts, ws):
            out = pol(stage, counts, ws)
            if forced["pending"] and out:
                forced["pending"] = False
            return out

        return policy

    def expected_duration(self, first: str, init: Dict[str, int],
                          steps: int = 12) -> float:
        prog = self.doc.catalog.ground(dict(init))
        pr = ForwardProjection(prog, steps,
                               policy=self._policy(prog, first))
        # E[total duration] = sum over slices of event-probability x
        # that event's duration (names from the event table)
        names = [ev.name for _, _, ev in prog.events()] + ["(done)"]
        total = 0.0
        for t in range(steps):
            for e, p in enumerate(pr.event_distribution(t)):
                if p > 0 and e < len(names) - 1:
                    total += p * self.durations.get(names[e], 1.0)
        # unfinished recipes at the horizon pay a completion penalty
        p_served = sum(p for (_, k), p in pr.final.items()
                       if dict(k).get("served"))
        total += (1.0 - p_served) * self.durations.get("penalty", 30.0)
        return total

    def choose(self, init: Dict[str, int], partner_next=None,
               clash: float = 40.0):
        """`partner_next` (the partner posterior's p_next) prices
        OVERLAP: a candidate the partner is about to take pays
        clash x P(partner takes it) inside the same argmin —
        complementary division of labor from the measure, no rules
        added. Alone (partner_next=None) this is v1 unchanged.
        clash=40 on the default duration scale (penalty 120, full
        recipe ~67): a duplicated side costs the wasted fetch+load
        leg PLUS the delay of the uncovered complementary side —
        tuned by the complement-flip gate, the same status as
        penalty=120 (tuned by the intent-ordering gate)."""
        prog = self.doc.catalog.ground(dict(init))
        enabled = []
        from metis.kernel import transition_count
        for kind, stage, ev in prog.events():
            if kind == "clause" and \
                    transition_count(ev, init) > 0 and \
                    ev.name in INTENTS:
                enabled.append(ev.name)
        if not enabled:
            return None
        overlap = partner_next or {}
        return plan(enabled, do=lambda i: i, project=lambda i: i,
                    score=lambda i: (self.expected_duration(i, init)
                                     + clash * overlap.get(i, 0.0),),
                    doomed_action=None)
