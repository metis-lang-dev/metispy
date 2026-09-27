"""The partner-posterior overlay — the designated stage-5 hook,
attached (model.py's v1 note: "a partner-help weight port fed by an
online posterior over partner styles").

A partner STYLE is a hypothesis measure over the SAME abstract
recipe machine: a weight row over the intent ports. The posterior
over styles updates online from observed partner intents — the
likelihood P_h(intent | abstract state) is read straight off the
style-weighted projection's slice-0 event distribution (model
filtering, the behavior_posterior pattern with event-level
evidence). The strategist adapts by pricing OVERLAP: a candidate
intent the partner is about to take pays a clash penalty inside the
same expected-duration argmin — complementary division of labor
falls out of the measure, no rules added.

Env integration note: classifying the live partner's intent from
Overcooked state deltas is transport, not semantics — the overlay
below is complete at the abstract level and gated there."""

import random
from typing import Dict, Optional

from metis.kernel import ForwardProjection
from metis.lang import load_llp, weight_policy

from .model import CATALOG, INTENTS

STYLES: Dict[str, Dict[str, float]] = {
    # onion-side worker: fetches and loads, rarely touches dishes
    "loader": {"fetch_onion": 1.0, "load": 1.0, "cook": 0.6,
               "fetch_dish": 0.05, "plate": 0.05, "serve": 0.05},
    # dish-side worker: plates and serves, rarely loads
    "server": {"fetch_onion": 0.05, "load": 0.05, "cook": 0.6,
               "fetch_dish": 1.0, "plate": 1.0, "serve": 1.0},
    # no discernible preference
    "uniform": {i: 1.0 for i in INTENTS},
}


class PartnerPosterior:
    def __init__(self, styles: Optional[dict] = None,
                 prior: Optional[Dict[str, float]] = None):
        self.doc = load_llp(CATALOG)
        self.styles = dict(styles or STYLES)
        n = len(self.styles)
        self.post = dict(prior or {h: 1.0 / n for h in self.styles})

    def _dist(self, style: str,
              init: Dict[str, int]) -> Dict[str, float]:
        """P_style(intent | abstract state): slice-0 event
        distribution of the style-weighted one-step projection."""
        row = self.styles[style]
        bindings = {f"w_{i}": (lambda a, c, _v=row[i]: _v)
                    for i in INTENTS}
        prog = self.doc.catalog.ground(dict(init))
        pr = ForwardProjection(
            prog, 1, policy=weight_policy(prog, self.doc.ports,
                                          bindings))
        names = [ev.name for _, _, ev in prog.events()]
        out: Dict[str, float] = {}
        for e, p in enumerate(pr.event_distribution(0)):
            if p > 0 and e < len(names) and names[e] in INTENTS:
                out[names[e]] = out.get(names[e], 0.0) + p
        return out

    def observe(self, init: Dict[str, int], intent: str):
        """One Bayes step from an observed partner intent; the
        posterior is the next observation's prior (the stage-5
        feedback loop)."""
        liks = {h: self._dist(h, init).get(intent, 0.0)
                for h in self.styles}
        z = sum(self.post[h] * liks[h] for h in self.styles)
        if z <= 0.0:
            return                      # unexplained by every style
        self.post = {h: self.post[h] * liks[h] / z
                     for h in self.styles}

    def p_next(self, init: Dict[str, int]) -> Dict[str, float]:
        """Posterior-mixture prediction of the partner's next
        intent — what the strategist prices as overlap."""
        out: Dict[str, float] = {}
        for h, ph in self.post.items():
            if ph <= 0.0:
                continue
            for i, p in self._dist(h, init).items():
                out[i] = out.get(i, 0.0) + ph * p
        return out

    def sample(self, style: str, init: Dict[str, int],
               rng: random.Random) -> Optional[str]:
        """Draw a partner intent from a TRUE style — the gates'
        world simulator."""
        dist = self._dist(style, init)
        if not dist:
            return None
        x = rng.random() * sum(dist.values())
        for i, p in dist.items():
            x -= p
            if x <= 0:
                return i
        return next(iter(dist))
