"""Partner-posterior gates (the stage-5 hook, attached): online
style identification from observed intents (planted-style recovery,
never actively wrong); the strategist's choice adapts to COMPLEMENT
an identified loader (flips off the onion side) and does NOT flip
against a server; posterior feedback is the next prior. Abstract
level — no Overcooked install needed."""

import random

from metis.domains.kitchen.model import KitchenModel, abstract
from metis.domains.kitchen.partner import PartnerPosterior


def default_durations():
    # agent.default_durations, inlined so these gates never import
    # the Overcooked bridge
    return {"fetch_onion": 6.0, "load": 4.0, "cook": 1.0,
            "fetch_dish": 8.0, "plate": 22.0, "serve": 6.0,
            "penalty": 120.0}


def scenes():
    """Abstract states a partner acts from (varied so likelihoods
    are informative across the intent vocabulary)."""
    return [abstract(0, False, False, "none"),
            abstract(2, False, False, "onion"),
            abstract(0, True, False, "none"),
            abstract(0, True, True, "dish"),
            abstract(1, False, False, "none")]


def identify(true_style, n=25, seed=9):
    pp = PartnerPosterior()
    rng = random.Random(seed)
    sc = scenes()
    for k in range(n):
        init = sc[k % len(sc)]
        intent = pp.sample(true_style, init, rng)
        if intent is not None:
            pp.observe(init, intent)
    return pp


def test_planted_style_identified_and_never_actively_wrong():
    for style in ("loader", "server"):
        finals = []
        for seed in (9, 23, 41):
            pp = identify(style, seed=seed)
            finals.append(pp.post[style])
        assert min(finals) > 0.5           # never actively wrong
        assert max(finals) > 0.9           # and typically decisive
        assert sum(f > 0.85 for f in finals) >= 2


def test_strategist_complements_an_identified_loader():
    """Empty pot, hand free: alone the measure fetches onions; with
    a partner the posterior has pinned as a LOADER, the onion side
    is theirs — the strategist flips to the dish side. Against a
    SERVER partner it keeps the onions (no false flip)."""
    model = KitchenModel(default_durations())
    init = abstract(0, False, False, "none")
    alone = model.choose(init).action
    assert alone == "fetch_onion"

    loader = identify("loader")
    adapted = model.choose(init,
                           partner_next=loader.p_next(init)).action
    assert adapted == "fetch_dish", adapted

    server = identify("server")
    kept = model.choose(init,
                        partner_next=server.p_next(init)).action
    assert kept == "fetch_onion", kept


def test_posterior_is_the_next_prior():
    pp = identify("loader", n=6)
    mid = dict(pp.post)
    pp2 = PartnerPosterior(prior=mid)
    assert pp2.post == mid
    rng = random.Random(3)
    init = scenes()[0]
    pp2.observe(init, pp2.sample("loader", init, rng))
    assert pp2.post["loader"] >= mid["loader"] - 1e-9
