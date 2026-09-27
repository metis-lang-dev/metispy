"""The metis Overcooked agent: GreedyHumanModel with its INTENT CHOICE
replaced by the catalog-projected measure; all motion execution
inherited (their MLAM plans, their low-level controller)."""

import numpy as _np

for _name, _val in (("Inf", _np.inf), ("NaN", _np.nan)):
    if not hasattr(_np, _name):
        setattr(_np, _name, _val)   # overcooked-ai predates numpy 2

from overcooked_ai_py.agents.agent import GreedyHumanModel  # noqa: E402

from .model import KitchenModel, abstract                   # noqa: E402


def default_durations():
    """Rough cramped_room travel costs in env steps; the penalty is
    the cost of an unfinished recipe at the projection horizon."""
    return {"fetch_onion": 6.0, "load": 4.0, "cook": 1.0,
            "fetch_dish": 8.0, "plate": 22.0, "serve": 6.0,
            "penalty": 120.0}   # must EXCEED full-recipe completion cost
            # (~67), else stalling beats cooking — found by the
            # intent-ordering gate


class MetisStrategist(GreedyHumanModel):
    def __init__(self, mlam, durations=None, partner=None):
        """`partner`: an optional kitchen.partner.PartnerPosterior;
        feed it observed partner intents via observe_partner() from
        the env loop, and choices price overlap against its
        prediction (classification of the live partner's intent from
        state deltas is the caller's transport concern)."""
        super().__init__(mlam)
        self.model = KitchenModel(durations or default_durations())
        self.partner = partner

    def observe_partner(self, init, intent):
        if self.partner is not None and intent is not None:
            self.partner.observe(init, intent)

    def ml_action(self, state):
        player = state.players[self.agent_index]
        am = self.mlam
        counter_objects = am.mdp.get_counter_objects_dict(
            state, list(am.mdp.terrain_pos_dict["X"]))
        pot_states = am.mdp.get_pot_states(state)

        n_onions = len(pot_states.get("1_items", [])) \
            + 2 * len(pot_states.get("2_items", [])) \
            + 3 * len(pot_states.get("3_items", []))
        cooking = bool(pot_states.get("cooking"))
        ready = bool(pot_states.get("ready"))
        held = player.get_object().name if player.has_object() \
            else "none"
        if held == "soup":
            held = "soup"
        init = abstract(n_onions, cooking, ready, held)
        pn = self.partner.p_next(init) if self.partner else None
        res = self.model.choose(init, partner_next=pn)
        intent = res.action if res else None

        goals = []
        if intent == "fetch_onion":
            goals = am.pickup_onion_actions(counter_objects)
        elif intent == "load":
            goals = am.put_onion_in_pot_actions(pot_states)
        elif intent == "cook":
            goals = am.start_cooking_actions(pot_states)
        elif intent == "fetch_dish":
            goals = am.pickup_dish_actions(counter_objects)
        elif intent == "plate":
            goals = am.pickup_soup_with_dish_actions(pot_states,
                                                     only_nearly_ready=True)
        elif intent == "serve":
            goals = am.deliver_soup_actions()
        if not goals:
            # measure said something MLAM can't ground right now (e.g.
            # plate before the pot exists) — fall back to the parent's
            # heuristics rather than stall
            return super().ml_action(state)
        return goals
