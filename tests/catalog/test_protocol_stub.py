"""metis.catalog.protocol_stub (specialist-tree s-404d001f step 3c; spec/15 amendment 7): a member's cancel rules
generated from its own rules and goal kinds — one per minimal state it can hold while it works, cancel_none for a
member with idle (the done-and-stop race), the task's (no idle) giving back `one`; the refusals by name. Moved
from metis-worlds' tools/, so the members here are generic, no world's."""
import pytest

from metis.catalog import protocol_stub as ps

WALK = """catalog walk 1.
include "iface/protocol_domain.llp".
layers (dyn).
stages (act).
namespace walk produce (dyn).
type domain {walk}.
type resource {motion}.
type place {p1 p2}.
pred goto(place) : walk.
pred idle : walk.
pred busy : walk.
pred moving(place) : walk.
stage act {
  %% I take my goal.
  begin [dyn] : $goto(X) * idle -o busy * tick__active(walk).
  %% I move toward it.
  go [dyn] : $goto(X) * busy * $tick__grant(walk, motion) -o busy * moving(X).
  %% I am there.
  arrive [dyn] : moving(X) * goto(X) * busy -o idle * tick__done(walk).
}
"""

TASK = """catalog task 1.
include "iface/protocol_domain.llp".
layers (dyn).
stages (act).
namespace task produce (dyn).
type domain {task}.
type place {p1 p2}.
pred reach(place) : task.
pred done : task.
pred asked(place) : task.
stage act {
  %% I ask for the place.
  ask [dyn] : reach(X) -o asked(X).
  %% It is reached.
  got [dyn] : asked(X) -o done.
}
"""


def test_a_member_with_idle_cancels_its_goal_and_absorbs_the_race():
    assert ps.states(WALK, "walk", ["goto"]) == [("busy", "goto")]       # moving holds busy*goto: left out (one cancel)
    assert [l for l in ps.cancel_rules(WALK, "walk", ["goto"]) if "[dyn]" in l] == [
        "  cancel_goto [dyn] : tick__cancel(walk) * busy * goto(V0) -o idle.",
        "  cancel_none [dyn] : idle * tick__cancel(walk) -o idle.   "
        "% the race: its done and the stop in one director beat (path_pop, stop_cancel: the sampled order)"]


def test_the_task_has_no_idle_and_gives_back_one():
    assert ps.cancel_rules(TASK, "task", ["reach"], ends=("done",)) == [
        "  %% Stopped: I drop my goal.", "  cancel_asked [dyn] : tick__cancel(task) * asked(V0) -o one."]


def test_the_cancels_load_beside_the_begin(tmp_path):
    from metis.lang.compiler import load_llp
    lines = WALK.split("\n")
    i = next(k for k, l in enumerate(lines) if l.lstrip().startswith("begin"))
    p = tmp_path / "walk.llp"
    p.write_text("\n".join(lines[:i + 1] + ps.cancel_rules(WALK, "walk", ["goto"]) + lines[i + 1:]))
    rules = {e.name.split("/")[-1] for e in load_llp(p).catalog.entries}
    assert {"begin", "cancel_goto", "cancel_none", "go", "arrive"} <= rules


def test_a_goal_no_rule_takes_is_refused_by_name():
    with pytest.raises(ps.StubError, match="task: its goal kinds fetch reach no state"):
        ps.cancel_rules(TASK, "task", ["fetch"], ends=("done",))


def test_a_tracked_token_with_no_declaration_is_refused_by_name():
    text = WALK.replace("pred goto(place) : walk.\n", "")             # the goal kind's arity is the family package's
    with pytest.raises(ps.StubError, match="walk: no declaration for goto"):
        ps.cancel_rules(text, "walk", ["goto"])
    assert ps.cancel_rules(text, "walk", ["goto"], decls="pred goto(place) : nav.\n")


LIFTED = """catalog worker_domain 1.
include "iface/protocol_domain.llp".
layers (dyn).
stages (act).
namespace worker produce (dyn).
type place {p1 p2}.
bwd self(domain).
pred job : worker.
pred calm(domain) : worker.
pred asked(domain) : worker.
pred bench(place) : worker.
stage act {
  %% I need the bench: I ask.
  ask [dyn] : $job * $self(S) * calm(S) -o asked(S) * tick__want(S, motion) * tick__active(S).
  %% Done at the bench.
  finish [dyn] : job * asked(S) * $self(S) -o calm(S) * tick__done(S).
}
"""


def test_a_lifted_member_binds_its_own_tokens_to_the_binder():
    # the base several members instantiate (its own namespace, the member a variable bound by self): calm its idle;
    # a token of the member's type takes the binder's variable (asked(S)), never a fresh one over the domains
    rules = ps.cancel_rules(LIFTED, "S", [], idle="calm", namespace="worker", bind="$self(S)", world=["job"])
    assert [l for l in rules if "[dyn]" in l] == [
        "  cancel [dyn] : tick__cancel(S) * $self(S) * asked(S) -o calm(S).",
        "  cancel_none [dyn] : calm(S) * tick__cancel(S) * $self(S) -o calm(S).   "
        "% the race: its done and the stop in one director beat (path_pop, stop_cancel: the sampled order)"]
    assert ps.refresh(ps.refresh(LIFTED, "S", idle="calm", namespace="worker", bind="$self(S)", world=["job"]),
                      "S", idle="calm", namespace="worker", bind="$self(S)", world=["job"]).count(ps.BEGIN) == 1
