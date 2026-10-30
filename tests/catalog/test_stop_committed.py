"""The committed action (metis-lang spec/15 amendment 9, wc-member s-4fe6cfd6): the scenes of metis-lang's gate 22
(corpus/stop) on metispy's sampler, the same counts per seed, over the vendored standard unit."""
import io
import re

import pytest

from langpath import LANG
from metis.lang.repl import Repl

STOP = LANG / "corpus" / "stop"
SEEDS = 10
pytestmark = pytest.mark.skipif(not STOP.is_dir(), reason=f"no metis-lang corpus/stop at {LANG} (METIS_LANG)")


def counts(scene, seed):
    repl = Repl(STOP / ("stop_%s.llp" % scene), seed=seed, stdout=io.StringIO())
    rules = [re.split(r"[\[(]", e)[0] for e in repl.run_steps(20)["fired"]]
    return rules.count


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
def test_a_stop_leaves_a_committed_holder_alone(seed):
    n = counts("committed", seed)
    assert n("stop_committed") == n("stop_dequeue") == 1 and n("stop_release") == n("grant_on_want") == 0


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
def test_its_yield_is_the_revoke_and_the_next_ticket_is_served(seed):
    n = counts("yield", seed)
    assert n("yield_owing") == n("owed_release") == n("grant_on_want") == 1 and n("stop_release") == 0


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
def test_a_yield_with_no_stop_keeps_the_resource(seed):
    n = counts("yield_nostop", seed)
    assert n("commit_off") == 1 and n("owed_release") == n("yield_owing") == 0


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
def test_preemption_beats_a_commitment(seed):
    assert counts("preempt", seed)("preempt") == 1


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
def test_a_commit_and_a_stop_in_one_beat_are_one_or_the_other(seed):
    n = counts("race", seed)
    assert (n("stop_release"), n("stop_committed")) in ((1, 0), (0, 1))
    assert n("stop_committed") == 0 or n("commit_on") == 1


@pytest.mark.parametrize("seed", range(1, SEEDS + 1))
@pytest.mark.parametrize("scene,dequeues", [("waiter", 1), ("nowaiter", 0)])
def test_a_stop_with_no_commitment_is_as_before(scene, dequeues, seed):
    n = counts(scene, seed)
    assert n("stop_release") == 1 and n("stop_dequeue") == dequeues and n("grant_on_want") == 0
