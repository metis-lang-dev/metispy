"""Oracle parity: bit-identical measures (<=1e-12) vs the ll2pg reference
in chariot_ws/packages/wayplan — the shared semantic core's referee.
Skips cleanly on hosts without the wayplan checkout. Parity mode: all
clause priors = 1."""

from pytest import approx

from metis.kernel import ForwardProjection
from fixtures import ALL
from oracle import oracle_projection, requires_oracle

pytestmark = requires_oracle

ATOMS = {"race": ["wa", "wb", "a"], "ff": ["p", "q", "a"],
         "persist": ["done", "k"], "staged": ["flag", "d", "t"]}


def test_parity_on_all_fixtures_all_horizons():
    for name, (make, max_steps) in ALL.items():
        for k in range(1, max_steps + 1):
            ours = ForwardProjection(make(), k)
            ref = oracle_projection(make(), k)
            for atom in ATOMS[name]:
                assert ours.final_distribution(atom) == approx(
                    ref.final_distribution(atom), abs=1e-12), \
                    (name, k, atom)
            assert ours.stage_distribution() == approx(
                ref.stage_distribution(), abs=1e-12), (name, k)
            assert ours.p_done() == approx(ref.p_done(), abs=1e-12)
            assert ours.n_reachable() == ref.n_reachable()
