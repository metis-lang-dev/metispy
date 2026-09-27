"""The love-triangle story world (port of the Ceptre tutorial).

No registries, no ports: the origin program is pure — priors are 1
(PARITY MODE: the measure is then bit-identical to the ceptre/ll2pg
choice semantics, per the kernel doctrine) and drama management is a
POLICY HOOK reweighting the same measure, keeping inference exact."""

from functools import lru_cache
from pathlib import Path

from metis.lang import load_llp

CATALOG = Path(__file__).parent / "catalogs" / "love_triangle.llp"

# the tutorial's init context, verbatim
INIT = {"eros(helena,hermia)": 1, "eros(hermia,helena)": 1,
        "eros(helena,lysander)": 1, "eros(lysander,hermia)": 1}


@lru_cache(maxsize=None)
def love_triangle():
    return load_llp(CATALOG)


def drama_policy(program, factor: float):
    """The drama manager (ARCHITECTURE: 'drama managers are
    reweightings of the same measure'): scale every move_on event's
    CHOICE weight — factor < 1 prolongs the tangle, > 1 hurries the
    heartbreak. Exactness is untouched; only the measure shifts."""
    idx = {i for i, (kind, _s, ev) in enumerate(program.events())
           if kind == "clause" and ev.name.startswith("move_on")}

    def policy(stage, counts, ws):
        return [(i, w * factor if i in idx else w) for i, w in ws]

    return policy
