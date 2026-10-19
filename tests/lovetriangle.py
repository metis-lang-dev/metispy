"""The Ceptre love triangle for the tests: metis-catalog's port,
domains/ceptre/love_triangle/love_triangle.llp. It is under interactive-lp's
Anti-Capitalist Software License v1.4 (Chris Martens), not metispy's
Apache-2.0 grant, so metispy no longer ships it (ceptre-carveout
s-76664d26): the lineage tests read the sibling checkout ($METIS_CATALOG,
else ../metis-catalog) and skip without it."""
import os
from functools import lru_cache
from pathlib import Path

import pytest

CATALOG = Path(os.environ.get(
    "METIS_CATALOG",
    Path(__file__).resolve().parents[2] / "metis-catalog")).resolve() \
    / "domains/ceptre/love_triangle/love_triangle.llp"

# the tutorial's init context (the program's own #init lines)
INIT = {"eros(helena,hermia)": 1, "eros(hermia,helena)": 1,
        "eros(helena,lysander)": 1, "eros(lysander,hermia)": 1}

needs = pytest.mark.skipif(not CATALOG.is_file(),
                           reason="no metis-catalog checkout (the love triangle is its Ceptre port)")


def require():
    """The program's path, or skip the calling test."""
    if not CATALOG.is_file():
        pytest.skip("no metis-catalog checkout (the love triangle is its Ceptre port)")
    return CATALOG


@lru_cache(maxsize=None)
def love_triangle():
    from metis.lang import load_llp
    return load_llp(require())


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
