"""The mini road catalog — LOADERS over the canonical .llp text.

The rules live in catalogs/mini.llp and catalogs/wheelchair.llp
(stage 41 standardization: the .llp + bindings format is the ONLY
rule format — the old hand-built ClauseSchema construction this
module used to carry is retired; it was not consumable by the OCaml
toolchain and duplicated the text). The API is unchanged:
base_catalog() / wheelchair_pack() return kernel Catalogs,
scenario() builds the demo init. Containment doctrine unchanged:
the pack owns only its namespace; admit()+typing guarantee it."""

from functools import lru_cache
from pathlib import Path

from metis.kernel import Catalog, pat
from metis.lang import load_llp

_CATALOGS = Path(__file__).parent / "catalogs"


@lru_cache(maxsize=None)
def _base_doc():
    return load_llp(_CATALOGS / "mini.llp")


def base_catalog() -> Catalog:
    return _base_doc().catalog


def wheelchair_pack() -> Catalog:
    """A well-typed pack: OWN namespace/predicates, reads kinematic
    state (persist — always allowed), produces only its own atoms.
    Cannot touch kinematics — admit()+typing guarantee it."""
    return load_llp(_CATALOGS / "wheelchair.llp",
                    base=_base_doc()).catalog


def scenario(ego=("l0", "c1", "v1"), alfa=("l0", "c2", "v0"),
             wc=False):
    """Initial atoms for the one-round demo scenario."""
    init = [pat("eturn"),
            pat("st", "ego", *ego),
            pat("ot", "alfa"),
            pat("st", "alfa", *alfa)]
    if wc:
        init.append(pat("wc_scan"))
    return init
