"""The routing director's lifted base in the iface (policy-tuning-2 s-e075f47f 2a, the architect's ruling): the generic
director a composition includes by name (`include "iface/director_domain.llp".`), taken verbatim from metis-catalog's
domains/director. Until the hand trees include it too, the two copies must not drift."""
import pytest

from metis.catalog import iface
from catalogs import installed_catalog


def test_the_iface_has_the_routing_base():
    text = iface.path("director_domain.llp").read_text()
    assert "bwd handles(domain, kind)." in text and "bwd planner(domain)." in text


def test_the_iface_copy_is_the_catalog_copy():
    root = installed_catalog()
    cat = root / "domains" / "director" / "director_domain.llp" if root else None
    if cat is None or not cat.is_file():
        pytest.skip("no metis-catalog here (METIS_CATALOG or an installed metis-catalog)")
    assert iface.path("director_domain.llp").read_bytes() == cat.read_bytes()
