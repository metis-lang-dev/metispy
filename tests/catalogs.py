"""The catalog the moved catalog tests read (metis-packages s-65778b98): $METIS_CATALOG, else the
installed metis-catalog's own (its checkout, installed -e, or the wheel's domains). None when
neither: the tests that need domains skip."""
import os
from pathlib import Path


def installed_catalog():
    env = os.environ.get("METIS_CATALOG")
    if env:
        return Path(env).resolve()
    from metis.catalog import manifests
    own = manifests.data_dir()
    if own is None:
        return None
    return own if (own / "domains").is_dir() else own.parent
