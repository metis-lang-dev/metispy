#!/usr/bin/env bash
# smoke_wheel.sh DIST — the built wheel in DIST installed into a fresh venv
# outside any checkout, and run there: the package imports from site-packages,
# the `metis` CLI answers, the package data is in (kitchen's catalogs), a
# catalog compiles and grounds, and a domain NAME without metis-catalog fails
# with the clear message. CI (ci.yml) and the release (publish.yml) run it.
set -euo pipefail
DIST=$(cd "$1" && pwd)
W=$(mktemp -d)
trap 'rm -r "$W"' EXIT
"${PYTHON:-python3}" -m venv "$W/venv"
"$W/venv/bin/pip" install -q "$DIST"/*.whl
cd "$W"                                   # no checkout on the path
env -u METIS_CATALOG -u PYTHONPATH "$W/venv/bin/python" - <<'PY'
import importlib.resources as r

import metis
assert "site-packages" in metis.__file__, metis.__file__
cats = sorted(p.name for p in (r.files("metis.domains.kitchen") / "catalogs").iterdir())
assert "kitchen_host.llp" in cats, cats
from metis.lang import compile_catalog, parse
doc = compile_catalog(parse("""catalog smoke 1.
layers (dyn).
stages (main).
namespace s produce (dyn).
pred a : s.
pred b : s.
stage main {
  %% a becomes b.
  ab [dyn] : a -o b.
}
"""))
assert doc.catalog.ground({"a": 1}).events()
from metis.kernel.artifact import _catalog
try:
    _catalog()
except RuntimeError as e:
    assert "metis-catalog" in str(e) and "METIS_CATALOG" in str(e), e
else:
    raise AssertionError("a domain name without metis-catalog did not fail")
print("smoke: metis %s from %s, %d kitchen catalogs" % (metis.__name__, metis.__file__, len(cats)))
PY
env -u METIS_CATALOG "$W/venv/bin/metis" --help > /dev/null
env -u METIS_CATALOG "$W/venv/bin/metis" --version
echo "SMOKE GREEN"
