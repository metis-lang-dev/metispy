"""Locate the metis-lang checkout (metisc + the C++ runtime + the
committed corpus) that this referee generates for and cross-checks.
Default: the sibling checkout; override with METIS_LANG."""
import os
from pathlib import Path

LANG = Path(os.environ.get(
    "METIS_LANG",
    Path(__file__).resolve().parents[2] / "metis-lang")).resolve()
