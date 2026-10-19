"""A wheel install carries every data file the package reads (catalog-packages
s-a2c529d6: a non-editable install could not build the kitchen's bundle, its
catalogs/*.llp were not in the wheel). The wheel is built from the checkout
and every tracked non-.py file under metis/ must be in it."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_wheel_carries_every_data_file(tmp_path):
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "metis"], capture_output=True, text=True)
    if tracked.returncode:
        pytest.skip("not a git checkout")
    data = {f for f in tracked.stdout.split() if not f.endswith(".py")} - {"metis/domains/road/README.md"}
    assert data, "no data files: the listing is wrong"
    src = tmp_path / "src"                       # built from a copy: the checkout gets no build/ or egg-info
    files = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    for f in files:
        if f.startswith("metis/") or f in ("pyproject.toml", "README.md", "LICENSE", "setup.py", "setup.cfg"):
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / f, src / f)
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(tmp_path), str(src)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    whl, = tmp_path.glob("metispy-*.whl")
    names = set(zipfile.ZipFile(whl).namelist())
    assert sorted(data - names) == []
