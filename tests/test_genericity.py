"""The genericity condition, mechanized: metis.kernel imports nothing
from metis.domains (or any domain-smelling module). Road behavior is an
application; the engine must never know it exists."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "metis"

FORBIDDEN = ("metis.domains", "domains")


def test_kernel_and_lang_are_domain_free():
    for pkg in ("kernel", "lang", "fw"):
        for py in (ROOT / pkg).rglob("*.py"):
            _check(py)


def _check(py):
    if True:
        tree = ast.parse(py.read_text(), filename=str(py))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not any(n == f or n.startswith(f + ".")
                               for f in FORBIDDEN), \
                    f"{py.name} imports domain module {n}"
