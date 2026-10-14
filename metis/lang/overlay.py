"""A fitted notch overlay as a file (the recorder's --overlay format, as
notch writes it): {"rows": [{port, args, row, notched}]} -> per port a
weight function of (args, counts), keyed by the args and the read-scope
presence row. Read by the REPL's `notch load` and by the recorder."""
import json
from pathlib import Path
from typing import Dict


def overlay_table(path):
    """A notch table file -> {port: fn(args, counts)} overlays keyed by
    (args, read-scope presence row)."""
    rows = json.loads(Path(path).read_text())["rows"]
    tables: Dict[str, Dict] = {}
    for r in rows:
        key = (tuple(r["args"]), tuple(tuple(x) for x in r["row"]))
        tables.setdefault(r["port"], {})[key] = float(r["notched"])

    def make(t):
        def ov(args, counts):
            for (a, row), v in t.items():
                if a == tuple(args) and all(
                        (counts.get(x, 0) > 0) == bool(b) for x, b in row):
                    return v
            return None
        return ov
    return {f: make(t) for f, t in tables.items()}
