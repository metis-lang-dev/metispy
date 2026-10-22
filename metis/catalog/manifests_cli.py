"""python -m metis_catalog list [--json] — what the consoles show (the loader's CLI, moved
from metis_catalog/__main__.py; metis-packages s-65778b98)."""
import argparse
import json
import sys

from .manifests import index, root


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metis_catalog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="every catalog: the domains first, with their display line and what they have")
    ls.add_argument("--json", action="store_true")
    ls.add_argument("--domains", action="store_true", help="the domains only")
    a = ap.parse_args(argv)
    rows = [e for e in index() if not a.domains or e["kind"] == "domain"]
    if a.json:
        json.dump(rows, sys.stdout, indent=1, default=str)
        print()
        return 0
    print("# %s" % root())
    for e in rows:
        has = " ".join(k for k, v in e["has"].items() if v and k != "program")
        print("%-7s %-16s %-44s %-28s %s" % (e["kind"], e["name"], e.get("path") or "—", has, e["line"][:70]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
