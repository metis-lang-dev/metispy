"""Catalog cards (runtime-answers §7, runtime-wire step 3): the domains
offerable at the served node, GENERATED from what every catalog already
carries — never authored:

  text      its `provenance text "..."` line (the one-line purpose)
  relation  `extends <served>` (a pack: admitted over the served world)
            or `sibling` (another catalog including the same base file
            as the served scene — an alternative world, not an addition)
  moves     the ground events the card adds over the served world — the
            only moves an explanation may name beside the fired path
  slots     the card's own declared types (binding slots for activate)

Discovery is structural (extends-pruning): the served file's folder,
`.llp` files with a provenance text line that either extend the served
catalog or share its include. The served catalog is listed first as the
active card."""

import re
from pathlib import Path
from typing import Dict, List, Optional

_CATALOG = re.compile(r"^\s*catalog\s+(\w+)", re.M)
_TEXT = re.compile(r'^\s*provenance\s+text\s+"([^"]*)"', re.M)
_EXTENDS = re.compile(r"^\s*extends\s+(\w+)\s*\.", re.M)
_INCLUDE = re.compile(r'^\s*include\s+"([^"]+)"\s*\.', re.M)
_TYPE = re.compile(r"^\s*type\s+(\w+)", re.M)


def _scan(path: Path) -> Optional[dict]:
    try:
        src = path.read_text()
    except OSError:
        return None
    m = _CATALOG.search(src)
    if not m:
        return None
    return {"name": m.group(1), "file": str(path),
            "text": (_TEXT.search(src) or [None, None])[1],
            "extends": (_EXTENDS.search(src) or [None, None])[1],
            "includes": [str((path.parent / i).resolve())
                         for i in _INCLUDE.findall(src)],
            "types": _TYPE.findall(src)}


def card_program(shell, card: dict, init: Dict[str, int], stage: str):
    """The card's world grounded at (init, stage): a sibling stands
    alone, a pack is admitted over the served catalog."""
    from metis.lang import load_llp
    if card["relation"] == "served":
        cat = shell.doc.catalog
    elif card["relation"] == "extends":
        cat = shell.doc.catalog.admit(
            load_llp(card["file"], base=shell.doc).catalog)
    else:
        cat = load_llp(card["file"]).catalog
    return cat, cat.ground(dict(init), init_stage=stage)


def cards(shell) -> List[dict]:
    served = _scan(Path(shell.path).resolve())
    if served is None:
        return []
    own = {ev.name for _, _, ev in shell.events_tbl}
    out = [{"name": shell.doc.catalog.name, "file": served["file"],
            "text": served["text"], "relation": "served", "moves": [],
            "slots": served["types"]}]
    for f in sorted(Path(served["file"]).parent.glob("*.llp")):
        c = _scan(f.resolve())
        if c is None or c["file"] == served["file"] or not c["text"]:
            continue
        if c["extends"] == shell.doc.catalog.name:
            rel = "extends"
        elif set(c["includes"]) & set(served["includes"]):
            rel = "sibling"
        else:
            continue
        card = {"name": c["name"], "file": c["file"], "text": c["text"],
                "relation": rel, "slots": c["types"]}
        try:
            _, prog = card_program(shell, card, shell.case["init"],
                                   shell.prog.init_stage)
            card["moves"] = sorted({ev.name for _, _, ev in prog.events()}
                                   - own)
        except Exception as e:            # a broken card is a finding
            card["moves"], card["error"] = [], f"{type(e).__name__}: {e}"
        out.append(card)
    return out


def card_line(c: dict) -> str:
    rel = {"served": "active", "extends": "pack",
           "sibling": "sibling"}[c["relation"]]
    moves = (" moves " + ", ".join(c["moves"])) if c["moves"] else ""
    return f'{c["name"]} [{rel}] "{c["text"] or ""}"{moves}'
