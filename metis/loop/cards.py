"""Catalog cards (runtime-answers §7, runtime-wire step 3): the domains
offerable at the served node, GENERATED from what every catalog already
carries — never authored:

  text      its `provenance text "..."` line (the one-line purpose)
  relation  `extends <served>` (a pack: admitted over the served world)
            or `sibling` (another catalog including the same base file
            as the served scene — an alternative world, not an addition)
  moves     the ground events the card adds over the served world — the
            only moves an explanation may name beside the fired path
  effects   each move's declared consume/produce/persist (the event
            signals' shape): what a reading's diff is matched against
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
            "effects": [],
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
            evs = {}
            for _, _, ev in prog.events():
                if ev.name not in own:
                    evs.setdefault(ev.name, ev)
            card["moves"] = sorted(evs)
            # each move's declared effect, the event signals' shape:
            # what a reading's diff can be matched against
            card["effects"] = [{"name": n,
                                "consume": dict(evs[n].consume),
                                "produce": dict(evs[n].produce),
                                "persist": dict(evs[n].persist)}
                               for n in card["moves"]]
        except Exception as e:            # a broken card is a finding
            card["moves"], card["effects"] = [], []
            card["error"] = f"{type(e).__name__}: {e}"
        out.append(card)
    return out


def card_line(c: dict) -> str:
    rel = {"served": "active", "extends": "pack",
           "sibling": "sibling"}[c["relation"]]
    def eff(e):
        ops = ([f"-{a}" + (f" {n}" if n != 1 else "")
                for a, n in sorted(e["consume"].items())]
               + [f"+{a}" + (f" {n}" if n != 1 else "")
                  for a, n in sorted(e["produce"].items())])
        return f"{e['name']} ({' '.join(ops)})"
    moves = (" moves " + ", ".join(eff(e) for e in c["effects"])
             if c.get("effects") else "")
    return f'{c["name"]} [{rel}] "{c["text"] or ""}"{moves}'


# ---- the script (metis-agent RAG §11.1, E8) -----------------------------

_TAG = re.compile(r"^(target):\s*(\w+)\s*$")


def _words(doc: str):
    """A clause's %% doc split into its words and its tags (`target:
    <catalog>` — the catalog whose world a cross-domain clause acts on)."""
    text, tags = [], {}
    for ln in doc.splitlines():
        m = _TAG.match(ln.strip())
        if m:
            tags[m.group(1)] = m.group(2)
        elif ln.strip():
            text.append(ln.strip())
    return " ".join(text), tags


def _side(atoms):
    """Atoms of one side of a clause by name, multiplicity folded, in
    order ($-premises kept: needed, not consumed)."""
    out, at = [], {}
    for a in atoms:
        if not hasattr(a, "pred") or a.pred == "one":
            continue                      # a Distinct guard, the LL unit
        k = (str(a).lstrip("$"), a.persist)
        if k in at:
            out[at[k]]["n"] += 1
            continue
        at[k] = len(out)
        out.append({"atom": k[0], "n": 1, "kept": a.persist})
    return out


def script_of(card: dict) -> dict:
    """One card's script from its catalog source (includes spliced): per
    clause {name, doc (its %% comment), premises, effects (atom names),
    target}, per atom {name, doc}. No authored interpretation: reading a
    fragment against the entries is the narrator's job (RAG §11.1). A
    sibling or pack lists only the clauses it adds over the served world
    (its card's moves) and the atoms they touch."""
    from metis.lang import ast
    from metis.lang.compiler import resolve_includes
    from metis.lang.parser import parse_file
    cat = resolve_includes(parse_file(card["file"]),
                           Path(card["file"]).parent)
    preds = [d for d in cat.decls if isinstance(d, ast.PredDecl)]
    moves = set(card.get("moves") or [])
    clauses = []
    for st in (d for d in cat.decls if isinstance(d, ast.Stage)):
        for r in st.rules:
            if card["relation"] != "served" and not any(
                    m == r.name or m.startswith(r.name + "(")
                    for m in moves):
                continue
            words, tags = _words(r.doc)
            clauses.append({"name": r.name, "stage": st.name,
                            "doc": words, "premises": _side(r.body),
                            "effects": _side(r.head),
                            "target": tags.get("target")})
    if card["relation"] != "served":      # only the atoms its moves use
        used = {x["atom"].split("(")[0] for c in clauses
                for x in c["premises"] + c["effects"]}
        preds = [d for d in preds if d.name in used]
    return {"name": card["name"], "relation": card["relation"],
            "text": card["text"], "clauses": clauses,
            "atoms": [{"name": d.name, "doc": d.doc} for d in preds],
            "missing": {"clause_doc": [c["name"] for c in clauses
                                       if not c["doc"]],
                        "atom_doc": [d.name for d in preds if not d.doc]}}


def script(shell) -> List[dict]:
    """The script of every card offerable at the served node: the served
    catalog's clauses and atoms, each sibling/pack's added clauses."""
    out = []
    for c in cards(shell):
        try:
            out.append(script_of(c))
        except Exception as e:            # a broken card is a finding
            out.append({"name": c["name"], "relation": c["relation"],
                        "text": c["text"], "clauses": [], "atoms": [],
                        "error": f"{type(e).__name__}: {e}"})
    return out


def script_line(s: dict) -> str:
    lines = [f'{s["name"]} [{s["relation"]}] "{s["text"] or ""}"']
    def side(xs):
        return " * ".join(("$" if x["kept"] else "") + x["atom"]
                          + (f"^{x['n']}" if x["n"] > 1 else "") for x in xs)
    for c in s["clauses"]:
        t = f" → {c['target']}" if c.get("target") else ""
        lines.append(f"  {c['name']}{t}: {c['doc'] or '(no comment)'}"
                     f" [{side(c['premises'])} -o {side(c['effects'])}]")
    if s["atoms"]:
        lines.append("  atoms: " + ", ".join(
            f"{a['name']} = {a['doc'] or '(no doc)'}" for a in s["atoms"]))
    return "\n".join(lines)
