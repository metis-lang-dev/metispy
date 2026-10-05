"""Renderers of the causality graph (spec 09 P1/P2): the run as a
partial order, in three deterministic mirrors of one object
(nets.causal_graph):

  - beats text — the primary surface, the answer to "where are we"
    without leaving the terminal: one block per Foata level, each
    occurrence with its covering parents (i<j), frontier marked *;
  - DOT — text only, rendering is the user's `dot -Tsvg` (no
    graphviz dependency); rank per beat, links as diamonds, the
    frontier bold;
  - JSON — the wire/console channel.

v1 edge styling is flow-only (D4): flow edges carry their atoms and
draw solid; read/order covering edges draw plain and unlabeled (the
JSON keeps their class and atoms — the vocabulary held in reserve).
Ordering: nodes by occurrence number, edges by (src, dst), atoms
sorted — so equal graphs give equal bytes."""
import json
from typing import Dict, List

from metis.kernel.nets import CausalGraph


def _parents(g: CausalGraph) -> Dict[int, List]:
    out: Dict[int, List] = {nd.i: [] for nd in g.nodes}
    for e in g.edges:
        out[e.dst].append(e)
    return out


def header(file: str, key: str, stage: str, steps: int) -> str:
    return (f"{file} · key {key[:12]} · stage {stage} · "
            f"{steps} fired")


def beats_text(g: CausalGraph, title: str = "") -> List[str]:
    """One block per beat, `_pretty_trace` style: occurrence number,
    event, its covering parents as i<j (flow atoms in parentheses),
    `*` on the frontier."""
    par = _parents(g)
    front = set(g.frontier)
    beats = max((nd.beat for nd in g.nodes), default=0) + 1
    num_w = len(str(max((nd.i for nd in g.nodes), default=0)))
    name_w = max((len(nd.name) for nd in g.nodes), default=0)
    beat_w = len(str(beats - 1))
    out = [title] if title else []
    for b in range(beats):
        first = True
        for nd in sorted((nd for nd in g.nodes if nd.beat == b),
                         key=lambda nd: nd.i):
            deps = " ".join(
                f"{e.src}<{e.dst}"
                + (f"({', '.join(e.atoms)})" if e.kind == "flow" else "")
                for e in par[nd.i])
            tag = f"{b:>{beat_w}}" if first else " " * beat_w
            mark = "*" if nd.i in front else " "
            shape = "◇" if nd.kind == "link" else " "
            out.append(f"  {tag} |{mark}{nd.i:>{num_w}} {shape}"
                       f"{nd.name:<{name_w}}  {deps}".rstrip())
            first = False
    out.append("  * frontier: " + " ".join(map(str, g.frontier)))
    return out


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dot(g: CausalGraph, label: str = "") -> str:
    """The DOT rendering: one rank per beat, links diamond, base
    ellipse, frontier bold; flow edges solid + atom-labeled, every
    other covering edge plain and unlabeled (D4)."""
    front = set(g.frontier)
    out = ["digraph causality {",
           "  rankdir=TB;",
           "  node [shape=box, fontname=\"monospace\"];"]
    if label:
        out += [f"  label={_q(label)};", "  labelloc=t;"]
    beats = max((nd.beat for nd in g.nodes), default=0) + 1
    for b in range(beats):
        ids = " ".join(f"n{nd.i};" for nd in g.nodes if nd.beat == b)
        out.append(f"  {{ rank=same; {ids} }}")
    for nd in sorted(g.nodes, key=lambda nd: nd.i):
        attrs = [f"label={_q(f'{nd.i} {nd.name}')}"]
        if nd.kind == "link":
            attrs.append("shape=diamond")
        elif nd.kind == "init":
            attrs.append("shape=ellipse")
        if nd.i in front:
            attrs.append("style=bold")
        out.append(f"  n{nd.i} [{', '.join(attrs)}];")
    for e in g.edges:
        attrs = (f" [label={_q(', '.join(e.atoms))}]"
                 if e.kind == "flow" else
                 " [color=gray]")
        out.append(f"  n{e.src} -> n{e.dst}{attrs};")
    out.append("}")
    return "\n".join(out) + "\n"


def as_obj(g: CausalGraph, stage: str, key: str) -> dict:
    """The JSON mirror (spec 09 P1): nodes, edges, frontier, stage,
    key — the wire op's result body."""
    return {
        "nodes": [{"i": nd.i, "name": nd.name, "kind": nd.kind,
                   "stage": nd.stage, "beat": nd.beat}
                  for nd in sorted(g.nodes, key=lambda nd: nd.i)],
        "edges": [{"from": e.src, "to": e.dst, "atoms": list(e.atoms),
                   "kind": e.kind} for e in g.edges],
        "frontier": list(g.frontier),
        "stage": stage,
        "key": key,
    }


def as_json(g: CausalGraph, stage: str, key: str) -> str:
    return json.dumps(as_obj(g, stage, key), indent=1,
                      ensure_ascii=False) + "\n"
