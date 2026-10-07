"""Renderers of the causality graph (spec 09 P1/P2): the run as a
partial order, in three deterministic mirrors of one object
(nets.causal_graph):

  - beats text — the primary surface, the answer to "where are we"
    without leaving the terminal: one block per Foata level, each
    occurrence with its covering parents (i<j), frontier marked *;
  - DOT — text only, rendering is the user's `dot -Tsvg` (no
    graphviz dependency): a causal diagram, rank per beat with a
    beat column, action over bindings in each node, links as
    diamonds, the frontier bold, a legend in the graph label;
  - JSON — the wire/console channel.

Edge classes in the DOT (D4, checked): flow solid + atom label,
read dashed + `$atom` label, order dotted gray with atoms only in a
tooltip. The beat text marks flow atoms; the JSON carries every
class and its atoms.
Ordering: nodes by occurrence number, edges by (src, dst), atoms
sorted — so equal graphs give equal bytes."""
import json
from typing import Dict, List

from metis.kernel.nets import CausalGraph

# spec 10: interventions are drawn as such. marks = {occurrence:
# "forced" | "do" | "fork" | "<how>+fork"}; occurrence 0 = the base,
# whose label names the state do's it is made of (base_do).
_TAG = {"forced": "[forced]", "do": "[do]", "fork": "[fork]"}


def _tags(mark: str) -> str:
    return " ".join(_TAG[m] for m in (mark or "").split("+") if m in _TAG)


def _parents(g: CausalGraph) -> Dict[int, List]:
    out: Dict[int, List] = {nd.i: [] for nd in g.nodes}
    for e in g.edges:
        out[e.dst].append(e)
    return out


def header(file: str, key: str, stage: str, steps: int) -> str:
    return (f"{file} · key {key[:12]} · stage {stage} · "
            f"{steps} fired")


def beats_text(g: CausalGraph, title: str = "", marks=None,
               base_do=()) -> List[str]:
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
            m = _tags((marks or {}).get(nd.i, ""))
            if nd.i == 0 and base_do:
                m = (m + " " if m else "") + "[do " + ", ".join(base_do) + "]"
            out.append((f"  {tag} |{mark}{nd.i:>{num_w}} {shape}"
                        f"{nd.name:<{name_w}}  {deps}".rstrip()
                        + (f"  {m}" if m else "")))
            first = False
    out.append("  * frontier: " + " ".join(map(str, g.frontier)))
    return out


def _q(s: str) -> str:
    """A DOT string; a newline becomes DOT's centered line break."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') \
        .replace("\n", "\\n") + '"'


LEGEND = ("solid, atom: token flow (made, then spent)  ·  "
          "dashed, $atom: read (needed, kept)  ·  "
          "dotted: order (competition, stage)  ·  "
          "bold: frontier  ·  diamond: stage switch")


def _node_label(nd) -> str:
    """`NN name` over the bindings (`C=montague D=romeo …`): the
    action first, its cast on a second line."""
    base, _, rest = nd.name.partition("[")
    binds = rest.rstrip("]").replace(",", " ")
    return f"{nd.i} {base}" + (f"\n{binds}" if binds else "")


def dot(g: CausalGraph, label: str = "", marks=None,
        base_do=()) -> str:
    """The causal diagram (spec 09 D4, CelfToGraph style: nodes are
    actions, edges causal relationships), one rank per beat with a
    beat column on the left. Edge classes: flow solid + atom label
    (the causal edge proper), read dashed + `$atom` label, order
    dotted gray with the atoms in a tooltip only (the noise class;
    constraint=false so it never bends the layout — the invisible
    beat chain keeps the ranks)."""
    front = set(g.frontier)
    out = ["digraph causality {",
           "  rankdir=TB; nodesep=0.25; ranksep=0.45;",
           '  graph [fontname="Helvetica", fontsize=12];',
           '  node [shape=box, fontname="Helvetica", fontsize=11, '
           'margin="0.12,0.04"];',
           '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.7];']
    legend = LEGEND + ("  ·  orange: intervention (do / forced — "
                       "not sampled)  ·  ⑂: the fork point"
                       if marks or base_do else "")
    title = "\n".join(x for x in (label, legend) if x)
    out += [f"  label={_q(title)};", "  labelloc=t;"]
    beats = max((nd.beat for nd in g.nodes), default=0) + 1
    out.append('  node [shape=plaintext, fontcolor=gray40] '
               + " ".join(f'b{b} [label="beat {b}"];'
                          for b in range(beats)))
    if beats > 1:
        out.append("  " + " -> ".join(f"b{b}" for b in range(beats))
                   + " [style=invis];")
    out.append('  node [shape=box, fontcolor=black];')
    ranks = []
    for b in range(beats):
        ids = " ".join(f"n{nd.i};" for nd in g.nodes if nd.beat == b)
        ranks.append(f"  {{ rank=same; b{b}; {ids} }}")

    def node_line(nd, indent="  "):
        mk = (marks or {}).get(nd.i, "")
        lab = _node_label(nd)
        if nd.i == 0 and base_do:
            lab = "0 do(" + ", ".join(base_do) + ")"
        if "do" in mk.split("+"):
            lab = "do: " + lab
        elif "forced" in mk.split("+"):
            lab = "forced: " + lab
        attrs = [f"label={_q(lab)}"]
        if "do" in mk.split("+") or "forced" in mk.split("+") \
                or (nd.i == 0 and base_do):
            attrs.append("color=darkorange3, fontcolor=darkorange4, "
                         "peripheries=2")
        if "fork" in mk.split("+"):
            attrs.append('xlabel="⑂ fork", fontcolor=red3')
        if nd.kind == "link":
            attrs.append("shape=diamond")
        elif nd.kind == "init":
            attrs.append("shape=ellipse")
        if nd.i in front:
            attrs.append("style=bold, penwidth=2")
        return f"{indent}n{nd.i} [{', '.join(attrs)}];"
    nodes = sorted(g.nodes, key=lambda nd: nd.i)
    if not any(nd.kind == "link" for nd in nodes):
        out += ranks + [node_line(nd) for nd in nodes]
    else:
        # D5 quiescence frames: one cluster per stage episode, link
        # diamonds between them. newrank lets the beat ranks (and the
        # beat column) align across cluster boundaries.
        out.insert(2, "  newrank=true;")
        episodes = sorted({nd.episode for nd in nodes
                           if nd.episode is not None})
        for ep in episodes:
            members = [nd for nd in nodes if nd.episode == ep]
            stage = members[0].stage
            out += [f"  subgraph cluster_e{ep} {{",
                    f"    label={_q(f'stage {stage} · episode {ep}')};",
                    "    labeljust=l; style=filled; fillcolor=gray96; "
                    "color=gray80; fontcolor=gray30;"]
            out += [node_line(nd, "    ") for nd in members]
            out.append("  }")
        out += [node_line(nd) for nd in nodes if nd.episode is None]
        out += ranks
    for e in g.edges:
        if e.kind == "flow":
            attrs = f"label={_q(chr(10).join(e.atoms))}, penwidth=1.6"
        elif e.kind == "read":
            attrs = (f"label={_q(chr(10).join('$' + a for a in e.atoms))}"
                     ", style=dashed, color=steelblue4, "
                     "fontcolor=steelblue4")
        else:
            attrs = (f"tooltip={_q(', '.join(e.atoms))}, style=dotted, "
                     "color=gray55, constraint=false")
        out.append(f"  n{e.src} -> n{e.dst} [{attrs}];")
    out.append("}")
    return "\n".join(out) + "\n"


def as_obj(g: CausalGraph, stage: str, key: str, marks=None,
           base_do=()) -> dict:
    """The JSON mirror (spec 09 P1): nodes, edges, frontier, stage,
    key — the wire op's result body. Nodes carry their stage episode
    (D5; null for a link, which stands between episodes)."""
    return {
        "nodes": [dict({"i": nd.i, "name": nd.name, "kind": nd.kind,
                        "stage": nd.stage, "beat": nd.beat,
                        "episode": nd.episode},
                       **({"mark": (marks or {})[nd.i]}
                          if (marks or {}).get(nd.i) else {}))
                  for nd in sorted(g.nodes, key=lambda nd: nd.i)],
        "edges": [{"from": e.src, "to": e.dst, "atoms": list(e.atoms),
                   "kind": e.kind} for e in g.edges],
        "frontier": list(g.frontier),
        "stage": stage,
        "key": key,
        **({"base_do": list(base_do)} if base_do else {}),
    }


def as_json(g: CausalGraph, stage: str, key: str, marks=None,
            base_do=()) -> str:
    return json.dumps(as_obj(g, stage, key, marks, base_do), indent=1,
                      ensure_ascii=False) + "\n"


def cone_text(g: CausalGraph, members, title: str,
              source=None) -> List[str]:
    """`why`: the cone grouped by beat, each occurrence with its rule's
    source line (`source(name) -> (lineno, text) | None`, as `list`
    renders rules), then what was pruned."""
    keep = set(members)
    num_w = len(str(max((nd.i for nd in g.nodes), default=0)))
    out = [title]
    beats = max((nd.beat for nd in g.nodes), default=0) + 1
    for b in range(beats):
        first = True
        for nd in sorted((nd for nd in g.nodes
                          if nd.beat == b and nd.i in keep),
                         key=lambda nd: nd.i):
            tag = f"{b:>{len(str(beats - 1))}}" if first else \
                " " * len(str(beats - 1))
            loc = source(nd.name) if source and nd.kind != "init" \
                else None
            where = (f"  L{loc[0]}: {loc[1].strip()}" if loc else
                     "  (the base state)" if nd.kind == "init" else "")
            out.append(f"  {tag} | {nd.i:>{num_w}} {nd.name}{where}")
            first = False
    pruned = [nd for nd in g.nodes if nd.i not in keep]
    out.append(f"  pruned: {len(pruned)} irrelevant event"
               f"{'' if len(pruned) == 1 else 's'}"
               + (": " + ", ".join(f"{nd.i} {nd.name}" for nd in pruned)
                  if pruned else ""))
    return out


def causes(g: CausalGraph, match_a, match_b) -> dict:
    """TeLLer's `x a b` / CelfToGraph's `link`: for each occurrence of
    B, the occurrences of A among its ancestors. verdict: yes (every
    B has one), some, no — or none when B never fired."""
    from metis.kernel.nets import cone
    rows = []
    for nd in g.nodes:
        if nd.kind != "init" and match_b(nd.name):
            anc = [i for i in cone(g, [nd.i]) if i != nd.i]
            hits = [i for i in anc
                    if match_a(next(n.name for n in g.nodes if n.i == i))]
            rows.append({"b": nd.i, "name": nd.name, "a": hits})
    n_yes = sum(1 for r in rows if r["a"])
    verdict = ("none" if not rows else "yes" if n_yes == len(rows)
               else "some" if n_yes else "no")
    return {"rows": rows, "verdict": verdict}
