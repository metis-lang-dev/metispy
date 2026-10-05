"""The causality graph (spec 09 P1/P2): the run as a partial order —
nets.causal_graph (the object), lang.causality (beats/DOT/JSON
mirrors) and the REPL's `graph` + the wire op over the session.

Goldens pin the tragedy fixture's DOT and JSON bytes; regenerate
deliberately with METIS_BLESS=1 (and review the diff)."""

import io
import json
import os
import random
from pathlib import Path

from metis.kernel.nets import (_footprint, canonical_graph, causal_graph,
                               foata_levels, run_key)
from metis.kernel.refsample import canonical_events, run_trace
from metis.lang.causality import as_json, beats_text, dot
from metis.lang.repl import Repl
from metis.loop.wire import dispatch

ROOT = Path(__file__).resolve().parents[2]
FICTION = ROOT.parent / "metis-catalog/domains/fiction"
GOLDENS = Path(__file__).parent / "goldens"


def _shell(name, seed=1):
    return Repl(FICTION / name, seed=seed, stdout=io.StringIO())


def _run(r, *cmds):
    for c in cmds:
        r.onecmd(c)
    out, r.stdout = r.stdout.getvalue(), io.StringIO()
    return out


def _trace(name, seed=1):
    """(program, case, event indices) of a seeded fixture trace."""
    r = _shell(name)
    idx, _ = run_trace(r.prog, r.case["steps"] or 6, seed)
    return r.prog, r.case, list(idx)


def _graph(prog, case, idx):
    tbl = canonical_events(prog)
    return causal_graph([tbl[i] for i in idx], case["init"],
                        prog.init_stage)


def _golden(name, text):
    p = GOLDENS / name
    if os.environ.get("METIS_BLESS"):
        p.parent.mkdir(exist_ok=True)
        p.write_text(text)
    assert text == p.read_text(), f"{name} drifted (METIS_BLESS=1 to " \
                                  "regenerate, then review the diff)"


# -- the pinned goldens ------------------------------------------------

def test_tragedy_graph_goldens(tmp_path):
    """The REPL's `graph F.dot` / `graph F.json` on a seeded tragedy
    session are pinned byte for byte."""
    r = _shell("tragedy.llp")
    _run(r, "run 40")
    _run(r, f"graph {tmp_path}/t.dot", f"graph {tmp_path}/t.json")
    _golden("causality_tragedy.dot", (tmp_path / "t.dot").read_text())
    _golden("causality_tragedy.json", (tmp_path / "t.json").read_text())


def test_herstory_graph_goldens(tmp_path):
    """The multi-stage pin (D5): herstory's four acts are four stage
    episodes; `graph F.dot` / `graph F.json` pinned byte for byte."""
    r = _shell("herstory.llp")
    _run(r, "run 40")
    _run(r, f"graph {tmp_path}/h.dot", f"graph {tmp_path}/h.json")
    _golden("causality_herstory.dot", (tmp_path / "h.dot").read_text())
    _golden("causality_herstory.json", (tmp_path / "h.json").read_text())


# -- the object's laws -------------------------------------------------

def _closure(g):
    """occurrence -> strict ancestors, from the edges alone."""
    par = {nd.i: {e.src for e in g.edges if e.dst == nd.i}
           for nd in g.nodes}
    memo = {}

    def anc(j):
        if j not in memo:
            memo[j] = set()
            for i in par[j]:
                memo[j] |= {i} | anc(i)
        return memo[j]
    return {nd.i: anc(nd.i) for nd in g.nodes}


def test_edges_are_the_hasse_diagram_of_the_dependence_order():
    """Every edge is a direct footprint conflict (D1: covering pairs
    only — no edge is implied by a longer path) and every dependent
    pair is ordered by the closure."""
    prog, case, idx = _trace("tragedy.llp")
    tbl = canonical_events(prog)
    rows = [tbl[i] for i in idx]
    g = _graph(prog, case, idx)
    anc = _closure(g)
    for e in g.edges:
        others = {e2.src for e2 in g.edges if e2.dst == e.dst} - {e.src}
        assert not any(e.src in anc[k] for k in others), e
    fps = [_footprint(*row) for row in rows]
    for j in range(len(rows)):
        for i in range(j):
            (ri, wi), (rj, wj) = fps[i], fps[j]
            if (wi & (wj | rj)) or (wj & ri):
                assert i + 1 in anc[j + 1], (i + 1, j + 1)


def test_beats_are_the_narrated_foata_levels_and_frontier_maximal():
    prog, case, idx = _trace("tragedy.llp")
    tbl = canonical_events(prog)
    g = _graph(prog, case, idx)
    levels = foata_levels([tbl[i] for i in idx])
    assert [nd.beat for nd in g.nodes[1:]] == [l + 1 for l in levels]
    anc = _closure(g)
    maximal = {j for j in anc if not any(j in a for a in anc.values())}
    assert set(g.frontier) == maximal


def test_flow_edges_carry_the_spent_tokens():
    """3lp seed 1: the stick house build_stick made is the token
    blow_stick spent — a flow edge labeled stickhouse."""
    prog, case, idx = _trace("3lp.llp")
    g = _graph(prog, case, idx)
    name = {nd.i: nd.name for nd in g.nodes}
    flows = {(name[e.src], name[e.dst]): e.atoms for e in g.edges
             if e.kind == "flow"}
    assert flows[("build_stick", "blow_stick")] == ("stickhouse",)
    assert flows[("#init", "build_stick")] == ("pig", "sticks")


def test_adjacent_independent_swap_leaves_canonical_bytes_invariant():
    """run_key's theorem made visible: swapping adjacent independent
    events changes neither run_key nor the canonical graph's bytes —
    checked on every swappable pair, then on a seeded random walk."""
    for name in ("tragedy.llp", "3lp.llp", "love_triangle.llp"):
        prog, case, idx = _trace(name)
        tbl = canonical_events(prog)

        def swappable(seq, k):
            (r1, w1), (r2, w2) = (_footprint(*tbl[seq[k]]),
                                  _footprint(*tbl[seq[k + 1]]))
            return not (w1 & (w2 | r2)) and not (w2 & r1)

        def canon(seq):
            g = _graph(prog, case, seq)
            return as_json(canonical_graph(g), prog.init_stage, "")
        # traces index canonical_events; run_key reads program.events()
        pos = {id(ev): k for k, (_, _, ev) in enumerate(prog.events())}

        def rkey(seq):
            return run_key(prog, [pos[id(tbl[i][2])] for i in seq])
        ref, key = canon(idx), rkey(idx)
        pairs = [k for k in range(len(idx) - 1) if swappable(idx, k)]
        assert pairs or name == "3lp.llp" and len(idx) < 3
        for k in pairs:
            seq = idx[:k] + [idx[k + 1], idx[k]] + idx[k + 2:]
            assert canon(seq) == ref and rkey(seq) == key
        rng, seq = random.Random(9), list(idx)
        for _ in range(200):
            ks = [k for k in range(len(seq) - 1) if swappable(seq, k)]
            if not ks:
                break
            k = rng.choice(ks)
            seq[k], seq[k + 1] = seq[k + 1], seq[k]
        assert canon(seq) == ref and rkey(seq) == key


# -- the REPL / wire surface -------------------------------------------

def test_graph_text_marks_frontier_and_parents():
    r = _shell("3lp.llp")
    _run(r, "run 6")
    out = _run(r, "graph")
    assert "6 fired" in out and "1<5(stickhouse)" in out
    assert "|*6" in out and "frontier: 6" in out
    assert "usage" in _run(r, "graph x.txt")


def test_graph_rebases_on_edits_and_undo_restores_the_base():
    """reset / assert / retract / #init are edits no event explains:
    the graph restarts there (node 0 = that state); undo brings the
    previous base back."""
    r = _shell("3lp.llp")
    _run(r, "run 3")
    assert "3 fired" in _run(r, "graph")
    _run(r, "+wolf")
    out = _run(r, "graph")
    assert "0 fired (since #3" in out and "frontier: 0" in out
    _run(r, "step")
    assert "|*4" in _run(r, "graph")
    _run(r, "undo", "undo")
    assert "3 fired" in _run(r, "graph")
    _run(r, "reset", "step")
    assert "1 fired (since #3" in _run(r, "graph")


def test_wire_graph_is_the_json_mirror():
    r = _shell("3lp.llp")
    _run(r, "run 6")
    out = dispatch(r, "graph", "")
    assert out["ok"] and out["frontier"] == [6]
    assert [n["i"] for n in out["nodes"]] == list(range(7))
    assert {"from", "to", "atoms", "kind"} <= set(out["edges"][0])
    g, _ = r.causal()
    assert json.loads(as_json(g, r.stage, r.key)) == \
        {k: v for k, v in out.items() if k != "ok"}
    assert dot(g).startswith("digraph causality {")
    assert beats_text(g)[-1].strip().startswith("* frontier")


# -- step 2: why / causes (spec 09 P3) ---------------------------------

def test_why_cone_is_the_hand_computed_ancestry():
    """3lp seed 1 (run 6): 1 build_stick, 2 build_straw, 3 blow_straw,
    4 build_brick, 5 blow_stick, 6 blow_brick. The final state is
    {brickhouse}: only build_brick wrote it; its causes are the pig
    competition 1<2<4 and the base — the three blows are pruned."""
    r = _shell("3lp.llp")
    _run(r, "run 6")
    out = _run(r, "why")
    assert out.splitlines()[0] == \
        "why the current state: 4 of 7 occurrences"
    assert "pruned: 3 irrelevant events: 3 blow_straw, 5 blow_stick, " \
           "6 blow_brick" in out
    assert "L35: build_brick [story]" in out     # source line, as list
    assert dispatch(r, "why", "5")["cone"] == [0, 1, 5]
    assert dispatch(r, "why", "6")["cone"] == [0, 1, 2, 3, 4, 5, 6]
    assert dispatch(r, "why", "brickhouse")["cone"] == [0, 1, 2, 4]


def test_why_atom_and_errors():
    r = _shell("3lp.llp")
    _run(r, "run 6")
    assert "last produced by event 4" in _run(r, "why brickhouse")
    out = _run(r, "why wolf")
    assert "from the base state" in out and "not present now" in out
    assert "covers 1..6" in _run(r, "why 9")
    assert "never produced" in _run(r, "why dragon")
    assert dispatch(r, "why", "9")["ok"] is False


def test_causes_truth_table_on_a_fork():
    """yes / no / some / none — love_triangle's jealousies fork: the
    two jealousy chains 1<2<4 and 3<5 run side by side."""
    r = _shell("3lp.llp")
    _run(r, "run 6")
    assert dispatch(r, "causes", "build_stick blow_stick")["verdict"] \
        == "yes"
    assert dispatch(r, "causes", "build_brick blow_straw")["verdict"] \
        == "no"
    assert "has not fired" in _run(r, "reset", "causes pig blow_brick")
    assert "usage" in _run(r, "causes x")
    t = _shell("love_triangle.llp")
    _run(t, "run 6")
    res = dispatch(t, "causes", "jealousy jealousy")
    assert res["verdict"] == "some"
    assert [(row["b"], row["a"]) for row in res["rows"]] == \
        [(1, []), (2, [1]), (3, []), (4, [1, 2]), (5, [3])]
    assert "some: jealousy causes 3 of 5 jealousy" in \
        _run(t, "causes jealousy jealousy")


def test_alternative_branches_match_their_source_rule(tmp_path):
    """⊕ branches (blow_stick-1/-2) answer to the rule's own name."""
    src = (FICTION / "3lp.llp").read_text()
    old = "blow_stick [story] : $wolf * stickhouse -o ()."
    assert old in src
    alt = tmp_path / "3lp_alt.llp"
    alt.write_text(src.replace(old, "blow_stick [story] : $wolf * "
                               "stickhouse\n    -o ( () @w 0.75 | "
                               "stickhouse @w 0.25 )."))
    r = Repl(alt, seed=1, stdout=io.StringIO())
    _run(r, "run 6")
    assert any(n.endswith("-1") or n.endswith("-2") for n in r.fired)
    assert dispatch(r, "causes", "build_stick blow_stick")["verdict"] \
        == "yes"


# -- step 3: the DOT as a causal diagram (spec 09 D4, checked) -------

def test_dot_draws_all_three_edge_classes_with_a_legend():
    """flow solid + atom label, read dashed + `$atom` label, order
    dotted gray with atoms only in a tooltip (never a label); node
    labels split action / bindings; the graph label carries a legend
    and a beat column keeps the ranks."""
    r = _shell("tragedy.llp")
    _run(r, "run 40")
    g, _ = r.causal()
    text = dot(g, "t")
    lines = text.splitlines()
    kinds = {e.kind for e in g.edges}
    assert kinds == {"flow", "read", "order"}
    for e in g.edges:
        line = next(l for l in lines
                    if l.strip().startswith(f"n{e.src} -> n{e.dst} ["))
        if e.kind == "flow":
            assert "label=" in line and "style=" not in line
            assert all(a in line for a in e.atoms)
        elif e.kind == "read":
            assert "style=dashed" in line
            assert all(f"${a}" in line for a in e.atoms)
        else:
            assert "style=dotted" in line and "label=" not in line
            assert "tooltip=" in line and "constraint=false" in line
    assert 'label="36 murder\\nC=romeo D=tybalt L=town"' in text
    assert "token flow" in text and "read" in text and "order" in text
    assert "b0 -> b1" in text and '{ rank=same; b0; n0; }' in text


# -- step 4: quiescence frames (spec 09 D5) ----------------------------

def test_episodes_cluster_the_dot_and_links_stand_between():
    """One cluster per stage episode (label: stage · episode), link
    diamonds outside every cluster, newrank so the beat column aligns
    across clusters; a run with no link has no cluster at all."""
    r = _shell("herstory.llp")
    _run(r, "run 40")
    g, _ = r.causal()
    text = dot(g, "h")
    eps = [nd.episode for nd in g.nodes]
    links = [nd for nd in g.nodes if nd.kind == "link"]
    assert [nd.episode for nd in links] == [None] * 3
    assert sorted({e for e in eps if e is not None}) == [0, 1, 2, 3]
    assert text.count("subgraph cluster_e") == 4 and "newrank=true" in text
    for stage, ep in (("beginning", 0), ("courtship", 1),
                      ("tragedy", 2), ("betrayal", 3)):
        assert f'label="stage {stage} · episode {ep}"' in text
    # each link node is declared at top level, never inside a cluster
    depth, where = 0, {}
    for line in text.splitlines():
        if "subgraph cluster" in line:
            depth += 1
        elif line.strip() == "}" and depth:
            depth -= 1
        for nd in links:
            if line.strip().startswith(f"n{nd.i} [label="):
                where[nd.i] = depth
    assert where == {nd.i: 0 for nd in links}
    # no beat spans two episodes
    for b in {nd.beat for nd in g.nodes}:
        assert len({nd.episode for nd in g.nodes if nd.beat == b}) == 1
    t = _shell("3lp.llp")
    _run(t, "run 6")
    tg, _ = t.causal()
    assert "cluster" not in dot(tg) and "newrank" not in dot(tg)


def test_json_mirror_carries_the_episode():
    r = _shell("herstory.llp")
    _run(r, "run 40")
    nodes = dispatch(r, "graph", "")["nodes"]
    assert {n["name"]: n["episode"] for n in nodes
            if n["kind"] == "link"} == {"act2": None, "act3": None,
                                        "act4": None}
    assert nodes[0]["episode"] == 0 and nodes[-1]["episode"] == 3
    assert {"i", "name", "kind", "stage", "beat"} <= set(nodes[0])
