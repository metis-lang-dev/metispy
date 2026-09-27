"""The IR parity corpus, manifest-driven. SINGLE SOURCE:
ocaml/parity/manifest.txt describes every case (llp, init, horizon,
registry name, queries, likelihood clamps); the Python golden
generator (cpp/parity/gen_ir_goldens.py), the OCaml driver
(ocaml/metisc) and the parity gates all read it. Registries are the
one thing a manifest cannot carry (bindings are native callables);
they are named here and resolved by name — the OCaml side ignores
the `registry` line (symbolic toolchain, no bindings by design)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from test_factorize import ring_registry, sys_registry

REGISTRIES = {
    "sysadmin": sys_registry,
    "cascade": lambda: {"pa": lambda a, c: 0.6,
                        "pb": lambda a, c: 0.9 if "a" in c else 0.2},
    "triway": lambda: {"w1": lambda a, c: 0.5,
                       "w2": lambda a, c: 0.3},
    "ring6": lambda: ring_registry(6),
    "packcoin": lambda: {"ph": lambda a, c: 0.7},
    "nav4": lambda: {"go": lambda a, c: 0.9,
                     "die": lambda a, c: 0.1},
    "roadclose": lambda: _road_close_bindings(),
    "xtraffic1": lambda: _xtraffic1_bindings(),
    "gol4": lambda: _gol4_bindings(),
}


def _gol4_bindings():
    from metis.domains.gol.model import life_bindings
    cells = ["g00", "g01", "g10", "g11"]
    nbrs = {c: [v for v in cells if v != c] for c in cells}
    return life_bindings(nbrs, {c: 0.1 for c in cells})


def xtraffic1_topo():
    """CrossingTraffic instance 1 (3x3, one obstacle row) — the same
    topology tests/xtraffic/test_xtraffic.py builds."""
    from metis.domains.xtraffic.model import Topology
    xs, ys = ["x1", "x2", "x3"], ["y1", "y2", "y3"]
    cells, adj = [], {}
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            c = f"{x}_{y}"
            cells.append(c)
            if yi + 1 < len(ys):
                adj[(c, "north")] = f"{x}_{ys[yi + 1]}"
            if yi - 1 >= 0:
                adj[(c, "south")] = f"{x}_{ys[yi - 1]}"
            if xi + 1 < len(xs):
                adj[(c, "east")] = f"{xs[xi + 1]}_{y}"
            if xi - 1 >= 0:
                adj[(c, "west")] = f"{xs[xi - 1]}_{y}"
    return Topology(cells, adj, ["x1_y2", "x2_y2", "x3_y2"],
                    {"x1_y2": "x2_y2", "x2_y2": "x3_y2"},
                    goal="x3_y3", input_rate=0.3, horizon=40)


def _xtraffic1_bindings():
    from metis.domains.xtraffic.model import XTrafficModel
    return XTrafficModel(xtraffic1_topo())._bindings


def _road_close_bindings():
    """The stage-33 semantic-handles scene (test_semantic_road's
    close_scene): geometry reaches the measure only through these
    bindings; the C++ corpus registry mirrors their values."""
    from metis.domains.road.semantic import Lane, Scene, \
        SemanticCompiler
    lanes = [Lane("l1", 500.0, 22.0, left=None, right="l2"),
             Lane("l2", 500.0, 27.8, left="l1", right="l3"),
             Lane("l3", 500.0, 27.8, left="l2", right=None)]
    scene = Scene({"ego": ("l2", 100.0, 25.0),
                   "alfa": ("l2", 110.0, 20.0)})
    return SemanticCompiler(lanes).bindings(scene)


@dataclass
class Case:
    name: str = ""
    llp: str = ""
    pack: str = ""                # base+pack admission when set
    registry: str = ""
    init: Dict[str, int] = field(default_factory=dict)
    steps: int = 0
    unknowns: Tuple[str, ...] = ()
    queries: List[str] = field(default_factory=list)
    liks: List[Dict[str, int]] = field(default_factory=list)
    seeds: int = 0                # sampler-wire cases (06-spec §5)


def parse_manifest(path) -> List[Case]:
    path = Path(path)
    cases, cur = [], Case()
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if tok[0] == "case":
            cur.name = tok[1]
        elif tok[0] == "llp":
            cur.llp = str((path.parent / tok[1]).resolve())
        elif tok[0] == "pack":
            cur.pack = str((path.parent / tok[1]).resolve())
        elif tok[0] == "registry":
            cur.registry = tok[1]
        elif tok[0] == "init":
            cur.init[tok[1]] = int(tok[2])
        elif tok[0] == "steps":
            cur.steps = int(tok[1])
        elif tok[0] == "seeds":
            cur.seeds = int(tok[1])
        elif tok[0] == "unknown":
            cur.unknowns += (tok[1],)
        elif tok[0] == "query":
            cur.queries.append(tok[1])
        elif tok[0] == "lik":
            cur.liks.append(dict(
                (kv.split("=")[0], int(kv.split("=")[1]))
                for kv in tok[1:]))
        elif tok[0] == "end":
            cases.append(cur)
            cur = Case()
        else:
            raise ValueError(f"bad manifest line: {line}")
    return cases


def build_program(case: Case):
    """Ground program only (no hooks) — the sampler-wire path runs
    prior-only (06-sampler-wire §4)."""
    from metis.lang import load_llp
    doc = load_llp(case.llp)
    if case.pack:
        packdoc = load_llp(case.pack, base=doc)
        return doc.catalog.admit(packdoc.catalog).ground(
            dict(case.init))
    return doc.catalog.ground(dict(case.init))


def build_case(case: Case):
    """Ground program + factor hooks for one manifest case. A `pack`
    case compiles the pack against the base, ADMITS it (containment)
    and grounds the merged catalog; hooks come from the base ports
    (packs that add weight ports would merge them — out of corpus
    scope), horn from the merged catalog."""
    from metis.lang import factor_hooks, load_llp
    doc = load_llp(case.llp)
    if case.pack:
        packdoc = load_llp(case.pack, base=doc)
        merged = doc.catalog.admit(packdoc.catalog)
        prog = merged.ground(dict(case.init))
        horn = merged.horn_db()
    else:
        prog = doc.catalog.ground(dict(case.init))
        horn = doc.catalog.horn_db()
    reg = REGISTRIES[case.registry]()
    hooks = factor_hooks(prog, doc.ports, reg, horn)
    return prog, hooks
