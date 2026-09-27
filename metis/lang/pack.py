"""The pack artifact: CatalogDoc <-> JSON-able dict. The pack is what
ships and hot-loads (the wayplan compile_pack doctrine); reloading must
reproduce the identical program (program_key equality is the gate)."""

from metis.kernel import Catalog, ClauseSchema, Entry, HornClause, \
    LinkSchema, Namespace, Provenance, Signature
from . import ast
from .compiler import CatalogDoc, Ports, WeightRef


def _atom_out(a: ast.Atom):
    return [a.pred, [t.value for t in a.terms]]


def _atom_in(d) -> ast.Atom:
    pred, terms = d
    return ast.Atom(pred, tuple(
        ast.Term("var" if t[0].isupper() else "const", t)
        for t in terms))


def _pats(ps):
    return [[p, list(a)] for p, a in ps]


def _unpats(ps):
    return [(p, tuple(a)) for p, a in ps]


def to_pack(doc: CatalogDoc) -> dict:
    c = doc.catalog
    entries = []
    for e in c.entries:
        d = {"name": e.name, "layer": e.layer, "kind": e.kind,
             "comment": e.comment,
             "provenance": [e.provenance.source, e.provenance.ref,
                            e.provenance.date]}
        p = e.payload
        if e.kind in ("schema", "link"):
            d["payload"] = {
                "name": p.name, "vars": dict(p.vars),
                "consume": _pats(p.consume), "produce": _pats(p.produce),
                "persist": _pats(p.persist), "guards": _pats(p.guards),
                "distinct": [list(x) for x in p.distinct],
                "weight": p.weight}
            if e.kind == "schema":
                d["payload"]["stage"] = p.stage
            else:
                d["payload"]["pre"] = p.pre
                d["payload"]["post"] = p.post
        elif e.kind == "horn":
            d["payload"] = {"name": p.name,
                            "head": [p.head[0], list(p.head[1])],
                            "body": _pats(p.body),
                            "vars": list(p.vars)}
        entries.append(d)
    return {
        "format": "metis-pack", "version": 1,
        "catalog": {
            "name": c.name, "version": c.version,
            "layers": list(c.layers), "stages": list(c.stages),
            "types": {t: list(v) for t, v in c.sig.types.items()},
            "namespaces": {n: {"produce": sorted(s.produce),
                               "consume": sorted(s.consume)}
                           for n, s in c.namespaces.items()},
            "preds": dict(c.preds), "bwd": sorted(c.bwd),
            "entries": entries},
        "ports": {
            "weights": {k: list(v) for k, v in doc.ports.weights.items()},
            "guards": {k: list(v) for k, v in doc.ports.guards.items()},
            "inputs": list(doc.ports.inputs),
            "outputs": list(doc.ports.outputs),
            "weight_refs": {k: [r.factor, list(r.args), r.complement]
                            for k, r in doc.ports.weight_refs.items()},
            "reads": {k: [[_atom_out(p.atom),
                           [_atom_out(g) for g in p.guards]]
                          for p in v]
                      for k, v in doc.ports.reads.items()}},
    }


def from_pack(pack: dict) -> CatalogDoc:
    c = pack["catalog"]
    entries = []
    for d in c["entries"]:
        prov = Provenance(*d["provenance"])
        p = d.get("payload")
        if d["kind"] == "schema":
            payload = ClauseSchema(
                p["name"], stage=p["stage"], vars=dict(p["vars"]),
                consume=_unpats(p["consume"]),
                produce=_unpats(p["produce"]),
                persist=_unpats(p["persist"]),
                guards=_unpats(p["guards"]),
                distinct=[tuple(x) for x in p["distinct"]],
                weight=p["weight"])
        elif d["kind"] == "link":
            payload = LinkSchema(
                p["name"], pre=p["pre"], post=p["post"],
                vars=dict(p["vars"]),
                consume=_unpats(p["consume"]),
                produce=_unpats(p["produce"]),
                persist=_unpats(p["persist"]),
                guards=_unpats(p["guards"]),
                distinct=[tuple(x) for x in p["distinct"]],
                weight=p["weight"])
        elif d["kind"] == "horn":
            payload = HornClause(p["name"],
                                 (p["head"][0], tuple(p["head"][1])),
                                 tuple(_unpats(p["body"])),
                                 vars=tuple(p["vars"]))
        else:
            payload = p
        entries.append(Entry(name=d["name"], layer=d["layer"],
                             kind=d["kind"], comment=d["comment"],
                             provenance=prov, payload=payload))
    cat = Catalog(
        name=c["name"], version=c["version"],
        layers=tuple(c["layers"]), stages=tuple(c["stages"]),
        sig=Signature({t: list(v) for t, v in c["types"].items()}),
        namespaces={n: Namespace(n, frozenset(s["produce"]),
                                 frozenset(s["consume"]))
                    for n, s in c["namespaces"].items()},
        preds=dict(c["preds"]), bwd=frozenset(c["bwd"]),
        entries=entries).checked()
    pp = pack["ports"]
    ports = Ports(
        weights={k: tuple(v) for k, v in pp["weights"].items()},
        guards={k: tuple(v) for k, v in pp["guards"].items()},
        inputs=tuple(pp["inputs"]), outputs=tuple(pp["outputs"]),
        weight_refs={k: WeightRef(f, tuple(a), c2)
                     for k, (f, a, c2) in pp["weight_refs"].items()},
        reads={k: tuple(ast.ReadPattern(_atom_in(a),
                                        tuple(_atom_in(g) for g in gs))
                        for a, gs in v)
               for k, v in pp.get("reads", {}).items()})
    return CatalogDoc(cat, ports)
