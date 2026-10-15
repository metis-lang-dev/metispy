"""The product ground-wire emitter — the artifact the C++ runtime
loads (cpp/include/metis/loader.hpp, evaluator.hpp). Same N/E/I/S
record grammar as the conformance goldens (cpp/parity/gen_goldens.py
keeps its own frozen copy for the parity corpus), PLUS:

    B <kind> <name>      port-manifest record (kind: in|out|guard|
                         weight) — what the artifact requires the
                         consuming registry to bind; the C++
                         Evaluator fail-fasts on a missing binding
                         (fw.Evaluator._validate parity)

and MINUS the Q/D/R expectation records: a product artifact carries
no expectations — certification happened on the authoring side.

python is the compiler: emit from a grounded Program + the resolved
per-event WeightRefs (weight_refs()) + the CatalogDoc's Ports."""

from typing import Dict, List, Optional, Tuple

Ref = Tuple[str, List[str], bool]        # factor, resolved args, compl


def weight_refs(doc, prog) -> Dict[int, Ref]:
    """Per ground event: (factor, resolved args, complement) from the
    catalog's weight refs + the event tag's substitution."""
    refs = {}
    for i, (kind, stage, ev) in enumerate(prog.events()):
        base = ev.name.split("[", 1)[0]
        ref = doc.ports.weight_refs.get(base)
        if ref is None:
            continue
        tag = ev.name.split("[", 1)[1].rstrip("]") \
            if "[" in ev.name else ""
        subst = dict(kv.split("=") for kv in tag.split(",")) \
            if tag else {}
        refs[i] = (ref.factor,
                   [subst.get(a, a) for a in ref.args],
                   ref.complement)
    return refs


def weight_text(w: float) -> str:
    """A weight as the wire carries it: the shortest text that reads back
    as the same double (repr; an integral value without its ".0"), so a
    C++ runtime samples on python's doubles exactly (the architect's
    ruling, runtime-binding step 4 — %.12g lost the last digits)."""
    w = float(w)
    if w.is_integer() and abs(w) < 2.0 ** 53:
        return str(int(w))
    return repr(w)


def emit_program(prog, steps: int,
                 refs: Optional[Dict[int, Ref]] = None,
                 ports=None) -> List[str]:
    """The artifact lines for one program. `ports` is a lang Ports
    (or None): declared inputs/outputs/guards/weights become the B
    manifest the consuming registry must satisfy."""
    out = []
    stage_idx = {s: i for i, s in enumerate(prog.stage_names)}
    out.append(f"N {len(prog.stage_names)} "
               f"{stage_idx[prog.init_stage]}")
    if ports is not None:
        for name in sorted(ports.inputs):
            out.append(f"B in {name}")
        for name in sorted(ports.outputs):
            out.append(f"B out {name}")
        for name in sorted(ports.guards):
            out.append(f"B guard {name}")
        for name in sorted(ports.weights):
            out.append(f"B weight {name}")

    def sections(ev):
        s = ""
        for tag, d in (("c", ev.consume), ("o", ev.produce),
                       ("p", ev.persist)):
            if d:
                s += f" ; {tag}"
                for a, n in sorted(d.items()):
                    s += f" {a} {n}"
        return s

    refs = refs or {}
    for i, (kind, stage, ev) in enumerate(prog.events()):
        post = stage_idx[ev.post] if kind == "link" else -1
        wsec = ""
        if refs.get(i):
            factor, args, compl = refs[i]
            wsec = f" ; w {factor} {1 if compl else 0}"
            for a in args:
                wsec += f" {a}"
        out.append(f"E {stage_idx[stage]} {post} {weight_text(ev.weight)} "
                   f"{ev.name.replace(' ', '_')}{sections(ev)}{wsec}")
    init = " ".join(f"{a} {n}" for a, n in sorted(prog.init.items()))
    out.append(f"I {init}")
    out.append(f"S {steps}")
    return out


def emit_artifact(doc, prog, steps: int) -> str:
    """The complete product artifact for one compiled+grounded
    catalog: header comment, manifest, program. Write this to the
    file the C++ Evaluator serves (hot-reload on mtime change)."""
    lines = [f"# metis ground-wire artifact — catalog "
             f"{doc.catalog.name} v{doc.catalog.version}; "
             f"GENERATED, do not edit."]
    lines += emit_program(prog, steps,
                          refs=weight_refs(doc, prog),
                          ports=doc.ports)
    return "\n".join(lines) + "\n"
