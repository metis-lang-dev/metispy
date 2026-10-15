"""The domain artifact — what the C++ runtime serves (runtime-binding
step 2; metis-agent docs/RUNTIME-BINDING.md §2): one directory per
domain, the SERVED program (the program plus its packs, merged in order,
so the key is the union's).

    <dir>/                     default: domains/<n>/build/<n>.metis
      ground.wire              N/E/I/S/B (wire.emit_artifact) — the runtime's load
      sidecar.json             what the harness needs beyond the wire, keyed
                               to it by event index (below)
      program.ir               the product IR (ir.ir_artifact_lines) — only
                               with --ir: marginals and the planner; no
                               runtime row needs it, and it factorizes only
                               a port-free program whose queries are variables

sidecar.json ("format": "metis-sidecar/1"):
  catalog, key      the served catalog's name and program_key (records,
                    replies, the KV prefix cache)
  packs             the merged packs' file names, in order
  init, steps       the case (#init, #steps) the program is grounded on
  stages            the wire's stage indices by name
  events            per wire event i (Program.events() order, the
                    wire's): canon (its index in the normative sampler's
                    order, refsample.canonical_events — the law the live
                    runtime samples by), name, rule, kind, stage, line
                    (the rule's %% doc in words, cards._words), and
                    target when the doc tags one
  cards             the cards offerable at the served node (cards.cards,
                    without the file paths): what explain may adopt
  queries           the #query atoms (the mirror line)
  script            cards.script(shell, packs): the narrator's and the
                    consult's text — the packs' clauses follow the
                    program's in the served card, each tagged "pack"
  ir                "program.ir" or null

    python -m metis.kernel.artifact <domain|file.llp> [-o DIR] [--pack P]... [--ir]

A domain name is resolved through metis_catalog (installed, or the
checkout at $METIS_CATALOG, else the sibling ../metis-catalog): its
manifest's served program, Domain.serves() — `serves: [files]`, else
the program and every packs/*.llp. A .llp path serves that file with
the --pack files given. The directory is replaced whole (written
beside it, then swapped), so a reader never sees half an artifact."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

FORMAT = "metis-sidecar/1"


def served_doc(program: Path, packs: Sequence[Path] = ()):
    """(doc, case, text): the program compiled with its includes, every
    pack compiled against the world so far and admitted over it, the
    ports merged — the loop's gate → accept chain, structurally (the
    packs in packs/ were gated when they were accepted). A pack that
    does not admit raises with its findings."""
    from metis.lang.compiler import (compile_catalog, load_llp,
                                     merge_ports, resolve_includes)
    from metis.lang.directives import embedded_case
    from metis.lang.parser import parse
    program = Path(program)
    text = program.read_text()
    doc = compile_catalog(resolve_includes(parse(text, file=str(program)),
                                           program.parent))
    for p in packs:
        pack = load_llp(str(p), base=doc)
        doc = replace(doc, catalog=doc.catalog.admit(pack.catalog),
                      ports=merge_ports(doc.ports, pack.ports))
    return doc, embedded_case(text), text


def rule_lines(files: Sequence[Path]) -> dict:
    """rule → (its %% doc in words, its tags), over the files with their
    includes spliced (a pack's rules speak their own lines)."""
    from metis.lang import ast
    from metis.lang.compiler import resolve_includes
    from metis.lang.parser import parse_file
    from metis.loop.cards import _words
    out = {}
    for f in files:
        cat = resolve_includes(parse_file(str(f)), Path(f).parent)
        for d in cat.decls:
            if isinstance(d, ast.Stage):
                for r in d.rules:
                    out[r.name] = _words(r.doc)
    return out


_RULE = re.compile(r"[\[(]")


def sidecar(program: Path, packs: Sequence[Path], doc, case, prog,
            ir: Optional[str] = None) -> dict:
    from metis.kernel.nets import program_key
    from metis.kernel.refsample import canonical_events
    from metis.loop.cards import cards, script
    lines = rule_lines([program, *packs])
    canon_tbl = canonical_events(prog)
    canon = {id(ev): j for j, (_, _, ev) in enumerate(canon_tbl)}
    events = []
    for i, (kind, stage, ev) in enumerate(prog.events()):
        rule = _RULE.split(ev.name, 1)[0]
        words, tags = lines.get(rule, ("", {}))
        e = {"i": i, "canon": canon[id(ev)], "name": ev.name,
             "rule": rule, "kind": kind, "stage": stage, "line": words}
        if tags.get("target"):
            e["target"] = tags["target"]
        events.append(e)
    # the harness's view of the served node (cards/script read a shell)
    shell = SimpleNamespace(path=Path(program), doc=doc, case=case,
                            prog=prog, events_tbl=canon_tbl)
    offered = [{k: v for k, v in c.items() if k != "file"}
               for c in cards(shell)]
    return {"format": FORMAT,
            "catalog": doc.catalog.name,
            "key": program_key(prog),
            "packs": [Path(p).name for p in packs],
            "init": dict(sorted(case["init"].items())),
            "steps": case["steps"],
            "stages": list(prog.stage_names),
            "events": events,
            "cards": offered,
            "queries": list(case["queries"]),
            "script": script(shell, packs),
            "ir": ir}


def ir_lines(doc, case, prog) -> List[str]:
    """program.ir: the product IR of a port-free program at its #steps
    (6 when none), its #query atoms the artifact's queries."""
    from metis.kernel.factorize import FactoredProgram
    from metis.kernel.ir import ir_artifact_lines
    from metis.lang import factor_hooks
    hooks = factor_hooks(prog, doc.ports, {}, doc.catalog.horn_db())
    fp = FactoredProgram(prog, case["steps"] or 6, hooks,
                         unknowns=tuple(case["unknowns"]))
    return ir_artifact_lines(doc.catalog.name, fp, list(case["queries"]),
                             ports=doc.ports)


def build(program: Path, packs: Sequence[Path] = (), with_ir=False) -> dict:
    """{file name: text} of the artifact."""
    from metis.kernel.wire import emit_artifact
    doc, case, _ = served_doc(program, packs)
    prog = doc.catalog.ground(dict(case["init"]))
    files = {"ground.wire": emit_artifact(doc, prog, case["steps"])}
    if with_ir:
        files["program.ir"] = "\n".join(ir_lines(doc, case, prog)) + "\n"
    side = sidecar(program, packs, doc, case, prog,
                   ir="program.ir" if with_ir else None)
    files["sidecar.json"] = json.dumps(side, indent=1,
                                       ensure_ascii=False) + "\n"
    return files


def write(out: Path, files: dict) -> None:
    """Replace `out` whole: the files go to a sibling directory first,
    then the two are swapped (a watcher sees the old or the new)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
    for name, text in files.items():
        (tmp / name).write_text(text)
    old = None
    if out.exists():
        old = out.with_name(f"{out.name}.old.{os.getpid()}")
        out.rename(old)
    tmp.rename(out)
    if old is not None:
        shutil.rmtree(old)


def _catalog():
    try:
        import metis_catalog
    except ImportError:
        root = os.environ.get("METIS_CATALOG") or str(
            Path(__file__).resolve().parents[3] / "metis-catalog")
        sys.path.insert(0, root)
        import metis_catalog
    return metis_catalog


def resolve(target: str, packs: Sequence[str] = ()):
    """(program, packs, default out dir) for a domain name or a file."""
    p = Path(target)
    if p.suffix == ".llp":
        return p.resolve(), [Path(x).resolve() for x in packs], \
            p.parent / "build" / f"{p.stem}.metis"
    d = _catalog().get(target)
    files = d.serves()                 # the manifest's served program
    if not files:
        raise SystemExit(f"domain {target!r} serves no program")
    return files[0], files[1:] + [Path(x).resolve() for x in packs], \
        d.dir / "build" / f"{d.name}.metis"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m metis.kernel.artifact",
        description="emit a domain's artifact: ground.wire + sidecar.json "
                    "(+ program.ir), the program plus its packs")
    ap.add_argument("target", help="a catalog domain name, or a .llp file")
    ap.add_argument("-o", "--out", type=Path,
                    help="the artifact directory (default: "
                         "domains/<n>/build/<n>.metis)")
    ap.add_argument("--pack", action="append", default=[],
                    help="a pack to merge after the domain's own (repeat)")
    ap.add_argument("--ir", action="store_true",
                    help="also write program.ir (a port-free program)")
    a = ap.parse_args(argv)
    program, packs, out = resolve(a.target, a.pack)
    files = build(program, packs, with_ir=a.ir)
    write(a.out or out, files)
    side = json.loads(files["sidecar.json"])
    print(f"{a.out or out}: {side['catalog']} key {side['key'][:12]}… "
          f"{len(side['events'])} events, packs {side['packs'] or '-'}, "
          f"{', '.join(sorted(files))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
