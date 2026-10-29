"""The domain bundle — what the C++ runtime serves (runtime-binding
step 2; metis-agent docs/RUNTIME-BINDING.md §2): one directory per
domain package, the SERVED program (the program plus its extensions,
merged in order, so the key is the union's). The module keeps its name
(metis.kernel.artifact); the word is bundle (catalog-entry step 2).

    <dir>/                     default: domains/<n>/build/<n>.metis
      ground.wire              N/E/I/S/B — the runtime's load: compiled by
                               metisc (`metisc --wire PROGRAM --extension E...`,
                               $METISC or on PATH; bundle-compiler s-3985caf6),
                               OCaml-authoritative; without metisc, metispy's
                               wire.emit_artifact covers (byte-equal, gate 4),
                               and the log says so
      sidecar.json             what the harness needs beyond the wire, keyed
                               to it by event index (below)
      tools.json               the REPL API's tool declarations (metis.lang.repl_api,
                               repl-tools step 3): what the Go runtime renders the
                               orchestrator's REPL tools from; a bundle without it
                               is stale (re-emit)
      program.ir               the product IR (ir.ir_artifact_lines) — only
                               with --ir: marginals and the planner; no
                               runtime row needs it, and it factorizes only
                               a port-free program whose queries are variables

sidecar.json ("format": "metis-sidecar/1"):
  catalog, key      the served catalog's name and program_key (records,
                    replies, the KV prefix cache)
  extensions        the merged extensions' file names, in order
  init, steps       the case (#init, #steps) the program is grounded on;
                    init in the program's #init order (the live counts'
                    insertion order, which Go decodes order-preserving)
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
  script            cards.script(shell, extensions): the narrator's and
                    the consult's text — the extensions' clauses follow
                    the program's in the served card, each tagged
                    "extension": <file name>
  ir                "program.ir" or null

    python -m metis.kernel.artifact <domain|file.llp> [-o DIR] [--extension E]... [--ir]

A domain name is resolved through metis_catalog (installed, or the
checkout at $METIS_CATALOG, else the sibling ../metis-catalog): its
manifest's served program, Domain.serves() — `serves: [files]`, else
the program and its extensions. A .llp path serves that file with
the --extension files given.

The swap is atomic: `<dir>` is a symlink to a versioned directory
`<dir>.v-<hash12>` (the files' content hash). A new version is written
whole beside it, then the link is replaced (os.replace); the previous
version stays one swap longer for a reader that resolved it. A reader
resolves `<dir>` ONCE and reads every file under that path, so
ground.wire and sidecar.json always come from the same version."""

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


def served_doc(program: Path, extensions: Sequence[Path] = (), manifest=None):
    """(doc, case, text): the program compiled with its includes, every
    extension compiled against the world so far and admitted over it,
    the ports merged — the loop's gate → accept chain, structurally (the
    package's extensions were gated when they were accepted). An
    extension that does not admit raises with its findings. `manifest`: the
    package's domain.yaml (or its dict): its levels' generated declarations
    join the program's (metis-lang spec/16 amendment 1)."""
    from metis.lang.compiler import (compile_catalog, load_llp,
                                     merge_ports, resolve_includes)
    from metis.lang.directives import embedded_case
    from metis.lang.levels import with_levels
    from metis.lang.parser import parse
    program = Path(program)
    text = program.read_text()
    doc = compile_catalog(with_levels(resolve_includes(parse(text, file=str(program)),
                                                       program.parent), manifest))
    for p in extensions:
        ext = load_llp(str(p), base=doc)
        doc = replace(doc, catalog=doc.catalog.admit(ext.catalog),
                      ports=merge_ports(doc.ports, ext.ports))
    return doc, embedded_case(text), text


def rule_lines(files: Sequence[Path]) -> dict:
    """rule → (its %% doc in words, its tags), over the files with their
    includes spliced (an extension's rules speak their own lines)."""
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


def sidecar(program: Path, extensions: Sequence[Path], doc, case, prog,
            ir: Optional[str] = None) -> dict:
    from metis.kernel.nets import program_key
    from metis.kernel.refsample import canonical_events
    from metis.loop.cards import cards, script
    lines = rule_lines([program, *extensions])
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
            "extensions": [Path(p).name for p in extensions],
            "init": dict(case["init"]),
            "steps": case["steps"],
            "stages": list(prog.stage_names),
            "events": events,
            "cards": offered,
            "queries": list(case["queries"]),
            "script": script(shell, extensions),
            "ir": ir}


def ir_lines(doc, case, prog) -> List[str]:
    """program.ir: the product IR of a port-free program at its #steps
    (6 when none), its #query atoms the bundle's queries."""
    from metis.kernel.factorize import FactoredProgram
    from metis.kernel.ir import ir_artifact_lines
    from metis.lang import factor_hooks
    hooks = factor_hooks(prog, doc.ports, {}, doc.catalog.horn_db())
    fp = FactoredProgram(prog, case["steps"] or 6, hooks,
                         unknowns=tuple(case["unknowns"]))
    return ir_artifact_lines(doc.catalog.name, fp, list(case["queries"]),
                             ports=doc.ports)


def metisc() -> Optional[str]:
    """The OCaml compiler that makes the bundle's ground.wire: $METISC, else
    `metisc` on PATH; None when neither is there (metispy then covers)."""
    m = os.environ.get("METISC")
    if m:
        return m
    return shutil.which("metisc")


def wire(program: Path, extensions: Sequence[Path], doc, prog, steps, manifest=None):
    """(ground.wire text, who made it): `metisc --wire` when metisc is
    there — the compiler the bundle points at (bundle-compiler s-3985caf6);
    a metisc that fails is an error, never covered — else metispy's
    emit_artifact, the cover (byte-equal: tests/test_ocaml_parity gate 4)."""
    import logging
    import subprocess
    from metis.kernel.wire import emit_artifact
    m = metisc()
    if m:
        cmd = [m, "--wire", str(program)]
        for e in extensions:
            cmd += ["--extension", str(e)]
        if manifest is not None and not isinstance(manifest, dict):
            cmd += ["--manifest", str(manifest)]
        from metis.catalog import metisc_env
        r = subprocess.run(cmd, capture_output=True, text=True, env=metisc_env())   # the same includes: metisc's own standard units
        if r.returncode != 0:
            raise RuntimeError(f"metisc --wire {Path(program).name}: exit "
                               f"{r.returncode}: {(r.stderr or r.stdout).strip()[:400]}")
        by = f"metisc ({m})"
        text = r.stdout
    else:
        by = "metispy emit_artifact (cover: no $METISC, no metisc on PATH)"
        text = emit_artifact(doc, prog, steps)
    logging.getLogger("metis.bundle").info("ground.wire of %s: %s",
                                           Path(program).name, by)
    return text, by


def build(program: Path, extensions: Sequence[Path] = (), with_ir=False,
          report: Optional[dict] = None, manifest=None) -> dict:
    """{file name: text} of the bundle; report, when given, gets
    report["wire_by"] = who compiled ground.wire (not a bundle file: the
    bundle is byte-equal either way)."""
    from metis.lang.repl_api import tools_json
    doc, case, _ = served_doc(program, extensions, manifest)
    prog = doc.catalog.ground(dict(case["init"]))
    text, by = wire(program, extensions, doc, prog, case["steps"], manifest)
    if report is not None:
        report["wire_by"] = by
    files = {"ground.wire": text}
    if with_ir:
        files["program.ir"] = "\n".join(ir_lines(doc, case, prog)) + "\n"
    side = sidecar(program, extensions, doc, case, prog,
                   ir="program.ir" if with_ir else None)
    files["sidecar.json"] = json.dumps(side, indent=1,
                                       ensure_ascii=False) + "\n"
    files["tools.json"] = tools_json()
    return files


def _umask() -> int:
    m = os.umask(0o022)
    os.umask(m)
    return m


def write(out: Path, files: dict) -> Path:
    """Publish `files` as the bundle `out`, atomically: the versioned
    directory `<out>.v-<hash12>` (written in full under a temporary name,
    then renamed into place, mode as the umask makes it — never mkdtemp's
    0700), then the symlink `out` replaced to point at it. Versions other
    than the new one and the one it replaced are removed. A real
    directory at `out` (an older emitter's) is replaced once.
    -> the versioned directory."""
    import hashlib
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name].encode() + b"\0")
    ver = out.with_name(f"{out.name}.v-{h.hexdigest()[:12]}")
    if not ver.is_dir():
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
        os.chmod(tmp, 0o777 & ~_umask())
        for name, text in files.items():
            (tmp / name).write_text(text)
        try:
            tmp.rename(ver)
        except OSError:                  # a concurrent writer made it
            shutil.rmtree(tmp)
    prev = os.readlink(out) if out.is_symlink() else None
    if out.exists() and not out.is_symlink():
        shutil.rmtree(out)               # an older emitter's real directory
    link = out.with_name(f".{out.name}.link.{os.getpid()}")
    if link.is_symlink():
        link.unlink()
    os.symlink(ver.name, link)           # relative: the build dir can move
    os.replace(link, out)
    keep = {ver.name, prev}
    for old in out.parent.glob(f"{out.name}.v-*"):
        if old.name not in keep:
            shutil.rmtree(old, ignore_errors=True)
    return ver


def _catalog():
    """The loader a domain NAME needs (a .llp path does not), metis.catalog.manifests
    (was metis_catalog; metis-packages s-65778b98) over the catalog $METIS_CATALOG names,
    else the cwd when it is a catalog (it has domains/: the catalog is the pwd), else the
    installed metis-catalog's own domains, else a metis-catalog checkout beside this source
    tree (the dev layout): its get(name) and domains(). Without any, a clear error, not a
    bare ImportError (a wheel install has no sibling)."""
    from metis.catalog import manifests
    env = os.environ.get("METIS_CATALOG")
    own = manifests.data_dir()
    if env:
        root = Path(env)
    elif (Path.cwd() / "domains").is_dir():
        root = Path.cwd()
    elif own is not None:
        root = own if (own / "domains").is_dir() else own.parent
    else:
        root = Path(__file__).resolve().parents[3] / "metis-catalog"
    if not (root / "domains").is_dir():
        raise RuntimeError(
            "a domain name needs metis-catalog: install it (pip install -e "
            "metis-catalog), or set METIS_CATALOG to a metis-catalog checkout "
            "(%s has none); a .llp path builds without it" % root)
    return SimpleNamespace(get=lambda name: manifests.get(name, root),
                           domains=lambda: manifests.domains(root))


def resolve(target: str, extensions: Sequence[str] = ()):
    """(program, extensions, default out dir) for a domain name or a file."""
    p = Path(target)
    if p.suffix == ".llp":
        return p.resolve(), [Path(x).resolve() for x in extensions], \
            p.parent / "build" / f"{p.stem}.metis"
    d = _catalog().get(target)
    files = d.serves()                 # the manifest's served program
    if not files:
        raise SystemExit(f"domain {target!r} serves no program")
    return files[0], files[1:] + [Path(x).resolve() for x in extensions], \
        d.dir / "build" / f"{d.name}.metis"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m metis.kernel.artifact",
        description="emit a domain package's bundle: ground.wire + sidecar.json "
                    "+ tools.json (+ program.ir), the program plus its extensions")
    ap.add_argument("target", help="a catalog domain name, or a .llp file")
    ap.add_argument("-o", "--out", type=Path,
                    help="the bundle directory (default: "
                         "domains/<n>/build/<n>.metis)")
    ap.add_argument("--extension", action="append", default=[],
                    help="an extension to merge after the package's own (repeat)")
    ap.add_argument("--ir", action="store_true",
                    help="also write program.ir (a port-free program)")
    a = ap.parse_args(argv)
    program, extensions, out = resolve(a.target, a.extension)
    manifest = None if Path(a.target).suffix == ".llp" else _catalog().get(a.target).dir / "domain.yaml"
    rep = {}
    files = build(program, extensions, with_ir=a.ir, report=rep, manifest=manifest)
    write(a.out or out, files)
    side = json.loads(files["sidecar.json"])
    print(f"{a.out or out}: {side['catalog']} key {side['key'][:12]}… "
          f"{len(side['events'])} events, extensions {side['extensions'] or '-'}, "
          f"{', '.join(sorted(files))}; ground.wire by {rep['wire_by']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
