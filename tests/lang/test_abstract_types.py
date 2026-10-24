"""Abstract types (spec 12): `types (kind domain).` forward-declares the
types a lifted domain file uses and its instance completes. The file
compiles alone (an advisory), grounding it is refused (type-incomplete
at the `types` line); a signature naming a type nobody declared is
type-unknown, once, at that signature. metisc's side is metis-lang
gate 11 (diag_abstract.llp; the same goldens)."""

import pytest

from metis.kernel import CatalogError
from metis.lang import ast
from metis.lang.compiler import LangError, compile_catalog, diagnose, \
    load_llp
from metis.lang.parser import parse

DOMAIN = """catalog dom 1.
layers (dyn).
stages (act).
types (kind domain).
namespace dir produce (dyn).
bwd handles(domain, kind).
pred seen(kind) : dir.
pred active(domain) : dir.
stage act {
  %% an arrival of a kind its handler is awake for is seen again.
  see [dyn] : seen(K) * $active(D) * handles(D, K) -o seen(K).
}
"""

INSTANCE = """catalog inst 1.
include "dom.llp".
type domain {nav task}.
type kind {goto done}.
handles(nav, goto).
"""


def test_the_declaration_parses_and_prints_back():
    cat = parse(DOMAIN, file="dom.llp")
    (td,) = [d for d in cat.decls if isinstance(d, ast.TypesDecl)]
    assert td.names == ("kind", "domain") and (td.line, td.col) == (4, 8)
    assert parse(ast.pretty(cat)) == cat
    assert "types (kind domain)." in ast.pretty(cat).splitlines()


def test_types_is_an_identifier_where_no_paren_follows():
    cat = parse("catalog t 1.\nlayers (s).\nstages (go).\ntype x {a}.\n"
                "bwd types(x).\ntypes(a).\n")
    assert [type(d).__name__ for d in cat.decls] == \
        ["TypeDecl", "BwdDecl", "TypesDecl"]
    cat = parse("catalog t 1.\nbwd types.\ntypes.\n")
    assert [type(d).__name__ for d in cat.decls] == ["BwdDecl", "Fact"]


def test_a_lifted_file_compiles_alone_with_an_advisory():
    doc = compile_catalog(parse(DOMAIN, file="dom.llp"))
    assert doc.catalog.abstract == ("kind", "domain")
    assert doc.catalog.sig.types["kind"] == []
    assert doc.warnings == [
        "type 'kind' is abstract (types (kind domain)): this unit grounds "
        "only through an instance that completes it",
        "type 'domain' is abstract (types (kind domain)): this unit grounds "
        "only through an instance that completes it"]
    assert [d.code for d in doc.diagnostics] == ["type-abstract"] * 2


def test_grounding_it_is_refused_at_the_types_line():
    doc = compile_catalog(parse(DOMAIN, file="dom.llp"))
    with pytest.raises(CatalogError) as e:
        doc.catalog.ground({})
    assert e.value.findings[0] == (
        "type 'kind' has no members: declared by `types (kind domain)`, "
        "completed by no `type kind {…}` — the instance that includes this "
        "file supplies it")
    d = e.value.diagnostics
    assert [x.code for x in d] == ["type-incomplete"] * 2
    assert [dict(x.data)["type"] for x in d] == ["kind", "domain"]
    assert (d[0].loc.file, d[0].loc.line, d[0].loc.decl) == \
        ("dom.llp", 4, "types/kind")
    assert d[0].subject == ("types (kind domain).",)


def test_the_instance_that_includes_it_grounds(tmp_path):
    (tmp_path / "dom.llp").write_text(DOMAIN)
    (tmp_path / "inst.llp").write_text(INSTANCE)
    doc = load_llp(tmp_path / "inst.llp")
    assert doc.catalog.abstract == () and doc.warnings == []
    assert doc.catalog.sig.types["kind"] == ["goto", "done"]
    prog = doc.catalog.ground({"seen(goto)": 1, "active(nav)": 1})
    assert [c.name for c in prog.stages["act"]] == ["see[K=goto,D=nav]"]
    assert diagnose(tmp_path / "inst.llp") == []


def test_a_completion_anywhere_in_the_unit_completes_it():
    doc = compile_catalog(parse(DOMAIN + "type kind {a b}.\n"))
    assert doc.catalog.abstract == ("domain",)
    assert doc.catalog.sig.types["kind"] == ["a", "b"]
    assert len(doc.warnings) == 1 and "'domain'" in doc.warnings[0]


def test_repeats_are_a_set_and_a_definition_is_still_one_decl():
    doc = compile_catalog(parse(DOMAIN.replace(
        "types (kind domain).", "types (kind domain).\ntypes (kind).")))
    assert doc.catalog.abstract == ("kind", "domain")
    with pytest.raises(LangError) as e:
        compile_catalog(parse(DOMAIN + "type kind {a}.\ntype kind {b}.\n"))
    assert [d.code for d in e.value.diagnostics if d.severity == "error"] \
        == ["decl-duplicate"]


def test_a_constant_at_an_abstract_position_waits_for_the_instance(
        tmp_path):
    src = DOMAIN + "handles(nav, goto).\n"
    assert compile_catalog(parse(src)).catalog.abstract == \
        ("kind", "domain")
    (tmp_path / "dom.llp").write_text(src)
    (tmp_path / "inst.llp").write_text(
        'catalog inst 1.\ninclude "dom.llp".\n'
        'type domain {task}.\ntype kind {goto}.\n')
    with pytest.raises(LangError) as e:
        load_llp(tmp_path / "inst.llp")
    assert [d.code for d in e.value.diagnostics] == ["const-not-in-type"]


def test_an_undeclared_type_is_one_finding_at_the_signature():
    with pytest.raises(LangError) as e:
        compile_catalog(parse(DOMAIN.replace("types (kind domain).\n", ""),
                              file="dom.llp"))
    d = e.value.diagnostics
    assert [x.code for x in d] == ["type-unknown"] * 2
    assert e.value.findings == [
        "bwd handles: 'domain' is not a type — declare it: `types (domain).` "
        "when an instance completes it, or `type domain {…}`.",
        "bwd handles: 'kind' is not a type — declare it: `types (kind).` "
        "when an instance completes it, or `type kind {…}`."]
    assert (d[0].loc.line, d[0].loc.decl, d[0].subject) == \
        (5, "bwd/handles", ("bwd handles(domain,kind).",))
    assert dict(d[0].data) == {"pred": "handles", "type": "domain"}


def test_a_pack_may_complete_its_bases_abstract_type():
    """spec 12 P5. The base's constants at the completed positions are
    not checked again here (a pack compiles against the base, it does
    not recompile it); an instance that includes the file checks them."""
    base = compile_catalog(parse(DOMAIN + "handles(nav, goto).\n"))
    pack = compile_catalog(parse(
        "catalog p 1.\nextends dom.\ntype kind {goto}.\n"
        "type domain {nav}.\n"), base=base)
    merged = base.catalog.admit(pack.catalog)
    assert merged.abstract == ()
    assert merged.sig.types["kind"] == ["goto"]
    prog = merged.ground({"seen(goto)": 1, "active(nav)": 1})
    assert [c.name for c in prog.stages["act"]] == ["see[K=goto,D=nav]"]


def test_a_pack_still_may_not_redeclare_a_defined_type():
    base = compile_catalog(parse(DOMAIN + "type kind {a}.\n"))
    pack = compile_catalog(parse(
        "catalog p 1.\nextends dom.\ntype kind {b}.\n"), base=base)
    with pytest.raises(CatalogError) as e:
        base.catalog.admit(pack.catalog)
    assert [d.code for d in e.value.diagnostics] == ["pack-redeclares"]


def test_the_refusal_is_metiscs_byte_for_byte():
    """metis-lang gate 11: `metisc --run diag_abstract.llp` prints the
    advisory and the finding on stderr and exits 1; metispy says the same
    words (its CLI surfaces: spec 12 step 2)."""
    from pathlib import Path
    corpus = Path(__file__).resolve().parents[3] / "metis-lang/corpus"
    doc = load_llp(corpus / "diag/diag_abstract.llp")
    with pytest.raises(CatalogError) as e:
        doc.catalog.ground({})
    got = [f"warning: {w}" for w in doc.warnings] + \
        [f"error: {f}" for f in e.value.findings]
    assert "\n".join(got) + "\n" == \
        (corpus / "goldens/abstract_run_goldens.txt").read_text()


# -- step 2: the surfaces — the metis shell serves a lifted file, refuses
#    only what needs its program; cannot-serve lines carry file:line ----

def _shell(path):
    import io
    from metis.lang.repl import Repl
    return Repl(path, seed=1, stdout=io.StringIO())


def _run(r, *cmds):
    import io
    for c in cmds:
        r.onecmd(c)
    out, r.stdout = r.stdout.getvalue(), io.StringIO()
    return out


def test_the_shell_serves_a_lifted_file_and_lists_its_rules(tmp_path):
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN + "#init seen(goto) 1\n#steps 3\n")
    r = _shell(p)
    assert r.prog is None and r.counts == {}       # no program, no #init
    assert "lifted: kind domain abstract" in r.intro
    assert r.intro.startswith("warning: type 'kind' is abstract")
    out = _run(r, "list")
    assert "type   kind: abstract, declared at dom.llp:4" in out
    assert "rules  1 (no program: none enabled)" in out
    assert "see [dyn] : seen(K) * $active(D) * handles(D, K)" in out
    assert _run(r, "key").strip() == \
        "no key: a lifted file has no program (spec 12)"
    assert [ln.split(". ", 1)[1].split(" @")[0]
            for ln in _run(r, "findings").splitlines()] == \
        ["warning type-abstract"] * 2 + ["error type-incomplete"] * 2


def test_the_shell_refuses_what_needs_the_program(tmp_path):
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN)
    r = _shell(p)
    for verb in ("run 3", "step", "state", "events", "query seen(goto)",
                 "assert seen(goto)", "reset"):
        out = _run(r, verb)
        assert out.startswith(f"{verb.split()[0]}: no program — dom.llp is "
                              "lifted; ground the instance that includes it")
        assert "\n  dom.llp:4: type 'kind' has no members" in out
    assert r.fired == [] and r.counts == {}


def test_reload_moves_between_lifted_and_served(tmp_path):
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN)
    r = _shell(p)
    p.write_text(DOMAIN + "type kind {goto}.\ntype domain {nav}.\n"
                 "handles(nav, goto).\n#init seen(goto) 1\n"
                 "#init active(nav) 1\n")
    _run(r, "reload")
    assert r.prog is not None and r.counts == {"seen(goto)": 1,
                                               "active(nav)": 1}
    assert "see[K=goto,D=nav]" in _run(r, "events")
    p.write_text(DOMAIN)
    assert "lifted: kind domain abstract, no program" in _run(r, "reload")
    assert r.prog is None


def test_the_wire_answers_a_lifted_file_the_same(tmp_path):
    from metis.loop.wire import dispatch
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN)
    r = _shell(p)
    res = dispatch(r, "step", "")
    assert res["ok"] is False and res["lifted"] == ["kind", "domain"]
    assert "dom.llp:4: type 'kind' has no members" in res["error"]
    assert dispatch(r, "findings", "")["ok"] is not False


def test_no_bundle_is_built_from_a_lifted_file(tmp_path):
    """the architect's ruling on step 1 (5): every emitter grounds through
    the one refusing check."""
    from metis.kernel.artifact import build
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN + "#steps 3\n")
    with pytest.raises(CatalogError) as e:
        build(p)
    assert [d.code for d in e.value.diagnostics] == ["type-incomplete"] * 2


def test_cannot_serve_lines_carry_the_location(tmp_path):
    from metis.lang.parser import ParseError
    from metis.lang.repl import located
    p = tmp_path / "dom.llp"
    p.write_text(DOMAIN.replace("types (kind domain).\n", ""))
    with pytest.raises(LangError) as e:
        load_llp(p)
    assert located(e.value, "dom.llp") == [
        "dom.llp:5: bwd handles: 'domain' is not a type — declare it: "
        "`types (domain).` when an instance completes it, or "
        "`type domain {…}`.",
        "dom.llp:5: bwd handles: 'kind' is not a type — declare it: "
        "`types (kind).` when an instance completes it, or "
        "`type kind {…}`."]
    p.write_text("catalog bad 1.\nlayers (s.\n")
    with pytest.raises(ParseError) as e:
        load_llp(p)
    assert located(e.value, "bad.llp") == \
        ["bad.llp:2:10: expected ident, got '.'"]


def test_cannot_serve_is_metiscs_compile_file_byte_for_byte():
    """metis-lang gate 11: `metisc FILE.llp` over three fixtures — the
    lifted one passes (its advisory, the ok line), the failing ones list
    `file:line: text` per error; metispy's cannot-serve lines (located)
    say the same."""
    import os
    from pathlib import Path
    from metis.lang.repl import located
    corpus = Path(__file__).resolve().parents[3] / "metis-lang/corpus"
    here = os.getcwd()
    os.chdir(corpus / "diag")
    got = []
    try:
        for f in ("diag_abstract.llp", "diag_compile.llp",
                  "diag_kernel.llp"):
            try:
                doc = load_llp(f)
            except (LangError, CatalogError) as e:
                got += located(e, f) + ["rc 1"]
                continue
            c = doc.catalog
            got += [f"warning: {w}" for w in doc.warnings]
            got.append(f"ok {c.name}: {len(c.entries)} entries"
                       + (f"; abstract {' '.join(c.abstract)} (grounds "
                          "through an instance that completes it)"
                          if c.abstract else ""))
            got.append("rc 0")
    finally:
        os.chdir(here)
    assert "\n".join(got) + "\n" == \
        (corpus / "goldens/compile_file_goldens.txt").read_text()
