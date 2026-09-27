"""The containment gate: the ported road catalog typechecks; a crafted
suite of violation packs is REJECTED (ill-typed, not lint-flagged)."""

import pytest

from metis.kernel import Catalog, CatalogError, ClauseSchema, Entry, \
    Provenance, Signature, ns, pat
from metis.domains.road import base_catalog, wheelchair_pack

PROV = Provenance("hand", "test", "2026-07-18")
LAYERS = ("detection", "object-behavior", "ego", "road", "rss", "pack")
STAGES = ("ego", "others", "check")


def pack_with(entries, preds=None, namespaces=None, bwd=frozenset(),
              types=None):
    return Catalog(name="crafted", version=1, layers=LAYERS,
                   stages=STAGES, sig=Signature(types or {}),
                   namespaces=namespaces or
                   {"pk": ns("pk", produce={"pack"})},
                   preds=preds if preds is not None else {"tok": "pk"},
                   bwd=bwd, entries=entries)


def entry(schema, comment="crafted violation", layer="pack"):
    return Entry(name=f"pack/{schema.name}", layer=layer, kind="schema",
                 comment=comment, provenance=PROV, payload=schema)


def test_road_catalog_typechecks():
    assert base_catalog().typecheck() == []


def test_wheelchair_pack_admitted():
    merged = base_catalog().admit(wheelchair_pack())
    assert merged.typecheck() == []
    assert "caution" in merged.preds


def test_pack_producing_kinematic_rejected():
    bad = pack_with([entry(ClauseSchema(
        "boost", stage="check",
        produce=[pat("st", "ego", "l0", "c4", "v2")]))])
    with pytest.raises(CatalogError, match="may not produce 'st'"):
        base_catalog().admit(bad)


def test_pack_consuming_kinematic_rejected():
    bad = pack_with([entry(ClauseSchema(
        "erase", stage="check",
        consume=[pat("st", "alfa", "l0", "c2", "v0")]))])
    with pytest.raises(CatalogError, match="may not consume 'st'"):
        base_catalog().admit(bad)


def test_pack_claiming_base_namespace_rejected():
    bad = pack_with([], preds={"boost": "kinematic"})
    with pytest.raises(CatalogError, match="claims base namespace"):
        base_catalog().admit(bad)


def test_pack_redeclaring_predicate_rejected():
    bad = pack_with([], preds={"st": "pk"})
    with pytest.raises(CatalogError, match="redeclares predicate"):
        base_catalog().admit(bad)


def test_guard_on_resource_predicate_rejected():
    bad = pack_with([entry(ClauseSchema(
        "peek", stage="check", consume=[pat("tok")],
        produce=[pat("tok")],
        guards=[pat("st", "ego", "l0", "c0", "v0")]))])
    with pytest.raises(CatalogError, match="guard on non-bwd"):
        base_catalog().admit(bad)


def test_bwd_predicate_as_resource_rejected():
    bad = pack_with([entry(ClauseSchema(
        "steal", stage="check",
        consume=[pat("adv", "c0", "v1", "c1")]))])
    with pytest.raises(CatalogError, match="used as a resource"):
        base_catalog().admit(bad)


def test_undeclared_predicate_rejected():
    bad = pack_with([entry(ClauseSchema(
        "ghost", stage="check", produce=[pat("mystery")]))])
    with pytest.raises(CatalogError, match="undeclared predicate"):
        base_catalog().admit(bad)


def test_missing_comment_rejected():
    bad = pack_with([entry(ClauseSchema(
        "mute", stage="check", consume=[pat("tok")]), comment="  ")])
    with pytest.raises(CatalogError, match="missing comment"):
        base_catalog().admit(bad)
