"""The card a package's mapper reads is the one its manifest names (intent: {spec}), as Domain.intent resolves it
(alf-bundle s-b53cc7ab step 2, the architect's ruling A): a bundle run fills a generated card and the committed
intent.yaml stays its template. No intent key: intent.yaml, as before."""
from metis.domain.package import build


def _package(d, manifest):
    d.mkdir()
    (d / "domain.yaml").write_text("program: p.llp\n" + manifest)
    (d / "intent.yaml").write_text("map:\n  - {verb: GO, atom: template}\n")
    (d / "intent").mkdir()
    (d / "intent" / "card.yaml").write_text("map:\n  - {verb: GO, atom: filled}\n")


def test_the_mapper_reads_the_card_the_manifest_names(tmp_path):
    _package(tmp_path / "p", "intent: {spec: intent/card.yaml}\n")
    ns = build("p", dir=tmp_path / "p")
    assert [r["atom"] for r in ns["CARD"]["rows"]] == ["filled"] and ns["INTENT"]["map"][0]["atom"] == "filled"


def test_without_an_intent_key_the_mapper_reads_intent_yaml(tmp_path):
    _package(tmp_path / "p", "")
    assert [r["atom"] for r in build("p", dir=tmp_path / "p")["CARD"]["rows"]] == ["template"]
    _package(tmp_path / "q", "intent: {spec: intent.yaml}\n")
    assert [r["atom"] for r in build("q", dir=tmp_path / "q")["CARD"]["rows"]] == ["template"]
