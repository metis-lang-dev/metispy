"""The REPL API as data (repl-tools step 1): declared = dispatched — every
verb served by the REPL is one of its read commands, the surprise verbs
are exactly the watch's, and nothing is declared that no layer serves;
the schema is well formed; --tools-json prints the data."""
import json
import subprocess
import sys

from metis.lang.repl import Repl
from metis.lang.repl_api import REPL_API, repl_api, tools_json
from metis.loop.watch import Watch

SERVERS = {"repl", "watch", "agent", "tree", "orchestrator"}


def by(server):
    return {t["name"] for t in REPL_API if t["served_by"] == server}


def test_the_repl_verbs_are_its_read_commands():
    # a read the API declares is a REPL command (do_<name>), read-only
    for name in by("repl"):
        assert callable(getattr(Repl, "do_" + name, None)), name
    assert by("repl") == {"state", "ctx", "events", "why", "findings"}


def test_the_surprise_verbs_are_the_watchs():
    assert by("watch") == set(Watch.VERBS)


def test_every_verb_has_its_server_and_one_name():
    names = [t["name"] for t in REPL_API]
    assert len(names) == len(set(names))
    assert {t["served_by"] for t in REPL_API} <= SERVERS
    assert by("agent") == {"consult"} and by("tree") == {"activate", "deactivate"} and by("orchestrator") == {"keep"}


def test_the_schema_is_well_formed():
    for t in REPL_API:
        assert t["description"] and t["final"]
        assert "accepted" not in t["final"]      # a REPL reply is final (E13 step 1)
        p = t["parameters"]
        if p is None:
            continue
        assert p["type"] == "OBJECT" and set(p["required"]) <= set(p["properties"])
        for name, prop in p["properties"].items():
            assert prop["type"] in ("STRING", "INTEGER", "NUMBER", "BOOLEAN"), (t["name"], name)
            assert prop["description"]
            assert prop.get("enum_from") in (None, "loaded", "catalog")


def test_tools_json_is_the_data():
    out = subprocess.run([sys.executable, "-m", "metis.lang.repl", "--tools-json"],
                         capture_output=True, text=True, check=True).stdout
    assert out == tools_json()
    doc = json.loads(out)
    assert doc["format"] == "metis-repl-api/1" and doc["tools"] == repl_api()


def test_repl_api_is_a_copy():
    a = repl_api()
    a[0]["description"] = "changed"
    assert REPL_API[0]["description"] != "changed"
