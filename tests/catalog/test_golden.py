"""metis.domain.day and `python -m metis.catalog golden` (dagent-fixtures s-86bbd1e7 step 2): a
package's golden is what the package makes — the same catalog writes it byte for byte again, --check
passes on it, and an edit of the program that changes what the day says fails --check until the
golden is written again (the catalog's push rule). On a scratch copy of the installed catalog's plant,
with a day of its own (the plant's committed goldens are the catalog's, checked by its CI)."""
import json
import shutil
import subprocess
import sys

import pytest

from catalogs import installed_catalog

CAT = installed_catalog()
if not (CAT / "domains" / "plant").is_dir():
    pytest.skip(f"no metis-catalog plant at {CAT} (METIS_CATALOG)", allow_module_level=True)

DAY = [{"ports": {}, "id": "p0"}, {"clock": "beat"}, {"id": "c1", "domain": "plant", "verb": "CHECK", "args": {}},
       {"clock": "beat"}, {"tool": "state", "args": {}}, {"id": "w", "domain": "plant", "verb": "WATER", "args": {}}]
DAY += [{"clock": "beat"}] * 8 + [{"tool": "why", "args": {"atom": "dry"}}, {"tool": "consult", "args": {}, "id": "k"},
                                  {"world": ["-dry"], "id": "w1"}, {"clock": "beat"}, {"clock": "beat"}]
INTENTS = [{"verb": "CHECK", "args": {}}, {"verb": "WATER", "args": {}}, {"verb": "NO_SUCH_VERB", "args": {}}]


@pytest.fixture
def scratch(tmp_path):
    shutil.copytree(CAT / "domains" / "plant", tmp_path / "domains" / "plant",
                    ignore=shutil.ignore_patterns("records", "build", "goldens", "__pycache__"))
    g = tmp_path / "domains" / "plant" / "goldens"
    g.mkdir()
    (g / "day.jsonl").write_text("".join(json.dumps(l) + "\n" for l in DAY))
    (g / "intents.jsonl").write_text("".join(json.dumps(l) + "\n" for l in INTENTS))
    return tmp_path


def cli(root, *args):
    return subprocess.run([sys.executable, "-m", "metis.catalog", "--root", str(root), "golden", *args],
                          capture_output=True, text=True)


def test_the_golden_is_what_the_package_makes_and_an_edit_fails_the_check(scratch):
    g = scratch / "domains" / "plant" / "goldens"
    r = cli(scratch, "plant")
    assert r.returncode == 0, r.stderr
    first = {f: (g / f).read_text() for f in ("day.out.jsonl", "intents.out.jsonl")}
    assert cli(scratch, "plant").returncode == 0
    assert {f: (g / f).read_text() for f in first} == first           # the same catalog: byte for byte
    r = cli(scratch, "plant", "--check")
    assert r.returncode == 0 and "the goldens are what the package makes" in r.stdout, r.stdout + r.stderr

    rows = [json.loads(x) for x in first["day.out.jsonl"].splitlines()]
    assert rows[0]["ready"] == {"clock": "ready", "domain": "plant", "beat": 0} and "move dry out" in rows[0]["script"]
    assert [r["in"] for r in rows[1:-1]] == DAY and set(rows[-1]["end"]) == {"stage", "counts", "fired", "records"}
    said = [o for r in rows[1:-1] for o in r["out"]]
    assert {"kind": "result", "status": "done", "id": "c1"}.items() <= next(o for o in said if o.get("id") == "c1"
                                                                            and o.get("status") == "done").items()
    assert not any("voice" in o or "ack" in o for o in said)
    assert next(r for r in rows if r.get("in", {}).get("tool") == "consult")["out"][0]["kind"] == "error"   # the agent's
    assert all("metis_domain_day__" not in json.dumps(r) for r in rows)                                    # <host>
    ints = [json.loads(x) for x in first["intents.out.jsonl"].splitlines()]
    assert ints[0]["out"]["atom"] == "cmd_check" and "error" in ints[2]["out"]

    llp = scratch / "domains" / "plant" / "plant.llp"
    llp.write_text(llp.read_text().replace("my soil dries out.", "my soil dries out in the sun.", 1))
    r = cli(scratch, "plant", "--check")
    assert r.returncode == 1 and "day.out.jsonl" in r.stdout and "in the sun" in r.stdout, r.stdout + r.stderr
    assert cli(scratch, "plant").returncode == 0 and cli(scratch, "plant", "--check").returncode == 0


def test_a_package_without_a_day_says_so(tmp_path):
    shutil.copytree(CAT / "domains" / "plant", tmp_path / "domains" / "plant",
                    ignore=shutil.ignore_patterns("records", "build", "goldens", "__pycache__"))
    r = cli(tmp_path, "plant", "--check")
    assert r.returncode == 1 and "no goldens/day.jsonl" in r.stdout
