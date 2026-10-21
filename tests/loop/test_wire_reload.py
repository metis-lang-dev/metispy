"""The wire's `reload [PATH]` (loop-parity s-72d9d36b): an applied program that won't
load never replaces the one served — the program, its candidate and the session stay;
one that loads is served fresh, as a new open, with the host as attached (D4)."""
import io
import json

from metis.lang.repl import Repl
from metis.loop.wire import INTENT_VERBS, parse_intent, serve_json

A = "catalog a 1.\nlayers (d).\nstages (s).\nnamespace w produce (d).\npred x : w.\npred y : w.\n" \
    "stage s {\n  %% x becomes y.\n  xy [d] : x -o y.\n}\n#init x 1\n"
B = A.replace("catalog a 1.", "catalog b 1.").replace("#init x 1", "#init x 2")


def wire(shell, *reqs):
    out = io.StringIO()
    serve_json(shell, io.StringIO("".join(json.dumps(r) + "\n" for r in reqs)), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_reload_serves_another_program_fresh(tmp_path):
    (tmp_path / "a.llp").write_text(A)
    (tmp_path / "b.llp").write_text(B)
    sh = Repl(tmp_path / "a.llp", stdout=io.StringIO())
    got = wire(sh, {"op": "llp", "arg": "catalog cand 1.\n"}, {"op": "reload", "arg": str(tmp_path / "b.llp")},
               {"op": "show"}, {"op": "ctx"})
    assert got[1]["ok"] and got[1]["path"] == str(tmp_path / "b.llp") and got[1]["host"] is None
    assert got[1]["key"] != sh.key                                    # another program
    assert got[2]["file"].endswith("b.llp") and not got[2]["candidate"]   # fresh: no candidate carried
    assert got[3]["counts"]["x"] == 2


def test_a_program_that_wont_load_never_replaces_the_one_served(tmp_path):
    (tmp_path / "a.llp").write_text(A)
    (tmp_path / "bad.llp").write_text(A.replace("#init x 1", "#init x 1."))
    sh = Repl(tmp_path / "a.llp", stdout=io.StringIO())
    got = wire(sh, {"op": "llp", "arg": "catalog cand 1.\n"}, {"op": "reload", "arg": str(tmp_path / "bad.llp")},
               {"op": "show"}, {"op": "show", "arg": "llp"})
    assert got[1]["ok"] is False and got[1]["findings"] and got[1]["served"] == str(tmp_path / "a.llp")
    assert got[2]["file"].endswith("a.llp") and got[2]["candidate"]  # the program and its candidate stay
    assert got[3]["llp"] == "catalog cand 1.\n"
    nope = wire(Repl(tmp_path / "a.llp", stdout=io.StringIO()), {"op": "reload", "arg": str(tmp_path / "none.llp")})
    assert nope[0]["ok"] is False and nope[0]["findings"]


def test_reload_rereads_the_file_served_and_is_no_page_intent(tmp_path):
    f = tmp_path / "a.llp"
    f.write_text(A)
    sh = Repl(f, stdout=io.StringIO())
    k = sh.key
    f.write_text(B.replace("catalog b 1.", "catalog a 1."))
    got = wire(sh, {"op": "reload"}, {"op": "ctx"})
    assert got[0]["ok"] and got[0]["key"] != k and got[1]["counts"]["x"] == 2
    assert "reload" not in INTENT_VERBS and parse_intent(":reload /etc/x.llp") is None   # it names a file


def test_the_host_stays_as_attached(tmp_path):
    """A runtime process keeps the host its operator named on argv (D4); reload never names one."""
    from metis.domains.kitchen.model import CATALOG
    scene, spec = CATALOG.parent / "kitchen_host.llp", "metis.domains.kitchen.host:make"
    sh = Repl(scene, plugin=spec, host=True, stdout=io.StringIO())
    got = wire(sh, {"op": "reload"}, {"op": "reload", "arg": str(scene)})
    assert got[0]["ok"] and got[0]["host"] == spec and got[1]["host"] == spec
