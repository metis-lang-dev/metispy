"""rack step 1 (metis-agent docs/RACK.md §5): the kitchen pair — one
kitchen, two cooks, two served domains on one env. The second cook's
onion is an EVENT on its own tape and a SURPRISE on the first cook's,
explained by the kitchen_shared card."""
import io

from metis.domains.kitchen.pair import SCENES, run_pair
from metis.lang.repl import Repl
from metis.loop.watch import Watch
from metis.loop.wire import dispatch


def test_the_onion_is_an_event_on_one_tape_and_a_surprise_on_the_other():
    r = run_pair(1)
    sigs = r["signals"]
    help_, = [s for s in sigs if s.get("name") == "partner_load"]
    assert help_["path"] == "kitchen_shared" and help_["signal"] == "event"
    sur, = [s for s in sigs if s["signal"] == "surprise"]
    assert sur["path"] == "kitchen"
    assert sur["diff"] == {"pot_space": -1, "pot_onion": 1}
    assert sur["t"] > help_["t"]                     # the event came first
    res, = [s for s in sigs if s["signal"] == "result"]
    assert res["status"] == "admitted" and res["card"] == "kitchen_shared"
    assert r["truth"] == [{"kind": "partner_load", "by": "kitchen_shared",
                           "order": 2, "beat": help_["beat"]}]
    # orders on both; the first cook serves all three of its orders
    exo = {p: sum(1 for s in sigs if s["path"] == p and s.get("kind") == "exo")
           for p in SCENES}
    assert exo == {"kitchen": 3, "kitchen_shared": 3}
    assert r["served"]["kitchen"] == 3 and r["served"]["kitchen_shared"] >= 2
    # one clock, in order
    assert [s["t"] for s in sigs] == sorted(s["t"] for s in sigs)
    assert run_pair(1)["signals"] == sigs            # deterministic


def test_separate_processes_share_the_env_through_a_file(tmp_path,
                                                         monkeypatch):
    """The harness's wiring: two runtimes, each its own host, the env in
    METIS_PAIR_DIR — the same surprise as in-process."""
    monkeypatch.setenv("METIS_PAIR_DIR", str(tmp_path))
    sh = {p: Repl(SCENES[p], plugin="metis.domains.kitchen.pair:" +
                  ("make_pair_kitchen" if p == "kitchen" else
                   "make_pair_shared"), host=True, stdout=io.StringIO())
          for p in SCENES}
    w = {p: Watch(sh[p], beats=400) for p in SCENES}
    sigs = []
    while not all(x.done for x in w.values()):
        for p in SCENES:
            if not w[p].done:
                for s in w[p].step():
                    sigs.append(s)
                    if s["signal"] == "surprise":
                        dispatch(sh[p], "intent", f"explain {s['record']} "
                                 "kitchen_shared partner_load")
    assert any(s.get("name") == "partner_load" and s["path"] ==
               "kitchen_shared" for s in sigs)
    sur = [s for s in sigs if s["signal"] == "surprise"]
    assert [s["path"] for s in sur] == ["kitchen"]
    assert (tmp_path / "kitchen-pair.json").exists()
