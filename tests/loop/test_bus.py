"""The loop bus: two real `metis --json` interpreters cooperating
through the dict-as-blackboard, plus the staleness protections
(channel init, supersession, duplicate/stale drops)."""
import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from metis.loop.bus import Broker, BusClient, Tx, ingress_path
from metis.loop import node

FIX = Path(__file__).resolve().parent / "fixtures"
METIS = os.environ.get(
    "METIS", str(Path(sys.executable).parent / "metis"))
if not Path(METIS).exists():
    METIS = shutil.which("metis") or METIS

pytestmark = [
    pytest.mark.skipif(not Path(METIS).exists(),
                       reason="no metis executable"),
    # CI.md D2: the bus FIFO intermittently raises ENXIO (OSError) when a
    # writer opens before the reader is up — rerun ONLY that, twice; any
    # other failure fails. Root cause stays on the metis-exact-audit
    # backlog. (Needs pytest-rerunfailures; without it the marker is inert.)
    pytest.mark.flaky(reruns=2, only_rerun=["OSError"]),
]


@pytest.fixture()
def bus(tmp_path):
    d = str(tmp_path / "bus")
    broker = Broker(d)
    t = threading.Thread(target=broker.serve, daemon=True)
    t.start()
    yield d, broker


def start_node(name, world, d):
    t = threading.Thread(
        target=node.serve, args=(name, str(world)),
        kwargs={"metis": METIS, "dir": d, "timeout": 60},
        daemon=True)
    t.start()
    return t


def wait_node(driver, name, tries=50):
    for _ in range(tries):
        r = driver.op(name, "help", "ops")     # a ping that fits the bus
        if r.get("ok"):
            return r
        time.sleep(0.1)
    raise AssertionError(f"node {name} never answered: {r}")


def test_two_worlds_cooperate_through_the_blackboard(bus):
    d, _ = bus
    start_node("scout", FIX / "scout.llp", d)
    start_node("wagon", FIX / "wagon.llp", d)
    driver = BusClient("driver", d)
    try:
        wait_node(driver, "scout")
        wait_node(driver, "wagon")

        # world A: the scout's exact verdict
        m = driver.op("scout", "marginals")
        assert m["ok"] and m["marginals"]["clear"] == 0.75

        # driver policy: the verdict becomes a blackboard word
        word = ("clear" if m["marginals"]["clear"] >= 0.5
                else "blocked")
        assert driver.bb_set("pass.word", word)["ok"]

        # world B: reads the word as its scene — no shared state
        got = driver.bb_get("pass.word")
        assert got["ok"] and got["src"] == "driver"
        r = driver.op("wagon", "init",
                      f"wagon=1,word_{got['value']}=1")
        assert r["ok"]
        m2 = driver.op("wagon", "marginals")
        assert m2["ok"] and m2["marginals"]["arrived"] == 1.0

        # and B's outcome goes back up for anyone watching
        assert driver.bb_set("wagon.state", "arrived")["ok"]
        assert driver.bb_dump()["bb"] == {
            "pass.word": "clear", "wagon.state": "arrived"}
    finally:
        driver.close()


def test_a_result_over_the_envelope_cap_is_an_error_and_the_node_lives(bus):
    """agentic-loop s-3ddb3e87 step 2 (nina's review of 44fa6be): bare
    help grew past the bus's PIPE_BUF envelope, node.serve's post-back
    raised in its thread and the node died silently. The asker now gets
    an error result naming the cap, and the node answers the next op;
    `help ops` / `help VERB` are the forms that fit."""
    d, _ = bus
    start_node("scout", FIX / "scout.llp", d)
    driver = BusClient("driver", d)
    try:
        assert wait_node(driver, "scout")["ops"]
        big = driver.op("scout", "help")
        assert big["ok"] is False and "PIPE_BUF" in big["error"], big
        one = driver.op("scout", "help", "marginals")
        assert one["ok"] and list(one["descriptions"]) == ["marginals"]
        m = driver.op("scout", "marginals")
        assert m["ok"] and m["marginals"]["clear"] == 0.75
    finally:
        driver.close()


def test_bb_provenance_and_unknown_node(bus):
    d, _ = bus
    a = BusClient("alice", d)
    try:
        a.bb_set("k", 1)
        got = a.bb_get("k")
        assert got == {"ok": True, "key": "k", "value": 1,
                       "src": "alice"}
        miss = a.bb_get("absent")
        assert miss["ok"] is False
        r = a.op("nobody", "help")
        assert r["ok"] is False and "no such node" in r["error"]
    finally:
        a.close()


def raw(d, env):
    t = Tx(ingress_path(d))
    t.send(env)
    t.close()


def test_stale_epoch_and_duplicates_are_dropped(bus):
    d, broker = bus
    a = BusClient("alice", d)
    try:
        a.bb_set("k", "fresh")
        # a duplicated line (same seq, replayed verbatim) is dropped
        raw(d, {"src": "alice", "dst": "bb", "epoch": a.epoch,
                "seq": a.seq, "op": "set",
                "arg": {"key": "k", "value": "replayed"}})
        # a stale repl (old epoch) keeps writing: dropped
        raw(d, {"src": "alice", "dst": "bb",
                "epoch": uuid.uuid4().hex[:12], "seq": 99,
                "op": "set", "arg": {"key": "k", "value": "stale"}})
        # no hello at all: dropped
        raw(d, {"src": "ghost", "dst": "bb", "epoch": "e", "seq": 1,
                "op": "set", "arg": {"key": "k", "value": "ghost"}})
        assert a.bb_get("k")["value"] == "fresh"
    finally:
        a.close()


def test_new_hello_supersedes_the_old_channel(bus):
    d, broker = bus
    old = BusClient("alice", d)
    old.bb_set("who", "old")
    new = BusClient("alice", d)   # same src, fresh epoch
    # the old channel is told, then cut
    note = old.recv(timeout=10)
    assert note["op"] == "superseded"
    # the old epoch can no longer write
    raw(d, {"src": "alice", "dst": "bb", "epoch": old.epoch,
            "seq": old.seq + 1, "op": "set",
            "arg": {"key": "who", "value": "zombie"}})
    assert new.bb_get("who")["value"] == "old"
    new.bb_set("who", "new")
    assert new.bb_get("who")["value"] == "new"
    new.close()
    old.rx.close()
    old.tx.close()


def test_envelope_size_guard(bus):
    d, _ = bus
    a = BusClient("alice", d)
    try:
        with pytest.raises(ValueError):
            a.bb_set("blob", "x" * 8192)
    finally:
        a.close()
