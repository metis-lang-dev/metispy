"""The gossip wiring — the bus acceptance test proposed in
CEPTRE-PORTS.md: three person-worlds (gossip_node.llp, no social
graph inside) cooperating over the blackboard must reproduce the
single-world closure (gossip3.llp, the graph baked in). The
`follows` relation exists ONLY as driver wiring; `posted` atoms
become bb keys; each world's `knows` stays private."""
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from metis.loop.bus import Broker, BusClient
from metis.loop import node

FIX = Path(__file__).resolve().parent / "fixtures"
METIS = os.environ.get(
    "METIS", str(Path(sys.executable).parent / "metis"))
if not Path(METIS).exists():
    METIS = shutil.which("metis") or METIS

pytestmark = pytest.mark.skipif(
    not Path(METIS).exists(), reason="no metis executable")

FOLLOWS = {"alice": ["bob"], "bob": ["carol"], "carol": ["alice"]}
KNOWS0 = {"alice": {"m1"}, "bob": {"m2"}, "carol": {"m3"}}
MSGS = {"m1", "m2", "m3"}


def start_node(name, world, d):
    threading.Thread(
        target=node.serve, args=(name, str(world)),
        kwargs={"metis": METIS, "dir": d, "timeout": 60},
        daemon=True).start()


def wait_node(driver, name, tries=50):
    for _ in range(tries):
        r = driver.op(name, "help", "ops")     # a ping that fits the bus
        if r.get("ok"):
            return
        time.sleep(0.1)
    raise AssertionError(f"node {name} never answered")


def final_atoms(result, pred):
    out = set()
    for atom, n in result["final"].items():
        m = re.fullmatch(rf"{pred}\(([^)]*)\)", atom)
        if m and n > 0:
            out.add(m.group(1))
    return out


def test_three_person_worlds_reproduce_the_single_world(tmp_path):
    d = str(tmp_path / "bus")
    broker = Broker(d)
    threading.Thread(target=broker.serve, daemon=True).start()
    for p in FOLLOWS:
        start_node(p, FIX / "gossip_node.llp", d)
    start_node("all", FIX / "gossip3.llp", d)
    driver = BusClient("driver", d)
    try:
        for p in list(FOLLOWS) + ["all"]:
            wait_node(driver, p)

        # --- the wired run: graph as driver policy, posts as bb keys
        knows = {p: set(s) for p, s in KNOWS0.items()}
        posted = {p: set() for p in FOLLOWS}
        delivered = {p: set(s) for p, s in KNOWS0.items()}
        for _ in range(5):  # rounds; ring diameter + slack
            changed = False
            for p in FOLLOWS:
                fresh = set()
                for q in FOLLOWS[p]:  # p hears what q posted
                    got = driver.bb_get(f"posted.{q}")
                    if got["ok"]:
                        fresh |= set(got["value"]) - delivered[p]
                scene = (
                    [f"knows({m})=1" for m in knows[p]] +
                    [f"not_posted({m})=1"
                     for m in knows[p] - posted[p]] +
                    [f"unheard({m})=1" for m in MSGS - knows[p]] +
                    [f"heard({m})=1" for m in fresh])
                delivered[p] |= fresh
                assert driver.op(p, "init", ",".join(scene))["ok"]
                res = driver.op(p, "map")
                assert res["ok"], res
                new_knows = final_atoms(res, "knows")
                new_posts = final_atoms(res, "posted")
                if new_knows - knows[p] or new_posts - posted[p]:
                    changed = True
                knows[p] |= new_knows
                posted[p] |= new_posts
                driver.bb_set(f"posted.{p}", sorted(posted[p]))
            if not changed:
                break

        # --- the reference: one world, the graph baked in
        ref = driver.op("all", "map")
        assert ref["ok"], ref
        for p in FOLLOWS:
            ref_knows = {m for pm in final_atoms(ref, "knows")
                         for q, m in [pm.split(",")] if q == p}
            ref_posted = {m for pm in final_atoms(ref, "posted")
                          for q, m in [pm.split(",")] if q == p}
            assert knows[p] == ref_knows, (p, knows[p], ref_knows)
            assert posted[p] == ref_posted, (
                p, posted[p], ref_posted)
        # and the ring actually closed: everyone knows everything
        assert all(knows[p] == MSGS for p in FOLLOWS)
    finally:
        driver.close()
