"""metis.transport (metis-lang spec/17 §5.3–5.4, linker s-5908724a step 5b): the M message, the H handshake, the
two-socket carrier: a tick's atoms cross whole, a handshake on another image refuses, a missing message past the
budget is Stuck."""
import socket
import threading

import pytest

from metis import transport as T


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_a_message_is_the_header_signed_wire_atoms_and_a_dot():
    m = T.message("17fe7f088b9a", "a5c04e4d019a", "b", 3, {"tick__grant(cook_a,pot)": 1, "tick__want(cook_a,pot)": -1,
                                                          "idle": 0})
    assert m == ("M image=17fe7f088b9a plan=a5c04e4d019a from=b tick=3\n"
                 "+ tick__grant(cook_a,pot) 1\n- tick__want(cook_a,pot) 1\n.\n")
    back = T.parse_message(m)
    assert back == {"image": "17fe7f088b9a", "plan": "a5c04e4d019a", "from": "b", "tick": 3,
                    "diff": {"tick__grant(cook_a,pot)": 1, "tick__want(cook_a,pot)": -1}}
    for bad in ("M image=x plan=y from=b tick=1\n+ p (x) 1\n.\n", "M image=x plan=y from=b\n.\n",
                "M image=x plan=y from=b tick=1\n+ p(x) 1\n", "M image=x plan=y from=b tick=1\n* p(x) 1\n.\n"):
        with pytest.raises(T.TransportError):
            T.parse_message(bad)


def pair(budget=1, a_hash=("img", "pln", "lck"), b_hash=("img", "pln", "lck")):
    pa, pb = free_port(), free_port()
    a = T.SocketCarrier("a", ("127.0.0.1", pa), ("127.0.0.1", pb), budget=budget, connect_timeout=5)
    b = T.SocketCarrier("b", ("127.0.0.1", pb), ("127.0.0.1", pa), budget=budget, connect_timeout=5)
    out = {}

    def hi(c, h, k):
        try:
            out[k] = c.hello(*h)
        except Exception as e:  # noqa: BLE001
            out[k] = e
    ts = [threading.Thread(target=hi, args=(a, a_hash, "a")), threading.Thread(target=hi, args=(b, b_hash, "b"))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    return a, b, out


def test_the_carrier_crosses_a_tick_whole_and_traces_tick_and_latch_beat():
    a, b, out = pair()
    try:
        assert out["a"]["process"] == "b" and out["b"]["process"] == "a"
        a.send(1, {"tick__want(cook_a,pot)": 1, "tick__active(cook_a)": 1})
        assert b.receive(1, beat=2, wait=2) == {"tick__want(cook_a,pot)": 1, "tick__active(cook_a)": 1}
        b.send(1, {"tick__grant(cook_a,pot)": 1})
        assert a.receive(1, beat=2, wait=2) == {"tick__grant(cook_a,pot)": 1}
        assert b.trace == [{"dir": "recv", "from": "a", "tick": 1, "beat": 2, "atoms": 2},
                           {"dir": "send", "from": "b", "tick": 1, "atoms": 1}]
    finally:
        a.close()
        b.close()


def test_a_handshake_on_another_image_refuses_naming_it():
    a, b, out = pair(b_hash=("other", "pln", "lck"))
    try:
        assert isinstance(out["a"], T.HandshakeRefused) and "image img here, other at b" in str(out["a"])
        assert isinstance(out["b"], T.HandshakeRefused)
    finally:
        a.close()
        b.close()


def test_a_message_missing_past_the_budget_is_stuck():
    a, b, out = pair(budget=1)
    try:
        with pytest.raises(T.Stuck, match="stuck: no message from a for tick 1 after 2 beat"):
            b.receive(1, beat=2, wait=0.05)
    finally:
        a.close()
        b.close()
