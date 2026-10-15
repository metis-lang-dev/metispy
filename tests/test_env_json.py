"""The neutral JSON env (metis-agent RUNTIME-PARITY row 22, ruling (d)):
a host's snapshot is JSON data in the record, hashed over its canonical
text, so py and Go write and read the same env."""
import hashlib

import pytest

from metis.kernel.record import (EnvError, env_data, env_hash, env_json,
                                 env_record, env_text)


def test_a_tuple_snapshot_is_carried_as_arrays():
    snap = (3, ((5, "+lost"),), ((2, "sit", ("now",)),), True)
    rec = env_record(snap)
    assert rec["format"] == "json"
    assert env_data(rec) == [3, [[5, "+lost"]], [[2, "sit", ["now"]]], True]
    assert env_json(snap) == env_data(rec)
    assert rec["hash"] == env_hash(snap) == env_hash(env_data(rec))


def test_the_hash_is_over_the_canonical_text():
    a, b = {"moist": 1, "beat": 2}, {"beat": 2, "moist": 1}
    assert env_text(a) == env_text(b) == '{"beat":2,"moist":1}'
    assert env_hash(a) == hashlib.sha256(b'{"beat":2,"moist":1}').hexdigest()
    assert env_text({"s": "é"}) == '{"s":"\\u00e9"}'


@pytest.mark.parametrize("snap", [{1, 2}, object(), float("nan"),
                                  {1: "a", "b": 2}])
def test_what_json_cannot_carry_is_an_error(snap):
    with pytest.raises(EnvError):
        env_record(snap)


def test_a_big_snapshot_carries_its_hash_only():
    rec = env_record(list(range(50)), data_cap=40)
    assert "data" not in rec and rec["hash"] == env_hash(list(range(50)))
    with pytest.raises(KeyError):
        env_data(rec)
