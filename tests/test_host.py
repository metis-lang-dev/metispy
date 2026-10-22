"""metis.host (metis-packages s-65778b98): the PortPlugin contract, the JSON-lines gym child, the
interpreter a world's gym runs under, the generated package's way to its world's host."""
import os
import sys
import textwrap

import pytest

from metis.host import GymChild, PortPlugin, find_interpreter, world_host

ECHO = textwrap.dedent('''
    import json, sys
    state = {}
    for line in sys.stdin:
        m = json.loads(line)
        op = m.pop("op")
        if op == "close":
            break
        if op == "open":
            state.update(m)
            print(json.dumps({"ok": True, "horizon": 5, "opened": m}), flush=True)
        elif op == "step":
            print(json.dumps({"ok": True, "obs": [m.get("action")], "done": False}), flush=True)
        elif op == "crash":
            sys.stderr.write("the gym fell over\\n"); sys.exit(3)
        else:
            print(json.dumps({"ok": False, "error": "no op " + op}), flush=True)
''')


def test_the_gym_child_answers_one_line_per_request(tmp_path):
    script = tmp_path / "echo_gym.py"
    script.write_text(ECHO)
    g = GymChild(script, sys.executable, what="game", game="g1", max_steps=60)
    assert g.info == {"horizon": 5, "opened": {"game": "g1", "max_steps": 60}}
    assert g.call("step", action="go") == {"obs": ["go"], "done": False}
    with pytest.raises(RuntimeError, match=r"^game nope: no op nope$"):
        g.call("nope")
    with pytest.raises(RuntimeError, match=r"the game process ended \(\w+\): the gym fell over"):   # its code, or None while it exits
        g.call("crash")
    g.close()


def test_the_interpreter_is_the_env_else_this_one_with_the_module_else_a_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("METIS_TOY_PY", "/opt/toy/bin/python")
    assert find_interpreter("METIS_TOY_PY") == "/opt/toy/bin/python"
    monkeypatch.delenv("METIS_TOY_PY")
    assert find_interpreter("METIS_TOY_PY", module="json") == sys.executable
    venv = tmp_path / ".venv/bin/python"
    venv.parent.mkdir(parents=True)
    venv.write_text("")
    assert find_interpreter("METIS_TOY_PY", module="no_such_module_here", candidates=[tmp_path / "nope", venv]) == str(venv)
    assert find_interpreter("METIS_TOY_PY", module="no_such_module_here") == sys.executable
    with pytest.raises(RuntimeError, match="set METIS_TOY_PY"):
        find_interpreter("METIS_TOY_PY", module="no_such_module_here", default=False)


def test_a_port_plugin_needs_tick_and_on_event():
    with pytest.raises(TypeError):
        PortPlugin()

    class Host(PortPlugin):
        def tick(self, counts):
            return ["+tick"]

        def on_event(self, name, args):
            self.last_event = name
    h = Host()
    assert h.tick({}) == ["+tick"] and h.observe({}) == {} and h.ports() == {} and h.sync({}) == []
    with pytest.raises(ValueError, match="no world lines"):
        h.world(["+x"])
    assert (h.beat, h.due, h.script, h.last) == (0, (), {}, None)


def test_a_generated_package_finds_its_worlds_host(tmp_path, monkeypatch):
    world = tmp_path / "repo" / "world"
    world.mkdir(parents=True)
    (world / "toy_world_host.py").write_text("class ToyHost:\n    def __init__(self, d, case=None):\n        self.d = d\n")
    pkg = tmp_path / "repo" / "build" / "games" / "g1"
    pkg.mkdir(parents=True)
    m = world_host(str(pkg / "toy_host.py"), "toy_world_host")              # found walking up to repo/world
    assert m.ToyHost(str(pkg)).d == str(pkg)
    sys.modules.pop("toy_world_host", None)
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "toy_world_host.py").write_text("WHERE = 'env'\n")
    monkeypatch.setenv("METIS_TOY_WORLD", str(other))
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(world)])
    assert world_host(str(tmp_path / "x"), "toy_world_host", env="METIS_TOY_WORLD").WHERE == "env"
    sys.modules.pop("toy_world_host", None)
