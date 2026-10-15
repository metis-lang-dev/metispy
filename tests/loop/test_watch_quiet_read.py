"""A quiet domain is read on every beat (e2e-workflow s-143c3088 A5, the
plant): the env can move on its own while no rule fires — a visitor waters
the dried soil — and the watch must see that reading as the surprise, not
only after the next fired row. A host without a sensor face is unchanged
(the reading is only a checkpoint)."""
import io
import sys

from metis.lang.repl import Repl
from metis.loop.watch import Watch

SOIL = """catalog soil 1.
layers (dyn).
stages (main).
namespace soil produce (dyn).
pred moist : soil.
pred dry : soil.
pred sun : soil.
stage main {
  %% The sun dries the soil out.
  dry_out [dyn] : moist * sun -o dry.
}
#init moist 1
"""

HOST = '''
class Soil:
    def __init__(self, case, sensor):
        self.beat, self.moist, self.sensor = 0, True, sensor
    def bind(self, reg):
        pass
    def tick(self, counts):
        if self.beat >= 12:
            return None
        self.beat += 1
        if self.beat == 6:
            self.moist = True        # a visitor waters it: no rule, no event
        return ["+sun"] if self.beat == 2 else []
    def on_event(self, name, args):
        if name.split("[")[0] == "dry_out":
            self.moist = False
    def snapshot(self):
        return (self.beat, self.moist)
    def restore(self, s):
        self.beat, self.moist = s


class Sensing(Soil):
    def observe(self, counts):
        return {"moist": int(self.moist), "dry": int(not self.moist)}


def make(case):
    return Sensing(case, True)


def make_blind(case):
    return Soil(case, False)
'''


def _run(tmp_path, monkeypatch, factory):
    (tmp_path / "soil.llp").write_text(SOIL)
    (tmp_path / "soil_host.py").write_text(HOST)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("soil_host", None)
    sh = Repl(str(tmp_path / "soil.llp"), seed=1, plugin="soil_host:" + factory, host=True, stdout=io.StringIO())
    w = Watch(sh, beats=12)
    sigs = []
    while not w.done:
        sigs += w.step()
    return sh, sigs


def test_the_env_moving_on_a_quiet_beat_is_the_surprise(tmp_path, monkeypatch):
    sh, sigs = _run(tmp_path, monkeypatch, "make")
    sur = [s for s in sigs if s["signal"] == "surprise"]
    assert len(sur) == 1 and sur[0]["diff"] == {"moist": 1, "dry": -1}
    assert sur[0]["beat"] == 6 and sh.fired == ["do +sun", "dry_out"]   # nothing fired at the visit: read while quiet


def test_a_host_without_a_sensor_face_is_never_surprised(tmp_path, monkeypatch):
    _, sigs = _run(tmp_path, monkeypatch, "make_blind")
    assert not [s for s in sigs if s["signal"] == "surprise"]
