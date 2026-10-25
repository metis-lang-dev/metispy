"""metis.catalog.iface — the generic interface packages (specialist-tree s-404d001f; metis-lang spec/15): declaration-
only modules every composition shares. protocol_domain.llp: the director interface (reports in, grants out).
A source includes one by its package path, `include "protocol_domain.llp".`; path(name)
gives the file (an installer that roots a world copies it from here: copy(name, dst))."""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def path(name="protocol_domain.llp"):
    return HERE / name


def copy(dst_dir, name="protocol_domain"):
    """NAME.llp and its natures NAME.yaml into DST_DIR (an installed root: relative includes, metisc); → the .llp."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for ext in (".llp", ".yaml"):
        shutil.copy(HERE / (name + ext), dst / (name + ext))
    return dst / (name + ".llp")
