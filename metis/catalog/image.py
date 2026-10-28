"""The image (metis-lang spec/17, linker s-5908724a step 3): a composition linked by metisc, what a runtime loads.

metispy never resolves a composition itself (spec/17 §6: one resolver, the OCaml linker). This module only calls it
and reads what it wrote:

    link(path, out=None, check=False)   metisc --link COMPOSITION [-o OUT] [--check] → its listing; the image dir
    build(dir, out=None)                 metisc build DIR -o OUT (stale members recompiled, the rest reused) → the image dir
    Image(path)                          image.json (+ image.lock) of a built image: the members, their clocks, the
                                         direct member, the protocol's natures, a member's manifest table
    use(image)                           a generated member (the derived director) as a package metis.catalog.get finds
    key_mismatches(image)                the members whose source, compiled here, is not the program the image linked

A path given to image() is an image dir (image.json in it), a composition dir (composition.yaml in it: built into
DIR/build), or a composition file (linked into the directory beside it, build/)."""
import json
import os
import re
import subprocess
from pathlib import Path

import yaml

_IMAGE = re.compile(r"^image (\S+) v-([0-9a-f]+)$", re.M)

# the generated members of the images in use: name → Domain (metis.catalog.get looks here first)
USED = {}
PASS_HOST = "metis.host:pass_host"


SIBLING_METISC = Path(__file__).resolve().parents[3] / "metis-lang" / "src" / "metisc"   # a checkout beside metispy's


def metisc():
    """The compiler the link and the build run (the loop's rule, linker step 3): $METISC, else the metis-lang
    checkout beside metispy's (src/metisc), else metisc on PATH; None when there is none."""
    import shutil
    if os.environ.get("METISC"):
        return os.environ["METISC"]
    if SIBLING_METISC.is_file() and os.access(SIBLING_METISC, os.X_OK):
        return str(SIBLING_METISC)
    return shutil.which("metisc")


def _metisc():
    from metis.catalog import NO_METISC, metisc_env
    m = metisc()
    if not m:
        raise ValueError(NO_METISC)
    return m, metisc_env()


def _run(cmd, cwd):
    m, env = _metisc()
    p = subprocess.run([m] + cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=1800)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out


def _image_dir(out, listing):
    m = _IMAGE.search(listing)
    if not m:
        raise ValueError("metisc printed no `image NAME v-HASH` line:\n" + listing.strip()[-1200:])
    d = Path(out) / ("%s.image.v-%s" % m.groups())
    if not (d / "image.json").is_file():
        raise ValueError("no image at %s (metisc: %s)" % (d, m.group(0)))
    return d


def link(path, out=None, check=False, plan=None):
    """metisc --link COMPOSITION: (the listing, the image dir or None with --check). A composition dir means its
    composition.yaml; the image goes to OUT, by default build/ beside the composition; PLAN (a plans/<p>.yaml) is
    linked with it into the image's plans/<p>.json (spec/17 §5). Raises ValueError with the listing when the link
    reports errors (a non-zero exit)."""
    comp = Path(path).resolve()
    if comp.is_dir():
        comp = comp / "composition.yaml"
    out = Path(out).resolve() if out else comp.parent / "build"
    cmd = ["--link", comp.name] + (["--check"] if check else ["-o", str(out)])
    if plan:
        cmd += ["--plan", str(Path(plan).resolve())]
    rc, listing = _run(cmd, comp.parent)
    if rc:
        raise ValueError("metisc --link %s: exit %d\n%s" % (comp, rc, listing.strip()[-2000:]))
    return listing, None if check else _image_dir(out, listing)


def build(path, out=None):
    """metisc build DIR -o OUT (spec/17 §7): the image dir. OUT is DIR/build by default (git-ignored where a
    project builds; the record OUT/<name>.build keeps the stale-member keys)."""
    d = Path(path).resolve()
    out = Path(out).resolve() if out else d / "build"
    rc, listing = _run(["build", str(d), "-o", str(out)], d)
    if rc:
        raise ValueError("metisc build %s: exit %d\n%s" % (d, rc, listing.strip()[-2000:]))
    return _image_dir(out, listing)


def image(path):
    """The image at PATH (an image dir), or built from a composition dir / file there (§ the module's doc)."""
    p = Path(path).resolve()
    if (p / "image.json").is_file():
        return Image(p)
    if p.is_dir() and (p / "composition.yaml").is_file():
        return Image(build(p))
    if p.is_file():
        return Image(link(p)[1])
    raise FileNotFoundError("%s: no image.json, no composition.yaml" % p)


class Image:
    """A linked image (image.json, metis-image/1): read, never re-derived."""

    def __init__(self, path):
        self.dir = Path(path).resolve()
        self.json = json.loads((self.dir / "image.json").read_text())
        lock = self.dir / "image.lock"
        self.lock = json.loads(lock.read_text()) if lock.is_file() else {}
        if self.json.get("format") != "metis-image/1":
            raise ValueError("%s: format %r, not metis-image/1" % (self.dir, self.json.get("format")))
        self.name = self.json["name"]
        self.hash = self.dir.name.rpartition(".v-")[2]
        self.members = list(self.json["members"])
        self.by_name = {m["name"]: m for m in self.members}

    @property
    def names(self):
        """The members in the image's order (the direct member first): the order a runtime beats them in."""
        return [m["name"] for m in self.members]

    @property
    def direct(self):
        return next((m["name"] for m in self.members if m.get("role") == "direct"), None)

    @property
    def clocks(self):
        """[{domain, n}]: each member's beats a tick (its manifest's clock.beats, as the link recorded it)."""
        return [{"domain": m["name"], "n": int((m.get("clock") or {}).get("beats") or 1)} for m in self.members]

    @property
    def natures(self):
        """The protocol's natures {pred: report | signal | injected | …} (image.json protocol.tokens)."""
        return dict((self.json.get("protocol") or {}).get("tokens") or {})

    def member(self, name):
        """The member's table as the link recorded it: shared, status, gating, clock, levels, beliefs, evaluate, …
        (the manifest's fields after the link checked them)."""
        return self.by_name[name]

    def generated(self, name):
        return bool(self.by_name[name].get("generated")) or (
            self.json.get("director") or {}).get("name") == name and bool((self.json.get("director") or {}).get("generated"))

    def bundle(self, name):
        return self.dir / self.by_name[name]["bundle"]

    def member_dir(self, name):
        """The member's package directory as the link resolved it (image.lock `dir`, metis-lang 53c5eec), or None
        from a lock before it (or a generated member's, which lives in its bundle)."""
        d = ((self.lock.get("members") or {}).get(name) or {}).get("dir")
        return Path(d["path"]).resolve() if isinstance(d, dict) and d.get("path") else None


def use(img):
    """The image's generated members as packages metis.catalog.get finds (the derived director's source, which
    the link wrote into its bundle): spec/17 §6, a source runtime builds the generated director from the image."""
    from metis.catalog.manifests import Domain
    for m in img.members:
        if img.generated(m["name"]):
            d = img.bundle(m["name"])
            man = yaml.safe_load((d / "domain.yaml").read_text()) or {}
            # a director has no world of its own, but the agent takes the relayed reports into its session through a
            # host: the generic pass host when the link wrote none (the link should record it: linker step 3's finding)
            man.setdefault("host", PASS_HOST)
            USED[m["name"]] = Domain(m["name"], d, man)
    return img


def unuse(img=None):
    for n in list(USED) if img is None else [m["name"] for m in img.members]:
        USED.pop(n, None)


def program_key(dom):
    """The program key of a package's served program (Domain.serves(): the program, then what extends it;
    grounded at its #init): what a bundle's sidecar keys and the image's wire_key holds (metis.kernel.artifact)."""
    from metis.kernel import artifact
    from metis.kernel.nets import program_key as key
    files = dom.serves()
    doc, case, _ = artifact.served_doc(Path(files[0]), files[1:])
    return key(doc.catalog.ground(dict(case["init"])))


def key_mismatches(img, get=None):
    """[(member, the image's wire_key, the key of the source compiled here)] for every member whose package,
    resolved as this process resolves it (metis.catalog.get), is not in the directory the image's lock names
    (the third element then `dir <path>`), or whose source compiles to another program than the image linked:
    load-key-mismatch (spec/17 §6). Empty when they all agree."""
    if get is None:
        from metis.catalog import get
    out = []
    for m in img.members:
        dom = get(m["name"])
        want = None if img.generated(m["name"]) else img.member_dir(m["name"])
        if want is not None and Path(dom.dir).resolve() != want:
            out.append((m["name"], str(want), "dir %s" % Path(dom.dir).resolve()))
            continue
        k = program_key(dom)
        if k != m.get("wire_key"):
            out.append((m["name"], m.get("wire_key"), k))
    return out
