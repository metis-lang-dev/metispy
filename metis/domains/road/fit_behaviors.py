"""Fit others' CHOICE weights from an observed-transitions JSONL
(carla_run --behavior-log): the first learning slot of DESIGN.md,
implemented as the simplest sound thing — smoothed categorical
frequencies, split by the danger flag (the feature our policy already
branches on). Output feeds carla_run --behaviors.

    python -m rss.fit_behaviors behavior.jsonl > behaviors.json
    python -m rss.fit_behaviors behavior.jsonl --by-class > behaviors.json

--by-class groups rows by catalog class (detection layer over the
per-row type_id the harvester logs) and emits one weight set per class
with enough data, plus the overall "default". CeptrePlanner expands
class keys to per-actor weights each replan (bridge_core.
expand_class_behaviors), so learning stays per-class while the policy
stays per-name."""

import collections
import json
import sys

ACTS = ("keep", "accel", "brake", "hbrake")
WALKER_ACTS = ("keep", "accel", "brake", "left", "right")
MIN_CLASS_ROWS = 500   # below this, a class inherits "default"


def fit(rows, alpha=1.0, acts=ACTS):
    by_flag = {False: collections.Counter(), True: collections.Counter()}
    for r in rows:
        by_flag[bool(r.get("dangerous"))][r["action"]] += 1
    out = {}
    for flag, counts in by_flag.items():
        tot = sum(counts.values()) + alpha * len(acts)
        out["dangerous" if flag else "default"] = {
            a: round((counts[a] + alpha) / tot, 4) for a in acts}
        out["n_" + ("dangerous" if flag else "default")] = \
            sum(counts.values())
    return out


def fit_by_class(rows, alpha=1.0):
    """Class-keyed weight sets: {"default": overall, "<class>": ...}
    for every catalog class with at least MIN_CLASS_ROWS rows."""
    from .detection import classify
    overall = fit(rows, alpha)
    out = {"default": overall["default"]}
    by_class = collections.defaultdict(list)
    for r in rows:
        cls = classify(r.get("type", ""))
        if cls is not None:
            by_class[cls].append(r)
    stats = {}
    for cls, cr in sorted(by_class.items()):
        stats[cls] = len(cr)
        if len(cr) >= MIN_CLASS_ROWS:
            acts = WALKER_ACTS if cls == "pedestrian" else ACTS
            out[cls] = fit(cr, alpha, acts)["default"]
    return out, stats


def main():
    by_class = "--by-class" in sys.argv[1:]
    path = [a for a in sys.argv[1:] if not a.startswith("--")][0]
    rows = [json.loads(line) for line in open(path)]
    if by_class:
        out, stats = fit_by_class(rows)
        print(json.dumps(out, indent=1))
        print(f"# {len(rows)} rows; per-class: "
              + ", ".join(f"{c}={n}" + ("" if c in out else " (thin, "
                          "inherits default)")
                          for c, n in sorted(stats.items())),
              file=sys.stderr)
        return
    fitted = fit(rows)
    n_d, n_s = fitted.pop("n_dangerous"), fitted.pop("n_default")
    print(json.dumps({"default": fitted["default"]}, indent=1))
    print(f"# fitted from {n_s} safe + {n_d} dangerous transitions; "
          f"dangerous-mode empirical: {fitted['dangerous']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
