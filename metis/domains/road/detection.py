"""Perception output bound to typed classes (functional module).

The CARLA object catalog (`data/carla_objects.json`, dumped from the
live blueprint library) grounds the mapping: a detected actor's type_id
becomes a *class*, and the class selects the object-behavior overlay.
An id no class covers returns None — the new-actor-type discovery
trigger.

Catalog-reconciliation note: this used to live inside the road catalog
Entry registry (`road/catalog/detection.py`); the registry moved to
chariot (wayplan `py/rss/catalog`, dissolving into the `.llp` catalogs
— see chariot_ws docs/catalog-reconciliation.md). This plain module
remains because the SUMO learning loop (`fit_behaviors`) classifies by
it; the registry Entries documenting these bindings live chariot-side."""

import json
import os

_DATA = os.path.join(os.path.dirname(__file__), "data",
                     "carla_objects.json")

# CARLA base_type -> class. `bulky` matters beyond behavior:
# semantics.hidden_cells gives bulky vehicles an adjacent-lane occlusion
# shadow, so the class assignment feeds the ghost machinery too.
BASE_TO_CLASS = {"car": "vehicle", "van": "bulky", "truck": "bulky",
                 "Bus": "bulky", "motorcycle": "twowheeler",
                 "bicycle": "twowheeler"}

# blueprints that predate the base_type attribute (empty string in the
# dump) — classified by hand from the model
UNTYPED = {"vehicle.bmw.grandtourer": "vehicle",
           "vehicle.mini.cooper_s": "vehicle"}


_CACHE = None


def _load():
    global _CACHE
    if _CACHE is None:
        with open(_DATA) as f:
            _CACHE = json.load(f)
    return _CACHE


def classify(type_id: str):
    """type_id -> class, or None (unknown: discovery trigger).
    Works from the dumped catalog when the id is known, and falls back
    to namespace heuristics so SUMO-side vtypes and future blueprints
    degrade gracefully rather than silently misclassify."""
    if type_id in UNTYPED:
        return UNTYPED[type_id]
    d = _load()
    v = d["vehicles"].get(type_id)
    if v is not None:
        return BASE_TO_CLASS.get(v["base"])
    if type_id in d["walkers"] or type_id.startswith("walker.pedestrian"):
        return "pedestrian"
    if type_id.startswith("walker."):
        return None
    if type_id.startswith("vehicle."):
        # unknown vehicle blueprint: conservative default is bulky
        # (worst-case occlusion shadow) until a proper entry exists
        return "bulky"
    return None
