"""The REPL API as data (repl-tools s-0c6879a1 step 1; metis-agent
docs/DESIGN.md "the REPL API"): the tool calls an orchestrator makes on a
domain agent directly — synchronous, answered now and finally — published
once here, where the REPL lives, for every renderer to read (py imports
it; the emitter writes it into the bundle as tools.json for the Go
runtime). The orchestrator's model sees them as Gemma native tool
declarations; nothing renders them from a copy.

Each verb:

    name         the tool's name
    description  one line, as the model reads it
    parameters   {"type": "OBJECT", "properties": {name: {type,
                 description[, enum_from]}}, "required": [...]} — or None;
                 enum_from names the set a renderer fills the enum from:
                 "loaded" (the domains a runtime serves) or "catalog" (the
                 domains activate can attach)
    final        the words its answer may carry (DESIGN: a REPL reply is
                 final — applied · admitted · refused · failed; never
                 "accepted", the intent interface's word)
    served_by    who dispatches it: "repl" (a read-only REPL command,
                 metis.lang.repl do_<name>), "watch" (a surprise verb,
                 metis.loop.watch Watch.VERBS), "agent" (the domain agent's
                 consult), "tree" (the runtime tree: activate/deactivate),
                 "orchestrator" (its own: keep)

    python -m metis.lang.repl --tools-json     # the data, as JSON
"""
import copy
import json

_DOMAIN = {"type": "STRING", "description": "a loaded domain", "enum_from": "loaded"}
_CARD = {"type": "STRING", "description": "the catalog domain (card)", "enum_from": "catalog"}
_RECORD = {"type": "STRING", "description": "the id of the reply this closes: the refused call's id, "
                                            "or the surprise's record (#N)"}


def _params(required, **props):
    return {"type": "OBJECT", "properties": props, "required": list(required)}


def _read(name, description, **more):
    req = ["domain"] + [k for k in more]
    return {"name": name, "description": description,
            "parameters": _params(req, domain=_DOMAIN, **more),
            "final": ["applied", "failed"], "served_by": "repl"}


REPL_API = [
    {"name": "activate", "description": "attach a catalog domain to the runtime tree",
     "parameters": _params(["domain"], domain=_CARD,
                           bindings={"type": "STRING", "description": "the card's bindings, when any"}),
     "final": ["applied", "refused", "failed"], "served_by": "tree"},
    {"name": "deactivate", "description": "detach a catalog domain from the runtime tree",
     "parameters": _params(["domain"], domain=_CARD),
     "final": ["applied", "refused", "failed"], "served_by": "tree"},
    {"name": "keep", "description": "change nothing: the context change needs no domain attached or detached",
     "parameters": None, "final": ["applied"], "served_by": "orchestrator"},
    _read("state", "the domain's current state, as its agent reads it"),
    _read("ctx", "the domain's context band: catalog, stage, beat, what is enabled"),
    _read("events", "the domain's recent events"),
    _read("why", "why an atom holds or not, in the domain",
          atom={"type": "STRING", "description": "the atom"}),
    _read("findings", "what the domain's agent has found so far"),
    {"name": "consult", "description": "ask the domain's agent a question; its reply has kind consult and an answer_kind",
     "parameters": _params(["domain", "question"], domain=_DOMAIN,
                           question={"type": "STRING", "description": "one question, usually why <the reason>?"},
                           max_calls={"type": "INTEGER", "description": "the agent's call budget, when you set one"}),
     "final": ["applied", "failed"], "served_by": "agent"},
    {"name": "reset", "description": "the reading was noise: set it aside and resume on the model",
     "parameters": _params(["domain", "record"], domain=_DOMAIN, record=_RECORD),
     "final": ["admitted", "refused"], "served_by": "watch"},
    {"name": "explain", "description": "a sibling card's move explains it",
     "parameters": _params(["domain", "record", "card", "move"], domain=_DOMAIN, record=_RECORD,
                           card={"type": "STRING", "description": "the sibling card"},
                           move={"type": "STRING", "description": "the card's move"}),
     "final": ["admitted", "admitted+notice", "refused"], "served_by": "watch"},
    {"name": "escalate", "description": "nothing explains it: describe it for the people who revise the model",
     "parameters": _params(["domain", "record", "text"], domain=_DOMAIN, record=_RECORD,
                           text={"type": "STRING", "description": "one plain sentence: what was seen"}),
     "final": ["admitted", "refused"], "served_by": "watch"},
]


def repl_api():
    """The REPL API's verbs as data (a copy: a renderer may fill enums)."""
    return copy.deepcopy(REPL_API)


def tools_json():
    """The data as the bundle's tools.json and --tools-json print it."""
    return json.dumps({"format": "metis-repl-api/1", "tools": repl_api()}, indent=1) + "\n"
