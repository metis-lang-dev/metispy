"""The narrate-out boundary (SPEC §1): a certified trace becomes
prose. The LLM sees ONLY certified material — the trace in its Foata
normal form (beats), the final state, and the rules' own %% docs —
and the prompt binds it to the concurrency-narration doctrine:

  - within a beat: simultaneity language only (same-beat events are
    pairwise independent; their order carries no information);
  - across beats: sequence/causation language, and causal claims only
    where the rule semantics licenses them;
  - threads (causal chains) may be told one after another with an
    explicit rewind; repeated similar events are aggregated.

Assembly runs on metispy's own renderers (the beat view IS
nets.foata_levels, the layering run_key hashes) — narration never
re-derives semantics, it only re-tells it."""
from typing import Dict, List, Tuple

from . import llm

PROMPT = """\
You narrate one certified run of a linear-logic story engine. Use
ONLY the material below — never invent events, sentiments, places or
motives that are not in it.

The trace is given in beats (its Foata normal form). Events inside
one beat are causally INDEPENDENT: simultaneous, and their listed
order carries no information — never imply sequence inside a beat
(use "meanwhile", "at the same time", or aggregate into one plural
sentence). Order ACROSS beats is causal: "then"/"because" language is
licensed there, but only claim a causal link the rule semantics
supports (the rules and their doc comments are included). You may
follow one causal thread over several beats, then rewind for a
parallel thread ("while all this unfolded, ..."). Aggregate repeated
similar events instead of enumerating them. The final state may
inform the closing summary. Event bindings name the roles: C acts, D
is acted on, Witness looks on, L/M are places.

Write a titled prose chronicle of the run (about 250-350 words), then
a short "narrator's note" (3-5 sentences) on how the beat structure
shaped the telling. No markdown lists.

== the run, in beats ==
{beats}

== the final state ==
{state}

== the rules (with their approved doc comments) ==
{rules}
"""


def material(rows, final_stage: str,
             kappa: List[Tuple[str, int]], rules_src: str,
             seed: int = 1) -> Dict[str, str]:
    """Certified material: beats + state via metispy's renderers,
    rules verbatim from the approved source."""
    from metis.lang.directives import _pretty_state, _pretty_trace
    return {"beats": "\n".join(_pretty_trace(seed, rows)),
            "state": "\n".join(_pretty_state(seed, final_stage,
                                             sorted(kappa))),
            "rules": rules_src.strip()}


def narrate(spec, rows, final_stage, kappa, rules_src,
            seed: int = 1, timeout: int = 600) -> str:
    m = material(rows, final_stage, kappa, rules_src, seed)
    return llm.complete(spec, PROMPT.format(**m), timeout=timeout)
