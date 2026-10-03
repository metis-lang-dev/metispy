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

# The two halves of the prompt, separately addressable (`narrate
# system` / `narrate user` in the shell and on the wire): the
# SYSTEM doctrine lives in loop.prompts (multishot by default,
# session- and file-overridable per seat — narrators differ in
# capability); USER_TEMPLATE carries the certified material of one
# run.
from .prompts import NARRATE as SYSTEM

USER_TEMPLATE = """\
== the run, in beats ==
{beats}

== the final state ==
{state}

== the rules (with their approved doc comments) ==
{rules}
"""

PROMPT = SYSTEM + "\n\n" + USER_TEMPLATE


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


def user_prompt(rows, final_stage, kappa, rules_src,
                seed: int = 1) -> str:
    """The material half alone — what `narrate user` shows."""
    return USER_TEMPLATE.format(
        **material(rows, final_stage, kappa, rules_src, seed))


def narrate(spec, rows, final_stage, kappa, rules_src,
            seed: int = 1, timeout: int = 600,
            system: str = None) -> str:
    from .prompts import resolve
    head, _ = resolve("narrate", system)
    body = user_prompt(rows, final_stage, kappa, rules_src, seed)
    return llm.complete(spec, head + "\n\n" + body, timeout=timeout)
