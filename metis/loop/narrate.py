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
import re
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

Now narrate this run.
"""

PROMPT = SYSTEM + "\n\n" + USER_TEMPLATE


HONESTY = """\

== interventions (NOT chance) ==
{lines}
These beats were decided OUTSIDE the story's chances. [forced]: the
player chose it — tell it as the choice that was made, never as luck,
fate or the dice. [do]: an intervention, a what-if imposed on the
story — tell it as a supposition ("suppose…", "had…", "in this
telling…"). [env]: the world outside the rules moved (an arrival, a
clock, a sensor) — tell it as something that happened TO the story,
from outside, never as a character's doing. A [do +X]/[do -X] start
state was set by hand: say so.
Never present any of these as a character's spontaneous impulse
unless a rule's doc says so."""

_HOW = {"forced": "[forced]", "do": "[do]", "exo": "[env]"}


def material(rows, final_stage: str,
             kappa: List[Tuple[str, int]], rules_src: str,
             seed: int = 1, marks=None, base_do=()) -> Dict[str, str]:
    """Certified material: beats + state via metispy's renderers,
    rules from the approved source (story_rules: noise stripped)."""
    from metis.lang.directives import _pretty_state, _pretty_trace
    from metis.kernel.nets import foata_levels
    docs = rule_docs(rules_src)
    lines = _pretty_trace(seed, rows)
    # _pretty_trace lists beat lines by (level, firing order): the
    # same order recovers each line's event for its effect
    order = sorted(range(len(rows)),
                   key=lambda i: (foata_levels(rows)[i], i)) \
        if rows else []
    evs = [None] + [rows[i][2] for i in order]
    # spec 10: faithfulness extends to agency — an intervention is
    # never presented as sampled
    hows = [None] + [(marks or [None] * len(rows))[i] for i in order]
    beats = "\n".join(gloss(l, docs, ev) + (
        f"  {_HOW[h]}" if h in _HOW else "")
        for l, ev, h in zip(lines, evs, hows))
    forced = [rows[i][2].name for i in range(len(rows))
              if (marks or [None] * len(rows))[i] in _HOW]
    if forced or base_do:
        beats += HONESTY.format(lines="\n".join(
            [f"[do {op}] — the start state, set by hand"
             for op in base_do]
            + [f"{_HOW[(marks or [])[i]]} {rows[i][2].name}"
               for i in range(len(rows))
               if (marks or [None] * len(rows))[i] in _HOW]))
    return {"beats": beats,
            "state": "\n".join(_pretty_state(seed, final_stage,
                                             sorted(kappa))),
            "rules": story_rules(rules_src)}


def story_rules(src: str) -> str:
    """The rules minus file-level noise: single-% comments and
    #-directive lines (init/steps/query) are notes for the author and
    the interpreter, not story — a small narrator retells them as
    plot. %% doc comments, the narrator's vocabulary, stay."""
    keep = [l for l in src.splitlines()
            if not (l.lstrip().startswith("#")
                    or (l.lstrip().startswith("%")
                        and not l.lstrip().startswith("%%")))]
    return "\n".join(keep).strip()


_RULE = re.compile(r"^\s*(?:qui\s+)?([\w-]+)\s*\[")
_BEAT = re.compile(r"^(\s*\d*\s*\|\s*)([\w-]+)\s*(?:\[([^\]]*)\])?\s*$")


def rule_docs(src: str) -> Dict[str, str]:
    """rule name -> its %% doc comment (the block right above it,
    joined into one line)."""
    docs, block = {}, []
    for l in src.splitlines():
        t = l.strip()
        if t.startswith("%%"):
            block.append(t[2:].strip())
            continue
        m = _RULE.match(l)
        if m and block:
            docs[m.group(1)] = " ".join(block)
        if t:
            block = []
    return docs


def effect(ev) -> str:
    """An event's effect from its ground semantics, in words a small
    narrator copies: what is gone, what is new, what was only read
    ($) and stays. Direction (who did what to whom) lives in the
    atoms' argument order, which glosses alone cannot carry."""
    def atoms(m):
        return ", ".join(a if n == 1 else f"{a}×{n}"
                         for a, n in sorted(m.items()))
    gone, new = dict(ev.consume), dict(ev.produce)
    stays = dict(ev.persist)
    for a in set(gone) & set(new):       # consumed and reproduced:
        k = min(gone[a], new[a])         # it stays (the held house)
        stays[a] = stays.get(a, 0) + k
        gone[a] -= k
        new[a] -= k
    parts = [f"{label}: {atoms({a: n for a, n in m.items() if n})}"
             for label, m in (("gone", gone), ("new", new),
                              ("stays", stays))
             if any(m.values())]
    return "; ".join(parts)


def gloss(line: str, docs: Dict[str, str], ev=None) -> str:
    """One beat line plus what the event means: the first sentence of
    its rule's %% doc, parenthetical asides dropped, role letters
    replaced by the event's bindings (C=ada turns "C warns D" into
    "ada warns D"), then the event's effect (`effect`). A presentation of the approved doc, not new
    semantics — a small narrator copies glosses far more faithfully
    than it performs the substitution itself."""
    m = _BEAT.match(line)
    if not m:
        return line
    eff = f"  ⇒ {effect(ev)}" if ev is not None and effect(ev) else ""
    doc = docs.get(m.group(2)) \
        or docs.get(re.sub(r"-\d+$", "", m.group(2)))  # ⊕ branches
    if not doc:
        return line.rstrip() + eff
    first = re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0]
    first = re.sub(r"\s*\([^)]*\)", "", first).strip()
    binds = dict(kv.split("=", 1) for kv in (m.group(3) or "").split(",")
                 if "=" in kv)
    if binds:
        keys = "|".join(map(re.escape, sorted(binds, key=len,
                                                reverse=True)))
        first = re.sub(rf"\b({keys})\b",
                       lambda k: binds[k.group(1)], first)
    return f"{line.rstrip()}  — {first}{eff}"


def user_prompt(rows, final_stage, kappa, rules_src,
                seed: int = 1, marks=None, base_do=()) -> str:
    """The material half alone — what `narrate user` shows."""
    return USER_TEMPLATE.format(
        **material(rows, final_stage, kappa, rules_src, seed,
                   marks, base_do))


def narrate(spec, rows, final_stage, kappa, rules_src,
            seed: int = 1, timeout: int = 600,
            system: str = None, marks=None, base_do=()) -> str:
    from .prompts import resolve
    head, _ = resolve("narrate", system)
    body = user_prompt(rows, final_stage, kappa, rules_src, seed,
                       marks, base_do)
    return llm.complete(spec, body, timeout=timeout,
                        options=options(), system=head)


def options() -> Dict[str, object]:
    """The narrate seat's sampling knobs: the dot file's [narrate]
    table verbatim (temperature, num_ctx, num_predict, ... — Ollama
    option names); empty keeps the endpoint defaults (temperature
    0)."""
    from . import config
    _, conf = config.load()
    t = conf.get("narrate")
    return dict(t) if isinstance(t, dict) else {}
