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
re-derives semantics, it only re-tells it.

Machinery folds. A rule whose %% doc opens with `[fold]` (or
`[fold P Q]`) is the catalog's bookkeeping — an unrolled count, a walk
over cells — not story: its events are not listed one by one; each run
of them becomes ONE line, its docs' words and the net-new atoms of the
named predicates (`[fold tally]` on the wildfire count shows the
counts it left: tally(x2,y3,k1)); a bare `[fold]` has no words, and a
run with neither words nor atoms left is not told at all. A
forced/do/env event is never folded (agency is told), and folded
rules leave the rules section. A ⊕ rule's doc may give one phrase per
branch, " | " between them: branch N's event tells phrase N.

The cut. A narration ends on a sentence: when the endpoint stops at
its token cap, the text is cut back to its last full sentence and says
so — "(narration cut at N tokens; ask fewer beats)" — never a word
broken off mid-sentence."""
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

Now narrate this run, in at most {words} words.
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
    folds = fold_rules(docs)
    lines = _pretty_trace(seed, rows)
    # _pretty_trace lists beat lines by (level, firing order): the
    # same order recovers each line's event for its effect
    levels = foata_levels(rows) if rows else []
    order = sorted(range(len(rows)), key=lambda i: (levels[i], i)) \
        if rows else []
    evs = [None] + [rows[i][2] for i in order]
    # spec 10: faithfulness extends to agency — an intervention is
    # never presented as sampled
    hows = [None] + [(marks or [None] * len(rows))[i] for i in order]
    lvls = [None] + [levels[i] for i in order]
    beats = "\n".join(_fold_beats(lines, evs, hows, lvls, docs, folds))
    forced = [rows[i][2].name for i in range(len(rows))
              if (marks or [None] * len(rows))[i] in _HOW]
    if forced or base_do:
        beats += HONESTY.format(lines="\n".join(
            [f"[do {op}] — the start state, set by hand"
             for op in base_do]
            + [f"{_HOW[(marks or [])[i]]} {rows[i][2].name}"
               for i in range(len(rows))
               if (marks or [None] * len(rows))[i] in _HOW]))
    told = sum(1 for ev, h in zip(evs[1:], hows[1:])
               if not (_base(ev.name) in folds and h not in _HOW))
    return {"beats": beats,
            "state": "\n".join(_pretty_state(seed, final_stage,
                                             sorted(kappa))),
            "rules": story_rules(rules_src, set(folds)),
            "words": budget(told)}


def budget(told: int) -> int:
    """The narration's length from the events it has to tell (folded
    machinery lines count as one each): about 25 words an event,
    80 at least, 300 at most — a cap a 700-token endpoint can keep."""
    return max(80, min(300, 40 + 25 * told))


_FOLD = re.compile(r"^\[fold((?:\s+[\w-]+)*)\]\s*")


def fold_rules(docs: Dict[str, str]) -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    """rule -> (its words, the predicates whose net-new atoms a fold
    shows), for the rules whose doc opens with [fold …]."""
    out = {}
    for r, d in docs.items():
        m = _FOLD.match(d)
        if m:
            words = re.split(r"(?<=[.!?])\s", d[m.end():], maxsplit=1)[0]
            out[r] = (re.sub(r"\s*\([^)]*\)", "", words).strip().rstrip("."),
                      tuple(m.group(1).split()))
    return out


def _base(name: str) -> str:
    return re.sub(r"-\d+$", "", name.split("[")[0])


def _fold_beats(lines, evs, hows, lvls, docs, folds):
    """The beat lines with each run of folded (machinery) events
    turned into one line: the beats it spans, its rules' words, the
    net-new atoms of the predicates the folds name."""
    out, group = [lines[0]], []
    width = len(str(max([l for l in lvls[1:] if l is not None], default=0) + 1))

    def flush():
        if not group:
            return
        a, b = group[0][1] + 1, group[-1][1] + 1
        span = f"{a}" if a == b else f"{a}–{b}"
        said, preds = [], set()
        for ev, _ in group:
            w, ps = folds[_base(ev.name)]
            if w and w not in said:
                said.append(w)
            preds |= set(ps)
        net = {}
        for ev, _ in group:
            for at, n in ev.produce.items():
                net[at] = net.get(at, 0) + n
            for at, n in ev.consume.items():
                net[at] = net.get(at, 0) - n
        left = sorted(at for at, n in net.items()
                      if n > 0 and at.split("(")[0] in preds)
        eff = f"  ⇒ left: {', '.join(left)}" if left else ""
        n = len(group)
        group.clear()
        if not said and not left:          # bare bookkeeping ([fold] with no words): nothing to tell
            return
        out.append(f"  {span:>{width}} | ⋯ {n} event{'s' if n != 1 else ''} of machinery, folded"
                   f" — {'; '.join(said) or 'bookkeeping'}{eff}")

    last = None
    for line, ev, h, lvl in zip(lines[1:], evs[1:], hows[1:], lvls[1:]):
        if _base(ev.name) in folds and h not in _HOW:
            group.append((ev, lvl))
            continue
        flush()
        # a beat's first told event carries the beat number, even when
        # folded events of the same beat came first in _pretty_trace
        m = _BEAT.match(line)
        if m and lvl != last and not m.group(1).strip().rstrip("|").strip():
            line = f"  {lvl + 1:>{width}} | " + line[m.end(1):]
        elif m and lvl == last and m.group(1).strip().rstrip("|").strip():
            line = f"  {'':>{width}} | " + line[m.end(1):]
        last = lvl
        out.append(gloss(line, docs, ev) + (f"  {_HOW[h]}" if h in _HOW else ""))
    flush()
    folded = sum(1 for ev, h in zip(evs[1:], hows[1:]) if _base(ev.name) in folds and h not in _HOW)
    if folded:
        out[0] += f" ({folded} of them machinery, folded)"
    return out


def story_rules(src: str, folded=()) -> str:
    """The rules minus file-level noise: single-% comments and
    #-directive lines (init/steps/query) are notes for the author and
    the interpreter, not story — a small narrator retells them as
    plot. %% doc comments, the narrator's vocabulary, stay. A folded
    rule (its doc opens with [fold …]) leaves with its doc: machinery
    is not the story's vocabulary."""
    keep, block, skipping = [], [], False
    for l in src.splitlines():
        t = l.strip()
        if skipping:                       # the folded rule's statement, to its final '.'
            if t.endswith("."):
                skipping = False
            continue
        if t.startswith("#") or (t.startswith("%") and not t.startswith("%%")):
            continue
        if t.startswith("%%"):
            block.append(l)
            continue
        m = _RULE.match(l)
        if m and m.group(1) in folded:
            block = []
            skipping = not t.endswith(".")
            continue
        keep += block + [l]
        block = []
    return "\n".join(keep + block).strip()


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
    # a ⊕ rule's doc may give one phrase per branch, " | " between
    # them: branch N (the event name's -N) tells its own outcome
    br = re.search(r"-(\d+)$", m.group(2))
    alts = [a.strip() for a in first.split(" | ")]
    if br and len(alts) > 1 and int(br.group(1)) <= len(alts):
        first = alts[int(br.group(1)) - 1]
        if first and first[-1] not in ".!?":
            first += "."
    binds = dict(kv.split("=", 1) for kv in (m.group(3) or "").split(",")
                 if "=" in kv)
    if binds:
        keys = "|".join(map(re.escape, sorted(binds, key=len,
                                                reverse=True)))
        # a quoted span is source text verbatim (wgc's row doc "A farmer
        # with a wolf…" under A=far): role letters are replaced outside
        # quotes only, never inside a quotation
        parts = re.split(r'("[^"]*")', first)
        first = "".join(p if i % 2 else re.sub(
            rf"\b({keys})\b", lambda k: binds[k.group(1)], p)
            for i, p in enumerate(parts))
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
    meta = {}
    text = llm.complete(spec, body, timeout=timeout,
                        options=options(), system=head, meta=meta)
    return end_on_sentence(text, meta)


_END = re.compile(r"""[.!?…]["'”’)\]*]*(?=\s|$)""")


def end_on_sentence(text: str, meta: dict) -> str:
    """The cut rule: a narration the endpoint stopped at its cap
    (meta['cut']) is cut back to its last full sentence and says where
    it was cut; an uncut one is returned as it is."""
    if not meta.get("cut"):
        return text
    text = text.rstrip()
    ends = list(_END.finditer(text))
    kept = text[:ends[-1].end()] if ends else ""
    n = meta.get("tokens")
    note = f"(narration cut at {n} tokens; ask fewer beats)" if n \
        else "(narration cut at the token cap; ask fewer beats)"
    return (kept + "\n\n" + note) if kept else note


def options() -> Dict[str, object]:
    """The narrate seat's sampling knobs: the dot file's [narrate]
    table verbatim (temperature, num_ctx, num_predict, ... — Ollama
    option names); empty keeps the endpoint defaults (temperature
    0)."""
    from . import config
    _, conf = config.load()
    t = conf.get("narrate")
    return dict(t) if isinstance(t, dict) else {}
