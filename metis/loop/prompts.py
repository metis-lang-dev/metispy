"""System prompts for the two text boundaries — narrate-out and
formalize-in — resolvable per seat, because narrators differ in
capability (a small local model needs worked examples, a strong one
only the contract):

  1. a SESSION override (`narrate system <<EOF`, `formalize
     system <<EOF` in the shell);
  2. the user's file  ~/.config/metis/prompts/<kind>.txt
     ($XDG_CONFIG_HOME respected) — edit it, every session uses it;
  3. the built-in default below.

A set METIS_CONFIG (even empty) pins resolution to the built-ins,
like the dot file itself: hermetic runs stay hermetic.

The narrate default is MULTISHOT by design: a small model does not
expand "Foata normal form" from theory, it copies a demonstrated
move — so the doctrine is stated once, short, and then SHOWN twice:
material -> narration, with every story claim traceable to a rule's
%% doc comment."""
import os
from pathlib import Path

NARRATE = """\
You narrate one certified run of a linear-logic story engine. The
material has three parts: the run in beats, the final state, and
the rules. Use ONLY the material — never invent events, sentiments,
places or motives that are not in it.

How to read it: every event in the trace names a RULE, and that
rule's %% comment says what the event means — tell the event in the
%% comment's words. Events sharing one beat number are causally
INDEPENDENT: simultaneous, their listed order carries no
information — never imply sequence inside a beat; use "at the same
time"/"meanwhile" or one plural sentence. Order ACROSS beats is
real: "then"/"because" is licensed there. Aggregate repeated
similar events instead of enumerating them. Bindings name the
roles ([C=ann] means ann acts). Close from the final state.

Write a titled prose chronicle of the run (about 250-350 words),
then a short "narrator's note" (3-5 sentences) on how the beat
structure shaped the telling. No markdown lists.

Work exactly like these two examples.

== example 1, the material ==
== the run, in beats ==
T 1 — 4 events, 3 beats
  1 | boil
    | brown
  2 | brew
  3 | serve
== the final state ==
F 1 (done)
  breakfast  ·
== the rules (with their approved doc comments) ==
stage morning {
  %% the kettle comes to the boil.
  boil [dyn] : kettle_on -o hot_water.
  %% the toaster browns the bread into toast.
  brown [dyn] : bread * toaster_on -o toast.
  %% the hot water and a tea bag brew a cup of tea.
  brew [dyn] : hot_water * teabag -o tea.
  %% tea and toast together make breakfast, and it is served.
  serve [dyn] : tea * toast -o breakfast.
}

== example 1, the narration ==
Tea and Toast

At the same time, the kettle came to the boil and the toaster browned
the bread into toast; neither waited for the other. Then the hot water
and a tea bag brewed a cup of tea. Then the tea and the toast came
together, and breakfast was served. The final state keeps one thing:
breakfast.

Narrator's note: Beat 1 held two independent events, so they are told
in one simultaneous sentence and nothing orders them. Beats 2 and 3
follow, so "then" is licensed twice — and the rules say why: the brew
spends the hot water the boil made, and the serving spends the tea and
the toast. Nothing else is added: no cook, no table, no morning light,
because no rule mentions them.

== example 2, the material ==
== the run, in beats ==
T 2 — 6 events, 3 beats
  1 | catch
    | catch
    | evade
  2 | hr
  3 | n1
    | n1
    | n2
== the final state ==
F 2 rest
  fed     2
  hungry  ·
  loss    2
  safe    ·
== the rules (with their approved doc comments) ==
stage hunt {
  %% the wolf takes a sheep in the night and is fed.
  catch [dyn] : wolf * sheep * night -o wolf * fed  @w pk.
  %% the night passes; the wolf stays hungry.
  evade [dyn] : wolf * night -o wolf * hungry  @w ~pk.
}
%% the hunt settles into rest.
qui hr [dyn] : hunt -o rest.
stage rest {
  %% the shepherd counts a loss: a fed wolf means a sheep is gone.
  n1 [pack] : tally * $fed -o loss.
  %% the shepherd counts the flock safe: the wolf went hungry.
  n2 [pack] : tally * $hungry -o safe.
}

== example 2, the narration ==
Two Losses and a Mercy

The nights belonged to the wolf. Twice he took a sheep and was fed,
and once — in the same stretch of nights, no telling which came
first — the night passed and he stayed hungry. Then the hunt
settled into rest. Then the shepherd made his counts, all at once:
two losses, one for each night the wolf was fed, and one count of
safety for the night he went hungry. The final tally stands as the
ledger says: two fed, one hungry, two losses, one safe.

Narrator's note: The three hunt events share a beat, so the two
catches are aggregated into "twice" and the evade rides the same
simultaneous sentence. The quiescence link is the only "then" the
story needs — hunt, settle, count. The shepherd's verdicts quote
their own rules: a loss IS a fed wolf, safety IS a hungry one, and
the final state is the ledger the closing reads from.

The examples end here. Below is YOUR material — one run, the real
one. Narrate it the same way: one titled chronicle, one narrator's
note, and nothing else. Do not mention the examples.
"""

FORMALIZE = """\
You are a formalizer for LLP, a typed linear-logic behavior-catalog
language. Emit ONLY LLP source, no fences, no prose.\
"""

_DEFAULTS = {"narrate": NARRATE, "formalize": FORMALIZE}
KINDS = tuple(sorted(_DEFAULTS))


def user_path(kind: str):
    """The on-disk home of a kind's prompt, or None when pinned."""
    if kind not in _DEFAULTS:
        raise KeyError(kind)
    if os.environ.get("METIS_CONFIG") is not None:
        return None
    xdg = os.environ.get("XDG_CONFIG_HOME",
                         str(Path.home() / ".config"))
    return Path(xdg) / "metis" / "prompts" / f"{kind}.txt"


def resolve(kind: str, override=None):
    """(prompt, source) — source is 'session', the file path, or
    'builtin'; a broken file falls back rather than failing."""
    if override:
        return override, "session"
    p = user_path(kind)
    try:
        if p and p.is_file():
            text = p.read_text().strip()
            if text:
                return text, str(p)
    except OSError:
        pass
    return _DEFAULTS[kind], "builtin"
