"""The loop core. Pure orchestration — every semantic judgement is
metispy's (admit_text_pack, recover); every model call goes through
one seam: an external COMMAND that reads a prompt on stdin and
writes .llp on stdout. `claude -p --model opus` is such a command;
so is an Ollama wrapper — the loop cannot tell the difference."""
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from metis.kernel.record import recover
from metis.lang import load_llp
from metis.lang.formalize import Formalization, admit_text_pack


# ---- SurpriseSource ------------------------------------------------

def load_records(path) -> List[dict]:
    """JSONL surprise records (kernel/record.py wire), newest last."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


# ---- Gates ---------------------------------------------------------

def base_shaped(llp: str) -> bool:
    """A candidate that declares its own stages and extends nothing is
    a BASE catalog, not a pack."""
    import re
    return (not re.search(r"(?m)^\s*extends\s", llp)
            and bool(re.search(r"(?m)^\s*stages\s*\(", llp)))


def admit(text: str, llp: str, base_path=None,
          scenarios: Sequence[Dict[str, int]] = (),
          steps: int = 6, mode: str = "pack") -> Formalization:
    """Gate a candidate: as a PACK against the base (mode "pack"), or as
    a BASE EDIT (mode "revise" — no base, as `revise` gates)."""
    base = load_llp(base_path) if base_path and mode == "pack" else None
    return admit_text_pack(text, llp, base=base,
                           scenarios=tuple(scenarios), steps=steps)


# ---- Formalizer (the LLM seam) -------------------------------------

# The SYSTEM half lives in loop.prompts (session- and file-
# overridable per seat); this is the material half.
USER_TEMPLATE = """\
The base catalog vocabulary you may CONSUME (packs never produce
base-namespace atoms — containment admission enforces it):

{vocabulary}

The observed, unmodeled behavior to formalize (this text is the
approved unit; your rules must carry %% docs quoting it):

{text}
{findings}"""


# D5 (spec 08): the REPAIR prompt carries no base catalog — the
# candidate's OWN source (it is being repaired) and the diagnostics,
# whose context lines are the only vocabulary that travels.
REPAIR_TEMPLATE = """\
The observed, unmodeled behavior to formalize (this text is the
approved unit; your rules must carry %% docs quoting it):

{text}

Your previous candidate:

{candidate}

It FAILED these gates. Each diagnostic quotes the offending
declaration, the declarations it references, and a repair hint —
repair and re-emit the full source:

{diagnostics}"""


# A model that runs away (thousands of tokens of non-LLP) must not
# balloon every later prompt: the echoed candidate is capped, and the
# cut is SAID in the prompt (measured 2026-10-04: an 8k-token runaway
# made the next repair prompt 132 KB before this cap).
CANDIDATE_CAP = 4000


def _capped(candidate: str) -> str:
    c = candidate.strip()
    if len(c) <= CANDIDATE_CAP:
        return c
    return (c[:CANDIDATE_CAP] + f"\n[... candidate truncated: "
            f"{len(c) - CANDIDATE_CAP} more characters omitted — far past "
            f"a catalog's size; re-emit ONLY the LLP source]")


def render_repair_prompt(text: str, candidate: str, report,
                         system: str = None) -> str:
    """The diagnostics-only repair prompt (D5): system + TEXT + the
    candidate's own source + the diagnostics' text renderings. Findings
    with no structured diagnostic (parse / roundtrip) travel as lines."""
    from metis.kernel.diagnostics import render_text
    from .prompts import resolve
    # only what FAILED the gate: warnings ride the report, not the repair
    diags = [d for d in (getattr(report, "diagnostics", []) or [])
             if d.severity == "error"]
    covered = {d.text for d in diags}
    blocks = [render_text(d).rstrip("\n") for d in diags]
    blocks += [f"error {f}" for f in report.findings
               if not any(f.endswith(t) for t in covered)]
    head, _ = resolve("formalize", system)
    return head + "\n\n" + REPAIR_TEMPLATE.format(
        text=text, candidate=_capped(candidate),
        diagnostics="\n\n".join(blocks))


def render_prompt(text: str, vocabulary: str,
                  findings: Sequence[str] = (),
                  system: str = None) -> str:
    from .prompts import resolve
    f = ""
    if findings:
        f = ("\nYour previous candidate FAILED these gates — repair "
             "and re-emit the full source:\n- " + "\n- ".join(findings))
    head, _ = resolve("formalize", system)
    return head + "\n\n" + USER_TEMPLATE.format(
        vocabulary=vocabulary, text=text, findings=f)


def run_backend(command, prompt: str, timeout: int = 600) -> str:
    """The seam takes an external COMMAND (argv list; prompt on
    stdin, LLP on stdout) or a CALLABLE(prompt) -> LLP — the
    in-process shape the Ollama adapter uses."""
    if callable(command):
        return command(prompt)
    r = subprocess.run(list(command), input=prompt, text=True,
                       capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"formalizer backend failed: {r.stderr[:400]}")
    return r.stdout


@dataclass
class FormalizeResult:
    ok: bool
    rounds: int
    llp: str = ""
    report: Optional[Formalization] = None
    history: List[Tuple[str, List[str]]] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)   # sent, in order
    mode: str = "pack"                  # how the last round was gated


def formalize_loop(text: str, vocabulary: str, backend: Sequence[str],
                   base_path=None, scenarios=(), steps: int = 6,
                   max_rounds: int = 4,
                   system: str = None, repair: str = "diagnostics",
                   start: Optional[str] = None,
                   mode: str = "pack") -> FormalizeResult:
    """text -> candidate -> gates -> findings -> repair, <= max_rounds.
    The candidate is UNTRUSTED; only the gate verdict is.

    repair: "diagnostics" (D5, default) — repair rounds send the TEXT,
    the candidate's own source and the diagnostics, never the base
    vocabulary; "full" — every round re-sends vocabulary + findings
    (the pre-D5 prompt, kept for measurement).
    start: a first candidate to repair instead of drafting one (round 1
    is then this candidate's gate verdict — repair-only measurement).
    mode: "pack" gates every round as a pack against the base;
    "revise" as a base edit; "auto" per round — a base-shaped candidate
    (own stages, no extends) as revise, else as a pack (a blank world
    passes "revise": there is nothing to extend)."""
    if repair not in ("diagnostics", "full"):
        raise ValueError(f"repair must be 'diagnostics' or 'full', "
                         f"got {repair!r}")
    history: List[Tuple[str, List[str]]] = []
    prompts: List[str] = []
    rep = None
    llp = ""
    for n in range(1, max_rounds + 1):
        if n == 1 and start is not None:
            llp = start
        else:
            if rep is None:
                prompt = render_prompt(text, vocabulary, (), system)
            elif repair == "diagnostics":
                prompt = render_repair_prompt(text, llp, rep, system)
            else:
                prompt = render_prompt(text, vocabulary, rep.findings,
                                       system)
            prompts.append(prompt)
            llp = run_backend(backend, prompt)
        how = mode if mode != "auto" else (
            "revise" if base_shaped(llp) else "pack")
        rep = admit(text, llp, base_path=base_path,
                    scenarios=scenarios, steps=steps, mode=how)
        history.append((llp, list(rep.findings)))
        if rep.ok:
            return FormalizeResult(True, n, llp, rep, history, prompts,
                                   how)
    return FormalizeResult(False, max_rounds, llp, rep, history, prompts,
                           how)


# ---- Explanation / replay ------------------------------------------

def replay(record: dict, catalog_path, init: Dict[str, int],
           steps: Optional[int] = None) -> dict:
    """recover() a record against a candidate catalog grounded on the
    record's scene — likelihood, posterior, MAP explanation."""
    doc = load_llp(catalog_path)
    prog = doc.catalog.ground(init)
    return recover(record, prog, steps=steps)
