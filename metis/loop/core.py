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

def admit(text: str, llp: str, base_path=None,
          scenarios: Sequence[Dict[str, int]] = (),
          steps: int = 6) -> Formalization:
    base = load_llp(base_path) if base_path else None
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


def formalize_loop(text: str, vocabulary: str, backend: Sequence[str],
                   base_path=None, scenarios=(), steps: int = 6,
                   max_rounds: int = 4,
                   system: str = None) -> FormalizeResult:
    """text -> candidate -> gates -> findings -> repair, <= max_rounds.
    The candidate is UNTRUSTED; only the gate verdict is."""
    findings: List[str] = []
    history: List[Tuple[str, List[str]]] = []
    for n in range(1, max_rounds + 1):
        llp = run_backend(backend, render_prompt(text, vocabulary,
                                                 findings, system))
        rep = admit(text, llp, base_path=base_path,
                    scenarios=scenarios, steps=steps)
        history.append((llp, list(rep.findings)))
        if rep.ok:
            return FormalizeResult(True, n, llp, rep, history)
        findings = rep.findings
    return FormalizeResult(False, max_rounds, llp, rep, history)


# ---- Explanation / replay ------------------------------------------

def replay(record: dict, catalog_path, init: Dict[str, int],
           steps: Optional[int] = None) -> dict:
    """recover() a record against a candidate catalog grounded on the
    record's scene — likelihood, posterior, MAP explanation."""
    doc = load_llp(catalog_path)
    prog = doc.catalog.ground(init)
    return recover(record, prog, steps=steps)
