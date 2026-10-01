"""metis.loop — the formalization/explanation loop, IN the kernel
package (user decision 2026-10-01: the metis shell is the one REPL;
the metis-loop repo is wrappers — web console, Claude Code skill, CLI
bindings). Four seams: SurpriseSource (JSONL records), Formalizer /
Narrator (one LLM endpoint, loop/llm.py), Gates (admit_text_pack),
Sink (the approval artifact). The shell mounts the verbs
(metis.loop.verbs), the console speaks the intent wire
(metis.loop.wire)."""
from .core import (FormalizeResult, admit, formalize_loop, load_records,
                   render_prompt, replay)

__all__ = ["FormalizeResult", "admit", "formalize_loop", "load_records",
           "render_prompt", "replay"]
