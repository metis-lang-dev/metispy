"""D5 measurement (spec 08): the formalize REPAIR loop with diagnostics-
only prompts vs today's full-vocabulary prompts, on a fixture suite of
broken candidates, against a local Ollama model.

Each case starts from a FIXED broken candidate (`formalize_loop(...,
start=...)`), so both modes repair the same input and only the repair
prompt differs. Per mode and case it records: converged?, rounds, and
per model call the prompt / completion tokens Ollama reports
(prompt_eval_count / eval_count) plus the prompt's characters.

    python tools/measure_repair.py [--model phi4-mini:latest]
        [--rounds 4] [--out results.json]
"""
import argparse
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from metis.loop import ollama
from metis.loop.core import formalize_loop

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "metis-lang/corpus"
TRAGEDY = ROOT / "metis-catalog/domains/tragedy/tragedy.llp"

MEADOW = (ROOT / "metispy/tests/loop/test_loop.py").read_text()
MEADOW_SRC = MEADOW.split('MEADOW = """', 1)[1].split('"""', 1)[0]
SHEPHERD_SRC = MEADOW.split('SHEPHERD = """', 1)[1].split('"""', 1)[0]
PACKMARK = (CORPUS / "packmark.llp").read_text()

TEXT_COIN = "%% market coin: a mark trades on the coin flip outcome."
TEXT_TALLY = ("%% the shepherd tallies the flock at rest: a fed wolf "
              "means a loss, a hungry one safety.")
TEXT_GRIEF = ("%% grief: whoever loved the dead grieves for them.")

TRAGEDY_PACK = """catalog grief 1.
extends tragedy.
provenance text "grief pack" "2026-10-04".
namespace gr produce (story).
pred grieves(character,character) : gr.
stage all {
  %% grief: whoever loved the dead grieves for them.
  grieve [story] : $philia(C,D) * $dead(D) -o grieves(C,D,D).
}
"""


def cases():
    """(name, text, base_source, broken candidate) — base as SOURCE so
    meadow (a test-file fixture) and corpus files are handled alike."""
    pb = (CORPUS / "packbase.llp").read_text()
    return [
        ("packmark-drift", TEXT_COIN, pb,
         PACKMARK.replace("mkt", "unlock", 1)),
        ("packmark-produce", TEXT_COIN, pb,
         PACKMARK.replace("-o seen.", "-o seen * heads.")),
        ("shepherd-doc", TEXT_TALLY, MEADOW_SRC,
         SHEPHERD_SRC.replace("  %% loss.\n", "")),
        ("shepherd-consume", TEXT_TALLY, MEADOW_SRC,
         SHEPHERD_SRC.replace("tally * $fed", "tally * fed")),
        ("shepherd-namespace", TEXT_TALLY, MEADOW_SRC,
         SHEPHERD_SRC.replace("pred tally : sh.", "pred tally : md.")),
        ("shepherd-arity", TEXT_TALLY, MEADOW_SRC,
         SHEPHERD_SRC.replace("-o loss.", "-o loss(n0).")),
        ("tragedy-arity (large base)", TEXT_GRIEF, TRAGEDY.read_text(),
         TRAGEDY_PACK),
    ]


def make_backend(model, host, calls):
    """A loop backend that records Ollama's own token counts per call
    (same payload as metis.loop.ollama.generate: temperature 0)."""
    def backend(prompt):
        req = urllib.request.Request(
            host.rstrip("/") + "/api/generate",
            data=json.dumps({"model": model, "prompt": prompt,
                             "stream": False,
                             "options": {"temperature": 0}}).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.loads(r.read())
        calls.append({"prompt_chars": len(prompt),
                      "prompt_tokens": out.get("prompt_eval_count"),
                      "completion_tokens": out.get("eval_count"),
                      "seconds": round(time.time() - t0, 1)})
        return ollama.strip_fences(out["response"])
    return backend


def run_case(name, text, base_src, broken, mode, model, host, rounds):
    with tempfile.NamedTemporaryFile("w", suffix=".llp",
                                     delete=False) as f:
        f.write(base_src)
        base_path = f.name
    calls = []
    try:
        res = formalize_loop(text, base_src,
                             make_backend(model, host, calls),
                             base_path=base_path, max_rounds=rounds,
                             repair=mode, start=broken)
    finally:
        Path(base_path).unlink()
    return {"case": name, "mode": mode, "ok": res.ok,
            "rounds": res.rounds, "calls": calls,
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
            "final_findings": list(res.report.findings
                                   if res.report else [])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ollama.default_model())
    ap.add_argument("--host", default=ollama.default_host())
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    results = []
    for name, text, base_src, broken in cases():
        for mode in ("full", "diagnostics"):
            r = run_case(name, text, base_src, broken, mode, a.model,
                         a.host, a.rounds)
            results.append(r)
            first = r["calls"][0]["prompt_tokens"] if r["calls"] else None
            print(f"{name:28s} {mode:11s} ok={r['ok']!s:5s} "
                  f"rounds={r['rounds']} first-repair-prompt={first} "
                  f"total-prompt={r['prompt_tokens']}", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "rounds": a.rounds, "results": results},
            indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
