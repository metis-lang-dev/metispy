"""Regenerate metisc's diagnostic registry + hints (metis-lang
src/diag.ml) FROM metispy's metis/kernel/diagnostics.py, so the two are
never transcribed twice (spec 08 §B1). Rewrites the block between the
GENERATED markers; metis-lang gate 11 and tests/lang/test_diagnostics.py
diff the registry anyway.

    python tools/gen_metisc_registry.py [path/to/metis-lang/src/diag.ml]
"""
import re
import sys
from pathlib import Path

from metis.kernel.diagnostics import GATED_PHASES, HINTS, REGISTRY

BEGIN = "(* BEGIN GENERATED"
END = "(* END GENERATED *)"


def ocaml_hint(code, tmpl):
    parts = re.split(r'(\{\{|\}\}|\{[a-z0-9_]+\})', tmpl)
    out = []
    for p in parts:
        if p == "{{":
            out.append('"{"')
        elif p == "}}":
            out.append('"}"')
        elif p.startswith("{") and p.endswith("}"):
            out.append(f'g "{p[1:-1]}"')
        elif p:
            out.append('"' + p.replace("\\", "\\\\").replace('"', '\\"')
                       + '"')
    return f'  | "{code}" ->\n    String.concat "" [ ' + "; ".join(out) + " ]"


def block() -> str:
    reg = "let registry = [\n" + ";\n".join(
        f'  ("{c}", ("{s}", "{ph}"))' for c, (s, ph) in REGISTRY.items()
    ) + " ]\n"
    gated = "let gated_phases = [ " + "; ".join(
        f'"{p}"' for p in GATED_PHASES) + " ]\n"
    hints = ("let hint code data =\n"
             "  let g k = match List.assoc_opt k data with Some v -> v "
             "| None -> \"\" in\n  match code with\n"
             + "\n".join(ocaml_hint(c, t) for c, t in HINTS.items())
             + "\n  | _ -> \"\"\n")
    return (f"{BEGIN} from metispy metis/kernel/diagnostics.py by\n"
            f"   metispy/tools/gen_metisc_registry.py — never hand-edit *)\n"
            f"{reg}\n{gated}\n{hints}{END}\n")


def main(path):
    src = Path(path).read_text()
    i, j = src.index(BEGIN), src.index(END) + len(END) + 1
    Path(path).write_text(src[:i] + block() + src[j:])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         Path(__file__).resolve().parents[2] / "metis-lang/src/diag.ml")
