"""Write metis/trace_tools.json, the trace verbs and lint as declared tools, FROM their argparse
(metis.trace.parser, metis.lint.parser; loop-workspace s-5b4be5fa, metis-loop docs/WORKSPACE.md §3.1): the
backend's trace channel serves exactly these, the pi loop's extension exposes exactly these.

    python tools/gen_trace_tools.py           # rewrite the file
    python tools/gen_trace_tools.py --check   # exit 1 when the file is not what the verbs declare
"""
import sys

from metis.catalog.workspace import TOOLS_FILE, tools_json


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    want = tools_json()
    if "--check" in argv:
        have = TOOLS_FILE.read_text() if TOOLS_FILE.is_file() else ""
        if have != want:
            print("%s is not what the verbs declare: python tools/gen_trace_tools.py" % TOOLS_FILE, file=sys.stderr)
            return 1
        print("%s: as the verbs declare" % TOOLS_FILE.name)
        return 0
    TOOLS_FILE.write_text(want)
    print("wrote %s (%d tools)" % (TOOLS_FILE, want.count('"name": ')))
    return 0


if __name__ == "__main__":
    sys.exit(main())
