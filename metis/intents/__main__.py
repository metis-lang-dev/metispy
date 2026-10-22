"""python -m metis.intents gen|check|schema|skeleton NAME — from a catalog's directory (or $METIS_CATALOG)."""
import sys

from . import main

sys.exit(main(sys.argv[1:]))
