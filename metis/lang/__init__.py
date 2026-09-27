"""rulescript v2 — the metis catalog language (docs/language.md).

Ceptre-core rules + the metis header (types, namespaces, pred
signatures, ports) + weights-as-ports. Pipeline:

    text --parse--> lossless AST --compile--> CatalogDoc
         (parse(pretty(ast)) == ast)          (kernel Catalog + ports)
    CatalogDoc --to_pack--> JSON pack --from_pack--> CatalogDoc

Generic like the kernel: this package never imports metis.domains
(enforced by tests/test_genericity.py)."""

from .parser import parse, parse_file                          # noqa: F401
from .ast import pretty                                        # noqa: F401
from .compiler import (CatalogDoc, LangError, compile_catalog,  # noqa: F401
                       factor_hooks, ground_reads, load_llp,
                       resolve_includes, weight_policy)
from .pack import to_pack, from_pack                           # noqa: F401
