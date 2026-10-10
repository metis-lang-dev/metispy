"""The evaluator framework — BehaviorTree.cpp-shaped ergonomics over
the metis engine (docs/framework.md): Registry (port bindings with
learning-overlay resolution), Blackboard (the shared store), Evaluator
(tick + certificate-gated hot pack loading).

Generic like kernel and lang: never imports metis.domains (enforced by
tests/test_genericity.py)."""

from .registry import PortError, Registry           # noqa: F401
from .blackboard import Blackboard                  # noqa: F401
from .evaluator import Evaluator, catalog_domain    # noqa: F401
from .plugin import (PluginError, PortPlugin,       # noqa: F401
                     load_plugin)
