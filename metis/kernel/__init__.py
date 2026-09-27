"""The domain-free engine kernel.

GENERICITY CONDITION: nothing under metis.kernel may import from
metis.domains — enforced by tests/test_genericity.py. Domain knowledge
enters only through data (clauses, weights, policies) passed in."""

from .clause import Clause, Link                              # noqa: F401
from .program import Program                                  # noqa: F401
from .weight import (falling_factorial, transition_count,     # noqa: F401
                     choice_weight)
from .filter import ForwardProjection                         # noqa: F401
from .sampler import sample_final, run_once, run_trace        # noqa: F401
from .nets import (canonical, canonical_bytes,                # noqa: F401
                   program_key, run_key,
                   independent_events, ProjectionCache)
from .decision import (PlanResult, cached_projector, plan,  # noqa: F401
                       round9)
from .filter import apply_event                               # noqa: F401
from .posterior import (Conditioned, MapResult,               # noqa: F401
                        behavior_posterior, map_derivation,
                        surprise)
from .sync import (sync_program, sync_policy,                 # noqa: F401
                   steps_per_tick, marginal_on, expected_on)
from .sample_eval import SampledProjection                    # noqa: F401
from .thts import mcts_plan                                   # noqa: F401
from .schema import (Pattern, pat, ground_atom, Signature,    # noqa: F401
                     ClauseSchema, LinkSchema)
from .horn import HornClause, HornDB, fact                    # noqa: F401
from .ground import (ground_schema, ground_link,              # noqa: F401
                     ground_program)
from .typing import Namespace, ns                             # noqa: F401
from .catalog import (Catalog, CatalogError, Entry,           # noqa: F401
                      Provenance)
