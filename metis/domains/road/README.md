# The road domain — where it stands (2026-07-22)

The road domain is fully catalog-native: `.llp` catalogs + weight-
port bindings are the ONLY encoding. The old hand-built oracle
(`lldsl.py` build_program, `planner.py` make_policy/plan) is RETIRED
(stage 42): the simulator and the oracle had diverged, the recode-
identity proof it served is history — it held from stage 33 until
retirement, and its final certified values are pinned as goldens in
`tests/test_gridcat.py` (program keys per scene class, unpoliced
P(crash), the RSS-shield zero-crash-mass property, and plan()
decisions/scores per objective). Those pins are what keep future
catalog edits honest in-repo; the triangle corpus
(ocaml/parity/manifest.txt: road-close) keeps the handle encoding
byte-verified across OCaml ≙ Python ≙ C++.

## The catalogs

- **`catalogs/grid_domain.llp`** — the full lolli behavior kernel,
  LIFTED: five ego actions, four vehicle actions, directed walkers
  (both signs), holonomic ped discs, occlusion ghosts, the crash
  sweep, and the run-token stage cycle. Rules mention only relations
  and HANDLES; every number lives in compiler-generated Horn tables
  or behind ports.
- **`catalogs/handles_domain.llp`** — the geometry-as-handles
  encoding (RSS gap classes; `docs/road-encoding.md`). The grid is a
  finer choice of semantic classes inside this same handle
  discipline — one lane vocabulary (`leftof`/`rightof`), two
  granularities.
- `catalogs/mini.llp` + `wheelchair.llp` — the typed catalog demo
  and its containment pack (`mini.py` is a thin loader over them;
  also the crash-freeze COUNTEREXAMPLE for the factored emission,
  tests/test_counterexamples.py).

## The stack

- **`gridcat.py`** — the semantic compiler: scene geometry → Horn
  tables + instance + init (`build`); the policy as weight-port
  bindings (`road_bindings` — RSS shield, behavior rows, danger
  response, manifest + occlusion priors); the Python planner
  (`plan` / `plan_seam`, the analog of the C++ `plan_metis`); the
  state/window helpers (`decode_state`, `windows`, `slices_for`)
  and the objective readouts (`OBJECTIVES`, `crash_probability`,
  `expected_dof`/`_danger_time`/`_progress`, `reachable_set_size`)
  absorbed at oracle retirement — scoring is production machinery.
- **`semantic.py`** — the handle-encoding compiler
  (`SemanticCompiler`, `RoadModel`, `handoff`): geometry as `Lane`
  objects, the semantic graph as facts, the motion handoff by
  handle.
- **`fw.py`** — the framework bindings (`road_evaluator`): the
  `metis.fw.Evaluator` seam, projecting through `gridcat`.
- **`sim.py`** — `Car`/`Road`/`RoadState`, `kinematics` (the tick
  dynamics; the Horn tables are GENERATED from these).
- **`semantics.py`** — `rss_actions`, `dangerous_ahead`, occlusion
  frontiers (the `road_bindings` ports call these).
- **`profile.py`** — vehicle/wheelchair kinematic profiles.
- **`sumo_harness.py`** — the SUMO co-simulation harness (plans via
  `gridcat.plan`).

## What retired with the oracle (stage 42)

`lldsl.py`, `planner.py` (policy/planner half — the objective
readouts moved to gridcat), `experiments.py` (oracle-era CLI), and
the oracle-comparison suites (test_peds, test_rss,
test_wheelchair_closure, test_decision_parity). Behavioral coverage
of walkers/peds/ghosts lives on in the pinned gridcat gates and the
grid catalog itself; wayplan `plans.txt` parity retired with the
divergence.
