# metispy

[![ci](https://github.com/metis-lang-dev/metispy/actions/workflows/ci.yml/badge.svg)](https://github.com/metis-lang-dev/metispy/actions/workflows/ci.yml) [![parity](https://github.com/metis-lang-dev/metispy/actions/workflows/parity.yml/badge.svg)](https://github.com/metis-lang-dev/metispy/actions/workflows/parity.yml)

The metis python referee + research kit. Import name stays `metis`.

This is the SEMANTIC side of the metis toolchain: the exact measure
(forward filter, factorized tables, clique trees), learning
(Dirichlet notching, MAP-EM), decision, the surprise record +
`recover()` replay, and the stage-45 formalization harness
(`metis.lang.formalize.admit_text_pack`) — the gate cascade the
metis-loop drives. It is NOT the shipping toolchain: the compiler is
`metisc` (metis-lang, OCaml) and the runtime is metis-lang's C++
headers; metispy referees both.

- `metis/kernel/` — measure semantics, factorization, IR emission,
  canonical identity, notching, records.
- `metis/lang/` — parser/compiler/typecheck (the python front-end,
  kept as differential oracle for metisc), the LSP dev server, the
  formalize harness, and the discovery pair over `#`-directives
  (00-language-spec §8): `directives.py` (the `llp.trace` /
  `metisc --run` replay) and `repl.py` — the `metis <file.llp>`
  shell (readline editing/history/completion; step/sample on the
  reference sampler, assert/retract, exact `query` from the current
  state, `reload` with degrade-to-last-good, `save` back to
  directives).
- `metis/fw/` — the python evaluator product loop (truth generator
  for the C++ Evaluator).
- `metis/domains/` — the benchmark playground models (fiction/
  Ceptre, GoL, kitchen/Overcooked, nav, sysadmin, xtraffic, road).
  Their catalogs live in metis-catalog.
- `tools/` — golden/artifact generators writing INTO the metis-lang
  checkout (`METIS_LANG`, default: the sibling `../metis-lang`).
- `tests/` — the full gate suite, including the cross-toolchain
  parity gates (OCaml compiles ≙ python referees ≙ C++ interprets).

## Install

```sh
pip install metispy         # Python 3.12+; the import name is `metis`, the CLI `metis`
```

What you get: the referee and research kit above — the exact measure,
learning, decision, records, the `metis <file.llp>` shell and the python
front-end, with the domains' catalogs it builds from. It is not the
compiler: `metisc` is metis-lang's (OCaml), built from that repo. A domain
NAME (rather than a `.llp` path) needs the catalog's loader: install
metis-catalog, or set `METIS_CATALOG` to a checkout.

For development, beside its siblings:

```sh
pip install -e .            # into the metis-lang dev env
METIS_LANG=../metis-lang python3 -m pytest tests/ -q
python3 -m pytest tests/ -q -m 'not slow'   # the dev loop
```

Optional extras: `pip install overcooked-ai scipy` (kitchen),
`pyRDDLGym` (sysadmin), `pgmpy` (graph-evaluator comparisons).
Oracle parity vs the wayplan ll2pg lineage: `METIS_ORACLE_PATH`.

## Releasing

metispy goes to PyPI from GitHub by trusted publishing
(`.github/workflows/publish.yml`; no token anywhere): bump `version` in
`pyproject.toml`, commit, `git tag vX.Y.Z` (the tag must be `v` + that
version: the workflow refuses any other), push the tag: it publishes straight
through (the `pypi` environment has no required reviewer). The run builds the sdist and the wheel,
smoke-installs the wheel in a fresh venv outside the checkout
(`tools/smoke_wheel.sh`, which ci.yml runs on every push too) and uploads.
`workflow_dispatch` is the dry run: everything but the upload.

## Narration (`narrate`, metis/loop/narrate.py)

`narrate` tells the session's fired events as prose (`narrate SEED`: a
fresh trace sampled with that seed). The narrator sees only certified
material — the Foata beats, the final state and the catalog's `%%`
docs — so the docs are its vocabulary: one plain sentence per story
row. Four conventions keep a mined or machinery-heavy catalog
tellable:

| in the catalog | what the narrator gets |
|---|---|
| `%% [fold]` | bookkeeping: not told |
| `%% [fold P …] words` | a run of these events → one beat line: the words, `⇒ left:` the net-new atoms of P |
| `%% phrase one \| phrase two` (⊕ rule) | branch N's event tells phrase N |
| `% evidence: "…"` | nothing — single-`%` lines never reach the narrator |

Example, metis-catalog `domains/rddl/wildfire.llp`:

```
  % evidence: "k, the number of its neighbors burning at the start of the step"
  %% [fold tally] the fire counts the burning neighbours of each cell
  see_fire [dyn] : scan(X, Y, S) * tally(X, Y, K) * ... -o scan(X, Y, S2) * tally(X, Y, K2).
  %% X,Y, with K burning neighbours, catches fire | X,Y, with K burning neighbours, stays unburnt.
  catch [dyn] : fresh(X, Y) * tally(X, Y, K) * not_burning(X, Y) -o (burning(X, Y) @w pign(K) | not_burning(X, Y) @w ~pign(K)).
```

The material asks for a length sized to what is told, and a narration
the endpoint cut at its token cap ends on its last full sentence with
"(narration cut at N tokens; ask fewer beats)".

## License

Apache License 2.0 (see `LICENSE`, `NOTICE`).
