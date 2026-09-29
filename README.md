# metispy

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

```sh
pip install -e .            # into the metis-lang dev env
METIS_LANG=../metis-lang python3 -m pytest tests/ -q
python3 -m pytest tests/ -q -m 'not slow'   # the dev loop
```

Optional extras: `pip install overcooked-ai scipy` (kitchen),
`pyRDDLGym` (sysadmin), `pgmpy` (graph-evaluator comparisons).
Oracle parity vs the wayplan ll2pg lineage: `METIS_ORACLE_PATH`.
