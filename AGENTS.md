# Notes for coding agents

giglib stores **genetic inheritance graphs** (GIGs): like _tskit_ tree sequences, but each edge
("iedge") records the inherited interval in both the child's and the parent's coordinates, so
genomes can rearrange (inversions, duplications, deletions, translocations). It is a proof of
concept, aimed at forward simulation of structural variants. Start with `README.md`, then
`docs/concepts.md`, `docs/usage.md` and `docs/validation.md`.

## Layout

| Path | Contents |
|---|---|
| `giglib/tables.py` | The tables (`Tables`, `IEdgeTable`, `NodeTable`, `MutationTable`, ...), validation, and table algorithms (`sort`, `sample_resolve`, `decapitate`, `find_mrca_regions`, ...) |
| `giglib/graph.py` | `Graph`: a validated, frozen view of a set of tables, with indexes and allele decoding |
| `giglib/constants.py` | `ValidFlags` (the validity requirements) and other constants |
| `giglib/tools/` | Tools that build on the core, e.g. converting tree sequences with indels (`giglib.tools.indels`) and simple forward simulators (`giglib.tools.simulation`) |
| `tests/` | Unit tests (`tests/tools/` for the tools). `tests/gigutil.py` has small helpers for building test fixtures |
| `docs/` | Documentation (jupyter-book; code cells are executed). `docs/design/` has design notes, which record decisions and open questions |
| `evaluation/` | Scripts run by hand to measure performance (see `evaluation/README.md`), not part of the package or the tests |

## Commands

```
python -m pytest                                   # tests (Python >= 3.12)
pre-commit run --all-files                         # ruff (line length 120) and other checks
(cd docs && ./build.sh)                            # docs; warnings are errors
python evaluation/performance.py --quick           # a quick performance smoke test
```

## Priorities

- **Correctness and clarity first**, then **internal consistency** of the API, ahead of
  compatibility with _tskit_ or with earlier versions of giglib (breaking changes are fine),
  and ahead of micro-optimisation (optimise later, guided by measurements).
- **One implementation of each rule.** For example, all the iedge validity rules are in
  `Tables._check_iedges()`, and all mapping of positions through iedges uses `PositionMap`. Don't
  add a second version of either; extend the existing one.
- **Follow the API conventions** in the "Conventions" section of `docs/usage.md`, e.g.
  methods on a `Graph` return objects and those on `Tables` return IDs; nodes come first when
  adding rows, and iedge coordinates and chromosomes are given by keyword.

## Coding style and quality

- Prefer semantic correctness, well-defined invariants and clear error messages to clever tricks.
- Consider the **algorithmic** time and memory complexity of code on critical paths (e.g. adding
  rows, validation, tracing through the graph): avoid work that grows with the size of the tables
  when it needn't. This is not micro-optimisation, which should wait for measurements.
- Apply DRY carefully: share logic when it makes the code clearer and keeps rules in one place,
  but avoid over-abstraction or generic frameworks that obscure intent.
- Write readable, explicit code with simple control flow and minimal "magic"; prefer
  straightforward implementations to compact but opaque idioms. Keep code concise but not cryptic,
  and remove dead or unused code and unnecessary indirection.
- Keep responsibilities coherent: e.g. the tables own their data, indexes and validity flags, and
  a `Graph` is a read-only view of validated tables.
- In tests, prefer simple, direct assertions that fail fast. Test code should not be defensive,
  over-general, or try to recover from failures.

## Validity

Validity is tracked as separate requirements (`ValidFlags`), not as a single yes or no; see
`docs/validation.md`. In particular:

- Table algorithms must check for the flags they need (e.g. with `Tables._require_iedge_flags()`),
  and should keep the flags that their output satisfies by construction.
- `Tables.validate()` is incremental: it only checks the rows added since it was last called (the
  first `iedges.num_validated` rows are the validated rows). Code that changes existing rows,
  rather than appending, must keep this record correct.
- Requirements that depend on node times are tied to a version of the node table, which changes
  when its existing rows change (`NodeTable._changed()`). Code that edits node rows in place must
  call it.
- Edit table columns through methods that keep the indexes and flags up to date: the public
  columns are read-only, and the internal `_data` arrays should only be written by table code.

## Working practices

- **Measure performance changes** before and after, with `evaluation/performance.py` and
  `evaluation/compare.py`, and record findings that justify a design decision in the findings
  table of `evaluation/README.md`, linked to the PR.
- **Record design discussions** (decisions, alternatives, open questions) in the relevant note in
  `docs/design/`, or in an issue, so they can be picked up later.
- **Fix unrelated bugs in separate PRs.**
- Add tests for each change, and keep the docs (including the executed examples) up to date.
- **The user may be editing files while you work:** don't revert or overwrite changes you didn't
  make, unless asked, and don't reformat or touch files that the task doesn't need.
- **Allow long-running commands to finish:** the docs build and evaluation runs can take several
  minutes, so set timeouts generously (or run them in the background and check on them).
- Run performance cases one at a time on an otherwise idle machine, not in parallel, so that
  timings aren't distorted.
