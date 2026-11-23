# Evaluation

Scripts to evaluate giglib **by hand**, modelled on
[tsdate's](https://github.com/tskit-dev/tsdate/tree/main/evaluation). They are not part of the
package or of the unit tests (pytest only collects `tests/`), and may need extra dependencies:
`pip install -r evaluation/requirements.txt`. They run giglib from this checkout.

Results go to `evaluation/results/`, which is git-ignored except for `performance.csv`, which
is committed as a reference to check progress against. Each row records the giglib version, git
SHA (suffixed `-dirty` if tracked files have uncommitted changes), CPU, OS, Python version and
timestamp.

| Script | Purpose |
|---|---|
| `performance.py` | Times core operations, in named cases (`--cases`). Writes one row per case, size and repeat. |
| `compare.py` | Compares two `performance.py` CSVs case by case (see below). |

```
python evaluation/performance.py
python evaluation/performance.py --quick   # a fast smoke test, written to performance_quick.csv
```

## Comparing performance across commits

To test whether a change affects speed, run the same cases on each version and compare:

```
git checkout main
python evaluation/performance.py --repeats 3 --output base.csv
git checkout my-branch
python evaluation/performance.py --repeats 3 --output new.csv
python evaluation/compare.py evaluation/results/base.csv evaluation/results/new.csv
```

If the base commit predates these scripts, run them from a separate worktree of that commit
instead, with this `evaluation/` folder copied in. The scripts run the giglib in the folder above
them, so this measures the base commit's code, and the copy is untracked, so the git SHA is not
marked `-dirty`:

```
git worktree add ../giglib-base <base-commit>
cp -r evaluation ../giglib-base/
python ../giglib-base/evaluation/performance.py --repeats 3 --output base.csv
python evaluation/performance.py --repeats 3 --output new.csv
python evaluation/compare.py ../giglib-base/evaluation/results/base.csv evaluation/results/new.csv
git worktree remove --force ../giglib-base
```

Cases that use features the base commit lacks will fail there: use `--cases` to choose those
that run on both.

For each case it prints the median value in each run, and the ratio new/base (below 1 is
better). Differences smaller than the min-to-max spread of the repeats are shown as `~` (noise),
so use at least 3 repeats and keep the machine otherwise idle. `compare.py` also prints the CPU
of each run: if they differ, the ratios include hardware effects.

To compare with the committed reference, write the new results to a *different* file (so as not
to overwrite it), and compare with `evaluation/results/performance.csv`. To compare with an
older commit's reference, use `git show <commit>:evaluation/results/performance.csv > old.csv`.
To update the reference, re-run without `--output`, and commit `performance.csv`.

## Adding a case

Add a function to `performance.py` that takes a size and yields rows (dicts with `variant`,
`size` and `value`), and add it to `CASES` with its unit and the sizes to use (normally, and
with `--quick`). Time with `common.best_time()`, which warms up first, then takes the best of
several timings.

## Findings

Measurements that justified changes, with the PR in which they were made (times on an Apple M2):

| Finding | PR |
|---|---|
| Mapping positions with `PositionMap` makes decoding haplotypes about 10% slower (a small object per iedge traced), accepted for correctness | #189 |
| Adding an iedge takes about 1.5 µs without validation. Validating incrementally costs about 0.5 µs per iedge if done once at the end, but each call to `validate()` has a fixed cost of about 50 µs, so validating after every row costs about 60 µs per iedge. Simulations therefore validate once per generation, using the validated rows | #190 |
