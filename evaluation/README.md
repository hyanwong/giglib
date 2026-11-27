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
| `find_homology(mode="closest_homologues")` traced every lineage to the roots. In a graph where every position descends from a single base (20 genomes, 400 generations, genomes of length 46 and 84), that took 90 s, against 0.1 s for `"first_mrca"`. Stopping once every position still being traced has found homology gives the same results in 2.5 s | #230 |
| Most of the time of `find_homology` and `sample_resolve` went into the general-purpose `portion` interval library (3.7 µs per intersection of two intervals, against 0.23 µs for plain `(left, right)` tuples). With giglib's own operations on sorted lists of intervals (and a single sweep to cut a node's genome into pieces with the same copies), the same results take 3.5 to 50 times less time for `find_homology` (e.g. 0.40 s to 0.02 s for `"closest_homologues"` between genomes of a forward simulation with microsatellites, over 80 generations), and about 3 times less for `sample_resolve` | #234 |
| Simplifying before finding homology (cases `simplify_microsatellites` and `simplify_single_origin`, against `homology_microsatellites` and `homology_single_origin`) gives the same homology records, and the gain grows with the depth of the graph. For microsatellites over 20 to 80 generations, simplifying (15 to 35 ms) removes 75% to 93% of the nodes, and makes `find_homology` 2.5 to 9 times faster per pair (e.g. 20.7 to 2.2 ms for `"all"` after 80 generations). For genomes grown from a single base over 400 generations, it removes 99% of the nodes in 17 ms, and `"all"` takes 53 ms per pair rather than 1.55 s. So simplifying pays for itself after a few pairs. Rebuilding `intervals.pieces` on `coverage` (#241), to share one sweep with `simplify`, costs at most about 5% in `find_homology` | #247 |
