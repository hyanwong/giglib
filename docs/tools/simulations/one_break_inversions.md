---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_tools_simulation_one_break)=

# One crossover, avoiding inversions

`DTWF_one_break_no_rec_inversions_slow_sim` is a forward-time simulator (see
{ref}`sec_tools_simulation`) with recombination. Each meiosis has one crossover, placed by
using {meth}`~giglib.Tables.find_homology` to find positions in the two parent genomes
that are homologous. If those positions are in opposite orientations (i.e. in an inversion
relative to each other), the simulator picks another position, so it never recombines across
an inversion.

To give the first generation a common ancestor, the simulator adds a temporary root at an
infinite time, which it removes at the end.

```{code-cell}
from giglib.tools.simulation.one_break_inversions import DTWF_one_break_no_rec_inversions_slow_sim

simulator = DTWF_one_break_no_rec_inversions_slow_sim()
gig = simulator.run(num_diploids=10, seq_lens=[100], gens=5, random_seed=1)
print(f"{len(gig.nodes)} nodes and {len(gig.iedges)} iedges")
print(f"As a tree sequence, this has {gig.to_tree_sequence().num_trees} trees")
```

```{eval-rst}
.. automodule:: giglib.tools.simulation.one_break_inversions
  :members:
```
